"""多患者局部搜索训练：每轮更新last.pt，每10轮另存独立的.pth模型。"""
import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path

import runtime_config
import numpy as np
import torch
from network import SiamFC, response_loss
from data_process.prepare_center_labels import LABEL_DEFINITION, sha256
from manifest_data import ManifestSource, training_loader, write_json

VIEWS = ('kv1', 'kv2')
ROOT = runtime_config.PROJECT_ROOT  # 保留旧ROI准备脚本读取的项目目录常量。

# 地址修改runtime_config.py；这里仅设置训练参数，命令行参数可覆盖。
TRAIN_CONFIG = dict(
    output_root=Path(runtime_config.TRAIN_OUTPUT_ROOT),
    epochs=100,
    batch_size=6,
    steps_per_epoch=1,        # 每轮完整遍历训练集一次
    lr=0.001,
    seed=42,
    margin=22,
    search_radius_mm=10.,    # 计划中心各轴±10mm；超范围状态不参与训练
    positive_radius=16./3.,
    center_label='gaussian',
    gaussian_sigma=8./3.,    # 完整DRR像素，默认等于圆形正区域半径的一半
    device='auto',
    num_workers=4,
)


def augment(search, centers, local_centers, template_hw, max_shift=16, center_bounds=None):
    """Training-only integer translation; image and center get identical shift."""
    moved = torch.zeros_like(search)
    labels = centers.clone()
    h, w = search.shape[-2:]
    for i in range(len(search)):
        # Keep the target's template footprint inside the image, without wrapping.
        origin = (centers[i]-local_centers[i]).detach().cpu().numpy()
        upper = np.array([w-template_hw[1], h-template_hw[0]])-origin
        lower = np.maximum(-max_shift, np.ceil(-origin)).astype(int)
        upper = np.minimum(max_shift, np.floor(upper)).astype(int)
        if center_bounds is not None:
            bounds = center_bounds[i].cpu().numpy()-centers[i].detach().cpu().numpy()
            lower = np.maximum(lower, np.ceil(bounds[0]).astype(int))
            upper = np.minimum(upper, np.floor(bounds[1]).astype(int))
        if np.any(lower > upper):
            raise ValueError('No valid training translation for target')
        du, dv = [np.random.randint(lo, hi+1) for lo, hi in zip(lower, upper)]
        x0, x1 = max(0, du), min(w, w+du)
        y0, y1 = max(0, dv), min(h, h+dv)
        moved[i, :, y0:y1, x0:x1] = search[i, :, y0-dv:y1-dv, x0-du:x1-du]
        moved[i] = (moved[i]*float(np.random.uniform(.9, 1.1))).clamp(0, 1)
        labels[i] += labels.new_tensor([du, dv])
    return moved, labels


def fit_model(model, loader, args, config):
    """只使用训练集；不按训练损失挑选best，不访问独立测试集。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    with (args.out/'history.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss'])
        for epoch in range(1, args.epochs+1):
            model.train()
            loss_sum, samples_seen = 0., 0
            for _ in range(args.steps_per_epoch):
                for batch in loader:
                    local = batch['template_center_local_uv'].to(args.device)
                    search, centers = augment(
                        batch['search'].to(args.device), batch['search_center_uv'].to(args.device),
                        local, batch['template'].shape[-2:], max_shift=21,
                        center_bounds=batch.get('search_center_bounds_uv'))
                    logits = model(batch['template'].to(args.device), search)
                    loss = response_loss(logits, centers, local, radius=args.positive_radius,
                                         center_label=args.center_label, gaussian_sigma=args.gaussian_sigma)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite training loss')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()*len(search)
                    samples_seen += len(search)
            train_loss = loss_sum/samples_seen
            checkpoint = dict(model=model.state_dict(), config=config, epoch=epoch,
                              train_loss=train_loss, checkpoint_kind='last_epoch')
            torch.save(checkpoint, str(args.out/'last.pt'))
            # 保留指定轮次的模型，后续轮次不会覆盖这些快照。
            if epoch % 10 == 0:
                torch.save(checkpoint, str(args.out/'epoch_{:03d}.pth'.format(epoch)))
            writer.writerow([epoch, train_loss])
            file.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print('epoch {}: loss={:.4f}; last.pt saved (no validation split)'.format(
                    epoch, train_loss), flush=True)
    print('Saved checkpoint: {}'.format(args.out/'last.pt'))


def train_manifest(args):
    """359例新清单入口：不划分患者，不读取独立测试数据，不按训练误差选best。"""
    root = Path(args.root or runtime_config.TRAIN_DATA_ROOT)
    preparation = Path(args.preparation or runtime_config.TRAIN_PREPARATION_ROOT)
    manifest = args.manifest or preparation/'manifest_train'/'dataset.json'
    source = ManifestSource(manifest, root, preparation, expected_split='train')
    if args.limit_patients:
        source = ManifestSource(manifest, root, preparation, expected_split='train',
                                patients=source.patient_ids[:args.limit_patients])
    if args.search_radius_mm != source.radius:
        raise ValueError('Training search radius must equal manifest radius')
    if any((args.out/name).exists() for name in ('last.pt', 'best.pt', 'config.json')):
        raise ValueError('Output already contains a run; choose a new --out')
    datasets = {}
    for patient in source.patient_ids:
        sets, _ = source.datasets(patient, with_targets=True, main_only=True, margin=args.margin,
                                  align_search_grid=True)
        datasets.update(sets)
    loader = training_loader(datasets, args.batch_size, args.num_workers, args.device.startswith('cuda'))
    config = dict(schema_version=5, dataset_format='portable_patient_state_manifest',
                  root=str(source.data_root), preparation=str(source.preparation_root),
                  manifest=str(source.path), manifest_sha256=sha256(source.path),
                  patient_split={'train':source.patient_ids},
                  training_subjects=sorted({source.patient(p)['subject_id'] for p in source.patient_ids}),
                  views=list(VIEWS), shared_weights=True, label_definition=LABEL_DEFINITION,
                  geometry_status='demo_verified_against_saved_data', margin=args.margin,
                  search_radius_mm=source.radius, positive_radius_px=args.positive_radius,
                  align_search_grid=True, center_label=args.center_label,
                  gaussian_sigma_px=args.gaussian_sigma,
                  response_loss='balanced_bce_with_logits_fixed_disk_weights',
                  stride=SiamFC.stride, epochs=args.epochs, batch_size=args.batch_size,
                  steps_per_epoch=args.steps_per_epoch, lr=args.lr, seed=args.seed,
                  device=args.device, num_workers=args.num_workers, tf32=False,
                  evaluation_protocol='fixed_epochs_no_validation; separate test manifest evaluated manually',
                  main_states=sum(len(ds) for (p,v),ds in datasets.items() if v=='kv1'),
                  geometry_sha256={p:source.patient(p)['source_json_sha256']['geometry'] for p in source.patient_ids},
                  preprocessing={}, torch_version=str(torch.__version__))
    for (patient, view), ds in datasets.items():
        config['preprocessing'].setdefault(patient, {})[view] = dict(
            normalization=[ds.low, ds.high], crop_box=list(ds.crop_box),
            search_crop_box=list(ds.search_crop_box), template_center_uv=ds.template_center.tolist())
    print('Patients: {}; main states: {}; paired views: {}; device: {}'.format(
        len(source.patient_ids), config['main_states'], 2*config['main_states'], args.device), flush=True)
    print('Python: {}\nTorch: {}\nData: {}\nPreparation: {}\nManifest: {}\nOutput: {}'.format(
        sys.executable, torch.__version__, source.data_root, source.preparation_root, source.path, args.out), flush=True)
    print('模板网格对齐；中心标签={}；圆形半径={:.3f}px；高斯sigma={:.3f}px'.format(
        args.center_label, args.positive_radius, args.gaussian_sigma), flush=True)
    model = SiamFC().to(args.device)
    # 只做一个批次的前向检查；其余批次在实际训练中逐批验证，不额外遍历整套图像。
    with torch.no_grad():
        batch = next(iter(loader))
        logits = model(batch['template'].to(args.device), batch['search'].to(args.device))
        loss = response_loss(logits, batch['search_center_uv'].to(args.device),
                             batch['template_center_local_uv'].to(args.device), args.positive_radius,
                             center_label=args.center_label, gaussian_sigma=args.gaussian_sigma)
        if not torch.isfinite(loss):
            raise ValueError('Initial manifest batch has invalid loss')
    if args.check_only:
        print('PASS: manifest, template/search shapes, labels and forward loss; no training/output saved.', flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'config.json', config)
    fit_model(model, loader, args, config)
    print('Training complete. Test the separate patient dataset with test.py; no test data loaded here.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=None)
    parser.add_argument('--preparation', type=Path, default=None, help='矩阵、标签和清单根目录')
    parser.add_argument('--manifest', type=Path, default=None, help='默认preparation/manifest_train/dataset.json')
    parser.add_argument('--out', type=Path, default=None, help='默认创建新的时间目录')
    for name in ('epochs', 'batch_size', 'steps_per_epoch', 'seed', 'margin', 'num_workers'):
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=TRAIN_CONFIG[name])
    for name in ('lr', 'search_radius_mm', 'positive_radius', 'gaussian_sigma'):
        parser.add_argument('--'+name.replace('_', '-'), type=float, default=TRAIN_CONFIG[name])
    parser.set_defaults(center_label=TRAIN_CONFIG['center_label'])
    parser.add_argument('--device', default=TRAIN_CONFIG['device'])
    parser.add_argument('--limit-patients', type=int, default=None, help='小规模试运行；正式训练不指定')
    parser.add_argument('--check-only', action='store_true', help='只检查一个批次前向，不训练、不保存输出')
    args = parser.parse_args()
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this Python environment')
    if (min(args.epochs, args.batch_size, args.steps_per_epoch) < 1 or args.num_workers < 0 or
            (args.limit_patients is not None and args.limit_patients < 1)):
        parser.error('epochs, batch-size, steps-per-epoch and limit-patients must be positive; workers >=0')
    if (not np.isfinite(args.lr) or args.lr <= 0 or args.margin < 0 or
            not np.isfinite(args.positive_radius) or args.positive_radius < 3):
        parser.error('lr must be positive; margin >=0; positive-radius >=3 for stride 4')
    if not np.isfinite(args.search_radius_mm) or not 0 < args.search_radius_mm <= 18:
        parser.error('search-radius-mm must be within (0,18] and match the manifest (currently 10)')
    if not np.isfinite(args.gaussian_sigma) or args.gaussian_sigma <= 0:
        parser.error('gaussian-sigma必须为正有限值，单位为完整DRR像素')
    args.out = (args.out or TRAIN_CONFIG['output_root']/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    train_manifest(args)


if __name__ == '__main__':
    main()
