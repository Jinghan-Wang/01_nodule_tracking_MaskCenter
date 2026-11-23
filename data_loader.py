"""Patient_1 / LIDC 单患者模板搜索 Dataset；多患者按患者/视角组合加载。
用 view 选择 kv1/kv2；定位标签是三维 Mask 中心的投影，二维 Mask 用于模板裁剪/显示。
"""
# 本版数据加载器将定位 Ground Truth 从“二维投影 Mask 质心”
# 修改为“患者 LPS 坐标系下三维肿瘤 Mask 中心经已验证双视角几何
# 投影得到的二维坐标”。二维 Mask 仍用于模板 ROI 裁剪、显示和质控，
# 但不再作为新的定位损失中心；这样 kv1/kv2 的二维监督都对应同一个
# 三维物理点，为后续双视角三角定位和三维联合搜索保持几何一致性。

import argparse
import csv
import json
from pathlib import Path

import runtime_config
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data_process.prepare_center_labels import DEFAULT_LABELS, LABEL_DEFINITION
from data_process.search_roi import build_search_box, projected_center_bounds, crop_search

DEFAULT_ROOT = Path(__file__).resolve().parent / 'datasets' / 'DRR' / 'Patient_1'


def read_pair(folder, view='kv1', with_mask=True, paths=None):
    """读取旧命名或 LIDC 目录的一对图像；image[v,u]，不改变图像方向。"""
    if view not in ('kv1', 'kv2'):
        raise ValueError('view must be kv1 or kv2')
    folder = Path(folder)
    lidc = (folder / 'CTDRR_ROI_NULL').is_dir()
    if paths is not None:
        paths = [Path(p) if p is not None else None for p in paths]
    elif lidc:
        suffix = '0000_angle_0.0' if view == 'kv1' else '0001_angle_150.0'
        paths = [folder / 'CTDRR_ROI_NULL' / ('CTDRR_' + suffix + '.nii.gz'),
                 folder / 'MASKDRR_ROI_NULL' / ('MASKDRR_' + suffix + '.nii.gz')]
    else:
        stem = 'patient-1-{}-sample-0000-{}-'.format(folder.name, view)
        paths = [folder / (stem + suffix + '.nii.gz')
                 for suffix in ('ct-drr', 'mask-binary')]
    for path in (paths if with_mask else paths[:1]):
        if path is None or not path.is_file():
            raise FileNotFoundError(str(path))
    if not with_mask:
        im = sitk.ReadImage(str(paths[0]))
        image = sitk.GetArrayFromImage(im).astype(np.float32)
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]  # 仅去掉 DRR 的单层 z 轴；不转置、不翻转、不缩放。
        if image.ndim != 2 or not np.isfinite(image).all() or image.max() <= image.min():
            raise ValueError('Invalid search image: '+str(paths[0]))
        return image, None, None, (im.GetSize(), im.GetSpacing(), im.GetOrigin(), im.GetDirection())
    images = [sitk.ReadImage(str(p)) for p in paths]
    a, b = images
    if any(x.GetDimension() not in (2, 3) or
           (x.GetDimension() == 3 and x.GetSize()[2] != 1) for x in images):
        raise ValueError('Expected 2D DRR and mask: {}'.format(folder))
    for getter in ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection'):
        if not np.allclose(getattr(a, getter)(), getattr(b, getter)(), rtol=0, atol=1e-6):
            raise ValueError('DRR/mask grid mismatch: {}'.format(folder))
    # SimpleITK already returns [row, column]; do not transpose or flip.
    image, mask = [sitk.GetArrayFromImage(x).astype(np.float32) for x in images]
    if image.ndim == 3:
        image, mask = image[0], mask[0]
    if not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError('Non-finite input: {}'.format(folder))
    if image.max() <= image.min():
        raise ValueError('Constant DRR: {}'.format(folder))
    if lidc:
        if np.any(mask < 0):
            raise ValueError('Negative projected Mask: {}'.format(folder))
        mask = (mask > 0).astype(np.float32)
    if not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError('Expected nonempty binary mask: {}'.format(folder))
    v, u = np.nonzero(mask)
    center = np.array([u.mean(), v.mean()], dtype=np.float32)
    grid = (a.GetSize(), a.GetSpacing(), a.GetOrigin(), a.GetDirection())
    return image, mask, center, grid


class DRRPairDataset(Dataset):
    """One original template paired with each selected breathing phase.

    Images/masks: float32 [1,H,W]. Coordinates: [u,v], zero-based pixel
    centers, u right and v down. No resizing or augmentation.
    search_radius_mm启用时：搜索图和训练标签均为局部坐标；search_origin_uv
    用于换回全图，response_origin_uv用于把响应网格直接换成全图坐标。
    with_targets=False reads no current-phase Mask or target center.
    """
    def __init__(self, root=DEFAULT_ROOT, phases=None, margin=16, view='kv1',
                 labels=DEFAULT_LABELS, with_targets=True, search_radius_mm=None,
                 pair_paths=None, load_search_masks=True, align_search_grid=False):
        self.root = Path(root)
        self.view = view
        self.with_targets = with_targets
        self.pair_paths = pair_paths
        self.load_search_masks = load_search_masks
        # labels=None is reserved for explicitly loading an old 2D-centroid checkpoint.
        self.labels = (json.loads(Path(labels).read_text(encoding='utf-8'))
                       if isinstance(labels,(str,Path)) else labels)
        if self.labels is not None and self.labels.get('label_definition') != LABEL_DEFINITION:
            raise ValueError('Unexpected center label definition')
        if with_targets and not load_search_masks and self.labels is None:
            raise ValueError('Centre labels required when current projected Mask is not loaded')
        if not isinstance(margin, int) or margin < 0:
            raise ValueError('margin must be a nonnegative integer')
        available = sorted(p.name for p in self.root.glob('phase_*') if p.is_dir())
        self.phases = available if phases is None else list(phases)
        if not self.phases or len(set(self.phases)) != len(self.phases):
            raise ValueError('Select at least one phase, without duplicates')
        if any(p not in available for p in self.phases):
            raise ValueError('Unknown phase; available: {}'.format(available))
        template_folder = self.labels.get('template_folder', 'original') if self.labels is not None else 'original'
        full, mask, self.template_center, self.grid = read_pair(
            self.root/template_folder, view, paths=None if pair_paths is None else pair_paths['original'])
        self.template_mask_center = self.template_center.copy()
        if self.labels is not None:
            self.template_center = np.asarray(self.labels['states']['original']['projected_uv'][view],np.float32)
            if list(full.shape) != self.labels['detector_image_shape']:
                raise ValueError('DRR size differs from calibrated geometry')
        low, high = np.percentile(full, [1, 99])
        if high <= low:
            raise ValueError('Invalid original normalization range')
        self.low, self.high = float(low), float(high)
        v, u = np.nonzero(mask)
        h, w = full.shape
        x0, y0 = max(0, int(u.min())-margin), max(0, int(v.min())-margin)
        x1, y1 = min(w, int(u.max())+margin+1), min(h, int(v.max())+margin+1)
        # 模板 ROI 在 original DRR 里的裁剪框：左边界、上边界、右边界、下边界
        self.crop_box = (x0, y0, x1, y1)
        self.crop_origin = np.array([x0, y0], dtype=np.float32)
        self.template = self.normalize(full[y0:y1, x0:x1])
        self.template_mask = mask[y0:y1, x0:x1].copy()
        self.search_radius_mm = search_radius_mm
        self.full_image_shape = list(full.shape)
        self.search_crop_box = (0, 0, w, h)
        if search_radius_mm is not None:
            if self.labels is None:
                raise ValueError('Local search requires verified 3D projection labels')
            self.search_crop_box = build_search_box(
                self.labels['states']['original']['center_lps_mm'],
                self.labels['cameras'][0 if view == 'kv1' else 1], self.template.shape,
                self.template_center-self.crop_origin, search_radius_mm,
                template_origin_uv=self.crop_origin if align_search_grid else None)
        self.search_origin = np.asarray(self.search_crop_box[:2], dtype=np.float32)
        # 全图响应起点用float64，避免大坐标先相加再相减造成精度损失。
        self.response_origin = self.template_center.astype(float)-self.crop_origin+self.search_origin
        if search_radius_mm is not None:
            self.search_center_bounds = projected_center_bounds(
                self.labels['states']['original']['center_lps_mm'],
                self.labels['cameras'][0 if view == 'kv1' else 1], search_radius_mm)-self.search_origin

    def normalize(self, image):
        return np.clip((image-self.low)/(self.high-self.low), 0, 1).astype(np.float32)

    def __len__(self):
        return len(self.phases)

    def __getitem__(self, index):
        phase = self.phases[index]
        image, mask, center, grid = read_pair(
            self.root/phase, self.view, with_mask=self.with_targets and self.load_search_masks,
            paths=None if self.pair_paths is None else self.pair_paths[phase])
        if any(not np.allclose(a, b, rtol=0, atol=1e-6) for a, b in zip(self.grid, grid)):
            raise ValueError('Original/search grid mismatch: {}'.format(phase))
        tensor = lambda a: torch.from_numpy(a.copy())
        search = self.normalize(image)
        if self.search_radius_mm is not None:
            search = crop_search(search, self.search_crop_box)
        result = {
            'template': tensor(self.template[None]),
            'search': tensor(search[None]),
            'template_mask': tensor(self.template_mask[None]),
            'template_center_uv': tensor(self.template_center),
            'template_center_local_uv': tensor(self.template_center-self.crop_origin),
            'template_origin_uv': tensor(self.crop_origin),
            'search_origin_uv': tensor(self.search_origin),
            'response_origin_uv': tensor(self.response_origin),
            'phase': phase,
            'view': self.view,
        }
        if self.with_targets:
            if self.search_radius_mm is not None:
                result['search_center_bounds_uv'] = tensor(self.search_center_bounds.astype(np.float32))
            if mask is not None:
                if self.search_radius_mm is not None:
                    mask = crop_search(mask, self.search_crop_box)
                result['search_mask'] = tensor(mask[None])  # 仅供显示，不输入网络
                result['search_mask_center_uv'] = tensor(center-self.search_origin)
            if self.labels is not None:
                center = np.asarray(self.labels['states'][phase]['projected_uv'][self.view],np.float32)
            #  当前 phase 三维中心的投影
            result['search_center_uv'] = tensor(center-self.search_origin)
            result['delta_uv'] = tensor(center-self.template_center)
        return result

""""
每个 dataset[i] 最终包含:
template                 原始肿瘤局部 DRR
search                   当前 phase DRR（可选固定局部ROI）

template_mask            模板肿瘤 mask
search_mask              当前 phase 肿瘤 GT mask

template_center_uv        original 三维中心投影的全图坐标
template_center_local_uv  模板局部坐标中的中心
template_origin_uv        模板左上角全局坐标

search_center_uv          当前 phase 三维中心投影，在实际search图中的坐标（仅训练/评估）
search_mask_center_uv     旧二维 Mask 质心，仅供对照，不用于新定位损失
delta_uv                  相对 original 的位移
search_origin_uv          search图左上角在完整DRR中的坐标
response_origin_uv        响应图[0,0]所代表中心在完整DRR中的坐标

phase                     当前 phase 名称
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--phases', nargs='+', default=None)
    parser.add_argument('--view', choices=['kv1', 'kv2'], default='kv1')
    parser.add_argument('--labels',type=Path,default=DEFAULT_LABELS)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--margin', type=int, default=16)
    parser.add_argument('--preview', action='store_true', help='Save preview.png and centers.csv')
    args = parser.parse_args()
    dataset = DRRPairDataset(args.root, args.phases, args.margin, view=args.view,labels=args.labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    count = 0
    for batch in loader:
        assert torch.isfinite(batch['template']).all() and torch.isfinite(batch['search']).all()
        count += batch['search'].shape[0]
        print('{} template={} search={}'.format(batch['phase'], tuple(batch['template'].shape), tuple(batch['search'].shape)))
    print('PASS: {} pairs; torch {}; original crop {}'.format(count, torch.__version__, dataset.crop_box))
    if args.preview:
        out = Path(__file__).resolve().parent / 'outputs'
        out.mkdir(exist_ok=True)
        cols = 4
        rows = (len(dataset)+1+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        sample = dataset[0]
        ax = axes.flat[0]
        ax.imshow(sample['template'][0], cmap='gray', vmin=0, vmax=1)
        ax.contour(sample['template_mask'][0], [.5], colors=['lime'])
        ax.plot(*sample['template_center_local_uv'].numpy(), 'r+')
        ax.set_title('original template (local pixels)')
        records = []
        for i in range(len(dataset)):
            item = dataset[i]
            ax = axes.flat[i+1]
            ax.imshow(item['search'][0], cmap='gray', vmin=0, vmax=1)
            ax.contour(item['search_mask'][0], [.5], colors=['lime'])
            ax.plot(*item['search_center_uv'].numpy(), 'r+')
            ax.plot(*item['template_center_uv'].numpy(), 'c+')
            ax.set_title('{} du={:.2f} dv={:.2f}'.format(item['phase'], *item['delta_uv'].tolist()))
            records.append([item['phase']] + item['search_center_uv'].tolist() + item['delta_uv'].tolist())
        fig.tight_layout()
        fig.savefig(str(out / ('preview_' + args.view + '.png')), dpi=120)
        plt.close(fig)
        with (out / ('centers_' + args.view + '.csv')).open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['phase', 'u_px', 'v_px', 'du_px', 'dv_px'])
            writer.writerows(records)
        print('Preview: {}'.format(out))


if __name__ == '__main__':
    main()
