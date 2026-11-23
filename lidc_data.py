"""LIDC 数据准备与加载；训练默认使用实际 Demo 验证后的几何标签。

prepare() 保留旧 TIGRE 暂定实验；新训练使用 VERIFIED_PREPARATION。
Python 3.7；不修改原始数据。训练/测试复用 data_loader.DRRPairDataset。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from geometry import fit_camera, project, triangulate
from data_process.prepare_center_labels import ROOT, LABEL_DEFINITION, sha256

DEFAULT_ROOT = ROOT / 'Data' / 'LIDC_one'
DEFAULT_OUT = ROOT / 'outputs' / 'lidc_dual_provisional'
VERIFIED_PREPARATION = ROOT / 'outputs' / 'lidc_demo_geometry'
VIEWS = ('kv1', 'kv2')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def patient_split(patients, seed=42):
    """24 位患者默认 18/3/3；同一患者的所有状态、视角只能属于一个集合。"""
    names = sorted(patients)
    if len(names) < 6:
        raise ValueError('Need at least six patients for three separate groups')
    shuffled = np.random.RandomState(seed).permutation(names).tolist()
    held = max(1, int(round(len(names) * .125)))
    return dict(train=sorted(shuffled[2*held:]), val=sorted(shuffled[:held]),
                test=sorted(shuffled[held:2*held]))


def check_split(split, patients):
    """三个集合非空、各自无重复、相互不交叉，且覆盖全部患者。"""
    groups = [split[k] for k in ('train','val','test')]
    names = sum(groups, [])
    if (any(not g for g in groups) or len(names) != len(set(names)) or
            set(names) != set(patients)):
        raise ValueError('Invalid patient split or patient leakage')


def prepare_patient(folder, out, pp, cfg):
    """每患者独立标定自己的完整网格，不假设不同 CT 网格共享同一投影矩阵。"""
    from data_process.check_geometry import read_volume, same_ct_grid, forward, point_phantom, centroid
    from data_loader import read_pair

    out.mkdir(parents=True, exist_ok=True)
    volume, affine, spacing, index, plan = read_volume(folder/'original'/'mask.nii.gz')
    shape = volume.shape
    same_ct_grid(folder/'original'/'CT.nii.gz', shape, affine)
    xyz, pixels = [], []
    for number, displacement in enumerate(np.random.RandomState(2026).uniform(-18, 18, (12, 3))):
        ijk = index+displacement/spacing
        if np.any(ijk < 6) or np.any(ijk > np.asarray(shape)-7):
            raise ValueError('Calibration phantom would leave CT grid: '+folder.name)
        images = forward(pp, cfg, point_phantom(shape, ijk, spacing), spacing)
        xyz.append(affine[:3,:3].dot(ijk)+affine[:3,3])
        pixels.append([centroid(image) for image in images])
        if (number+1) % 4 == 0:
            print('{} calibration {}/12'.format(folder.name, number+1), flush=True)
    xyz, pixels = np.asarray(xyz), np.asarray(pixels)
    cameras = [fit_camera(xyz[:8], pixels[:8,v]) for v in range(2)]
    error_px = max(np.linalg.norm(project(xyz[8:],p)[0]-pixels[8:,v],axis=1).max()
                   for v,p in enumerate(cameras))
    reconstructed = [triangulate(uv[0],uv[1],cameras)['xyz_lps_mm'] for uv in pixels[8:]]
    error_mm = np.linalg.norm(np.asarray(reconstructed)-xyz[8:],axis=1).max()
    if error_px > 1 or error_mm > 2:
        raise ValueError('Provisional camera numerical calibration failed: '+folder.name)
    states = ['original']+sorted(p.name for p in folder.glob('phase_*') if p.is_dir())
    records, audit = {}, []
    for state in states:
        path = folder/state/'mask.nii.gz'
        vol, a, sp, _, center = read_volume(path)
        same_ct_grid(folder/state/'CT.nii.gz', vol.shape, a)
        if vol.shape != shape or not np.allclose(a, affine, rtol=0, atol=1e-5):
            raise ValueError('State grid changed: '+folder.name+'/'+state)
        offset = np.linalg.solve(affine[:3, :3], center-plan)*spacing
        if np.any(abs(offset) > 18):
            raise ValueError('State outside provisional calibrated region')
        projected = {}
        # original 比较复投影 IoU；所有状态记录中心差异，不用差异反推/矫正相机。
        repro = forward(pp, cfg, (vol > 0).astype(np.float32), sp) if state == 'original' else None
        for v, (view, camera) in enumerate(zip(VIEWS, cameras)):
            image, mask, mask_center, _ = read_pair(folder/state, view)
            if image.shape != (512, 512):
                raise ValueError('Expected 512x512 DRR')
            uv, valid = project(center, camera)
            if not valid[0] or np.any(uv[0] < 0) or np.any(uv[0] >= 512):
                raise ValueError('Provisional target outside image')
            projected[view] = uv[0].tolist()
            iou = None
            if repro is not None:
                binary = repro[v] > 1e-6
                iou = float(np.logical_and(mask > 0, binary).sum()/np.logical_or(mask > 0, binary).sum())
            audit.append(dict(state=state, view=view, mask_iou=iou,
                              projected_center_to_mask_center_px=float(np.linalg.norm(uv[0]-mask_center))))
        records[state] = dict(center_lps_mm=center.tolist(), projected_uv=projected,
                              mask_sha256=sha256(path))
    geometry = dict(patient=folder.name, cameras=[p.tolist() for p in cameras],
                    coordinate_system='patient LPS mm', mode='original',
                    geometry_verified=False, geometry_status='provisional_old_projector_convention',
                    detector_image_shape=[512, 512], volume_shape=list(shape),
                    voxel_to_lps=affine.tolist(), plan_lps_mm=plan.tolist(),
                    config=cfg, calibrated_radius_mm=18,
                    heldout_reprojection_max_px=float(error_px),
                    heldout_reconstruction_max_mm=float(error_mm),
                    minimum_reproduced_mask_iou=min(r['mask_iou'] for r in audit if r['mask_iou'] is not None),
                    saved_mask_reprojection_scope='original only; all states have projected-center vs mask-center audit',
                    source_files_sha256={name:sha256(folder/name) for name in
                        ('angle_all.txt', 'congfigfiles_drr_LIDC.ini', 'original/paraset.ini')},
                    projection_sha256=sha256(ROOT/'preprocessing'/'preprocess.py'),
                    note='Old TIGRE convention assumed by user request. INI ISO/corrections are NOT applied. New-data geometry is NOT verified.')
    write_json(out/'geometry.json', geometry)
    labels = dict(label_definition=LABEL_DEFINITION, geometry_status=geometry['geometry_status'],
                  geometry_sha256=sha256(out/'geometry.json'),
                  geometry_path=str((out/'geometry.json').resolve()), cameras=geometry['cameras'],
                  detector_image_shape=[512, 512], states=records)
    write_json(out/'projected_centers.json', labels)
    original_labels = dict(labels)
    original_labels['states'] = {'original':records['original']}
    write_json(out/'original_labels.json', original_labels)
    write_json(out/'mask_projection_audit.json', audit)
    return dict(phases=states[1:], geometry_status=geometry['geometry_status'],
                minimum_mask_iou=geometry['minimum_reproduced_mask_iou'],
                heldout_projection_max_px=float(error_px), heldout_3d_max_mm=float(error_mm))


def prepare(root=DEFAULT_ROOT, out=DEFAULT_OUT, seed=42):
    # 先导入 TIGRE，再加载图像库，避免 Windows DLL 装载顺序问题。
    from data_process.check_geometry import load_projector
    root, out = Path(root).resolve(), Path(out).resolve()
    patients = sorted(p.name for p in root.glob('LIDC_*') if p.is_dir())
    split = patient_split(patients, seed)
    check_split(split,patients)
    if (out/'best.pt').exists() or (out/'training'/'best.pt').exists():
        raise ValueError('Choose a new output; never replace a trained run geometry')
    out.mkdir(parents=True, exist_ok=True)
    pp, import_mode = load_projector()
    cfg = pp.load_config(str(ROOT/'config'/'preprocess_config.json'))
    cfg.update(kv_size=[512, 512], drr_detector_pixel_mm=[1.2, 1.2])
    report = dict(root=str(root), patient_split=split, seed=seed,
                  geometry_status='provisional_old_projector_convention', patients={},
                  projection_import=import_mode, status='preparing')
    write_json(out/'dataset.json', report)
    for name in patients:
        report['patients'][name] = prepare_patient(root/name, out/'patients'/name, pp, cfg)
        write_json(out/'dataset.json', report)
        print('{}: {} states; min saved-mask IoU={:.4f}; provisional'.format(
            name, len(report['patients'][name]['phases']), report['patients'][name]['minimum_mask_iou']), flush=True)
    report['status'] = 'prepared_provisional'
    write_json(out/'dataset.json', report)
    return report


def load_datasets(root, preparation, patients, margin=22, with_targets=True, search_radius_mm=None,
                  phase_selection=None):
    """每患者/视角一个 Dataset，批次内模板尺寸一致，不对 ROI 做缩放。"""
    from data_loader import DRRPairDataset
    result = {}
    for patient in patients:
        phases = None if phase_selection is None else phase_selection[patient]
        if phases == []:
            continue
        folder = Path(preparation)/'patients'/patient
        name = 'projected_centers.json' if with_targets else 'original_labels.json'
        labels = json.loads((folder/name).read_text(encoding='utf-8'))
        geometry = json.loads((folder/'geometry.json').read_text(encoding='utf-8'))
        if not with_targets and set(labels['states']) != {'original'}:
            raise ValueError('Prediction labels must contain original only: '+patient)
        if with_targets:
            available_phases = {p.name for p in (Path(root)/patient).glob('phase_*') if p.is_dir()}
            if set(labels['states']) != available_phases | {'original'}:
                raise ValueError('State list changed since label preparation: '+patient)
        if (sha256(folder/'geometry.json') != labels['geometry_sha256'] or
                labels['geometry_status'] != geometry['geometry_status'] or
                not np.array_equal(labels['cameras'], geometry['cameras'])):
            raise ValueError('Geometry/labels mismatch: '+patient)
        # 预测时只检查 original；不读取当前状态 Mask 或真值。
        for state, record in labels['states'].items():
            if sha256(Path(root)/patient/state/'mask.nii.gz') != record['mask_sha256']:
                raise ValueError('Mask changed since label preparation: '+patient+'/'+state)
            for view, camera in zip(VIEWS, geometry['cameras']):
                uv, valid = project(record['center_lps_mm'], camera)
                if not valid[0] or not np.allclose(uv[0], record['projected_uv'][view], rtol=0, atol=1e-6):
                    raise ValueError('Projected center mismatch: '+patient+'/'+state+'/'+view)
        for view in VIEWS:
            result[(patient, view)] = DRRPairDataset(Path(root)/patient, phases=phases, margin=margin,
                                                    view=view, labels=labels, with_targets=with_targets,
                                                    search_radius_mm=search_radius_mm)
    return result


def preview_datasets(datasets, patients, out):
    """显示实际归一化输入与投影标签；每患者两行，两视角不混用模板。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for patient in patients:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for row, view in enumerate(VIEWS):
            ds = datasets[(patient, view)]
            sample = ds[0]
            template = sample['template'][0].numpy()
            search = sample['search'][0].numpy()
            local = sample['template_center_local_uv'].numpy()
            target = sample['search_center_uv'].numpy()
            for col, (image, center) in enumerate([(template, local), (search, target), (search, target)]):
                ax = axes[row, col]
                ax.imshow(image, cmap='gray', vmin=0, vmax=1)
                ax.plot(*center, 'g+', markersize=12, label='Projected 3D center')
                ax.set_xlabel('u (pixels)')
                ax.set_ylabel('v (pixels)')
                ax.legend(loc='upper right', fontsize=8)
            axes[row, 0].contour(sample['template_mask'][0].numpy(), [.5], colors=['cyan'])
            axes[row, 0].set_title(view+' original template / local coordinates')
            axes[row, 1].set_title(view+' '+sample['phase']+' / search input')
            axes[row, 2].set_title(view+' search zoom / input coordinates')
            u, v = target
            axes[row, 2].set_xlim(max(0, u-50), min(search.shape[1]-1, u+50))
            axes[row, 2].set_ylim(min(search.shape[0]-1, v+50), max(0, v-50))
        fig.suptitle(patient+' | green +: label; cyan: original projected Mask')
        fig.tight_layout()
        fig.savefig(str(out/(patient+'.png')), dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.root, args.out, args.seed)
