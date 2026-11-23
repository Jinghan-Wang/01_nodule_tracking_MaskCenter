"""Phase 6 steps 1-3: project one 3D Mask centroid to both detector images.

Only label preparation reads current-phase 3D masks. No TIGRE rerun is needed.
Uses the verified, locally calibrated patient-LPS cameras from Phase 5.
"""
# 作用说明：
# 对每个呼吸状态读取对应的三维肿瘤 Mask，先计算该 Mask 在患者 LPS 坐标系中的
# 三维中心位置 [x, y, z]（单位：mm），再利用 Phase 5 已经验证通过的 kv1、kv2
# 两个投影矩阵，将同一个三维肿瘤中心分别投影到两个探测器平面，得到对应的二维
# 真值位置 (u1, v1) 和 (u2, v2)。这些二维位置作为后续孪生网络训练和评价时的
# Ground Truth，用来生成响应图标签或计算定位误差，而不能作为实际推理阶段的输入。
# 同时保存每个状态的三维中心、两个视角的二维投影坐标、几何信息及 Mask 文件
# SHA256 指纹，并检查 original 状态重新计算得到的三维中心是否与 Phase 5 标定时
# 保存的计划中心 plan_lps_mm 一致，从而确保标签、患者数据和投影几何属于同一套数据。
# 核心流程：
# 3D Mask -> 计算3D中心 -> 投影到kv1/kv2 -> 得到两个视角的2D真值标签

import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import data_process  # 支持直接运行。
import numpy as np
import SimpleITK as sitk
from geometry import project

DEFAULT_OUT = ROOT/'outputs'/'siamfc_dual_center3d'
DEFAULT_LABELS = DEFAULT_OUT/'projected_centers.json'
LABEL_DEFINITION = 'projected_3d_mask_centroid_lps'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_labels(volume_root=ROOT/'datasets'/'Patient_1',
                   geometry_path=ROOT/'outputs'/'phase5'/'geometry.json'):
    """Return a small reusable label dictionary; never import GPU preprocessing."""
    volume_root, geometry_path = Path(volume_root), Path(geometry_path)
    geo = json.loads(geometry_path.read_text(encoding='utf-8'))
    if geo.get('geometry_verified') is not True or geo.get('mode') != 'original':
        raise ValueError('Need verified Phase 5 original-mode geometry')
    original = volume_root/'original'/'original_1_tumor_mask.nii.gz'
    if sha256(original) != geo['mask_sha256']:
        raise ValueError('Original Mask differs from the calibrated patient')
    cameras = np.asarray(geo['cameras'], float)
    if cameras.shape != (2,3,4) or not np.isfinite(cameras).all():
        raise ValueError('Invalid calibrated cameras')
    records = {}
    states = ['original']+sorted(p.name for p in volume_root.glob('phase_*') if p.is_dir())
    for state in states:
        path = original if state == 'original' else volume_root/state/'rand_0000-tumor.nii.gz'
        im = sitk.ReadImage(str(path))
        if im.GetDimension() != 3 or list(im.GetSize()) != geo['volume_shape']:
            raise ValueError('3D Mask grid shape mismatch: '+state)
        affine = np.eye(4)
        affine[:3,:3] = np.asarray(im.GetDirection()).reshape(3,3).dot(np.diag(im.GetSpacing()))
        affine[:3,3] = im.GetOrigin()
        if not np.allclose(affine,geo['voxel_to_lps'],atol=1e-5,rtol=0):
            raise ValueError('3D Mask affine differs from calibrated grid: '+state)
        mask = sitk.GetArrayFromImage(im)
        if not np.isfinite(mask).all() or not np.any(mask > 0):
            raise ValueError('Invalid 3D Mask: '+state)
        index = np.argwhere(mask > 0).mean(0)[::-1]  # z,y,x -> i,j,k
        center = affine[:3,:3].dot(index)+affine[:3,3]
        # Calibration has only been tested locally; fail instead of extrapolating silently.
        offset = np.linalg.solve(affine[:3,:3], center-np.asarray(geo['plan_lps_mm']))*im.GetSpacing()
        if np.any(abs(offset) > 18):
            raise ValueError('Center outside locally verified calibration region: '+state)
        projected = {}
        for view,p in zip(('kv1','kv2'),cameras):
            uv, valid = project(center,p)
            h,w = geo['detector_image_shape']
            if not valid[0] or not (0 <= uv[0,0] < w and 0 <= uv[0,1] < h):
                raise ValueError('Projected target outside detector: '+state+'/'+view)
            projected[view] = uv[0].tolist()
        # records['phase_000x']
        # center_lps_mm：这个呼吸状态真实的3D肿瘤中心    projected_uv：这个3D中心投影到kv1、kv2的位置   mask_sha256：当前3D Mask文件的“数字指纹”
        records[state] = dict(center_lps_mm=center.tolist(), projected_uv=projected,
                              mask_sha256=sha256(path))
    if not np.allclose(records['original']['center_lps_mm'],geo['plan_lps_mm'],atol=1e-5,rtol=0):
        raise ValueError('Original center differs from calibration')
    # states=records :所有状态的3D中心 kv1二维中心、kv2二维中心、Mask文件指纹
    return dict(label_definition=LABEL_DEFINITION, geometry_sha256=sha256(geometry_path),
                geometry_path=str(geometry_path.resolve()), cameras=cameras.tolist(),
                detector_image_shape=geo['detector_image_shape'], states=records,
                note='Current-phase centers are supervision/evaluation only, never inference inputs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volume-root',type=Path,default=ROOT/'datasets'/'Patient_1')
    parser.add_argument('--geometry',type=Path,default=ROOT/'outputs'/'phase5'/'geometry.json')
    parser.add_argument('--out',type=Path,default=DEFAULT_LABELS)
    args=parser.parse_args()
    labels=prepare_labels(args.volume_root,args.geometry)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(labels,indent=2),encoding='utf-8')
    print('Prepared {} states: {}'.format(len(labels['states']),args.out))


if __name__ == '__main__':
    main()
