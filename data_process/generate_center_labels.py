"""直接运行：使用已验证矩阵生成三维中心与双视角二维标签，不调用Demo/DLL。"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import data_process
import numpy as np
import SimpleITK as sitk

from data_process.calibrate_drr_demo import binary_center, plan_folder_name, save_json
from data_process.generate_projection_matrices import input_digest
from data_process.prepare_center_labels import LABEL_DEFINITION, sha256
from geometry import project, triangulate


# 改为400多例患者所在目录。默认从前一步对应输出目录读取矩阵并保存标签。
# DATA_ROOT = ROOT/'Data'/'New_data'
DATA_ROOT = "F:/AIDataSet/Nodule_tracking/Error_ID"

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def label_patient(folder, destination):
    """每患者先完成全部检查，再保存正式标签；不改写geometry.json。"""
    geometry_path = destination/'geometry.json'
    geometry = read_json(geometry_path)
    geometry_hash = sha256(geometry_path)
    if (geometry.get('patient') != folder.name or geometry.get('geometry_verified') is not True or
            geometry.get('geometry_status') != 'demo_verified_against_saved_data'):
        raise ValueError('A verified matrix for this patient is required')
    plan_name = plan_folder_name(folder)
    if geometry.get('template_folder') != plan_name:
        raise ValueError('Template directory differs from verified geometry')
    digest = input_digest(folder)
    if digest != geometry.get('input_digest'):
        raise ValueError('Input data/config changed since matrix validation; validate geometry first')
    phases = sorted(p.name for p in folder.glob('phase_*') if p.is_dir())
    if not phases:
        raise ValueError('No phase directories found')
    sources = [('original',plan_name)]+[(phase,phase) for phase in phases]
    recipe = dict(label_definition=LABEL_DEFINITION, generator_sha256=sha256(Path(__file__)))
    full_path, original_path = destination/'projected_centers.json', destination/'original_labels.json'
    if full_path.exists() and original_path.exists():
        full, original = read_json(full_path), read_json(original_path)
        if (full.get('geometry_sha256') == geometry_hash and full.get('input_digest') == digest and
                full.get('recipe') == recipe and set(full['states']) == {'original'} | set(phases)):
            if original != dict(full, states={'original':full['states']['original']}):
                raise ValueError('Original-only labels differ from the complete labels')
            check_projections(full, geometry)
            return full, True

    cameras = np.asarray(geometry['cameras'], float)
    if cameras.shape != (2,3,4) or not np.isfinite(cameras).all():
        raise ValueError('Invalid two-view camera matrices')
    affine = np.asarray(geometry['voxel_to_lps'], float)
    direction = affine[:3,:3]/np.linalg.norm(affine[:3,:3], axis=0)
    plan = np.asarray(geometry['plan_lps_mm'])
    records, max_centroid_difference = {}, 0.
    for state, source in sources:
        path = folder/source/'mask.nii.gz'
        before = sha256(path)
        image = sitk.ReadImage(str(path))
        if image.GetDimension()!=3 or list(image.GetSize())!=geometry['volume_shape']:
            raise ValueError('Mask size differs from calibrated volume: '+source)
        current_affine = np.eye(4)
        current_affine[:3,:3] = np.asarray(image.GetDirection()).reshape(3,3).dot(np.diag(image.GetSpacing()))
        current_affine[:3,3] = image.GetOrigin()
        if not np.allclose(current_affine, affine, atol=1e-6, rtol=0):
            raise ValueError('Mask physical grid differs from calibrated volume: '+source)
        index = binary_center(image)  # 验证非空0/1 Mask；返回xyz顺序的体素中心均值。
        center = np.asarray(image.TransformContinuousIndexToPhysicalPoint(index.tolist()))
        # 独立使用ITK的物理质心实现，核查数组轴顺序与物理坐标转换。
        statistics = sitk.LabelShapeStatisticsImageFilter()
        statistics.Execute(sitk.Cast(image, sitk.sitkUInt8))
        difference = float(np.linalg.norm(center-np.asarray(statistics.GetCentroid(1))))
        max_centroid_difference = max(max_centroid_difference, difference)
        if difference > 1e-5:
            raise ValueError('Independent physical centroid check failed: '+source)
        offset_ct_axes = np.linalg.solve(direction, center-plan)
        if np.max(abs(offset_ct_axes)) > geometry['calibrated_radius_mm']+1e-6:
            raise ValueError('State center outside verified geometry region: '+source)
        if state=='original' and not np.allclose(center, plan, atol=1e-5, rtol=0):
            raise ValueError('Plan center differs from matrix calibration')
        projected = {}
        h, w = geometry['detector_image_shape']
        for view, camera in zip(('kv1','kv2'), cameras):
            uv, valid = project(center, camera)
            if not valid[0] or not (0 <= uv[0,0] < w and 0 <= uv[0,1] < h):
                raise ValueError('Projected center outside detector: '+source+'/'+view)
            projected[view] = uv[0].tolist()
        if sha256(path) != before:
            raise ValueError('Mask changed during label generation: '+source)
        records[state] = dict(source_folder=source, center_lps_mm=center.tolist(),
                              projected_uv=projected, mask_sha256=before)
    labels = dict(patient=folder.name, label_definition=LABEL_DEFINITION,
                  geometry_status=geometry['geometry_status'], geometry_sha256=geometry_hash,
                  geometry_path='geometry.json', template_folder=plan_name,
                  cameras=cameras.tolist(), detector_image_shape=geometry['detector_image_shape'],
                  input_digest=digest, recipe=recipe, states=records,
                  note='3D binary Mask voxel-centre centroid in patient LPS mm; projected_uv uses full DRR pixels. Current-phase targets are supervision/evaluation only.')
    roundtrip = check_projections(labels, geometry)
    # 再次核对全部标定输入和矩阵，确保处理期间未被其他任务修改。
    if input_digest(folder) != digest or sha256(geometry_path) != geometry_hash:
        raise ValueError('Input or geometry changed during processing')
    labels['qc'] = dict(states=len(records), projected_points=2*len(records),
                        max_independent_centroid_difference_mm=max_centroid_difference,
                        max_algebraic_roundtrip_error_mm=roundtrip,
                        all_projected_centers_inside_detector=True, source_files_unchanged=True,
                        geometry_file_unchanged=True)
    save_json(full_path, labels)
    save_json(original_path, dict(labels, states={'original':records['original']}))
    return labels, False


def check_projections(labels, geometry):
    """代数闭环核验；独立几何精度仍以前一步实际Demo标定为准。"""
    max_error = 0.
    h, w = geometry['detector_image_shape']
    for state, record in labels['states'].items():
        for view, camera in zip(('kv1','kv2'), geometry['cameras']):
            uv, valid = project(record['center_lps_mm'], camera)
            expected = np.asarray(record['projected_uv'][view])
            if (not valid[0] or not np.isfinite(expected).all() or
                    not (0 <= expected[0] < w and 0 <= expected[1] < h) or
                    not np.allclose(uv[0], expected, atol=1e-7, rtol=0)):
                raise ValueError('Saved label projection mismatch: '+state+'/'+view)
        result = triangulate(record['projected_uv']['kv1'], record['projected_uv']['kv2'], geometry['cameras'])
        error = float(np.linalg.norm(np.asarray(result['xyz_lps_mm'])-record['center_lps_mm']))
        max_error = max(max_error, error)
    if max_error > 1e-5:
        raise ValueError('Two-view algebraic roundtrip failed')
    return max_error


def write_outputs(output, root, results):
    rows = []
    lines = ['# 中心标签质检结论', '', '数据根目录：'+str(root),
             '使用前一步已验证的geometry.json；本步骤不调用Demo、DLL或TIGRE。', '',
             '| 患者 | 结论 | 状态数（含Origin） | 二维标签数 | ±10mm内phase数 | 独立质心核对最大差mm |',
             '|---|---|---:|---:|---:|---:|']
    for result in results:
        if result['status']!='passed':
            lines.append('| {} | 未通过 | — | — | — | — |'.format(result['patient']))
            continue
        labels = result['labels']
        plan = np.asarray(labels['states']['original']['center_lps_mm'])
        within = 0
        for state, record in labels['states'].items():
            xyz = record['center_lps_mm']
            delta = np.asarray(xyz)-plan
            inside = bool(np.max(abs(delta)) <= 10.+1e-6)
            within += int(state!='original' and inside)
            rows.append(dict(patient=result['patient'], state=state, source_folder=record['source_folder'],
                             center_x_mm=xyz[0],center_y_mm=xyz[1],center_z_mm=xyz[2],
                             kv1_u_px=record['projected_uv']['kv1'][0],kv1_v_px=record['projected_uv']['kv1'][1],
                             kv2_u_px=record['projected_uv']['kv2'][0],kv2_v_px=record['projected_uv']['kv2'][1],
                             dx_mm=float(delta[0]),dy_mm=float(delta[1]),dz_mm=float(delta[2]),
                             within_10mm_box=inside))
        q = labels['qc']
        lines.append('| {} | {} | {} | {} | {} | {:.2e} |'.format(
            result['patient'], '通过（复用）' if result['reused'] else '通过',q['states'],q['projected_points'],
            within,q['max_independent_centroid_difference_mm']))
    if rows:
        with (output/'centers.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines += ['', '检查包括：输入指纹与标定时一致；三维Mask非空且为0/1；物理网格与标定一致；',
              '体素均值转换的物理质心与ITK独立质心一致（≤1e-5mm）；Origin中心与计划中心一致；',
              '两个视角标签在完整DRR内；三角定位代数闭环误差≤1e-5mm；输入和矩阵未被改写。', '',
              '代数闭环是坐标实现自洽检查，不是新的独立标定精度或网络定位精度。',
              '二维目标是同一个三维中心的投影，不是二维投影Mask的质心。',
              'JSON将计划状态统一记作original；source_folder=Origin记录实际目录，未重命名原数据。', '',
              '## 正式输出', '',
              '- patients/患者ID/projected_centers.json：计划状态及所有phase的三维中心、两视角投影位置和来源指纹。',
              '- patients/患者ID/original_labels.json：只保留计划状态，预测时避免读取当前状态真值。',
              '- centers.csv：上述中心坐标的人类可读总表，三维单位mm、二维单位像素。',
              '- 本结论：每患者是否通过；不保存体积副本、可视化或检查过程文件。', '',
              'within_10mm_box只标注研究范围，不删除任何状态，也不指定训练/验证/测试归属。',
              '本次未生成患者划分清单、未适配新数据训练入口、未重新训练。', '']
    for result in results:
        if result['status']!='passed':
            lines += ['未通过：'+result['patient'],result['error'],'']
    (output/'中心标签质检结论.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DATA_ROOT)
    parser.add_argument('--geometry-root',type=Path,default='D:/AI_code/6nodule_tracking/nodule_tracking/outputs/geometry_error22_maskcenter')
    args = parser.parse_args()
    root = args.root.resolve()
    # output = (args.geometry_root or ROOT/'outputs'/('geometry_'+root.name)).resolve()
    output = (
            args.geometry_root or Path(r"D:/AI_code/6nodule_tracking/nodule_tracking/outputs/geometry_error22_maskcenter")
    ).resolve()

    if not root.is_dir() or not output.is_dir():
        raise ValueError('Data and previous geometry output directories must exist')
    patients = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if not patients:
        raise ValueError('No patient directories found')
    results = []
    for folder in patients:
        try:
            labels, reused = label_patient(folder,output/'patients'/folder.name)
            results.append(dict(patient=folder.name,status='passed',labels=labels,reused=reused))
            print('{}: {} states passed{}'.format(folder.name,len(labels['states']), ' (reused)' if reused else ''),flush=True)
        except Exception as error:
            results.append(dict(patient=folder.name,status='failed',error=str(error)))
            print(folder.name+': failed; see conclusion',flush=True)
    write_outputs(output,root,results)
    print('Results: '+str(output/'中心标签质检结论.md'),flush=True)
    if any(r['status']!='passed' for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
