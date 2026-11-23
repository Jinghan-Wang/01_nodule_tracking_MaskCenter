"""实际Demo标定双视角矩阵。PyCharm直接运行；只保留矩阵与质检结论。"""
import argparse
import contextlib
import hashlib
import io
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import data_process
from data_process.calibrate_drr_demo import (
    DEMO, FILENAMES, DemoRunner, calibrate, patient_fingerprints, plan_folder_name, save_json)
from data_process.prepare_center_labels import sha256


# 后续将此处改为400多例训练患者所在目录，其下直接放各患者文件夹。
# DATA_ROOT = ROOT/'Data'/'New_data'
DATA_ROOT = "D:/AI_code/6nodule_tracking/nodule_tracking/datasets/data"
DEMO_ROOT = "D:/AI_code/6nodule_tracking/nodule_tracking_ISO/Demo"
FIDUCIAL_COUNT = 32              # 前24点拟合，后8点独立验证
MAX_PROJECTION_ERROR_PX = 0.5
MAX_RECONSTRUCTION_ERROR_MM = 1.


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def input_digest(folder):
    values = patient_fingerprints(folder)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode('utf-8')).hexdigest()


def calibrate_patient(folder, destination, demo, output, recipe):
    """只有全部通过才写正式geometry.json；临时体模与日志自动清理。"""
    plan_name = plan_folder_name(folder)
    digest = input_digest(folder)
    matrix_path = destination/'geometry.json'
    if matrix_path.exists():
        old = read_json(matrix_path)
        if (old.get('geometry_verified') is True and old.get('input_digest') == digest and
                old.get('recipe') == recipe):
            return old, True
        raise ValueError('Existing matrix provenance differs; use a new --out to preserve the old result')

    # 临时目录明确位于本次输出根目录，cleanup只删除这个新建目录。
    with tempfile.TemporaryDirectory(prefix='_temporary_', dir=str(output)) as temporary:
        work = Path(temporary).resolve()
        if work.parent != output.resolve():
            raise ValueError('Temporary directory escaped output root')
        runner = DemoRunner(work/'runtime', demo)
        # 旧工具的逐点输出仅留在内存，不产生正式过程报告。
        with contextlib.redirect_stdout(io.StringIO()):
            geometry = calibrate(folder, work/'calibration', runner, FIDUCIAL_COUNT, export_labels=False)
        audit = read_json(work/'calibration'/'saved_data_audit.json')
        fingerprints = read_json(work/'calibration'/'source_fingerprints.json')
        qc = dict(
            fiducials=FIDUCIAL_COUNT, fit_points=FIDUCIAL_COUNT*3//4,
            independent_points=FIDUCIAL_COUNT-FIDUCIAL_COUNT*3//4,
            max_projection_error_px=geometry['heldout_reprojection_max_px'],
            max_reconstruction_error_mm=geometry['heldout_reconstruction_max_mm'],
            states_checked=len({r['state'] for r in audit}), paired_views_checked=len(audit),
            minimum_mask_iou=min(r['mask']['mask_iou'] for r in audit),
            maximum_ct_mean_abs_difference=max(r['ct']['mean_abs_difference'] for r in audit),
            maximum_mask_mean_abs_difference=max(r['mask']['mean_abs_difference'] for r in audit),
            all_ct_pixels_identical=all(r['ct']['identical_pixels'] for r in audit),
            all_mask_pixels_identical=all(r['mask']['identical_pixels'] for r in audit),
            source_files_unchanged=fingerprints['unchanged'])
        passed = (geometry['geometry_verified'] and fingerprints['unchanged'] and
                  qc['max_projection_error_px'] <= MAX_PROJECTION_ERROR_PX and
                  qc['max_reconstruction_error_mm'] <= MAX_RECONSTRUCTION_ERROR_MM and
                  qc['maximum_mask_mean_abs_difference'] <= 1e-5)
        if not passed:
            raise ValueError('Geometry validation failed: '+json.dumps(qc))
        geometry.update(template_folder=plan_name, views=['kv1','kv2'], angles_deg=[0.,150.],
                        pixel_convention='zero-based pixel centres; u right, v down; image[v,u]',
                        projection_formula='h=P@[x,y,z,1]; u=h[0]/h[2], v=h[1]/h[2]',
                        method='actual Demo fiducials; normalized DLT; held-out and saved-image validation',
                        input_digest=digest, recipe=recipe, qc=qc)
        destination.mkdir(parents=True, exist_ok=True)
        save_json(matrix_path, geometry)
        return geometry, False


def write_conclusion(output, root, records):
    lines = ['# 双视角投影矩阵质检结论', '',
             '数据根目录：'+str(root),
             '方法：实际DRRDemo.exe调用DRR.dll投影已知位置体模，Python拟合矩阵。',
             '32个点中24个拟合、8个独立验证；逐状态重投影CT和Mask，对照已有DRR。',
             '几何局部标定范围为计划中心附近CT各轴±18mm，覆盖当前±10mm研究范围；不是搜索范围变更。', '',
             '| 患者 | 结论 | 最大投影误差px | 最大三维重建误差mm | 核对状态数 | 最小Mask IoU | CT/Mask像素完全一致 |',
             '|---|---|---:|---:|---:|---:|---|']
    for record in records:
        if record['status'] == 'passed':
            q = record['qc']
            lines.append('| {} | {} | {:.4f} | {:.4f} | {} | {:.4f} | {} / {} |'.format(
                record['patient'], '通过（复用）' if record['reused'] else '通过',
                q['max_projection_error_px'], q['max_reconstruction_error_mm'],q['states_checked'],
                q['minimum_mask_iou'],q['all_ct_pixels_identical'],q['all_mask_pixels_identical']))
        else:
            lines.append('| {} | 未通过 | — | — | — | — | — |'.format(record['patient']))
    lines += ['', '阈值：独立投影误差≤0.5px、三维重建误差≤1mm、保存Mask IoU≥0.98，',
              'CT和Mask重投影平均绝对像素差均≤1e-5，原输入文件内容未改变。以上为本次工程核验阈值，不是定位精度结论。', '',
              '每患者patients/患者ID/geometry.json中，cameras[0]为kv1、cameras[1]为kv2，均为3×4矩阵。',
              '输入为患者LPS物理坐标（mm），输出为完整DRR的零起点像素坐标；后续裁剪时需另加坐标变换。',
              '只验证几何，不生成训练中心标签，不训练网络；不代表Dynagan形变本身已完成物理验证。',
              '程序、配置、输入指纹与简短质检数字保存在矩阵文件中，用于防止复用过期结果。',
              '体模、重投影图、Demo副本和过程日志均为临时文件，处理结束后清理。', '']
    for record in records:
        if record['status'] != 'passed':
            lines += ['未通过：'+record['patient'], record['error'], '']
    (output/'质检结论.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATA_ROOT)
    parser.add_argument('--demo', type=Path, default=DEMO_ROOT)
    parser.add_argument('--out', type=Path, default="outputs/geometry_target_center")
    args = parser.parse_args()
    root, demo = args.root.resolve(), args.demo.resolve()
    # output = (args.out or ROOT/'outputs'/('geometry_'+root.name)).resolve()
    output = (args.out or Path(r"outputs/geometry_target_center")).resolve()
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    if output == root or root in output.parents:
        raise ValueError('Choose an output outside the original dataset')
    # 输入原程序只读；实际执行的是临时目录里的副本。
    recipe = dict(count=FIDUCIAL_COUNT, max_projection_px=MAX_PROJECTION_ERROR_PX,
                  max_reconstruction_mm=MAX_RECONSTRUCTION_ERROR_MM,
                  binaries={name:sha256(demo/name) for name in FILENAMES+('paraset.ini',)},
                  implementation={str(path.relative_to(ROOT)):sha256(path) for path in (
                      Path(__file__).resolve(),ROOT/'data_process'/'calibrate_drr_demo.py',ROOT/'geometry.py')})
    patients = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if not patients:
        raise ValueError('No patient directories found')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    print('Data: {}\nOutput: {}\nPatients: {}'.format(root,output,len(patients)),flush=True)
    for folder in patients:
        print('Calibrating '+folder.name+' ...',flush=True)
        try:
            geometry, reused = calibrate_patient(folder, output/'patients'/folder.name, demo, output, recipe)
            records.append(dict(patient=folder.name, status='passed', reused=reused, qc=geometry['qc']))
            print(folder.name+': passed'+(' (reused)' if reused else ''),flush=True)
        except Exception as error:
            records.append(dict(patient=folder.name,status='failed',error=str(error)))
            print(folder.name+': failed; see conclusion',flush=True)
        write_conclusion(output,root,records)
    print('Conclusion: '+str(output/'质检结论.md'),flush=True)
    if any(r['status']!='passed' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
