"""遍历患者目录，用真实 Origin Mask 固定等中心并生成双视角投影矩阵。"""
import argparse
import configparser
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data_process
import numpy as np
import SimpleITK as sitk
from data_process.calibrate_drr_demo import (
    FILENAMES, DemoRunner, calibrate, plan_folder_name, projection_pair, read_pose, save_json)
from data_process.generate_projection_matrices import input_digest, read_json, write_conclusion
from data_process.prepare_center_labels import sha256

DATA_ROOT = Path(r'F:\AIDataSet\Nodule_tracking\Error_ID')
DEMO_ROOT = ROOT.parent/'nodule_tracking_ISO'/'Demo'
OUTPUT_ROOT = ROOT/'outputs'/'geometry_maskcenter'


class MaskCenterRunner(DemoRunner):
    """让 Demo 扫描 Origin 和 phase_0000，人工体模共用真实计划中心。"""

    def __init__(self, out, source):
        super().__init__(out, source)
        self.state = self.patient/'phase_0000'
        self.state.mkdir()

    def run(self, source_patient, ct, mask, tag, project_ct=False, expected_pose=None):
        origin = self.patient/'Origin'
        if not origin.exists():
            origin.mkdir()
            source = source_patient/plan_folder_name(source_patient)
            for name in ('CT', 'mask'):
                image = sitk.ReadImage(str(source/(name+'.nii.gz')))
                for key in image.GetMetaDataKeys():
                    image.EraseMetaData(key)
                sitk.WriteImage(image, str(origin/(name+'.mhd')))
        for name in ('congfigfiles_drr_LIDC.ini', 'angle_all.txt'):
            shutil.copy2(str(source_patient/name), str(self.patient/name))
        for image, name in ((ct, 'CT.mhd'), (mask, 'mask.mhd')):
            clean = sitk.Image(image)
            for key in clean.GetMetaDataKeys():
                clean.EraseMetaData(key)
            sitk.WriteImage(clean, str(self.state/name))
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(str(self.source/'paraset.ini'), encoding='utf-8-sig')
        config['Batch'].update(Mode='Single', SinglePatientRoot=str(self.patient.resolve()),
                               DataDirectoryPattern='*')
        config['Inputs'].update(ProcessCT=str(project_ct).lower(), ProcessMask='true')
        with (self.work/'paraset.ini').open('w', encoding='utf-8') as stream:
            config.write(stream)
        # 每次清除当前体模投影，禁止误读上一次调用留下的文件。
        for prefix in ('CT', 'MASK'):
            for path in (self.state/(prefix+'DRR_ROI_NULL')).glob('*.nii.gz'):
                path.unlink()
        pose_path = self.state/'paraset.ini'
        if pose_path.exists():
            pose_path.unlink()
        process = subprocess.run([str((self.work/'DRRDemo.exe').resolve())],
            cwd=str(self.work), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)
        (self.work/(tag+'.stdout.txt')).write_bytes(process.stdout)
        if process.returncode:
            raise RuntimeError('Demo 退出码 {}：{}\n{}'.format(process.returncode, tag,
                               process.stdout.decode('mbcs', errors='replace')[-2000:]))
        if expected_pose is None:
            raise ValueError('本脚本要求 MaskCenter 配置和固定计划姿态')
        actual = read_pose(pose_path)
        if not np.allclose(actual, expected_pose, atol=1e-5, rtol=0):
            raise ValueError('实际 MaskCenter 姿态与保存数据不符：{}；实际={}；预期={}'.format(
                tag, actual.tolist(), expected_pose.tolist()))
        return projection_pair(self.state, 'MASK'), pose_path.read_text(encoding='utf-8-sig')


def prepare_patient(folder, destination, demo, output, recipe):
    """沿用32点标定及全部状态的逐像素验收，仅替换 DemoRunner。"""
    digest = input_digest(folder)
    matrix = destination/'geometry.json'
    if matrix.exists():
        old = read_json(matrix)
        if old.get('geometry_verified') and old.get('input_digest') == digest and old.get('recipe') == recipe:
            return old, True
        raise ValueError('已有矩阵来源不同，请指定新的输出目录：'+str(matrix))
    with tempfile.TemporaryDirectory(prefix='_temporary_', dir=str(output)) as temporary:
        work = Path(temporary)
        runner = MaskCenterRunner(work/'runtime', demo)
        with contextlib.redirect_stdout(io.StringIO()):
            geometry = calibrate(folder, work/'calibration', runner, count=32, export_labels=False)
        audit = read_json(work/'calibration'/'saved_data_audit.json')
        fingerprints = read_json(work/'calibration'/'source_fingerprints.json')
        qc = dict(fiducials=32, fit_points=24, independent_points=8,
            max_projection_error_px=geometry['heldout_reprojection_max_px'],
            max_reconstruction_error_mm=geometry['heldout_reconstruction_max_mm'],
            states_checked=len({r['state'] for r in audit}), paired_views_checked=len(audit),
            minimum_mask_iou=min(r['mask']['mask_iou'] for r in audit),
            maximum_ct_mean_abs_difference=max(r['ct']['mean_abs_difference'] for r in audit),
            maximum_mask_mean_abs_difference=max(r['mask']['mean_abs_difference'] for r in audit),
            all_ct_pixels_identical=all(r['ct']['identical_pixels'] for r in audit),
            all_mask_pixels_identical=all(r['mask']['identical_pixels'] for r in audit),
            source_files_unchanged=fingerprints['unchanged'])
        if not (geometry['geometry_verified'] and qc['source_files_unchanged'] and
                qc['max_projection_error_px'] <= .5 and qc['max_reconstruction_error_mm'] <= 1 and
                qc['maximum_mask_mean_abs_difference'] <= 1e-5):
            raise ValueError('几何验收未通过：'+str(qc))
        geometry.update(template_folder=plan_folder_name(folder), views=['kv1', 'kv2'],
            angles_deg=[0., 150.], input_digest=digest, recipe=recipe, qc=qc,
            pixel_convention='zero-based pixel centres; u right, v down; image[v,u]',
            projection_formula='h=P@[x,y,z,1]; u=h[0]/h[2], v=h[1]/h[2]',
            method='actual Demo fiducials; fixed real Origin MaskCenter; normalized DLT')
        destination.mkdir(parents=True, exist_ok=True)
        save_json(matrix, geometry)
        return geometry, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATA_ROOT)
    parser.add_argument('--demo', type=Path, default=DEMO_ROOT)
    parser.add_argument('--out', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--qt-bin', type=Path,
                        default=Path(r'C:\Qt\Qt5.15.2\5.15.2\msvc2019_64\bin'))
    args = parser.parse_args()
    root, demo, output = args.root.resolve(), args.demo.resolve(), args.out.resolve()
    if output == root or root in output.parents:
        parser.error('输出目录必须位于原始数据之外')
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    patients = sorted(folder for folder in root.iterdir()
                      if folder.is_dir() and not folder.name.startswith('.'))
    if not patients:
        raise ValueError('数据根目录下没有患者文件夹：'+str(root))
    # 入口确认中心模式，防止使用其他 Demo 配置生成错误几何。
    config = configparser.ConfigParser()
    config.read(str(demo/'paraset.ini'), encoding='utf-8-sig')
    if config.get('DRR', 'ProjectionCenter').lower() != 'maskcenter':
        raise ValueError('Demo 必须配置 ProjectionCenter=MaskCenter')
    os.environ['PATH'] = str(args.qt_bin)+os.pathsep+os.environ['PATH']
    recipe = dict(count=32, max_projection_px=.5, max_reconstruction_mm=1.,
        center_method='real Origin MaskCenter',
        binaries={name:sha256(demo/name) for name in FILENAMES+('paraset.ini',)},
        implementation={str(path.relative_to(ROOT)):sha256(path) for path in (
            Path(__file__).resolve(), ROOT/'data_process'/'calibrate_drr_demo.py', ROOT/'geometry.py')})
    output.mkdir(parents=True, exist_ok=True)
    records = []
    print('数据目录：'+str(root), flush=True)
    print('Demo目录：'+str(demo), flush=True)
    print('输出目录：'+str(output), flush=True)
    print('患者数：'+str(len(patients)), flush=True)
    for folder in patients:
        print('Calibrating '+folder.name+' ...', flush=True)
        try:
            geometry, reused = prepare_patient(folder, output/'patients'/folder.name, demo, output, recipe)
        except (ValueError, RuntimeError) as error:
            records.append(dict(patient=folder.name, status='failed', error=str(error)))
            write_conclusion(output, root, records)
            raise
        records.append(dict(patient=folder.name, status='passed', reused=reused, qc=geometry['qc']))
        write_conclusion(output, root, records)
        print(folder.name+': passed', flush=True)
    print('完成：'+str(output/'质检结论.md'), flush=True)


if __name__ == '__main__':
    main()
