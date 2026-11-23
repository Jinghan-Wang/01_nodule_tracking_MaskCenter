"""统计所有患者各状态三维二值Mask的连通区域，不改写原始数据。"""
import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# 支持Windows下在PyCharm中直接选择Conda解释器运行。
if os.name == 'nt':
    environment = Path(sys.executable).parent
    os.environ['PATH'] = os.pathsep.join([
        str(environment), str(environment / 'Library' / 'bin'), os.environ.get('PATH', '')])

import numpy as np
import SimpleITK as sitk


DATA_ROOT = Path('/data/yingxiang2_storage/LIDCTestDataSet/TestDataSet')
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'mask_components'


def inspect_mask(path):
    """采用三维26邻域；不按大小删除任何连通区域。"""
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayViewFromImage(image)
    if image.GetDimension() != 3:
        raise ValueError('Mask必须是三维：' + str(path))
    if not np.isfinite(array).all() or not np.all((array == 0) | (array == 1)):
        raise ValueError('Mask必须只包含有限的0和1：' + str(path))
    # 面、边、角相接的前景体素均视为相连。
    connected = sitk.ConnectedComponent(sitk.Cast(image == 1, sitk.sitkUInt8), True)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(connected)
    labels = sorted(statistics.GetLabels(),
                    key=statistics.GetNumberOfPixels, reverse=True)
    total = sum(statistics.GetNumberOfPixels(label) for label in labels)
    parts = []
    for rank, label in enumerate(labels, 1):
        count = statistics.GetNumberOfPixels(label)
        center = statistics.GetCentroid(label)
        parts.append(dict(component_rank=rank, voxels=count,
                          volume_mm3=statistics.GetPhysicalSize(label),
                          fraction=count / total,
                          center_x_mm=center[0], center_y_mm=center[1], center_z_mm=center[2]))
    if not parts:
        return dict(status='empty', component_count=0, foreground_voxels=0,
                    largest_fraction='', all_to_largest_center_mm=''), parts
    centers = np.asarray([[p['center_x_mm'], p['center_y_mm'], p['center_z_mm']]
                          for p in parts])
    counts = np.asarray([p['voxels'] for p in parts])
    all_center = np.average(centers, axis=0, weights=counts)
    return dict(status='ok', component_count=len(parts), foreground_voxels=total,
                largest_fraction=parts[0]['fraction'],
                all_to_largest_center_mm=float(np.linalg.norm(all_center - centers[0]))), parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATA_ROOT)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    patients = sorted(p for p in root.glob('LIDC*') if p.is_dir())
    if not patients:
        raise ValueError('数据根目录下未找到LIDC患者文件夹')
    output = (args.out or OUTPUT_ROOT / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    # 每次使用新目录，避免覆盖已有统计结果。
    output.mkdir(parents=True, exist_ok=False)
    mask_fields = ['patient', 'state', 'mask_path', 'status', 'component_count',
                   'foreground_voxels', 'largest_fraction', 'all_to_largest_center_mm']
    part_fields = ['patient', 'state', 'component_rank', 'voxels', 'volume_mm3',
                   'fraction', 'center_x_mm', 'center_y_mm', 'center_z_mm']
    mask_rows = []
    with (output / 'mask_summary.csv').open('w', encoding='utf-8-sig', newline='') as mf, \
            (output / 'components.csv').open('w', encoding='utf-8-sig', newline='') as cf:
        mask_writer = csv.DictWriter(mf, fieldnames=mask_fields)
        part_writer = csv.DictWriter(cf, fieldnames=part_fields)
        mask_writer.writeheader()
        part_writer.writeheader()
        for index, patient in enumerate(patients, 1):
            originals = [patient / name for name in ('Origin', 'original')
                         if (patient / name).is_dir()]
            phases = sorted(p for p in patient.glob('phase_*') if p.is_dir())
            if len(originals) != 1 or not phases:
                raise ValueError('患者必须有唯一Origin/original目录及phase目录：' + str(patient))
            for state in originals + phases:
                path = state / 'mask.nii.gz'
                summary, parts = inspect_mask(path)
                row = dict(patient=patient.name, state=state.name,
                           mask_path=path.relative_to(root).as_posix(), **summary)
                mask_writer.writerow(row)
                mask_rows.append(row)
                for part in parts:
                    part_writer.writerow(dict(patient=patient.name, state=state.name, **part))
            print('[{}/{}] {}'.format(index, len(patients), patient.name), flush=True)
    multiple = [r for r in mask_rows if r['component_count'] > 1]
    empty = [r for r in mask_rows if r['component_count'] == 0]
    lines = ['# 三维Mask连通区域统计', '', '数据根目录：' + str(root), '',
             '- 患者数：{}'.format(len(patients)),
             '- Mask数（Origin/original及所有phase）：{}'.format(len(mask_rows)),
             '- 含多个连通区域的Mask数：{}'.format(len(multiple)),
             '- 含多个连通区域的患者数：{}'.format(len({r['patient'] for r in multiple})),
             '- 空Mask数：{}'.format(len(empty)), '',
             '先看mask_summary.csv：每行一个状态，component_count为三维26邻域连通区域数。',
             'largest_fraction为最大区域占前景体素的比例；all_to_largest_center_mm为全部前景质心与最大区域质心的距离。',
             '再看components.csv：每行一个区域，按体素数降序编号，列出体积（mm³）、比例及患者LPS物理坐标质心（mm）。', '',
             '连通区域数不等于确诊结节数：小碎片可能是假标注，同一个结节也可能因分割断裂形成多个区域。',
             '最大区域不自动认定为目标；本脚本不删除小区域、不修改Mask、中心标签或投影矩阵。',
             '空Mask记录为empty；缺失、损坏或非二值Mask直接报错，不静默跳过。',
             '只有程序正常结束并生成本说明时，本次统计才完整。']
    (output / '统计说明.md').write_text('\n'.join(lines), encoding='utf-8')
    print('完成：{}位患者，{}个Mask，其中{}个多区域、{}个空Mask。'.format(
        len(patients), len(mask_rows), len(multiple), len(empty)))
    print('输出：' + str(output))


if __name__ == '__main__':
    main()
