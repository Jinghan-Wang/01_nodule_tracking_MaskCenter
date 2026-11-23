"""统计单个MHA Mask或目录内全部MHA Mask的三维连通区域，不修改原文件。"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.count_mask_components import inspect_mask


# 也可以改成单个患者文件，例如该目录下的LIDC-IDRI-0171.mha。
INPUT_PATH = Path(r'C:\Users\OUR\Desktop\7_mask\mask')
OUTPUT_ROOT = ROOT / 'outputs' / 'mha_mask_components'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT_PATH)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    source = args.input.resolve()
    if source.is_file() and source.suffix.lower() == '.mha':
        paths = [source]
    elif source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == '.mha')
    else:
        raise ValueError('请指定MHA文件或包含MHA文件的目录：' + str(source))
    if not paths:
        raise ValueError('目录中没有MHA文件：' + str(source))
    output = (args.out or OUTPUT_ROOT / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary_fields = ['patient', 'mask_path', 'status', 'component_count',
                      'foreground_voxels', 'largest_fraction', 'all_to_largest_center_mm']
    part_fields = ['patient', 'component_rank', 'voxels', 'volume_mm3', 'fraction',
                   'center_x_mm', 'center_y_mm', 'center_z_mm']
    with (output / 'mask_summary.csv').open('w', encoding='utf-8-sig', newline='') as sf, \
            (output / 'components.csv').open('w', encoding='utf-8-sig', newline='') as pf:
        summary_writer = csv.DictWriter(sf, fieldnames=summary_fields)
        part_writer = csv.DictWriter(pf, fieldnames=part_fields)
        summary_writer.writeheader()
        part_writer.writeheader()
        for path in paths:
            summary, parts = inspect_mask(path)
            summary_writer.writerow(dict(patient=path.stem, mask_path=str(path), **summary))
            print('{}：{}个三维连通区域'.format(path.stem, summary['component_count']), flush=True)
            for part in parts:
                part_writer.writerow(dict(patient=path.stem, **part))
                print('  区域{}：{}个体素，体积{:.3f} mm³，占比{:.2%}'.format(
                    part['component_rank'], part['voxels'], part['volume_mm3'], part['fraction']))
    print('统计完成，输出：' + str(output))
    print('区域数不等于确诊结节数；微小区域也全部计入，需结合原图确认。')


if __name__ == '__main__':
    main()
