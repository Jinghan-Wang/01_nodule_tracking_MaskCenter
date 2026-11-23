"""批量将各患者 original/MASKDRR_ROI_NULL 的肿瘤投影转成二值 Mask。

默认前景为灰度 > 0；输出 uint8（0/1），保留原图维度和空间信息。
兼容 Python 3.7，依赖 numpy、SimpleITK。
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_process  # 初始化当前Conda环境的DLL路径。

import numpy as np
import SimpleITK as sitk

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'Data' / 'LIDC_one'


def make_mask(source, target, threshold):
    """不裁剪、不缩放、不翻转，只对像素二值化。"""
    image = sitk.ReadImage(str(source))
    array = sitk.GetArrayFromImage(image)
    if image.GetNumberOfComponentsPerPixel() != 1 or not np.isfinite(array).all():
        raise ValueError('Expected finite scalar image')
    binary = (array > threshold).astype(np.uint8)
    foreground = int(binary.sum())
    if foreground == 0:
        print('WARNING: Empty binary Mask (no pixels above threshold): {}'.format(source))
    mask = sitk.GetImageFromArray(binary)
    mask.CopyInformation(image)
    target.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(mask, str(target), True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--threshold', type=float, default=0., help='前景条件：像素值严格大于阈值')
    parser.add_argument('--overwrite', action='store_true', help='重新生成已有输出；默认跳过')
    args = parser.parse_args()
    if not args.root.is_dir() or not np.isfinite(args.threshold):
        parser.error('Root must exist and threshold must be finite')
    patients = sorted(p for p in args.root.glob('LIDC_*') if p.is_dir())
    if not patients:
        parser.error('No LIDC_* patient folders found')
    generated = skipped = failed = 0
    for patient in patients:
        source_dir = patient / 'original' / 'MASKDRR_ROI_NULL'
        files = sorted(source_dir.glob('*.nii.gz')) + sorted(source_dir.glob('*.nii'))
        if not files:
            print('ERROR: No NIfTI files in {}'.format(source_dir))
            failed += 1
            continue
        for source in files:
            stem = source.name[:-7] if source.name.endswith('.nii.gz') else source.stem
            target = patient / 'original' / 'nodule_mask_DRR' / (stem + '_binary.nii.gz')
            if target.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                make_mask(source, target, args.threshold)
                generated += 1
            except (RuntimeError, ValueError, OSError) as error:
                print('ERROR: {}: {}'.format(source, error))
                failed += 1
    print('Patients: {}; generated: {}; skipped existing: {}; failed: {}'.format(
        len(patients), generated, skipped, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
