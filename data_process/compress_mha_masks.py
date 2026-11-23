"""将MHA Mask无损压缩到独立目录，保留编辑后的体素值和数据类型，使用old目录同名文件的空间几何信息。"""
import argparse
import os
import sys
from pathlib import Path

# 支持Windows下在PyCharm中直接使用Conda解释器。
if os.name == 'nt':
    environment = Path(sys.executable).parent
    os.environ['PATH'] = os.pathsep.join([
        str(environment), str(environment / 'Library' / 'bin'), os.environ.get('PATH', '')])

import SimpleITK as sitk


INPUT_DIR = Path(r'C:\Users\OUR\Desktop\7_mask\new')
REFERENCE_DIR = Path(r'C:\Users\OUR\Desktop\7_mask\old')
OUTPUT_DIR = Path(r'C:\Users\OUR\Desktop\7_mask\1compressed_fixed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT_DIR)
    parser.add_argument('--reference', type=Path, default=REFERENCE_DIR,
                        help='提供同名MHA文件空间几何信息的参考目录')
    parser.add_argument('--out', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.out.resolve()
    reference_dir = args.reference.resolve()
    if not reference_dir.is_dir():
        raise NotADirectoryError(reference_dir)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if output == reference_dir:
        raise ValueError('输出目录不能与参考目录相同，避免覆盖参考文件')
    if source == output:
        raise ValueError('输出目录必须与输入目录不同，避免覆盖原文件')
    paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == '.mha')
    if not paths:
        raise ValueError('输入目录中没有MHA文件：' + str(source))
    # 写入前检查全部参考文件和目标文件，不覆盖已有输出。
    for path in paths:
        reference_path = reference_dir / path.name
        if not reference_path.is_file():
            raise FileNotFoundError('缺少同名参考文件：' + str(reference_path))
        destination = output / path.name
        if destination.exists():
            raise FileExistsError('不覆盖已有文件：' + str(destination))
    output.mkdir(parents=True, exist_ok=True)
    total_before, total_after = 0, 0
    for path in paths:
        image = sitk.ReadImage(str(path))
        destination = output / path.name
        reference = sitk.ReadImage(str(reference_dir / path.name))
        if image.GetDimension() != reference.GetDimension() or image.GetSize() != reference.GetSize():
            raise ValueError('输入mask与参考文件维度或尺寸不一致：' + path.name)
        # 保留编辑后的体素值和类型，仅复制原点、间距、方向；不重采样。
        image.CopyInformation(reference)
        # 同步附加的历史几何字段，避免与有效几何参数矛盾。
        for key in ('ITK_original_spacing', 'ITK_original_direction'):
            if reference.HasMetaDataKey(key):
                image.SetMetaData(key, reference.GetMetaData(key))
            elif image.HasMetaDataKey(key):
                image.EraseMetaData(key)
        sitk.WriteImage(image, str(destination), useCompression=True)
        before, after = path.stat().st_size, destination.stat().st_size
        total_before += before
        total_after += after
        print('{}：{:.2f} MB → {:.2f} KB'.format(
            path.name, before / 1024 ** 2, after / 1024), flush=True)
    print('完成{}个文件：合计{:.2f} MB → {:.2f} MB'.format(
        len(paths), total_before / 1024 ** 2, total_after / 1024 ** 2))
    print('几何参考目录：' + str(reference_dir))
    print('输出目录：' + str(output))


if __name__ == '__main__':
    main()
