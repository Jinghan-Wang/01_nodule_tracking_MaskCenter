#!/usr/bin/env python3
"""按 Excel 第一张工作表 C 列移动患者目录。依赖：pip install openpyxl。

默认实际移动；传入 --dry-run 只检查并输出报告。请勿并发运行本脚本。
"""
import argparse
import csv
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook


def read_ids(excel, sheet=None):
    wb = load_workbook(excel, read_only=True, data_only=True)
    try:
        ws = wb[sheet] if sheet else wb.worksheets[0]
        title = ws.title
        if str(ws.cell(1, 3).value or '').strip() != 'LIDC患者编号':
            raise ValueError(f'{title} 的 C1 不是“LIDC患者编号”，请检查表头。')
        ids = []
        seen = set()
        for row, (value,) in enumerate(ws.iter_rows(min_row=2, min_col=3,
                                                   max_col=3, values_only=True), 2):
            if value is None or not str(value).strip():
                continue
            patient_id = str(value).strip()
            # 严格校验目录名，保留编号前导零，禁止路径穿越。
            if not re.fullmatch(r'LIDC-IDRI-\d{4}', patient_id):
                raise ValueError(f'{title}!C{row} 的编号格式无效：{patient_id!r}')
            if patient_id not in seen:
                ids.append(patient_id)
                seen.add(patient_id)
        if not ids:
            raise ValueError('C 列没有可用患者编号。')
        return title, ids
    finally:
        wb.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--excel', required=True, type=Path, help='Excel 文件路径')
    parser.add_argument('--sheet', help='工作表名称；默认第一个工作表')
    parser.add_argument('--source', type=Path,
                        default=Path('F:\AAA\LIDCTrainDataSet'))
    parser.add_argument('--target', type=Path,
                        default=Path('F:\AAA\Delete_data'))
    parser.add_argument('--report-dir', type=Path, default=Path('lidc_move_reports'))
    parser.add_argument('--dry-run', action='store_true', help='预演，不移动数据')
    args = parser.parse_args()
    title, ids = read_ids(args.excel, args.sheet)
    source, target = args.source.resolve(), args.target.resolve()
    if not source.is_dir():
        raise ValueError(f'源目录不存在或不是目录：{source}')
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('源目录与目标目录不能相同或相互包含。')
    if target.exists() and not target.is_dir():
        raise ValueError(f'目标路径不是目录：{target}')
    # 每次建立独立报告目录，重复运行也不会覆盖历史缺失记录。
    report = args.report_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    report.mkdir(parents=True, exist_ok=False)
    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    print(f'工作表：{title}；去重后患者数：{len(ids)}；预演：{args.dry_run}')
    with (report / 'results.csv').open('w', encoding='utf-8-sig', newline='') as log, \
            (report / 'missing_ids.txt').open('w', encoding='utf-8') as missing:
        writer = csv.writer(log)
        writer.writerow(['患者编号', '状态', '说明'])
        for patient_id in ids:
            src, dst = source / patient_id, target / patient_id
            detail = ''
            if not src.exists() and not src.is_symlink():
                status = '源目录缺失'
                detail = '目标中已有同名路径' if dst.exists() or dst.is_symlink() else ''
                missing.write(patient_id + '\n')
                missing.flush()
            elif src.is_symlink() or not src.is_dir():
                status = '跳过异常路径'
                detail = '源路径不是普通目录，或为符号链接'
            elif dst.exists() or dst.is_symlink():
                status = '目标已存在'
                detail = '跳过，不覆盖或合并'
            elif args.dry_run:
                status = '待移动'
            else:
                try:
                    shutil.move(str(src), str(dst))
                    status = '移动成功'
                except (OSError, shutil.Error) as exc:
                    status = '移动失败'
                    detail = str(exc)
            writer.writerow([patient_id, status, detail])
            log.flush()
            counts[status] += 1
            print(f'{patient_id}: {status} {detail}')
    print(f'统计：{dict(counts)}\n报告目录：{report.resolve()}')
    return 1 if counts['移动失败'] or counts['跳过异常路径'] else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        sys.exit(1)
