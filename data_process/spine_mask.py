#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量分割 LIDC 患者 Origin/CT.nii.gz 中的椎骨。

每位患者在输出根目录保存一个 <患者ID>.nii.gz，1 为脊柱，0 为背景。
"""

import csv
import os
import traceback
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np


# ==================== 只需按需修改这里 ====================
#DATASET_ROOT = Path("/data/yingxiang2_storage/DataSet/LIDCTrainMaskCenter")
DATASET_ROOT = Path("/data/yingxiang2_storage/DataSet/LIDCTestMaskCenter")
OUTPUT_ROOT = Path("/data/yingxiang2_storage/DataSet/LIDCTestMaskCenter_Spine_Mask")
DEVICE = "gpu"               # 没有 CUDA GPU 时改为 "cpu"
FAST = False                 # False：标准精度；True：较快、较低分辨率
SKIP_EXISTING = True         # 已存在同名患者掩膜时直接跳过
MAX_PATIENTS = None          # 首次测试可设为 2；正式运行设为 None
# ===========================================================

VERTEBRAE_CLASSES = [
    f"vertebrae_{level}{index}"
    for level, count in (("C", 7), ("T", 12), ("L", 5))
    for index in range(1, count + 1)
]


def load_ct_reference(ct_path: Path):
    """检查输入 CT 的维度和空间信息，不修改 CT 强度/HU。"""
    image = nib.load(str(ct_path))
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError(f"输入不是有效的三维 CT：shape={image.shape}")
    affine = image.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("输入 CT 的 affine 矩阵无效")
    return image


def save_label(data: np.ndarray, reference, destination: Path):
    """将 uint8 标签保存于 CT 原始体素网格，并继承 qform/sform。"""
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    header["cal_min"] = 0
    header["cal_max"] = int(data.max())

    image = nib.Nifti1Image(data, reference.affine, header=header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, str(destination))


def segment_one_patient(ct_path: Path, target_path: Path):
    """使用 TotalSegmentator 分割椎骨并合并为二值脊柱掩膜。"""
    from totalsegmentator.python_api import totalsegmentator

    reference = load_ct_reference(ct_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="_tmp_spine_", dir=str(target_path.parent)) as tmp:
        temporary_dir = Path(tmp)
        segmentation_path = temporary_dir / "vertebrae.nii.gz"
        totalsegmentator(
            input=str(ct_path),
            output=str(segmentation_path),
            task="total",
            roi_subset=VERTEBRAE_CLASSES,
            ml=True,
            fast=FAST,
            device=DEVICE,
            nr_thr_resamp=1,
            nr_thr_saving=1,
        )

        segmentation = nib.load(str(segmentation_path))
        if segmentation.shape != reference.shape or not np.allclose(
            segmentation.affine, reference.affine, atol=1e-4, rtol=0
        ):
            raise ValueError(f"脊柱 mask 与 CT 的空间网格不一致：{segmentation_path}")

        mask = (np.asanyarray(segmentation.dataobj) > 0).astype(np.uint8)
        count = int(np.count_nonzero(mask))
        if not 0 < count < mask.size:
            raise ValueError("脊柱 mask 必须同时包含前景和背景")

        # 先完整写入同一文件系统内的临时文件，再原子替换。
        working_file = temporary_dir / target_path.name
        save_label(mask, reference, working_file)
        os.replace(working_file, target_path)

    return count


def main():
    root = DATASET_ROOT.expanduser().resolve()
    output_root = OUTPUT_ROOT.expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"数据集根目录不存在：{root}")

    patient_dirs = sorted(
        p for p in root.glob("LIDC-IDRI-*") if p.is_dir()
    )
    if MAX_PATIENTS is not None:
        patient_dirs = patient_dirs[:MAX_PATIENTS]
    if not patient_dirs:
        raise RuntimeError(f"未找到 LIDC-IDRI-* 患者目录：{root}")

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "segmentation_log.csv"
    error_path = output_root / "segmentation_errors.txt"
    print(f"患者总数：{len(patient_dirs)}", flush=True)
    print(f"输入目录：{root}", flush=True)
    print(f"输出目录：{output_root}", flush=True)
    print("只读取每位患者的 Origin/CT.nii.gz", flush=True)

    fieldnames = ["patient_id", "status", "ct_path", "mask_path", "message", "label_1_voxels"]
    stats = {"success": 0, "skipped_existing": 0, "missing_ct": 0, "failed": 0}

    # 每处理完一例就写入 CSV，可直接查看进度和失败原因。
    with log_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        for idx, patient_dir in enumerate(patient_dirs, start=1):
            patient_id = patient_dir.name
            ct_path = patient_dir / "Origin" / "CT.nii.gz"
            target_path = output_root / f"{patient_id}.nii.gz"
            print(f"\n[{idx}/{len(patient_dirs)}] {patient_id}", flush=True)

            row = {
                "patient_id": patient_id,
                "ct_path": str(ct_path),
                "mask_path": str(target_path),
                "message": "",
            }
            try:
                if not ct_path.is_file():
                    row["status"] = "missing_ct"
                    row["message"] = "未找到 Origin/CT.nii.gz"
                    print(f"  跳过：{row['message']}", flush=True)

                elif SKIP_EXISTING and target_path.is_file():
                    row["status"] = "skipped_existing"
                    row["message"] = "已有分割文件，跳过"
                    print(f"  跳过：{target_path}", flush=True)

                else:
                    row["label_1_voxels"] = segment_one_patient(ct_path, target_path)
                    row["status"] = "success"
                    row["message"] = "完成"
                    print(f"  完成：{target_path}", flush=True)

            except Exception as exc:
                row["status"] = "failed"
                row["message"] = f"{type(exc).__name__}: {exc}"
                print(f"  失败：{row['message']}", flush=True)
                with error_path.open("a", encoding="utf-8") as error_file:
                    error_file.write(
                        f"\n{'=' * 60}\n{datetime.now().isoformat()}  {patient_id}\n"
                        f"{traceback.format_exc()}\n"
                    )

            writer.writerow(row)
            csv_file.flush()
            stats[row["status"]] += 1

    print("\n========== 批量处理结束 ==========", flush=True)
    print(f"成功：{stats['success']}", flush=True)
    print(f"已存在而跳过：{stats['skipped_existing']}", flush=True)
    print(f"缺少 CT：{stats['missing_ct']}", flush=True)
    print(f"失败：{stats['failed']}", flush=True)
    print(f"处理日志：{log_path}", flush=True)
    if stats["failed"]:
        print(f"错误详情：{error_path}", flush=True)
    print("标签：0 背景，1 脊柱", flush=True)


if __name__ == "__main__":
    main()
