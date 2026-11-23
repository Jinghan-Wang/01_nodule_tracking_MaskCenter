"""直接运行：复用已验证数据，准备并检查±10mm固定搜索ROI；不训练、不重投影。"""
import csv
import json
import tempfile
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train  # 初始化pytorch_3.7的Windows DLL路径。
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from data_loader import read_pair
from geometry import project, response_camera
from lidc_data import load_datasets
from network import logits_to_scores, response_loss
from run_localization import load_pair


PREPARATION = train.ROOT/'outputs'/'lidc_demo_geometry'
CHECKPOINT = train.ROOT/'outputs'/'template_margin_comparison'/'20261015_154208'/'margin_22'/'best.pt'
OUTPUT = train.ROOT/'outputs'/'local_search_roi'
RADIUS_MM = 10.
MARGIN = 22


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_example(patient, datasets, root, out):
    """一张图展示两视角的模板、固定搜索框和实际裁剪输入。"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, view in enumerate(train.VIEWS):
        ds = datasets[(patient, view)]
        sample = ds[0]
        full, mask, _, _ = read_pair(root/patient/ds.phases[0], view)
        x0, y0, x1, y1 = ds.search_crop_box
        local = ds.template_center-ds.crop_origin
        plan = ds.template_center-ds.search_origin
        axes[row, 0].imshow(ds.template, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].contour(ds.template_mask, [.5], colors='cyan')
        axes[row, 0].plot(*local, 'yo', mfc='none', label='Plan center')
        axes[row, 0].set_title(view+' original template / margin=22')
        axes[row, 1].imshow(ds.normalize(full), cmap='gray', vmin=0, vmax=1)
        axes[row, 1].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, color='orange'))
        axes[row, 1].set_title(view+' fixed search ROI / all phases')
        axes[row, 2].imshow(sample['search'][0], cmap='gray', vmin=0, vmax=1)
        axes[row, 2].contour(sample['search_mask'][0], [.5], colors='cyan')
        axes[row, 2].plot(*plan, 'yo', mfc='none', label='Plan center')
        axes[row, 2].plot(*sample['search_center_uv'], 'g+', label='GT (display only)')
        axes[row, 2].set_title(view+' actual cropped input / '+ds.phases[0])
        axes[row, 2].legend(fontsize=8)
    fig.suptitle(patient+' | no resizing | current Mask never defines the ROI')
    fig.tight_layout()
    fig.savefig(str(out/'example.png'), dpi=140)
    plt.close(fig)


@torch.no_grad()
def main():
    manifest = read_json(PREPARATION/'dataset.json')
    root = Path(manifest['root'])
    checkpoint = torch.load(str(CHECKPOINT), map_location='cpu')
    if checkpoint['config']['margin'] != MARGIN:
        raise ValueError('Use the matching margin22 baseline')
    torch.set_num_threads(4)
    model = train.SiamFC().eval()
    model.load_state_dict(checkpoint['model'])
    OUTPUT.mkdir(exist_ok=True)
    boxes, exceptions, contour_losses = [], [], []
    max_label_error, max_map_error, max_logit_error = 0., 0., 0.
    state_count, views_checked, loss_checked = 0, 0, 0
    # 固定计划盒密集采样，确认所有候选投影均落在有效响应范围内。
    offsets = np.asarray(list(product(np.linspace(-RADIUS_MM, RADIUS_MM, 11), repeat=3)))
    for split, patients in manifest['patient_split'].items():
        for patient in patients:
            labels = read_json(PREPARATION/'patients'/patient/'projected_centers.json')
            plan = np.asarray(labels['states']['original']['center_lps_mm'])
            # 先只用original建立ROI；当前状态真值随后才用于检查。
            prediction_sets = load_datasets(root, PREPARATION, [patient], MARGIN,
                                            with_targets=False, search_radius_mm=RADIUS_MM)
            for phase in prediction_sets[(patient, 'kv1')].phases:
                delta = np.asarray(labels['states'][phase]['center_lps_mm'])-plan
                state_count += 1
                if np.max(abs(delta)) > RADIUS_MM+1e-6:
                    exceptions.append(dict(split=split, patient=patient, phase=phase,
                                           dx_mm=delta[0], dy_mm=delta[1], dz_mm=delta[2],
                                           max_axis_mm=float(np.max(abs(delta)))))
            supervised = load_datasets(root, PREPARATION, [patient], MARGIN,
                                        with_targets=True, search_radius_mm=RADIUS_MM)
            for view in train.VIEWS:
                ds, target_ds = prediction_sets[(patient, view)], supervised[(patient, view)]
                assert ds.search_crop_box == target_ds.search_crop_box
                sample = ds[0]
                assert 'search_center_uv' not in sample and 'search_mask' not in sample
                template, search = sample['template'][None], sample['search'][None]
                logits = model(template, search)[0, 0]
                assert torch.isfinite(logits).all()
                camera = labels['cameras'][0 if view == 'kv1' else 1]
                projected = project(plan+offsets, camera)[0]
                mapped = project(plan+offsets, response_camera(camera, ds.response_origin, 4))[0]
                max_map_error = max(max_map_error, float(np.max(abs(mapped-(projected-ds.response_origin)/4))))
                assert (mapped >= 0).all() and (mapped <= np.asarray(logits.shape[::-1])-1).all()
                full = ds.normalize(read_pair(root/patient/ds.phases[0], view, with_mask=False)[0])
                full_logits = model(template, torch.from_numpy(full)[None, None])[0, 0]
                # 只比较两种输入共同覆盖的响应；4像素对齐保证网格相同。
                ox, oy = (ds.search_origin/4).astype(int)
                left, top = max(0, ox), max(0, oy)
                right = min(full_logits.shape[1], ox+logits.shape[1])
                bottom = min(full_logits.shape[0], oy+logits.shape[0])
                gap = float((full_logits[top:bottom, left:right]-
                             logits[top-oy:bottom-oy, left-ox:right-ox]).abs().max())
                max_logit_error = max(max_logit_error, gap)
                assert gap < 1e-3, 'Local/full response mismatch'
                x0, y0, x1, y1 = ds.search_crop_box
                h, w = ds.full_image_shape
                boxes.append(dict(split=split, patient=patient, view=view, x0=x0, y0=y0, x1=x1, y1=y1,
                                  width=x1-x0, height=y1-y0,
                                  area_ratio=(x1-x0)*(y1-y0)/(h*w),
                                  padded=int(x0 < 0 or y0 < 0 or x1 > w or y1 > h)))
                for index, phase in enumerate(ds.phases):
                    b = target_ds[index]
                    gt = np.asarray(labels['states'][phase]['projected_uv'][view])
                    error = abs(b['search_center_uv'].numpy()+ds.search_origin-gt).max()
                    max_label_error = max(max_label_error, float(error))
                    delta = np.asarray(labels['states'][phase]['center_lps_mm'])-plan
                    if np.max(abs(delta)) <= RADIUS_MM+1e-6:
                        if index == 0:
                            value = response_loss(logits[None, None], b['search_center_uv'][None],
                                                  b['template_center_local_uv'][None], radius=16./3.)
                            assert torch.isfinite(value)
                            loss_checked += 1
                        original_mask = read_pair(root/patient/phase, view)[1]
                        if int(original_mask.sum()) != int(b['search_mask'].sum()):
                            contour_losses.append(patient+'/'+phase+'/'+view)
                # 使用生产响应读取函数验证全图坐标元数据；临时文件不保留。
                views_checked += 1
            with tempfile.TemporaryDirectory() as temporary:
                payload, preprocessing = {}, {}
                for view in train.VIEWS:
                    ds = prediction_sets[(patient, view)]
                    b = ds[0]
                    value = model(b['template'][None], b['search'][None])[0, 0]
                    payload.update({view+'_logits':value.numpy(), view+'_scores':logits_to_scores(value).numpy(),
                                    view+'_origin_uv':ds.response_origin, view+'_stride':np.array(4),
                                    view+'_image_shape':np.array(ds.full_image_shape)})
                    preprocessing[view] = dict(template_center_uv=ds.template_center.tolist(),
                                               crop_box=list(ds.crop_box), search_crop_box=list(ds.search_crop_box))
                path = Path(temporary)/'pair.npz'
                np.savez_compressed(str(path), **payload)
                geometry = read_json(PREPARATION/'patients'/patient/'geometry.json')
                load_pair(path, dict(preprocessing=preprocessing, stride=4), geometry)
            if patient == manifest['patient_split']['train'][0]:
                draw_example(patient, supervised, root, OUTPUT)
    assert max_label_error < 1e-4 and max_map_error < 1e-8
    write_csv(OUTPUT/'roi_boxes.csv', boxes)
    write_csv(OUTPUT/'range_exceptions.csv', exceptions)
    report(boxes, exceptions, contour_losses, state_count,
           max_label_error, max_map_error, max_logit_error, loss_checked)
    print('Checked {} patients, {} states, {} fixed view ROIs; {} states outside +/-10 mm.'.format(
        len(manifest['patients']), state_count, views_checked, len(exceptions)))
    print('Results: '+str(OUTPUT/'说明.md'))


def report(boxes, exceptions, contour_losses, states, label_error, map_error, logit_error, loss_checked):
    lines = ['# outputs梳理与固定搜索ROI', '',
             '本次复用既有数据和margin22模型；没有重投影、重标注或重新训练，没有删除原文件。', '',
             '## 既有内容如何使用', '',
             '| 目录 | 用途及本次处理 |', '|---|---|',
             '| lidc_demo_geometry | 已验证的24患者几何、中心标签及患者划分；直接复用。 |',
             '| template_margin_comparison/20261015_154208 | 已有12/22/32模型及全数据误差；复用22检查响应坐标。 |',
             '| lidc_dual_training/20261013_115834_453018 | 原正式模型、val/test定位及响应分析；保留作全图基线。 |',
             '| lidc_dual_provisional | 早期暂定旧几何结果；不作为本次数据来源。 |',
             '| phase5、phase6、siamfc、siamfc_dual、siamfc_dual_center3d | 单患者早期实验；保留作历史参考。 |',
             '| state_deduplication | 数据去重记录；不重复去重。 |', '',
             '## 本次ROI规则', '',
             '- 模板沿用original投影Mask外接框+22像素，不缩放。',
             '- 用计划中心各轴±10mm盒的8角投影，得到中心可活动范围。',
             '- 向四侧增加模板参考点到边界的距离，再额外留8像素；边界对齐4像素响应网格。',
             '- 每患者每视角固定一个搜索框，所有phase共用；边界超出原图时归一化后补0。',
             '- 当前Mask和真实中心不参与ROI生成，仅用于下面的范围/轮廓检查。', '',
             '## 结果', '',
             '- {}状态、48个固定ROI；{}个ROI需要边界补0。'.format(states, sum(r['padded'] for r in boxes)),
             '- 搜索图面积占原图：平均{:.1%}，范围{:.1%}～{:.1%}；不是实测速度提升。'.format(
                 np.mean([r['area_ratio'] for r in boxes]), min(r['area_ratio'] for r in boxes), max(r['area_ratio'] for r in boxes)),
             '- 范围外{}状态（训练{}、验证{}、测试{}）；全部保留，详见range_exceptions.csv。'.format(
                 len(exceptions), *[sum(r['split']==s for r in exceptions) for s in ('train','val','test')]),
             '- 范围内状态轮廓裁切丢失：{}张视角图。'.format(len(contour_losses)),
             '- 标签换回全图最大误差{:.8f}px；几何映射最大误差{:.2e}px。'.format(label_error, map_error),
             '- 48个ROI与原全图共同网格的网络logits最大差{:.2e}；{}个代表输入损失有限。'.format(logit_error, loss_checked),
             '- 每视角密集检查1331个三维候选投影，全部位于有效响应范围内；24患者响应文件读取通过。', '',
             '±10mm是本次研究假设，不等同体表±1.5mm阈值。范围外状态不能被声称可由该搜索盒正确定位。',
             '训练入口默认search_radius_mm=10，按state_usage清单仅用范围内训练/验证状态；超范围状态另作测试。',
             '此脚本只检查ROI；训练由train.py执行，此处不产生新的训练或精度结论。', '',
             '## 只需看这些文件', '',
             '- 本说明：已有结果如何复用、裁剪规则与检查结论。',
             '- example.png：两视角的模板、原图搜索框和实际裁剪输入。',
             '- roi_boxes.csv：48个固定框，完整DRR坐标，x1/y1是切片开区间。',
             '- range_exceptions.csv：10个三维范围外状态及位移。', '']
    if contour_losses:
        lines += ['范围内轮廓裁切例外：'+', '.join(contour_losses)]
    (OUTPUT/'说明.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
