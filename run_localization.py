"""Phase 6（4–6）：成对网络响应图 -> 三维定位 -> 独立读取真值评估。

默认使用既有第 76 轮模型导出的响应图；不重新训练、不调用 TIGRE。
预测函数只接收响应图、成像几何和计划中心，不接收当前状态真值。
Python 3.7 compatible.
--multi-run 将多患者 val/test 响应图接入同一求解器；暂定几何会明确标记。
"""
import argparse
import csv
import json
from pathlib import Path

import runtime_config
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

from geometry import project, response_camera
from localize_3d import peaks, predict_pair
from data_process.prepare_center_labels import ROOT, DEFAULT_OUT, LABEL_DEFINITION, sha256
from data_loader import read_pair
from lidc_data import check_split

VIEWS = ('kv1', 'kv2')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

# 读取同一呼吸状态的两张完整响应图，检查分数、起点、步长及尺寸
def load_pair(path, config, geometry):
    """核对元数据，防止视角、坐标起点、步长或分数转换混用。"""
    maps, cameras, top1, diagnostics = [], [], [], {}
    with np.load(str(path), allow_pickle=False) as pair:
        for view, camera in zip(VIEWS, geometry['cameras']):
            scores = pair[view+'_scores'].astype(float)
            logits = pair[view+'_logits'].astype(float)
            origin = pair[view+'_origin_uv'].astype(float)
            stride = float(pair[view+'_stride'])
            prep = config['preprocessing'][view]
            expected_origin = (np.asarray(prep['template_center_uv'])-prep['crop_box'][:2]+
                               np.asarray(prep.get('search_crop_box', [0, 0])[:2]))
            if (scores.ndim != 2 or min(scores.shape) < 2 or logits.shape != scores.shape or
                    not np.isfinite(scores).all() or not np.isfinite(logits).all() or
                    not np.allclose(scores, np.clip(expit(logits), 1e-6, 1-1e-6), atol=1e-6, rtol=0) or
                    origin.shape != (2,) or not np.allclose(origin, expected_origin, atol=1e-5, rtol=0) or
                    stride != config['stride'] or
                    pair[view+'_image_shape'].tolist() != geometry['detector_image_shape']):
                raise ValueError('Response metadata mismatch: '+str(path)+'/'+view)
            maps.append(scores)
            cameras.append(response_camera(camera, origin, stride))
            row, col = np.unravel_index(np.argmax(logits), logits.shape)
            top1.append((np.array([col, row])*stride+origin).tolist())
            diagnostics[view] = dict(response_shape=list(scores.shape), origin_uv=origin.tolist(),
                                     stride=stride, top1_uv=top1[-1],
                                     peaks_uv_score=[[(u*stride+origin[0]), (v*stride+origin[1]), s]
                                                     for u, v, s in peaks(scores)])
    return maps, cameras, top1, diagnostics

# 计算计划中心 baseline、top-1 三角定位 baseline 和三维联合定位。

# 预测完成后读取真值，计算三维及各方向误差。
def evaluate(records, labels, manifest, out):
    """全部预测已写盘后才调用；真值只参与误差和图示。"""
    rows = []
    for record in records:
        gt = np.asarray(labels['states'][record['phase']]['center_lps_mm'])
        for method, result in record['methods'].items():
            row = dict(phase=record['phase'], split=record['split'], method=method, status=result['status'])
            if result['status'] == 'ok':
                pred = np.asarray(result['xyz_lps_mm'])
                error = pred-gt
                row.update(error_mm=float(np.linalg.norm(error)))
                for axis, value, delta, component, truth in zip('xyz', pred, result['delta_lps_mm'], error, gt):
                    row['gt_'+axis+'_mm'] = float(truth)
                    row['pred_'+axis+'_mm'] = float(value)
                    row['delta_'+axis+'_mm'] = float(delta)
                    row['error_'+axis+'_mm'] = float(component)
            rows.append(row)
    fields = ['phase', 'split', 'method', 'status', 'error_mm'] + [
        prefix+axis+'_mm' for prefix in ('gt_', 'pred_', 'delta_', 'error_') for axis in 'xyz']
    with (out/'errors.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(evaluation_protocol='validation_retest_not_independent',
                   evaluation_phases=manifest['split']['test'], groups={})
    for group in ('validation_retest', 'training_diagnostic'):
        metrics = {}
        for method in ('plan', 'top1', 'joint'):
            selected = [r for r in rows if r['split'] == group and r['method'] == method]
            successful = [r for r in selected if r['status'] == 'ok']
            errors = [r['error_mm'] for r in successful]
            metrics[method] = dict(total=len(selected), successful=len(successful), failed=len(selected)-len(successful),
                                   mean_error_mm=float(np.mean(errors)) if errors else None,
                                   max_error_mm=float(np.max(errors)) if errors else None,
                                   mean_absolute_axis_error_mm=[float(np.mean([abs(r['error_'+a+'_mm']) for r in successful]))
                                                                for a in 'xyz'] if errors else None)
        summary['groups'][group] = metrics
    write_json(out/'summary.json', summary)
    return summary

# 展示双视角 DRR、完整响应图和三维结果的回投影。
def plot_results(records, labels, config, geometry, out, show_baselines=True, include_plan=False):
    """蓝色简化图可额外显示计划中心；保留原版全部 baseline 的绘图方式。"""

    out.mkdir(parents=True, exist_ok=True)
    colors = dict(plan='cyan', top1='orange', joint='red')
    for record in records:
        phase = record['phase']
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        with np.load(record['response_file'], allow_pickle=False) as pair:
            for i, (view, camera) in enumerate(zip(VIEWS, geometry['cameras'])):
                image = read_pair(Path(config['root'])/phase, view, with_mask=False)[0]
                lo, hi = config['preprocessing'][view]['normalization']
                axes[i, 0].imshow(np.clip((image-lo)/(hi-lo), 0, 1), cmap='gray', vmin=0, vmax=1)
                origin, stride = pair[view+'_origin_uv'], float(pair[view+'_stride'])
                response = pair[view+'_scores']
                extent = [origin[0]-stride/2, origin[0]+stride*(response.shape[1]-.5),
                          origin[1]+stride*(response.shape[0]-.5), origin[1]-stride/2]
                im = axes[i, 1].imshow(response, cmap='magma' if show_baselines else 'Blues',
                                       extent=extent, vmin=0, vmax=1,
                                       interpolation=None if show_baselines else 'nearest')
                fig.colorbar(im, ax=axes[i, 1], label='sigmoid score')
                gt = np.asarray(labels['states'][phase]['projected_uv'][view])
                for ax in axes[i]:
                    if show_baselines:
                        ax.plot(*gt, 'g+', markersize=12, label='GT projection')
                        for method, result in record['methods'].items():
                            if result['status'] == 'ok':
                                uv = project(result['xyz_lps_mm'], camera)[0][0]
                                ax.plot(*uv, marker='x', color=colors[method], linestyle='none', label=method)
                    else:
                        if include_plan:
                            uv = project(geometry['plan_lps_mm'], camera)[0][0]
                            ax.plot(*uv, marker='o', color='darkorange', markersize=14,
                                    markerfacecolor='none', markeredgewidth=2, linestyle='none',
                                    label='Plan center projection', zorder=4)
                        ax.plot(*gt, marker='+', color='limegreen', markersize=12,
                                markeredgewidth=2, linestyle='none', label='GT projection')
                        result = record['methods']['joint']
                        if result['status'] == 'ok':
                            uv = project(result['xyz_lps_mm'], camera)[0][0]
                            ax.plot(*uv, marker='x', color='red', markersize=9,
                                    markeredgewidth=2, linestyle='none', label='3D joint projection')
                    ax.set_xlabel('u / DRR pixels')
                    ax.set_ylabel('v / DRR pixels')
                if show_baselines:
                    axes[i, 1].plot(*record['views'][view]['top1_uv'], 'o', color='orange',
                                    fillstyle='none', label='raw top-1')
                axes[i, 0].legend(fontsize=8)
                axes[i, 1].legend(fontsize=8)
                axes[i, 0].set_title(view+(' DRR: projected 3D results' if show_baselines
                                          else ' DRR: plan, GT and 3D joint' if include_plan
                                          else ' DRR: GT and 3D joint localization'))
                axes[i, 1].set_title(view+' full network response')
        title = phase+' / '+record['split']
        if geometry.get('patient'):
            title = geometry['patient']+' / '+title
        if geometry.get('geometry_status') == 'provisional_old_projector_convention':
            title += '\nPROVISIONAL geometry: workflow check, not validated accuracy'
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(str(out/(phase+'.png')), dpi=140)
        plt.close(fig)


def write_report(records, labels, summary, out):
    """把机器可读结果整理成可直接阅读的工作报告。"""
    lines = ['# Phase 6（4–6）结果', '',
             '网络完整响应图已接入 Phase 5 三维候选联合评分；未重新训练网络。',
             '坐标为患者 LPS 毫米：+X 向左，+Y 向后，+Z 向头侧。Δ = 预测中心 − 计划中心；误差 = 预测中心 − 当前真实中心。', '',
             '## 验证集复测', '',
             'phase_0002、phase_0007 同时用于模型选择和测试，不能作为独立泛化结果。', '',
             '| 方法 | 平均三维误差 / mm | 最大误差 / mm | 成功 / 总数 |',
             '|---|---:|---:|---:|']
    names = dict(plan='固定计划中心', top1='两个最高峰三角定位', joint='完整响应图三维联合评分')
    for method, metric in summary['groups']['validation_retest'].items():
        mean = '{:.3f}'.format(metric['mean_error_mm']) if metric['successful'] else 'failed'
        maximum = '{:.3f}'.format(metric['max_error_mm']) if metric['successful'] else 'failed'
        lines.append('| {} | {} | {} | {} / {} |'.format(names[method],
                     mean, maximum, metric['successful'], metric['total']))
    lines += ['', '## 逐状态结果', '',
              '| 状态 | 分组 | 计划误差 / mm | top-1误差 / mm | 联合误差 / mm | 联合 Δx, Δy, Δz / mm |',
              '|---|---|---:|---:|---:|---|']
    for record in records:
        gt = np.asarray(labels['states'][record['phase']]['center_lps_mm'])
        errors = ['{:.3f}'.format(np.linalg.norm(np.asarray(record['methods'][m]['xyz_lps_mm'])-gt))
                  if record['methods'][m]['status'] == 'ok' else 'failed' for m in ('plan', 'top1', 'joint')]
        joint = record['methods']['joint']
        delta = ', '.join('{:.3f}'.format(x) for x in joint['delta_lps_mm']) if joint['status'] == 'ok' else 'failed'
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(record['phase'], record['split'], *errors, delta))
    lines += ['', '## 如何阅读', '',
              '- predictions.json：无当前真值参与的三维预测；包括三维中心、位移、支持分数、峰间差、边界标记，以及 top-1 射线间距。',
              '- errors.csv：读取当前真值之后计算的三维总误差与三个方向误差。training_diagnostic 仅为训练状态排查结果。',
              '- summary.json：训练状态和验证集复测分别统计；失败数单独报告。',
              '- phase_xxxx.png：保留原版全部对照标记，magma 响应配色；绿 + 真值、红 x 联合定位、橙 x top-1 三角定位、青 x 计划中心、橙色空心圆原始二维最高峰。',
              '- joint_only/phase_xxxx.png：蓝色简化图，响应值越高颜色越深；仅显示绿 +（真值）和红 x（三维联合定位中心投影）。', '',
              '默认搜索范围为计划中心附近 ±18 mm，粗步长 2 mm、细步长 0.25 mm，沿用 Phase 5 软先验权重 0.05、尺度 20 mm。',
              '三维细步长只是搜索精细程度，不代表定位精度达到 0.25 mm。网络响应图步长为 4 个 DRR 像素，以双线性插值取分数。',
              'view_support 按每张响应图最大值归一化；raw_view_scores 是原始 sigmoid 分数。高分、边界标记或峰间差均不是经过校准的置信概率。',
              '应同时检查三维误差、响应图是否宽平、多峰是否接近、射线间距和搜索边界。',
              '本次只有两个复测状态；没有新增三维损失，没有扩展多患者或真实 kV。', '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run_multi(args):
    """多患者响应图接入已有求解器；预测和真值评估分两个阶段。"""
    config = read_json(args.multi_run/'config.json')
    if config.get('schema_version') != 4:
        raise ValueError('Use a multi-patient checkpoint output directory')
    split = config['patient_split']
    check_split(split,config['geometry_sha256'])
    if not np.isfinite(args.radius_mm) or not 0 < args.radius_mm <= 18:
        raise ValueError('Search radius must be within (0,18] mm')
    if config.get('search_radius_mm') is not None and args.radius_mm != config['search_radius_mm']:
        raise ValueError('3D search radius must match the local response ROI configuration')
    out = args.multi_run/args.split/'localization'
    out.mkdir(parents=True, exist_ok=True)
    records, patient_info = [], {}
    preparation = Path(config['preparation'])
    prepared = read_json(preparation/'dataset.json')
    if config['geometry_status'] != 'demo_verified_against_saved_data' or prepared['status'] != 'prepared_verified':
        raise ValueError('Multi-patient localization requires verified Demo geometry')
    for patient in split[args.split]:
        response_dir = args.multi_run/args.split/'responses'/patient
        manifest = read_json(response_dir/'manifest.json')
        geo_path = preparation/'patients'/patient/'geometry.json'
        geometry = read_json(geo_path)
        if (sha256(geo_path) != config['geometry_sha256'][patient] or
                manifest['geometry_sha256'] != config['geometry_sha256'][patient] or
                manifest['patient'] != patient or manifest['split'] != args.split or
                manifest['geometry_status'] != config['geometry_status'] or
                not geometry['geometry_verified'] or geometry['geometry_status'] != config['geometry_status'] or
                manifest['checkpoint_sha256'] != sha256(args.multi_run/'best.pt') or
                manifest['config_sha256'] != sha256(args.multi_run/'config.json') or
                manifest['label_definition'] != LABEL_DEFINITION or
                manifest['views'] != list(VIEWS) or manifest['stride'] != config['stride'] or
                manifest['phases'] != prepared['patients'][patient]['phases'] or
                not np.allclose(manifest['cameras'], geometry['cameras'], atol=1e-10, rtol=0) or
                not np.allclose(manifest['plan_lps_mm'], geometry['plan_lps_mm'], atol=1e-5, rtol=0)):
            raise ValueError('Multi-patient response provenance mismatch: '+patient)
        direction = np.asarray(geometry['voxel_to_lps'])[:3,:3]
        direction /= np.linalg.norm(direction,axis=0)
        corners = np.array(np.meshgrid(*[[-args.radius_mm,args.radius_mm]]*3)).reshape(3,-1).T
        if np.max(abs(np.linalg.solve(direction,corners.T))) > 18.+1e-8:
            raise ValueError('LPS search box exceeds calibrated CT-axis region')
        patient_config = dict(root=str(Path(config['root'])/patient),
                              preprocessing=manifest['preprocessing'], stride=config['stride'])
        patient_info[patient] = (geometry, patient_config)
        plan = np.asarray(geometry['plan_lps_mm'])
        for phase in manifest['phases']:
            path = response_dir/(phase+'.npz')
            maps, cameras, top1, diagnostics = load_pair(path, patient_config, geometry)
            methods = predict_pair(maps,cameras,geometry['cameras'],plan,top1,args.radius_mm)
            records.append(dict(patient=patient,phase=phase,split=args.split,
                                response_file=str(path.resolve()),response_sha256=sha256(path),
                                views=diagnostics,methods=methods))
    write_json(out/'predictions.json', dict(geometry_status=config['geometry_status'],records=records,
               checkpoint_sha256=manifest['checkpoint_sha256'],checkpoint_epoch=manifest['checkpoint_epoch'],
               search=dict(radius_mm=args.radius_mm,coarse_mm=2.,fine_mm=.25,prior_weight=.05,prior_sigma_mm=20.),
               note='Prediction uses plan and response maps only. Scores are not calibrated probabilities.'))
    if args.no_evaluation:
        return
    rows = []
    for patient, (geometry, patient_config) in patient_info.items():
        labels = read_json(preparation/'patients'/patient/'projected_centers.json')
        if labels['geometry_sha256'] != config['geometry_sha256'][patient]:
            raise ValueError('Evaluation geometry/labels mismatch: '+patient)
        selected = [r for r in records if r['patient'] == patient]
        for record in selected:
            truth = np.asarray(labels['states'][record['phase']]['center_lps_mm'])
            for method, result in record['methods'].items():
                row = dict(patient=patient,phase=record['phase'],method=method,status=result['status'])
                if result['status'] == 'ok':
                    pred = np.asarray(result['xyz_lps_mm'])
                    error = pred-truth
                    row['error_mm'] = float(np.linalg.norm(error))
                    for axis, p, t, delta, e in zip('xyz',pred,truth,result['delta_lps_mm'],error):
                        row.update({'pred_'+axis+'_mm':p,'gt_'+axis+'_mm':t,
                                    'delta_'+axis+'_mm':delta,'error_'+axis+'_mm':e})
                rows.append(row)
        plot_results(selected,labels,patient_config,geometry,out/'figures'/patient,show_baselines=False)
        plot_results(selected,labels,patient_config,geometry,out/'figures_with_plan'/patient,
                     show_baselines=False,include_plan=True)
    fields = ['patient','phase','method','status','error_mm']+[
        prefix+axis+'_mm' for prefix in ('pred_','gt_','delta_','error_') for axis in 'xyz']
    with (out/'errors.csv').open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metrics = {}
    for method in ('plan','top1','joint'):
        selected = [r for r in rows if r['method'] == method]
        errors = [r['error_mm'] for r in selected if r['status'] == 'ok']
        per_patient, per_patient_max, per_patient_failed = {}, {}, {}
        for patient in patient_info:
            values = [r['error_mm'] for r in selected if r['patient']==patient and r['status']=='ok']
            per_patient_failed[patient] = sum(r['status'] != 'ok' for r in selected if r['patient']==patient)
            if values:
                per_patient[patient] = float(np.mean(values))
                per_patient_max[patient] = float(np.max(values))
        metrics[method] = dict(total=len(selected),successful=len(errors),failed=len(selected)-len(errors),
            mean_error_mm=float(np.mean(errors)) if errors else None,
            median_error_mm=float(np.median(errors)) if errors else None,
            p95_error_mm=float(np.percentile(errors,95)) if errors else None,
            max_error_mm=float(np.max(errors)) if errors else None,
            per_patient_mean_mm=per_patient,per_patient_max_mm=per_patient_max,
            per_patient_failed=per_patient_failed)
    summary = dict(split=args.split,patients=split[args.split],geometry_status=config['geometry_status'],
                   checkpoint_epoch=manifest['checkpoint_epoch'],methods=metrics,
                   note=('Validation used for checkpoint selection.' if args.split == 'val' else
                         'Independent test patients; fixed checkpoint and search settings. Simulated DRR results only.'))
    write_json(out/'summary.json',summary)
    write_patient_summary(summary,out)
    write_result_overview(args.multi_run)
    print('{}: mean 3D error (mm): plan={}, top1={}, joint={}'.format(args.split,
          *['{:.3f}'.format(metrics[m]['mean_error_mm']) if metrics[m]['successful'] else 'failed'
            for m in ('plan','top1','joint')]),flush=True)
    print('Results: {}'.format(args.multi_run/'evaluation_report.md'),flush=True)


def write_patient_summary(summary, out):
    """每患者一行，保留三种方法的平均、最大误差与失败数。"""
    names = ('plan','top1','joint')
    fields = ['patient']+[m+s for m in names for s in ('_mean_mm','_max_mm','_failed')]
    patient_rows = []
    for patient in summary['patients']:
        row = dict(patient=patient)
        for method, metric in summary['methods'].items():
            row[method+'_mean_mm'] = metric['per_patient_mean_mm'].get(patient)
            row[method+'_max_mm'] = metric['per_patient_max_mm'].get(patient)
            row[method+'_failed'] = metric['per_patient_failed'][patient]
        patient_rows.append(row)
    with (out/'patient_summary.csv').open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(patient_rows)


def write_result_overview(run):
    """只生成一份结果总览，验证与独立测试分开列出。"""
    lines = ['# 三维定位结果总览','',
             '单位：mm。优先看独立测试；验证集参与过模型选择。','',
             '| 集合 | 状态数 | 计划平均 | top-1平均 | 联合平均 | 联合最大 | 联合失败数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    summaries = []
    def number(value):
        return '失败' if value is None else '{:.3f}'.format(value)
    for group, name in [('val','验证'),('test','独立测试')]:
        path = run/group/'localization'/'summary.json'
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append((group,name,summary))
        metrics = summary['methods']
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(name,metrics['joint']['total'],
                     *[number(metrics[m]['mean_error_mm']) for m in ('plan','top1','joint')],
                     number(metrics['joint']['max_error_mm']),metrics['joint']['failed']))
    lines += ['', '| 集合 | 患者 | 计划平均 | top-1平均 | 联合平均 |',
              '|---|---|---:|---:|---:|']
    for group, name, summary in summaries:
        for patient in summary['patients']:
            lines.append('| {} | {} | {} | {} | {} |'.format(name,patient,
                         *[number(summary['methods'][m]['per_patient_mean_mm'].get(patient))
                           for m in ('plan','top1','joint')]))
    lines += ['', '联合定位并非始终更准，需比较每位患者的三种误差。结果仅适用于当前模拟 DRR 数据。','',
              '只需关注以下内容（test 为独立测试，val 为验证）：','',
              '- localization/patient_summary.csv：每位患者的三种方法比较。',
              '- localization/errors.csv：逐状态误差、预测中心及各方向偏差。',
              '- localization/figures：绿色 + 真值，红色 x 联合定位；响应越高蓝色越深。','',
              '- localization/figures_with_plan：另加橙色空心圆（计划中心），三者均为三维中心的二维投影。','',
              'responses、predictions.json、summary.json 供程序使用，不必逐个查看。','']
    (run/'evaluation_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', type=Path, default=DEFAULT_OUT/'responses')
    parser.add_argument('--geometry', type=Path, default=ROOT/'outputs'/'phase5'/'geometry.json')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs'/'phase6')
    parser.add_argument('--radius-mm', type=float, default=None)
    parser.add_argument('--no-evaluation', action='store_true', help='只做预测，不读取当前状态真值或画评估图')
    parser.add_argument('--multi-run',type=Path,default=None,help='多患者 train.py 的输出目录')
    parser.add_argument('--split',choices=['val','test'],default='test')
    args = parser.parse_args()
    if args.radius_mm is None:
        args.radius_mm = (read_json(args.multi_run/'config.json').get('search_radius_mm') or 18.
                          if args.multi_run is not None else 18.)
    if args.multi_run is not None:
        run_multi(args)
        return
    manifest = read_json(args.responses/'manifest.json')
    config = read_json(args.responses.parent/'config.json')
    geometry = read_json(args.geometry)
    geometry_hash = sha256(args.geometry)
    if (manifest.get('artifact') != 'paired_full_response_maps' or manifest['views'] != list(VIEWS) or
            manifest['label_definition'] != LABEL_DEFINITION or config.get('schema_version') != 3 or
            config['split'] != manifest['split'] or config['stride'] != 4 or
            manifest['score_transform'] != config['score_transform'] or
            geometry.get('geometry_verified') is not True or geometry.get('mode') != 'original' or
            manifest['geometry_sha256'] != geometry_hash or
            config['center_labels']['geometry_sha256'] != geometry_hash or
            not np.allclose(config['center_labels']['cameras'], geometry['cameras'], atol=1e-10, rtol=0) or
            not np.allclose(config['center_labels']['states']['original']['center_lps_mm'],
                            geometry['plan_lps_mm'], atol=1e-5, rtol=0) or
            not np.allclose(manifest['cameras'], geometry['cameras'], atol=1e-10, rtol=0)):
        raise ValueError('Response/config/geometry provenance mismatch')
    phases = sorted(set(manifest['split']['train']+manifest['split']['test']))
    if (phases != manifest['phases'] or set(manifest['split']['train']) & set(manifest['split']['test']) or
            set(manifest['split']['val']) != set(manifest['split']['test'])):
        raise ValueError('Invalid phase pairing or split')
    # 只读取计划中心；当前状态中心不传入预测函数。
    plan = np.asarray(geometry['plan_lps_mm'], float)
    direction = np.asarray(geometry['voxel_to_lps'])[:3, :3]
    direction = direction/np.linalg.norm(direction, axis=0)
    corners = np.array(np.meshgrid(*[[-args.radius_mm, args.radius_mm]]*3)).reshape(3, -1).T
    if (not np.isfinite(args.radius_mm) or args.radius_mm <= 0 or
            np.max(abs(np.linalg.solve(direction, corners.T))) > 18.+1e-8):
        raise ValueError('Search box exceeds the locally verified +/-18 mm CT-axis region')
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for phase in phases:
        path = args.responses/(phase+'.npz')
        maps, cameras, top1, diagnostics = load_pair(path, config, geometry)
        methods = predict_pair(maps, cameras, geometry['cameras'], plan, top1, args.radius_mm)
        records.append(dict(phase=phase, split='validation_retest' if phase in manifest['split']['test']
                            else 'training_diagnostic', response_file=str(path.resolve()),
                            response_sha256=sha256(path), views=diagnostics, methods=methods))
    # 先写预测，后读取评估真值；两阶段互不反馈。
    write_json(args.out/'predictions.json', dict(coordinate_system=geometry['coordinate_system'],
               geometry_sha256=geometry_hash, checkpoint_epoch=manifest['checkpoint_epoch'],
               search=dict(radius_mm=args.radius_mm, coarse_mm=2., fine_mm=.25,
                           prior_weight=.05, prior_sigma_mm=20.), records=records,
               note='No current-phase targets used in prediction. Support and gap are heuristic, not probability.'))
    print('Saved {} 3D predictions: {}'.format(len(records), args.out/'predictions.json'))
    if not args.no_evaluation:
        labels = read_json(args.responses.parent/'projected_centers.json')
        if labels != config['center_labels']:
            raise ValueError('Evaluation labels differ from training snapshot')
        summary = evaluate(records, labels, manifest, args.out)
        plot_results(records, labels, config, geometry, args.out)
        plot_results(records, labels, config, geometry, args.out/'joint_only', show_baselines=False)
        write_report(records, labels, summary, args.out)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
