"""完整测试入口：双视角响应 -> 三维联合定位 -> 位移输出 -> 独立真值误差评估。"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import runtime_config
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from network import SiamFC, logits_to_scores
from data_process.prepare_center_labels import sha256
from geometry import response_camera
from manifest_data import ManifestSource, write_json
from localize_3d import predict_pair


def localization_metrics(selected, cohort, patient='ALL'):
    values = dict(cohort=cohort, patient=patient, states=len(selected)//3)
    for method in ('plan', 'top1', 'joint'):
        data = [r['error_mm'] for r in selected if r['method'] == method and r['status'] == 'ok']
        values[method+'_mean_mm'] = float(np.mean(data)) if data else None
        values[method+'_max_mm'] = max(data) if data else None
        if method != 'plan':
            values[method+'_failed'] = values['states']-len(data)
    return values


def detector_mm_row(row, spacing, roi_sizes):
    """用横纵像素间距分别换算，坐标以完整DRR的首个像素中心为原点。"""
    su, sv = spacing
    result = {key: row[key] for key in ('patient', 'phase', 'split', 'usage', 'view')}
    for prefix in ('pred', 'gt', 'error'):
        result[prefix+'_u_mm'] = row[prefix+'_u_px']*su
        result[prefix+'_v_mm'] = row[prefix+'_v_px']*sv
    result['error_mm'] = float(np.hypot(result['error_u_mm'], result['error_v_mm']))
    for key, (width, height) in roi_sizes.items():
        result[key.replace('_px', '_mm')] = '{:.3f}×{:.3f}'.format(width*su, height*sv)
    result.update(spacing_u_mm_per_px=float(su), spacing_v_mm_per_px=float(sv))
    if 'gt_in_response' in row:
        result['gt_in_response'] = row['gt_in_response']
    return result


def mean_2d_metrics(rows_px, rows_mm):
    """对同一分组内各状态的两个视角等权求平均。"""
    return dict(mean_2d_error_px=float(np.mean([r['error_px'] for r in rows_px])),
                mean_2d_error_mm=float(np.mean([r['error_mm'] for r in rows_mm])))


def read_patient_info(path):
    """仅按指定工作表C列匹配患者，原样读取J、L列；空值记为斜杠。"""
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook['01首选20例']
    rows = iter(sheet.values)
    header = next(rows)
    if [header[i] for i in (2, 9, 11)] != ['LIDC患者编号', '解剖位置', '实际HU中位数']:
        raise ValueError('患者信息表的C、J、L列表头与约定不符')
    info = {}
    for row in rows:
        if row[2] is None:
            continue
        patient = str(row[2]).strip()
        values = tuple('/' if row[i] is None or row[i] == '' else row[i] for i in (9, 11))
        if patient in info and info[patient] != values:
            raise ValueError('指定工作表中同一患者有冲突信息：'+patient)
        info[patient] = values
    workbook.close()
    return info


def write_excel(path, tables):
    """各类结果分别放入工作表；按表头和显示值计算列宽，保留原始数值精度。"""
    workbook = Workbook()
    workbook.remove(workbook.active)
    descriptions = {
        'true_delta_x_mm': '当前真实中心x减计划中心x，单位mm；同一状态在三种方法下相同。',
        'true_delta_y_mm': '当前真实中心y减计划中心y，单位mm；同一状态在三种方法下相同。',
        'true_delta_z_mm': '当前真实中心z减计划中心z，单位mm；同一状态在三种方法下相同。',
        'error_mm': '预测中心与当前真实中心的三维欧氏距离，单位mm。',
        'error_x_mm': '预测中心x减当前真实中心x，单位mm。',
        'error_y_mm': '预测中心y减当前真实中心y，单位mm。',
        'error_z_mm': '预测中心z减当前真实中心z，单位mm。',
        'error_px': '网络独立二维最高峰与真实三维中心投影的距离，单位完整DRR像素。',
        'mean_2d_error_px': '该分组全部状态、两个视角的二维网络峰值误差平均值，单位完整DRR像素。',
        'mean_2d_error_mm': '该分组各状态和视角的二维峰值误差平均值，单位探测器平面mm。',
        'displacement_group': '按真实中心到计划中心的三维直线距离分组；zero为≤1e-6 mm，0_to_5_mm为(1e-6,5] mm，5_to_10_mm为(5,10] mm；all为本范围的全部状态。',
        'template_roi_size_mm': '模板宽×高，分别乘以横纵像素间距，单位探测器平面mm。',
        'search_roi_size_mm': '搜索ROI宽×高，含边缘补零，单位探测器平面mm。',
        'template_roi_size_px': '本行视角的实际模板宽×高，单位像素。',
        'search_roi_size_px': '本行视角的实际搜索ROI宽×高，单位像素；包含需要的边缘补零。'}
    for name, rows in tables.items():
        sheet = workbook.create_sheet(name)
        fields = list(rows[0])
        sheet.append(fields)
        for row in rows:
            sheet.append([row[field] for field in fields])
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill('solid', fgColor='D9EAF7')
            description = descriptions.get(cell.value)
            if name == 'errors_2d_mm' and cell.value == 'error_mm':
                description = '网络二维最高峰与真值投影的探测器平面距离，单位mm。'
            if description:
                cell.comment = Comment(description, '字段说明')
        for index, column in enumerate(sheet.columns, 1):
            width = 0
            for cell in column:
                value = cell.value
                if isinstance(value, float):
                    cell.number_format = '0.000'
                    display = format(value, '.3f')
                elif value is None:
                    display = ''
                else:
                    display = str(value)
                width = max(width, sum(2 if ord(char) > 127 else 1 for char in display))
            sheet.column_dimensions[get_column_letter(index)].width = max(12, width + 2)
    workbook.save(str(path))


def plot_2d_results(record, datasets, maps, rows, path, stride):
    """一状态一张图：每行一个视角，显示计划模板、局部搜索图和网络响应。"""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    figure.subplots_adjust(left=.06, right=.94, bottom=.13, top=.85, wspace=.35, hspace=.50)
    figure.suptitle('{} | {} | {}'.format(record['patient'], record['phase'], record['usage']), fontsize=15)
    for index, row in enumerate(rows):
        view = row['view']
        dataset = datasets[(record['patient'], view)]
        sample = dataset[dataset.phases.index(record['phase'])]
        template = sample['template'][0].numpy()
        search = sample['search'][0].numpy()
        template_origin = sample['template_origin_uv'].numpy()
        search_origin = sample['search_origin_uv'].numpy()
        response_origin = sample['response_origin_uv'].numpy()
        reference = sample['template_center_uv'].numpy()
        prediction = np.asarray([row['pred_u_px'], row['pred_v_px']])
        truth = np.asarray([row['gt_u_px'], row['gt_v_px']])

        # extent按像素中心定义；响应图每格对应stride个完整DRR像素。
        for column, (image, origin, step) in enumerate([
                (template, template_origin, 1), (search, search_origin, 1),
                (maps[index], response_origin, stride)]):
            ax = axes[index, column]
            height, width = image.shape
            extent = (origin[0]-step/2, origin[0]+(width-.5)*step,
                      origin[1]+(height-.5)*step, origin[1]-step/2)
            shown = ax.imshow(image, cmap='Blues' if column == 2 else 'gray',
                              vmin=0, vmax=1, origin='upper', extent=extent, interpolation='nearest')
            if column == 0:
                # 轮廓与模板共用完整DRR像素中心坐标，不改变网络输入。
                mask = sample['template_mask'][0].numpy()
                ax.contour(template_origin[0]+np.arange(mask.shape[1]),
                           template_origin[1]+np.arange(mask.shape[0]), mask,
                           levels=[.5], colors=['#ff9900'], linewidths=1.2)
                ax.plot(*reference, '+', color='#9b59d0', markersize=12, markeredgewidth=2)
                ax.set_title(view+' | original template')
            else:
                ax.plot(*truth, '+', color='limegreen', markersize=13, markeredgewidth=2)
                ax.plot(*prediction, 'x', color='red', markersize=10, markeredgewidth=2)
                title = 'local search DRR' if column == 1 else 'network response'
                ax.set_title('{} | {}\n2D peak error = {:.3f} px'.format(view, title, row['error_px']))
                if column == 2:
                    figure.colorbar(shown, ax=ax, fraction=.046, pad=.04, label='Sigmoid score')
                    if not row['gt_in_response']:
                        ax.text(.02, .02, 'GT outside valid response', transform=ax.transAxes,
                                fontsize=9, color='red', bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
            # 范围外真值不应把坐标轴自动拉长，图片保持实际输入/响应范围。
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_xlabel('u (full DRR pixels)')
            ax.set_ylabel('v (full DRR pixels)')
    figure.legend(handles=[
        Line2D([], [], marker='+', color='#9b59d0', linestyle='None', markersize=11, label='Template plan reference'),
        Line2D([], [], color='#ff9900', linewidth=1.2, label='Template projected Mask contour'),
        Line2D([], [], marker='+', color='limegreen', linestyle='None', markersize=11, label='GT projection'),
        Line2D([], [], marker='x', color='red', linestyle='None', markersize=9, label='Network 2D peak')],
        loc='lower center', bbox_to_anchor=(.5, .02), ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=150)
    plt.close(figure)


@torch.no_grad()
def evaluate_manifest(checkpoint, args, align_search_grid=False):
    """新清单评估：先完成全部无当前真值预测，再加载标签计算误差。"""
    cfg = checkpoint['config']
    training_fit = args.split == 'train'
    root = args.root or (runtime_config.TRAIN_DATA_ROOT if training_fit else runtime_config.TEST_DATA_ROOT)
    print('TEST_DATA_ROOT:',root)
    preparation = args.preparation or (
        runtime_config.TRAIN_PREPARATION_ROOT if training_fit else runtime_config.TEST_PREPARATION_ROOT)
    print('TEST_PREPARATION_ROOT:',preparation)
    if root is None or preparation is None:
        raise ValueError('Set TEST_DATA_ROOT/TEST_PREPARATION_ROOT in runtime_config.py or use --root/--preparation')
    split = 'train' if training_fit else 'test'
    manifest = args.manifest or Path(preparation)/('manifest_'+split)/'dataset.json'
    source = ManifestSource(manifest, root, preparation, expected_split=split)
    if args.limit_patients:
        source = ManifestSource(manifest, root, preparation, expected_split=split,
                                patients=source.patient_ids[:args.limit_patients])
    if source.radius != cfg['search_radius_mm'] or args.radius_mm != source.radius:
        raise ValueError('Manifest/checkpoint/search radii differ')
    if not training_fit:
        subjects = {source.patient(p)['subject_id'] for p in source.patient_ids}
        overlap = subjects.intersection(cfg['training_subjects'])
        if overlap:
            raise ValueError('Independent test overlaps training subjects: '+', '.join(sorted(overlap)))
    config_path = Path(args.checkpoint).parent/'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != cfg:
        raise ValueError('Checkpoint/config mismatch')
    out = args.out or runtime_config.TEST_OUTPUT_ROOT/(
        Path(args.checkpoint).parent.name+'_'+split+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out = Path(out).resolve()
    if (out == source.data_root or source.data_root in out.parents or out == source.preparation_root or
            out == source.preparation_root/'patients' or source.preparation_root/'patients' in out.parents):
        raise ValueError('Evaluation output must not overwrite input data')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('输出目录非空，请使用新的--out，避免覆盖已有结果：'+str(out))
    out.mkdir(parents=True, exist_ok=True)
    model = SiamFC().to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    predictions = []
    search_grids = {}
    roi_sizes = {}
    detector_spacing = {}
    # 仅在需要绘图时保留CPU响应；不存搜索图，不产生磁盘响应缓存。
    response_maps = {}
    for patient in source.patient_ids:
        datasets, geometry = source.datasets(patient, with_targets=False, margin=cfg['margin'],
                                              align_search_grid=align_search_grid)
        search_grids[patient] = {}
        for view in cfg['views']:
            dataset = datasets[(patient, view)]
            height, width = dataset.template.shape
            x0, y0, x1, y1 = dataset.search_crop_box
            roi_sizes[(patient, view)] = dict(template_roi_size_px=(width, height),
                                             search_roi_size_px=(x1-x0, y1-y0))
            # 加载器已核对计划图和搜索图的物理网格一致。
            detector_spacing[(patient, view)] = dataset.grid[1][:2]
            search_grids[patient][view] = dict(
                template_crop_box=list(dataset.crop_box), search_crop_box=list(dataset.search_crop_box),
                response_origin_uv=dataset.response_origin.tolist(),
                plan_response_uv=((dataset.template_center-dataset.response_origin)/model.stride).tolist())
        if training_fit and patient in cfg['preprocessing']:
            for view in cfg['views']:
                check_preprocessing(datasets[(patient,view)], {'preprocessing':cfg['preprocessing'][patient]}, view)
        states = source.patient(patient)['states']
        for index, state in enumerate(states):
            maps, cameras, top1 = [], [], []
            for view, camera in zip(cfg['views'], geometry['cameras']):
                sample = datasets[(patient,view)][index]
                if sample['phase'] != state['state'] or 'search_center_uv' in sample:
                    raise ValueError('Paired prediction or target isolation failed')
                logits = model(sample['template'][None].to(args.device), sample['search'][None].to(args.device))
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite network response')
                origin = sample['response_origin_uv'].numpy()
                maps.append(logits_to_scores(logits)[0,0].cpu().numpy().astype(float))
                cameras.append(response_camera(camera, origin, model.stride))
                row, col = np.unravel_index(int(logits[0,0].argmax()), logits.shape[-2:])
                top1.append((np.array([col,row])*model.stride+origin).tolist())
            methods = predict_pair(maps, cameras, geometry['cameras'], geometry['plan_lps_mm'], top1, source.radius)
            predictions.append(dict(patient=patient, phase=state['state'], split=split,
                                    usage=state['usage'], top1_uv=top1, methods=methods))
            if args.save_figures:
                response_maps[(patient, state['state'])] = [m.astype(np.float32) for m in maps]
    checkpoint_hash, manifest_hash = sha256(args.checkpoint), sha256(source.path)
    if args.predict_only:
        write_json(out/'predictions.json', dict(checkpoint_sha256=checkpoint_hash,
                   manifest_sha256=manifest_hash, radius_mm=source.radius, records=predictions,
                   grid_alignment='template' if align_search_grid else 'image', search_grids=search_grids))
        print('Prediction complete (no current labels read): '+str(out/'predictions.json'), flush=True)
        return
    # 真值仅从这里开始读取；原始CT、三维Mask及当前投影Mask均不需要加载。
    labels = {p:source.read_verified(p, 'training_labels') for p in source.patient_ids}
    errors_2d = []
    errors_2d_mm = []
    figure_patient, figure_datasets = None, None
    for record in predictions:
        target = labels[record['patient']]['states'][record['phase']]['projected_uv']
        if args.save_figures and figure_patient != record['patient']:
            figure_patient = record['patient']
            figure_datasets, _ = source.datasets(figure_patient, with_targets=False, margin=cfg['margin'],
                                                 align_search_grid=align_search_grid)
        view_rows = []
        for index, view in enumerate(cfg['views']):
            prediction = np.asarray(record['top1_uv'][index])
            truth = np.asarray(target[view])
            error = prediction-truth
            row = dict(patient=record['patient'], phase=record['phase'], split=split,
                       usage=record['usage'], view=view, pred_u_px=float(prediction[0]),
                       pred_v_px=float(prediction[1]), gt_u_px=float(truth[0]), gt_v_px=float(truth[1]),
                       error_u_px=float(error[0]), error_v_px=float(error[1]), error_px=float(np.linalg.norm(error)))
            key = (record['patient'], view)
            row.update({name: '{}×{}'.format(*size) for name, size in roi_sizes[key].items()})
            if args.save_figures:
                dataset = figure_datasets[(record['patient'], view)]
                response_uv = (truth-dataset.response_origin)/model.stride
                height, width = response_maps[(record['patient'], record['phase'])][index].shape
                row['gt_in_response'] = bool(0 <= response_uv[0] <= width-1 and 0 <= response_uv[1] <= height-1)
            view_rows.append(row)
            errors_2d_mm.append(detector_mm_row(row, detector_spacing[key], roi_sizes[key]))
        errors_2d.extend(view_rows)
        if args.save_figures:
            plot_2d_results(record, figure_datasets, response_maps.pop((record['patient'], record['phase'])),
                            view_rows, out/'figures_2d'/record['patient']/(record['phase']+'.png'), model.stride)
    errors = []
    displacement = {}
    for record in predictions:
        truth = np.asarray(labels[record['patient']]['states'][record['phase']]['center_lps_mm'])
        plan = np.asarray(labels[record['patient']]['states']['original']['center_lps_mm'])
        true_delta = truth-plan
        displacement[(record['patient'], record['phase'])] = float(np.linalg.norm(true_delta))
        for method, result in record['methods'].items():
            row = dict(patient=record['patient'], phase=record['phase'],
                       usage=record['usage'], method=method, status=result['status'], error_mm=None)
            for prefix in ('pred_', 'gt_', 'true_delta_', 'error_'):
                row.update({prefix+axis+'_mm':None for axis in 'xyz'})
            # 真实中心与真实位移独立于预测是否成功，失败行也保留这些真值。
            for axis, t, delta in zip('xyz', truth, true_delta):
                row.update({'gt_'+axis+'_mm':float(t), 'true_delta_'+axis+'_mm':float(delta)})
            if result['status'] == 'ok':
                pred = np.asarray(result['xyz_lps_mm'])
                row['error_mm'] = float(np.linalg.norm(pred-truth))
                for axis, p, t in zip('xyz', pred, truth):
                    row.update({'pred_'+axis+'_mm':float(p), 'error_'+axis+'_mm':float(p-t)})
            errors.append(row)
    summary, patient_rows = [], []
    state_key = lambda row: (row['patient'], row['phase'])
    for usage in ('main', 'out_of_range_test'):
        selected = [r for r in errors if r['usage'] == usage]
        if not selected:
            continue
        cohort = split+('_main' if usage == 'main' else '_outside')
        selected_px = [r for r in errors_2d if r['usage'] == usage]
        selected_mm = [r for r in errors_2d_mm if r['usage'] == usage]
        groups = [('all', selected)]
        if usage == 'main':
            # 分组使用真实三维直线位移；超过10 mm的范围内状态计入all。
            bands = [('zero', lambda d: d <= 1e-6),
                     ('0_to_5_mm', lambda d: 1e-6 < d <= 5),
                     ('5_to_10_mm', lambda d: 5 < d <= 10)]
            groups.extend((name, [r for r in selected if inside(displacement[state_key(r)])])
                          for name, inside in bands)
        for name, group in groups:
            if not group:
                continue
            keys = {state_key(r) for r in group}
            metrics = localization_metrics(group, cohort)
            summary.append(dict(cohort=metrics.pop('cohort'), displacement_group=name, **metrics,
                **mean_2d_metrics([r for r in selected_px if state_key(r) in keys],
                                  [r for r in selected_mm if state_key(r) in keys])))
        for patient in sorted({r['patient'] for r in selected}):
            values = localization_metrics([r for r in selected if r['patient'] == patient], cohort, patient)
            location, hu = args.patient_metadata.get(patient, ('/', '/'))
            values = dict(cohort=values.pop('cohort'), patient=values.pop('patient'),
                          解剖位置=location, 实际HU中位数=hu, **values)
            values.pop('top1_failed')
            values.pop('joint_failed')
            values.update(mean_2d_metrics([r for r in selected_px if r['patient'] == patient],
                                          [r for r in selected_mm if r['patient'] == patient]))
            patient_rows.append(values)
    write_excel(out/'results.xlsx', dict(summary=summary, patient_summary=patient_rows,
                                        errors=errors, errors_2d=errors_2d, errors_2d_mm=errors_2d_mm))
    note = ('训练患者的拟合表现，不是独立测试。' if training_fit else
            '独立测试患者；已检查患者/同患者多结节不与训练对象重叠。')
    (out/'结果说明.md').write_text('\n'.join([
        '# 三维定位结果', '', note,
        '模型：{}；训练轮次：{}。'.format(Path(args.checkpoint).name, checkpoint['epoch']),
        '训练中心标签：{}。'.format(cfg.get('center_label', 'disk')),
        '搜索ROI网格：{}。'.format('对齐模板裁剪原点' if align_search_grid else '原有全图4像素网格'),
        'results.xlsx包含summary、patient_summary、errors、errors_2d和errors_2d_mm。',
        '列宽按表头和显示值自动计算；小数显示3位，底层仍保存原始精度，空白表示预测失败而非0。',
        '模型指纹：'+checkpoint_hash+'；清单指纹：'+manifest_hash+'。',
        'plan=固定计划中心，top1=两视角最高峰三角定位，joint=完整响应图三维联合定位。',
        'main与outside分开汇总；summary的all为该范围总计。main另按真实三维位移分组：',
        'zero≤1e-6 mm，0_to_5_mm为(1e-6,5] mm，5_to_10_mm为(5,10] mm。',
        '各轴±10 mm内三维直线位移大于10 mm的状态计入main/all；无样本的分组省略。',
        'errors：true_delta_x/y/z_mm=当前真实中心减计划中心；同一状态在三种方法下相同。',
        'errors：error_x/y/z_mm=预测中心减当前真实中心；error_mm为三维距离，均为mm。',
        '不再输出旧delta_x/y/z_mm预测位移列。全部预测在内存完成后才读取当前状态真值；默认不保存predictions.json及响应缓存。',
        'errors_2d：网络各视角独立最高峰与三维真值投影点的二维像素误差；不是三角定位误差。',
        'errors_2d_mm：同一批二维峰值结果，u、v各乘以图像头中的横纵间距，误差取两个分量的欧氏距离。',
        '二维mm为探测器平面单位，以完整DRR首个像素中心为原点，与患者LPS三维坐标分开。',
        'summary和patient_summary的mean_2d_error_px/mm对相应状态和两个视角等权求平均。',
        'errors_2d的template_roi_size_px、search_roi_size_px：该视角实际输入的宽×高，单位像素；不另存search_grids工作表。',
        '患者信息来源：{}，仅01首选20例工作表，C列匹配患者ID，J列解剖位置、L列实际HU中位数；缺失填/，文字及数值原样保留。'.format(args.patient_info),
        '保留errors.status；summary保留top1_failed、joint_failed，不再输出plan_failed及患者级失败汇总列。',
        ('figures_2d/患者/状态.png：上排kv1、下排kv2；左为模板、中为局部搜索DRR、右为响应图。'
         '模板橙色轮廓为计划投影Mask，紫色+为模板计划参考点，绿色+为当前真值投影，红色x为网络二维最高峰；响应越高蓝色越深。'
         if args.save_figures else '需要全部状态的二维可视化时，运行命令增加--save-figures。'),
        '失败数为求解失败次数；0不代表误差为0。', '']), encoding='utf-8')
    print('Evaluation complete: '+str(out/'结果说明.md'), flush=True)
    return dict(errors_2d=errors_2d, errors=errors)


def compare_sampling_grids(results, out, split):
    """在相同患者/状态上汇总二维和三维误差；不根据结果删样本。"""
    baseline, aligned = results['image_grid'], results['template_grid']
    key = lambda row: (row['patient'], row['phase'], row['view'])
    if {key(row) for row in baseline['errors_2d']} != {key(row) for row in aligned['errors_2d']}:
        raise ValueError('两组患者、状态或视角不同，不能进行配对对照')
    # plan的三维误差就是实际位移大小；只用于评估分组，不参与ROI或预测。
    motion = {(r['patient'], r['phase']):r['error_mm']
              for r in baseline['errors'] if r['method'] == 'plan'}
    main = {(r['patient'], r['phase']) for r in baseline['errors_2d'] if r['usage'] == 'main'}
    outside = {(r['patient'], r['phase']) for r in baseline['errors_2d'] if r['usage'] == 'out_of_range_test'}
    groups = [('main_all', main),
              ('main_zero', {k for k in main if motion[k] <= 1e-6}),
              ('main_small_0_to_3mm', {k for k in main if 1e-6 < motion[k] < 3.}),
              ('main_moving_ge3mm', {k for k in main if motion[k] >= 3.}),
              ('outside', outside)]
    rows = []
    for cohort, selected in groups:
        if not selected:
            continue
        for mode, result in results.items():
            errors_2d = [r for r in result['errors_2d'] if (r['patient'], r['phase']) in selected]
            errors_3d = [r for r in result['errors'] if (r['patient'], r['phase']) in selected]
            metrics = localization_metrics(errors_3d, split+'_'+cohort)
            row = dict(cohort=metrics['cohort'], grid=mode, states=len(selected))
            for view in ('kv1', 'kv2'):
                values = [r['error_px'] for r in errors_2d if r['view'] == view]
                row.update({view+'_mean_px':float(np.mean(values)),
                            view+'_p95_px':float(np.percentile(values, 95)),
                            view+'_max_px':float(np.max(values))})
            for method in ('plan', 'top1', 'joint'):
                for metric in ('mean_mm', 'max_mm', 'failed'):
                    if method == 'plan' and metric == 'failed':
                        continue
                    row[method+'_'+metric] = metrics[method+'_'+metric]
            rows.append(row)
    write_excel(out/'comparison.xlsx', dict(comparison=rows))
    (out/'对照说明.md').write_text('\n'.join([
        '# 采样网格对齐对照', '',
        '两组使用同一模型、患者、状态和±10mm范围；不重新训练，不插值响应图。',
        'image_grid：原有搜索框起点对齐全图4像素网格。',
        'template_grid：搜索框起点与模板裁剪原点模4相同，每轴最多平移2像素。',
        '两组模板、归一化及ROI宽高保持不变；ROI原有8像素余量仍至少剩6像素。',
        '物理覆盖范围有轻微平移，仍覆盖规定候选位置及模板支持；不是让两个不同格点集合具有完全相同的采样坐标。',
        '', '先看comparison.xlsx：',
        '- main_all：±10mm范围内全部状态，是主要对照结果。',
        '- main_zero：真实三维中心位移≤1e-6mm，检查零位移时是否减少漂移。',
        '- main_small_0_to_3mm：真实位移大于1e-6且小于3mm。',
        '- main_moving_ge3mm：真实位移至少3mm，检查是否牺牲有位移状态。',
        '- outside：范围外状态，单独查看，不混入主要结果。',
        'kv1/kv2_mean_px、p95_px、max_px越低越好；plan/top1/joint指标仍为三维mm，失败数需单独检查。',
        '不只看零位移组，也要看main_all、两视角P95和最终joint三维误差，不能预先认定对齐后更好。',
        '', 'image_grid/和template_grid/各自保留常规测试结果；加--save-figures后各自输出全部二维图片。',
        '各组results.xlsx的errors_2d工作表记录每个视角的模板和搜索ROI尺寸，不另存search_grids。',
        '两组都先完成全部预测，再读取当前状态真值；真值不参与ROI选择和预测。',
        '已有数据、标签、投影矩阵和模型文件均不改写。', '']), encoding='utf-8')
    print('Comparison complete: '+str(out/'comparison.xlsx'), flush=True)


def check_preprocessing(ds, cfg, view):
    prep=cfg['preprocessing'][view]
    if (not np.allclose([ds.low,ds.high],prep['normalization']) or
        list(ds.crop_box) != prep['crop_box'] or
        not np.allclose(ds.template_center,prep['template_center_uv'])):
        raise ValueError('Template preprocessing differs from checkpoint: '+view)
    if 'search_crop_box' in prep and list(ds.search_crop_box) != prep['search_crop_box']:
        raise ValueError('Search ROI differs from checkpoint: '+view)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=runtime_config.TEST_CHECKPOINT)
    parser.add_argument('--root', type=Path, default=None)
    parser.add_argument('--preparation', type=Path, default=None)
    parser.add_argument('--manifest', type=Path, default=None)
    parser.add_argument('--patient-info', type=Path,
                        default=runtime_config.PROJECT_ROOT/'LIDC_10mm-30mm.xlsx',
                        help='患者信息参考表；仅读取01首选20例的C、J、L列')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--limit-patients', type=int, default=None, help='小规模试运行')
    parser.add_argument('--predict-only', action='store_true', help='输出三维坐标及相对计划中心位移；不读取当前真值')
    parser.add_argument('--save-figures', action='store_true', help='保存清单内全部状态的双视角模板、搜索DRR、响应图和二维误差')
    parser.add_argument('--compare-grids', action='store_true', help='不重新训练，比较原有全图网格与模板对齐网格')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--split', choices=['train', 'test'], default='test', help='默认独立测试；train仅用于拟合评估')
    parser.add_argument('--radius-mm', type=float, default=None, help='默认沿用模型和清单中的搜索范围')
    args = parser.parse_args()
    if args.predict_only and args.save_figures:
        parser.error('--save-figures需要真值点，不能与不读取真值的--predict-only同时使用')
    if args.compare_grids and (args.predict_only or args.split != 'test'):
        parser.error('--compare-grids用于测试集配对误差对照，不与--predict-only或--split train同时使用')
    if args.checkpoint is None:
        parser.error('Set TEST_CHECKPOINT in runtime_config.py or pass --checkpoint /path/to/last.pt')
    if args.limit_patients is not None and args.limit_patients < 1:
        parser.error('limit-patients must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this Python environment')
    # 普通评估在网络执行前检查参考表；仅预测模式不需要患者信息。
    args.patient_metadata = {} if args.predict_only else read_patient_info(args.patient_info)
    torch.set_num_threads(4)
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu')
    cfg = checkpoint['config']
    if cfg.get('schema_version') != 5:
        parser.error('This entry uses manifest/schema_version=5 checkpoints; old models: python -m legacy.test')
    args.radius_mm = cfg['search_radius_mm'] if args.radius_mm is None else args.radius_mm
    if args.radius_mm != cfg['search_radius_mm']:
        parser.error('Search radius must match the checkpoint and manifest')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.compare_grids:
        out = args.out or runtime_config.TEST_OUTPUT_ROOT/(
            Path(args.checkpoint).parent.name+'_grid_compare_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out = Path(out).resolve()
        # 显式指定目录时也不覆盖前次实验。
        if out.exists():
            parser.error('对照输出目录已存在，请指定一个新的--out')
        results = {}
        for mode, aligned in (('image_grid', False), ('template_grid', True)):
            run_args = argparse.Namespace(**vars(args))
            run_args.out = out/mode
            print('Grid comparison: '+mode, flush=True)
            results[mode] = evaluate_manifest(checkpoint, run_args, align_search_grid=aligned)
        compare_sampling_grids(results, out, args.split)
    else:
        # 新模型沿用训练时记录的对齐网格；历史模型未记录时保持原全图网格。
        evaluate_manifest(checkpoint, args, align_search_grid=cfg.get('align_search_grid', False))


if __name__ == '__main__':
    main()
