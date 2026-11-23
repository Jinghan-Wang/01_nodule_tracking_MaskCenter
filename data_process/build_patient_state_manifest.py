"""建立可迁移的患者/状态清单；只复用已有矩阵和标签，不划分数据、不修改源文件。

PyCharm直接运行。仅需Python标准库，不依赖PyTorch、SimpleITK、Demo或其他项目脚本。
默认输出：PREPARATION_ROOT/manifest_train。上传服务器后，只需提供新的数据根目录
和预处理结果根目录；清单中的文件路径均为相对路径，使用正斜杠。
"""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


# 整个目录已由用户指定为训练集；不会抽取验证集或测试集。
DATA_ROOT = Path(r"D:/AI_code/6nodule_tracking/nodule_tracking/datasets/data")
PREPARATION_ROOT = Path(r'outputs/geometry_target_center')
DATASET_SPLIT = 'test'              # 另一批独立测试数据可另运行本脚本，改为test。
RADIUS_MM = 10.
TOLERANCE_MM = 1e-6
VIEWS = {'kv1': '0000_angle_0.0', 'kv2': '0001_angle_150.0'}
LABEL_DEFINITION = 'projected_3d_mask_centroid_lps'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(value, length):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError('Invalid coordinate length')
    numbers = [float(x) for x in value]
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError('Non-finite coordinate')
    return numbers


def relative_file(path, root):
    """根目录只用于本机检查，不写进正式清单。"""
    relative = path.resolve().relative_to(root).as_posix()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('Missing or zero-byte file: '+relative)
    return relative


def state_files(folder, source, data_root):
    base = folder/source
    inputs, projected_masks = {}, {}
    for view, suffix in VIEWS.items():
        inputs[view] = relative_file(base/'CTDRR_ROI_NULL'/('CTDRR_'+suffix+'.nii.gz'), data_root)
        projected_masks[view] = relative_file(base/'MASKDRR_ROI_NULL'/('MASKDRR_'+suffix+'.nii.gz'), data_root)
    # 训练图像与监督/评估数据分开列出，避免将当前状态Mask当作预测输入。
    annotations = dict(ct=relative_file(base/'CT.nii.gz', data_root),
                       mask_3d=relative_file(base/'mask.nii.gz', data_root),
                       projected_masks=projected_masks)
    return inputs, annotations


def check_projection(center, projected, cameras, shape):
    height, width = shape
    for view, camera in zip(VIEWS, cameras):
        uv = vector(projected[view], 2)
        homogeneous = [sum(a*b for a, b in zip(row, center+[1.])) for row in camera]
        if not all(math.isfinite(x) for x in homogeneous) or homogeneous[2] <= 1e-9:
            raise ValueError('Label behind camera: '+view)
        expected = [homogeneous[i]/homogeneous[2] for i in (0, 1)]
        if (not (0 <= uv[0] < width and 0 <= uv[1] < height) or
                max(abs(a-b) for a, b in zip(uv, expected)) > 1e-6):
            raise ValueError('Projected label inconsistent with matrix: '+view)


def prepare_patient(folder, preparation, data_root, split):
    """整位患者检查通过才入清单；计划模板不作为训练phase。"""
    destination = preparation/'patients'/folder.name
    paths = {key: destination/name for key, name in (
        ('geometry', 'geometry.json'), ('training_labels', 'projected_centers.json'),
        ('plan_labels', 'original_labels.json'))}
    geometry, labels, original = [read_json(paths[key]) for key in ('geometry', 'training_labels', 'plan_labels')]
    hashes = {key: sha256(path) for key, path in paths.items()}
    if any(item.get('patient') != folder.name for item in (geometry, labels, original)):
        raise ValueError('Patient identity differs between folder, matrix and labels')
    if geometry.get('geometry_verified') is not True or geometry.get('geometry_status') != 'demo_verified_against_saved_data':
        raise ValueError('Geometry has not passed actual Demo verification')
    for item in (labels, original):
        if (item.get('label_definition') != LABEL_DEFINITION or
                item.get('geometry_status') != geometry['geometry_status'] or
                item.get('geometry_sha256') != hashes['geometry'] or
                not item.get('input_digest') or item['input_digest'] != geometry.get('input_digest') or
                item.get('cameras') != geometry.get('cameras') or
                item.get('detector_image_shape') != geometry.get('detector_image_shape')):
            raise ValueError('Label definition or matrix provenance differs')
    if original.get('states') != {'original': labels['states']['original']}:
        raise ValueError('Plan-only labels differ from complete labels')
    cameras = [[vector(row, 4) for row in camera] for camera in geometry['cameras']]
    if len(cameras) != 2 or any(len(camera) != 3 for camera in cameras):
        raise ValueError('Expected two 3x4 camera matrices')
    shape = vector(geometry['detector_image_shape'], 2)
    if any(x <= 0 or not x.is_integer() for x in shape):
        raise ValueError('Invalid detector dimensions')
    plan_folder = labels.get('template_folder')
    if plan_folder not in ('Origin', 'original') or geometry.get('template_folder') != plan_folder:
        raise ValueError('Plan folder differs from verified geometry')
    plan = vector(labels['states']['original']['center_lps_mm'], 3)
    if max(abs(a-b) for a, b in zip(plan, vector(geometry['plan_lps_mm'], 3))) > 1e-5:
        raise ValueError('Plan label differs from calibration centre')
    phases = sorted(p.name for p in folder.glob('phase_*') if p.is_dir())
    if not phases or set(labels['states']) != {'original'} | set(phases):
        raise ValueError('Phase directories and label states differ')
    records = []
    for state in ['original']+phases:
        target = labels['states'][state]
        source = plan_folder if state == 'original' else state
        if target.get('source_folder') != source:
            raise ValueError('Label source folder differs: '+state)
        if not re.fullmatch(r'[0-9a-f]{64}', target.get('mask_sha256', '')):
            raise ValueError('Mask fingerprint absent: '+state)
        center = vector(target['center_lps_mm'], 3)
        check_projection(center, target['projected_uv'], cameras, shape)
        delta = [a-b for a, b in zip(center, plan)]
        inside = max(map(abs, delta)) <= RADIUS_MM+TOLERANCE_MM
        usage = 'template_only' if state == 'original' else ('main' if inside else 'out_of_range_test')
        inputs, annotations = state_files(folder, source, data_root)
        annotations.update(label_state=state, mask_sha256=target['mask_sha256'])
        records.append(dict(state=state, source_folder=source, split=split, usage=usage,
                            within_10mm_box=inside, displacement_lps_mm=delta,
                            use_for_training=(split == 'train' and usage == 'main'),
                            inputs=inputs, annotations=annotations))
    base = re.fullmatch(r'(LIDC-IDRI-\d{4})(?:-\d+)?', folder.name)
    patient = dict(patient_id=folder.name, subject_id=base.group(1) if base else folder.name,
                   split=split, **{key: relative_file(path, preparation) for key, path in paths.items()})
    patient.update(source_json_sha256=hashes, image_shape_hw=shape,
                   template=records[0], states=records[1:])
    return patient


def build_manifest(data_root, preparation, split, patient_ids=None):
    """只读取已有成果；缺少完整预处理成果的患者列为未准备完成，不混入有效清单。"""
    folders = sorted(p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if patient_ids is not None:
        requested = set(patient_ids)
        if len(requested) != len(patient_ids) or not requested.issubset({p.name for p in folders}):
            raise ValueError('Selected patient IDs are missing or duplicated')
        folders = [p for p in folders if p.name in requested]
    patients, patient_rows, state_rows = {}, [], []
    for folder in folders:
        row = dict(patient=folder.name, split=split, status='', phase_count=0,
                   main_phases=0, out_of_range_phases=0, reason='')
        destination = preparation/'patients'/folder.name
        missing = [name for name in ('geometry.json', 'projected_centers.json', 'original_labels.json')
                   if not (destination/name).is_file()]
        if missing:
            row.update(status='not_prepared', reason='Missing prepared output: '+', '.join(missing))
        else:
            try:
                patient = prepare_patient(folder, preparation, data_root, split)
                patients[folder.name] = patient
                row.update(status='ready', phase_count=len(patient['states']),
                           main_phases=sum(r['usage'] == 'main' for r in patient['states']),
                           out_of_range_phases=sum(r['usage'] == 'out_of_range_test' for r in patient['states']))
                for record in [patient['template']]+patient['states']:
                    delta = record['displacement_lps_mm']
                    state_rows.append(dict(patient=folder.name, state=record['state'],
                        source_folder=record['source_folder'], split=split, usage=record['usage'],
                        within_10mm_box=record['within_10mm_box'], use_for_training=record['use_for_training'],
                        dx_mm=delta[0], dy_mm=delta[1], dz_mm=delta[2], max_abs_axis_mm=max(map(abs, delta)),
                        kv1_drr=record['inputs']['kv1'], kv2_drr=record['inputs']['kv2']))
            except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
                row.update(status='invalid', reason=str(error))
        patient_rows.append(row)
    if not patients:
        raise ValueError('No fully prepared valid patients; check data/preparation roots')
    summary = dict(scanned_patients=len(folders), ready_patients=len(patients),
                   not_prepared_patients=sum(r['status'] == 'not_prepared' for r in patient_rows),
                   invalid_patients=sum(r['status'] == 'invalid' for r in patient_rows),
                   templates=len(patients), phases=sum(len(p['states']) for p in patients.values()),
                   main_phases=sum(r['usage'] == 'main' for r in state_rows),
                   out_of_range_phases=sum(r['usage'] == 'out_of_range_test' for r in state_rows))
    dataset = dict(schema_version=1, split=split, label_definition=LABEL_DEFINITION,
                   required_runtime_roots=['data_root', 'preparation_root'],
                   path_convention='POSIX relative paths; inputs/annotations relative to data_root; geometry/training_labels/plan_labels relative to preparation_root',
                   coordinate_system='patient LPS mm; full DRR zero-based u-right/v-down pixels',
                   radius_mm=RADIUS_MM, tolerance_mm=TOLERANCE_MM,
                   range_rule='max(abs(dx), abs(dy), abs(dz)) <= radius_mm + tolerance_mm',
                   patient_split={split: list(patients)}, summary=summary, patients=patients,
                   generator_sha256=sha256(Path(__file__)),
                   verification_scope='Reuse previous image QC; check JSON provenance, coordinate projection consistency and file existence. No CT/Mask decompression or raw-image rehashing.')
    return dataset, patient_rows, state_rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(output, dataset, patient_rows, state_rows):
    output.mkdir(parents=True, exist_ok=True)
    (output/'dataset.json').write_text(json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    write_csv(output/'patients.csv', patient_rows)
    write_csv(output/'states.csv', state_rows)
    q = dataset['summary']
    lines = ['# 患者与状态清单说明', '',
             '该数据目录整体归属：'+dataset['split']+'；未随机划分或抽取任何患者。',
             '扫描{}例，有效{}例，预处理未完成{}例，清单核对异常{}例。'.format(
                 q['scanned_patients'], q['ready_patients'], q['not_prepared_patients'], q['invalid_patients']),
             '计划模板{}个；phase共{}个，其中范围内{}个、范围外{}个。'.format(
                 q['templates'], q['phases'], q['main_phases'], q['out_of_range_phases']), '',
             '- dataset.json：后续程序读取；包含患者、配对视角、模板、标签和矩阵的相对路径。',
             '- patients.csv：每患者一行；ready才进入dataset.json，其他患者保留原因供追踪。',
             '- states.csv：每状态一行；两视角保持配对，不把同一phase拆成两个样本。', '',
             '## 用途规则', '',
             'template_only：Origin计划模板，不作为训练phase。',
             'main：相对计划中心的LPS三个方向位移都≤10mm（容差1e-6mm）。',
             'out_of_range_test：至少一个方向超范围，保留作补充评估，不参与训练或模型选择。',
             'split与usage不同：训练患者的超范围状态仍是训练患者来源，不是另一批独立测试集。',
             'use_for_training仅在split=train且usage=main时为True；不自动建立验证集。', '',
             '## 上传服务器', '',
             '上传完整数据目录、完整预处理目录（含本清单子目录）及项目代码，保留患者子目录结构。',
             '清单内不写Windows绝对路径；几何和标签JSON保持原样，不为迁移而修改它们。',
             '服务器接入代码分别传入data_root和preparation_root，按以下方式解析：', '',
             '```python',
             'import json',
             'from pathlib import Path',
             'data_root = Path("/your/server/LIDCTrainDataSet")',
             'preparation_root = Path("/your/server/1022_data_process")',
             'manifest = json.loads((preparation_root / "manifest_'+dataset['split']+'" / "dataset.json").read_text(encoding="utf-8"))',
             'patient_id = next(iter(manifest["patients"]))',
             'patient = manifest["patients"][patient_id]',
             'state = patient["states"][0]',
             'search_kv1 = data_root / state["inputs"]["kv1"]',
             'geometry = preparation_root / patient["geometry"]',
             'labels = preparation_root / patient["training_labels"]',
             '```', '',
             'inputs为DRR图像；annotations、training_labels仅供监督/评估。实际预测使用计划参考与当前DRR，',
             '不要将当前状态Mask或中心标签作为预测输入。当前训练入口尚未适配本清单，不能直接视为已接通训练。', '',
             '本步骤复用已有质检，不重新解压图像、不重新核对全部原始大文件指纹、不调用Demo、不重新计算标签。',
             '本脚本只写本目录内四个清单文件，不修改已有脚本、矩阵、标签或原始数据。', '']
    (output/'说明.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATA_ROOT)
    parser.add_argument('--preparation', type=Path, default=PREPARATION_ROOT)
    parser.add_argument('--split', choices=('train', 'test'), default=DATASET_SPLIT)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--patients', nargs='+', default=None, help='可选：仅检查指定患者；默认扫描全部')
    args = parser.parse_args()
    root, preparation = args.root.resolve(), args.preparation.resolve()
    output = (args.out or preparation/('manifest_'+args.split)).resolve()
    if not root.is_dir() or not (preparation/'patients').is_dir():
        raise ValueError('Data root and preparation/patients must exist')
    if (output == root or root in output.parents or output == preparation or
            output == preparation/'patients' or preparation/'patients' in output.parents):
        raise ValueError('Choose a dedicated manifest folder, outside raw data and prepared patients')
    dataset, patient_rows, state_rows = build_manifest(root, preparation, args.split, args.patients)
    save_outputs(output, dataset, patient_rows, state_rows)
    q = dataset['summary']
    print('Ready patients: {}; main phases: {}; outside phases: {}'.format(
        q['ready_patients'], q['main_phases'], q['out_of_range_phases']))
    print('Not prepared: {}; invalid: {}'.format(q['not_prepared_patients'], q['invalid_patients']))
    print('Output: '+str(output))
    if q['invalid_patients']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
