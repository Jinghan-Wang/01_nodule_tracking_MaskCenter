"""用实际 DRRDemo 黑盒标定：已知三维点 -> 保存图像像素，不猜测角度约定。

所有 Demo 输入、日志和输出只写入独立实验目录，原始数据/旧训练结果不变。
Python 3.7。先运行 --patients LIDC_0001，再决定是否推广到全部患者。
"""
import argparse
import configparser
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import data_process  # 初始化当前Conda环境的DLL路径。

import numpy as np
import SimpleITK as sitk

from geometry import fit_camera, project, triangulate
from data_process.prepare_center_labels import sha256

DEMO = ROOT.parent / '00_GenDRRTools' / 'GenDRRDemo'
FILENAMES = ('DRRDemo.exe', 'DRR.dll', 'cudart64_102.dll')
ANGLES = ('0000_angle_0.0', '0001_angle_150.0')


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def image_array(path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3 or array.shape[0] != 1 or not np.isfinite(array).all():
        raise ValueError('Invalid saved DRR: ' + str(path))
    return array[0].astype(np.float64)


def projection_pair(folder, prefix):
    return np.asarray([image_array(folder / (prefix+'DRR_ROI_NULL') /
                                  (prefix+'DRR_'+suffix+'.nii.gz')) for suffix in ANGLES])


def weighted_center(array):
    weights = np.maximum(array, 0)
    if weights.sum() <= 0:
        raise ValueError('Empty fiducial projection')
    v, u = np.indices(weights.shape)
    return np.array([(weights*u).sum(), (weights*v).sum()])/weights.sum()


def binary_center(image):
    """非空二值三维 Mask 的体素质心；与 Demo 的 mask==1 规则一致。"""
    array = sitk.GetArrayViewFromImage(image)
    if (image.GetDimension() != 3 or not np.isfinite(array).all() or
            not np.all((array == 0) | (array == 1)) or not np.any(array == 1)):
        raise ValueError('Expected a finite, nonempty, binary 3D Mask')
    return np.argwhere(array == 1).mean(0)[::-1]


def read_pose(path):
    """读取状态文件中的 ISO 和旋转，各为三个分量。"""
    fields = dict(line.split('=', 1) for line in path.read_text(encoding='utf-8-sig').splitlines()
                  if '=' in line)
    fields = {key.strip(): value for key, value in fields.items()}
    return np.array([[float(v) for v in fields[key].split(',')] for key in ('ISO', 'Rotation')])


class DemoRunner:
    """复制可执行文件到实验目录运行；绝不修改原 Demo 的配置或数据。"""
    def __init__(self, out, source):
        self.work = out / 'demo_work'
        self.patient = self.work / 'inputs' / 'patient'
        self.state = self.patient / 'trial'
        self.state.mkdir(parents=True, exist_ok=True)
        for name in FILENAMES:
            shutil.copy2(str(source/name), str(self.work/name))
        self.source = source

    def run(self, source_patient, ct, mask, tag, project_ct=False, expected_pose=None):
        if ct.GetSize() != mask.GetSize() or ct.GetSpacing() != mask.GetSpacing() or \
                ct.GetOrigin() != mask.GetOrigin() or ct.GetDirection() != mask.GetDirection():
            raise ValueError('CT / Mask grid mismatch')
        for name in ('congfigfiles_drr_LIDC.ini', 'angle_all.txt'):
            shutil.copy2(str(source_patient/name), str(self.patient/name))
        # 精确删除上一轮的四个实验投影，避免误读旧结果；不递归删除目录。
        for prefix in ('CT', 'MASK'):
            for suffix in ANGLES:
                path = self.state/(prefix+'DRR_ROI_NULL')/(prefix+'DRR_'+suffix+'.nii.gz')
                if path.exists():
                    path.unlink()
        # NIfTI 的说明性字符串元数据不能写入 MHD；只保留像素及真实空间信息。
        for image,name in ((ct,'CT.mhd'),(mask,'mask.mhd')):
            clean = sitk.Image(image)
            for key in clean.GetMetaDataKeys():
                clean.EraseMetaData(key)
            sitk.WriteImage(clean,str(self.state/name))
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(str(self.source/'paraset.ini'), encoding='utf-8-sig')
        config['Batch'].update(Mode='Single', SinglePatientRoot=str(self.patient.resolve()),
                               PatientsRoot=str(self.patient.parent.resolve()),
                               DataDirectoryPattern='trial')
        config['Inputs'].update(ProcessCT=str(project_ct).lower(), ProcessMask='true')
        pose_path = self.state/'paraset.ini'
        if expected_pose is not None:
            # 用固定的实际 ISO 投影，避免 MaskCenter 随人工体模位置改变。
            index = ((np.asarray(ct.GetSize(), dtype=float)-1)/2).tolist()
            center = ct.TransformContinuousIndexToPhysicalPoint(index)
            offset = expected_pose[0] - center
            config['DRR'].update(ProjectionCenter='ImageCenter',
                                 ISOOffsetX=str(offset[0]), ISOOffsetY=str(offset[1]),
                                 ISOOffsetZ=str(offset[2]))
            if pose_path.exists():
                pose_path.unlink()
        with (self.work/'paraset.ini').open('w', encoding='utf-8') as stream:
            config.write(stream)
        # Demo结束会等待回车；关闭标准输入，确保PyCharm批量运行时自动退出。
        process = subprocess.run([str((self.work/'DRRDemo.exe').resolve())],
                                 cwd=str(self.work), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=180,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        (self.work/(tag+'.stdout.txt')).write_bytes(process.stdout)
        if process.returncode != 0:
            raise RuntimeError('Demo failed: '+tag+'; see experiment logs')
        images = projection_pair(self.state, 'MASK')
        if expected_pose is not None and not np.allclose(read_pose(pose_path), expected_pose,
                                                         atol=1e-5, rtol=0):
            raise ValueError('Demo 实际 ISO 或旋转与保存的状态不符：'+tag)
        pose = pose_path.read_text(encoding='utf-8-sig')
        return images, pose


def phantom(reference, center_index, sigma_mm=1.5):
    """离散高斯小体模：空气 -1000 HU，记录实际衰减权重的三维质心。"""
    spacing = np.asarray(reference.GetSpacing())
    size = np.asarray(reference.GetSize())
    radius = np.ceil(4*sigma_mm/spacing).astype(int)
    index = np.asarray(center_index)
    lo, hi = index-radius, index+radius+1
    if np.any(lo < 0) or np.any(hi > size):
        raise ValueError('Fiducial outside CT grid')
    axes = [(np.arange(lo[d],hi[d])-index[d])*spacing[d] for d in range(3)]
    x, y, z = np.meshgrid(*axes, indexing='ij')
    attenuation = np.rint(4000*np.exp(-(x*x+y*y+z*z)/(2*sigma_mm**2))).astype(np.int32)
    array = np.full(tuple(size[::-1]), -1000, dtype=np.int32)
    array[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]] = attenuation.transpose(2,1,0)-1000
    ct = sitk.GetImageFromArray(array)
    ct.CopyInformation(reference)
    mask = sitk.Cast(ct > -1000, sitk.sitkUInt8)
    indices = np.stack(np.meshgrid(*[np.arange(lo[d],hi[d]) for d in range(3)],
                                  indexing='ij'), axis=-1)
    actual_index = (indices*attenuation[...,None]).sum(axis=(0,1,2))/attenuation.sum()
    xyz = reference.TransformContinuousIndexToPhysicalPoint(actual_index.tolist())
    return ct, mask, np.asarray(xyz)


def compare_images(generated, saved):
    """同时核查形状、像素数值和 >0 支持区域，不用二维质心去调整相机。"""
    if generated.shape != saved.shape:
        raise ValueError('重投影图像和已保存图像的尺寸不一致')
    a, b = generated > 0, saved > 0
    union = np.logical_or(a,b).sum()
    return dict(mask_iou=float(np.logical_and(a,b).sum()/union) if union else 0.,
                max_abs_difference=float(np.max(abs(generated-saved))),
                mean_abs_difference=float(np.mean(abs(generated-saved))),
                identical_pixels=bool(np.array_equal(generated,saved)))


def plan_folder_name(folder):
    """兼容旧original和新Origin；同时存在时拒绝猜测模板。"""
    names = [name for name in ('original', 'Origin') if (folder/name).is_dir()]
    if len(names) != 1:
        raise ValueError('Expected exactly one original/Origin folder: '+str(folder))
    return names[0]


def patient_fingerprints(folder):
    source_states = [plan_folder_name(folder)]+sorted(p.name for p in folder.glob('phase_*') if p.is_dir())
    sources = ['angle_all.txt','congfigfiles_drr_LIDC.ini']
    for state in source_states:
        sources.extend(state+'/'+name for name in ('CT.nii.gz','mask.nii.gz','paraset.ini'))
        for prefix in ('CT','MASK'):
            sources.extend(state+'/'+prefix+'DRR_ROI_NULL/'+prefix+'DRR_'+suffix+'.nii.gz'
                           for suffix in ANGLES)
    return {name:sha256(folder/name) for name in sources}


def calibrate(folder, out, runner, count=32, export_labels=True):
    out.mkdir(parents=True, exist_ok=True)
    plan_name = plan_folder_name(folder)
    reference = sitk.ReadImage(str(folder/plan_name/'mask.nii.gz'))
    source_states = [plan_name]+sorted(p.name for p in folder.glob('phase_*') if p.is_dir())
    fingerprints = patient_fingerprints(folder)
    center_index = binary_center(reference)
    plan = np.asarray(reference.TransformContinuousIndexToPhysicalPoint(center_index.tolist()))
    plan_drr = folder/plan_name/'CTDRR_ROI_NULL'/('CTDRR_'+ANGLES[0]+'.nii.gz')
    detector = sitk.ImageFileReader()
    detector.SetFileName(str(plan_drr))
    detector.ReadImageInformation()
    w, h = detector.GetSize()[:2]
    config = configparser.ConfigParser()
    config.read(str(runner.source/'paraset.ini'), encoding='utf-8-sig')
    fixed_pose = {}
    if config.get('DRR', 'ProjectionCenter', fallback='ImageCenter').lower() == 'maskcenter':
        expected = read_pose(folder/plan_name/'paraset.ini')
        for state in source_states:
            if not np.allclose(read_pose(folder/state/'paraset.ini'), expected, atol=1e-5, rtol=0):
                raise ValueError('状态姿态不同于计划态，不能共用一对投影矩阵：'+state)
        fixed_pose['expected_pose'] = expected
    # 同样的完整 CT 网格/物理中心；标定区域为计划结节附近的 CT 轴向 ±18 mm。
    spacing = np.asarray(reference.GetSpacing())
    rng = np.random.RandomState(2026)
    xyz, pixels, poses = [], [], []
    for number, offset in enumerate(rng.uniform(-18,18,(count,3))):
        index = np.rint(center_index+offset/spacing).astype(int)
        ct, mask, position = phantom(reference, index)
        images, pose = runner.run(folder, ct, mask, folder.name+'_fiducial_{:02d}'.format(number),
                                  **fixed_pose)
        if images.shape[1:] != (h, w):
            raise ValueError('体模投影与已保存 DRR 尺寸不一致')
        uv = np.asarray([weighted_center(image) for image in images])
        if np.any(uv < 5) or np.any(uv > np.array([w, h])-6):
            raise ValueError('Fiducial near detector border')
        xyz.append(position)
        pixels.append(uv)
        poses.append(pose)
        np.savez_compressed(str(out/'fiducial_{:02d}.npz'.format(number)), images=images)
        if (number+1) % 4 == 0:
            print('{} actual Demo fiducials {}/{}'.format(folder.name,number+1,count), flush=True)
    xyz, pixels = np.asarray(xyz), np.asarray(pixels)
    fit_count = count*3//4
    cameras = [fit_camera(xyz[:fit_count],pixels[:fit_count,v]) for v in range(2)]
    repro = np.asarray([np.linalg.norm(project(xyz[fit_count:],p)[0]-pixels[fit_count:,v],axis=1)
                       for v,p in enumerate(cameras)])
    reconstructed = np.asarray([triangulate(uv[0],uv[1],cameras)['xyz_lps_mm']
                                for uv in pixels[fit_count:]])
    error = np.linalg.norm(reconstructed-xyz[fit_count:],axis=1)
    save_json(out/'fiducials.json', dict(xyz_lps_mm=xyz.tolist(),pixels_uv=pixels.tolist(),
                                       fit_count=fit_count,generated_poses=poses,
                                       heldout_projection_error_px=repro.tolist(),
                                       heldout_reconstruction_error_mm=error.tolist()))
    affine = np.eye(4)
    affine[:3,:3] = np.asarray(reference.GetDirection()).reshape(3,3).dot(np.diag(spacing))
    affine[:3,3] = reference.GetOrigin()
    geometry = dict(patient=folder.name,coordinate_system='patient LPS mm',
                    cameras=[p.tolist() for p in cameras],plan_lps_mm=plan.tolist(),
                    voxel_to_lps=affine.tolist(),volume_shape=list(reference.GetSize()),
                    detector_image_shape=[h,w], detector_spacing_uv_mm=list(detector.GetSpacing()[:2]),
                    calibrated_radius_mm=18,
                    geometry_verified=False,geometry_status='demo_calibrated_pending_saved_data_check',
                    heldout_reprojection_max_px=float(repro.max()),
                    heldout_reconstruction_max_mm=float(error.max()),
                    binaries_sha256={name:sha256(runner.source/name) for name in FILENAMES},
                    source_files_sha256={name:sha256(folder/name) for name in
                        ('angle_all.txt','congfigfiles_drr_LIDC.ini',plan_name+'/paraset.ini')})
    save_json(out/'geometry.json',geometry)
    if repro.max() > 1 or error.max() > 2:
        raise ValueError('Held-out fiducial check failed; do not update labels')
    audit = []
    records = {}
    for state in source_states:
        ct = sitk.ReadImage(str(folder/state/'CT.nii.gz'))
        mask = sitk.ReadImage(str(folder/state/'mask.nii.gz'))
        if (mask.GetSize() != reference.GetSize() or mask.GetSpacing() != reference.GetSpacing()
                or mask.GetOrigin() != reference.GetOrigin() or mask.GetDirection() != reference.GetDirection()):
            raise ValueError('State grid differs from calibrated original: '+state)
        center_index = binary_center(mask)
        center = np.asarray(mask.TransformContinuousIndexToPhysicalPoint(center_index.tolist()))
        if np.any(abs(np.linalg.solve(affine[:3,:3],center-plan)*spacing) > 18):
            raise ValueError('State center leaves local calibrated region: '+state)
        uv = [project(center,p)[0][0] for p in cameras]
        if not np.isfinite(uv).all() or np.any(np.asarray(uv) < 0) or np.any(np.asarray(uv) >= [w,h]):
            raise ValueError('State projected label outside detector: '+state)
        records[state] = dict(center_lps_mm=center.tolist(),
                              projected_uv={v:p.tolist() for v,p in zip(('kv1','kv2'),uv)},
                              mask_sha256=fingerprints[state+'/mask.nii.gz'])
        images, pose = runner.run(folder,ct,mask,folder.name+'_'+state,project_ct=True, **fixed_pose)
        original_pose = (folder/state/'paraset.ini').read_text(encoding='utf-8-sig')
        current = []
        generated_ct = projection_pair(runner.state,'CT')
        for v,view in enumerate(('kv1','kv2')):
            mask_result = compare_images(images[v],projection_pair(folder/state,'MASK')[v])
            ct_result = compare_images(generated_ct[v],projection_pair(folder/state,'CT')[v])
            record = dict(state=state,view=view,mask=mask_result,ct=ct_result,
                          generated_pose=pose,saved_pose=original_pose)
            audit.append(record)
            current.append(mask_result['mask_iou'])
        np.savez_compressed(str(out/(state+'_reproduced.npz')),mask=images,ct=generated_ct)
        save_json(out/'saved_data_audit.json',audit)
        print('{} {} saved Mask IoU: {}'.format(folder.name,state,current),flush=True)
    # 重现保存的 CT 和 Mask 是关键门槛；失败时保持独立实验，不导出训练标签。
    unchanged = all(sha256(folder/name) == digest for name,digest in fingerprints.items())
    save_json(out/'source_fingerprints.json',dict(files=fingerprints,unchanged=unchanged))
    matched = unchanged and all(r['mask']['mask_iou'] >= .98 and
                               r['ct']['mean_abs_difference'] <= 1e-5 for r in audit)
    geometry.update(geometry_verified=bool(matched),
                    geometry_status='demo_verified_against_saved_data' if matched else
                                    'demo_calibrated_saved_data_mismatch',
                    minimum_reproduced_mask_iou=min(r['mask']['mask_iou'] for r in audit),
                    saved_mask_reprojection_scope='all states; CT and Mask pixel comparison')
    save_json(out/'geometry.json',geometry)
    if matched and export_labels:
        labels = dict(label_definition='projected_3d_mask_centroid_lps',
                      geometry_status=geometry['geometry_status'],
                      geometry_sha256=sha256(out/'geometry.json'),
                      geometry_path=str((out/'geometry.json').resolve()),
                      cameras=geometry['cameras'],detector_image_shape=geometry['detector_image_shape'],states=records,
                      note='Current-state Mask centers are training/evaluation targets only.')
        save_json(out/'projected_centers.json',labels)
        labels = dict(labels,states={'original':records[plan_name]})
        save_json(out/'original_labels.json',labels)
    return geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'Data'/'LIDC_one')
    parser.add_argument('--demo',type=Path,default=DEMO)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs'/'lidc_demo_geometry')
    parser.add_argument('--patients',nargs='+',default=['LIDC_0001'])
    parser.add_argument('--all',action='store_true',help='独立核查所有患者')
    parser.add_argument('--resume',action='store_true',help='跳过已验证且数据/程序指纹不变的患者')
    parser.add_argument('--count',type=int,default=32)
    parser.add_argument('--workers',type=int,default=1,choices=(1,2),help='最多两个独立 Demo 进程')
    args = parser.parse_args()
    if args.count < 16:
        raise ValueError('At least 16 fiducials required')
    if (args.out/'training'/'best.pt').exists():
        raise ValueError('This preparation already has trained weights; choose a new --out')
    args.out.mkdir(parents=True,exist_ok=True)
    report = {}
    patients = sorted(p.name for p in args.root.glob('LIDC_*') if p.is_dir()) if args.all else args.patients
    if not patients or len(patients) != len(set(patients)):
        raise ValueError('Patient list must be nonempty and unique')
    pending = []
    for name in patients:
        folder, destination = args.root/name, args.out/'patients'/name
        geometry_path = destination/'geometry.json'
        if args.resume and geometry_path.exists() and (destination/'projected_centers.json').exists():
            old = json.loads(geometry_path.read_text(encoding='utf-8'))
            fingerprints = json.loads((destination/'source_fingerprints.json').read_text(encoding='utf-8'))
            if (old['geometry_verified'] and
                    all(sha256(folder/file) == digest for file,digest in fingerprints['files'].items()) and
                    all(sha256(args.demo/file) == digest for file,digest in old['binaries_sha256'].items())):
                report[name] = old
                print('Reused verified '+name,flush=True)
                continue
        pending.append(name)
    def run_patient(name):
        runner = DemoRunner(args.out/'runtime'/name,args.demo)
        geometry = calibrate(args.root/name,args.out/'patients'/name,runner,args.count)
        # 仅清理本次运行生成的三个完整体积，保留小体模投影、日志、配置与审计。
        for stem in ('CT','mask','MASK_mul'):
            for suffix in ('.mhd','.raw'):
                path = runner.state/(stem+suffix)
                if path.exists():
                    path.unlink()
        return geometry
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_patient,name):name for name in pending}
        for future in as_completed(futures):
            name = futures[future]
            report[name] = future.result()
            save_json(args.out/'summary.json',report)
            print('Completed {}/{} patients'.format(len(report),len(patients)),flush=True)
    if args.all:
        from lidc_data import patient_split
        status = 'prepared_verified' if all(p['geometry_verified'] for p in report.values()) else 'geometry_check_failed'
        dataset = dict(root=str(args.root.resolve()),seed=42,patient_split=patient_split(patients),
                       status=status,geometry_status='demo_verified_against_saved_data',
                       patients={name:dict(phases=sorted(p.name for p in (args.root/name).glob('phase_*') if p.is_dir()),
                                          geometry_status=g['geometry_status']) for name,g in report.items()})
        save_json(args.out/'dataset.json',dataset)


if __name__ == '__main__':
    main()
