"""Phase 5 end-to-end QC, original mode only; no network or training changes.

Run in pytorch_3.7: python data_process/check_geometry.py
Uses actual preprocessing/TIGRE to calibrate patient-LPS -> saved DRR pixels.
“双视角投影几何 + 三角定位 + 三维联合搜索”整套流程质检

"""
import argparse
import ast
import csv
import hashlib
import json
import sys
import types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import data_process  # 初始化DLL路径，再加载TIGRE/CUDA。
# Load TIGRE/CUDA before imaging/plotting DLLs on Windows.
import tigre
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import fit_camera, project, triangulate
from localize_3d import joint_search, peaks



def load_projector():
    """Import original module; if nibabel is absent, load ONLY its unchanged math.

    SimpleITK handles I/O here. The fallback compiles an explicit function allowlist
    from the actual source, rather than maintaining a second projection formula.
    """
    sys.path.insert(0, str(ROOT/'preprocessing'))
    try:
        import preprocess
        return preprocess, 'direct import'
    except ModuleNotFoundError as error:
        if error.name != 'nibabel':
            raise
    import tigre
    source = ROOT/'preprocessing'/'preprocess.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = {'load_config', '_cfg', 'pad_and_shift', 'build_tigre_geometry', 'project_drrs'}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_CONFIG' for t in node.targets):
            selected.append(node)
    namespace = dict(np=np, tigre=tigre, types=types, json=json, ndimage=ndimage, _TIGRE_AVAILABLE=True)
    module = ast.Module(body=selected) if sys.version_info < (3, 8) else ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(source), 'exec'), namespace)
    return types.SimpleNamespace(**namespace), 'original functions compiled; SimpleITK I/O (nibabel unavailable)'


def read_volume(path):
    im = sitk.ReadImage(str(path))
    if im.GetDimension() != 3:
        raise ValueError('Expected 3D input: '+str(path))
    # Reverse SimpleITK array order to reproduce nibabel data[i,j,k].
    volume = sitk.GetArrayFromImage(im).transpose(2, 1, 0).astype(np.float32)
    if not np.isfinite(volume).all() or not np.any(volume > 0):
        raise ValueError('Invalid or empty mask: '+str(path))
    affine = np.eye(4)
    affine[:3, :3] = np.array(im.GetDirection()).reshape(3, 3).dot(np.diag(im.GetSpacing()))
    affine[:3, 3] = im.GetOrigin()
    center_index = np.argwhere(volume > 0).mean(0)
    center = affine[:3, :3].dot(center_index)+affine[:3, 3]
    return volume, affine, np.array(im.GetSpacing()), center_index, center


def same_ct_grid(ct_path, shape, affine):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(ct_path))
    reader.ReadImageInformation()
    a = np.eye(4)
    a[:3, :3] = np.array(reader.GetDirection()).reshape(3, 3).dot(np.diag(reader.GetSpacing()))
    a[:3, 3] = reader.GetOrigin()
    if tuple(reader.GetSize()) != tuple(shape) or not np.allclose(a, affine, atol=1e-5, rtol=0):
        raise ValueError('CT/mask grid mismatch: '+str(ct_path))


def forward(pp, cfg, volume, spacing):
    padded, _ = pp.pad_and_shift(volume, spacing, np.zeros(3), cval=0.)
    projections = pp.project_drrs(padded, spacing, cfg)
    # project_lidc_* saves raw [a,b] with nibabel, our DataLoader reads [b,a].
    return np.asarray(projections).transpose(0, 2, 1)


def centroid(image):
    v, u = np.indices(image.shape)
    total = image.sum()
    if total <= 0:
        raise ValueError('Empty point projection')
    return np.array([(u*image).sum()/total, (v*image).sum()/total])


def point_phantom(shape, index, spacing, sigma_mm=1.5):
    """Small isotropic Gaussian fiducial; preserve complete original volume grid."""
    volume = np.zeros(shape, np.float32)
    radius = np.ceil(4*sigma_mm/spacing).astype(int)
    lo = np.maximum(0, np.floor(index).astype(int)-radius)
    hi = np.minimum(shape, np.ceil(index).astype(int)+radius+1)
    axes = [(np.arange(lo[d], hi[d])-index[d])*spacing[d] for d in range(3)]
    x, y, z = np.meshgrid(*axes, indexing='ij')
    volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = np.exp(-(x*x+y*y+z*z)/(2*sigma_mm**2))
    return volume


def heatmap(shape, uv, sigma=1.5):
    v, u = np.indices(shape)
    return np.exp(-((u-uv[0])**2+(v-uv[1])**2)/(2*sigma**2))

# 人工构造一个“完全已知答案”的理想几何环境，检查 fit_camera()、project()、triangulate()、joint_search() 这些核心数学函数有没有写错
def numerical_checks():
    """Independent analytic pinhole model, not the fitted TIGRE cameras."""
    a = np.array([[800,0,128,128000],[0,800,128,128000],[0,0,1,1000.]],float)
    b = np.array([[128,800,0,128000],[128,0,800,128000],[1,0,0,1000.]],float)
    points = np.random.RandomState(7).uniform(-20,20,(30,3))
    fitted = fit_camera(points[:20],project(points[:20],a)[0])
    assert np.max(abs(project(points[20:],fitted)[0]-project(points[20:],a)[0])) < 1e-8
    for point in points:
        result=triangulate(project(point,a)[0][0],project(point,b)[0][0],[a,b])
        assert np.linalg.norm(np.array(result['xyz_lps_mm'])-point) < 1e-7
    try:
        triangulate([128,128],[128,128],[a,a])
    except ValueError:
        pass
    else:
        raise AssertionError('Parallel-ray rejection failed')
    for empty in [np.zeros((8,8)),np.ones((8,8))]:
        try:
            joint_search([empty,empty],[a,b],[0,0,0])
        except ValueError:
            pass
        else:
            raise AssertionError('Uninformative map rejection failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'config'/'preprocess_config.json')
    p.add_argument('--volume-root', type=Path, default=ROOT/'datasets'/'Patient_1')
    p.add_argument('--drr-root', type=Path, default=ROOT/'datasets'/'DRR'/'Patient_1')
    p.add_argument('--out', type=Path, default=ROOT/'outputs'/'phase5')
    args = p.parse_args()
    numerical_checks()
    args.out.mkdir(parents=True, exist_ok=True)
    pp, import_mode = load_projector()
    cfg = pp.load_config(str(args.config))
    if len(cfg['drr_angles_deg']) != 2:
        raise ValueError('Exactly two angles required')
    mask_path = args.volume_root/'original'/'original_1_tumor_mask.nii.gz'
    volume, affine, spacing, index, plan = read_volume(mask_path)
    shape = volume.shape
    same_ct_grid(mask_path.with_name('original_1_ct.nii.gz'), shape, affine)
    observed_shape = tuple(cfg['kv_size'][::-1])
    print('Calibrating actual TIGRE, shape={}, plan LPS={}'.format(shape, plan), flush=True)
    rng = np.random.RandomState(2026)
    points, pixels = [], []
    # Fit and held-out phantoms use different independently sampled locations.
    for number, displacement in enumerate(rng.uniform(-18, 18, (32, 3))):
        ijk = index + displacement/spacing
        world = affine[:3, :3].dot(ijk)+affine[:3, 3]
        images = forward(pp, cfg, point_phantom(shape, ijk, spacing), spacing)
        uv = np.array([centroid(im) for im in images])
        if np.any(uv < 4) or np.any(uv[:, 0] > observed_shape[1]-5) or np.any(uv[:, 1] > observed_shape[0]-5):
            raise ValueError('Calibration fiducial near detector border; reduce calibration range')
        points.append(world)
        pixels.append(uv)
        if (number+1) % 8 == 0:
            print('Projected calibration fiducials {}/32'.format(number+1), flush=True)
    points, pixels = np.array(points), np.array(pixels)
    cameras = [fit_camera(points[:24], pixels[:24, v]) for v in range(2)]
    errors = [np.linalg.norm(project(points[24:], cameras[v])[0]-pixels[24:, v], axis=1) for v in range(2)]
    reconstructed = np.array([triangulate(uv[0],uv[1],cameras)['xyz_lps_mm'] for uv in pixels[24:]])
    roundtrip = [np.linalg.norm(np.array(triangulate(project(x,cameras[0])[0][0],project(x,cameras[1])[0][0],cameras)['xyz_lps_mm'])-x) for x in points[24:]]
    calibration = dict(coordinate_system='patient LPS mm; +X left, +Y posterior, +Z superior',
                       mode='original', cameras=[c.tolist() for c in cameras],
                       volume_shape=list(shape), voxel_to_lps=affine.tolist(), plan_lps_mm=plan.tolist(),
                       valid_region='Locally validated within approximately 18 mm per CT array axis around original target',
                       config=cfg, projection_import=import_mode,
                       projection_sha256=hashlib.sha256((ROOT/'preprocessing'/'preprocess.py').read_bytes()).hexdigest(),
                       mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                       tigre_path=str(pp.tigre.__file__), detector_image_shape=list(observed_shape),
                       heldout_reprojection_max_px=float(np.max(errors)),
                       heldout_reconstruction_max_mm=float(np.linalg.norm(reconstructed-points[24:],axis=1).max()),
                       algebraic_roundtrip_max_mm=float(max(roundtrip)))
    np.savez_compressed(str(args.out/'calibration_points.npz'), points_lps_mm=points, pixels_uv=pixels, fit_count=24)
    print('Calibration held-out max pixel error {:.4f}'.format(calibration['heldout_reprojection_max_px']), flush=True)
    rows, centers, mask_centers = [], {}, {}
    states = ['original']+sorted(d.name for d in args.volume_root.glob('phase_*') if d.is_dir())
    for state in states:
        folder = args.volume_root/state
        path = folder/'original_1_tumor_mask.nii.gz' if state == 'original' else folder/'rand_0000-tumor.nii.gz'
        vol, a, sp, _, center = read_volume(path)
        ct_path = folder/('original_1_ct.nii.gz' if state == 'original' else 'rand_0000-warped.nii.gz')
        same_ct_grid(ct_path, vol.shape, a)
        if vol.shape != shape or not np.allclose(a, affine, atol=1e-5, rtol=0):
            raise ValueError('Phase grid differs; needs its own geometry calibration: '+state)
        projected = forward(pp, cfg, (vol>0).astype(np.float32), sp)
        centers[state] = center
        uv_pair = []
        for v in range(2):
            saved_path = args.drr_root/state/('patient-1-{}-sample-0000-kv{}-mask-binary.nii.gz'.format(state,v+1))
            saved = sitk.GetArrayFromImage(sitk.ReadImage(str(saved_path)))
            if saved.shape != observed_shape or not np.isin(saved, [0,1]).all() or not saved.any():
                raise ValueError('Invalid saved binary projection: '+str(saved_path))
            repro = projected[v] > 1e-6
            iou = float(np.logical_and(saved,repro).sum()/np.logical_or(saved,repro).sum())
            mask_uv = centroid(saved)
            projected_center = project(center,cameras[v])[0][0]
            rows.append([state,'kv'+str(v+1),iou,*mask_uv,*projected_center,float(np.linalg.norm(mask_uv-projected_center))])
            uv_pair.append(mask_uv)
        mask_centers[state] = uv_pair
        print('Reproduced masks: '+state, flush=True)
    with (args.out/'mask_center_audit.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['state','view','reproduced_mask_iou','mask_u','mask_v','projected_3d_center_u','projected_3d_center_v','center_difference_px'])
        writer.writerows(rows)
    calibration['minimum_reproduced_mask_iou'] = min(r[2] for r in rows)
    calibration['geometry_verified'] = bool(calibration['heldout_reprojection_max_px'] < .3 and
        calibration['heldout_reconstruction_max_mm'] < 1.0 and calibration['minimum_reproduced_mask_iou'] >= .98)
    (args.out/'geometry.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')

    experiments = []
    examples = []
    for state in states:
        truth = centers[state]
        true_uv = [project(truth,c)[0][0] for c in cameras]
        ideal = [heatmap(observed_shape,uv) for uv in true_uv]
        clean = joint_search(ideal,cameras,plan)
        # False peaks deliberately inconsistent across views (vertical shift in one view).
        false_uv = [true_uv[0]+[-22,0], true_uv[1]+[-20,24]]
        noisy = [np.maximum(h,1.5*heatmap(observed_shape,f)) for h,f in zip(ideal,false_uv)]
        top = [peaks(h,1)[0][:2] for h in noisy]
        tri = triangulate(top[0],top[1],cameras)
        joint = joint_search(noisy,cameras,plan)
        no_prior = joint_search(noisy,cameras,plan,prior_weight=0)
        mask_tri = triangulate(*mask_centers[state],cameras)
        experiments.append(dict(state=state,truth_lps_mm=truth.tolist(),
            displacement_lps_mm=(truth-plan).tolist(),
            plan_error_mm=float(np.linalg.norm(truth-plan)),
            ideal_joint_error_mm=float(np.linalg.norm(np.array(clean['xyz_lps_mm'])-truth)),
            false_peak_top1_error_mm=float(np.linalg.norm(np.array(tri['xyz_lps_mm'])-truth)),
            false_peak_ray_gap_mm=tri['ray_gap_mm'],
            false_peak_joint_error_mm=float(np.linalg.norm(np.array(joint['xyz_lps_mm'])-truth)),
            no_prior_joint_error_mm=float(np.linalg.norm(np.array(no_prior['xyz_lps_mm'])-truth)),
            binary_centroid_triangulation_error_mm=float(np.linalg.norm(np.array(mask_tri['xyz_lps_mm'])-truth)),
            joint=joint))
        if state == 'phase_0007':
            examples = [noisy,true_uv,top,joint,truth]
    report = dict(phase='5 only; no neural response maps used', geometry_verified=calibration['geometry_verified'],
                  calibration=calibration, experiments=experiments,
                  caveats=['Synthetic maps centred on projected 3D truth are solver tests, not learned localization accuracy.',
                           'Binary projection centroid and projected 3D centroid are distinct labels.',
                           'Camera calibration is locally validated; score gap is not calibrated confidence.'])
    report['solver_checks_passed'] = bool(max(e['ideal_joint_error_mm'] for e in experiments) < 2.0 and
        all(e['false_peak_joint_error_mm'] < e['false_peak_top1_error_mm'] for e in experiments))
    report['overall_phase5_passed'] = calibration['geometry_verified'] and report['solver_checks_passed']
    with (args.out/'experiments.csv').open('w',newline='') as f:
        fields = ['state','plan_error_mm','ideal_joint_error_mm','false_peak_top1_error_mm',
                  'false_peak_ray_gap_mm','false_peak_joint_error_mm','no_prior_joint_error_mm',
                  'binary_centroid_triangulation_error_mm']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(experiments)
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if examples:
        maps, uv, top, result, truth = examples
        fig, axes = plt.subplots(1,2,figsize=(11,5))
        for v,ax in enumerate(axes):
            ax.imshow(maps[v],origin='upper',cmap='magma')
            ax.plot(*uv[v],'g+',label='true projection')
            ax.plot(*top[v],'rx',label='false top-1')
            ax.plot(*project(result['xyz_lps_mm'],cameras[v])[0][0],'co',fillstyle='none',label='joint 3D projection')
            ax.set(title='kv{} synthetic multi-peak QC'.format(v+1),xlabel='u (pixels)',ylabel='v (pixels)')
            ax.legend()
        fig.tight_layout(); fig.savefig(str(args.out/'false_peak_qc.png'),dpi=140); plt.close(fig)
    print(json.dumps(dict(geometry_verified=calibration['geometry_verified'],
                         heldout_projection_max_px=calibration['heldout_reprojection_max_px'],
                         heldout_3d_max_mm=calibration['heldout_reconstruction_max_mm'],
                         min_mask_iou=calibration['minimum_reproduced_mask_iou'],
                         mean_ideal_error_mm=float(np.mean([x['ideal_joint_error_mm'] for x in experiments])),
                         mean_false_top1_error_mm=float(np.mean([x['false_peak_top1_error_mm'] for x in experiments])),
                         mean_false_joint_error_mm=float(np.mean([x['false_peak_joint_error_mm'] for x in experiments]))),indent=2))
    if not report['overall_phase5_passed']:
        raise SystemExit('Geometry QC failed: inspect report before using on real DRRs. Phase 6 is not enabled.')


if __name__ == '__main__':
    main()
