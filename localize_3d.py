"""第 5 阶段：在由粗到细的三维候选网格上联合计算完整二维热力图的得分。
无神经网络依赖。输入得分必须为非负证据图，而非原始对数概率（logits）。输出得分差值属于启发式指标，并非概率。

用两张二维响应图寻找三维位置，这个脚本也不直接生成文件。它也是工具模块。
它解决的问题是：不再只取 kv1、kv2 各自的 top-1，然后硬做三角定位，而是在三维空间里搜索候选点，看哪个三维点能同时得到两个视角的支持。
"""
import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter
from geometry import project, triangulate

# 从一张二维 response/evidence map 中找到五个局部峰。返回：[u, v, score]
def peaks(response, count=5, min_distance=5):
    response = np.asarray(response, float)
    maxima = response == maximum_filter(response, size=2*min_distance+1)
    indices = np.argwhere(maxima & (response > 0))
    indices = sorted(indices, key=lambda p: response[tuple(p)], reverse=True)
    # 取前 count 个最强峰，然后返回“横坐标、纵坐标、峰值分数”。
    return [[int(p[1]), int(p[0]), float(response[tuple(p)])] for p in indices[:count]]

# 已知两张响应图、两个 X 光相机几何、计划 CT 中的三维结节位置，在计划位置附近搜索一个三维点，使它投影到两张响应图以后，两边都获得较高响应。
# 它不是简单地“两个视角各取最高点再三角定位”，而是让每一个三维候选点都同时去两个响应图里取分数，最后找一个“两边都支持、又离计划位置合理”的三维点。
# 输入maps：两张二维响应图：maps = [response_kv1, response_kv2] 每张图上数值越高，表示网络越认为“结节可能在这里”。
# 输入cameras：两个视角的投影几何
# 输入plan：计划CT中的三维结节中心：
def joint_search(maps, cameras, plan, radius_mm=20., coarse_mm=2., fine_mm=.25,
                 prior_weight=.05, prior_sigma_mm=20.):
    """All candidates sample BOTH maps. The plan determines search box and prior."""
    if len(maps) != 2 or len(cameras) != 2:
        raise ValueError('Exactly two views required')
    if min(radius_mm, coarse_mm, fine_mm, prior_sigma_mm) <= 0 or fine_mm > coarse_mm or prior_weight < 0:
        raise ValueError('Invalid search settings')
    normalized = []
    for a in maps:
        a = np.asarray(a, float)
        if a.ndim != 2 or not np.isfinite(a).all() or a.min() < 0 or a.max() <= 0:
            raise ValueError('Expected finite nonnegative, nonempty evidence map')
        if np.ptp(a) <= 1e-12:
            raise ValueError('Flat map has no localization evidence')
        normalized.append(a/a.max())
    plan = np.asarray(plan, float)
    if plan.shape != (3,) or not np.isfinite(plan).all():
        raise ValueError('Plan must be a finite 3D point')

    def grid(center, radius, step):
        axes = [np.arange(max(plan[d]-radius_mm, center[d]-radius),
                          min(plan[d]+radius_mm, center[d]+radius)+step*.01, step) for d in range(3)]
        return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)

    def score(points):
        values = np.zeros(len(points))
        valid = np.ones(len(points), bool)
        for a, p in zip(normalized, cameras):
            uv, front = project(points, p)
            valid &= front & (uv[:, 0] >= 0) & (uv[:, 0] <= a.shape[1]-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= a.shape[0]-1)
            sample = map_coordinates(a, np.nan_to_num(uv).T[::-1], order=1, mode='constant', cval=0)
            values += np.log(sample+1e-8)
        values -= prior_weight*np.sum(((points-plan)/prior_sigma_mm)**2, axis=1)
        values[~valid] = -np.inf
        return values

    coarse = grid(plan, radius_mm, coarse_mm)
    scores = score(coarse)
    if not np.isfinite(scores).any():
        raise ValueError('No candidate projects inside both response maps')
    # Refine multiple separated coarse peaks; do not commit to just one early peak.
    seeds = []
    for i in np.argsort(scores)[::-1]:
        if not np.isfinite(scores[i]):
            break
        if all(np.linalg.norm(coarse[i]-s) > 2*coarse_mm for s in seeds):
            seeds.append(coarse[i])
        if len(seeds) == 4:
            break
    refined = np.concatenate([grid(s, coarse_mm, fine_mm) for s in seeds])
    candidates = np.concatenate([coarse, refined])
    all_scores = np.r_[scores, score(refined)]
    best = int(np.argmax(all_scores))
    xyz = candidates[best]
    separate = np.linalg.norm(candidates-xyz, axis=1) >= 3.
    runner = np.max(all_scores[separate]) if separate.any() else -np.inf
    support = [float(map_coordinates(a, project(xyz,p)[0].T[::-1], order=1, mode='constant', cval=0)[0])
               for a,p in zip(normalized,cameras)]
    return dict(xyz_lps_mm=xyz.tolist(), delta_lps_mm=(xyz-plan).tolist(),
                view_support=support,
                score=float(all_scores[best]),
                separated_peak_score_gap=float(all_scores[best]-runner) if np.isfinite(runner) else None,
                at_search_boundary=bool(np.any(abs(xyz-plan) >= radius_mm-fine_mm)),
                candidate_count=len(candidates),
                confidence_note='Heuristic score gap; not calibrated probability')


def predict_pair(maps, response_cameras, drr_cameras, plan, top1, radius):
    """主方法和 top-1 baseline；签名中没有当前状态 Mask/中心。"""
    results = dict(plan=dict(status='ok', xyz_lps_mm=list(plan), delta_lps_mm=[0., 0., 0.]))
    try:
        baseline = triangulate(top1[0], top1[1], drr_cameras)
        baseline.update(status='ok', delta_lps_mm=(np.asarray(baseline['xyz_lps_mm'])-plan).tolist())
        results['top1'] = baseline
    except ValueError as error:
        results['top1'] = dict(status='failed', reason=str(error))
    try:
        result = joint_search(maps, response_cameras, plan, radius_mm=radius)
        result['status'] = 'ok'
        # view_support 是按每张响应图最大值归一化后的支持；另保存原始 sigmoid 分数。
        xyz = result['xyz_lps_mm']
        result['raw_view_scores'] = [float(map_coordinates(a, project(xyz, p)[0].T[::-1],
                                          order=1, mode='constant', cval=0)[0])
                                     for a, p in zip(maps, response_cameras)]
        result['projected_uv'] = {v: project(xyz, p)[0][0].tolist()
                                  for v, p in zip(('kv1', 'kv2'), drr_cameras)}
        results['joint'] = result
    except ValueError as error:
        results['joint'] = dict(status='failed', reason=str(error))
    return results
