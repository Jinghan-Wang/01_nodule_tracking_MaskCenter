"""Phase 5: calibrated perspective cameras. Points use patient LPS millimetres.

Pixel convention matches data_loader.py: image[v,u], zero-based pixel centres.
Camera matrices are calibrated against the actual preprocessing/TIGRE projector,
so no guessed Euler-angle, array-axis or detector-sign convention is needed.
最底层的“双视角几何工具”
这个文件本身不会生成 CSV、JSON 或图片文件。
它更像一个“数学工具箱”，被另外两个脚本调用。

"""
import numpy as np


def response_camera(camera, origin_uv, stride):
    """DRR 投影矩阵转为响应图投影矩阵：response_uv=(DRR_uv-origin)/stride。"""
    camera = np.asarray(camera, float)
    origin = np.asarray(origin_uv, float)
    if (camera.shape != (3, 4) or origin.shape != (2,) or
            not np.isfinite(camera).all() or not np.isfinite(origin).all() or
            not np.isfinite(stride) or stride <= 0):
        raise ValueError('Invalid response coordinate mapping')
    transform = np.array([[1./stride, 0., -origin[0]/stride],
                          [0., 1./stride, -origin[1]/stride], [0., 0., 1.]])
    return transform.dot(camera)

#把患者 LPS 坐标中的三维点 [X,Y,Z] 投影成 DRR 上的二维坐标 [u,v]。同时返回 valid，告诉你这个点是否在相机前方、投影是否有效。
def project(points, camera):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    h = np.c_[points, np.ones(len(points))].dot(np.asarray(camera).T)
    valid = np.isfinite(h).all(1) & (h[:, 2] > 1e-9)
    uv = np.full((len(points), 2), np.nan)
    uv[valid] = h[valid, :2] / h[valid, 2:3]
    return uv, valid

# 根据很多已知的“3D点 ↔ 2D投影点”对应关系，反推出一个 3×4 相机矩阵。
def fit_camera(points, pixels):
    """Normalized DLT from >=6 noncoplanar point/pixel correspondences."""
    xyz, uv = np.asarray(points, float), np.asarray(pixels, float)
    if xyz.shape != (len(uv), 3) or uv.shape[1:] != (2,) or len(xyz) < 6:
        raise ValueError('Need at least six 3D/2D correspondences')
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all() or np.linalg.matrix_rank(xyz-xyz.mean(0)) < 3:
        raise ValueError('Invalid or coplanar calibration points')
    def normalize(x):
        c = x.mean(0)
        scale = np.sqrt(x.shape[1]) / np.linalg.norm(x-c, axis=1).mean()
        t = np.eye(x.shape[1]+1)
        t[:-1, :-1] *= scale
        t[:-1, -1] = -c*scale
        return np.c_[x, np.ones(len(x))].dot(t.T), t
    x, t3 = normalize(xyz)
    y, t2 = normalize(uv)
    a = []
    for point, pixel in zip(x, y):
        a.append(np.r_[point, np.zeros(4), -pixel[0]*point])
        a.append(np.r_[np.zeros(4), point, -pixel[1]*point])
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    if s[-2] < 1e-9:
        raise ValueError('Degenerate camera calibration')
    p = np.linalg.inv(t2).dot(vt[-1].reshape(3, 4)).dot(t3)
    p /= np.linalg.norm(p[2, :3])
    if np.mean(np.c_[xyz, np.ones(len(xyz))].dot(p[2])) < 0:
        p = -p
    return p

# 把一个二维像素点 [u,v] 反投影成一条三维射线。
def backproject(pixel, camera):
    p = np.asarray(camera, float)
    source = -np.linalg.solve(p[:, :3], p[:, 3])
    direction = np.linalg.solve(p[:, :3], np.r_[pixel, 1.0])
    direction /= np.linalg.norm(direction)
    return source, direction

# 用 kv1 和 kv2 两个二维点对应的两条三维射线，反推出三维位置。
# 返回的字典内容：
# xyz_lps_mm：最终三维位置
# ray_gap_mm：两条射线最近位置之间还有多大间隙
# ray_angle_deg：两条射线夹角
# reprojection_error_px：重建出来的三维点重新投回 kv1、kv2 后，与原来的二维点相差多少像素。
def triangulate(pixel_a, pixel_b, cameras, min_angle_deg=.1):
    """Closest points on two rays; report gap and reprojection errors, not just XYZ."""
    sa, da = backproject(pixel_a, cameras[0])
    sb, db = backproject(pixel_b, cameras[1])
    angle = np.degrees(np.arccos(np.clip(abs(da.dot(db)), 0, 1)))
    if angle < min_angle_deg:
        raise ValueError('Near-parallel rays: triangulation is ill-conditioned')
    distances = np.linalg.lstsq(np.stack([da, -db], axis=1), sb-sa, rcond=None)[0]
    if np.any(distances <= 0):
        raise ValueError('Intersection lies behind a source')
    a, b = sa+distances[0]*da, sb+distances[1]*db
    xyz = (a+b)/2
    reprojection = []
    for pixel, camera in zip([pixel_a, pixel_b], cameras):
        uv, valid = project(xyz, camera)
        if not valid[0]:
            raise ValueError('Invalid reconstructed depth')
        reprojection.append(float(np.linalg.norm(uv[0]-pixel)))
    return dict(xyz_lps_mm=xyz.tolist(), ray_gap_mm=float(np.linalg.norm(a-b)),
                ray_angle_deg=float(angle), reprojection_error_px=reprojection)
