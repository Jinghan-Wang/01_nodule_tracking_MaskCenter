"""固定搜索ROI工具：只使用计划中心、投影矩阵和计划模板，不使用当前状态真值。"""
from itertools import product

import numpy as np
from geometry import project


def projected_center_bounds(plan, camera, radius_mm):
    """三维搜索盒在完整DRR中的二维包围范围：[最小uv,最大uv]。"""
    corners = np.asarray(plan)+np.asarray(list(product([-radius_mm, radius_mm], repeat=3)))
    uv, valid = project(corners, camera)
    if not valid.all():
        raise ValueError('Search box projection is invalid')
    return np.asarray([uv.min(0), uv.max(0)])


def build_search_box(plan, camera, template_hw, template_center_local, radius_mm=10.,
                     template_origin_uv=None):
    """返回完整DRR坐标[x0,y0,x1,y1]；右/下为切片开区间。

    三维盒投影后，按模板参考点到四边的距离外扩，再留8像素余量。
    默认左上角对齐全图4像素网格，保持已有实验方式。
    提供template_origin_uv时，ROI同尺寸平移至模板网格，每轴最多2像素。
    不截断边界：必要时补零，避免边缘候选失去模板支持。
    """
    if not np.isfinite(radius_mm) or not 0 < radius_mm <= 18:
        raise ValueError('Search radius must be within (0,18] mm')
    bounds = projected_center_bounds(plan, camera, radius_mm)
    local = np.asarray(template_center_local)
    wh = np.asarray(template_hw[::-1])
    start = np.floor((bounds[0]-local-8)/4).astype(int)*4
    stop = np.ceil((bounds[1]-local+wh+8)/4).astype(int)*4
    if template_origin_uv is not None:
        # 起点与模板裁剪原点模4相同，使计划参考点落在响应格点上。
        # 同时平移两条边，保持ROI宽高；原有8像素余量仍至少剩6像素。
        shift = (np.asarray(template_origin_uv, dtype=int)+2) % 4-2
        start += shift
        stop += shift
    return tuple(int(x) for x in np.r_[start, stop])


def crop_search(image, box):
    """保持原始像素分辨率；超出原图部分填0，输入应已归一化。"""
    x0, y0, x1, y1 = box
    h, w = image.shape
    result = np.zeros((y1-y0, x1-x0), dtype=image.dtype)
    left, top, right, bottom = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if left < right and top < bottom:
        result[top-y0:bottom-y0, left-x0:right-x0] = image[top:bottom, left:right]
    return result
