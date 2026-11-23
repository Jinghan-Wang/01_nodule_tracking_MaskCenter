"""SiamFC-inspired shared encoder + dense cross-correlation for grayscale DRR.

Reference: local siamfc-pytorch-master/siamfc/{heads,backbones,losses}.py.
Compact valid-convolution encoder and energy-normalized correlation are adaptations,
not an exact reproduction of AlexNet SiamFC. No pretrained weights.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

# 把网络输出的原始 logits 转成 0～1 之间的响应分数，供后面的响应图和三维联合评分使用。
def logits_to_scores(logits):
    """Fixed Phase 6 export transform; retains all peaks, not calibrated probability."""
    return torch.sigmoid(logits).clamp(1e-6, 1-1e-6)


class SiamFC(nn.Module):
    # kv1/kv2 调用同一个实例，共用参数；每次仍是对应视角的 template + search。
    # 网络结构、损失和坐标解码不因增加第二个视角而改变。
    stride = 4

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 16, 3))
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, template, search):
        z, x = self.encoder(template), self.encoder(search)
        batch, channels, height, width = x.shape
        # One convolution kernel per sample; no mixing between batch elements.
        dot = F.conv2d(x.reshape(1, batch*channels, height, width), z, groups=batch)
        dot = dot.reshape(batch, 1, dot.shape[-2], dot.shape[-1])
        energy_x = F.conv2d(x.square().sum(1, keepdim=True),
                            x.new_ones(1, 1, z.shape[-2], z.shape[-1]))
        energy_z = z.square().sum((1, 2, 3)).view(batch, 1, 1, 1)
        correlation = dot / (energy_x * energy_z).clamp_min(1e-12).sqrt()
        return self.log_scale.exp().clamp(max=100) * correlation + self.bias

# 根据真实中心制作圆形0/1标签或连续高斯标签，监督响应图中心与周围的分数。
# 真实二维中心 → 响应图坐标 → 目标分数图 → 与网络logits计算损失。
# 在这种相关匹配里，响应图上的位置表示“模板应该放到搜索图的哪个位置”，也就是相关响应图应该出现高响应的位置
def response_loss(logits, center_uv, local_center_uv, radius=4.0,
                  center_label='disk', gaussian_sigma=8./3.):
    """同一坐标网格及平衡权重下，比较圆形硬标签与高斯软标签。"""
    # H = 高度 = v方向   W = 宽度 = u方向
    h, w = logits.shape[-2:]
    u = torch.arange(w, device=logits.device).float()[None, None, :] * SiamFC.stride
    v = torch.arange(h, device=logits.device).float()[None, :, None] * SiamFC.stride
    offset = center_uv-local_center_uv
    if ((offset[:, 0] < 0) | (offset[:, 0] > (w-1)*SiamFC.stride) |
        (offset[:, 1] < 0) | (offset[:, 1] > (h-1)*SiamFC.stride)).any():
        raise ValueError('Target outside valid correlation support')
    # 计算 response map 每个位置距离真实位置多远。
    distance = (u-offset[:, 0, None, None]).square() + (v-offset[:, 1, None, None]).square()
    positive = (distance <= radius**2).unsqueeze(1).float()
    # 高斯以连续真值为中心，sigma单位为完整DRR像素；不将最近格点强制归一化为1。
    target = positive if center_label == 'disk' else torch.exp(-distance/(2*gaussian_sigma**2)).unsqueeze(1)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    axes = (1, 2, 3)
    # 两组都用同一个圆形区域分配权重，避免同时改变标签形状与平衡策略。
    return (.5*(loss*positive).sum(axes)/positive.sum(axes).clamp_min(1) +
            .5*(loss*(1-positive)).sum(axes)/(1-positive).sum(axes).clamp_min(1)).mean()

# 从响应图找到最可能的位置，再转换回 DRR 坐标。
def decode_response(logits, local_center_uv):
    """Valid identical encoders: center = response_index * stride + local center.

    Each valid receptive field has the same encoder offset in both branches,
    so that offset cancels. No resizing, centered-crop assumption or prior window.
    """
    width = logits.shape[-1]
    peak = logits.flatten(1).argmax(1)
    uv = torch.stack((peak % width, torch.div(peak, width, rounding_mode='floor')), dim=1).float()
    return uv*SiamFC.stride + local_center_uv
