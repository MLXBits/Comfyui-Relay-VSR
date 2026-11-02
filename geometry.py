"""RelayVSR 原生超分倍率的空间几何关系。

x2 权重：latent 步长 8，Wan 主干步长 16。
x4 权重：latent 步长 4，Wan 主干步长 8。
"""
# Portions of this file are derived from RelayVSR (https://github.com/kopperx/RelayVSR)
# and LightX2V (https://github.com/ModelTC/LightX2V), both under the Apache License 2.0.
# Modifications: adapted for this ComfyUI plugin.

from dataclasses import dataclass


@dataclass(frozen=True)
class RelayVSRGeometry:
    scale: int

    def __post_init__(self):
        if self.scale not in (2, 4):
            raise ValueError("RelayVSR 原生倍率必须是 2 或 4")

    @property
    def latent_stride(self):
        return 16 // self.scale

    @property
    def wan_stride(self):
        return 2 * self.latent_stride

    @property
    def encoder_strides(self):
        # Flash 模型 LR 编码器的三级下采样步长（最后一级为 1x1 投影）。
        return (2, 2, self.latent_stride // 4)

    def latent_grid(self, height, width):
        return height // self.latent_stride, width // self.latent_stride

    def wan_grid(self, height, width):
        return height // self.wan_stride, width // self.wan_stride

    def padded_size(self, height, width):
        alignment = self.wan_stride
        return height + (-height) % alignment, width + (-width) % alignment

    def output_size(self, sr_ratio, height, width):
        if self.scale == 2:
            if sr_ratio != 2:
                raise ValueError("RelayVSR x2 权重仅支持 2 倍放大")
        elif sr_ratio not in (2, 4):
            raise ValueError("RelayVSR x4 权重的 sr_ratio 必须为 2 或 4")
        target = (int(height * sr_ratio), int(width * sr_ratio))
        if min(target) <= 0:
            raise ValueError("RelayVSR 输出尺寸必须为正")
        return target
