"""RelayVSR 的 3D RoPE（Wan）与逐帧相对 RoPE（Flash）。

与官方实现保持一致：
- Wan：时间/高/宽三轴，维度按 head_dim 切分，cos/sin 以 float32 存储，
  因此 apply_rope 的乘加在 float32 里完成，再写回推理 dtype。
- Flash：三轴维度固定为 (16, 24, 24)，cos/sin 直接使用推理 dtype。
"""
# Portions of this file are derived from RelayVSR (https://github.com/kopperx/RelayVSR)
# and LightX2V (https://github.com/ModelTC/LightX2V), both under the Apache License 2.0.
# Modifications: adapted for this ComfyUI plugin.

import torch


def apply_rope(x, cosine, sine):
    """x: (tokens, heads, dim)，cosine/sine: (tokens, dim)。"""
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cosine[:, None, 0::2] - odd * sine[:, None, 1::2]
    out[..., 1::2] = even * sine[:, None, 1::2] + odd * cosine[:, None, 0::2]
    return out


class RotaryEmbedding:
    def __init__(self, *, wan, device, dtype, head_dim=128, max_length=1024, theta=10000.0):
        self.wan = wan
        self.device, self.dtype = device, dtype
        self.max_length = max_length
        if wan:
            # 与官方一致：时间轴占 head_dim - 4*(head_dim//6)，其余两轴各 2*(head_dim//6)。
            self.dims = (
                head_dim - 4 * (head_dim // 6),
                2 * (head_dim // 6),
                2 * (head_dim // 6),
            )
        else:
            self.dims = (16, 24, 24)
        self.cache = {}
        self.tables = []
        for dim in self.dims:
            if wan:
                # 官方在 CPU 上用 float64 建表，再转 float32 放到设备上。
                inverse = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
                angles = torch.outer(torch.arange(max_length, dtype=torch.float64), inverse)
                self.tables.append(
                    (
                        angles.cos().repeat_interleave(2, -1).float().to(device),
                        angles.sin().repeat_interleave(2, -1).float().to(device),
                    )
                )
            else:
                self.tables.append(
                    1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
                )

    def for_positions(self, height, width, positions):
        key = (height, width, positions)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self.wan and (
            max(height, width) > self.max_length or min(positions) < 0 or max(positions) >= self.max_length
        ):
            raise ValueError("RelayVSR 的空间或时间位置超出了权重的 RoPE 范围")
        frames = len(positions)
        coordinates = (
            torch.tensor(positions, device=self.device).view(frames, 1, 1).expand(frames, height, width).reshape(-1),
            torch.arange(height, device=self.device).view(1, height, 1).expand(frames, height, width).reshape(-1),
            torch.arange(width, device=self.device).view(1, 1, width).expand(frames, height, width).reshape(-1),
        )
        cosines, sines = [], []
        for coordinate, table in zip(coordinates, self.tables):
            if self.wan:
                cosine, sine = table[0][coordinate], table[1][coordinate]
            else:
                angles = coordinate.float().unsqueeze(-1) * table
                cosine = angles.cos().repeat_interleave(2, -1)
                sine = angles.sin().repeat_interleave(2, -1)
            cosines.append(cosine)
            sines.append(sine)
        result = (torch.cat(cosines, -1), torch.cat(sines, -1))
        if not self.wan:
            result = tuple(value.to(self.dtype) for value in result)
        if len(self.cache) >= 32:
            # 只保留最近的位置表，避免长视频时无限增长。
            self.cache.clear()
        self.cache[key] = result
        return result

    def clear(self):
        self.cache.clear()
