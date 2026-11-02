"""Flash 解码器（逐帧输出）的推理核心。

每个窗口内的帧都复用同一组 start/end condition KV；帧间通过一个
“上一帧”静态 K/V 缓存做时间一致性，SDPA 使用居中矩形窗口。
"""
# Portions of this file are derived from RelayVSR (https://github.com/kopperx/RelayVSR)
# and LightX2V (https://github.com/ModelTC/LightX2V), both under the Apache License 2.0.
# Modifications: adapted for this ComfyUI plugin.

import torch
import torch.nn.functional as F

from .attention import flash_local_attention
from .rope import apply_rope
from .weights import rms_norm


def _embedding(tensor):
    # 官方权重把类型嵌入存成 (1, dim) 或 (1, 1, dim)。
    return tensor[0].reshape(-1)


class FlashKV:
    def __init__(self, num_layers, tokens, kv_heads, head_dim, dtype, device):
        shape = (num_layers, tokens, kv_heads, head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty_like(self.k)

    def view(self, layer):
        return self.k[layer], self.v[layer]

    def store(self, layer, key, value):
        self.k[layer, : key.shape[0]].copy_(key)
        self.v[layer, : value.shape[0]].copy_(value)


class FlashPre:
    def __init__(self, weights, config, geometry):
        self.w = weights
        self.geometry = geometry
        self.eps = float(config.get("norm_eps", 1e-5))
        self.strides = geometry.encoder_strides

    def frame(self, lr):
        x = lr
        for index, stride in zip((0, 2, 4), self.strides):
            weight, bias = self.w[f"c{index}"]
            x = F.silu(F.conv2d(x, weight, bias, stride=stride, padding=1))
        weight, bias = self.w["c6"]
        x = F.conv2d(x, weight, bias).flatten(2).transpose(1, 2)[0]
        return rms_norm(x, self.w["lrn"], self.eps) + _embedding(self.w["vtype"])

    def condition(self, latent):
        weight, bias = self.w["cond"]
        x = F.conv2d(latent, weight, bias).flatten(2).transpose(1, 2)[0]
        return rms_norm(x, self.w["condn"], self.eps) + _embedding(self.w["ctype"])


class FlashTransformer:
    def __init__(self, blocks, post, config, geometry, rope, dtype, device):
        self.blocks = blocks
        self.post = post
        self.geometry = geometry
        self.rope = rope
        self.dtype = dtype
        self.eps = float(config.get("norm_eps", 1e-5))
        spatial = config.get("spatial_attention") or {}
        self.condition_window = tuple(spatial.get("condition_self_window_size", [48, 48]))
        self.start_window = tuple(spatial.get("start_condition_window_size", [48, 48]))
        self.end_window = tuple(spatial.get("end_condition_window_size", [48, 48]))
        self.video_window = tuple(spatial.get("video_window_size", [24, 24]))
        self.window_size = int(config.get("window_size", 2))

        num_kv_groups = int(config.get("num_kv_groups", 2))
        head_dim = blocks[0]["k"][0].shape[0] // num_kv_groups
        self.head_dim = head_dim
        self.num_heads = blocks[0]["q"][0].shape[0] // head_dim
        self.num_kv_heads = blocks[0]["k"][0].shape[0] // head_dim
        self.head = post["head"]
        self.projection = post["proj"]

    def _block(self, t, x, condition, past_key, past_value, q_rope, k_rope, windows, height, width):
        batch_tokens = x.shape[0]
        norm = rms_norm(x, t["an"], self.eps)
        query = F.linear(norm, t["q"][0], t["q"][1]).view(batch_tokens, self.num_heads, self.head_dim)
        key_op, value_op = (t["ck"], t["cv"]) if condition else (t["k"], t["v"])
        key = F.linear(norm, key_op[0], key_op[1]).view(batch_tokens, self.num_kv_heads, self.head_dim)
        value = F.linear(norm, value_op[0], value_op[1]).view(batch_tokens, self.num_kv_heads, self.head_dim)
        key = rms_norm(key, t["kn"], self.eps)
        value = rms_norm(value, t["vn"], self.eps)

        all_key = key if condition else torch.cat((past_key, key), 0)
        all_value = value if condition else torch.cat((past_value, value), 0)
        query = apply_rope(query, *q_rope)
        all_key = apply_rope(all_key, *k_rope)
        attended = flash_local_attention(query, all_key, all_value, height, width, windows)
        x = x + F.linear(attended, t["o"][0], t["o"][1])

        gate, up = F.linear(rms_norm(x, t["fn"], self.eps), t["gu_w"]).chunk(2, -1)
        x = x + F.linear(F.silu(gate) * up, t["dn_w"])
        return x, key, value

    def prepare_condition(self, x, height, width):
        frequencies = self.rope.for_positions(height, width, (0,))
        windows = (self.condition_window,)
        caches = []
        for t in self.blocks:
            x, key, value = self._block(
                t, x, True, None, None, frequencies, frequencies, windows, height, width
            )
            caches.append((key, value))
        return caches

    def infer(self, x, conditions, history, frame, height, width):
        previous = min(frame, self.window_size - 1)
        q_rope = self.rope.for_positions(height, width, (0,))
        k_rope = self.rope.for_positions(height, width, (0, 0, *range(-previous, 1)))
        windows = (self.start_window, self.end_window) + (self.video_window,) * (previous + 1)
        for index, t in enumerate(self.blocks):
            condition_key, condition_value = conditions[index]
            if previous:
                past_key = torch.cat((condition_key, history.view(index)[0]), 0)
                past_value = torch.cat((condition_value, history.view(index)[1]), 0)
            else:
                past_key, past_value = condition_key, condition_value
            x, key, value = self._block(
                t, x, False, past_key, past_value, q_rope, k_rope, windows, height, width
            )
            history.store(index, key, value)
        return x

    def decode(self, hidden, lr):
        height, width = self.geometry.latent_grid(*lr.shape[-2:])
        pixels = F.linear(
            F.silu(F.linear(hidden, self.head[0], self.head[1])),
            self.projection[0],
            self.projection[1],
        )
        channels = pixels.shape[1]
        residual = F.pixel_shuffle(pixels.t().reshape(1, channels, height, width), 16)
        return F.interpolate(lr, scale_factor=self.geometry.scale, mode="bicubic", align_corners=False) + residual
