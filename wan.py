"""Wan 主干（RelayVSR 参考实现）的推理核心。

与官方实现的对应关系：
- pre：latent patch 嵌入 + LQ 像素投影（pixel_unshuffle / lq_proj）+ 条件嵌入
- blocks：自注意力（环形帧缓存 + 贴边矩形窗口）、固定 prompt 的 cross-attn、FFN
- post：FP32 LayerNorm + modulation + proj_out + unpatchify

注意：官方在 fp32 中计算 LayerNorm / modulation，再把残差写回 bf16，
这里严格保持这些精度边界。
"""
# Portions of this file are derived from RelayVSR (https://github.com/kopperx/RelayVSR)
# and LightX2V (https://github.com/ModelTC/LightX2V), both under the Apache License 2.0.
# Modifications: adapted for this ComfyUI plugin.

import math

import torch
import torch.nn.functional as F

from .attention import full_attention, wan_local_attention
from .rope import apply_rope
from .weights import fp32_layer_norm, rms_norm, channel_rms_norm


class WanKV:
    """按帧槽位原地覆盖的自注意力 K/V 缓存。"""

    def __init__(self, num_layers, tokens_per_frame, heads, head_dim, frames, dtype, device):
        shape = (num_layers, tokens_per_frame * frames, heads, head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty(shape, dtype=dtype, device=device)
        self.tokens = tokens_per_frame
        self.frames = frames

    def write(self, layer, slot, key, value):
        start = slot * self.tokens
        self.k[layer, start : start + self.tokens].copy_(key)
        self.v[layer, start : start + self.tokens].copy_(value)

    def visible(self, layer, key_frames):
        end = key_frames * self.tokens
        return self.k[layer, :end], self.v[layer, :end]


class WanPre:
    def __init__(self, weights, config, geometry, dtype):
        self.w = weights
        self.geometry = geometry
        self.dtype = dtype
        self.eps = float(config.get("eps", 1e-6))
        self.freq_dim = weights["time1_w"].shape[1]
        self.dim = weights["time1_w"].shape[0]
        self.num_heads = int(config.get("num_attention_heads", config.get("num_heads", 24)))
        self.head_dim = int(config.get("attention_head_dim", self.dim // self.num_heads))
        self.patch_stride = tuple(config.get("patch_size", [1, 2, 2]))

    def conditioning(self):
        w = self.w
        dim = self.freq_dim
        exponent = (
            -math.log(10000)
            * torch.arange(dim // 2, device=w["prompt"].device, dtype=torch.float32)
            / (dim // 2)
        )
        angles = 1000 * torch.exp(exponent)
        timestep = torch.cat((angles.cos(), angles.sin())).unsqueeze(0).to(w["prompt"].dtype)
        temb = F.linear(
            F.silu(F.linear(timestep, w["time1_w"], w["time1_b"])), w["time2_w"], w["time2_b"]
        )
        modulation = F.linear(F.silu(temb), w["timep_w"], w["timep_b"]).view(6, -1)
        prompt = w["prompt"]
        if prompt.dim() == 3:
            prompt = prompt[0]
        context = F.linear(
            F.gelu(F.linear(prompt, w["text1_w"], w["text1_b"]), approximate="tanh"),
            w["text2_w"],
            w["text2_b"],
        )
        return temb, modulation, context

    def prepare_lr(self, lr):
        if self.geometry.scale == 2:
            return F.pixel_unshuffle(lr, 8)
        return F.pixel_unshuffle(F.interpolate(lr, scale_factor=4, mode="bilinear", align_corners=False), 16)

    def prepare(self, latents, lr):
        w = self.w
        hidden = F.conv3d(
            latents, w["patch_w"], w["patch_b"], stride=self.patch_stride
        ).flatten(2).transpose(1, 2)[0]
        lq = self.prepare_lr(lr)
        lq = F.silu(channel_rms_norm(F.conv2d(lq, w["lq1_w"], w["lq1_b"]), w["lq_n1"]))
        lq = F.silu(channel_rms_norm(F.conv2d(lq, w["lq2_w"], w["lq2_b"], stride=2, padding=1), w["lq_n2"]))
        lq = lq.flatten(2).transpose(1, 2).flatten(0, 1)
        lq = F.linear(lq, w["lqo_w"], w["lqo_b"])
        return hidden + lq


class WanTransformer:
    def __init__(self, blocks, post, cross_kv, config, modulation, geometry, rope, dtype, device, offload):
        self.blocks = blocks
        self.post = post
        self.cross_kv = cross_kv
        self.geometry = geometry
        self.rope = rope
        self.dtype = dtype
        self.device = device
        self.eps = float(config.get("eps", 1e-6))
        self.num_heads = int(config.get("num_attention_heads", config.get("num_heads", 24)))
        self.head_dim = int(config.get("attention_head_dim", self.blocks[0]["sq_w"].shape[0] // self.num_heads))
        self.window = tuple(config.get("spatial_window_size", [22, 40]))
        self.temporal_window = int(config.get("temporal_context_frames", 4))
        temporal_rope = config.get("temporal_rope") or {}
        if temporal_rope.get("policy", "fixed_end") != "fixed_end":
            raise ValueError("RelayVSR 要求 Wan 使用 fixed_end 时间 RoPE")
        self.training_frames = int(temporal_rope.get("training_frames", 8))
        # proj_out 输出是 out_channels * prod(patch_size)，这里还原每帧的通道数。
        self.out_channels = int(config.get("out_channels") or self.post["proj_w"].shape[0] // 4)

        # 每个 block 的 6 组 modulation 只依赖固定 timestep，预先算好放在显存里。
        modulation = modulation.float()
        self.mod_vectors = [
            tuple(block["sst"][i].to(device) + modulation[i] for i in range(6))
            for block in blocks
        ]

        self.staging = None
        if offload:
            self.staging = {key: torch.empty_like(value, device=device) for key, value in blocks[0].items()}

    def _tensors(self, index):
        if self.staging is None:
            return self.blocks[index]
        for key, value in self.blocks[index].items():
            self.staging[key].copy_(value)
        return self.staging

    def prepare_rope(self, frame, height, width):
        visible = min(frame + 1, self.temporal_window)
        origin = max(0, frame - self.training_frames + 1)
        positions = tuple(
            frame - ((frame - slot) % self.temporal_window) - origin for slot in range(visible)
        )
        return (
            self.rope.for_positions(height, width, (frame - origin,)),
            self.rope.for_positions(height, width, positions),
            visible,
        )

    def _block(self, index, x, q_rope, k_rope, kv, frame, height, width, visible):
        t = self._tensors(index)
        shift, scale, gate, c_shift, c_scale, c_gate = self.mod_vectors[index]

        norm1 = fp32_layer_norm(x, self.eps)
        norm1 = (norm1 * (1 + scale) + shift).to(self.dtype)
        q = rms_norm(F.linear(norm1, t["sq_w"], t["sq_b"]), t["sqn"], self.eps)
        k = rms_norm(F.linear(norm1, t["sk_w"], t["sk_b"]), t["skn"], self.eps)
        v = F.linear(norm1, t["sv_w"], t["sv_b"])
        tokens = q.shape[0]
        q = q.view(tokens, self.num_heads, self.head_dim)
        k = k.view(tokens, self.num_heads, self.head_dim)
        v = v.view(tokens, self.num_heads, self.head_dim)

        kv.write(index, frame % self.temporal_window, k, v)
        key, value = kv.visible(index, visible)
        q = apply_rope(q, *q_rope)
        key = apply_rope(key, *k_rope)

        if height > self.window[0] or width > self.window[1]:
            attended = wan_local_attention(
                q, key, value, height, width, self.window, visible
            )
        else:
            attended = full_attention(q, key, value)
        y = F.linear(attended, t["so_w"], t["so_b"])
        x = (x.float() + y.float() * gate).to(self.dtype)

        # cross-attention：固定 prompt 的 KV 已在加载时算好。
        norm3 = fp32_layer_norm(x, self.eps, t["norm2_w"], t["norm2_b"], cast=True)
        cross_q = rms_norm(F.linear(norm3, t["cq_w"], t["cq_b"]), t["cqn"], self.eps)
        cross_q = cross_q.view(tokens, self.num_heads, self.head_dim)
        cross_k, cross_v = self.cross_kv[index]
        attended = full_attention(cross_q, cross_k, cross_v)
        attn_out = F.linear(attended, t["co_w"], t["co_b"])
        x = x + attn_out

        # FFN 前残差仍是 bf16，LayerNorm 与 modulation 在 fp32 中完成。
        norm2 = fp32_layer_norm(x, self.eps)
        norm2 = (norm2 * (1 + c_scale) + c_shift).to(self.dtype)
        y = F.linear(F.gelu(F.linear(norm2, t["f0_w"], t["f0_b"]), approximate="tanh"), t["f2_w"], t["f2_b"])
        x = (x.float() + y.float() * c_gate).to(self.dtype)
        return x

    def forward(self, x, modulation, q_rope, k_rope, kv, frame, height, width, visible):
        for index in range(len(self.blocks)):
            x = self._block(index, x, q_rope, k_rope, kv, frame, height, width, visible)
        return self.finish(x, modulation)

    def finish(self, x, temb):
        w = self.post
        table = w["sst"]
        if table.dim() == 3:
            table = table[0]
        shift, scale = (table + temb).unbind(0)
        x = (fp32_layer_norm(x, self.eps) * (1 + scale) + shift).to(self.dtype)
        return F.linear(x, w["proj_w"], w["proj_b"])

    def unpatchify(self, x, height, width):
        x = x[: height * width].view(1, height, width, 1, 2, 2, self.out_channels)
        x = torch.einsum("fhwpqrc->cfphqwr", x)
        return x.reshape(self.out_channels, 1, height * 2, width * 2).unsqueeze(0)
