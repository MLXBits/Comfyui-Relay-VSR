"""RelayVSR 模型装配与流式推理。

一次 restore 调用会：
1. 按 Wan 参考帧建立 Flash 的 start/end condition；
2. 把视频切成 chunk_len 帧的窗口；
3. 逐帧用 Flash 解码并输出裁剪后的像素。

Wan 主干只处理每个窗口的首/尾帧（参考帧），因此长视频的主要开销在 Flash。
"""
# Portions of this file are derived from RelayVSR (https://github.com/kopperx/RelayVSR)
# and LightX2V (https://github.com/ModelTC/LightX2V), both under the Apache License 2.0.
# Modifications: adapted for this ComfyUI plugin.

from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from .attention import clear_layout_cache
from .checkpoints import checkpoint_config, validate_configs
from .flash import FlashKV, FlashPre, FlashTransformer
from .geometry import RelayVSRGeometry
from .rope import RotaryEmbedding
from .wan import WanKV, WanPre, WanTransformer
from .weights import (
    Checkpoint,
    fp32_layer_norm,
    load_flash_block,
    load_flash_post,
    load_flash_pre,
    load_wan_block,
    load_wan_post,
    load_wan_pre,
    rms_norm,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class RunState:
    def __init__(self, generator, wan_kv, flash_kv):
        self.generator = generator
        self.wan_kv = wan_kv
        self.flash_kv = flash_kv
        self.reference_count = 0

    def close(self):
        self.wan_kv = None
        self.flash_kv = None
        self.generator = None


class RelayVSREngine:
    def __init__(self, bundle, dtype="bf16", offload=False, device=None):
        self.bundle = bundle
        self.geometry = RelayVSRGeometry(bundle.scale)
        self.dtype = DTYPES.get(dtype)
        if self.dtype is None:
            raise ValueError(f"不支持的 dtype: {dtype}")
        self.device = device or torch.device("cuda")
        self.offload = bool(offload)
        self.loaded = False

    # ---------------------------------------------------------------- 装配
    def load(self):
        wan_config = checkpoint_config(self.bundle.wan_path, "wan", self.geometry.scale)
        flash_config = checkpoint_config(self.bundle.flash_path, "flash", self.geometry.scale)
        validate_configs(wan_config, flash_config, self.geometry.scale)

        with safe_open(str(self.bundle.wan_path), framework="pt", device="cpu") as handle:
            wan = Checkpoint(handle)
            pre_weights = load_wan_pre(wan, self.device, self.dtype)
            self.wan_pre = WanPre(pre_weights, wan_config, self.geometry, self.dtype)
            self.temb, self.modulation, self.context = self.wan_pre.conditioning()
            self.latent_channels = int(pre_weights["patch_w"].shape[1])
            eps = self.wan_pre.eps

            num_layers = int(wan_config.get("num_layers", 30))
            blocks, cross_kv = [], []
            for index in range(num_layers):
                block = load_wan_block(wan, index, self.device, self.dtype)
                key = rms_norm(
                    F.linear(self.context, block["ck_w"], block["ck_b"]), block["ckn"], eps
                ).view(-1, self.wan_pre.num_heads, self.wan_pre.head_dim)
                value = F.linear(self.context, block["cv_w"], block["cv_b"]).view(
                    -1, self.wan_pre.num_heads, self.wan_pre.head_dim
                )
                cross_kv.append((key, value))
                if self.offload:
                    block = {name: tensor.to("cpu") for name, tensor in block.items()}
                blocks.append(block)
            post_weights = load_wan_post(wan, self.device, self.dtype)

        with safe_open(str(self.bundle.flash_path), framework="pt", device="cpu") as handle:
            flash = Checkpoint(handle)
            flash_pre_weights = load_flash_pre(flash, self.device, self.dtype)
            depth = int(flash_config.get("depth", 10))
            flash_blocks = [
                load_flash_block(flash, index, self.device, self.dtype) for index in range(depth)
            ]
            flash_post_weights = load_flash_post(flash, self.device, self.dtype)

        self.wan_rope = RotaryEmbedding(
            wan=True,
            device=self.device,
            dtype=self.dtype,
            head_dim=self.wan_pre.head_dim,
            max_length=int(wan_config.get("rope_max_seq_len", 1024)),
        )
        self.flash_rope = RotaryEmbedding(
            wan=False,
            device=self.device,
            dtype=self.dtype,
            theta=float(flash_config.get("rope_theta", 10000.0)),
        )
        self.wan = WanTransformer(
            blocks,
            post_weights,
            cross_kv,
            wan_config,
            self.modulation,
            self.geometry,
            self.wan_rope,
            self.dtype,
            self.device,
            self.offload,
        )
        self.flash_pre = FlashPre(flash_pre_weights, flash_config, self.geometry)
        self.flash = FlashTransformer(
            flash_blocks, flash_post_weights, flash_config, self.geometry, self.flash_rope, self.dtype, self.device
        )
        self.flash_depth = depth
        self.flash_kv_heads = self.flash.num_kv_heads
        self.flash_head_dim = self.flash.head_dim
        self.temporal_window = int(wan_config.get("temporal_context_frames", 4))
        self.loaded = True

    def free(self):
        for name in (
            "wan_pre", "wan", "wan_rope", "flash_pre", "flash", "flash_rope", "temb", "modulation", "context",
        ):
            self.__dict__.pop(name, None)
        self.loaded = False
        clear_layout_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 运行
    def _preprocess(self, frame):
        height, width = frame.shape[-2:]
        x = frame[None].to(self.device).to(self.dtype)
        x = x.div(127.5).sub(1.0)
        padded_height, padded_width = self.geometry.padded_size(height, width)
        return F.pad(x, (0, padded_width - width, 0, padded_height - height), mode="replicate")

    def _init_run(self, padded_height, padded_width, seed):
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        wan_h, wan_w = self.geometry.wan_grid(padded_height, padded_width)
        wan_kv = WanKV(
            len(self.wan.blocks),
            wan_h * wan_w,
            self.wan_pre.num_heads,
            self.wan_pre.head_dim,
            self.temporal_window,
            self.dtype,
            self.device,
        )
        latent_h, latent_w = self.geometry.latent_grid(padded_height, padded_width)
        flash_kv = FlashKV(
            self.flash_depth,
            latent_h * latent_w,
            self.flash_kv_heads,
            self.flash_head_dim,
            self.dtype,
            self.device,
        )
        return RunState(generator, wan_kv, flash_kv)

    def _reference(self, frame, state):
        lr = self._preprocess(frame)
        latent_h, latent_w = self.geometry.latent_grid(*lr.shape[-2:])
        latents = torch.randn(
            (1, self.latent_channels, 1, latent_h, latent_w),
            generator=state.generator,
            device=self.device,
            dtype=torch.float32,
        )
        height, width = self.geometry.wan_grid(*lr.shape[-2:])
        hidden = self.wan_pre.prepare(latents.to(self.dtype), lr)
        q_rope, k_rope, visible = self.wan.prepare_rope(state.reference_count, height, width)
        hidden = self.wan.forward(
            hidden, self.temb, q_rope, k_rope, state.wan_kv, state.reference_count, height, width, visible
        )
        noise = self.wan.unpatchify(hidden, height, width)
        latents = latents - noise.float()
        state.reference_count += 1
        latent = latents[:, :, 0].to(self.dtype)
        condition = self.flash_pre.condition(latent)
        return self.flash.prepare_condition(condition, latent.shape[-2], latent.shape[-1])

    @staticmethod
    def _postprocess(prediction, target):
        if prediction.shape[-2:] != target:
            prediction = F.interpolate(
                prediction.float().clamp(-1, 1), size=target, mode="bicubic", align_corners=False, antialias=True
            )
        else:
            prediction = prediction.float().clamp(-1, 1)
        pixels = prediction[0].add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
        return pixels.permute(1, 2, 0).contiguous().cpu().numpy()

    def restore(self, frames, seed, sr_ratio=2, chunk_len=16, check=None):
        """frames: 可迭代的 uint8 (3,H,W) CPU 张量；产出 HWC uint8 numpy。"""
        chunk_len = max(2, int(chunk_len))
        check = check or (lambda: None)
        iterator = iter(frames)
        first = next(iterator, None)
        if first is None:
            raise ValueError("RelayVSR 输入为空")
        height, width = first.shape[-2:]
        padded_height, padded_width = self.geometry.padded_size(height, width)
        target = self.geometry.output_size(sr_ratio, height, width)
        state = self._init_run(padded_height, padded_width, seed)
        scale = self.geometry.scale
        try:
            start_condition = self._reference(first, state)
            first_window = True
            start = first
            while True:
                window = [start, *islice(iterator, chunk_len - 1)]
                if any(frame.shape != first.shape for frame in window):
                    raise ValueError("RelayVSR 要求所有输入帧尺寸一致")
                if len(window) == 1 and not first_window:
                    return
                check()
                end_condition = self._reference(window[-1], state) if len(window) > 1 else start_condition
                conditions = [
                    (torch.cat((a[0], b[0]), 0), torch.cat((a[1], b[1]), 0))
                    for a, b in zip(start_condition, end_condition)
                ]
                for offset, frame in enumerate(window):
                    check()
                    lr = self._preprocess(frame)
                    hidden = self.flash_pre.frame(lr)
                    hidden = self.flash.infer(
                        hidden, conditions, state.flash_kv, offset, lr.shape[-2] // self.geometry.latent_stride,
                        lr.shape[-1] // self.geometry.latent_stride,
                    )
                    prediction = self.flash.decode(hidden, lr)
                    prediction = prediction[..., : height * scale, : width * scale]
                    if first_window or offset > 0:
                        yield self._postprocess(prediction, target)
                first_window = False
                start, start_condition = window[-1], end_condition
                if len(window) < chunk_len:
                    return
        finally:
            state.close()
            clear_layout_cache()
            self.wan_rope.clear()
            self.flash_rope.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()