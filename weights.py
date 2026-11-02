"""RelayVSR 权重读取与低层算子。

为了不引入 LightX2V 依赖，这里只使用 safetensors + torch。
权重键名与官方 checkpoint 完全一致；Wan 与 Flash 的权重分别打包成
普通 dict，便于在显存不足时做整块 CPU offload。

Wan 线性层可以存成 ComfyUI 原生 int8 convrot 格式：带 ``<module>.comfy_quant``
的模块会被还原成 ``QuantizedTensor``，``F.linear`` 直接走 int8 内核；
没有该标记的模块仍按普通浮点权重读取。
"""
# Portions of this file are derived from RelayVSR (https://github.com/kopperx/RelayVSR)
# and LightX2V (https://github.com/ModelTC/LightX2V), both under the Apache License 2.0.
# Modifications: adapted for this ComfyUI plugin.

import json

import torch
import torch.nn.functional as F

from comfy.quant_ops import QuantizedTensor, get_layout_class

INT8_LAYOUT = "TensorWiseINT8Layout"


def fp32_layer_norm(x, eps, weight=None, bias=None, cast=False):
    """LayerNorm 在 float32 下计算；cast=True 时输出转回输入 dtype。"""
    output = F.layer_norm(
        x.float(),
        (x.shape[-1],),
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        eps,
    )
    return output.to(x.dtype) if cast else output


def channel_rms_norm(x, weight, eps=1e-6):
    """按通道（dim=1）归一化，用于 lq_proj 的 4D 特征。"""
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=1, keepdim=True) + eps)
    return (value * weight).to(x.dtype)


def rms_norm(x, weight, eps):
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


class Checkpoint:
    """safetensors 句柄的薄包装，缓存 key 集合以识别量化模块。"""

    def __init__(self, handle):
        self.handle = handle
        self.keys = frozenset(handle.keys())

    def tensor(self, key):
        return self.handle.get_tensor(key)

    def quantized(self, key):
        return key[: -len(".weight")] + ".comfy_quant" in self.keys


def _get(handle, key, device, dtype):
    if key.endswith(".weight") and handle.quantized(key):
        return _quantized_weight(handle, key, device, dtype)
    return handle.tensor(key).to(device=device, dtype=dtype)


def _quantized_weight(handle, key, device, dtype):
    module = key[: -len(".weight")]
    conf = json.loads(handle.tensor(module + ".comfy_quant").numpy().tobytes())
    if conf.get("format") != "int8_tensorwise":
        raise ValueError(f"不支持的量化格式 {conf.get('format')!r}: {module}")
    qdata = handle.tensor(key).to(device=device)
    scale = handle.tensor(module + ".weight_scale").to(device=device, dtype=torch.float32)
    params = get_layout_class(INT8_LAYOUT).Params(
        scale=scale,
        orig_dtype=dtype,
        orig_shape=tuple(qdata.shape),
        is_weight=True,
        convrot=bool(conf.get("convrot", False)),
        convrot_groupsize=int(conf.get("convrot_groupsize", 256)),
    )
    return QuantizedTensor(qdata, INT8_LAYOUT, params)


def _linear(handle, key, device, dtype, bias=True):
    weight = _get(handle, key + ".weight", device, dtype)
    if not bias:
        return weight, None
    bias_tensor = _get(handle, key + ".bias", device, dtype)
    return weight, bias_tensor


def load_wan_pre(handle, device, dtype):
    return {
        "patch_w": _get(handle, "patch_embedding.weight", device, dtype),
        "patch_b": _get(handle, "patch_embedding.bias", device, dtype),
        "lq1_w": _get(handle, "lq_proj.conv1.weight", device, dtype),
        "lq1_b": _get(handle, "lq_proj.conv1.bias", device, dtype),
        "lq2_w": _get(handle, "lq_proj.conv2.weight", device, dtype),
        "lq2_b": _get(handle, "lq_proj.conv2.bias", device, dtype),
        "lq_n1": _get(handle, "lq_proj.norm1.weight", device, dtype),
        "lq_n2": _get(handle, "lq_proj.norm2.weight", device, dtype),
        "lqo_w": _get(handle, "lq_proj.out_proj.weight", device, dtype),
        "lqo_b": _get(handle, "lq_proj.out_proj.bias", device, dtype),
        "text1_w": _get(handle, "condition_embedder.text_embedder.linear_1.weight", device, dtype),
        "text1_b": _get(handle, "condition_embedder.text_embedder.linear_1.bias", device, dtype),
        "text2_w": _get(handle, "condition_embedder.text_embedder.linear_2.weight", device, dtype),
        "text2_b": _get(handle, "condition_embedder.text_embedder.linear_2.bias", device, dtype),
        "time1_w": _get(handle, "condition_embedder.time_embedder.linear_1.weight", device, dtype),
        "time1_b": _get(handle, "condition_embedder.time_embedder.linear_1.bias", device, dtype),
        "time2_w": _get(handle, "condition_embedder.time_embedder.linear_2.weight", device, dtype),
        "time2_b": _get(handle, "condition_embedder.time_embedder.linear_2.bias", device, dtype),
        "timep_w": _get(handle, "condition_embedder.time_proj.weight", device, dtype),
        "timep_b": _get(handle, "condition_embedder.time_proj.bias", device, dtype),
        "prompt": _get(handle, "prompt_embeds", device, dtype),
    }


def _table(handle, name, device, dtype):
    # 官方权重把逐块调制表存成 (1, 6, dim)，这里统一压成 (6, dim)。
    tensor = _get(handle, name, device, dtype)
    return tensor[0] if tensor.dim() == 3 and tensor.shape[0] == 1 else tensor


def load_wan_block(handle, index, device, dtype):
    prefix = f"blocks.{index}."
    return {
        "sq_w": _get(handle, prefix + "attn1.to_q.weight", device, dtype),
        "sq_b": _get(handle, prefix + "attn1.to_q.bias", device, dtype),
        "sk_w": _get(handle, prefix + "attn1.to_k.weight", device, dtype),
        "sk_b": _get(handle, prefix + "attn1.to_k.bias", device, dtype),
        "sv_w": _get(handle, prefix + "attn1.to_v.weight", device, dtype),
        "sv_b": _get(handle, prefix + "attn1.to_v.bias", device, dtype),
        "so_w": _get(handle, prefix + "attn1.to_out.0.weight", device, dtype),
        "so_b": _get(handle, prefix + "attn1.to_out.0.bias", device, dtype),
        "sqn": _get(handle, prefix + "attn1.norm_q.weight", device, dtype),
        "skn": _get(handle, prefix + "attn1.norm_k.weight", device, dtype),
        "cq_w": _get(handle, prefix + "attn2.to_q.weight", device, dtype),
        "cq_b": _get(handle, prefix + "attn2.to_q.bias", device, dtype),
        "ck_w": _get(handle, prefix + "attn2.to_k.weight", device, dtype),
        "ck_b": _get(handle, prefix + "attn2.to_k.bias", device, dtype),
        "cv_w": _get(handle, prefix + "attn2.to_v.weight", device, dtype),
        "cv_b": _get(handle, prefix + "attn2.to_v.bias", device, dtype),
        "co_w": _get(handle, prefix + "attn2.to_out.0.weight", device, dtype),
        "co_b": _get(handle, prefix + "attn2.to_out.0.bias", device, dtype),
        "cqn": _get(handle, prefix + "attn2.norm_q.weight", device, dtype),
        "ckn": _get(handle, prefix + "attn2.norm_k.weight", device, dtype),
        "norm2_w": _get(handle, prefix + "norm2.weight", device, dtype),
        "norm2_b": _get(handle, prefix + "norm2.bias", device, dtype),
        "sst": _table(handle, prefix + "scale_shift_table", device, dtype),
        "f0_w": _get(handle, prefix + "ffn.net.0.proj.weight", device, dtype),
        "f0_b": _get(handle, prefix + "ffn.net.0.proj.bias", device, dtype),
        "f2_w": _get(handle, prefix + "ffn.net.2.weight", device, dtype),
        "f2_b": _get(handle, prefix + "ffn.net.2.bias", device, dtype),
    }


def load_wan_post(handle, device, dtype):
    return {
        "sst": _get(handle, "scale_shift_table", device, dtype),
        "proj_w": _get(handle, "proj_out.weight", device, dtype),
        "proj_b": _get(handle, "proj_out.bias", device, dtype),
    }


def load_flash_pre(handle, device, dtype):
    return {
        "c0": _linear(handle, "lr_encoder.layers.0", device, dtype),
        "c2": _linear(handle, "lr_encoder.layers.2", device, dtype),
        "c4": _linear(handle, "lr_encoder.layers.4", device, dtype),
        "c6": _linear(handle, "lr_encoder.layers.6", device, dtype),
        "cond": _linear(handle, "condition_encoder.projection", device, dtype),
        "lrn": _get(handle, "lr_encoder.norm.weight", device, dtype),
        "condn": _get(handle, "condition_encoder.norm.weight", device, dtype),
        "vtype": _get(handle, "video_type_embedding", device, dtype),
        "ctype": _get(handle, "condition_type_embedding", device, dtype),
    }


def load_flash_block(handle, index, device, dtype):
    prefix = f"backbone.blocks.{index}.attention."
    ff = f"backbone.blocks.{index}.feed_forward."
    return {
        "q": _linear(handle, prefix + "query", device, dtype),
        "k": _linear(handle, prefix + "key", device, dtype),
        "v": _linear(handle, prefix + "value", device, dtype),
        "ck": _linear(handle, prefix + "condition_key", device, dtype),
        "cv": _linear(handle, prefix + "condition_value", device, dtype),
        "o": _linear(handle, prefix + "output", device, dtype),
        "kn": _get(handle, prefix + "key_norm.weight", device, dtype),
        "vn": _get(handle, prefix + "value_norm.weight", device, dtype),
        "an": _get(handle, f"backbone.blocks.{index}.attention_norm.weight", device, dtype),
        "fn": _get(handle, f"backbone.blocks.{index}.feed_forward_norm.weight", device, dtype),
        "gu_w": _get(handle, ff + "gate_up_proj.weight", device, dtype),
        "dn_w": _get(handle, ff + "down.weight", device, dtype),
    }


def load_flash_post(handle, device, dtype):
    return {
        "head": _linear(handle, "spatial_head.0", device, dtype),
        "proj": _linear(handle, "residual_projection", device, dtype),
    }
