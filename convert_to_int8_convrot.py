"""把 RelayVSR 的 Wan 主干转换成 ComfyUI 原生 int8 convrot 权重。

只量化 ``blocks.*`` 里的线性层（attn1/attn2 的 q/k/v/out、FFN 的 net.0/net.2），
conv3d/conv2d、norm、bias、embedding 与 ``proj_out`` 保持原精度。
每个量化模块写成 ``<name>.weight``（int8）、``<name>.weight_scale``（fp32 行缩放）
和 ``<name>.comfy_quant``（JSON 配置）；safetensors metadata 原样保留，
所以输出目录可以直接当作 RelayVSR bundle 使用。

用法：
    python convert_to_int8_convrot.py <原始目录> [输出目录]
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

GROUP_SIZE = 256
QUANT_MODULES = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)
WAN_NAME = "wan_x2.safetensors"
FLASH_NAME = "flash_x2.safetensors"
OUTPUT_NAME = "wan_x2_int8_convrot.safetensors"


def _quantizable(name):
    if not name.startswith("blocks.") or not name.endswith(".weight"):
        return False
    module = name[: -len(".weight")]
    return any(module.endswith("." + suffix) for suffix in QUANT_MODULES)


def _quantize(tensor, name, device):
    if tensor.shape[-1] % GROUP_SIZE:
        raise ValueError(f"{name} 的输入维度 {tensor.shape[-1]} 不能被 {GROUP_SIZE} 整除")
    qdata, params = TensorWiseINT8Layout.quantize(
        tensor.to(device),
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=GROUP_SIZE,
    )
    conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}
    module = name[: -len(".weight")]
    return {
        name: qdata.to("cpu"),
        module + ".weight_scale": params.scale.to("cpu", dtype=torch.float32),
        module + ".comfy_quant": torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8),
    }


def convert(src, dst, device="cuda"):
    src, dst = Path(src), Path(dst)
    wan_path = src / WAN_NAME
    if not wan_path.is_file():
        raise FileNotFoundError(f"找不到 {wan_path}")
    dst.mkdir(parents=True, exist_ok=True)

    started = time.time()
    quantized = 0
    with safe_open(str(wan_path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        names = list(handle.keys())
        output = {}
        for index, name in enumerate(names, 1):
            if _quantizable(name):
                output.update(_quantize(handle.get_tensor(name), name, device))
                quantized += 1
            else:
                output[name] = handle.get_tensor(name)
            if index % 25 == 0 or index == len(names):
                print(f"\r读取 {index}/{len(names)}，已量化 {quantized} 个权重", end="", flush=True)
        print()

    out_path = dst / OUTPUT_NAME
    save_file(output, str(out_path), metadata=metadata)
    del output

    flash_src = src / FLASH_NAME
    if flash_src.is_file():
        shutil.copy2(flash_src, dst / FLASH_NAME)

    size = out_path.stat().st_size / 2**30
    print(f"完成：{out_path}（{size:.2f} GiB，{quantized} 个 int8 convrot 权重）")
    print(f"耗时 {time.time() - started:.1f} 秒")


def main():
    parser = argparse.ArgumentParser(description="RelayVSR Wan 主干 int8 convrot 转换")
    parser.add_argument("src", help="包含 wan_x2.safetensors 的原始目录")
    parser.add_argument("dst", nargs="?", help="输出目录，默认 <src>_int8_convrot")
    parser.add_argument("--device", default="cuda", help="量化设备，默认 cuda")
    options = parser.parse_args()

    src = Path(options.src).resolve()
    dst = Path(options.dst).resolve() if options.dst else src.with_name(src.name + "_int8_convrot")
    if torch.device(options.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("量化需要 CUDA 设备，请检查 PyTorch 安装")
    convert(src, dst, options.device)


if __name__ == "__main__":
    sys.exit(main())
