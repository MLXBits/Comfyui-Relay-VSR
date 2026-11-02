"""ComfyUI 节点：RelayVSR 模型加载与像素级视频超分。

模型按目录组织，插件会自动把 ``models/relayvsr`` 注册到 ComfyUI 的模型路径，
因此 extra_model_paths.yaml 里也可以用 ``relayvsr`` 作为键指向其它磁盘。
"""

import logging
import os
from pathlib import Path

import torch
import comfy.model_management as mm
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .checkpoints import inspect_bundle
from .runtime import RelayVSREngine

MODEL_TYPE = io.Custom("RELAYVSR_MODEL")
MODEL_DIR = "relayvsr"
PLACEHOLDER = "No RelayVSR weights found"

folder_paths.add_model_folder_path(MODEL_DIR, os.path.join(folder_paths.models_dir, MODEL_DIR))


def _scan_bundles():
    found = {}
    for root in folder_paths.get_folder_paths(MODEL_DIR):
        root = Path(root)
        if not root.is_dir():
            continue
        candidates = [root]
        candidates.extend(
            Path(entry.path) for entry in sorted(os.scandir(root), key=lambda entry: entry.name) if entry.is_dir()
        )
        for candidate in candidates:
            try:
                info = inspect_bundle(candidate)
            except ValueError as error:
                logging.warning(f"RelayVSR 跳过 {candidate}: {error}")
                continue
            if info is None:
                continue
            label = candidate.name if candidate != root else root.name
            if label in found:
                label = f"{root.name}/{candidate.name}"
            found[label] = info
    return found


BUNDLES = _scan_bundles()


class RelayVSRModel:
    """加载器输出：持有权重路径，首次超分时才把权重搬进显存。"""

    def __init__(self, info, dtype, offload):
        self.info = info
        self.dtype = dtype
        self.offload = offload
        self.device = mm.get_torch_device()
        self.engine = None

    def ensure_loaded(self):
        if self.engine is None:
            engine = RelayVSREngine(self.info, self.dtype, self.offload, self.device)
            engine.load()
            self.engine = engine
        return self.engine

    def free(self):
        if self.engine is not None:
            self.engine.free()
            self.engine = None


_current_model = None


def _release_current():
    global _current_model
    if _current_model is not None:
        _current_model.free()
        _current_model = None


class RelayVSRModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RelayVSRModelLoader",
            display_name="RelayVSR Model Loader",
            category="model/loaders",
            description="Load the RelayVSR Wan backbone and Flash decoder (supports native x2 and x4 weights).",
            inputs=[
                io.Combo.Input(
                    "bundle",
                    options=sorted(BUNDLES) or [PLACEHOLDER],
                    tooltip="Directory under models/relayvsr containing a matched wan/flash weight pair",
                ),
                io.Combo.Input("dtype", options=["bf16", "fp16"], default="bf16"),
                io.Boolean.Input(
                    "offload",
                    default=True,
                    label_on="Offload Wan blocks",
                    label_off="Keep all in VRAM",
                    tooltip="Move Wan blocks to the GPU one at a time. Enabled by default; disable it on 32 GB+ cards for faster reference passes",
                ),
                io.String.Input(
                    "path",
                    default="",
                    advanced=True,
                    tooltip="Absolute path to a custom weight directory; leave empty to use the bundle above",
                ),
            ],
            outputs=[MODEL_TYPE.Output("model", display_name="model")],
        )

    @classmethod
    def execute(cls, bundle, dtype, offload, path):
        global _current_model
        path = (path or "").strip()
        if path:
            info = inspect_bundle(Path(path))
            if info is None:
                raise ValueError(f"目录中没有配对的 RelayVSR Wan/Flash 权重: {path}")
        else:
            info = BUNDLES.get(bundle) or _scan_bundles().get(bundle)
            if info is None:
                raise ValueError(
                    f"未找到 RelayVSR 权重，请把 wan/flash 权重放进 {os.path.join(folder_paths.models_dir, MODEL_DIR)}"
                )
        _release_current()
        _current_model = RelayVSRModel(info, dtype, offload)
        return io.NodeOutput(_current_model)


class RelayVSRUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RelayVSRUpscale",
            display_name="RelayVSR Upscale",
            category="image/upscaling",
            description="Upscale an image sequence frame by frame with RelayVSR. Pixels in, pixels out.",
            inputs=[
                MODEL_TYPE.Input("model"),
                io.Image.Input("images"),
                io.Int.Input(
                    "sr_ratio",
                    default=2,
                    min=2,
                    max=4,
                    step=2,
                    tooltip="x2 weights require 2; x4 weights accept 2 or 4",
                ),
                io.Int.Input(
                    "chunk_len",
                    default=16,
                    min=2,
                    max=64,
                    advanced=True,
                    tooltip="Frames per streaming window; larger chunks avoid repeated work at window boundaries",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Only affects the initial noise of reference frames",
                ),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, model, images, sr_ratio, chunk_len, seed):
        engine = model.ensure_loaded()
        frames = [frame.mul(255).round().clamp(0, 255).to(torch.uint8).cpu() for frame in images.movedim(-1, 1)]
        progress = comfy.utils.ProgressBar(len(frames))
        outputs = []
        for pixel in engine.restore(frames, seed, sr_ratio, chunk_len, mm.throw_exception_if_processing_interrupted):
            outputs.append(torch.from_numpy(pixel))
            progress.update(1)
        return io.NodeOutput(torch.stack(outputs).float().div(255.0))


class RelayVSRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [RelayVSRModelLoader, RelayVSRUpscale]


async def comfy_entrypoint() -> RelayVSRExtension:
    return RelayVSRExtension()


