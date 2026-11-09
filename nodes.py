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
from comfy.model_patcher import ModelPatcher
import folder_paths
from comfy_api.latest import ComfyExtension, io, ui
from typing_extensions import override

from .checkpoints import inspect_bundle
from .runtime import RelayVSREngine
from .streaming import CODECS, PRESETS, stream_upscale

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


class _EngineSlot(torch.nn.Module):
    """ModelPatcher needs an nn.Module to track; the engine hangs off this one."""

    def __init__(self):
        super().__init__()
        self.engine = None


class RelayVSRPatcher(ModelPatcher):
    """Registers the engine with ComfyUI model management.

    The engine keeps its weights as plain tensors, not nn.Module parameters, so
    the stock patcher cannot move them. This subclass loads and frees the whole
    engine instead. "Unload models", /free and memory-pressure eviction then
    release it like any other model.
    """

    def __init__(self, info, dtype, offload, device):
        self.info = info
        self.relay_dtype = dtype
        self.offload = offload
        # Estimate before the first load; partially_load replaces it with the measured VRAM.
        size = os.path.getsize(info.wan_path) + os.path.getsize(info.flash_path)
        super().__init__(_EngineSlot(), device, mm.unet_offload_device(), size=size)

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        if self.model.engine is not None:
            return 0
        cuda = device_to.type == "cuda"
        before = torch.cuda.memory_allocated(device_to) if cuda else 0
        engine = RelayVSREngine(self.info, self.relay_dtype, self.offload, device_to)
        try:
            engine.load()
        except Exception:
            engine.free()
            raise
        self.model.engine = engine
        self.model.device = device_to
        if cuda:
            self.size = max(torch.cuda.memory_allocated(device_to) - before, 1)
        self.model.model_loaded_weight_memory = self.size
        return self.size

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        # The engine cannot be partly unloaded. Returning 0 makes ComfyUI call detach() and free all of it.
        return 0

    def detach(self, unpatch_all=True):
        if self.model.engine is not None:
            self.model.engine.free()
            self.model.engine = None
        self.model.device = self.offload_device
        self.model.model_loaded_weight_memory = 0
        return self.model


class RelayVSRModel:
    """加载器输出：持有权重路径，首次超分时才把权重搬进显存。"""

    def __init__(self, info, dtype, offload):
        self.patcher = RelayVSRPatcher(info, dtype, offload, mm.get_torch_device())

    def ensure_loaded(self):
        mm.load_models_gpu([self.patcher])
        return self.patcher.model.engine

    def free(self):
        self.patcher.detach()


_current_model = None


def _release_current():
    global _current_model
    if _current_model is not None:
        _current_model.free()
        _current_model = None


def _images_to_frames(images):
    return [frame.mul(255).round().clamp(0, 255).to(torch.uint8).cpu() for frame in images.movedim(-1, 1)]


def _frames_to_images(frames):
    return torch.stack([torch.from_numpy(frame) for frame in frames]).float().div(255.0)


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
        frames = _images_to_frames(images)
        progress = comfy.utils.ProgressBar(len(frames))
        outputs = []
        for pixel in engine.restore(frames, seed, sr_ratio, chunk_len, mm.throw_exception_if_processing_interrupted):
            outputs.append(pixel)
            progress.update(1)
        return io.NodeOutput(_frames_to_images(outputs))


class RelayVSRUpscaleStreaming(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RelayVSRUpscaleStreaming",
            display_name="RelayVSR Upscale Streaming",
            category="video/upscaling",
            description=(
                "Upscale a video with RelayVSR without materializing it as IMAGE tensors. Frames are decoded "
                "one time chunk at a time, each chunk is upscaled in spatial tiles, temporal overlaps are "
                "crossfaded, and finished frames go straight to the encoder. The whole clip never sits in RAM."
            ),
            inputs=[
                MODEL_TYPE.Input("model"),
                io.Video.Input(
                    "video",
                    tooltip="The video to upscale. It is read and written incrementally; only one time chunk is in memory",
                ),
                io.Int.Input(
                    "upscale_width",
                    default=0,
                    min=0,
                    max=16384,
                    step=2,
                    tooltip="Final output width. Height follows the source aspect ratio and is rounded to even. 0 keeps width * sr_ratio",
                ),
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
                    default=24,
                    min=2,
                    max=64,
                    tooltip="Frames per streaming pass and RelayVSR window; larger values blend fewer temporal seams but use more RAM",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Only affects the initial noise of reference frames",
                ),
                io.Int.Input(
                    "chunk_overlap",
                    default=4,
                    min=0,
                    max=256,
                    tooltip="Frames crossfaded between adjacent time chunks; must be smaller than chunk_len",
                ),
                io.Int.Input(
                    "spatial_grid_cols",
                    default=1,
                    min=1,
                    max=8,
                    tooltip="Number of tiles along the width; 1 disables horizontal tiling",
                ),
                io.Int.Input(
                    "spatial_grid_rows",
                    default=1,
                    min=1,
                    max=8,
                    tooltip="Number of tiles along the height; 1 disables vertical tiling",
                ),
                io.Int.Input(
                    "spatial_overlap",
                    default=64,
                    min=0,
                    max=1024,
                    step=8,
                    tooltip="Overlap between spatial tiles in input pixels, crossfaded linearly",
                ),
                io.Combo.Input(
                    "codec",
                    options=list(CODECS),
                    default="h264",
                    tooltip="h264: compatible default. h265: smaller at the same quality. av1: smallest, slowest, WebM output",
                ),
                io.Float.Input(
                    "crf",
                    default=18.0,
                    min=0.0,
                    max=63.0,
                    step=1.0,
                    tooltip="Quality: lower is better and bigger. Values above each codec's limit are clamped",
                ),
                io.Combo.Input(
                    "preset",
                    options=PRESETS,
                    default="medium",
                    tooltip="Encoder speed vs compression; maps onto the numeric speed knob for AV1",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="RelayVSR/upscaled",
                    tooltip="Output name under ComfyUI's output directory; a counter is appended",
                ),
            ],
            outputs=[
                io.String.Output(
                    "filename",
                    tooltip="Absolute path of the saved video",
                )
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls, model, video, upscale_width, sr_ratio, chunk_len, seed, chunk_overlap,
        spatial_grid_cols, spatial_grid_rows, spatial_overlap, codec, crf, preset, filename_prefix,
    ):
        engine = model.ensure_loaded()
        out_path, file_name, subfolder = stream_upscale(
            engine, video, upscale_width, sr_ratio, chunk_len, seed, chunk_overlap,
            spatial_grid_cols, spatial_grid_rows, spatial_overlap, codec, crf, preset, filename_prefix,
            mm.throw_exception_if_processing_interrupted,
        )
        return io.NodeOutput(
            out_path,
            ui=ui.PreviewVideo([ui.SavedResult(file_name, subfolder, io.FolderType.output)]),
        )


class RelayVSRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [RelayVSRModelLoader, RelayVSRUpscale, RelayVSRUpscaleStreaming]


async def comfy_entrypoint() -> RelayVSRExtension:
    return RelayVSRExtension()


