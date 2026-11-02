# Comfyui-Relay-VSR

A ComfyUI custom node pack for [Relay-VSR](https://github.com/kopperx/RelayVSR) and [LightX2V Relay-VSR-2x](https://github.com/ModelTC/LightX2V/tree/main/scripts/relayvsr) — a one-step distilled,
pixel-space video super-resolution model. It runs the Wan backbone and the Flash decoder directly,
with no VAE, CLIP or text encoder: **images in, images out**.

- Native LightX2V 2× weights (`RelayVSR_X2`)
- Original 4× weights (`RelayVSR_X4`)
- Streaming windows: only a small number of frames are in flight, so memory stays flat for long clips
- Optional INT8 conv-rotation weights for the Wan backbone
- Optional per-block Wan CPU offload

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-account>/Comfyui-Relay-VSR.git
```

Restart ComfyUI. No additional pip packages are required: the nodes only use `torch`, `safetensors`,
`numpy` and ComfyUI's own dependencies (`comfy.quant_ops`, `comfy-kitchen`). There is intentionally
no `requirements.txt`, so ComfyUI's pinned versions stay authoritative.

## Models

| Scale | Source |
| --- | --- |
| 2× | https://huggingface.co/lightx2v/RelayVSR_X2 |
| 4× | https://github.com/kopperx/RelayVSR |

Place one directory per bundle under `ComfyUI/models/relayvsr/`:

```
ComfyUI/models/relayvsr/
├── RelayVSR_X2/
│   ├── wan_x2.safetensors
│   └── flash_x2.safetensors
└── RelayVSR_X4/
    ├── wan.safetensors
    └── flash.safetensors
```

Each directory is discovered automatically and its scale is detected from the safetensors metadata.
The `relayvsr` key is registered with ComfyUI, so you can also point other disks at it through
`extra_model_paths.yaml`. After adding or moving folders, restart ComfyUI (or use the loader's
custom path field).

## Nodes

### RelayVSR Model Loader — category `model/loaders`

| Input | Description |
| --- | --- |
| `bundle` | Directory under `models/relayvsr` containing a matched Wan/Flash pair |
| `dtype` | `bf16` (default) or `fp16` for the floating-point weights |
| `offload` | Move Wan blocks to the GPU one at a time. Enabled by default; disable it on 32 GB+ cards for faster reference passes |
| `path` | Advanced: absolute path to a weight directory, overrides `bundle` |

Output: `RELAYVSR_MODEL` (connect to the upscale node).

### RelayVSR Upscale — category `image/upscaling`

| Input | Default | Description |
| --- | --- | --- |
| `model` | — | Output of the loader |
| `images` | — | IMAGE batch; each frame is processed as part of one sequence |
| `sr_ratio` | 2 | Output scale. 2× weights require `2`; 4× weights accept `2` or `4` |
| `chunk_len` | 16 | Frames per streaming window. Larger chunks are faster but use more VRAM |
| `seed` | 0 | Only affects the initial noise of the reference frames |

Output: `IMAGE` at `height * sr_ratio` × `width * sr_ratio`. Progress and cancellation are supported.

## INT8 conv-rotation weights

`convert_to_int8_convrot.py` rewrites the Wan backbone into ComfyUI's native INT8 tensor-wise
format with conv rotation. Only the attention/FFN linear layers inside `blocks.*` are quantized;
convolutions, norms, embeddings, biases and `proj_out` stay in full precision. The Flash decoder is
copied unchanged. The loader detects `<module>.comfy_quant` automatically and routes `F.linear`
through ComfyUI's INT8 kernel.

Run it with ComfyUI's Python, since the script imports `torch` and `comfy_kitchen` (both come with
ComfyUI). From the portable root (`ComfyUI_windows_portable`) on Windows:

```bat
python_embeded\python.exe ComfyUI\custom_nodes\Comfyui-Relay-VSR\convert_to_int8_convrot.py ComfyUI\models\relayvsr\RelayVSR_X2
```

For a manual install, use the Python from your ComfyUI virtual environment instead. The script
requires CUDA (`--device cuda` is the default).

The output goes to `RelayVSR_X2_int8_convrot/` by default and is selected as a separate bundle:

```
ComfyUI/models/relayvsr/
├── RelayVSR_X2_int8_convrot/
│   ├── wan_x2_int8_convrot.safetensors
│   └── flash_x2.safetensors
```

INT8 bundles support the same block offload as BF16 bundles. The converter currently targets the 2×
filenames (`wan_x2.safetensors` / `flash_x2.safetensors`).

## Notes

- Wan in BF16 is about 9.4 GB of weights; INT8 conv-rotation roughly halves that. The Flash decoder
  is about 120 MB.
- A 768×1344 clip with 16-frame windows needs about 18–19 GB in BF16. On 24 GB cards, leave `offload`
  enabled: the Wan KV cache alone is about 12 GB at 1080p, and holding the Wan weights in VRAM as well
  will run out of memory.
- There is no prompt input: RelayVSR uses a fixed prompt embedding.
- The model is 1-step distilled (sigma 1 → 0, no CFG), so `seed` has very limited influence.

## License and attribution

This project is licensed under the Apache License 2.0.

It contains code ported from:

- RelayVSR — https://github.com/kopperx/RelayVSR (Apache-2.0)
- LightX2V — https://github.com/ModelTC/LightX2V (Apache-2.0)

Modifications: removed the LightX2V framework/registry dependencies, rewrote attention, KV caching
and streaming-window orchestration for ComfyUI, and added ComfyUI nodes, INT8 loading and
model-directory handling.

RelayVSR, LightX2V and their logos belong to their respective owners; this project is not endorsed
by or affiliated with them. Model weights are not distributed here and may be governed by separate
terms on Hugging Face / GitHub.
