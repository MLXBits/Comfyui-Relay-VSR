"""RelayVSR 的空间分块与线性融合。

每个时间块内把一帧切成矩形区域，分块边界用线性交叉权重融合，避免硬接缝。
"""

import math

import numpy as np


def axis_tiles(size, count, overlap, stride):
    """把一条边切成 count 段，返回等长的 (start, end) 区间。

    起点尽量对齐 stride，末端区间贴住 size，避免单独处理边角。
    """
    if count <= 1 or size < stride * 2:
        return [(0, size)]
    count = min(count, size // stride)
    if count <= 1:
        return [(0, size)]
    tile = math.ceil((size + (count - 1) * overlap) / count)
    tile = min(size, math.ceil(tile / stride) * stride)
    tiles = []
    for index in range(count):
        if index == count - 1:
            start = size - tile
        else:
            start = round(index * (size - tile) / (count - 1))
            start -= start % stride
            start = min(max(start, 0), size - tile)
        if tiles and start <= tiles[-1][0]:
            continue
        tiles.append((start, start + tile))
    if tiles[-1][1] < size:
        tiles[-1] = (size - tile, size)
    return tiles


def _ramp(length, fade_in, fade_out):
    alpha = np.ones(length, dtype=np.float32)
    if fade_in > 0:
        count = min(int(fade_in), length)
        alpha[:count] = (np.arange(count, dtype=np.float32) + 0.5) / count
    if fade_out > 0:
        count = min(int(fade_out), length)
        alpha[length - count:] = (np.arange(count, dtype=np.float32)[::-1] + 0.5) / count
    return alpha


def axis_alphas(tiles, scale):
    """每个分块在该轴上的线性权重，长度按输出像素计算。"""
    alphas = []
    for index, (start, end) in enumerate(tiles):
        fade_in = (tiles[index - 1][1] - start) * scale if index > 0 else 0
        fade_out = (end - tiles[index + 1][0]) * scale if index + 1 < len(tiles) else 0
        alphas.append(_ramp((end - start) * scale, fade_in, fade_out))
    return alphas


def stitch(tile_frames, tiles_y, tiles_x, y_alphas, x_alphas, scale):
    """把同一帧的各空间分块按权重拼回整帧，返回 HWC uint8。"""
    height = tiles_y[-1][1] * scale
    width = tiles_x[-1][1] * scale
    pixels = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    index = 0
    for row, (y0, y1) in enumerate(tiles_y):
        for col, (x0, x1) in enumerate(tiles_x):
            alpha = (y_alphas[row][:, None] * x_alphas[col][None, :])[..., None]
            top, left = y0 * scale, x0 * scale
            tile_height, tile_width = (y1 - y0) * scale, (x1 - x0) * scale
            region = (slice(top, top + tile_height), slice(left, left + tile_width))
            pixels[region] += tile_frames[index].astype(np.float32) * alpha
            weights[region] += alpha
            index += 1
    return np.clip(pixels / np.maximum(weights, 1e-6), 0, 255).round().astype(np.uint8)
