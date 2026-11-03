"""流式视频解码/编码，以及 RelayVSR 的时空分块调度。

节点拿到的是 ComfyUI 的 VIDEO（本质是文件路径），因此整段视频不会被读成
IMAGE 张量。这里按时间块解码、逐块做空间分块超分、把重叠帧线性融合，再把
完成的帧交给后台编码线程。任一时刻内存里只有一个时间块。音轨与输出容器
兼容时直接复用压缩包，否则解码成 PCM 后重新编码。
"""

import os
import queue
import threading
from collections import deque
from fractions import Fraction

import av
import numpy as np
import torch
from av.video.reformatter import Interpolation

import comfy.utils
import folder_paths

from .tiling import axis_alphas, axis_tiles, stitch

PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]

# codec -> (encoder, container, crf max)
CODECS = {
    "h264": ("libx264", "mp4", 51),
    "h265": ("libx265", "mp4", 51),
    "av1": ("libsvtav1", "webm", 63),
}

# Source audio codecs that each output container can remux untouched.
COPYABLE_AUDIO = {
    "mp4": {"aac", "mp3", "ac3", "eac3", "alac"},
    "webm": {"opus", "vorbis"},
}

# AAC only opens on its standard rate table; anything else has to be resampled.
AAC_RATES = {96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350}


def _transcode_audio(stream, container):
    """Pick a layout and rate that the output audio encoder accepts."""
    channels = stream.layout.nb_channels
    if container == "webm":  # libopus only takes mono/stereo at 48 kHz here.
        return ("mono" if channels == 1 else "stereo"), 48000
    rate = int(stream.sample_rate)
    return {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo"), rate if rate in AAC_RATES else 48000


def _rotation_quadrant(frame):
    return int(round(frame.rotation // 90)) % 4 if frame.rotation else 0


def _frame_to_tensor(frame):
    pixels = frame.to_ndarray(format="rgb24")
    quadrant = _rotation_quadrant(frame)
    if quadrant:
        pixels = np.rot90(pixels, k=quadrant, axes=(0, 1)).copy()
    return torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1)


def _video_crop(video):
    """ComfyUI 没有公开的 crop 读取接口，VideoFromFile 把它存在私有字段里。"""
    return getattr(video, "_VideoFromFile__crop", None)


def _normalize_crop(crop, width, height):
    x, y, crop_width, crop_height = (int(value) for value in crop)
    x = max(0, min(x, width - 1)) & ~1
    y = max(0, min(y, height - 1)) & ~1
    crop_width = min(crop_width, width - x) & ~1
    crop_height = min(crop_height, height - y) & ~1
    if crop_width <= 0 or crop_height <= 0 or (x, y, crop_width, crop_height) == (0, 0, width, height):
        return None
    return x, y, crop_width, crop_height


class VideoReader:
    """按时间顺序交出 uint8 (3,H,W) 帧，同时收集音轨。

    能与输出容器直接兼容的音频保留为压缩包，交给编码线程原样复用；否则解码成
    PCM，由编码线程重新编码。
    """

    def __init__(self, video, audio_container):
        self.crop = _video_crop(video)
        self._crop_rect = None
        self._crop_resolved = False
        self.source = video.get_stream_source()
        self.container = av.open(self.source, mode="r")
        if not self.container.streams.video:
            self.container.close()
            raise ValueError("RelayVSR: 输入视频没有视频流")
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.fps = Fraction(self.stream.average_rate) if self.stream.average_rate else Fraction(24)

        self.start_time, self.duration = video.get_active_trim_window()
        self._start_pts = int(self.start_time / self.stream.time_base)
        self._end_pts = int((self.start_time + self.duration) / self.stream.time_base) if self.duration else None
        if self._start_pts:
            self.container.seek(self._start_pts, stream=self.stream)

        self.audio_stream = next(
            (stream for stream in reversed(self.container.streams.audio) if stream.codec_context is not None), None
        )
        self.audio_layout = self.audio_stream.layout.name if self.audio_stream is not None else None
        self.audio_rate = int(self.audio_stream.sample_rate) if self.audio_stream is not None else 0
        self.audio_copy = (
            self.audio_stream is not None
            and not self.start_time
            and self.audio_stream.codec_context.name in COPYABLE_AUDIO[audio_container]
        )
        if self.audio_stream is not None and not self.audio_copy:
            self.audio_layout, self.audio_rate = _transcode_audio(self.audio_stream, audio_container)
        self._audio_resampler = (
            av.audio.resampler.AudioResampler(format="fltp", layout=self.audio_layout, rate=self.audio_rate)
            if self.audio_stream is not None and not self.audio_copy
            else None
        )

        streams = [self.stream] + ([self.audio_stream] if self.audio_stream is not None else [])
        self._packets = iter(self.container.demux(*streams))
        self._video = deque()
        self._audio = []
        self._eof = False

        if self.duration:
            seconds = self.duration
        elif self.stream.duration is not None and self.stream.time_base:
            seconds = float(self.stream.duration * self.stream.time_base) - self.start_time
        elif self.container.duration is not None:
            seconds = float(self.container.duration / av.time_base) - self.start_time
        else:
            seconds = 0.0
        self.frame_estimate = max(0, int(round(seconds * float(self.fps))))

        if self.stream.duration is not None and self.stream.time_base:
            raw_end = float(self.stream.duration * self.stream.time_base)
        elif self.container.duration is not None:
            raw_end = float(self.container.duration / av.time_base)
        else:
            raw_end = None
        if self.duration:
            end = self.start_time + self.duration
            self._audio_end = min(end, raw_end) if raw_end is not None else end
        else:
            self._audio_end = raw_end

    def read_chunk(self, count):
        frames = []
        while len(frames) < count:
            while not self._video and not self._eof:
                self._pump()
            if not self._video:
                break
            frames.append(self._video.popleft())
        audio, self._audio = self._audio, []
        return frames, audio

    def _pump(self):
        packet = next(self._packets, None)
        if packet is None:
            self._eof = True
            return
        if packet.stream is self.stream:
            for frame in packet.decode():
                if frame.pts is not None and frame.pts < self._start_pts:
                    continue
                if self._end_pts is not None and frame.pts >= self._end_pts:
                    self._eof = True
                    return
                tensor = _frame_to_tensor(frame)
                if self.crop is not None:
                    if not self._crop_resolved:
                        self._crop_rect = _normalize_crop(self.crop, tensor.shape[2], tensor.shape[1])
                        self._crop_resolved = True
                    if self._crop_rect is not None:
                        x, y, width, height = self._crop_rect
                        tensor = tensor[:, y:y + height, x:x + width].contiguous()
                self._video.append(tensor)
        elif packet.stream is self.audio_stream:
            if self.audio_copy:
                self._collect_audio_packet(packet)
            else:
                self._decode_audio(packet)

    def _collect_audio_packet(self, packet):
        pts = packet.pts if packet.pts is not None else packet.dts
        if pts is None:
            return
        if self._audio_end is not None and pts * self.audio_stream.time_base >= self._audio_end:
            return
        self._audio.append(packet)

    def _decode_audio(self, packet):
        for frame in packet.decode():
            for resampled in self._audio_resampler.resample(frame):
                self._append_audio(resampled)

    def _append_audio(self, frame):
        start = frame.time
        if start is None:
            return
        samples = frame.to_ndarray()
        rate = frame.sample_rate
        if start + samples.shape[1] / rate <= self.start_time:
            return
        if start < self.start_time:
            skip = int((self.start_time - start) * rate)
            samples = samples[:, skip:]
            start += skip / rate
        if self._audio_end is not None:
            remaining = self._audio_end - start
            if remaining <= 0:
                return
            samples = samples[:, :int(remaining * rate)]
        if samples.shape[1]:
            self._audio.append(np.ascontiguousarray(samples))

    def close(self):
        self.container.close()


class VideoWriter:
    """PyAV 编码器 + 有界队列 + 独立线程，GPU 处理下一块时同时压缩。

    输出容器和流在主线程创建，编码线程只负责编码与 mux；兼容的音轨由
    ``add_stream_from_template`` 建流后直接复用输入包。
    """

    def __init__(
        self, path, width, height, fps, codec, crf, preset,
        audio_layout=None, audio_rate=0, audio_template=None, queue_depth=2,
    ):
        encoder, container, crf_max = CODECS[codec]
        self.path = path
        self.width = width + width % 2
        self.height = height + height % 2
        self.audio_copy = audio_template is not None
        self.audio_layout = audio_layout
        self.audio_rate = audio_rate or 48000
        self.audio_samples = 0
        self._queue = queue.Queue(maxsize=max(1, int(queue_depth)))
        self._error = None
        self._stop = False
        self.output = None

        try:
            self.output = av.open(path, mode="w", format=container)
            self.video_stream = self.output.add_stream(encoder, rate=fps)
            self.video_stream.width = self.width
            self.video_stream.height = self.height
            self.video_stream.pix_fmt = "yuv420p"
            crf = int(round(min(max(crf, 0), crf_max)))
            if encoder == "libsvtav1":
                speed = max(0, min(13, 12 - PRESETS.index(preset)))
                self.video_stream.options = {"preset": str(speed), "crf": str(crf), "svtav1-params": "tune=0"}
            elif encoder == "libx265":
                self.video_stream.options = {"preset": preset, "crf": str(crf), "x265-params": "log-level=error"}
            else:
                self.video_stream.options = {"preset": preset, "crf": str(crf)}

            self.audio_stream = None
            if audio_template is not None:
                self.audio_stream = self.output.add_stream_from_template(audio_template, opaque=True)
            elif audio_layout:
                self.audio_stream = self.output.add_stream(
                    "libopus" if container == "webm" else "aac",
                    rate=self.audio_rate,
                    layout=audio_layout,
                )
        except BaseException:
            if self.output is not None:
                self.output.close()
            try:
                os.remove(path)
            except OSError:
                pass
            raise

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, frames, audio):
        if not frames and not audio:
            return
        payload = (frames, audio)
        while self._thread.is_alive():
            if self._error is not None:
                raise RuntimeError("RelayVSR: 视频编码失败") from self._error
            try:
                self._queue.put(payload, timeout=0.2)
                return
            except queue.Full:
                continue
        if self._error is not None:
            raise RuntimeError("RelayVSR: 视频编码失败") from self._error

    def finish(self):
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("RelayVSR: 视频编码失败") from self._error

    def abort(self):
        self._stop = True
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        self._thread.join()
        try:
            os.remove(self.path)
        except OSError:
            pass

    def _run(self):
        try:
            while True:
                payload = self._queue.get()
                if payload is None or self._stop:
                    break
                frames, audio = payload
                for frame in frames:
                    video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                    if frame.shape[0] != self.height or frame.shape[1] != self.width:
                        video_frame = video_frame.reformat(
                            width=self.width, height=self.height, format="yuv420p",
                            interpolation=Interpolation.LANCZOS,
                        )
                    else:
                        video_frame = video_frame.reformat(format="yuv420p")
                    self.output.mux(self.video_stream.encode(video_frame))
                if self.audio_stream is not None:
                    if self.audio_copy:
                        self._mux_audio(audio)
                    else:
                        self.audio_samples = self._write_audio(audio, self.audio_samples)

            if not self._stop:
                self.output.mux(self.video_stream.encode(None))
                if self.audio_stream is not None and not self.audio_copy:
                    self.output.mux(self.audio_stream.encode(None))
        except BaseException as error:
            self._error = error
        finally:
            try:
                self.output.close()
            except Exception as error:
                if self._error is None:
                    self._error = error

    def _mux_audio(self, packets):
        for packet in packets:
            packet.stream = self.audio_stream
            self.output.mux(packet)

    def _write_audio(self, arrays, samples):
        frames = []
        for array in arrays:
            frame = av.AudioFrame.from_ndarray(array, format="fltp", layout=self.audio_layout)
            frame.sample_rate = self.audio_rate
            frames.append(frame)

        for frame in frames:
            frame.pts = samples
            frame.time_base = Fraction(1, self.audio_rate)
            samples += frame.samples
            self.output.mux(self.audio_stream.encode(frame))
        return samples


def _upscale_chunk(engine, frames, sr_ratio, chunk_len, seed, tiles_y, tiles_x, y_alphas, x_alphas, check):
    if len(tiles_y) == 1 and len(tiles_x) == 1:
        return list(engine.restore(frames, seed, sr_ratio, chunk_len, check))

    tiles = []
    for y0, y1 in tiles_y:
        for x0, x1 in tiles_x:
            check()
            tile_input = [frame[:, y0:y1, x0:x1].contiguous() for frame in frames]
            tiles.append(list(engine.restore(tile_input, seed, sr_ratio, chunk_len, check)))

    outputs = []
    for offset in range(len(frames)):
        outputs.append(stitch([tile[offset] for tile in tiles], tiles_y, tiles_x, y_alphas, x_alphas, sr_ratio))
        for tile in tiles:
            tile[offset] = None
    return outputs


def _output_size(width, height, sr_ratio, upscale_width):
    if not upscale_width:
        return width * sr_ratio, height * sr_ratio
    out_width = max(2, int(round(upscale_width / 2)) * 2)
    out_height = max(2, int(round(out_width * height / width / 2)) * 2)
    return out_width, out_height


def stream_upscale(
    engine,
    video,
    upscale_width,
    sr_ratio,
    chunk_len,
    seed,
    chunk_overlap,
    spatial_grid_cols,
    spatial_grid_rows,
    spatial_overlap,
    codec,
    crf,
    preset,
    filename_prefix,
    check,
):
    """返回 (输出文件路径, 文件名, 子目录)。"""
    chunk_len = max(2, int(chunk_len))
    overlap = min(max(0, int(chunk_overlap)), chunk_len - 1)

    reader = VideoReader(video, CODECS[codec][1])
    writer = None
    try:
        frames, audio = reader.read_chunk(chunk_len)
        if not frames:
            raise ValueError("RelayVSR: 输入视频没有可解码的帧")

        height, width = frames[0].shape[-2:]
        out_width, out_height = _output_size(width, height, sr_ratio, upscale_width)
        tiles_y = axis_tiles(height, spatial_grid_rows, spatial_overlap, engine.geometry.wan_stride)
        tiles_x = axis_tiles(width, spatial_grid_cols, spatial_overlap, engine.geometry.wan_stride)
        y_alphas = axis_alphas(tiles_y, sr_ratio)
        x_alphas = axis_alphas(tiles_x, sr_ratio)

        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), out_width, out_height
        )
        file_name = "{}_{:05}_.{}".format(name, counter, CODECS[codec][1])
        out_path = os.path.join(folder, file_name)
        writer = VideoWriter(
            out_path, out_width, out_height, reader.fps, codec, crf, preset,
            reader.audio_layout, reader.audio_rate,
            reader.audio_stream if reader.audio_copy else None,
        )

        progress = comfy.utils.ProgressBar(reader.frame_estimate) if reader.frame_estimate else None
        written = 0
        pending = None

        while True:
            check()
            outputs = _upscale_chunk(
                engine, frames, sr_ratio, chunk_len, seed, tiles_y, tiles_x, y_alphas, x_alphas, check
            )
            if pending is not None:
                for index, previous in enumerate(pending):
                    weight = (index + 0.5) / overlap
                    blended = previous.astype(np.float32) * (1.0 - weight) + outputs[index].astype(np.float32) * weight
                    outputs[index] = np.clip(blended, 0, 255).round().astype(np.uint8)

            next_frames, next_audio = reader.read_chunk(chunk_len - overlap)
            if next_frames:
                if overlap:
                    writer.write(outputs[:-overlap], audio)
                    pending = outputs[-overlap:]
                else:
                    writer.write(outputs, audio)
                frames = (frames[-overlap:] if overlap else []) + next_frames
                audio = next_audio
                written += len(outputs) - overlap
            else:
                writer.write(outputs, audio + next_audio)
                written += len(outputs)
            if progress is not None:
                progress.update_absolute(min(written, reader.frame_estimate))
            if not next_frames:
                break

        writer.finish()
        writer = None
        return out_path, file_name, subfolder
    finally:
        reader.close()
        if writer is not None:
            writer.abort()
