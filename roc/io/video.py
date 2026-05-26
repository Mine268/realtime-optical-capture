from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def build_mjpg_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open temporary video writer: {path}")
    return writer


def encode_h264_mp4(temp_video_path: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-r",
        f"{max(fps, 1.0):.6f}",
        "-i",
        str(temp_video_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to encode compatible H.264 mp4 files") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to encode H.264 mp4: {output_path}") from exc


class H264VideoWriter:
    def __init__(self, output_path: Path, width: int, height: int, fps: float, temp_suffix: str = ".tmp.avi") -> None:
        self.output_path = output_path
        self.fps = max(fps, 1.0)
        self.temp_path = output_path.with_name(output_path.stem + temp_suffix)
        self.writer = build_mjpg_writer(self.temp_path, width, height, self.fps)
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError(f"Cannot write to closed video writer: {self.output_path}")
        self.writer.write(frame)

    def close(self) -> None:
        if self.closed:
            return
        self.writer.release()
        try:
            encode_h264_mp4(self.temp_path, self.output_path, self.fps)
        finally:
            self.temp_path.unlink(missing_ok=True)
        self.closed = True

    def __enter__(self) -> "H264VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_frames_to_h264_mp4(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    if not frame_paths:
        raise RuntimeError(f"No frames provided for video output: {output_path}")

    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_paths[0]}")

    writer = H264VideoWriter(output_path, first_frame.shape[1], first_frame.shape[0], fps)
    try:
        writer.write(first_frame)
        for frame_path in frame_paths[1:]:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            writer.write(frame)
    finally:
        writer.close()
