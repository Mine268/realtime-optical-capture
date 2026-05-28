from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from roc.mvs.camera import CameraInfo


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(slots=True)
class OfflineCameraSource:
    serial: str
    path: Path


class OfflineMvsSystem:
    """MVS-like camera system backed by recorded videos or image folders."""

    def __init__(self, source_dir: Path, serials: list[str] | None = None, loop: bool = False) -> None:
        self.source_dir = source_dir.resolve()
        self.loop = loop
        self._closed = False
        self._sources = _discover_sources(self.source_dir, serials)
        if not self._sources:
            raise RuntimeError(f"No offline camera sources found in {self.source_dir}")

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "OfflineMvsSystem":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def enumerate_devices(self) -> list[CameraInfo]:
        devices = []
        for index, source in enumerate(self._sources):
            width, height = _read_source_size(source.path)
            devices.append(
                CameraInfo(
                    index=index,
                    serial=source.serial,
                    model_name="OfflineFileCamera",
                    transport_type=0,
                    transport_name="Offline",
                    width=width,
                    height=height,
                )
            )
        return devices

    def open_camera(self, camera_index: int) -> "OfflineMvsCamera":
        if camera_index < 0 or camera_index >= len(self._sources):
            raise RuntimeError(f"Offline camera index out of range: {camera_index}")
        source = self._sources[camera_index]
        return OfflineMvsCamera(source=source, camera_index=camera_index, loop=self.loop)


class OfflineMvsCamera:
    def __init__(self, source: OfflineCameraSource, camera_index: int, loop: bool = False) -> None:
        self.source = source
        self.serial = source.serial
        self.model_name = "OfflineFileCamera"
        self.transport_type = 0
        self.transport_name = "Offline"
        self.camera_index = camera_index
        self.loop = loop
        self._cap: cv2.VideoCapture | None = None
        self._image_paths: list[Path] = []
        self._image_index = 0
        self._opened = False
        self._grabbing = False
        self._pending_trigger = False
        self.open()

    def open(self) -> None:
        if self.source.path.is_dir():
            self._image_paths = sorted(path for path in self.source.path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
            if not self._image_paths:
                raise RuntimeError(f"No image frames found for offline camera {self.serial}: {self.source.path}")
        else:
            self._cap = cv2.VideoCapture(str(self.source.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open offline video for camera {self.serial}: {self.source.path}")
        self._opened = True

    def close(self) -> None:
        self._grabbing = False
        self._opened = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def apply_manual_capture(self, exposure_us: float, gain_db: float, pixel_format: str) -> None:
        return None

    def start_grabbing(self) -> None:
        self._grabbing = True

    def stop_grabbing(self) -> None:
        self._grabbing = False

    def trigger_software(self) -> None:
        self._pending_trigger = True

    def grab_frame(self, timeout_ms: int = 1000) -> np.ndarray | None:
        if not self._opened:
            raise RuntimeError(f"Offline camera {self.serial} is closed")
        if not self._grabbing:
            self.start_grabbing()
        self._pending_trigger = False
        if self._cap is not None:
            return self._grab_video_frame()
        return self._grab_image_frame()

    def snapshot(self, fps_sleep: float = 0.0) -> np.ndarray | None:
        self.trigger_software()
        return self.grab_frame()

    def __enter__(self) -> "OfflineMvsCamera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _grab_video_frame(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if ret:
            return frame
        if not self.loop:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap.read()
        return frame if ret else None

    def _grab_image_frame(self) -> np.ndarray | None:
        if self._image_index >= len(self._image_paths):
            if not self.loop:
                return None
            self._image_index = 0
        path = self._image_paths[self._image_index]
        self._image_index += 1
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read offline frame for camera {self.serial}: {path}")
        return frame


def _discover_sources(source_dir: Path, serials: list[str] | None) -> list[OfflineCameraSource]:
    if not source_dir.is_dir():
        raise RuntimeError(f"Offline source directory not found: {source_dir}")
    allowed = set(serials or [])
    sources: list[OfflineCameraSource] = []
    for path in sorted(source_dir.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_file() and path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue
        if not path.is_file() and not path.is_dir():
            continue
        serial = path.stem if path.is_file() else path.name
        if allowed and serial not in allowed:
            continue
        sources.append(OfflineCameraSource(serial=serial, path=path))
    return sources


def _read_source_size(path: Path) -> tuple[int, int]:
    if path.is_dir():
        frame_paths = sorted(candidate for candidate in path.iterdir() if candidate.suffix.lower() in IMAGE_SUFFIXES)
        if not frame_paths:
            return 0, 0
        frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if frame is None:
            return 0, 0
        height, width = frame.shape[:2]
        return width, height
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0, 0
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
