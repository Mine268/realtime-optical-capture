from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
import time

import cv2


@dataclass(slots=True)
class SyncCaptureWorker:
    serial: str
    camera: object
    raw_dir: Path
    trigger_delay_s: float
    frame_index: int = 0
    last_frame: object = None
    error: Exception | None = None
    start_sem: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(0))
    done_sem: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(0))
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def start(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, name=f"capture-{self.serial}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            self.start_sem.acquire()
            if self.stop_event.is_set():
                self.done_sem.release()
                break
            try:
                self.camera.trigger_software()
                if self.trigger_delay_s > 0:
                    time.sleep(self.trigger_delay_s)
                frame = self.camera.grab_frame(timeout_ms=2000)
                if frame is None:
                    raise RuntimeError(f"No frame received for camera {self.serial}")
                frame_path = self.raw_dir / f"frame_{self.frame_index:06d}.bmp"
                ok = cv2.imwrite(str(frame_path), frame)
                if not ok:
                    raise RuntimeError(f"Failed to write raw frame: {frame_path}")
                self.last_frame = frame
            except Exception as exc:  # noqa: BLE001
                self.error = exc
            finally:
                self.done_sem.release()

    def stop(self) -> None:
        self.stop_event.set()
        self.start_sem.release()
        self.done_sem.acquire()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


def transcode_raw_frames_to_videos(
    raw_frame_dirs: dict[str, Path],
    final_video_dir: Path,
    timestamps_ns: list[int],
    fallback_fps: float,
) -> float:
    if len(timestamps_ns) >= 2:
        duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
        actual_fps = (len(timestamps_ns) - 1) / duration_s if duration_s > 0 else fallback_fps
    else:
        actual_fps = fallback_fps
    actual_fps = max(actual_fps, 1.0)

    final_video_dir.mkdir(parents=True, exist_ok=True)
    for serial, raw_dir in raw_frame_dirs.items():
        frame_paths = sorted(raw_dir.glob("frame_*.bmp"))
        if not frame_paths:
            raise RuntimeError(f"No raw frames found for camera {serial} in {raw_dir}")
        writer = None
        try:
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Failed to read raw frame: {frame_path}")
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_path = final_video_dir / f"{serial}.mp4"
                    writer = cv2.VideoWriter(str(video_path), fourcc, actual_fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open final mocap video writer: {video_path}")
                writer.write(frame)
        finally:
            if writer is not None:
                writer.release()

    return actual_fps
