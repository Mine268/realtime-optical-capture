"""Parallel synchronized trigger+grab for multiple MVS cameras."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(slots=True)
class _Worker:
    serial: str
    camera: object
    trigger_delay_s: float = 0.0
    start_sem: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(0))
    done_sem: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(0))
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    frame: object = None
    error: Exception | None = None

    def _run(self) -> None:
        while True:
            self.start_sem.acquire()
            if self.stop_event.is_set():
                self.done_sem.release()
                return
            try:
                self.camera.trigger_software()
                aborted = self.trigger_delay_s > 0 and self.stop_event.wait(self.trigger_delay_s)
                if not aborted:
                    self.frame = self.camera.grab_frame()
            except Exception as exc:
                self.error = exc
            finally:
                self.done_sem.release()

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.start_sem.release()
        self.done_sem.acquire()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


class ParallelCapture:
    """Context manager for parallel synchronized multi-camera trigger+grab."""

    def __init__(self, cameras: list[tuple[str, object]], trigger_delay_s: float = 0.0) -> None:
        if not cameras:
            raise ValueError("At least one camera required")
        self._workers = [
            _Worker(serial=serial, camera=cam, trigger_delay_s=trigger_delay_s)
            for serial, cam in cameras
        ]

    def __enter__(self) -> ParallelCapture:
        for w in self._workers:
            w.start()
        return self

    def __exit__(self, *_: object) -> None:
        for w in self._workers:
            w.stop()

    def snapshot_all(self) -> dict[str, object]:
        """Trigger all cameras simultaneously, wait for all frames, return {serial: frame}."""
        for w in self._workers:
            w.start_sem.release()

        result: dict[str, object] = {}
        for w in self._workers:
            w.done_sem.acquire()
            if w.error is not None:
                raise w.error
            result[w.serial] = w.frame
        return result
