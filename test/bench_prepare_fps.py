"""Benchmark max FPS for the roc prepare camera pipeline.

Compares serial vs multi-threaded synchronized capture.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from roc.mvs import MvsSystem


@dataclass(slots=True)
class _Worker:
    serial: str
    camera: object
    start_sem: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(0))
    done_sem: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(0))
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    frame_count: int = 0
    null_count: int = 0
    error: Exception | None = None

    def _run(self) -> None:
        while True:
            self.start_sem.acquire()
            if self.stop_event.is_set():
                self.done_sem.release()
                return
            try:
                self.camera.trigger_software()
                frame = self.camera.grab_frame()
                if frame is None:
                    self.null_count += 1
                else:
                    self.frame_count += 1
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


def _bench_serial(cameras: list[tuple[str, object]], duration_s: float) -> dict:
    frame_counts = {serial: 0 for serial, _ in cameras}
    null_counts = {serial: 0 for serial, _ in cameras}
    total_start = time.perf_counter()

    while time.perf_counter() - total_start < duration_s:
        for serial, cam in cameras:
            cam.trigger_software()
            frame = cam.grab_frame()
            if frame is None:
                null_counts[serial] += 1
            else:
                frame_counts[serial] += 1

    elapsed = time.perf_counter() - total_start
    return {
        "frame_counts": frame_counts,
        "null_counts": null_counts,
        "elapsed": elapsed,
    }


def _bench_synced(cameras: list[tuple[str, object]], duration_s: float) -> dict:
    workers = [_Worker(serial=serial, camera=cam) for serial, cam in cameras]
    for w in workers:
        w.start()

    frame_set_index = 0
    total_start = time.perf_counter()

    try:
        while time.perf_counter() - total_start < duration_s:
            for w in workers:
                w.start_sem.release()

            for w in workers:
                w.done_sem.acquire()
                if w.error is not None:
                    raise w.error

            frame_set_index += 1
    finally:
        for w in workers:
            w.stop()

    elapsed = time.perf_counter() - total_start
    return {
        "frame_counts": {w.serial: w.frame_count for w in workers},
        "null_counts": {w.serial: w.null_count for w in workers},
        "elapsed": elapsed,
        "frame_sets": frame_set_index,
    }


def main() -> None:
    with MvsSystem() as mvs:
        devices = mvs.enumerate_devices()
        print(f"Found {len(devices)} devices")
        for d in devices:
            print(f"  [{d.index}] {d.serial} {d.model_name}")

        # Open cameras once, then run both benchmarks
        cameras: list[tuple[str, object]] = []
        for device in devices:
            cam = mvs.open_camera(device.index)
            cam.apply_manual_capture(
                exposure_us=8000,
                gain_db=6.0,
                pixel_format="BayerRG8",
            )
            cam.start_grabbing()
            cameras.append((device.serial, cam))

        try:
            print(f"\nOpened {len(cameras)} cameras.\n")

            # ---- serial ----
            print("=== Serial (trigger+grab each camera in sequence) ===")
            serial_result = _bench_serial(cameras, duration_s=5.0)
            _print_result(serial_result, cameras)

            # ---- synced ----
            print("=== Synchronized (all cameras trigger+grab in parallel) ===")
            synced_result = _bench_synced(cameras, duration_s=5.0)
            _print_result(synced_result, cameras)

            # ---- comparison ----
            print("=== Comparison ===")
            ser_fps = sum(serial_result["frame_counts"].values()) / serial_result["elapsed"]
            sync_fps = sum(synced_result["frame_counts"].values()) / synced_result["elapsed"]
            sync_fs = synced_result["frame_sets"] / synced_result["elapsed"]
            print(f"  Serial:        {ser_fps:.1f} total frames/sec, ~{ser_fps / len(cameras):.1f} FPS per camera")
            print(f"  Synchronized:  {sync_fps:.1f} total frames/sec, {sync_fs:.1f} frame-sets/sec")
            speedup = sync_fs / (ser_fps / len(cameras))
            print(f"  Frame-set rate speedup: {speedup:.1f}x")
        finally:
            for _, cam in cameras:
                cam.close()


def _print_result(result: dict, cameras: list) -> None:
    elapsed = result["elapsed"]
    frame_counts = result["frame_counts"]
    null_counts = result["null_counts"]
    frame_sets = result.get("frame_sets")

    print(f"  Elapsed: {elapsed:.3f}s")
    for serial in sorted(frame_counts):
        count = frame_counts[serial]
        nulls = null_counts[serial]
        print(f"    {serial}: {count} frames ({count / elapsed:.1f} FPS), {nulls} nulls")
    total = sum(frame_counts.values())
    print(f"  Total: {total} frames ({total / elapsed:.1f} frames/sec)")
    if frame_sets is not None:
        print(f"  Frame sets: {frame_sets} ({frame_sets / elapsed:.1f} frame-sets/sec)")
    print()


if __name__ == "__main__":
    main()
