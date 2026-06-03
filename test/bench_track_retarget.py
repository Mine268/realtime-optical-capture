"""Benchmark track retarget speed and accuracy on an existing mocap NPZ."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from roc.mocap.retarget import RetargetConfig, RetargetMode
from roc.mocap.track import RealtimeSmplxTracker


def main() -> None:
    npz_path = Path("sessions/mocap_live_rkw_20260603_test01/mocap_live_rkw_20260603_test01.npz")
    output_dir = Path("sessions/mocap_live_rkw_20260603_test01/smplx_track_gpu")
    mocap_session = Path("sessions/mocap_live_rkw_20260603_test01")

    data = np.load(npz_path, allow_pickle=True)
    points_3d = data["points_3d"]
    frames = points_3d.shape[0]
    print(f"NPZ: {npz_path}")
    print(f"Frames: {frames}, landmarks: {points_3d.shape[1]}")

    # ---- Track mode ----
    print("\n=== Track Mode (GPU) ===")
    track_config = RetargetConfig(
        model_dir=Path("models/smplx"),
        mode=RetargetMode.TRACK,
        output_dir=output_dir,
        device="cuda",
        track_pose_steps=18,
        track_temporal_weight=0.20,
        track_velocity_weight=0.02,
        track_acceleration_weight=0.004,
        track_recovery_pose_steps=60,
        profile=True,
    )
    tracker = RealtimeSmplxTracker(track_config, output_dir)

    track_times = []
    body_errors = []
    for frame_idx in range(frames):
        t0 = time.perf_counter()
        result = tracker.update(frame_idx, points_3d[frame_idx])
        elapsed = time.perf_counter() - t0
        track_times.append(elapsed)
        body_errors.append(result["body_mean_error_m"].item() if hasattr(result["body_mean_error_m"], 'item') else float(result["body_mean_error_m"]))

    track_times = np.array(track_times)
    body_errors = np.array(body_errors)

    steady_mask = np.arange(frames) > 0
    steady_times = track_times[steady_mask]
    steady_errors = body_errors[steady_mask]

    print(f"\n  All {frames} frames:")
    print(f"    time: mean={np.mean(track_times)*1000:.1f}ms, p50={np.percentile(track_times,50)*1000:.1f}ms, p95={np.percentile(track_times,95)*1000:.1f}ms")
    print(f"    body_err: mean={np.mean(body_errors)*1000:.1f}mm, p50={np.percentile(body_errors,50)*1000:.1f}mm, p95={np.percentile(body_errors,95)*1000:.1f}mm")
    print(f"    FPS: {1.0 / np.mean(track_times):.1f}")

    print(f"\n  Steady (frames 1-{frames-1}):")
    print(f"    time: mean={np.mean(steady_times)*1000:.1f}ms, p50={np.percentile(steady_times,50)*1000:.1f}ms")
    print(f"    body_err: mean={np.mean(steady_errors)*1000:.1f}mm, p50={np.percentile(steady_errors,50)*1000:.1f}mm, p95={np.percentile(steady_errors,95)*1000:.1f}mm")
    print(f"    FPS: {1.0 / np.mean(steady_times):.1f}")

    track_npz = tracker.save(source_npz=npz_path)
    print(f"\n  Saved to: {track_npz}")


if __name__ == "__main__":
    main()
