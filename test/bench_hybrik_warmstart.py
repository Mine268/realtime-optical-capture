"""Benchmark: HybrIK IK warm-start + N Adam refinement steps vs pure Adam."""
from __future__ import annotations

import os, time
from pathlib import Path
import numpy as np
import torch
import smplx

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import sys
sys.path.insert(0, str(Path("test").resolve()))
from test_hybrik_ik import _map_mediapipe_to_smplx_body

from roc.mocap.hybrik_ik import twist_and_swing_ik
from roc.mocap.track import RealtimeSmplxTracker
from roc.mocap.retarget import RetargetConfig, RetargetMode


def main() -> None:
    npz_path = Path("sessions/mocap_live_rkw_20260603_test01/mocap_live_rkw_20260603_test01.npz")
    data = np.load(npz_path, allow_pickle=True)
    points_3d = data["points_3d"]
    frames = min(50, points_3d.shape[0])

    device = torch.device("cuda")
    smplx_model = smplx.create("models/smplx", model_type="smplx", gender="neutral").to(device).eval()

    # Pre-map all MediaPipe joints → SMPL-X body joints
    smpl_body = _map_mediapipe_to_smplx_body(points_3d[:frames])  # (F, 22, 3)
    smpl_body_t = torch.from_numpy(smpl_body).float().to(device)
    print(f"Pre-mapped {frames} frames")

    # ---- Baseline: pure Adam track (no warm-start) ----
    print("\n=== Baseline: Pure Adam track (18 steps) ===")
    config = RetargetConfig(
        model_dir=Path("models/smplx"), mode=RetargetMode.TRACK,
        output_dir=Path("/tmp/hybrik_bench"), device="cuda",
        track_pose_steps=18, track_recovery_pose_steps=60,
        track_temporal_weight=0.20, track_velocity_weight=0.02, track_acceleration_weight=0.004,
    )
    tracker = RealtimeSmplxTracker(config, Path("/tmp/hybrik_bench/track"))
    track_times, track_errors = [], []
    for i in range(frames):
        t0 = time.perf_counter()
        result = tracker.update(i, points_3d[i])
        torch.cuda.synchronize()
        track_times.append(time.perf_counter() - t0)
        track_errors.append(float(result["body_mean_error_m"]))
    print(f"  mean: {np.mean(track_times[1:])*1000:.1f}ms, FPS: {1/np.mean(track_times[1:]):.1f}")
    print(f"  body_err: {np.mean(track_errors[1:])*1000:.1f}mm")

    # ---- HybrIK IK only (no Adam) ----
    print("\n=== HybrIK IK only (no refinement) ===")
    ik_times = []
    with torch.no_grad():
        for i in range(frames):
            t0 = time.perf_counter()
            bp, go, tr = twist_and_swing_ik(smpl_body_t[i:i+1], smplx_model)
            torch.cuda.synchronize()
            ik_times.append(time.perf_counter() - t0)
    print(f"  mean: {np.mean(ik_times)*1000:.1f}ms, FPS: {1/np.mean(ik_times):.1f}")

    # ---- HybrIK IK warm-start + N Adam steps ----
    for n_steps in [2, 3, 4, 5, 8]:
        print(f"\n=== IK warm-start + {n_steps} Adam steps ===")
        config2 = RetargetConfig(
            model_dir=Path("models/smplx"), mode=RetargetMode.TRACK,
            output_dir=Path(f"/tmp/hybrik_bench/ik{n_steps}"), device="cuda",
            track_pose_steps=n_steps, track_recovery_pose_steps=max(n_steps, 10),
            track_temporal_weight=0.20, track_velocity_weight=0.02, track_acceleration_weight=0.004,
        )
        tracker2 = RealtimeSmplxTracker(config2, Path(f"/tmp/hybrik_bench/ik{n_steps}"))

        # Patch: override the warm-start with IK result
        _orig_update = tracker2.update

        def patched_update(frame_idx, pts):
            # Run IK to get initial pose
            sj = _map_mediapipe_to_smplx_body(pts[None, ...])
            sj_t = torch.from_numpy(sj).float().to(device)
            with torch.no_grad():
                bp_ik, go_ik, tr_ik = twist_and_swing_ik(sj_t, smplx_model)
            # Override warm-start state with IK result
            tracker2._prev_bp = bp_ik.clone().detach()
            tracker2._prev_go = go_ik.clone().detach()
            tracker2._prev_tr = tr_ik.clone().detach()
            tracker2._has_prev = (frame_idx > 0)
            return _orig_update(frame_idx, pts)

        tracker2.update = patched_update

        times, errors = [], []
        for i in range(frames):
            t0 = time.perf_counter()
            result = tracker2.update(i, points_3d[i])
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            errors.append(float(result["body_mean_error_m"]))
        steady_t = np.array(times[1:])  # skip first frame (recovery)
        steady_e = np.array(errors[1:])
        ik_overhead = np.mean(ik_times) * 1000
        total = np.mean(steady_t) * 1000
        adam_only = total - ik_overhead
        print(f"  total: {total:.1f}ms = IK({ik_overhead:.1f}ms) + Adam({adam_only:.1f}ms)")
        print(f"  FPS: {1/np.mean(steady_t):.1f}")
        print(f"  body_err: {np.mean(steady_e)*1000:.1f}mm")


if __name__ == "__main__":
    main()
