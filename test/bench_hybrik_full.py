"""Full evaluation: HybrIK tracker vs Adam track vs fit ground truth."""
from __future__ import annotations

import os, time
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from roc.mocap.hybrik_tracker import HybrikTracker
from roc.mocap.track import RealtimeSmplxTracker
from roc.mocap.retarget import RetargetConfig, RetargetMode


def main() -> None:
    session = Path("sessions/mocap_live_rkw_20260603_test01")
    npz_path = session / "mocap_live_rkw_20260603_test01.npz"
    data = np.load(npz_path, allow_pickle=True)
    points_3d = data["points_3d"]
    frames = points_3d.shape[0]

    # ---- Load fit ground truth ----
    fit_npz = session / "smplx_retarget" / "smplx_fit_sequence.npz"
    fit_data = np.load(fit_npz, allow_pickle=True)
    fit_joints = np.asarray(fit_data["smplx_joints"], dtype=np.float32)
    if fit_joints.ndim == 4 and fit_joints.shape[1] == 1:
        fit_joints = fit_joints[:, 0, :, :]
    fit_frames = np.asarray(fit_data["frame_indices"], dtype=np.int32).reshape(-1)
    input_scale = float(np.asarray(fit_data["input_scale"]).reshape(()))
    fit_body = fit_joints[:, :22] / input_scale  # mm
    fit_fi_to_idx = {int(fi): i for i, fi in enumerate(fit_frames)}
    print(f"GT fit: {fit_body.shape[0]} frames, {np.sum(~np.isfinite(fit_body))} NaN")

    # ---- HybrIK Tracker ----
    print("\n=== HybrIK Tracker ===")
    hybrik_config = RetargetConfig(
        model_dir=Path("models/smplx"),
        mode=RetargetMode.TRACK,
        output_dir=session / "smplx_hybrik",
        device="cuda",
        profile=True,
    )
    hybrik = HybrikTracker(hybrik_config, session / "smplx_hybrik")

    hybrik_times = []
    for i in range(frames):
        t0 = time.perf_counter()
        result = hybrik.update(i, points_3d[i])
        torch.cuda.synchronize()
        hybrik_times.append(time.perf_counter() - t0)

    hybrik_times = np.array(hybrik_times)
    hybrik_npz = hybrik.save(source_npz=npz_path)
    print(f"  speed: {np.mean(hybrik_times)*1000:.1f}ms, FPS: {1/np.mean(hybrik_times):.1f}")
    print(f"  saved: {hybrik_npz}")

    # ---- Adam Track (baseline) ----
    print("\n=== Adam Track ===")
    track_config = RetargetConfig(
        model_dir=Path("models/smplx"),
        mode=RetargetMode.TRACK,
        output_dir=session / "smplx_track_cmp",
        device="cuda",
        track_pose_steps=18,
        track_recovery_pose_steps=60,
        track_temporal_weight=0.20,
        track_velocity_weight=0.02,
        track_acceleration_weight=0.004,
    )
    tracker = RealtimeSmplxTracker(track_config, session / "smplx_track_cmp")

    track_times = []
    for i in range(frames):
        t0 = time.perf_counter()
        tracker.update(i, points_3d[i])
        torch.cuda.synchronize()
        track_times.append(time.perf_counter() - t0)

    track_times = np.array(track_times)
    track_npz = tracker.save(source_npz=npz_path)
    print(f"  speed: {np.mean(track_times[1:])*1000:.1f}ms steady, FPS: {1/np.mean(track_times[1:]):.1f}")
    print(f"  saved: {track_npz}")

    # ---- Compare both vs fit ----
    print("\n=== Comparison vs Fit ===")

    for label, result_npz in [("HybrIK", hybrik_npz), ("Adam Track", track_npz)]:
        rdata = np.load(result_npz, allow_pickle=True)
        rjoints = np.asarray(rdata["smplx_joints"], dtype=np.float32)
        if rjoints.ndim == 4 and rjoints.shape[1] == 1:
            rjoints = rjoints[:, 0, :, :]
        rframes = np.asarray(rdata["frame_indices"], dtype=np.int32).reshape(-1)
        ris = np.asarray(rdata["input_scale"], dtype=np.float32).reshape(())
        rbody = rjoints[:, :22] / ris  # mm

        errors = []
        joint_errors = {j: [] for j in range(22)}
        joint_names = [
            "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee",
            "spine2", "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot",
            "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
            "l_elbow", "r_elbow", "l_wrist", "r_wrist",
        ]

        for ti, fi in enumerate(rframes):
            if fi not in fit_fi_to_idx:
                continue
            ffi = fit_fi_to_idx[fi]
            for j in range(22):
                rp = rbody[ti, j]
                fp = fit_body[ffi, j]
                if np.all(np.isfinite(rp)) and np.all(np.isfinite(fp)):
                    err = np.linalg.norm(rp - fp)
                    joint_errors[j].append(err)
                    errors.append(err)

        errors = np.array(errors)
        print(f"\n  {label}:")
        print(f"    body mean={np.mean(errors):.0f}mm, p50={np.percentile(errors,50):.0f}mm, p90={np.percentile(errors,90):.0f}mm")

        # Show worst joints
        joint_means = [(np.mean(v), name) for name, v in zip(joint_names, [joint_errors[j] for j in range(22)])]
        joint_means.sort(key=lambda x: -x[0])
        print(f"    Worst joints:")
        for err, name in joint_means[:8]:
            print(f"      {name:<15}: {err:.0f}mm")
        print(f"    Best joints:")
        for err, name in joint_means[-5:]:
            print(f"      {name:<15}: {err:.0f}mm")


if __name__ == "__main__":
    main()
