"""Compare track vs fit per-joint errors on shared frames."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from roc.mocap.retarget import (
    RetargetConfig, RetargetMode, run_mocap_retarget,
    _ensure_retarget_dependencies, _load_reference_fitter,
)
from roc.mocap.track import RealtimeSmplxTracker


def main() -> None:
    npz_path = Path("sessions/mocap_live_rkw_20260603_test01/mocap_live_rkw_20260603_test01.npz")
    output_dir = Path("sessions/mocap_live_rkw_20260603_test01/smplx_compare")
    frames = 50  # compare first 50 frames

    data = np.load(npz_path, allow_pickle=True)
    points_3d = data["points_3d"]

    # ---- Track ----
    print("=== Running Track ===")
    track_config = RetargetConfig(
        model_dir=Path("models/smplx"),
        mode=RetargetMode.TRACK,
        output_dir=output_dir / "track",
        device="cuda",
        track_pose_steps=18,
        track_temporal_weight=0.20,
        track_velocity_weight=0.02,
        track_acceleration_weight=0.004,
        track_recovery_pose_steps=60,
    )
    tracker = RealtimeSmplxTracker(track_config, output_dir / "track")
    for i in range(min(frames, points_3d.shape[0])):
        tracker.update(i, points_3d[i])
    track_npz = tracker.save(source_npz=npz_path)
    track_data = np.load(track_npz, allow_pickle=True)
    print(f"Track OK: {track_npz}")

    # ---- Fit (batch, first {frames} frames) ----
    print(f"\n=== Running Fit ({frames} frames) ===")
    fit_config = RetargetConfig(
        model_dir=Path("models/smplx"),
        mode=RetargetMode.FIT,
        output_dir=output_dir / "fit",
        device="cuda",
        pose_steps=120,
        betas_steps=80,
        lower_body_refine=True,
        optimize_hands=False,
        use_vposer=False,
        max_frames=frames,
        profile=True,
    )
    fit_mocap_session = output_dir / "fit_session"
    fit_mocap_session.mkdir(parents=True, exist_ok=True)
    fit_npz = run_mocap_retarget(npz_path, fit_mocap_session, fit_config)
    fit_data = np.load(fit_npz, allow_pickle=True)
    print(f"Fit OK: {fit_npz}")

    # ---- Compare per-joint ----
    print(f"\n=== Per-joint comparison ===")

    # Load SMPL-X joint name mapping
    _ensure_retarget_dependencies(use_vposer=False)
    fitter = _load_reference_fitter()
    joint_names = fitter.get_joint_name_to_index()
    idx_to_name = {v: k for k, v in joint_names.items()}

    # Get smplx_joints from both
    track_joints = np.asarray(track_data["smplx_joints"], dtype=np.float32)
    fit_joints = np.asarray(fit_data["smplx_joints"], dtype=np.float32)

    # Squeeze extra dims
    if track_joints.ndim == 4 and track_joints.shape[1] == 1:
        track_joints = track_joints[:, 0, :, :]
    if fit_joints.ndim == 4 and fit_joints.shape[1] == 1:
        fit_joints = fit_joints[:, 0, :, :]

    # Align frame counts
    track_frames = track_joints.shape[0]
    fit_frames = fit_joints.shape[0]
    n_common = min(track_frames, fit_frames)

    # Get track frame indices to align with fit
    track_fi = np.asarray(track_data["frame_indices"], dtype=np.int32).reshape(-1)
    fit_fi = np.asarray(fit_data["frame_indices"], dtype=np.int32).reshape(-1)
    print(f"Track frames: {track_fi[:5]}...{track_fi[-5:]} ({len(track_fi)} total)")
    print(f"Fit frames: {fit_fi[:5]}...{fit_fi[-5:]} ({len(fit_fi)} total)")

    # Build track frame index -> track array index map
    track_fi_to_idx = {int(fi): ti for ti, fi in enumerate(track_fi)}

    n_joints = track_joints.shape[-2]
    print(f"Track joints shape: {track_joints.shape}, Fit joints shape: {fit_joints.shape}")

    joint_errors = {j: [] for j in range(n_joints)}

    input_scale = float(np.asarray(track_data["input_scale"]).reshape(()))
    compared = 0
    for fit_idx in range(fit_frames):
        fi = int(fit_fi[fit_idx])
        if fi not in track_fi_to_idx:
            continue
        ti = track_fi_to_idx[fi]
        for j in range(n_joints):
            tj = track_joints[ti, j] / input_scale
            fj = fit_joints[fit_idx, j] / input_scale
            if np.all(np.isfinite(tj)) and np.all(np.isfinite(fj)):
                err = np.linalg.norm(tj - fj)
                joint_errors[j].append(err)
        compared += 1
    print(f"Compared {compared} frames across {n_joints} joints")

    # Report top joints by mean error
    joint_stats = []
    for j in range(n_joints):
        errs = np.array(joint_errors[j])
        if len(errs) > 0:
            joint_stats.append((j, idx_to_name.get(j, f"joint_{j}"), np.mean(errs), np.median(errs), np.percentile(errs, 90)))

    # Filter: body-only joints (exclude hand/finger/wrist joints)
    hand_patterns = ["thumb", "index", "middle", "ring", "pinky", "wrist"]
    body_joints = []
    for j, name, mean_e, p50_e, p90_e in joint_stats:
        if not any(p in name for p in hand_patterns):
            body_joints.append((j, name, mean_e, p50_e, p90_e))

    body_joints.sort(key=lambda x: -x[2])
    print(f"\nTop 20 BODY joints with largest track-vs-fit mean error (mm):")
    print(f"{'Joint':<30} {'Mean':>7} {'p50':>7} {'p90':>7}")
    print("-" * 55)
    for j, name, mean_e, p50_e, p90_e in body_joints[:20]:
        print(f"{name:<30} {mean_e:7.1f} {p50_e:7.1f} {p90_e:7.1f}")

    # Body-only overall
    body_errs = []
    for j, _, _, _, _ in body_joints:
        body_errs.extend(joint_errors[j])
    body_errs = np.array(body_errs)
    print(f"\nBody-only track-vs-fit: mean={np.mean(body_errs):.1f}mm, p50={np.percentile(body_errs,50):.1f}mm, p90={np.percentile(body_errs,90):.1f}mm")


if __name__ == "__main__":
    main()
