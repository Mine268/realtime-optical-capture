"""Run track mode with latest fixes and compare against fit baseline."""
from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

from roc.mocap.retarget import RetargetConfig, RetargetMode
from roc.mocap.track import RealtimeSmplxTracker

# Load data
mocap_npz = Path("sessions/mocap_live_rkw_20260603_test01/mocap_live_rkw_20260603_test01.npz")
data = np.load(mocap_npz, allow_pickle=True)
points_3d_raw = data["points_3d_raw"]

config = RetargetConfig(
    mode=RetargetMode.TRACK,
    device="cpu",
    model_dir=Path("models/smplx"),
    profile=True,
)
config.track_pose_steps = 18
config.track_recovery_pose_steps = 60

output_dir = Path("sessions/mocap_live_rkw_20260603_test01/smplx_track_priors")
output_dir.mkdir(parents=True, exist_ok=True)

tracker = RealtimeSmplxTracker(config, output_dir)

n_frames = min(300, points_3d_raw.shape[0])
print(f"Running track on {n_frames} frames...")
t0 = time.perf_counter()

for i in range(n_frames):
    result = tracker.update(i, points_3d_raw[i])
    if i % 50 == 0:
        err = result.get("body_mean_error_m", float("nan"))
        print(f"  frame {i}: body_err={err:.4f}m")

elapsed = time.perf_counter() - t0
fps = n_frames / elapsed
print(f"\nDone: {n_frames} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

npz_path = tracker.save(source_npz=mocap_npz)
print(f"Saved: {npz_path}")

# Compare with fit
print("\n--- Comparison with fit baseline ---")
track_data = np.load(npz_path, allow_pickle=True)
fit_data = np.load(
    "sessions/mocap_live_rkw_20260603_test01/smplx_retarget/smplx_fit_sequence.npz",
    allow_pickle=True,
)

track_bp = track_data["body_pose"]
fit_bp = fit_data["body_pose"]
if track_bp.ndim == 3:
    track_bp = track_bp[:, 0, :]
if fit_bp.ndim == 3:
    fit_bp = fit_bp[:, 0, :]

min_f = min(len(track_bp), len(fit_bp))

names = {
    0: "left_hip", 1: "right_hip", 2: "spine1", 3: "left_knee", 4: "right_knee",
    5: "spine2", 6: "left_ankle", 7: "right_ankle", 8: "spine3", 9: "left_foot",
    10: "right_foot", 11: "neck", 12: "left_collar", 13: "right_collar", 14: "head",
    15: "left_shoulder", 16: "right_shoulder", 17: "left_elbow", 18: "right_elbow",
    19: "left_wrist", 20: "right_wrist",
}

print(f'\n{"Joint":<20s} {"Track":>8s} {"Fit":>8s} {"Ratio":>8s} {"Note"}')
print("-" * 60)
for j in range(21):
    tn = np.mean(np.linalg.norm(track_bp[:min_f, j*3:(j+1)*3], axis=1))
    fn = np.mean(np.linalg.norm(fit_bp[:min_f, j*3:(j+1)*3], axis=1))
    ratio = tn / fn if fn > 0.01 else float("inf")
    note = ""
    if ratio < 0.3:
        note = "*** TOO LOW"
    elif ratio < 0.5:
        note = "* low"
    elif ratio > 2.0:
        note = "! HIGH"
    elif 0.7 <= ratio <= 1.4:
        note = "OK"
    print(f"{names.get(j, str(j)):<20s} {tn:8.4f} {fn:8.4f} {ratio:8.3f}  {note}")

# Per-joint error
track_joints = track_data["smplx_joints"]
fit_joints = fit_data["smplx_joints"]
if track_joints.ndim == 4:
    track_joints = track_joints[:, 0, :, :]
if fit_joints.ndim == 4:
    fit_joints = fit_joints[:, 0, :, :]

input_scale = float(fit_data["input_scale"].reshape(())) if "input_scale" in fit_data.files else 0.001
from smplx.joint_names import JOINT_NAMES

ti = track_data["frame_indices"][:min_f]
fi = fit_data["frame_indices"][:min_f]
common = np.intersect1d(ti, fi)

if len(common) > 0:
    t_map = {f: i for i, f in enumerate(ti)}
    f_map = {f: i for i, f in enumerate(fi)}
    errors = []
    for f in common:
        tj = track_joints[t_map[f]]
        fj = fit_joints[f_map[f]]
        n = min(22, min(len(tj), len(fj)))
        err = np.linalg.norm(tj[:n] - fj[:n], axis=1) / input_scale
        errors.append(err)
    errors = np.array(errors)
    mean_err = np.mean(errors, axis=0)
    print(f"\nPer-joint error vs fit (mm, {len(common)} frames):")
    for i in range(min(22, len(mean_err))):
        jname = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"j{i}"
        flag = " ***" if mean_err[i] > 90 else ""
        print(f"  {jname:<20s}: {mean_err[i]:6.1f} mm{flag}")
    overall = np.mean(mean_err[:22])
    print(f"  {'Overall':<20s}: {overall:6.1f} mm")
