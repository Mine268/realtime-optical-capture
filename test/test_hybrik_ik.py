"""Test twist-and-swing IK solver on real triangulated 3D data."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import torch
import smplx

from roc.mocap.hybrik_ik import twist_and_swing_ik


def _map_mediapipe_to_smplx_body(points_3d: np.ndarray) -> np.ndarray:
    """Map 75 MediaPipe 3D landmarks to 22 SMPL-X body joint positions.

    points_3d: (F, 75, 3) in mm (ROC format: 33 pose + 21 left_hand + 21 right_hand)
    Returns: (F, 22, 3) in meters

    MediaPipe pose landmark indices (33 landmarks):
      0:nose  1:left_eye_inner  2:left_eye  3:left_eye_outer  4:right_eye_inner
      5:right_eye  6:right_eye_outer  7:left_ear  8:right_ear  9:mouth_left
      10:mouth_right  11:left_shoulder  12:right_shoulder  13:left_elbow
      14:right_elbow  15:left_wrist  16:right_wrist  17:left_pinky  18:right_pinky
      19:left_index  20:right_index  21:left_thumb  22:right_thumb  23:left_hip
      24:right_hip  25:left_knee  26:right_knee  27:left_ankle  28:right_ankle
      29:left_heel  30:right_heel  31:left_foot_index  32:right_foot_index

    SMPL-X 22 body joints:
      0:pelvis  1:left_hip  2:right_hip  3:spine1  4:left_knee  5:right_knee
      6:spine2  7:left_ankle  8:right_ankle  9:spine3  10:left_foot  11:right_foot
      12:neck  13:left_collar  14:right_collar  15:head  16:left_shoulder
      17:right_shoulder  18:left_elbow  19:right_elbow  20:left_wrist  21:right_wrist
    """
    # MediaPipe indices
    MP = {
        "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
        "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
        "left_heel": 29, "right_heel": 30, "left_foot_index": 31, "right_foot_index": 32,
    }

    F = points_3d.shape[0]
    # Convert mm → meters
    pts = points_3d.astype(np.float32) * 0.001  # (F, 75, 3)

    # Initialize SMPL-X body joints with NaN
    smpl_joints = np.full((F, 22, 3), np.nan, dtype=np.float32)

    # Direct mappings
    direct_map = [
        (1, MP["left_hip"]), (2, MP["right_hip"]),
        (4, MP["left_knee"]), (5, MP["right_knee"]),
        (7, MP["left_ankle"]), (8, MP["right_ankle"]),
        (11, MP["right_foot_index"]),  # approximate right_foot from foot_index
        (16, MP["left_shoulder"]), (17, MP["right_shoulder"]),
        (18, MP["left_elbow"]), (19, MP["right_elbow"]),
        (20, MP["left_wrist"]), (21, MP["right_wrist"]),
    ]
    for smpl_idx, mp_idx in direct_map:
        smpl_joints[:, smpl_idx] = pts[:, mp_idx]

    # Left foot from ankle+heel+foot_index average
    left_foot = (pts[:, MP["left_ankle"]] + pts[:, MP["left_heel"]] + pts[:, MP["left_foot_index"]]) / 3.0
    smpl_joints[:, 10] = left_foot

    # Pelvis: midpoint of hips
    smpl_joints[:, 0] = (pts[:, MP["left_hip"]] + pts[:, MP["right_hip"]]) / 2.0

    # Spine chain: linearly interpolate between pelvis and neck
    neck = (
        pts[:, MP["left_shoulder"]] + pts[:, MP["right_shoulder"]]
    ) / 2.0  # neck ≈ shoulder midpoint
    # spine3 ≈ 75% from pelvis toward neck
    smpl_joints[:, 9] = smpl_joints[:, 0] + 0.75 * (neck - smpl_joints[:, 0])
    # spine2 ≈ 50%
    smpl_joints[:, 6] = smpl_joints[:, 0] + 0.50 * (neck - smpl_joints[:, 0])
    # spine1 ≈ 25%
    smpl_joints[:, 3] = smpl_joints[:, 0] + 0.25 * (neck - smpl_joints[:, 0])
    # neck from shoulder midpoint
    smpl_joints[:, 12] = neck

    # Collars: offset from spine3 toward respective shoulders
    collar_offset = 0.10  # 10cm lateral offset
    smpl_joints[:, 13] = smpl_joints[:, 9].copy()  # left_collar
    smpl_joints[:, 14] = smpl_joints[:, 9].copy()  # right_collar

    # Head: from nose, ears midpoint
    head_points = np.stack([pts[:, MP["nose"]], pts[:, MP["left_ear"]], pts[:, MP["right_ear"]]], axis=1)
    with np.errstate(all="ignore"):
        smpl_joints[:, 15] = np.nanmean(head_points, axis=1)

    # Fill remaining NaN with zeros
    smpl_joints = np.where(np.isfinite(smpl_joints), smpl_joints, 0.0)

    return smpl_joints


def main() -> None:
    npz_path = Path("sessions/mocap_live_rkw_20260603_test01/mocap_live_rkw_20260603_test01.npz")
    data = np.load(npz_path, allow_pickle=True)
    points_3d = data["points_3d"]  # (300, 75, 3) mm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load SMPL-X model
    model = smplx.create("models/smplx", model_type="smplx", gender="neutral")
    model.to(device)
    model.eval()

    # Map MediaPipe → SMPL-X joints
    smpl_body_joints = _map_mediapipe_to_smplx_body(points_3d)  # (300, 22, 3) meters
    print(f"Mapped joints shape: {smpl_body_joints.shape}")
    print(f"Any NaN: {np.any(np.isnan(smpl_body_joints))}")

    # Test single frame
    batch = torch.from_numpy(smpl_body_joints[0:1]).float().to(device)
    print(f"\nSingle-frame test:")
    t0 = time.perf_counter()
    with torch.no_grad():
        bp, go, tr = twist_and_swing_ik(batch, model)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  Time: {elapsed * 1000:.2f}ms")
    print(f"  body_pose: {bp.shape}, NaN={torch.isnan(bp).any().item()}")
    print(f"  global_orient: {go.shape}, NaN={torch.isnan(go).any().item()}")
    print(f"  transl: {tr.shape}, NaN={torch.isnan(tr).any().item()}")
    print(f"  body_pose range: [{bp.min().item():.4f}, {bp.max().item():.4f}]")

    # Verify: forward SMPL-X with computed params and compare joint positions
    with torch.no_grad():
        output = model(
            betas=torch.zeros(1, 10, device=device),
            body_pose=bp,
            global_orient=go,
            transl=tr,
            return_verts=False,
        )
    recon_joints = output.joints[:, :22]  # First 22 body joints
    recon_error = torch.norm(recon_joints - batch, dim=2)
    joint_names = [
        "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee",
        "spine2", "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot",
        "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
        "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    ]
    print(f"\n  Reconstruction error (mm):")
    for j, name in enumerate(joint_names):
        print(f"    {name:<15}: {recon_error[0, j].item() * 1000:.1f}mm")
    print(f"  Mean: {recon_error.mean().item() * 1000:.1f}mm")

    # Benchmark 50 frames
    print(f"\nBenchmark 50 frames:")
    batch = torch.from_numpy(smpl_body_joints[:50]).float().to(device)
    times = []
    with torch.no_grad():
        for i in range(50):
            t0 = time.perf_counter()
            twist_and_swing_ik(batch[i:i + 1], model)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times = np.array(times)
    print(f"  mean: {np.mean(times) * 1000:.1f}ms, p50: {np.percentile(times, 50) * 1000:.1f}ms")
    print(f"  FPS: {1.0 / np.mean(times):.1f}")


if __name__ == "__main__":
    main()
