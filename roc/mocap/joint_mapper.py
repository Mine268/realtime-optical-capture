"""Learned MLP mapper: MediaPipe 75×3 → SMPL-X 22 body joints."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class JointMapper(nn.Module):
    """MLP mapping 75 MediaPipe 3D landmarks → 22 SMPL-X body joint positions.

    Input: (B, 75, 3) meters, pelvis-centered
    Output: (B, 22, 3) meters, pelvis-relative
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(75 * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 22 * 3),
        )

    def forward(self, mp_pts: torch.Tensor) -> torch.Tensor:
        B = mp_pts.shape[0]
        x = mp_pts.reshape(B, -1)
        return self.net(x).reshape(B, 22, 3)


def load_mapper(checkpoint: Path | str | None = None) -> JointMapper:
    """Load the trained JointMapper."""
    if checkpoint is None:
        checkpoint = Path("models/joint_mapper.pt")
    model = JointMapper(hidden=128)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    model.eval()
    return model


def fill_nan_landmarks(points_3d: np.ndarray) -> np.ndarray:
    """Fill NaN landmarks with per-frame mean of valid landmarks."""
    filled = points_3d.astype(np.float32, copy=True)
    for f in range(len(filled)):
        valid = np.isfinite(filled[f]).all(axis=1)
        if valid.any():
            vm = filled[f, valid].mean(axis=0)
            for l in range(points_3d.shape[1]):
                if not np.isfinite(filled[f, l]).all():
                    filled[f, l] = vm
    return filled


def map_mediapipe_to_smpl(
    points_3d: np.ndarray,
    model: JointMapper,
    device: torch.device | None = None,
) -> np.ndarray:
    """Map MediaPipe 3D keypoints (F, 75, 3) mm → SMPL-X body joints (F, 22, 3) meters."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fill NaN
    filled = fill_nan_landmarks(points_3d)

    # Center at pelvis (mid-hip)
    pelvis = (filled[:, 23] + filled[:, 24]) / 2.0  # (F, 3) mm
    centered = filled - pelvis[:, None, :]  # (F, 75, 3) mm
    centered_m = centered * 0.001  # meters

    with torch.no_grad():
        x = torch.from_numpy(centered_m).float().to(device)
        smpl_joints_rel = model(x).cpu().numpy()  # (F, 22, 3) meters

    # Add pelvis back
    pelvis_m = pelvis * 0.001
    smpl_joints = smpl_joints_rel + pelvis_m[:, None, :]
    return smpl_joints.astype(np.float32)
