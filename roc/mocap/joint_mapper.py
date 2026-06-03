"""Learned MLP mapper: MediaPipe 75×3 → SMPL-X 22 body joints."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Normalizer:
    """Per-coordinate mean/std normalization."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        n = cls()
        n.mean = d["mean"]
        n.std = d["std"]
        return n


class JointMapper(nn.Module):
    """MLP mapping 75 MediaPipe 3D landmarks → 22 SMPL-X body joint positions.

    Input: (B, 75, 3) meters, pelvis-centered, NORMALIZED
    Output: (B, 22, 3) meters, pelvis-relative, DENORMALIZED
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.net(x.reshape(B, -1)).reshape(B, 22, 3)


def load_mapper(
    checkpoint: Path | str | None = None,
    xnorm_path: Path | str | None = None,
    ynorm_path: Path | str | None = None,
    device: torch.device | None = None,
) -> tuple[JointMapper, Normalizer, Normalizer]:
    """Load the trained JointMapper and its normalizers."""
    if checkpoint is None:
        checkpoint = Path("models/joint_mapper.pt")
    if xnorm_path is None:
        xnorm_path = Path("models/joint_mapper_xnorm.pkl")
    if ynorm_path is None:
        ynorm_path = Path("models/joint_mapper_ynorm.pkl")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = JointMapper(hidden=128)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    model.to(device).eval()

    x_norm = Normalizer.load(Path(xnorm_path))
    y_norm = Normalizer.load(Path(ynorm_path))
    return model, x_norm, y_norm


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
    x_norm: Normalizer,
    y_norm: Normalizer,
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

    # Normalize → MLP → denormalize
    F = centered.shape[0]
    flat = centered.reshape(F, -1)  # (F, 225)
    flat_norm = x_norm.normalize(flat)

    with torch.no_grad():
        x = torch.from_numpy(flat_norm).float().to(device)
        pred_norm = model(x).cpu().numpy()  # (F, 22, 3)

    pred_flat = y_norm.denormalize(pred_norm.reshape(F, -1))  # (F, 66) mm
    smpl_rel_mm = pred_flat.reshape(F, 22, 3)  # (F, 22, 3) mm

    # Add pelvis back, convert to meters
    smpl_joints_m = (smpl_rel_mm + pelvis[:, None, :]) * 0.001
    return smpl_joints_m.astype(np.float32)
