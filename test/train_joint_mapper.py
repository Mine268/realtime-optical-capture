"""Train MLP mapper: MediaPipe 3D keypoints → SMPL-X body joint positions.

Collects all available (mocap_npz, fit_npz) pairs across sessions, aligns
frames, and trains a tiny MLP to map surface landmarks to internal SMPL-X joints.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


class JointMapper(nn.Module):
    """MLP: 75×3 MediaPipe → 22×3 SMPL-X body joint positions (pelvis-relative, meters)."""

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
        """mp_pts: (B, 75, 3) in meters, pelvis-centered. Returns (B, 22, 3) in meters."""
        B = mp_pts.shape[0]
        x = mp_pts.reshape(B, -1)
        return self.net(x).reshape(B, 22, 3)


def _fill_nan_landmarks(pts: np.ndarray) -> np.ndarray:
    """Fill NaN landmarks with per-frame mean of valid landmarks."""
    filled = pts.copy()
    for f in range(len(filled)):
        valid = np.isfinite(filled[f]).all(axis=1)
        if valid.any():
            vm = filled[f, valid].mean(axis=0)
            for l in range(pts.shape[1]):
                if not np.isfinite(filled[f, l]).all():
                    filled[f, l] = vm
        else:
            filled[f] = 0.0
    return filled


def _load_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Collect all (MediaPipe, SMPL-X body joint) pairs from sessions."""
    sessions_dir = Path("sessions")
    X_list, Y_list = [], []

    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        # Find mocap NPZ
        mocap_npz = next(session_dir.glob("*.npz"), None)
        if mocap_npz is None or "mocap_" not in mocap_npz.name:
            continue

        # Find fit NPZ (highest quality: smplx_compare/fit or smplx_retarget)
        fit_candidates = [
            session_dir / "smplx_compare" / "fit" / "smplx_fit_sequence.npz",
            session_dir / "smplx_retarget" / "smplx_fit_sequence.npz",
        ]
        fit_npz = None
        for candidate in fit_candidates:
            if candidate.is_file():
                fit_npz = candidate
                break
        if fit_npz is None:
            continue

        try:
            mp_data = np.load(mocap_npz, allow_pickle=True)
            fit_data = np.load(fit_npz, allow_pickle=True)
        except Exception:
            continue

        mp_pts = np.asarray(mp_data["points_3d"], dtype=np.float32)
        fit_joints = np.asarray(fit_data["smplx_joints"], dtype=np.float32)
        fit_frames = np.asarray(fit_data["frame_indices"], dtype=np.int32).reshape(-1)
        input_scale = float(np.asarray(fit_data["input_scale"]).reshape(()))

        if fit_joints.ndim == 4 and fit_joints.shape[1] == 1:
            fit_joints = fit_joints[:, 0, :, :]

        if np.any(~np.isfinite(fit_joints)):
            continue  # skip sessions with NaN in fit output

        # Align by frame indices
        n_common = min(len(mp_pts), fit_frames.max() + 1)
        fit_frames_clipped = fit_frames[fit_frames < n_common]

        # Get aligned data
        mp_aligned = _fill_nan_landmarks(mp_pts[fit_frames_clipped].astype(np.float32))
        fit_body = fit_joints[:len(fit_frames_clipped), :22].astype(np.float32) / input_scale  # back to mm

        # Center at pelvis
        mp_pelvis = (mp_aligned[:, 23] + mp_aligned[:, 24]) / 2.0  # mid-hip ≈ pelvis
        fit_pelvis = fit_body[:, 0]

        mp_centered = mp_aligned - mp_pelvis[:, None, :]
        fit_centered = fit_body - fit_pelvis[:, None, :]

        # Convert to meters
        X_list.append(mp_centered * 0.001)
        Y_list.append(fit_centered * 0.001)

        print(f"  {session_dir.name}: {len(fit_frames_clipped)} frames from {fit_npz.relative_to(session_dir)}")

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    print(f"\n  Total: {X.shape[0]} frames")
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


def main() -> None:
    print("Collecting training data...")
    X, Y = _load_training_data()

    # Shuffle and split
    n_total = X.shape[0]
    perm = torch.randperm(n_total)
    X, Y = X[perm], Y[perm]
    n_train = int(n_total * 0.85)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]
    print(f"Training: {n_train}, Validation: {n_total - n_train}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointMapper(hidden=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)

    joint_names = [
        "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee",
        "spine2", "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot",
        "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
        "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    ]

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(800):
        model.train()
        optimizer.zero_grad()
        train_loss = nn.functional.mse_loss(model(X_train), Y_train)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(X_val), Y_val)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 200 == 0 or epoch == 799:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                errors = torch.norm(val_pred - Y_val, dim=2).mean(0) * 1000
            print(f"\n  epoch {epoch}: train={train_loss.item():.6f}, val={val_loss.item():.6f}")
            for j in range(22):
                if errors[j] > 40:
                    print(f"    {joint_names[j]:<15}: {errors[j].item():.0f}mm")
            if epoch == 799:
                print(f"    Best val: {best_val_loss:.6f}")

    # Load best
    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation
    with torch.no_grad():
        all_pred = model(X_val.to(device))
        all_errors = torch.norm(all_pred - Y_val.to(device), dim=2) * 1000

    print(f"\n{'='*50}")
    print(f"Final per-joint validation error (mm):")
    mean_err = all_errors.mean(0)
    for j in range(22):
        marker = " ***" if mean_err[j] > 50 else ""
        print(f"  {joint_names[j]:<15}: {mean_err[j].item():.0f}mm{marker}")
    print(f"  {'OVERALL':<15}: {all_errors.mean().item():.0f}mm (p50={all_errors.median().item():.0f}mm)")
    print(f"{'='*50}")

    # Save model
    save_path = Path("models/joint_mapper.pt")
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved to {save_path} ({sum(p.numel() for p in model.parameters()):,} params)")


if __name__ == "__main__":
    main()
