"""Train MLP mapper: MediaPipe 3D → SMPL-X body joints, with proper normalization."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


class JointMapper(nn.Module):
    """MLP: 75×3 MediaPipe → 22×3 SMPL-X body joints (normalized I/O)."""

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


class Normalizer:
    """Per-coordinate mean/std normalization with persistence."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        """data: (N, D)"""
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std = data.std(axis=0).astype(np.float32) + 1e-8

    def normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        n = cls()
        n.mean = d["mean"]
        n.std = d["std"]
        return n


def _fill_nan_landmarks(pts: np.ndarray) -> np.ndarray:
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


def _collect_raw_data() -> tuple[np.ndarray, np.ndarray]:
    """Collect all (MediaPipe_raw, SMPL_body) pairs in mm, pelvis-centered."""
    sessions_dir = Path("sessions")
    X_list, Y_list = [], []

    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        mocap_npz = next(session_dir.glob("*.npz"), None)
        if mocap_npz is None or "mocap_" not in mocap_npz.name:
            continue

        fit_candidates = [
            session_dir / "smplx_compare" / "fit" / "smplx_fit_sequence.npz",
            session_dir / "smplx_retarget" / "smplx_fit_sequence.npz",
        ]
        fit_npz = None
        for c in fit_candidates:
            if c.is_file():
                fit_npz = c
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
            continue

        n_common = min(len(mp_pts), fit_frames.max() + 1)
        fit_frames_clipped = fit_frames[fit_frames < n_common]

        mp_aligned = _fill_nan_landmarks(mp_pts[fit_frames_clipped])
        fit_body = fit_joints[:len(fit_frames_clipped), :22] / input_scale  # mm

        # Center both at pelvis
        mp_pelvis = (mp_aligned[:, 23] + mp_aligned[:, 24]) / 2.0
        fit_pelvis = fit_body[:, 0]
        mp_centered = mp_aligned - mp_pelvis[:, None, :]
        fit_centered = fit_body - fit_pelvis[:, None, :]

        X_list.append(mp_centered.reshape(mp_centered.shape[0], -1))
        Y_list.append(fit_centered.reshape(fit_centered.shape[0], -1))
        print(f"  {session_dir.name}: {len(fit_frames_clipped)} frames from {fit_npz.relative_to(session_dir)}")

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    print(f"\n  Total: {X.shape[0]} frames, X shape={X.shape}, Y shape={Y.shape}")
    return X, Y


def main() -> None:
    print("Collecting training data...")
    X_raw, Y_raw = _collect_raw_data()  # (N, 225), (N, 66) in mm

    # ---- Fit normalizers on training split ----
    n_total = X_raw.shape[0]
    perm = np.random.RandomState(42).permutation(n_total)
    n_train = int(n_total * 0.85)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    x_norm = Normalizer()
    y_norm = Normalizer()
    x_norm.fit(X_raw[train_idx])
    y_norm.fit(Y_raw[train_idx])

    print(f"\nInput stats: mean={x_norm.mean.mean():.1f}mm, std={x_norm.std.mean():.1f}mm")
    print(f"Output stats: mean={y_norm.mean.mean():.1f}mm, std={y_norm.std.mean():.1f}mm")

    # Normalize
    X = x_norm.normalize(X_raw)
    Y = y_norm.normalize(Y_raw)

    X_train = torch.from_numpy(X[train_idx]).float()
    Y_train = torch.from_numpy(Y[train_idx]).float()
    X_val = torch.from_numpy(X[val_idx]).float()
    Y_val = torch.from_numpy(Y[val_idx]).float()
    print(f"Training: {len(train_idx)}, Validation: {len(val_idx)}")

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
        pred = model(X_train)
        train_loss = nn.functional.mse_loss(pred.reshape(X_train.shape[0], -1), Y_train)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.functional.mse_loss(val_pred.reshape(X_val.shape[0], -1), Y_val)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 200 == 0 or epoch == 799:
            model.eval()
            with torch.no_grad():
                # Evaluate in original mm space
                val_pred_norm = model(X_val).reshape(-1, 22, 3).cpu().numpy()
                val_pred_norm_flat = val_pred_norm.reshape(len(val_pred_norm), -1)
                val_pred_mm = y_norm.denormalize(val_pred_norm_flat).reshape(-1, 22, 3)
                val_true_mm = y_norm.denormalize(Y_val.cpu().numpy()).reshape(-1, 22, 3)
                errors = np.linalg.norm(val_pred_mm - val_true_mm, axis=2).mean(0)

            print(f"\n  epoch {epoch}: train={train_loss.item():.6f}, val={val_loss.item():.6f}")
            for j in range(22):
                if errors[j] > 35:
                    print(f"    {joint_names[j]:<15}: {errors[j]:.0f}mm")
            if epoch == 799:
                print(f"    Best val: {best_val_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation
    with torch.no_grad():
        all_pred_norm = model(X_val.to(device)).reshape(-1, 22, 3).cpu().numpy()
        all_pred_norm_flat = all_pred_norm.reshape(len(all_pred_norm), -1)
        all_pred_mm = y_norm.denormalize(all_pred_norm_flat).reshape(-1, 22, 3)
        all_true_mm = y_norm.denormalize(Y_val.cpu().numpy()).reshape(-1, 22, 3)
        all_errors = np.linalg.norm(all_pred_mm - all_true_mm, axis=2)

    print(f"\n{'='*50}")
    print(f"Final per-joint validation error (mm):")
    mean_err = all_errors.mean(0)
    for j in range(22):
        marker = " ***" if mean_err[j] > 50 else ""
        print(f"  {joint_names[j]:<15}: {mean_err[j]:.0f}mm{marker}")
    print(f"  {'OVERALL':<15}: {all_errors.mean():.0f}mm (median={np.median(all_errors):.0f}mm, p90={np.percentile(all_errors,90):.0f}mm)")
    print(f"{'='*50}")

    # Save model and normalizers
    model_path = Path("models/joint_mapper.pt")
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    x_norm.save(Path("models/joint_mapper_xnorm.pkl"))
    y_norm.save(Path("models/joint_mapper_ynorm.pkl"))
    print(f"\nSaved to models/joint_mapper.pt (+ _xnorm.pkl, _ynorm.pkl)")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")


if __name__ == "__main__":
    main()
