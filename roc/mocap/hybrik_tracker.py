"""HybrIK-based SMPL-X tracker: learned joint mapper + analytical IK.

Replaces iterative Adam optimisation with a feed-forward pipeline:
  1. MLP mapper: MediaPipe 75×3 → SMPL-X 22 body joints
  2. Twist-and-swing IK → body_pose, global_orient, transl

Runs at ~8ms/frame (122 FPS) on CUDA.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import smplx
import torch

from roc.mocap.hybrik_ik import twist_and_swing_ik
from roc.mocap.joint_mapper import JointMapper, Normalizer, fill_nan_landmarks, load_mapper
from roc.mocap.retarget import RetargetConfig
from roc.mocap.track import _save_sequence_npz, _apply_so3_smooth, _write_track_report


class HybrikTracker:
    """Feed-forward SMPL-X body tracker using learned mapping + analytical IK."""

    def __init__(
        self,
        config: RetargetConfig,
        output_dir: Path,
        mapper_checkpoint: Path | str | None = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.output_dir = output_dir

        # Resolve model path
        from roc.mocap.retarget import _resolve_smplx_model_dir
        model_dir = _resolve_smplx_model_dir(config.model_dir)

        # Load SMPL-X model
        self.device = torch.device("cuda" if torch.cuda.is_available() and
                                    config.device in ("cuda", "gpu") else "cpu")
        self.smplx_model = smplx.create(
            str(model_dir), model_type="smplx", gender="neutral",
            num_betas=10, num_pca_comps=12,
        ).to(self.device).eval()

        # Load trained joint mapper + normalizers
        self.mapper, self.x_norm, self.y_norm = load_mapper(
            mapper_checkpoint,
            xnorm_path=Path("models/joint_mapper_xnorm.pkl"),
            ynorm_path=Path("models/joint_mapper_ynorm.pkl"),
            device=self.device,
        )

        self.aggregate: list[dict[str, np.ndarray]] = []
        self._prev_joints: np.ndarray | None = None

    def update(self, frame_index: int, points_3d: np.ndarray) -> dict[str, np.ndarray]:
        start = time.perf_counter()

        # Map MediaPipe → SMPL-X body joints (with normalization)
        from roc.mocap.joint_mapper import map_mediapipe_to_smpl
        smpl_joints_m = map_mediapipe_to_smpl(
            points_3d[None, ...], self.mapper, self.x_norm, self.y_norm, self.device,
        )

        with torch.no_grad():
            smpl_abs = torch.from_numpy(smpl_joints_m).float().to(self.device)

            # Twist-and-swing IK
            bp, go, tr = twist_and_swing_ik(smpl_abs, self.smplx_model)

            # Compute body error
            output = self.smplx_model(
                betas=torch.zeros(1, 10, device=self.device),
                body_pose=bp, global_orient=go, transl=tr,
                return_verts=False,
            )
            recon = output.joints[0, :22]
            body_err = float(torch.norm(recon - smpl_abs[0], dim=1).mean())

        # Build result in the same format as RealtimeSmplxTracker
        result = {
            "frame_index": np.array(frame_index, dtype=np.int32),
            "betas": np.zeros((1, 10), dtype=np.float32),
            "global_orient": go.cpu().numpy().astype(np.float32),
            "body_pose": bp.cpu().numpy().astype(np.float32),
            "left_hand_pose": np.zeros((1, 12), dtype=np.float32),
            "right_hand_pose": np.zeros((1, 12), dtype=np.float32),
            "transl": tr.cpu().numpy().astype(np.float32),
            "smplx_joints": output.joints.cpu().numpy().astype(np.float32),
            "overall_mean_error_m": np.array(body_err, dtype=np.float32),
            "body_mean_error_m": np.array(body_err, dtype=np.float32),
            "left_hand_mean_error_m": np.array(0.0, dtype=np.float32),
            "right_hand_mean_error_m": np.array(0.0, dtype=np.float32),
        }

        if self.config.profile:
            elapsed = time.perf_counter() - start
            result["stage_timings"] = {"hybrik_total_s": elapsed}
            if frame_index % 10 == 0:
                print(
                    f"[mocap-profile] frame={frame_index} stage=hybrik "
                    f"total={elapsed * 1000:.1f}ms body_err={body_err:.4f}m",
                    flush=True,
                )

        self.aggregate.append(result)
        self._prev_joints = smpl_abs.cpu().numpy()
        return result

    def save(self, source_npz: Path | None = None) -> Path:
        if not self.aggregate:
            raise RuntimeError("No HybrIK frames were produced")
        # Light temporal smoothing on body_pose (weaker than track mode since
        # the IK output is already relatively stable)
        _apply_so3_smooth(self.aggregate, sigma=0.15)
        self._refresh_joints()
        sequence_path = self.output_dir / "smplx_fit_sequence.npz"
        _save_sequence_npz(sequence_path, self.aggregate, self.config, source_npz=source_npz)
        _write_track_report(self.output_dir, source_npz, sequence_path, self.aggregate, self.config)
        return sequence_path

    def _refresh_joints(self) -> None:
        """Recompute smplx_joints from smoothed rotations."""
        zh = torch.zeros(1, 12, device=self.device)
        with torch.no_grad():
            for item in self.aggregate:
                bp = torch.tensor(item["body_pose"], dtype=torch.float32, device=self.device)
                go = torch.tensor(item["global_orient"], dtype=torch.float32, device=self.device)
                tr = torch.tensor(item["transl"], dtype=torch.float32, device=self.device)
                out = self.smplx_model(
                    betas=torch.zeros(1, 10, device=self.device),
                    body_pose=bp, global_orient=go, transl=tr,
                    left_hand_pose=zh, right_hand_pose=zh,
                    return_verts=False,
                )
                item["smplx_joints"] = out.joints.cpu().numpy().astype(np.float32)
