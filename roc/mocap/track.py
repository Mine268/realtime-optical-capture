from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from roc.mocap.retarget import (
    BODY_NAMES,
    HAND_NAMES,
    RetargetConfig,
    _ensure_retarget_dependencies,
    _load_reference_fitter,
    _resolve_smplx_model_dir,
    _save_sequence_npz,
    _write_report,
)


class RealtimeSmplxTracker:
    """Body-only SMPL-X tracker using vectorised Adam optimisation."""

    def __init__(self, config: RetargetConfig, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = _resolve_smplx_model_dir(config.model_dir)
        if not model_dir.is_dir():
            raise RuntimeError(f"SMPL-X model directory not found: {config.model_dir}")
        _ensure_retarget_dependencies(use_vposer=False)

        self.config = config
        self.output_dir = output_dir
        self.fitter = _load_reference_fitter()
        self.T = self.fitter.torch
        self.device = self.T.device(
            "cuda" if config.device == "cuda" and self.T.cuda.is_available() else "cpu"
        )
        self.joint_name_to_idx = self.fitter.get_joint_name_to_index()

        self.model = self.fitter.create_model(
            _resolve_smplx_model_dir(config.model_dir), "neutral", 10, 12, self.device,
        )
        self.shared_betas = self.T.zeros((1, 10), dtype=self.T.float32, device=self.device)

        # Pre-build index tensors
        body_smplx_map = getattr(self.fitter, "BODY_SMPLX_MAP", {})
        body_idx = {name: i for i, name in enumerate(BODY_NAMES)}
        src_l, tgt_l = [], []
        for bn, sn in body_smplx_map.items():
            if bn in body_idx and sn in self.joint_name_to_idx:
                src_l.append(self.joint_name_to_idx[sn])
                tgt_l.append(body_idx[bn])
        self._src = self.T.tensor(src_l, device=self.device, dtype=self.T.long)
        self._tgt_idx = self.T.tensor(tgt_l, device=self.device, dtype=self.T.long)

        # Warm-start state on GPU
        self._prev_bp = self.T.zeros(1, 63, device=self.device)
        self._prev_go = self.T.zeros(1, 3, device=self.device)
        self._prev_tr = self.T.zeros(1, 3, device=self.device)
        self._has_prev = False

        self.aggregate: list[dict[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    def update(self, frame_index: int, points_3d: np.ndarray) -> dict[str, np.ndarray]:
        T = self.T
        start = time.perf_counter()
        scaled = points_3d.astype(np.float32) * np.float32(self.config.input_scale)

        # Build target
        target_full = T.from_numpy(scaled).to(self.device)
        target_23 = target_full[self._tgt_idx]
        valid_23 = T.isfinite(target_23).all(dim=1)
        n_valid = int(valid_23.sum())
        if n_valid < 3:
            return self._empty_result(frame_index)

        # Fresh parameters from warm-start
        bp = self._prev_bp.clone().detach().requires_grad_(True)
        go = self._prev_go.clone().detach().requires_grad_(True)
        tr = self._prev_tr.clone().detach().requires_grad_(True)
        p_bp = self._prev_bp.detach().clone()
        p_go = self._prev_go.detach().clone()
        p_tr = self._prev_tr.detach().clone()
        has_prev = self._has_prev
        zh = T.zeros(1, 12, device=self.device)

        _src = self._src
        _valid = valid_23
        _tgt = target_23
        _model = self.model
        _betas = self.shared_betas
        tw = self.config.track_temporal_weight

        # Adam with pose prior to prevent unnatural joint angles
        n_steps = 55 if not has_prev else 18
        adapter_elapsed = time.perf_counter() - start

        optimizer = T.optim.Adam([
            {"params": [bp], "lr": 0.08},
            {"params": [go], "lr": 0.05},
            {"params": [tr], "lr": 0.03},
        ])

        wp = 0.10  # pose prior: L2 on body_pose to stay near rest pose
        wk = 0.005  # light knee penalty to avoid hyperextension
        ws = 0.03   # spine penalty to maintain torso length

        T.cuda.synchronize()
        opt_start = time.perf_counter()
        for _ in range(n_steps):
            optimizer.zero_grad()
            out = _model(
                betas=_betas, body_pose=bp, global_orient=go, transl=tr,
                left_hand_pose=zh, right_hand_pose=zh,
            )
            pred_pts = out.joints[0, _src]
            diffs = pred_pts[_valid] - _tgt[_valid]
            loss = (diffs * diffs).sum() / n_valid

            # Pose prior: keeps all joints near rest pose
            loss = loss + wp * T.mean(bp ** 2)

            # Knee: light penalty (joints 3=l_knee@9:12, 4=r_knee@12:15)
            loss = loss + wk * (T.mean(bp[:, 9:12] ** 2) + T.mean(bp[:, 12:15] ** 2))

            # Spine: maintain torso height (spine1@6:9, spine2@15:18, spine3@24:27)
            loss = loss + ws * (
                T.mean(bp[:, 6:9] ** 2) + T.mean(bp[:, 15:18] ** 2) + T.mean(bp[:, 24:27] ** 2)
            )

            if has_prev:
                loss = loss + tw * (
                    T.mean((bp - p_bp) ** 2)
                    + 0.5 * T.mean((go - p_go) ** 2)
                    + 0.3 * T.mean((tr - p_tr) ** 2)
                )
            loss.backward()
            optimizer.step()
        T.cuda.synchronize()
        opt_elapsed = time.perf_counter() - opt_start

        with T.no_grad():
            out = _model(
                betas=_betas, body_pose=bp, global_orient=go, transl=tr,
                left_hand_pose=zh, right_hand_pose=zh,
            )
            joints = out.joints.cpu().numpy().copy()
            pred_pts = out.joints[0, _src]
            df = pred_pts[_valid] - _tgt[_valid]
            body_err = float(T.mean(T.norm(df, dim=1)).cpu())

        result = {
            "frame_index": np.array(frame_index, dtype=np.int32),
            "betas": _betas.cpu().numpy().copy(),
            "global_orient": go.detach().cpu().numpy().copy(),
            "body_pose": bp.detach().cpu().numpy().copy(),
            "left_hand_pose": np.zeros((1, 12), dtype=np.float32),
            "right_hand_pose": np.zeros((1, 12), dtype=np.float32),
            "transl": tr.detach().cpu().numpy().copy(),
            "smplx_joints": joints,
            "overall_mean_error_m": np.array(body_err, dtype=np.float32),
            "body_mean_error_m": np.array(body_err, dtype=np.float32),
            "left_hand_mean_error_m": np.array(0.0, dtype=np.float32),
            "right_hand_mean_error_m": np.array(0.0, dtype=np.float32),
        }

        if self.config.profile:
            result["stage_timings"] = {
                "input_adapter_s": adapter_elapsed,
                "adam_optimize_s": opt_elapsed,
                "track_update_s": time.perf_counter() - start,
            }
            if len(self.aggregate) == 0 or frame_index % max(1, self.config.profile_interval) == 0:
                print(
                    f"[mocap-profile] frame={frame_index} stage=track adam "
                    f"opt={opt_elapsed*1000:.1f}ms body_err={body_err:.4f}m",
                    flush=True,
                )

        self.aggregate.append(result)
        self._prev_bp = bp.detach().clone()
        self._prev_go = go.detach().clone()
        self._prev_tr = tr.detach().clone()
        self._has_prev = True

        return result

    def _empty_result(self, frame_index: int = 0) -> dict[str, np.ndarray]:
        return {
            "frame_index": np.array(frame_index, dtype=np.int32),
            "betas": self.shared_betas.cpu().numpy().copy(),
            "global_orient": np.zeros((1, 3), dtype=np.float32),
            "body_pose": np.zeros((1, 63), dtype=np.float32),
            "left_hand_pose": np.zeros((1, 12), dtype=np.float32),
            "right_hand_pose": np.zeros((1, 12), dtype=np.float32),
            "transl": np.zeros((1, 3), dtype=np.float32),
            "smplx_joints": np.zeros((1, 127, 3), dtype=np.float32),
            "overall_mean_error_m": np.array(np.nan, dtype=np.float32),
            "body_mean_error_m": np.array(np.nan, dtype=np.float32),
            "left_hand_mean_error_m": np.array(0.0, dtype=np.float32),
            "right_hand_mean_error_m": np.array(0.0, dtype=np.float32),
        }

    def save(self, source_npz: Path | None = None) -> Path:
        if not self.aggregate:
            raise RuntimeError("No track frames were produced")
        _apply_so3_smooth(self.aggregate, sigma=0.15)
        sequence_path = self.output_dir / "smplx_fit_sequence.npz"
        _save_sequence_npz(sequence_path, self.aggregate, self.config, source_npz=source_npz)
        _write_track_report(self.output_dir, source_npz, sequence_path, self.aggregate, self.config)
        return sequence_path


def _apply_so3_smooth(aggregate: list[dict[str, np.ndarray]], sigma: float) -> None:
    if len(aggregate) < 3:
        return
    from scipy.ndimage import gaussian_filter1d
    bp = np.stack([item["body_pose"].reshape(63) for item in aggregate], axis=0)
    go = np.stack([item["global_orient"].reshape(3) for item in aggregate], axis=0)
    tr = np.stack([item["transl"].reshape(3) for item in aggregate], axis=0)
    sbp = gaussian_filter1d(bp, sigma=sigma, axis=0)
    sgo = gaussian_filter1d(go, sigma=sigma, axis=0)
    st = gaussian_filter1d(tr, sigma=sigma, axis=0)
    for i, item in enumerate(aggregate):
        item["body_pose"] = sbp[i].reshape(1, 63).astype(np.float32)
        item["global_orient"] = sgo[i].reshape(1, 3).astype(np.float32)
        item["transl"] = st[i].reshape(1, 3).astype(np.float32)


def _apply_track_overrides(base_args: argparse.Namespace, config: RetargetConfig) -> argparse.Namespace:
    base_args.pose_steps = max(1, config.track_pose_steps)
    base_args.root_steps = config.realtime_root_steps
    base_args.lower_body_refine = False
    base_args.lower_steps = None
    base_args.hand_weight = 0.0
    base_args.hand_prior_weight = 0.0
    base_args.temporal_weight = config.track_temporal_weight
    base_args.velocity_weight = config.track_velocity_weight
    base_args.acceleration_weight = config.track_acceleration_weight
    base_args.early_stop_patience = 3
    base_args.no_mesh = not config.save_debug_assets
    base_args.no_plot = not config.save_debug_assets
    base_args.use_vposer = False
    base_args.disable_post_smooth = True
    return base_args


def _default_track_root_steps(config: RetargetConfig) -> int:
    return int(config.root_steps or max(8, config.track_pose_steps // 2))


def _write_track_report(output_dir, source_npz, sequence_path, aggregate, config) -> None:
    report = {
        "mode": "track",
        "source_npz": str(source_npz) if source_npz is not None else None,
        "output_npz": str(sequence_path),
        "frames": len(aggregate),
        "model_dir": str(config.model_dir),
        "device": config.device,
        "track_pose_steps": config.track_pose_steps,
        "track_temporal_weight": config.track_temporal_weight,
        "track_velocity_weight": config.track_velocity_weight,
        "track_acceleration_weight": config.track_acceleration_weight,
        "input_scale": config.input_scale,
        "mean_overall_error_m": float(np.mean([item["overall_mean_error_m"] for item in aggregate])),
        "mean_body_error_m": float(np.mean([item["body_mean_error_m"] for item in aggregate])),
        "mean_left_hand_error_m": float(np.mean([item["left_hand_mean_error_m"] for item in aggregate])),
        "mean_right_hand_error_m": float(np.mean([item["right_hand_mean_error_m"] for item in aggregate])),
    }
    (output_dir / "track_report.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False, allow_unicode=False) or "",
        encoding="utf-8",
    )
