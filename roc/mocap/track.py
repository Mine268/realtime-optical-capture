from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

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

# Default track hyperparameter config shipped alongside this module.
_DEFAULT_TRACK_CONFIG_PATH = Path(__file__).resolve().parent / "track_config.yaml"


def _load_track_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load track hyperparameters from a YAML file.

    Returns a dict with keys: optimizer, loss_weights, temporal, bezier,
    axis_weights, post_smooth, target_weights, target_smooth_alpha,
    body_pose_prior_weights, body_pose_temporal_weights.
    """
    resolved = Path(path) if path else _DEFAULT_TRACK_CONFIG_PATH
    if not resolved.is_file():
        raise RuntimeError(f"Track config not found: {resolved}")
    with open(resolved, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Track config must be a mapping: {resolved}")
    return cfg


class RealtimeSmplxTracker:
    """Body-only SMPL-X tracker using vectorised Adam optimisation."""

    def __init__(
        self,
        config: RetargetConfig,
        output_dir: Path,
        track_config_path: Path | str | None = None,
    ) -> None:
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

        # Load hyperparameters from YAML
        tcfg = _load_track_config(track_config_path)
        self._tcfg = tcfg

        # Compile model on GPU for ~2x speedup
        if self.device.type == "cuda" and hasattr(self.T, "compile"):
            opt_cfg = tcfg.get("optimizer", {})
            if bool(opt_cfg.get("torch_compile", True)):
                try:
                    self.model = self.T.compile(self.model, mode="default")  # ~1.2x; CUDA graphs unsafe for training loop
                    zh = self.T.zeros(1, 12, device=self.device)
                    bp0 = self.T.zeros(1, 63, device=self.device)
                    with self.T.no_grad():
                        self.model(betas=self.shared_betas, body_pose=bp0,
                                   global_orient=bp0[:, :3], transl=bp0[:, :3],
                                   left_hand_pose=zh, right_hand_pose=zh,
                                   return_verts=False)
                except Exception:
                    pass

        # Pre-build index tensors
        body_smplx_map = dict(getattr(self.fitter, "BODY_SMPLX_MAP", {}))
        body_smplx_map.setdefault("left_shoulder", "left_shoulder")
        body_smplx_map.setdefault("right_shoulder", "right_shoulder")
        body_idx = {name: i for i, name in enumerate(BODY_NAMES)}
        src_l, tgt_l = [], []
        target_names = []
        for bn, sn in body_smplx_map.items():
            if bn in body_idx and sn in self.joint_name_to_idx:
                src_l.append(self.joint_name_to_idx[sn])
                tgt_l.append(body_idx[bn])
                target_names.append(bn)
        self._src = self.T.tensor(src_l, device=self.device, dtype=self.T.long)
        self._tgt_idx = self.T.tensor(tgt_l, device=self.device, dtype=self.T.long)
        self._target_names = target_names
        self._target_weights = self.T.tensor(
            [_target_weight(name, tcfg) for name in target_names],
            device=self.device,
            dtype=self.T.float32,
        )
        self._target_smooth_alpha = self.T.tensor(
            [_target_smooth_alpha(name, tcfg) for name in target_names],
            device=self.device,
            dtype=self.T.float32,
        )
        self._pose_prior_weights = self.T.tensor(
            _body_pose_prior_weights(tcfg),
            device=self.device,
            dtype=self.T.float32,
        ).view(1, 63)
        self._temporal_weights = self.T.tensor(
            _body_pose_temporal_weights(tcfg),
            device=self.device,
            dtype=self.T.float32,
        ).view(1, 63)
        self._smplx_knee_triplets = self.T.tensor(
            [
                [
                    self.joint_name_to_idx["left_hip"],
                    self.joint_name_to_idx["left_knee"],
                    self.joint_name_to_idx["left_ankle"],
                ],
                [
                    self.joint_name_to_idx["right_hip"],
                    self.joint_name_to_idx["right_knee"],
                    self.joint_name_to_idx["right_ankle"],
                ],
            ],
            device=self.device,
            dtype=self.T.long,
        )
        self._target_knee_triplets = self.T.tensor(
            [
                [body_idx["left_hip"], body_idx["left_knee"], body_idx["left_ankle"]],
                [body_idx["right_hip"], body_idx["right_knee"], body_idx["right_ankle"]],
            ],
            device=self.device,
            dtype=self.T.long,
        )
        self._smplx_elbow_triplets = self.T.tensor(
            [
                [
                    self.joint_name_to_idx["left_shoulder"],
                    self.joint_name_to_idx["left_elbow"],
                    self.joint_name_to_idx["left_wrist"],
                ],
                [
                    self.joint_name_to_idx["right_shoulder"],
                    self.joint_name_to_idx["right_elbow"],
                    self.joint_name_to_idx["right_wrist"],
                ],
            ],
            device=self.device,
            dtype=self.T.long,
        )
        self._target_elbow_triplets = self.T.tensor(
            [
                [body_idx["left_shoulder"], body_idx["left_elbow"], body_idx["left_wrist"]],
                [body_idx["right_shoulder"], body_idx["right_elbow"], body_idx["right_wrist"]],
            ],
            device=self.device,
            dtype=self.T.long,
        )
        self._smplx_axis_pairs = self.T.tensor(
            [
                [self.joint_name_to_idx["left_hip"], self.joint_name_to_idx["right_hip"]],
                [self.joint_name_to_idx["left_shoulder"], self.joint_name_to_idx["right_shoulder"]],
            ],
            device=self.device,
            dtype=self.T.long,
        )
        self._target_axis_pairs = self.T.tensor(
            [
                [body_idx["left_hip"], body_idx["right_hip"]],
                [body_idx["left_shoulder"], body_idx["right_shoulder"]],
            ],
            device=self.device,
            dtype=self.T.long,
        )
        axw = tcfg.get("axis_weights", {})
        self._axis_weights = self.T.tensor(
            [float(axw.get("hip", 2.0)), float(axw.get("shoulder", 1.0))],
            device=self.device, dtype=self.T.float32,
        )

        self._smplx_shoulder_idx = self.T.tensor(
            [self.joint_name_to_idx["left_shoulder"], self.joint_name_to_idx["right_shoulder"]],
            device=self.device, dtype=self.T.long,
        )
        self._target_shoulder_idx = self.T.tensor(
            [body_idx["left_shoulder"], body_idx["right_shoulder"]],
            device=self.device, dtype=self.T.long,
        )

        # Spine Bezier: 5 SMPL-X joints along the spine chain
        self._smplx_spine_chain = self.T.tensor([
            self.joint_name_to_idx[n] for n in
            ["pelvis", "spine1", "spine2", "spine3", "neck"]
        ], device=self.device, dtype=self.T.long)

        # Target: hips_center, neck_center, left_hip, right_hip for spine geometry
        self._target_hips_center = body_idx["hips_center"]
        self._target_neck_center = body_idx["neck_center"]
        self._target_hip_l = body_idx["left_hip"]
        self._target_hip_r = body_idx["right_hip"]

        # Bezier parameters (learned from fit data, normalized by spine length)
        bz = tcfg.get("bezier", {})
        self._bezier_t = self.T.tensor(
            bz.get("t_values", [0.0, 0.25, 0.50, 0.65, 1.0]),
            device=self.device, dtype=self.T.float32,
        )
        self._bezier_p1_along = float(bz.get("p1_along", 0.25))
        self._bezier_p1_perp = float(bz.get("p1_perp", 0.12))
        self._bezier_p2_along = float(bz.get("p2_along", 0.67))
        self._bezier_p2_perp = float(bz.get("p2_perp", 0.16))

        # Spine direction: SMPL-X pelvis→neck indices + target hip/shoulder refs
        self._smplx_spine_dir_idx = self.T.tensor(
            [self._smplx_spine_chain[0], self._smplx_spine_chain[4]],
            device=self.device, dtype=self.T.long,
        )

        # Pelvis frame: SMPL-X full-skeleton indices for pelvis+hips triangle
        self._smplx_pelvis_idx = self.joint_name_to_idx["pelvis"]
        self._smplx_hip_l_idx = self.joint_name_to_idx["left_hip"]
        self._smplx_hip_r_idx = self.joint_name_to_idx["right_hip"]

        # Warm-start state on GPU
        self._prev_bp = self.T.zeros(1, 63, device=self.device)
        self._prev_go = self.T.zeros(1, 3, device=self.device)
        self._prev_tr = self.T.zeros(1, 3, device=self.device)
        self._prev_prev_bp = self.T.zeros(1, 63, device=self.device)
        self._prev_prev_go = self.T.zeros(1, 3, device=self.device)
        self._prev_prev_tr = self.T.zeros(1, 3, device=self.device)
        self._has_prev = False
        self._has_prev_prev = False
        self._prev_target_23: object | None = None

        self.aggregate: list[dict[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    def update(self, frame_index: int, points_3d: np.ndarray) -> dict[str, np.ndarray]:
        T = self.T
        start = time.perf_counter()
        scaled = points_3d.astype(np.float32) * np.float32(self.config.input_scale)

        # Build target
        target_full = T.from_numpy(scaled).to(self.device)
        target_23 = target_full[self._tgt_idx]
        if self._prev_target_23 is not None:
            prev_target_23 = self._prev_target_23
            alpha = self._target_smooth_alpha
            smooth_mask = (alpha > 0.0) & T.isfinite(prev_target_23).all(dim=1) & T.isfinite(target_23).all(dim=1)
            smoothed = alpha[:, None] * prev_target_23 + (1.0 - alpha[:, None]) * target_23
            target_23 = T.where(smooth_mask[:, None], smoothed, target_23)
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
        pp_bp = self._prev_prev_bp.detach().clone()
        pp_go = self._prev_prev_go.detach().clone()
        pp_tr = self._prev_prev_tr.detach().clone()
        has_prev = self._has_prev
        has_pp = self._has_prev_prev
        zh = T.zeros(1, 12, device=self.device)

        _src = self._src
        _valid = valid_23
        _tgt = target_23
        _model = self.model
        _betas = self.shared_betas
        tw = self.config.track_temporal_weight  # default 0.05 — light smoothing, Bezier spine provides stability
        _target_weights = self._target_weights
        _pose_prior_weights = self._pose_prior_weights
        _temporal_weights = self._temporal_weights
        _smplx_knee_triplets = self._smplx_knee_triplets
        _target_knee_triplets = self._target_knee_triplets
        _smplx_elbow_triplets = self._smplx_elbow_triplets
        _target_elbow_triplets = self._target_elbow_triplets
        _smplx_axis_pairs = self._smplx_axis_pairs
        _target_axis_pairs = self._target_axis_pairs
        _axis_weights = self._axis_weights
        _smplx_shoulder_idx = self._smplx_shoulder_idx
        _target_shoulder_idx = self._target_shoulder_idx
        _smplx_spine_chain = self._smplx_spine_chain
        _bezier_t = self._bezier_t
        _target_hip_l = self._target_hip_l
        _target_hip_r = self._target_hip_r

        # Adam with joint-specific priors. Knees stay loose so squats can bend.
        steady_steps = max(1, int(self.config.track_pose_steps))
        recovery_steps = max(steady_steps, int(self.config.track_recovery_pose_steps))
        n_steps = recovery_steps if not has_prev else steady_steps
        adapter_elapsed = time.perf_counter() - start

        opt_cfg = self._tcfg.get("optimizer", {})
        lr_cfg = opt_cfg.get("learning_rates", {})
        optimizer = T.optim.Adam([
            {"params": [bp], "lr": float(lr_cfg.get("body_pose", 0.08))},
            {"params": [go], "lr": float(lr_cfg.get("global_orient", 0.05))},
            {"params": [tr], "lr": float(lr_cfg.get("transl", 0.03))},
        ])

        if self.device.type == "cuda":
            T.cuda.synchronize()
        opt_start = time.perf_counter()

        lw = self._tcfg.get("loss_weights", {})
        tcfg_temporal = self._tcfg.get("temporal", {})
        go_scale = float(tcfg_temporal.get("global_orient_scale", 0.5))
        tr_scale = float(tcfg_temporal.get("transl_scale", 0.3))
        v_go_scale = float(tcfg_temporal.get("velocity_global_orient_scale", 0.3))
        v_tr_scale = float(tcfg_temporal.get("velocity_transl_scale", 0.2))
        a_go_scale = float(tcfg_temporal.get("acceleration_global_orient_scale", 0.3))
        a_tr_scale = float(tcfg_temporal.get("acceleration_transl_scale", 0.2))

        for _ in range(n_steps):
            optimizer.zero_grad()
            out = _model(
                betas=_betas, body_pose=bp, global_orient=go, transl=tr,
                left_hand_pose=zh, right_hand_pose=zh,
                return_verts=False,
            )
            pred_pts = out.joints[0, _src]
            diffs = pred_pts[_valid] - _tgt[_valid]
            valid_weights = _target_weights[_valid]
            loss = (valid_weights[:, None] * diffs.square()).sum() / (3.0 * valid_weights.sum().clamp_min(1e-6))

            loss = loss + T.mean(_pose_prior_weights * bp.square())
            loss = loss + float(lw.get("knee_angle", 0.04)) * _knee_angle_loss(
                T,
                out.joints[0],
                target_full,
                _smplx_knee_triplets,
                _target_knee_triplets,
            )
            loss = loss + float(lw.get("elbow_angle", 0.03)) * _knee_angle_loss(
                T,
                out.joints[0],
                target_full,
                _smplx_elbow_triplets,
                _target_elbow_triplets,
                max_target_segment_m=float(lw.get("elbow_max_segment_m", 0.40)),
            )
            loss = loss + float(lw.get("axis_alignment", 0.10)) * _axis_alignment_loss(
                T,
                out.joints[0],
                target_full,
                _smplx_axis_pairs,
                _target_axis_pairs,
                _axis_weights,
            )
            loss = loss + float(lw.get("shoulder_line", 0.08)) * _shoulder_line_loss(
                T,
                out.joints[0],
                target_full,
                _smplx_shoulder_idx,
                _target_shoulder_idx,
            )
            loss = loss + float(lw.get("spine_bezier", 0.12)) * _spine_bezier_loss(
                T,
                out.joints[0],
                target_full,
                _smplx_spine_chain,
                _bezier_t,
                0, 0,
                _target_hip_l,
                _target_hip_r,
                p1_along=self._bezier_p1_along,
                p1_perp=self._bezier_p1_perp,
                p2_along=self._bezier_p2_along,
                p2_perp=self._bezier_p2_perp,
            )
            loss = loss + float(lw.get("spine_sagittal", 0.12)) * _spine_sagittal_loss(
                T,
                out.joints[0],
                _smplx_spine_chain,
                target_points=target_full,
                tgt_hip_l_idx=_target_hip_l,
                tgt_hip_r_idx=_target_hip_r,
            )
            loss = loss + float(lw.get("hip_symmetry", 0.0)) * _hip_symmetry_loss(T, bp)
            loss = loss + float(lw.get("upper_body_symmetry", 0.003)) * _upper_body_symmetry_loss(T, bp)
            loss = loss + float(lw.get("pelvis_frame", 0.20)) * _pelvis_frame_loss(
                T,
                out.joints[0],
                target_full,
                self._smplx_pelvis_idx,
                self._smplx_hip_l_idx,
                self._smplx_hip_r_idx,
                _target_hip_l,
                _target_hip_r,
            )

            if has_prev:
                loss = loss + tw * (
                    T.mean(_temporal_weights * (bp - p_bp).square())
                    + go_scale * T.mean((go - p_go) ** 2)
                    + tr_scale * T.mean((tr - p_tr) ** 2)
                )
            if has_pp:
                vw = self.config.track_velocity_weight
                if vw > 0:
                    cv_bp = bp - p_bp; pv_bp = p_bp - pp_bp
                    cv_go = go - p_go; pv_go = p_go - pp_go
                    cv_tr = tr - p_tr; pv_tr = p_tr - pp_tr
                    loss = loss + vw * (
                        T.mean(_temporal_weights * (cv_bp - pv_bp).square())
                        + v_go_scale * T.mean((cv_go - pv_go) ** 2)
                        + v_tr_scale * T.mean((cv_tr - pv_tr) ** 2)
                    )
                aw = self.config.track_acceleration_weight
                if aw > 0:
                    loss = loss + aw * (
                        T.mean(_temporal_weights * (bp - 2 * p_bp + pp_bp).square())
                        + a_go_scale * T.mean((go - 2 * p_go + pp_go) ** 2)
                        + a_tr_scale * T.mean((tr - 2 * p_tr + pp_tr) ** 2)
                    )
            loss.backward()
            optimizer.step()
        if self.device.type == "cuda":
            T.cuda.synchronize()
        opt_elapsed = time.perf_counter() - opt_start

        with T.no_grad():
            out = _model(
                betas=_betas, body_pose=bp, global_orient=go, transl=tr,
                left_hand_pose=zh, right_hand_pose=zh,
                return_verts=False,
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
        self._prev_prev_bp = self._prev_bp.clone()
        self._prev_prev_go = self._prev_go.clone()
        self._prev_prev_tr = self._prev_tr.clone()
        self._prev_bp = bp.detach().clone()
        self._prev_go = go.detach().clone()
        self._prev_tr = tr.detach().clone()
        self._prev_target_23 = target_23.detach().clone()
        self._has_prev = True
        self._has_prev_prev = has_prev

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
        sigma = float(self._tcfg.get("post_smooth", {}).get("sigma", 0.03))
        _apply_so3_smooth(self.aggregate, sigma=sigma)
        self._refresh_smoothed_joints()
        sequence_path = self.output_dir / "smplx_fit_sequence.npz"
        _save_sequence_npz(sequence_path, self.aggregate, self.config, source_npz=source_npz)
        _write_track_report(self.output_dir, source_npz, sequence_path, self.aggregate, self.config)
        return sequence_path

    def _refresh_smoothed_joints(self) -> None:
        T = self.T
        zh = T.zeros(1, 12, device=self.device)
        with T.no_grad():
            for item in self.aggregate:
                bp = T.tensor(item["body_pose"], dtype=T.float32, device=self.device)
                go = T.tensor(item["global_orient"], dtype=T.float32, device=self.device)
                tr = T.tensor(item["transl"], dtype=T.float32, device=self.device)
                out = self.model(
                    betas=self.shared_betas,
                    body_pose=bp,
                    global_orient=go,
                    transl=tr,
                    left_hand_pose=zh,
                    right_hand_pose=zh,
                    return_verts=False,
                )
                item["smplx_joints"] = out.joints.cpu().numpy().copy()


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


def _knee_angle_loss(
    T: object,
    pred_joints: object,
    target_points: object,
    pred_triplets: object,
    target_triplets: object,
    max_target_segment_m: float | None = None,
) -> object:
    target_sel = target_points[target_triplets]
    valid = T.isfinite(target_sel).all(dim=(1, 2))
    if max_target_segment_m is not None:
        upper = T.linalg.norm(target_sel[:, 0] - target_sel[:, 1], dim=1)
        lower = T.linalg.norm(target_sel[:, 2] - target_sel[:, 1], dim=1)
        valid = valid & (upper <= max_target_segment_m) & (lower <= max_target_segment_m)
    if not bool(valid.any()):
        return pred_joints.sum() * 0.0

    pred_cos = _triplet_cosine(T, pred_joints, pred_triplets)
    target_cos = _triplet_cosine(T, target_points, target_triplets)
    valid = valid & T.isfinite(pred_cos) & T.isfinite(target_cos)
    if not bool(valid.any()):
        return pred_joints.sum() * 0.0
    return (pred_cos[valid] - target_cos[valid]).square().mean()


def _triplet_cosine(T: object, points: object, triplets: object) -> object:
    p = points[triplets]
    v1 = p[:, 0] - p[:, 1]
    v2 = p[:, 2] - p[:, 1]
    denom = T.linalg.norm(v1, dim=1) * T.linalg.norm(v2, dim=1)
    return (v1 * v2).sum(dim=1) / denom.clamp_min(1e-6)


def _axis_alignment_loss(
    T: object,
    pred_joints: object,
    target_points: object,
    pred_pairs: object,
    target_pairs: object,
    weights: object,
) -> object:
    pred_sel = pred_joints[pred_pairs]
    target_sel = target_points[target_pairs]
    valid = T.isfinite(target_sel).all(dim=(1, 2))
    pred_axis = _horizontal_axis(T, pred_sel[:, 0] - pred_sel[:, 1])
    target_axis = _horizontal_axis(T, target_sel[:, 0] - target_sel[:, 1])
    valid = valid & T.isfinite(pred_axis).all(dim=1) & T.isfinite(target_axis).all(dim=1)
    if not bool(valid.any()):
        return pred_joints.sum() * 0.0
    diffs = pred_axis[valid] - target_axis[valid]
    valid_weights = weights[valid]
    return (valid_weights[:, None] * diffs.square()).sum() / (2.0 * valid_weights.sum().clamp_min(1e-6))


def _shoulder_line_loss(
    T: object,
    pred_joints: object,
    target_points: object,
    smplx_shoulder_idx: object,
    target_shoulder_idx: object,
) -> object:
    """Penalise perpendicular distance from SMPL-X shoulders to the MediaPipe shoulder line."""
    tgt_l = target_points[target_shoulder_idx[0]]
    tgt_r = target_points[target_shoulder_idx[1]]
    if not bool(T.isfinite(tgt_l).all() & T.isfinite(tgt_r).all()):
        return pred_joints.sum() * 0.0

    line = tgt_r - tgt_l
    line_len_sq = T.dot(line, line)
    if line_len_sq < 1e-10:
        return pred_joints.sum() * 0.0

    loss = 0.0
    for i in range(2):
        pred_pt = pred_joints[smplx_shoulder_idx[i]]
        to_l = pred_pt - tgt_l
        proj_factor = T.clamp(T.dot(to_l, line) / line_len_sq, 0.0, 1.0)
        perp = to_l - proj_factor * line
        loss = loss + T.sum(perp.square())
    return loss / 2.0


def _spine_direction_loss(
    T: object,
    pred_joints: object,
    target_points: object,
    smplx_spine_dir_idx: object,
    tgt_hip_l_idx: int,
    tgt_hip_r_idx: int,
    tgt_shoulder_l_idx: int = 11,
    tgt_shoulder_r_idx: int = 12,
) -> object:
    """Penalise angular deviation of SMPL-X spine direction vs target.

    Target spine direction is computed from MediaPipe hip and shoulder centres,
    not from derived BODY_NAMES indices (which are absent from the raw array).
    """
    pred_pelvis = pred_joints[smplx_spine_dir_idx[0]]
    pred_neck = pred_joints[smplx_spine_dir_idx[1]]

    tgt_hl = target_points[tgt_hip_l_idx]
    tgt_hr = target_points[tgt_hip_r_idx]
    tgt_sl = target_points[tgt_shoulder_l_idx]
    tgt_sr = target_points[tgt_shoulder_r_idx]

    valid = (T.isfinite(tgt_hl).all() & T.isfinite(tgt_hr).all() &
             T.isfinite(tgt_sl).all() & T.isfinite(tgt_sr).all())
    if not bool(valid):
        return pred_joints.sum() * 0.0

    tgt_hips = (tgt_hl + tgt_hr) / 2.0
    tgt_neck = (tgt_sl + tgt_sr) / 2.0

    pred_dir = pred_neck - pred_pelvis
    tgt_dir = tgt_neck - tgt_hips

    pred_norm = T.linalg.norm(pred_dir)
    tgt_norm = T.linalg.norm(tgt_dir)
    if pred_norm < 1e-8 or tgt_norm < 1e-8:
        return pred_joints.sum() * 0.0

    pred_dir = pred_dir / pred_norm
    tgt_dir = tgt_dir / tgt_norm
    return 1.0 - T.dot(pred_dir, tgt_dir)


def _bezier_point(
    t: object, P0: object, P1: object, P2: object, P3: object,
) -> object:
    """Cubic Bezier B(t) = (1-t)³P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³P3."""
    t = t.unsqueeze(-1) if t.dim() < 2 else t
    mt = 1.0 - t
    return mt**3 * P0 + 3 * mt**2 * t * P1 + 3 * mt * t**2 * P2 + t**3 * P3


def _spine_bezier_loss(
    T: object,
    pred_joints: object,       # (127, 3) SMPL-X joints
    target_points: object,     # (75, 3) raw MediaPipe landmarks
    smplx_spine_chain: object, # (5,) SMPL-X indices: pelvis, spine1, spine2, spine3, neck
    bezier_t: object,          # (5,) t parameters
    _hips_idx: int,            # unused (derived from hip_L + hip_R)
    _neck_idx: int,            # unused (derived from shoulder_L + shoulder_R)
    hip_l_idx: int,            # BODY_NAMES index for left_hip → raw MP index
    hip_r_idx: int,            # BODY_NAMES index for right_hip
    shoulder_l_idx: int = 11,  # BODY_NAMES index for left_shoulder → raw MP index 11
    shoulder_r_idx: int = 12,  # BODY_NAMES index for right_shoulder → raw MP index 12
    p1_along: float = 0.25, p1_perp: float = 0.12,
    p2_along: float = 0.67, p2_perp: float = 0.16,
) -> object:
    """Penalise spine joints deviating from a Bezier curve anchored by computed target positions.

    hips_center and neck_center are derived from raw MediaPipe landmarks rather than
    indexed directly (they are not present in the 75-element array).
    """
    # Compute target anchors from raw MediaPipe landmarks
    tgt_hl = target_points[hip_l_idx]
    tgt_hr = target_points[hip_r_idx]
    tgt_sl = target_points[shoulder_l_idx]
    tgt_sr = target_points[shoulder_r_idx]

    valid = (T.isfinite(tgt_hl).all() & T.isfinite(tgt_hr).all() &
             T.isfinite(tgt_sl).all() & T.isfinite(tgt_sr).all())
    if not bool(valid):
        return pred_joints.sum() * 0.0

    tgt_pelvis = (tgt_hl + tgt_hr) / 2.0   # hips_center
    tgt_neck = (tgt_sl + tgt_sr) / 2.0     # neck_center

    spine_vec = tgt_neck - tgt_pelvis
    spine_len = T.linalg.norm(spine_vec)
    if spine_len < 1e-8:
        return pred_joints.sum() * 0.0
    spine_dir = spine_vec / spine_len

    # Forward perpendicular direction: cross(hip_axis, spine_dir)
    hip_axis = tgt_hr - tgt_hl
    forward = T.cross(hip_axis, spine_vec)
    forward_norm = T.linalg.norm(forward)
    if forward_norm > 1e-8:
        forward = forward / forward_norm
    else:
        return pred_joints.sum() * 0.0

    # Build Bezier control points
    P0 = tgt_pelvis
    P1 = tgt_pelvis + p1_along * spine_vec + p1_perp * spine_len * forward
    P2 = tgt_pelvis + p2_along * spine_vec + p2_perp * spine_len * forward
    P3 = tgt_neck

    # Evaluate Bezier at t = 0.25 (spine1), 0.50 (spine2), 0.65 (spine3)
    t_eval = bezier_t[1:4]
    target_spine = _bezier_point(t_eval, P0, P1, P2, P3)

    pred_spine = pred_joints[smplx_spine_chain[1:4]]
    diffs = pred_spine - target_spine
    return T.mean(diffs.square())


def _spine_sagittal_loss(
    T: object,
    pred_joints: object,
    smplx_spine_chain: object,
    target_points: object | None = None,
    tgt_hip_l_idx: int = 23,
    tgt_hip_r_idx: int = 24,
) -> object:
    """Penalise lateral (side-to-side) bending of the spine.

    The spine should bend primarily in the sagittal plane (forward), not
    left-right.  Uses the target (MediaPipe) hip axis to define the lateral
    reference direction, avoiding circular dependency on predicted SMPL-X hips
    that may themselves be twisted.
    """
    # Use target hip axis as lateral reference (avoids circular dependency)
    if target_points is not None:
        tgt_hl = target_points[tgt_hip_l_idx]
        tgt_hr = target_points[tgt_hip_r_idx]
        if T.isfinite(tgt_hl).all() & T.isfinite(tgt_hr).all():
            hip_axis = tgt_hr - tgt_hl
        else:
            hip_axis = pred_joints[2] - pred_joints[1]
    else:
        hip_axis = pred_joints[2] - pred_joints[1]

    hip_axis_norm = T.linalg.norm(hip_axis)
    if hip_axis_norm < 1e-8:
        return pred_joints.sum() * 0.0
    lateral_dir = hip_axis / hip_axis_norm

    spine_pts = pred_joints[smplx_spine_chain]
    pelvis = spine_pts[0]
    neck = spine_pts[4]
    spine_dir = neck - pelvis
    spine_len = T.linalg.norm(spine_dir)
    if spine_len < 1e-8:
        return pred_joints.sum() * 0.0
    spine_dir = spine_dir / spine_len

    loss = 0.0
    for i in range(1, 4):
        to_pelvis = spine_pts[i] - pelvis
        along = T.dot(to_pelvis, spine_dir)
        proj = pelvis + along * spine_dir
        dev = spine_pts[i] - proj
        lateral = T.dot(dev, lateral_dir)
        loss = loss + lateral.square()
    return loss / 3.0


def _hip_symmetry_loss(T: object, body_pose: object) -> object:
    """Penalise asymmetric left/right hip rotation magnitudes.

    body_pose[0] = left_hip, body_pose[1] = right_hip.
    """
    l_hip = body_pose[:, 0:3]
    r_hip = body_pose[:, 3:6]
    l_norm = T.linalg.norm(l_hip, dim=1)
    r_norm = T.linalg.norm(r_hip, dim=1)
    return T.mean((l_norm - r_norm).square())


def _upper_body_symmetry_loss(T: object, body_pose: object) -> object:
    """Soft symmetry on collar, shoulder, elbow joint rotation magnitudes.

    Collars are weighted highest (attached to the same spine3 vertebra),
    shoulders moderate, elbows lowest.  The loss is intentionally weak so
    natural asymmetric motions (throwing, reaching) are preserved.
    """
    loss = 0.0
    # collars: bp[12], bp[13] — should be nearly symmetric
    l_c = T.linalg.norm(body_pose[:, 36:39], dim=1)
    r_c = T.linalg.norm(body_pose[:, 39:42], dim=1)
    loss = loss + 1.0 * T.mean((l_c - r_c).square())
    # shoulders: bp[15], bp[16]
    l_s = T.linalg.norm(body_pose[:, 45:48], dim=1)
    r_s = T.linalg.norm(body_pose[:, 48:51], dim=1)
    loss = loss + 0.7 * T.mean((l_s - r_s).square())
    # elbows: bp[17], bp[18]
    l_e = T.linalg.norm(body_pose[:, 51:54], dim=1)
    r_e = T.linalg.norm(body_pose[:, 54:57], dim=1)
    loss = loss + 0.4 * T.mean((l_e - r_e).square())
    return loss


def _pelvis_frame_loss(
    T: object,
    pred_joints: object,       # (127, 3) SMPL-X full skeleton
    target_points: object,     # (75, 3) MediaPipe landmarks
    pelvis_idx: int,           # SMPL-X full-skeleton index for pelvis
    hip_l_idx: int,            # SMPL-X full-skeleton index for left_hip
    hip_r_idx: int,            # SMPL-X full-skeleton index for right_hip
    tgt_hip_l_idx: int,        # BODY_NAMES index (also MediaPipe index) for left_hip
    tgt_hip_r_idx: int,        # BODY_NAMES index for right_hip
) -> object:
    """Enforce pelvis+hips triangle matches target as a rigid body.

    The vectors from pelvis to left/right hip in SMPL-X should match the
    corresponding vectors in the MediaPipe target. This couples pelvis
    orientation and hip rotations structurally, without blindly suppressing
    individual joint rotations.
    """
    tgt_L = target_points[tgt_hip_l_idx]
    tgt_R = target_points[tgt_hip_r_idx]

    if not bool(T.isfinite(tgt_L).all() & T.isfinite(tgt_R).all()):
        return pred_joints.sum() * 0.0

    tgt_center = (tgt_L + tgt_R) / 2.0
    tgt_vec_L = tgt_L - tgt_center
    tgt_vec_R = tgt_R - tgt_center

    pred_pelvis = pred_joints[pelvis_idx]
    pred_L = pred_joints[hip_l_idx]
    pred_R = pred_joints[hip_r_idx]

    pred_vec_L = pred_L - pred_pelvis
    pred_vec_R = pred_R - pred_pelvis

    return (T.sum((pred_vec_L - tgt_vec_L).square()) +
            T.sum((pred_vec_R - tgt_vec_R).square())) / 2.0


def _horizontal_axis(T: object, axis: object) -> object:
    xy = axis[:, :2]
    norm = T.linalg.norm(xy, dim=1, keepdim=True).clamp_min(1e-6)
    return xy / norm


def _target_weight(name: str, tcfg: dict[str, Any]) -> float:
    """Per-landmark MSE weight from track config."""
    tw = tcfg.get("target_weights") or {}
    if name in tw:
        return float(tw[name])
    return 1.0


def _target_smooth_alpha(name: str, tcfg: dict[str, Any]) -> float:
    """Per-landmark EMA smoothing alpha from track config."""
    sa = tcfg.get("target_smooth_alpha") or {}
    if name in sa:
        return float(sa[name])
    return 0.0


def _body_pose_prior_weights(tcfg: dict[str, Any]) -> np.ndarray:
    """Per-joint L2 regularisation weights from track config (21 joints × 3 = 63)."""
    pw = tcfg.get("body_pose_prior_weights", {})
    default = float(pw.get("default", 0.005))
    weights = np.full(63, default, dtype=np.float32)
    for joint_str, value in pw.get("joints", {}).items():
        _set_joint_weight(weights, int(joint_str), float(value))
    return weights


def _body_pose_temporal_weights(tcfg: dict[str, Any]) -> np.ndarray:
    """Per-joint temporal smoothing weights from track config (21 joints × 3 = 63)."""
    tw = tcfg.get("body_pose_temporal_weights", {})
    default = float(tw.get("default", 1.0))
    weights = np.full(63, default, dtype=np.float32)
    for group in tw.get("groups", []):
        w = float(group["weight"])
        for joint in group.get("joints", []):
            _set_joint_weight(weights, int(joint), w)
    return weights


def _set_joint_weight(weights: np.ndarray, joint_index: int, value: float) -> None:
    start = joint_index * 3
    weights[start : start + 3] = value


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
