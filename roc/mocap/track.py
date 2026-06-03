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
            [_target_weight(name) for name in target_names],
            device=self.device,
            dtype=self.T.float32,
        )
        self._target_smooth_alpha = self.T.tensor(
            [_target_smooth_alpha(name) for name in target_names],
            device=self.device,
            dtype=self.T.float32,
        )
        self._pose_prior_weights = self.T.tensor(
            _body_pose_prior_weights(),
            device=self.device,
            dtype=self.T.float32,
        ).view(1, 63)
        self._temporal_weights = self.T.tensor(
            _body_pose_temporal_weights(),
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
        self._axis_weights = self.T.tensor([2.0, 1.0], device=self.device, dtype=self.T.float32)

        self._smplx_shoulder_idx = self.T.tensor(
            [self.joint_name_to_idx["left_shoulder"], self.joint_name_to_idx["right_shoulder"]],
            device=self.device, dtype=self.T.long,
        )
        self._target_shoulder_idx = self.T.tensor(
            [body_idx["left_shoulder"], body_idx["right_shoulder"]],
            device=self.device, dtype=self.T.long,
        )

        # Spine direction: pelvis→neck in SMPL-X, hips_center→neck_center in target
        self._smplx_spine_idx = self.T.tensor(
            [self.joint_name_to_idx["pelvis"], self.joint_name_to_idx["neck"]],
            device=self.device, dtype=self.T.long,
        )
        self._target_spine_idx = self.T.tensor(
            [body_idx["hips_center"], body_idx["neck_center"]],
            device=self.device, dtype=self.T.long,
        )

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
        tw = self.config.track_temporal_weight
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

        # Adam with joint-specific priors. Knees stay loose so squats can bend.
        steady_steps = max(1, int(self.config.track_pose_steps))
        recovery_steps = max(steady_steps, int(self.config.track_recovery_pose_steps))
        n_steps = recovery_steps if not has_prev else steady_steps
        adapter_elapsed = time.perf_counter() - start

        optimizer = T.optim.Adam([
            {"params": [bp], "lr": 0.08},
            {"params": [go], "lr": 0.05},
            {"params": [tr], "lr": 0.03},
        ])

        if self.device.type == "cuda":
            T.cuda.synchronize()
        opt_start = time.perf_counter()
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
            loss = loss + 0.08 * _knee_angle_loss(
                T,
                out.joints[0],
                target_full,
                _smplx_knee_triplets,
                _target_knee_triplets,
            )
            loss = loss + 0.04 * _knee_angle_loss(
                T,
                out.joints[0],
                target_full,
                _smplx_elbow_triplets,
                _target_elbow_triplets,
                max_target_segment_m=0.40,
            )
            loss = loss + 0.10 * _axis_alignment_loss(
                T,
                out.joints[0],
                target_full,
                _smplx_axis_pairs,
                _target_axis_pairs,
                _axis_weights,
            )
            loss = loss + 0.08 * _shoulder_line_loss(
                T,
                out.joints[0],
                target_full,
                _smplx_shoulder_idx,
                _target_shoulder_idx,
            )

            if has_prev:
                loss = loss + tw * (
                    T.mean(_temporal_weights * (bp - p_bp).square())
                    + 0.5 * T.mean((go - p_go) ** 2)
                    + 0.3 * T.mean((tr - p_tr) ** 2)
                )
            if has_pp:
                vw = self.config.track_velocity_weight
                if vw > 0:
                    cv_bp = bp - p_bp; pv_bp = p_bp - pp_bp
                    cv_go = go - p_go; pv_go = p_go - pp_go
                    cv_tr = tr - p_tr; pv_tr = p_tr - pp_tr
                    loss = loss + vw * (
                        T.mean(_temporal_weights * (cv_bp - pv_bp).square())
                        + 0.3 * T.mean((cv_go - pv_go) ** 2)
                        + 0.2 * T.mean((cv_tr - pv_tr) ** 2)
                    )
                aw = self.config.track_acceleration_weight
                if aw > 0:
                    loss = loss + aw * (
                        T.mean(_temporal_weights * (bp - 2 * p_bp + pp_bp).square())
                        + 0.3 * T.mean((go - 2 * p_go + pp_go) ** 2)
                        + 0.2 * T.mean((tr - 2 * p_tr + pp_tr) ** 2)
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
        _apply_so3_smooth(self.aggregate, sigma=0.30)
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
    smplx_spine_idx: object,
    target_spine_idx: object,
) -> object:
    """Penalise deviation between SMPL-X spine direction (pelvis→neck) and target."""
    pred_pelvis = pred_joints[smplx_spine_idx[0]]
    pred_neck = pred_joints[smplx_spine_idx[1]]
    tgt_hips = target_points[target_spine_idx[0]]
    tgt_neck = target_points[target_spine_idx[1]]

    if not bool(T.isfinite(tgt_hips).all() & T.isfinite(tgt_neck).all()):
        return pred_joints.sum() * 0.0

    pred_dir = pred_neck - pred_pelvis
    tgt_dir = tgt_neck - tgt_hips

    pred_norm = T.linalg.norm(pred_dir)
    tgt_norm = T.linalg.norm(tgt_dir)
    if pred_norm < 1e-8 or tgt_norm < 1e-8:
        return pred_joints.sum() * 0.0

    pred_dir = pred_dir / pred_norm
    tgt_dir = tgt_dir / tgt_norm

    # Cosine distance: penalise any angular deviation of the spine axis
    return 1.0 - T.dot(pred_dir, tgt_dir)


def _horizontal_axis(T: object, axis: object) -> object:
    xy = axis[:, :2]
    norm = T.linalg.norm(xy, dim=1, keepdim=True).clamp_min(1e-6)
    return xy / norm


def _target_weight(name: str) -> float:
    if name in {"left_knee", "right_knee", "left_ankle", "right_ankle"}:
        return 2.3
    if name in {"left_heel", "right_heel", "left_foot_index", "right_foot_index"}:
        return 1.8
    if name in {"left_hip", "right_hip"}:
        return 1.8
    if name == "hips_center":
        return 0.35
    if name in {"left_shoulder", "right_shoulder"}:
        return 1.7
    if name in {"left_elbow", "right_elbow"}:
        return 1.05
    if name in {"left_wrist", "right_wrist"}:
        return 0.55
    if name in {"trunk_center", "neck_center", "head_center"}:
        return 0.25
    if name in {"nose", "left_eye", "right_eye", "left_ear", "right_ear", "head_center"}:
        return 0.35
    return 1.0


def _target_smooth_alpha(name: str) -> float:
    if name in {"left_wrist", "right_wrist"}:
        return 0.20
    if name in {"left_elbow", "right_elbow"}:
        return 0.35
    return 0.0


def _body_pose_prior_weights() -> np.ndarray:
    weights = np.full(63, 0.018, dtype=np.float32)
    _set_joint_weight(weights, 2, 0.045)   # spine1
    _set_joint_weight(weights, 5, 0.045)   # spine2
    _set_joint_weight(weights, 8, 0.045)   # spine3
    _set_joint_weight(weights, 11, 0.035)  # neck
    _set_joint_weight(weights, 12, 0.035)  # left_collar
    _set_joint_weight(weights, 13, 0.035)  # right_collar
    _set_joint_weight(weights, 3, 0.010)   # left_knee
    _set_joint_weight(weights, 4, 0.010)   # right_knee
    _set_joint_weight(weights, 6, 0.010)   # left_ankle
    _set_joint_weight(weights, 7, 0.010)   # right_ankle
    _set_joint_weight(weights, 9, 0.012)   # left_foot
    _set_joint_weight(weights, 10, 0.012)  # right_foot
    _set_joint_weight(weights, 19, 0.120)  # left_wrist
    _set_joint_weight(weights, 20, 0.120)  # right_wrist
    return weights


def _body_pose_temporal_weights() -> np.ndarray:
    weights = np.ones(63, dtype=np.float32)
    for joint in (12, 13):
        _set_joint_weight(weights, joint, 2.4)
    for joint in (15, 16, 17, 18):
        _set_joint_weight(weights, joint, 1.8)
    for joint in (19, 20):
        _set_joint_weight(weights, joint, 1.2)
    for joint in (3, 4, 6, 7, 9, 10):
        _set_joint_weight(weights, joint, 0.65)
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
