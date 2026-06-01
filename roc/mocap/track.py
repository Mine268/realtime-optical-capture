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
    _body_root_features,
    _build_reference_args,
    _ensure_retarget_dependencies,
    _format_timing_line,
    _load_reference_fitter,
    _resolve_smplx_model_dir,
    _save_sequence_npz,
    _sequence_from_points,
    _vector_angle_deg,
    _write_report,
)


def _geometric_root(points_3d_scaled, rest_joints, body_idx, joint_name_to_idx):
    """Estimate global_orient and transl from hip/shoulder keypoints.

    Root is in world space — geometric estimation is reliable.
    Returns flat arrays: global_orient (3,), transl (3,).
    """
    def _pt(name):
        idx = body_idx.get(name)
        if idx is None:
            return None
        p = points_3d_scaled[idx]
        return p.astype(np.float64) if np.all(np.isfinite(p)) else None

    def _jidx(name):
        return joint_name_to_idx[name]

    def _rot_between(v_from, v_to):
        v_from = v_from.astype(np.float64)
        v_to = v_to.astype(np.float64)
        fn = np.linalg.norm(v_from)
        tn = np.linalg.norm(v_to)
        if fn < 1e-10 or tn < 1e-10:
            return np.zeros(3, dtype=np.float32)
        v_from = v_from / fn
        v_to = v_to / tn
        cross = np.cross(v_from, v_to)
        dot = np.dot(v_from, v_to)
        cn = np.linalg.norm(cross)
        if cn < 1e-10:
            if dot > 0:
                return np.zeros(3, dtype=np.float32)
            perp = np.array([1.0, 0.0, 0.0]) if abs(v_from[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            return (np.cross(v_from, perp) / np.linalg.norm(np.cross(v_from, perp)) * np.pi).astype(np.float32)
        return (cross / cn * np.arctan2(cn, dot)).astype(np.float32)

    rj = rest_joints
    l_hip = _pt("left_hip"); r_hip = _pt("right_hip")
    l_sh = _pt("left_shoulder"); r_sh = _pt("right_shoulder")

    go = np.zeros(3, dtype=np.float32)
    tr = np.zeros(3, dtype=np.float32)

    if l_hip is not None and r_hip is not None:
        tr = ((l_hip + r_hip) / 2.0 - rj[_jidx("pelvis")]).astype(np.float32)
        hip_axis = r_hip - l_hip
        rest_hip = rj[_jidx("right_hip")] - rj[_jidx("left_hip")]
        if l_sh is not None and r_sh is not None:
            sh_axis = r_sh - l_sh
            rest_sh = rj[_jidx("right_shoulder")] - rj[_jidx("left_shoulder")]
            go = ((_rot_between(rest_hip, hip_axis) + _rot_between(rest_sh, sh_axis)) / 2.0).astype(np.float32)
        else:
            go = _rot_between(rest_hip, hip_axis)

    return {"global_orient": go, "transl": tr}


class RealtimeSmplxTracker:
    """Body-only per-frame SMPL-X tracker.

    Uses the reference fitter with stripped-down loss terms for faster
    per-step execution, temporal warm-start, early stopping, and SO(3)
    post-smoothing.
    """

    def __init__(self, config: RetargetConfig, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = _resolve_smplx_model_dir(config.model_dir)
        if not model_dir.is_dir():
            raise RuntimeError(f"SMPL-X model directory not found: {config.model_dir}")
        _ensure_retarget_dependencies(use_vposer=False)

        self.config = config
        self.output_dir = output_dir
        self.fitter = _load_reference_fitter()
        self.device = self.fitter.torch.device(
            "cuda" if config.device == "cuda" and self.fitter.torch.cuda.is_available() else "cpu"
        )
        self.joint_name_to_idx = self.fitter.get_joint_name_to_index()

        base_args = _build_reference_args(config, output_dir, model_dir)
        self.args = _apply_track_overrides(base_args, config)
        self.args.early_stop_patience = 3
        self.args.early_stop_check_interval = 1

        self.init_args = _build_reference_args(config, output_dir, model_dir)
        _apply_track_overrides(self.init_args, config)
        self.init_args.root_steps = config.realtime_root_recovery_steps or _default_track_root_steps(config)
        self.init_args.early_stop_patience = 10
        self.init_args.early_stop_check_interval = 1

        self.model = self.fitter.create_model(
            self.args.model_dir, self.args.gender,
            self.args.num_betas, self.args.num_pca_comps, self.device,
        )

        self.shared_betas = self.fitter.torch.zeros(
            (1, self.args.num_betas), dtype=self.fitter.torch.float32, device=self.device,
        )

        # Cache rest pose for geometric root estimation
        self._rest_joints = self._compute_rest_pose()
        self._body_idx = {name: i for i, name in enumerate(BODY_NAMES)}

        self.aggregate: list[dict[str, np.ndarray]] = []
        self.prev_result: dict[str, np.ndarray] | None = None
        self.prev_prev_result: dict[str, np.ndarray] | None = None
        self.prev_body_axis: np.ndarray | None = None
        self.prev_body_center: np.ndarray | None = None

    def _compute_rest_pose(self):
        with self.fitter.torch.no_grad():
            z = self.fitter.torch.zeros
            d = self.device
            output = self.model(
                betas=z(1, self.args.num_betas, device=d),
                body_pose=z(1, 63, device=d), global_orient=z(1, 3, device=d),
                transl=z(1, 3, device=d), left_hand_pose=z(1, 12, device=d),
                right_hand_pose=z(1, 12, device=d),
            )
        return output.joints[0].cpu().numpy().copy()

    def update(self, frame_index: int, points_3d: np.ndarray) -> dict[str, np.ndarray]:
        start = time.perf_counter()
        sequence = _sequence_from_points(points_3d[None, :, :], input_scale=self.config.input_scale)
        body_axis, body_center = _body_root_features(sequence)
        root_steps, root_reason = self._select_root_steps(body_axis, body_center, points_3d)
        self.args.root_steps = root_steps
        adapter_elapsed = time.perf_counter() - start

        is_init = self.prev_result is None
        fit_args = self.init_args if is_init else self.args
        if not is_init:
            fit_args.root_steps = root_steps

        # Geometric root init (world-space, reliable)
        scaled = points_3d.astype(np.float32) * np.float32(self.config.input_scale)
        geo_root = _geometric_root(scaled, self._rest_joints, self._body_idx, self.joint_name_to_idx)
        zero_hands = np.zeros(12, dtype=np.float32)

        if is_init:
            init_state = {
                "body_pose": np.zeros(63, dtype=np.float32),
                "global_orient": geo_root["global_orient"],
                "transl": geo_root["transl"],
                "left_hand_pose": zero_hands,
                "right_hand_pose": zero_hands,
            }
        elif self.prev_result is not None:
            prev_bp = np.asarray(self.prev_result["body_pose"], dtype=np.float32).reshape(63)
            prev_go = np.asarray(self.prev_result["global_orient"], dtype=np.float32).reshape(3)
            prev_tr = np.asarray(self.prev_result["transl"], dtype=np.float32).reshape(3)
            prev_lh = np.asarray(self.prev_result.get("left_hand_pose", zero_hands), dtype=np.float32).reshape(12)
            prev_rh = np.asarray(self.prev_result.get("right_hand_pose", zero_hands), dtype=np.float32).reshape(12)
            # Blend geo root 15% + prev 85% for temporal stability
            init_state = {
                "body_pose": prev_bp,
                "global_orient": (geo_root["global_orient"] * 0.15 + prev_go * 0.85).astype(np.float32),
                "transl": (geo_root["transl"] * 0.15 + prev_tr * 0.85).astype(np.float32),
                "left_hand_pose": prev_lh,
                "right_hand_pose": prev_rh,
            }
        else:
            init_state = None

        result = self.fitter.fit_single_frame(
            self.model, None, 0, sequence, self.shared_betas,
            fit_args, self.joint_name_to_idx, self.device,
            init_state=init_state,
            prev_state=self.prev_result,
            prev_prev_state=self.prev_prev_result,
        )
        result["frame_index"] = np.array(frame_index, dtype=np.int32)
        result["retarget_root_steps"] = np.array(root_steps, dtype=np.int32)
        result["retarget_root_reason"] = np.array(root_reason)

        if self.config.profile:
            timings = dict(result.get("stage_timings", {}))
            timings["input_adapter_s"] = adapter_elapsed
            timings["track_update_s"] = time.perf_counter() - start
            result["stage_timings"] = timings
            if len(self.aggregate) == 0 or frame_index % max(1, self.config.profile_interval) == 0:
                print(_format_timing_line(frame_index, timings, root_steps, root_reason), flush=True)

        self.aggregate.append(result)
        self.prev_prev_result = self.prev_result
        self.prev_result = result
        self.prev_body_axis = body_axis
        self.prev_body_center = body_center
        return result

    def _select_root_steps(self, body_axis, body_center, points_3d=None):
        if not self.config.realtime_adaptive_root:
            return self.config.root_steps, "fixed"
        recovery = self.config.realtime_root_recovery_steps or _default_track_root_steps(self.config)
        if self.prev_result is None:
            return recovery, "init"
        prev_err = float(self.prev_result.get("body_mean_error_m", float("inf")))
        if prev_err > self.config.realtime_root_error_threshold_m:
            return recovery, "error"
        if (body_axis is not None and self.prev_body_axis is not None
                and _vector_angle_deg(body_axis, self.prev_body_axis) > self.config.realtime_root_turn_threshold_deg):
            return recovery, "turn"
        if (body_center is not None and self.prev_body_center is not None
                and float(np.linalg.norm(body_center - self.prev_body_center))
                > self.config.realtime_root_translation_threshold_m):
            return recovery, "translation"
        return max(1, self.config.realtime_root_steps), "steady"

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
    str_ = gaussian_filter1d(tr, sigma=sigma, axis=0)
    for i, item in enumerate(aggregate):
        item["body_pose"] = sbp[i].reshape(1, 63).astype(np.float32)
        item["global_orient"] = sgo[i].reshape(1, 3).astype(np.float32)
        item["transl"] = str_[i].reshape(1, 3).astype(np.float32)


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
