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


class RealtimeSmplxTracker:
    """Body-only per-frame SMPL-X tracker for realtime mocap.

    Uses the reference fitter's full optimization for the first frame,
    then small-step body-only tracking with early stopping for
    subsequent frames.  Hands, face, and shape are frozen.
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
        # Re-enable early stopping for realtime
        self.args.early_stop_patience = 4
        self.args.early_stop_check_interval = 1

        # Build the init-frame args (full root steps, no early stop)
        self.init_args = _build_reference_args(config, output_dir, model_dir)
        _apply_track_overrides(self.init_args, config)
        self.init_args.root_steps = config.realtime_root_recovery_steps or _default_track_root_steps(config)
        self.init_args.early_stop_patience = 20
        self.init_args.early_stop_check_interval = 1

        self.model = self.fitter.create_model(
            self.args.model_dir,
            self.args.gender,
            self.args.num_betas,
            self.args.num_pca_comps,
            self.device,
        )

        self.shared_betas = self.fitter.torch.zeros(
            (1, self.args.num_betas),
            dtype=self.fitter.torch.float32,
            device=self.device,
        )

        self.aggregate: list[dict[str, np.ndarray]] = []
        self.prev_result: dict[str, np.ndarray] | None = None
        self.prev_prev_result: dict[str, np.ndarray] | None = None
        self.prev_body_axis: np.ndarray | None = None
        self.prev_body_center: np.ndarray | None = None

    def update(self, frame_index: int, points_3d: np.ndarray) -> dict[str, np.ndarray]:
        start = time.perf_counter()
        sequence = _sequence_from_points(points_3d[None, :, :], input_scale=self.config.input_scale)
        body_axis, body_center = _body_root_features(sequence)
        root_steps, root_reason = self._select_root_steps(body_axis, body_center, points_3d)
        self.args.root_steps = root_steps
        adapter_elapsed = time.perf_counter() - start

        # First frame: full root, no temporal init, longer early-stop patience
        is_init = self.prev_result is None
        fit_args = self.init_args if is_init else self.args
        if not is_init:
            fit_args.root_steps = root_steps

        result = self.fitter.fit_single_frame(
            self.model,
            None,
            0,
            sequence,
            self.shared_betas,
            fit_args,
            self.joint_name_to_idx,
            self.device,
            init_state=None if is_init else self.prev_result,
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

    def _select_root_steps(
        self,
        body_axis: np.ndarray | None,
        body_center: np.ndarray | None,
        points_3d: np.ndarray | None = None,
    ) -> tuple[int | None, str]:
        if not self.config.realtime_adaptive_root:
            return self.config.root_steps, "fixed"

        recovery_steps = self.config.realtime_root_recovery_steps or _default_track_root_steps(self.config)
        if self.prev_result is None:
            return recovery_steps, "init"

        prev_error = float(self.prev_result.get("body_mean_error_m", np.inf))
        if prev_error > self.config.realtime_root_error_threshold_m:
            return recovery_steps, "error"

        if (
            body_axis is not None
            and self.prev_body_axis is not None
            and _vector_angle_deg(body_axis, self.prev_body_axis) > self.config.realtime_root_turn_threshold_deg
        ):
            return recovery_steps, "turn"

        if (
            body_center is not None
            and self.prev_body_center is not None
            and float(np.linalg.norm(body_center - self.prev_body_center))
            > self.config.realtime_root_translation_threshold_m
        ):
            return recovery_steps, "translation"

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
    """Smooth body_pose rotations in-place using axis-angle SLERP."""
    if len(aggregate) < 3:
        return
    # Simple Gaussian-weighted average on axis-angle vectors
    from scipy.ndimage import gaussian_filter1d

    body_poses = np.stack([item["body_pose"].reshape(63) for item in aggregate], axis=0)
    global_orients = np.stack([item["global_orient"].reshape(3) for item in aggregate], axis=0)
    transls = np.stack([item["transl"].reshape(3) for item in aggregate], axis=0)

    smoothed_bp = gaussian_filter1d(body_poses, sigma=sigma, axis=0)
    smoothed_go = gaussian_filter1d(global_orients, sigma=sigma, axis=0)
    smoothed_tr = gaussian_filter1d(transls, sigma=sigma, axis=0)

    for i, item in enumerate(aggregate):
        item["body_pose"] = smoothed_bp[i].reshape(1, 63).astype(np.float32)
        item["global_orient"] = smoothed_go[i].reshape(1, 3).astype(np.float32)
        item["transl"] = smoothed_tr[i].reshape(1, 3).astype(np.float32)


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
    base_args.early_stop_patience = 4
    base_args.no_mesh = not config.save_debug_assets
    base_args.no_plot = not config.save_debug_assets
    base_args.use_vposer = False
    base_args.disable_post_smooth = True
    return base_args


def _default_track_root_steps(config: RetargetConfig) -> int:
    return int(config.root_steps or max(8, config.track_pose_steps // 2))


def _write_track_report(
    output_dir: Path,
    source_npz: Path | None,
    sequence_path: Path,
    aggregate: list[dict[str, np.ndarray]],
    config: RetargetConfig,
) -> None:
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
