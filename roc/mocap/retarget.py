from __future__ import annotations

import argparse
import enum
import importlib.util
import json
import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml


class RetargetMode(enum.Enum):
    FIT = "fit"
    TRACK = "track"


@dataclass(slots=True)
class RetargetConfig:
    model_dir: Path
    mode: RetargetMode = RetargetMode.FIT
    vposer_dir: Path | None = None
    output_dir: Path | None = None
    device: str = "cpu"
    gender: str = "neutral"
    num_betas: int = 10
    num_pca_comps: int = 12
    betas_sample_count: int = 16
    betas_steps: int = 80
    pose_steps: int = 120
    root_steps: int | None = None
    lower_steps: int | None = None
    lower_body_refine: bool = True
    early_stop_check_interval: int = 1
    temporal_weight: float = 0.0
    velocity_weight: float = 0.0
    acceleration_weight: float = 0.002
    realtime_adaptive_root: bool = True
    realtime_root_steps: int = 2
    realtime_root_recovery_steps: int | None = None
    realtime_root_error_threshold_m: float = 0.12
    realtime_root_turn_threshold_deg: float = 18.0
    realtime_root_translation_threshold_m: float = 0.20
    frame_step: int = 1
    max_frames: int = -1
    input_scale: float = 0.001
    optimize_hands: bool = False
    use_vposer: bool = False
    save_debug_assets: bool = False
    profile: bool = False
    profile_interval: int = 25
    # Tracking-specific parameters (used only in TRACK mode)
    track_pose_steps: int = 20
    track_temporal_weight: float = 0.05
    track_velocity_weight: float = 0.005
    track_acceleration_weight: float = 0.002
    track_recovery_interval: int = 30
    track_recovery_pose_steps: int = 60
    track_recovery_body_error_threshold_m: float = 0.15


class RealtimeSmplxRetargeter:
    def __init__(self, config: RetargetConfig, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = _resolve_smplx_model_dir(config.model_dir)
        if not model_dir.is_dir():
            raise RuntimeError(f"SMPL-X model directory not found: {config.model_dir}")
        if config.use_vposer and (config.vposer_dir is None or not config.vposer_dir.is_dir()):
            raise RuntimeError(f"VPoser directory not found: {config.vposer_dir}")
        _ensure_retarget_dependencies(config.use_vposer)

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        self.config = config
        self.output_dir = output_dir
        self.fitter = _load_reference_fitter()
        self.args = _build_reference_args(config, output_dir, model_dir)
        self.device = self.fitter.torch.device(
            "cuda" if self.args.device == "cuda" and self.fitter.torch.cuda.is_available() else "cpu"
        )
        self.joint_name_to_idx = self.fitter.get_joint_name_to_index()
        self.model = self.fitter.create_model(
            self.args.model_dir,
            self.args.gender,
            self.args.num_betas,
            self.args.num_pca_comps,
            self.device,
        )
        self.vposer = None
        if self.args.use_vposer:
            if self.args.vposer_dir is None:
                raise RuntimeError("--retarget-use-vposer requires --retarget-vposer-dir")
            self.vposer, _ = self.fitter.load_vposer(str(self.args.vposer_dir), vp_model="snapshot")
            self.vposer = self.vposer.to(device=self.device)
            self.vposer.eval()
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
        root_steps, root_reason = self._select_root_steps(body_axis, body_center)
        self.args.root_steps = root_steps
        adapter_elapsed = time.perf_counter() - start
        result = self.fitter.fit_single_frame(
            self.model,
            self.vposer,
            0,
            sequence,
            self.shared_betas,
            self.args,
            self.joint_name_to_idx,
            self.device,
            init_state=self.prev_result,
            prev_state=self.prev_result,
            prev_prev_state=self.prev_prev_result,
        )
        result["frame_index"] = np.array(frame_index, dtype=np.int32)
        result["retarget_root_steps"] = np.array(root_steps, dtype=np.int32)
        result["retarget_root_reason"] = np.array(root_reason)
        if self.config.profile:
            timings = dict(result.get("stage_timings", {}))
            timings["input_adapter_s"] = adapter_elapsed
            timings["retarget_update_s"] = time.perf_counter() - start
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
    ) -> tuple[int | None, str]:
        if not self.config.realtime_adaptive_root:
            return self.config.root_steps, "fixed"

        recovery_steps = self.config.realtime_root_recovery_steps or _default_root_steps(self.config)
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
            raise RuntimeError("No realtime SMPL-X retarget frames were produced")
        sequence_path = self.output_dir / "smplx_fit_sequence.npz"
        _save_sequence_npz(sequence_path, self.aggregate, self.config, source_npz=source_npz)
        _write_report(self.output_dir, source_npz, sequence_path, self.aggregate, self.config)
        return sequence_path


BODY_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
    "hips_center",
    "trunk_center",
    "neck_center",
    "head_center",
]

HAND_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


def run_mocap_retarget(npz_path: Path, mocap_session: Path, config: RetargetConfig) -> Path:
    npz_path = npz_path.resolve()
    mocap_session = mocap_session.resolve()
    output_dir = (config.output_dir or mocap_session / "smplx_retarget").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = _resolve_smplx_model_dir(config.model_dir)
    if not model_dir.is_dir():
        raise RuntimeError(f"SMPL-X model directory not found: {config.model_dir}")
    if config.use_vposer and (config.vposer_dir is None or not config.vposer_dir.is_dir()):
        raise RuntimeError(f"VPoser directory not found: {config.vposer_dir}")
    _ensure_retarget_dependencies(config.use_vposer)

    sequence = _load_roc_sequence(npz_path, input_scale=config.input_scale)
    fitter = _load_reference_fitter()
    args = _build_reference_args(config, output_dir, model_dir)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    device = fitter.torch.device("cuda" if args.device == "cuda" and fitter.torch.cuda.is_available() else "cpu")
    joint_name_to_idx = fitter.get_joint_name_to_index()
    model = fitter.create_model(args.model_dir, args.gender, args.num_betas, args.num_pca_comps, device)
    vposer = None
    if args.use_vposer:
        if args.vposer_dir is None:
            raise RuntimeError("--retarget-use-vposer requires --retarget-vposer-dir")
        vposer, _ = fitter.load_vposer(str(args.vposer_dir), vp_model="snapshot")
        vposer = vposer.to(device=device)
        vposer.eval()

    shared_betas = fitter.optimize_shared_betas(model, vposer, sequence, args, joint_name_to_idx, device)
    fit_indices = list(range(0, sequence.body.shape[0], max(1, args.frame_step)))
    if args.max_frames > 0:
        fit_indices = fit_indices[: args.max_frames]
    if not fit_indices:
        raise RuntimeError("No frames available for SMPL-X retargeting")

    aggregate: list[dict[str, np.ndarray]] = []
    per_frame_dir = output_dir / "per_frame"
    prev_result = None
    prev_prev_result = None
    for frame_index in fit_indices:
        result = fitter.fit_single_frame(
            model,
            vposer,
            frame_index,
            sequence,
            shared_betas,
            args,
            joint_name_to_idx,
            device,
            init_state=prev_result,
            prev_state=prev_result,
            prev_prev_state=prev_prev_result,
        )
        if args.profile:
            print(
                _format_timing_line(
                    frame_index,
                    dict(result.get("stage_timings", {})),
                ),
                flush=True,
            )
        fitter.save_result(result, per_frame_dir, frame_index)
        if not args.no_mesh:
            fitter.save_mesh(result, model, per_frame_dir, frame_index, device)
        if not args.no_plot:
            fitter.save_debug_plot(result, sequence, joint_name_to_idx, model, device, per_frame_dir, frame_index)
        aggregate.append(result)
        prev_prev_result = prev_result
        prev_result = result

    sequence_path = output_dir / "smplx_fit_sequence.npz"
    _save_sequence_npz(sequence_path, aggregate, config, source_npz=npz_path)
    _write_report(output_dir, npz_path, sequence_path, aggregate, config)
    return sequence_path


def _save_sequence_npz(
    sequence_path: Path,
    aggregate: list[dict[str, np.ndarray]],
    config: RetargetConfig,
    source_npz: Path | None,
) -> None:
    payload = {
        "frame_indices": np.array([item["frame_index"] for item in aggregate], dtype=np.int32),
        "betas": np.stack([item["betas"] for item in aggregate], axis=0),
        "global_orient": np.stack([item["global_orient"] for item in aggregate], axis=0),
        "body_pose": np.stack([item["body_pose"] for item in aggregate], axis=0),
        "left_hand_pose": np.stack([item["left_hand_pose"] for item in aggregate], axis=0),
        "right_hand_pose": np.stack([item["right_hand_pose"] for item in aggregate], axis=0),
        "transl": np.stack([item["transl"] for item in aggregate], axis=0),
        "smplx_joints": np.stack([item["smplx_joints"] for item in aggregate], axis=0),
        "input_scale": np.array(config.input_scale, dtype=np.float32),
        "overall_mean_error_m": np.array([item["overall_mean_error_m"] for item in aggregate], dtype=np.float32),
        "body_mean_error_m": np.array([item["body_mean_error_m"] for item in aggregate], dtype=np.float32),
        "left_hand_mean_error_m": np.array([item["left_hand_mean_error_m"] for item in aggregate], dtype=np.float32),
        "right_hand_mean_error_m": np.array([item["right_hand_mean_error_m"] for item in aggregate], dtype=np.float32),
    }
    if all("retarget_root_steps" in item for item in aggregate):
        payload["retarget_root_steps"] = np.array(
            [item["retarget_root_steps"] for item in aggregate],
            dtype=np.int32,
        )
    if all("retarget_root_reason" in item for item in aggregate):
        payload["retarget_root_reason"] = np.array(
            [_scalar_string(item["retarget_root_reason"]) for item in aggregate]
        )
    if source_npz is not None:
        payload["source_npz"] = str(source_npz)
    np.savez_compressed(sequence_path, **payload)


def _resolve_smplx_model_dir(model_dir: Path) -> Path:
    resolved = model_dir.resolve()
    if (resolved / "smplx").is_dir():
        return resolved
    if any((resolved / name).is_file() for name in ("SMPLX_NEUTRAL.npz", "SMPLX_NEUTRAL.pkl")):
        return resolved.parent
    return resolved


def _build_reference_args(config: RetargetConfig, output_dir: Path, model_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model_dir=model_dir,
        vposer_dir=config.vposer_dir.resolve() if config.vposer_dir is not None else None,
        output_dir=output_dir,
        device=config.device,
        gender=config.gender,
        frame_index=0,
        num_betas=config.num_betas,
        num_pca_comps=config.num_pca_comps,
        betas_sample_count=config.betas_sample_count,
        betas_steps=config.betas_steps,
        pose_steps=config.pose_steps,
        root_steps=config.root_steps,
        lower_steps=config.lower_steps,
        lower_body_refine=config.lower_body_refine,
        lr=0.05,
        betas_lr=0.05,
        body_weight=5.0,
        hand_weight=2.0 if config.optimize_hands else 0.0,
        foot_weight=8.0,
        foot_orient_weight=6.0,
        pose_prior_weight=0.02,
        shape_prior_weight=0.001,
        hand_prior_weight=0.01 if config.optimize_hands else 0.0,
        spine_prior_weight=0.0,
        no_body_landmarks=False,
        use_vposer=config.use_vposer,
        no_mesh=not config.save_debug_assets,
        no_plot=not config.save_debug_assets,
        early_stop_patience=20,
        early_stop_eps=1e-5,
        early_stop_check_interval=config.early_stop_check_interval,
        temporal_weight=config.temporal_weight,
        velocity_weight=config.velocity_weight,
        acceleration_weight=config.acceleration_weight,
        disable_post_smooth=True,
        smooth_window=0,
        smooth_sigma=0.0,
        frame_step=config.frame_step,
        max_frames=config.max_frames,
        run_full_sequence=True,
        profile=config.profile,
    )


def _default_root_steps(config: RetargetConfig) -> int:
    return int(config.root_steps or max(12, config.pose_steps // 4))


def _format_timing_line(
    frame_index: int,
    timings: dict[str, float],
    root_steps: int | None = None,
    root_reason: str | None = None,
) -> str:
    fields = (
        ("smplx_total", timings.get("retarget_update_s", 0.0)),
        ("target_setup", timings.get("target_setup_s", 0.0)),
        ("init", timings.get("init_s", 0.0)),
        ("root_optimize", timings.get("root_s", 0.0)),
        ("pose_optimize", timings.get("pose_s", 0.0)),
        ("lower_body_refine", timings.get("lower_s", 0.0)),
        ("final_forward", timings.get("final_s", 0.0)),
    )
    parts = [f"{name}={seconds * 1000.0:.1f}ms" for name, seconds in fields]
    if root_steps is not None:
        parts.insert(0, f"root_steps_used={root_steps}")
    if root_reason is not None:
        parts.insert(1, f"root_steps_reason={root_reason}")
    return f"[mocap-profile] frame={frame_index} stage=retarget " + " ".join(parts)


def _body_root_features(sequence: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        name_to_index = {name: idx for idx, name in enumerate(sequence.body_names)}
        body = sequence.body[0]
        left_hip = body[name_to_index["left_hip"]]
        right_hip = body[name_to_index["right_hip"]]
        center = body[name_to_index["hips_center"]]
    except (AttributeError, IndexError, KeyError):
        return None, None
    if not np.all(np.isfinite(left_hip)) or not np.all(np.isfinite(right_hip)) or not np.all(np.isfinite(center)):
        return None, None
    axis = right_hip - left_hip
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return None, center.astype(np.float32)
    return (axis / norm).astype(np.float32), center.astype(np.float32)


def _vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    cos_angle = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _load_reference_fitter() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "refs" / "smplx_from_freemocap_3d" / "fit_freemocap_smplx.py"
    if not script_path.is_file():
        raise RuntimeError(f"SMPL-X retarget reference script not found: {script_path}")
    _install_vposer_stub_if_missing()
    spec = importlib.util.spec_from_file_location("_roc_reference_smplx_fitter", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load SMPL-X retarget reference script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_retarget_dependencies(use_vposer: bool) -> None:
    missing = [name for name in ("torch", "smplx", "trimesh") if importlib.util.find_spec(name) is None]
    if use_vposer and importlib.util.find_spec("human_body_prior") is None:
        missing.append("human_body_prior")
    if missing:
        raise RuntimeError(
            "SMPL-X retarget dependencies are missing: "
            + ", ".join(missing)
            + ". Install them in the active roc environment before using --retarget."
        )


def _install_vposer_stub_if_missing() -> None:
    if importlib.util.find_spec("human_body_prior") is not None:
        return

    def load_vposer(*_args, **_kwargs):
        raise RuntimeError("VPoser is not installed; rerun without --retarget-use-vposer or install human_body_prior")

    human_body_prior = types.ModuleType("human_body_prior")
    tools = types.ModuleType("human_body_prior.tools")
    model_loader = types.ModuleType("human_body_prior.tools.model_loader")
    model_loader.load_vposer = load_vposer
    tools.model_loader = model_loader
    human_body_prior.tools = tools
    sys.modules.setdefault("human_body_prior", human_body_prior)
    sys.modules.setdefault("human_body_prior.tools", tools)
    sys.modules.setdefault("human_body_prior.tools.model_loader", model_loader)


def _load_roc_sequence(npz_path: Path, input_scale: float) -> Any:
    with np.load(npz_path, allow_pickle=True) as data:
        points_3d = np.asarray(data["points_3d"], dtype=np.float32) * np.float32(input_scale)
        landmark_names = [str(name) for name in data["landmark_names"].tolist()]
    return _sequence_from_points(points_3d, input_scale=1.0, landmark_names=landmark_names)


def _sequence_from_points(
    points_3d: np.ndarray,
    input_scale: float,
    landmark_names: list[str] | None = None,
) -> Any:
    points_3d = np.asarray(points_3d, dtype=np.float32) * np.float32(input_scale)
    if landmark_names is None:
        landmark_names = (
            BODY_NAMES[:33]
            + [f"left_hand_{name}" for name in HAND_NAMES]
            + [f"right_hand_{name}" for name in HAND_NAMES]
        )
    if points_3d.ndim != 3 or points_3d.shape[-1] != 3:
        raise RuntimeError(f"Expected points_3d with shape (frames, landmarks, 3), got {points_3d.shape}")
    name_to_idx = {name: idx for idx, name in enumerate(landmark_names)}

    body = _build_body_sequence(points_3d, name_to_idx)
    left_hand = _build_hand_sequence(points_3d, name_to_idx, "left")
    right_hand = _build_hand_sequence(points_3d, name_to_idx, "right")

    fitter = _load_reference_fitter()
    sequence = fitter.SequenceData(
        body=_fill_missing_sequence(body),
        left_hand=_fill_missing_sequence(left_hand),
        right_hand=_fill_missing_sequence(right_hand),
        body_names=BODY_NAMES,
        left_hand_names=[f"left_hand_{name}" for name in HAND_NAMES],
        right_hand_names=[f"right_hand_{name}" for name in HAND_NAMES],
    )
    return sequence


def _build_body_sequence(points_3d: np.ndarray, name_to_idx: dict[str, int]) -> np.ndarray:
    frames = points_3d.shape[0]
    body = np.full((frames, len(BODY_NAMES), 3), np.nan, dtype=np.float32)
    for out_idx, name in enumerate(BODY_NAMES):
        if name in name_to_idx:
            body[:, out_idx] = points_3d[:, name_to_idx[name]]

    body_idx = {name: idx for idx, name in enumerate(BODY_NAMES)}
    body[:, body_idx["hips_center"]] = _nanmean_points(body[:, [body_idx["left_hip"], body_idx["right_hip"]]])
    body[:, body_idx["neck_center"]] = _nanmean_points(body[:, [body_idx["left_shoulder"], body_idx["right_shoulder"]]])
    body[:, body_idx["trunk_center"]] = _nanmean_points(
        body[:, [body_idx["left_shoulder"], body_idx["right_shoulder"], body_idx["left_hip"], body_idx["right_hip"]]]
    )
    body[:, body_idx["head_center"]] = _nanmean_points(
        body[:, [body_idx["nose"], body_idx["left_ear"], body_idx["right_ear"], body_idx["left_eye"], body_idx["right_eye"]]]
    )
    return body


def _build_hand_sequence(points_3d: np.ndarray, name_to_idx: dict[str, int], side: str) -> np.ndarray:
    frames = points_3d.shape[0]
    hand = np.full((frames, len(HAND_NAMES), 3), np.nan, dtype=np.float32)
    for out_idx, name in enumerate(HAND_NAMES):
        source_name = f"{side}_hand_{name}"
        if source_name in name_to_idx:
            hand[:, out_idx] = points_3d[:, name_to_idx[source_name]]
    return hand


def _nanmean_points(points: np.ndarray) -> np.ndarray:
    valid = np.isfinite(points).all(axis=-1)
    count = np.maximum(valid.sum(axis=1), 1).astype(np.float32)
    values = np.where(valid[..., None], points, 0.0).sum(axis=1) / count[:, None]
    values[valid.sum(axis=1) == 0] = np.nan
    return values.astype(np.float32)


def _fill_missing_sequence(sequence: np.ndarray) -> np.ndarray:
    filled = sequence.copy()
    frame_count, point_count, coord_count = filled.shape
    x = np.arange(frame_count)
    for point_idx in range(point_count):
        for coord_idx in range(coord_count):
            values = filled[:, point_idx, coord_idx]
            finite = np.isfinite(values)
            if finite.all():
                continue
            if not finite.any():
                filled[:, point_idx, coord_idx] = 0.0
                continue
            filled[:, point_idx, coord_idx] = np.interp(x, x[finite], values[finite]).astype(np.float32)
    return filled.astype(np.float32)


def _write_report(
    output_dir: Path,
    source_npz: Path | None,
    sequence_path: Path,
    aggregate: list[dict[str, np.ndarray]],
    config: RetargetConfig,
) -> None:
    report = {
        "source_npz": str(source_npz) if source_npz is not None else None,
        "output_npz": str(sequence_path),
        "frames": len(aggregate),
        "model_dir": str(config.model_dir),
        "device": config.device,
        "frame_step": config.frame_step,
        "max_frames": config.max_frames,
        "input_scale": config.input_scale,
        "pose_steps": config.pose_steps,
        "root_steps": config.root_steps,
        "lower_steps": config.lower_steps,
        "lower_body_refine": config.lower_body_refine,
        "early_stop_check_interval": config.early_stop_check_interval,
        "temporal_weight": config.temporal_weight,
        "velocity_weight": config.velocity_weight,
        "acceleration_weight": config.acceleration_weight,
        "realtime_adaptive_root": config.realtime_adaptive_root,
        "realtime_root_steps": config.realtime_root_steps,
        "realtime_root_recovery_steps": config.realtime_root_recovery_steps,
        "realtime_root_error_threshold_m": config.realtime_root_error_threshold_m,
        "realtime_root_turn_threshold_deg": config.realtime_root_turn_threshold_deg,
        "realtime_root_translation_threshold_m": config.realtime_root_translation_threshold_m,
        "optimize_hands": config.optimize_hands,
        "mean_overall_error_m": float(np.mean([item["overall_mean_error_m"] for item in aggregate])),
        "mean_body_error_m": float(np.mean([item["body_mean_error_m"] for item in aggregate])),
        "mean_left_hand_error_m": float(np.mean([item["left_hand_mean_error_m"] for item in aggregate])),
        "mean_right_hand_error_m": float(np.mean([item["right_hand_mean_error_m"] for item in aggregate])),
    }
    timing_summary = _summarize_stage_timings(aggregate)
    if timing_summary:
        report["stage_timings_s"] = timing_summary
    root_summary = _summarize_root_adaptation(aggregate)
    if root_summary:
        report["realtime_root_adaptation"] = root_summary
    (output_dir / "trajectory_names.json").write_text(
        json.dumps(
            {
                "body": BODY_NAMES,
                "left_hand": [f"left_hand_{name}" for name in HAND_NAMES],
                "right_hand": [f"right_hand_{name}" for name in HAND_NAMES],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "retarget_report.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=False)


def _summarize_stage_timings(aggregate: list[dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    stage_names = sorted(
        {
            key
            for item in aggregate
            for key in item.get("stage_timings", {}).keys()
            if isinstance(item.get("stage_timings"), dict)
        }
    )
    summary: dict[str, dict[str, float]] = {}
    for stage_name in stage_names:
        values = np.array(
            [
                float(item["stage_timings"][stage_name])
                for item in aggregate
                if isinstance(item.get("stage_timings"), dict) and stage_name in item["stage_timings"]
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        summary[stage_name] = {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
        }
    return summary


def _summarize_root_adaptation(aggregate: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    if not all("retarget_root_steps" in item and "retarget_root_reason" in item for item in aggregate):
        return {}
    steps = np.array([int(item["retarget_root_steps"]) for item in aggregate], dtype=np.int32)
    reasons = [_scalar_string(item["retarget_root_reason"]) for item in aggregate]
    reason_counts = {reason: reasons.count(reason) for reason in sorted(set(reasons))}
    return {
        "mean_root_steps": float(np.mean(steps)),
        "max_root_steps": int(np.max(steps)),
        "reason_counts": reason_counts,
    }


def _scalar_string(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return str(value.item())
    return str(value)
