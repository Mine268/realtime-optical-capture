from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from roc.config.models import MocapConfig
from roc.config.yaml_io import load_capture_config, save_capture_config, save_mocap_config
from roc.io.sessions import MocapSessionPaths, create_mocap_session
from roc.io.video import build_mjpg_writer, encode_h264_mp4
from roc.mocap.postprocess import RealtimePostprocessor, postprocess_points_3d
from roc.tracking.mediapipe_tracker import HAND_LANDMARK_NAMES, POSE_LANDMARK_NAMES


def build_temp_video_writer(path: Path, frame_width: int, frame_height: int, fps_hint: float) -> cv2.VideoWriter:
    return build_mjpg_writer(path, frame_width, frame_height, fps_hint)


def finalize_videos_with_actual_fps(
    temp_video_paths: dict[str, Path],
    final_video_dir: Path,
    timestamps_ns: list[int],
    fallback_fps: float,
) -> float:
    if len(timestamps_ns) >= 2:
        duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
        actual_fps = (len(timestamps_ns) - 1) / duration_s if duration_s > 0 else fallback_fps
    else:
        actual_fps = fallback_fps
    actual_fps = max(actual_fps, 1.0)

    for serial, temp_path in temp_video_paths.items():
        final_path = final_video_dir / f"{serial}.mp4"
        try:
            encode_h264_mp4(temp_path, final_path, actual_fps)
        finally:
            temp_path.unlink(missing_ok=True)

    return actual_fps


def prepare_mocap_session(
    prepare_session: Path,
    calib_session: Path,
    session_root: Path,
    mode: str,
    fps: float,
    max_frames: int,
    hands_enabled: bool,
    model_complexity: int,
) -> tuple[MocapSessionPaths, object]:
    capture_config_path = prepare_session / "capture_config.yaml"
    calibration_yaml_path = calib_session / "calibration.yaml"
    if not capture_config_path.is_file():
        raise RuntimeError(f"Prepare capture_config.yaml not found: {capture_config_path}")
    if not calibration_yaml_path.is_file():
        raise RuntimeError(f"Calibration yaml not found: {calibration_yaml_path}")

    capture_config = load_capture_config(capture_config_path)
    session_paths = create_mocap_session(session_root)
    save_capture_config(session_paths.capture_config_path, capture_config)
    session_paths.calibration_yaml_path.write_text(calibration_yaml_path.read_text(encoding="utf-8"), encoding="utf-8")

    mocap_config = MocapConfig(
        schema_version=1,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        prepare_session=str(prepare_session),
        calib_session=str(calib_session),
        mode=mode,
        fps=fps,
        max_frames=max_frames,
        hands_enabled=hands_enabled,
        model_complexity=model_complexity,
        video_format="mp4",
        lossless=False,
    )
    save_mocap_config(session_paths.mocap_config_path, mocap_config)
    return session_paths, capture_config


def save_mocap_outputs(
    session_paths: MocapSessionPaths,
    capture_config,
    camera_serials: list[str] | None,
    timestamps,
    pose_2d_np: np.ndarray,
    pose_conf_np: np.ndarray,
    left_hand_2d_np: np.ndarray,
    left_hand_conf_np: np.ndarray,
    right_hand_2d_np: np.ndarray,
    right_hand_conf_np: np.ndarray,
    bboxes_np: np.ndarray,
    points_3d: np.ndarray,
    reprojection: np.ndarray,
    fps: float,
    hands_enabled: bool,
    model_complexity: int,
    postprocess_mode: str = "offline",
) -> None:
    landmark_names = (
        POSE_LANDMARK_NAMES
        + [f"left_hand_{name}" for name in HAND_LANDMARK_NAMES]
        + [f"right_hand_{name}" for name in HAND_LANDMARK_NAMES]
    )
    points_3d_raw = points_3d.astype(np.float32)
    if postprocess_mode == "offline":
        points_3d_processed, postprocess_report = postprocess_points_3d(points_3d_raw, fps=fps)
    elif postprocess_mode == "realtime":
        online_filter = RealtimePostprocessor(
            num_landmarks=points_3d_raw.shape[1],
            fps=fps,
            cutoff_hz=1.2,
            max_hold_frames=3,
        )
        points_3d_processed = np.stack([online_filter.update(frame) for frame in points_3d_raw], axis=0)
        postprocess_report = online_filter.report()
    else:
        raise ValueError(f"Unsupported postprocess mode: {postprocess_mode}")
    np.savez_compressed(
        session_paths.mocap_npz_path,
        timestamps=np.array(timestamps, dtype=np.int64),
        camera_serials=np.array(camera_serials or capture_config.camera_serials, dtype=object),
        pose_2d=pose_2d_np,
        pose_confidence=pose_conf_np,
        left_hand_2d=left_hand_2d_np,
        left_hand_confidence=left_hand_conf_np,
        right_hand_2d=right_hand_2d_np,
        right_hand_confidence=right_hand_conf_np,
        bboxes_2d=bboxes_np,
        points_3d=points_3d_processed.astype(np.float32),
        points_3d_raw=points_3d_raw,
        reprojection_error=reprojection.astype(np.float32),
        landmark_names=np.array(landmark_names, dtype=object),
    )
    with session_paths.mocap_report_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "frames": len(timestamps),
                "fps": fps,
                "hands_enabled": hands_enabled,
                "model_complexity": model_complexity,
                "output_npz": str(session_paths.mocap_npz_path),
                "postprocess_mode": postprocess_mode,
                "postprocess": postprocess_report.to_dict(),
            },
            handle,
            sort_keys=False,
            allow_unicode=False,
        )
