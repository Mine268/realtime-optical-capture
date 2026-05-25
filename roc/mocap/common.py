from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from roc.config.models import MocapConfig
from roc.config.yaml_io import load_capture_config, save_capture_config, save_mocap_config
from roc.io.sessions import MocapSessionPaths, create_mocap_session
from roc.tracking.mediapipe_tracker import HAND_LANDMARK_NAMES, POSE_LANDMARK_NAMES


def build_temp_video_writer(path: Path, frame_width: int, frame_height: int, fps_hint: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, max(fps_hint, 1.0), (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open temporary mocap video writer: {path}")
    return writer


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
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to reopen temporary mocap video: {temp_path}")
        final_path = final_video_dir / f"{serial}.mp4"
        writer = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(final_path), fourcc, actual_fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open final mocap video writer: {final_path}")
                writer.write(frame)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
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
) -> None:
    landmark_names = (
        POSE_LANDMARK_NAMES
        + [f"left_hand_{name}" for name in HAND_LANDMARK_NAMES]
        + [f"right_hand_{name}" for name in HAND_LANDMARK_NAMES]
    )
    np.savez_compressed(
        session_paths.mocap_npz_path,
        timestamps=np.array(timestamps, dtype=np.int64),
        camera_serials=np.array(capture_config.camera_serials, dtype=object),
        pose_2d=pose_2d_np,
        pose_confidence=pose_conf_np,
        left_hand_2d=left_hand_2d_np,
        left_hand_confidence=left_hand_conf_np,
        right_hand_2d=right_hand_2d_np,
        right_hand_confidence=right_hand_conf_np,
        bboxes_2d=bboxes_np,
        points_3d=points_3d.astype(np.float32),
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
            },
            handle,
            sort_keys=False,
            allow_unicode=False,
        )
