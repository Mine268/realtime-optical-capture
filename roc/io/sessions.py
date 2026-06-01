from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class PrepareSessionPaths:
    session_dir: Path
    snapshots_dir: Path
    logs_dir: Path
    capture_config_path: Path


@dataclass(slots=True)
class CalibrationSessionPaths:
    session_dir: Path
    videos_dir: Path
    raw_frames_dir: Path
    logs_dir: Path
    capture_config_path: Path
    calib_config_path: Path
    calibration_toml_path: Path
    calibration_yaml_path: Path
    charuco_2d_path: Path
    charuco_3d_path: Path
    calibration_report_path: Path
    calibration_visualization_path: Path
    charuco_overlays_dir: Path


@dataclass(slots=True)
class MocapSessionPaths:
    session_dir: Path
    videos_dir: Path
    raw_frames_dir: Path
    logs_dir: Path
    capture_config_path: Path
    calibration_yaml_path: Path
    mocap_config_path: Path
    mocap_npz_path: Path
    mocap_report_path: Path
    overlay_videos_dir: Path


def create_prepare_session(root: Path) -> PrepareSessionPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = root / f"prepare_{timestamp}"
    snapshots_dir = session_dir / "preview_snapshot"
    logs_dir = session_dir / "logs"
    capture_config_path = session_dir / "capture_config.yaml"

    snapshots_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return PrepareSessionPaths(
        session_dir=session_dir,
        snapshots_dir=snapshots_dir,
        logs_dir=logs_dir,
        capture_config_path=capture_config_path,
    )


def create_calibration_session(root: Path) -> CalibrationSessionPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = root / f"calib_{timestamp}"
    videos_dir = session_dir / "videos"
    raw_frames_dir = session_dir / "raw_frames"
    logs_dir = session_dir / "logs"

    videos_dir.mkdir(parents=True, exist_ok=False)
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    charuco_overlays_dir = session_dir / "charuco_overlays"
    charuco_overlays_dir.mkdir(parents=True, exist_ok=True)

    return CalibrationSessionPaths(
        session_dir=session_dir,
        videos_dir=videos_dir,
        raw_frames_dir=raw_frames_dir,
        logs_dir=logs_dir,
        capture_config_path=session_dir / "capture_config.yaml",
        calib_config_path=session_dir / "calib_config.yaml",
        calibration_toml_path=session_dir / "calibration.toml",
        calibration_yaml_path=session_dir / "calibration.yaml",
        charuco_2d_path=session_dir / "charuco_2d.npz",
        charuco_3d_path=session_dir / "charuco_3d.npy",
        calibration_report_path=session_dir / "calibration_report.yaml",
        calibration_visualization_path=session_dir / "calibration_visualization.png",
        charuco_overlays_dir=charuco_overlays_dir,
    )


def get_existing_mocap_session(session_dir: Path) -> MocapSessionPaths:
    session_dir = session_dir.resolve()
    if not session_dir.is_dir():
        raise RuntimeError(f"Mocap session directory not found: {session_dir}")

    videos_dir = session_dir / "videos"
    raw_frames_dir = session_dir / "raw_frames"
    logs_dir = session_dir / "logs"
    overlay_videos_dir = session_dir / "overlay_videos"

    videos_dir.mkdir(parents=True, exist_ok=True)
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    overlay_videos_dir.mkdir(parents=True, exist_ok=True)

    timestamp = session_dir.name.removeprefix("mocap_")
    return MocapSessionPaths(
        session_dir=session_dir,
        videos_dir=videos_dir,
        raw_frames_dir=raw_frames_dir,
        logs_dir=logs_dir,
        capture_config_path=session_dir / "capture_config.yaml",
        calibration_yaml_path=session_dir / "calibration.yaml",
        mocap_config_path=session_dir / "mocap_config.yaml",
        mocap_npz_path=session_dir / f"mocap_{timestamp}.npz",
        mocap_report_path=session_dir / "mocap_report.yaml",
        overlay_videos_dir=overlay_videos_dir,
    )
