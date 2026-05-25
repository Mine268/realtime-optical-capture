from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from roc.config.models import CalibrationConfig
from roc.config.models import CameraCaptureConfig, CaptureConfig, CharucoConfig


def capture_config_to_dict(config: CaptureConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "created_at": config.created_at,
        "camera_count": config.camera_count,
        "camera_serials": config.camera_serials,
        "sync": {
            "mode": config.sync_mode,
            "fps": config.sync_fps,
        },
        "capture": {
            "pixel_format": config.pixel_format,
            "output_format": config.output_format,
            "lossless": config.lossless,
            "preview_scale": config.preview_scale,
        },
        "cameras": {
            camera.serial: {
                key: value
                for key, value in asdict(camera).items()
                if key != "serial"
            }
            for camera in config.cameras
        },
    }


def save_capture_config(path: Path, config: CaptureConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            capture_config_to_dict(config),
            handle,
            sort_keys=False,
            allow_unicode=False,
        )


def load_capture_config(path: Path) -> CaptureConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    cameras = []
    for serial, camera_data in data["cameras"].items():
        cameras.append(CameraCaptureConfig(serial=serial, **camera_data))

    return CaptureConfig(
        schema_version=data["schema_version"],
        created_at=data["created_at"],
        camera_count=data["camera_count"],
        camera_serials=list(data["camera_serials"]),
        sync_mode=data["sync"]["mode"],
        sync_fps=float(data["sync"]["fps"]),
        pixel_format=data["capture"]["pixel_format"],
        output_format=data["capture"]["output_format"],
        lossless=bool(data["capture"]["lossless"]),
        preview_scale=float(data["capture"]["preview_scale"]),
        cameras=cameras,
    )


def calibration_config_to_dict(config: CalibrationConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "created_at": config.created_at,
        "prepare_session": config.prepare_session,
        "frames": config.frames,
        "fps": config.fps,
        "mode": config.mode,
        "charuco": {
            "squares_x": config.charuco.squares_x,
            "squares_y": config.charuco.squares_y,
            "dictionary": config.charuco.dictionary,
            "square_length_mm": config.charuco.square_length_mm,
            "marker_length_mm": config.charuco.marker_length_mm,
        },
        "world": {
            "mode": config.world_mode,
        },
        "video": {
            "format": config.video_format,
            "lossless": config.lossless,
        },
    }


def save_calibration_config(path: Path, config: CalibrationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            calibration_config_to_dict(config),
            handle,
            sort_keys=False,
            allow_unicode=False,
        )


def load_calibration_config(path: Path) -> CalibrationConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    charuco = data["charuco"]
    return CalibrationConfig(
        schema_version=data["schema_version"],
        created_at=data["created_at"],
        prepare_session=data["prepare_session"],
        frames=int(data["frames"]),
        fps=float(data["fps"]),
        mode=str(data.get("mode", "capture+solve")),
        world_mode=data["world"]["mode"],
        video_format=data["video"]["format"],
        lossless=bool(data["video"]["lossless"]),
        charuco=CharucoConfig(
            squares_x=int(charuco["squares_x"]),
            squares_y=int(charuco["squares_y"]),
            dictionary=str(charuco["dictionary"]),
            square_length_mm=float(charuco["square_length_mm"]),
            marker_length_mm=float(charuco["marker_length_mm"]),
        ),
    )
