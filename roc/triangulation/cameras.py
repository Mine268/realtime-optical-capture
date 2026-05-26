from __future__ import annotations

from pathlib import Path

from aniposelib.cameras import CameraGroup


def load_camera_group_from_toml(path: Path) -> CameraGroup:
    if not path.is_file():
        raise RuntimeError(f"Calibration toml not found: {path}")
    return CameraGroup.load(str(path))


def camera_group_names(camera_group: CameraGroup) -> list[str]:
    return [camera.get_name() for camera in camera_group.cameras]


def camera_order_indices(source_serials: list[str], target_serials: list[str]) -> list[int]:
    source_index = {serial: index for index, serial in enumerate(source_serials)}
    missing = [serial for serial in target_serials if serial not in source_index]
    if missing:
        raise RuntimeError(f"Missing cameras for calibrated order: {missing}")
    return [source_index[serial] for serial in target_serials]
