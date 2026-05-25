from __future__ import annotations

from pathlib import Path

from aniposelib.cameras import CameraGroup


def load_camera_group_from_toml(path: Path) -> CameraGroup:
    if not path.is_file():
        raise RuntimeError(f"Calibration toml not found: {path}")
    return CameraGroup.load(str(path))

