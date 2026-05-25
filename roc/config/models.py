from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CameraCaptureConfig:
    serial: str
    enabled: bool = True
    index_hint: int = 0
    exposure_us: float = 8000.0
    gain_db: float = 6.0
    width: int = 0
    height: int = 0
    model_name: str = ""
    transport_type: str = ""


@dataclass(slots=True)
class CaptureConfig:
    schema_version: int
    created_at: str
    camera_count: int
    camera_serials: list[str] = field(default_factory=list)
    sync_mode: str = "software_trigger"
    sync_fps: float = 5.0
    pixel_format: str = "BayerRG8"
    output_format: str = "mp4"
    lossless: bool = False
    preview_scale: float = 0.5
    cameras: list[CameraCaptureConfig] = field(default_factory=list)


@dataclass(slots=True)
class CharucoConfig:
    squares_x: int = 7
    squares_y: int = 5
    dictionary: str = "DICT_4X4_250"
    square_length_mm: float = 190.5
    marker_length_mm: float = 152.4


@dataclass(slots=True)
class CalibrationConfig:
    schema_version: int
    created_at: str
    prepare_session: str
    frames: int = 120
    fps: float = 3.0
    world_mode: str = "camera0"
    video_format: str = "mp4"
    lossless: bool = False
    mode: str = "capture+solve"
    charuco: CharucoConfig = field(default_factory=CharucoConfig)


@dataclass(slots=True)
class MocapConfig:
    schema_version: int
    created_at: str
    prepare_session: str
    calib_session: str
    mode: str = "realtime"
    fps: float = 20.0
    max_frames: int = 0
    hands_enabled: bool = True
    model_complexity: int = 1
    video_format: str = "mp4"
    lossless: bool = False
