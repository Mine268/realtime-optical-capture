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
