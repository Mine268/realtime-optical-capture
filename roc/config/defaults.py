DEFAULT_EXPOSURE_US = 8000.0
DEFAULT_GAIN_DB = 6.0
DEFAULT_PREVIEW_SCALE = 0.5
DEFAULT_PIXEL_FORMAT = "BayerRG8"
DEFAULT_SYNC_MODE = "software_trigger"
DEFAULT_TRIGGER_FPS = 5.0

DEFAULT_CAPTURE_CONFIG = {
    "schema_version": 1,
    "sync": {
        "mode": DEFAULT_SYNC_MODE,
        "fps": DEFAULT_TRIGGER_FPS,
    },
    "capture": {
        "pixel_format": DEFAULT_PIXEL_FORMAT,
        "output_format": "mp4",
        "lossless": False,
        "preview_scale": DEFAULT_PREVIEW_SCALE,
    },
}

