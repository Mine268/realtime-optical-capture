from __future__ import annotations

from pathlib import Path


def pose_model_path_for_complexity(model_complexity: int) -> Path:
    paths = {
        0: Path("models/mediapipe/pose_landmarker_lite.task"),
        1: Path("models/mediapipe/pose_landmarker_full.task"),
        2: Path("models/mediapipe/pose_landmarker_heavy.task"),
    }
    try:
        return paths[model_complexity]
    except KeyError as exc:
        raise ValueError(f"Unsupported pose model complexity: {model_complexity}") from exc


def hand_model_path() -> Path:
    return Path("models/mediapipe/hand_landmarker.task")
