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
