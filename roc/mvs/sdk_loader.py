from __future__ import annotations

import os
import sys
from pathlib import Path


MVS_PYTHON_SAMPLE_DIR = Path("/opt/MVS/Samples/64/Python")
MVS_IMPORT_DIR = MVS_PYTHON_SAMPLE_DIR / "MvImport"


def ensure_mvs_python_path() -> None:
    if str(MVS_IMPORT_DIR) not in sys.path:
        sys.path.insert(0, str(MVS_IMPORT_DIR))

    os.environ.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")

