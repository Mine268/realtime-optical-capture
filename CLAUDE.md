# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install the package in editable mode (venv must already exist)
env UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python -e .

# CLI entry point (after install)
roc <command> ...

# Or run directly without install
python -m roc.cli <command> ...

# Tab completion (argcomplete)
source .venv/bin/activate && eval "$(register-python-argcomplete roc)"
```

There is no test suite, linting, or type-checking setup. `pyproject.toml` defines the package metadata and dependencies only.

## Architecture

This is a fixed multi-view real-time optical (markerless) pose estimation system. It drives synchronized industrial cameras (MVS SDK), calibrates them with a ChArUco board, then captures multi-view video and triangulates 3D human pose using MediaPipe.

**Three-stage pipeline** — each stage produces a timestamped `sessions/<stage>_YYYYmmdd_HHMMSS/` directory that feeds into the next:

1. **`prepare`** (`roc/prepare/app.py`) — enumerates MVS cameras, opens an OpenCV preview window for per-camera exposure/gain tuning, saves `capture_config.yaml`.
2. **`calib`** (`roc/calib/`) — records multi-camera video of a ChArUco board, then solves intrinsics/extrinsics via `aniposelib`. Outputs `calibration.toml` + `calibration.yaml`. Supports `camera0` and `ground` world coordinate modes.
3. **`mocap`** (`roc/mocap/`) — three sub-modes:
   - `realtime`: captures from live cameras + runs MediaPipe + triangulation per frame. Uses causal online postprocessing (EMA low-pass + short hold for missing points).
   - `capture`: raw frame capture only (no pose estimation), saves raw BMPs then transcodes to H.264 mp4 via ffmpeg. Uses `SyncCaptureWorker` threads for synchronized multi-camera trigger+grab.
   - `capture_estimate`: offline pose estimation from pre-recorded mp4 files. Uses zero-phase Butterworth low-pass filtering (via `scipy.signal.filtfilt`).

### Key modules

| Module | Purpose |
|---|---|
| `roc/cli.py` | argparse CLI with 3 subcommands (`prepare`, `calib`, `mocap`). Lazy-imports stage modules. |
| `roc/config/` | Dataclass models (`CaptureConfig`, `CalibrationConfig`, `MocapConfig`) and YAML serialization/deserialization. |
| `roc/mvs/` | MVS SDK wrapper. `MvsSystem`/`MvsCamera` wrap the C++ SDK at `/opt/MVS/`. `OfflineMvsSystem`/`OfflineMvsCamera` provide the same interface backed by mp4 files or image folders — used with `--offline-source-dir` for testing without hardware. |
| `roc/tracking/` | `MediapipeTracker` wraps MediaPipe PoseLandmarker + HandLandmarker in VIDEO mode. Model paths are in `models/mediapipe/`. Supports CPU/GPU delegate selection. |
| `roc/triangulation/` | `triangulate_sequence()` uses `aniposelib.cameras.CameraGroup.triangulate()` for 3D reconstruction. `cameras.py` handles loading calibration TOML and reordering cameras to match the calibrated order. |
| `roc/mocap/postprocess.py` | Two postprocessing pipelines: `postprocess_points_3d()` (offline: velocity outlier removal + short-gap interpolation + zero-phase Butterworth) and `RealtimePostprocessor` (online: causal EMA low-pass + brief hold for dropouts). |
| `roc/io/` | `sessions.py` creates session directory structures. `video.py` provides `H264VideoWriter` (writes MJPG temp file then ffmpeg transcodes to H.264 baseline yuv420p). |
| `roc/mocap/render_*.py` | Standalone scripts for generating 2D overlay videos, 3D skeleton preview mp4, and 3D reprojection diagnostic videos from npz outputs. |
| `roc/mocap/benchmark_realtime.py` | Standalone benchmarking tool that measures per-model throughput using pre-recorded video. |

### Data flow (realtime mocap)

```
MVS cameras (software trigger)
  → MvsCamera.snapshot() per camera per frame-set
  → MediapipeTracker.detect_pose() + detect_hands() per view
  → accumulate 2D keypoints across cameras
  → camera_order_indices() reorder to match calibration
  → triangulate_sequence() → points_3d_raw
  → RealtimePostprocessor.update() per frame → points_3d
  → save_mocap_outputs() → .npz + mocap_report.yaml
```

### 2D→3D shape conventions

- `pose_2d`: `(num_cameras, num_frames, num_pose_landmarks, 2)` — raw pixel coordinates
- `points_3d`: `(num_frames, num_landmarks, 3)` — triangulated world coordinates
- Landmark names are stored in the npz as `landmark_names` array: 33 MediaPipe pose names + 21 left hand + 21 right hand
- 2D points with confidence ≤ 0.1 are set to NaN before triangulation
- Camera ordering in npz arrays (`camera_serials`) reflects the calibrated order, not the capture order

### Environment requirements

- MVS SDK at `/opt/MVS/` with Python bindings at `/opt/MVS/Samples/64/Python/MvImport/`
- `ffmpeg` on PATH for H.264 video encoding
- GUI environment for OpenCV preview windows (all stages support `--show-preview`)
- MediaPipe model files in `models/mediapipe/` (not committed; must be downloaded separately)
- GPU delegate requires working EGL/OpenGL context (currently non-functional in this environment)
