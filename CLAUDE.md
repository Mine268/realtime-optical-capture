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

This is a fixed multi-view real-time optical (markerless) pose estimation system. It drives synchronized industrial cameras (MVS SDK), calibrates them with a ChArUco board, then captures multi-view video and triangulates 3D human pose using MediaPipe. Optionally, the triangulated 3D keypoints can be retargeted to SMPL-X joint rotations via gradient-based fitting.

**Three-stage pipeline** — each stage produces a timestamped `sessions/<stage>_YYYYmmdd_HHMMSS/` directory that feeds into the next:

1. **`prepare`** (`roc/prepare/app.py`) — enumerates MVS cameras, opens an OpenCV preview window for per-camera exposure/gain tuning, saves `capture_config.yaml`.
2. **`calib`** (`roc/calib/`) — records multi-camera video of a ChArUco board, then solves intrinsics/extrinsics via `aniposelib`. Outputs `calibration.toml` + `calibration.yaml`. Supports `camera0` (origin at camera 0) and `ground` (ChArUco board defines ground plane) world coordinate modes.
3. **`mocap`** (`roc/mocap/`) — three sub-modes:
   - `realtime`: captures from live cameras + runs MediaPipe + triangulation per frame. Uses causal online postprocessing (EMA low-pass + short hold for missing points). Optionally runs per-frame SMPL-X retargeting inline.
   - `capture`: raw frame capture only (no pose estimation), saves raw BMPs then transcodes to H.264 mp4 via ffmpeg. Uses `SyncCaptureWorker` threads for synchronized multi-camera trigger+grab.
   - `capture_estimate`: offline pose estimation from pre-recorded mp4 files. Uses zero-phase Butterworth low-pass filtering (via `scipy.signal.filtfilt`). SMPL-X retargeting runs as a batch after triangulation.

### Key modules

| Module | Purpose |
|---|---|
| `roc/cli.py` | argparse CLI with 3 subcommands (`prepare`, `calib`, `mocap`). Lazy-imports stage modules. |
| `roc/config/` | Dataclass models (`CaptureConfig`, `CalibrationConfig`, `MocapConfig`) and YAML serialization/deserialization. |
| `roc/mvs/` | MVS SDK wrapper. `MvsSystem`/`MvsCamera` wrap the C++ SDK at `/opt/MVS/`. `OfflineMvsSystem`/`OfflineMvsCamera` provide the same interface backed by mp4 files or image folders — used with `--offline-source-dir` for testing without hardware. |
| `roc/tracking/` | `MediapipeTracker` wraps MediaPipe PoseLandmarker + HandLandmarker in VIDEO mode. Model paths are in `models/mediapipe/`. Supports CPU/GPU delegate selection. |
| `roc/triangulation/` | `triangulate_sequence()` uses `aniposelib.cameras.CameraGroup.triangulate()` for 3D reconstruction. `cameras.py` handles loading calibration TOML and reordering cameras to match the calibrated order. |
| `roc/mocap/postprocess.py` | Two postprocessing pipelines: `postprocess_points_3d()` (offline: velocity outlier removal + short-gap interpolation + zero-phase Butterworth) and `RealtimePostprocessor` (online: causal EMA low-pass + brief hold for dropouts). |
| `roc/mocap/retarget.py` | SMPL-X skeleton fitting. `RealtimeSmplxRetargeter` does per-frame fitting inline during realtime mocap with adaptive root-step selection. `run_mocap_retarget()` does batch fitting for offline mode. Dynamically imports the reference fitting script via `importlib`. |
| `roc/io/` | `sessions.py` creates session directory structures. `video.py` provides `H264VideoWriter` (writes MJPG temp file then ffmpeg transcodes to H.264 baseline yuv420p). |
| `roc/mocap/render_*.py` | Standalone scripts for generating 2D overlay videos, 3D skeleton preview mp4, 3D reprojection diagnostic videos, and SMPL-X reprojection overlays from npz outputs. |
| `roc/mocap/benchmark_realtime.py` | Standalone benchmarking tool that measures per-model throughput using pre-recorded video. |
| `roc/mocap/ablate_filters.py` | Offline comparison of postprocessing filter configurations against ground truth. |
| `roc/mocap/analyze_jitter.py` | Quantifies jitter characteristics (velocity, acceleration) in triangulated 3D sequences. |
| `roc/mocap/tune_postprocess.py` | Grid-search tool for tuning postprocessing hyperparameters. |

### Session directory structure

```
sessions/
  prepare_YYYYmmdd_HHMMSS/
    capture_config.yaml          # Camera enumeration + per-camera exposure/gain
    preview_snapshot/             # Optional preview snapshots
    logs/
  calib_YYYYmmdd_HHMMSS/
    calibration.toml              # aniposelib CameraGroup (binary)
    calibration.yaml              # Human-readable calibration export
    calibration_report.yaml       # Per-camera reprojection error statistics
    calibration_visualization.png # 3D camera/scene plot
    charuco_2d.npz / charuco_3d.npy  # Intermediate ChArUco detections
    capture_config.yaml           # Copy from prepare session
    calib_config.yaml             # Calibration settings snapshot
    videos/                       # Per-camera calibration capture mp4s
    charuco_overlays/             # Per-frame ChArUco overlay images
    logs/
  mocap_YYYYmmdd_HHMMSS/
    mocap_YYYYmmdd_HHMMSS.npz     # All 2D/3D keypoints, confidences, reprojection errors
    mocap_report.yaml             # Frame count, fps, postprocessing summary
    mocap_config.yaml             # Mocap settings snapshot
    calibration.yaml              # Copy from calib session
    capture_config.yaml           # Copy from prepare session
    videos/                       # Per-camera mp4s (captured or copied)
    overlay_videos/               # 2D pose overlay mp4s
    reprojection_videos/          # 3D reprojection diagnostic mp4s
    pose_videos/                  # 3D skeleton preview mp4
    smplx_retarget/               # SMPL-X fitting outputs (if --retarget)
    logs/
```

### Data flow (realtime mocap)

```
MVS cameras (software trigger)
  → MvsCamera.snapshot() per camera per frame-set
  → MediapipeTracker.detect_pose() + detect_hands() per view
  → accumulate 2D keypoints across cameras
  → camera_order_indices() reorder to match calibration
  → triangulate_sequence() → points_3d_raw
  → RealtimePostprocessor.update() per frame → points_3d
  → [optional] RealtimeSmplxRetargeter.update() per frame → smplx joint rotations
  → save_mocap_outputs() → .npz + mocap_report.yaml
  → render_all_overlays() + render_reprojection_overlays() + render_npz_to_video()
```

### SMPL-X retargeting architecture

The SMPL-X fitting pipeline in `roc/mocap/retarget.py` is a thin wrapper around a reference implementation at `refs/smplx_from_freemocap_3d/fit_freemocap_smplx.py`. Rather than vendoring or forking the reference script, `retarget.py` loads it at runtime via `importlib.util.spec_from_file_location()` and calls its functions directly (`optimize_shared_betas`, `fit_single_frame`, `create_model`).

The retargeter maps the 75 MediaPipe landmarks (33 pose + 21 left hand + 21 right hand) into the body/hand landmark taxonomies expected by the reference fitter, using BODY_NAMES (37 landmarks with derived hip/trunk/neck/head centers) and HAND_NAMES (21 per hand).

Two modes:
- **Offline batch** (`run_mocap_retarget`): fits shared betas from all frames, then per-frame pose optimization with optional temporal smoothing. Called after triangulation completes.
- **Realtime** (`RealtimeSmplxRetargeter`): fits per-frame inline during capture. Uses adaptive root-step selection based on body error, turn angle, and translation magnitude to balance speed and stability. Default uses 2 root steps for steady frames, higher for init/turn/translation/high-error frames.

Dependencies (`torch`, `smplx`, `trimesh`, optionally `human_body_prior` for VPoser) are not in `pyproject.toml` — they must be installed separately in the venv before using `--retarget`.

### 2D→3D shape conventions

- `pose_2d`: `(num_cameras, num_frames, num_pose_landmarks, 2)` — raw pixel coordinates
- `points_3d`: `(num_frames, num_landmarks, 3)` — triangulated world coordinates (millimeters)
- Landmark names are stored in the npz as `landmark_names` array: 33 MediaPipe pose names + 21 left hand + 21 right hand
- 2D points with confidence ≤ 0.1 are set to NaN before triangulation
- Camera ordering in npz arrays (`camera_serials`) reflects the calibrated order, not the capture order
- Calibration coordinate units are millimeters; SMPL-X retargeting scales input by `--retarget-input-scale` (default 0.001) to convert to meters

### Environment requirements

- MVS SDK at `/opt/MVS/` with Python bindings at `/opt/MVS/Samples/64/Python/MvImport/`
- `ffmpeg` on PATH for H.264 video encoding
- GUI environment for OpenCV preview windows (all stages support `--show-preview`)
- MediaPipe model files in `models/mediapipe/` (not committed; must be downloaded separately)
- SMPL-X model files in `models/smplx/` (not committed; required for `--retarget`)
- VPoser model files in `models/vposer_v1_0/` (not committed; required for `--retarget-use-vposer`)
- GPU delegate requires working EGL/OpenGL context (currently non-functional in this environment)
- SMPL-X retargeting requires `torch`, `smplx`, `trimesh` in the venv (not managed by pyproject.toml)
