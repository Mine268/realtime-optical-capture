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
| `roc/mocap/track.py` | `RealtimeSmplxTracker`: body-only SMPL-X tracking via Adam optimisation. Composite loss with 11 terms (MSE, priors, knee/elbow angles, axis alignment, Bezier spine, sagittal, pelvis frame, hip/upper-body symmetry). Hyperparameters in `track_config.yaml`. |
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

### Track mode architecture (`roc/mocap/track.py`)

`RealtimeSmplxTracker` is a body-only SMPL-X retargeter using vectorised Adam optimisation.  It replaces the full SMPL-X iterative fitting with a lightweight optimisation loop that directly optimises body_pose (63), global_orient (3), and transl (3) via composite loss minimisation.

**Model replacement — SkeletonFK**

The default SMPL-X model computes LBS on 10,475 vertices, even with `return_verts=False`.  A custom `SkeletonFK` module replaces this with pure skeletal forward kinematics (FK): `batch_rodrigues → batch_rigid_transform → joint positions`.  No vertices, no LBS, no blend shapes.  Body joints (0-21) are computed via FK; all other joints (22-126) are derived from nearest body joints with rotation-aware offsets computed from the rest-pose full model.

Speed comparison (CPU, 9 Adam steps):

| Model | Forward | FW+BW | Track FPS |
|-------|---------|-------|-----------|
| Full SMPL-X | 5.4ms | 12.5ms | 5.6 |
| SkeletonFK | 0.3ms | 1.3ms | 17.8 |

FK body joint positions match the full model to 0.0mm at betas=0.  Hand/face landmarks have 2-9mm approximation error (rotated offsets from body joints).  The FK model outputs 127 joints to maintain drop-in compatibility with the tracker's internal joint index mapping.

The FK model is defined in `test/bench_fk_track.py` and injected via `tracker.model = fk_model`.  It is not yet integrated as the default model in `track.py`.

**Hyperparameters** are in `roc/mocap/track_config.yaml`:

| Section | Key params |
|---------|-----------|
| `optimizer.learning_rates` | body_pose=0.08, global_orient=0.05, transl=0.03 |
| `loss_weights` | knee_angle=0.04, bezier=0.12, sagittal=0.12, pelvis_frame=0.20 |
| `bezier` | p1_perp=0.12, p2_perp=0.16 (spine curvature prior) |
| `target_weights` | Per-landmark MSE weights (knees 2.3, hips 1.8, wrists 0.55, etc.) |
| `target_smooth_alpha` | Per-landmark EMA smoothing (all zero — raw 3D positions used) |
| `post_smooth.sigma` | SO(3) Gaussian filter sigma at save time (0.03) |
| `body_pose_prior_weights` | Per-joint L2 regularisation |
| `body_pose_temporal_weights` | Per-joint temporal smoothing |

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

### Track vs Fit modes

| | Track (RealtimeSmplxTracker) | Fit (RealtimeSmplxRetargeter) |
|---|---|---|
| Optimizer | Adam inline (hand-written loop) | Reference fitter's `fit_single_frame` |
| Steps | 18 steady / 60 recovery | ~2-30 root + ~120 pose + ~170 lower-body refine |
| Speed | ~150ms (6.3 FPS GPU) | ~390ms (2.6 FPS GPU) |
| Hands/Betas | Frozen to zero | Configurable |
| Input | Raw triangulated `points_3d_raw` | Same (since fix) |
| Save smoothing | SO(3) Gaussian σ=0.03 | None |

### Track mode loss function

Track mode in `roc/mocap/track.py` uses a composite loss per frame:

1. **Weighted joint MSE** — per-landmark weights in `_target_weight()`: knees 2.3×, hips 1.8×, shoulders 1.7×, elbows 1.05×, wrists 0.55×, face landmarks 0.70×, head_center 0.80×, derived centers 0.25×
2. **Pose prior** — per-joint L2 on body_pose via `_body_pose_prior_weights()`: pelvis/hips 0.050-0.060, spine 0.045, knees 0.010, ankles 0.010, wrists 0.120. Hip Y (femur twist) weighted 3× higher (0.150)
3. **Knee angle loss** (0.08) — cosine similarity of hip-knee-ankle triplets
4. **Elbow angle loss** (0.04) — same for shoulder-elbow-wrist, gated by limb length ≤0.40m
5. **Hip/shoulder axis alignment** (0.10) — horizontal axis directions in XY plane
6. **Shoulder line loss** (0.08) — point-to-line distance from SMPL-X shoulders to MediaPipe shoulder line
7. **Spine Bezier** (0.12) — cubic Bezier anchored by pelvis→neck, control points from fit data (p1_perp=0.08, p2_perp=0.12)
8. **Spine sagittal** (0.10) — penalises lateral (left-right) spine bending via hip axis reference
9. **Hip symmetry** (0.06) — penalises L/R hip rotation magnitude asymmetry
10. **Temporal** (tw=0.05) — body_pose, global_orient, transl consistency with previous frame, per-joint weights in `_body_pose_temporal_weights()`
11. **Velocity** (0.005) / **Acceleration** (0.002) damping — 2nd/3rd order finite difference

### Key track evaluation techniques

**Per-joint error vs fit baseline**: Load both track and fit `smplx_fit_sequence.npz`, extract `smplx_joints[:, :22] / input_scale` (first 22 SMPL-X body joints in mm), align by `frame_indices`, compute per-joint Euclidean distance. Use `test/compare_track_fit.py`.

**Spine curvature**: Compute spine joint positions from `smplx_joints` indices [0,3,6,9,12], measure max angle between consecutive bone segments. Track should stay within ±5° of fit.

**Pelvis/hip stability**: Check body_pose axis-angle norms for joints 0 (pelvis), 1 (left_hip), 2 (right_hip). High Y-component (>500 mrad) indicates femur twist. Compare against fit's per-joint axis-angle distributions.

**Input data source**: Always use `points_3d_raw` (triangulated, no filtering) for retargeting. `points_3d` in NPZ is EMA-filtered with a lag of ~1 frame at low FPS. The batch fit loader `_load_roc_sequence` and realtime loop were both fixed to use raw.

**Visual validation**: Render SMPL-X reprojection overlays via `render_smplx_reprojection_overlays()` to compare cyan SMPL-X skeleton against green 2D detections. Common command pattern in `test/bench_*.py` files.

**Smoothing audit**: Smoothing in track pipeline has been minimised:
- `track_temporal_weight`: 0.20→0.05
- `track_velocity_weight`: 0.02→0.005
- `track_acceleration_weight`: 0.004→0.002
- `target_smooth_alpha`: removed entirely (all zero — raw 3D positions used)
- `post_smooth.sigma`: 0.03 (mild SO(3) Gaussian at save time only)
- Target EMA (`_target_smooth_alpha`): wrist 0.20→0.05, elbow 0.35→0.10
- Per-joint temporal weights (`_body_pose_temporal_weights`): collar 2.4→1.2, spine 1.0→0.5, etc.
- SO(3) save-time Gaussian sigma: 0.30→0.03

### Bezier spine prior

Learned from fit data across 3 sessions (300 frames). The spine chain (pelvis→spine1→spine2→spine3→neck) follows a cubic Bezier with pelvis and neck as anchors. Control points:
- P1: 25% along spine, 8% forward perpendicular bend
- P2: 67% along spine, 12% forward perpendicular bend

Values were reduced from initial 27%/24% and 69%/28% (learned from all-fit average including forward-leaning poses) to prevent over-curvature in standing/walking frames. The forward direction is computed as `cross(hip_axis, spine_vector)`.

### Known issues and mitigations

- **GPU NaN in fit mode**: `alignment_rotation()` in the reference fitter divides by zero when foot landmarks are NaN (filled to 0.0). Fixed with zero-norm guards.
- **EMA lag in postprocessor**: `RealtimePostprocessor` was using configured `--fps` (e.g. 30) instead of actual FPS (~5.7), causing effective cutoff to drop from 1.2Hz to 0.23Hz. Fixed with per-frame `dt_s` parameter.
- **Spine over-curvature**: Bezier perp values must be tuned per-session. Too high causes spine to bend forward even in upright poses.
- **Face tracking weakness**: Face landmarks (nose, eyes, ears) had weights 0.25-0.35, allowing head to drift. Increased to 0.70-0.80.
