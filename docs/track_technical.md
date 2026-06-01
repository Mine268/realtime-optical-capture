# SMPL-X Track Mode: Technical Reference

## Overview

Track mode (`--retarget-mode track`) is a body-only SMPL-X tracker for realtime character driving. It keeps the same output schema as fit mode (`smplx_fit_sequence.npz`) but replaces the reference fitter with a short Adam update from the previous frame. It freezes betas, hands, face, expression, and jaw.

Fit mode remains the quality baseline. Track mode is faster and more stable for body rotations, but the usable 18-step setting is still below the 10 FPS target on the RTX 2080 Ti test sequence.

## Fit vs Track Usage

Use `fit` when quality is more important than latency: offline reconstruction, baseline comparison, suspicious-frame debugging, and future tracker initialization or recovery references. It runs the reference optimizer per frame, supports lower-body refine and optional hand fitting, and is currently about `0.61 FPS` on the RTX 2080 Ti 20-frame wall-clock baseline.

Use `track` when latency matters: body-only realtime SMPL-X tracking, character driving, and quick previews. The current usable setting is 18 steady Adam steps via `--retarget-pose-steps 18`. Six-step runs can approach 10 FPS, but the rotations are visibly jittery and should not be used for driving.

Typical fit command:

```bash
roc mocap \
  --mode realtime \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --mocap-session sessions/mocap_test \
  --frames 200 \
  --offline-source-dir sessions/mocap_test/videos \
  --inference-device gpu \
  --model-complexity 2 \
  --no-hands \
  --retarget \
  --retarget-mode fit \
  --retarget-model-dir models/smplx \
  --profile
```

Typical track command:

```bash
roc mocap \
  --mode realtime \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --mocap-session sessions/mocap_test \
  --frames 200 \
  --offline-source-dir sessions/mocap_test/videos \
  --inference-device gpu \
  --model-complexity 2 \
  --no-hands \
  --retarget \
  --retarget-mode track \
  --retarget-model-dir models/smplx \
  --retarget-pose-steps 18 \
  --profile
```

## Pipeline

```text
ROC 3D keypoints, mm
  -> scale to meters
  -> map body landmarks to SMPL-X joints
  -> causal smoothing for elbow/wrist targets
  -> Adam optimize body_pose(63), global_orient(3), transl(3)
  -> losses: weighted joints + pose prior + knee/elbow angles + hip/shoulder axes + temporal terms
  -> save-time SO(3)-space Gaussian smoothing
  -> refresh smplx_joints from the smoothed rotations
```

The realtime loop calls `RealtimeSmplxTracker.update()` immediately after 3D triangulation and causal postprocessing. `save()` only aggregates already-produced frame results.

## Optimization State

| Parameter | Status |
|---|---|
| `body_pose` | Optimized, warm-started from previous frame |
| `global_orient` | Optimized, warm-started |
| `transl` | Optimized, warm-started |
| `betas` | Fixed zeros |
| hands | Fixed zeros |
| VPoser | Not used |

Adam learning rates are `0.08` for body pose, `0.05` for root orientation, and `0.03` for translation. In track mode, `--retarget-pose-steps` controls steady-frame Adam steps with a CLI cap of 20. The current usable-quality setting is 18 steady steps; first-frame and recovery updates use a higher automatic budget.

## Losses

Track mode uses weighted joint MSE rather than treating all mapped landmarks equally. Real landmarks such as hips, shoulders, knees, ankles, elbows, and wrists drive the fit. Derived centers such as `hips_center`, `trunk_center`, `neck_center`, and `head_center` have low weight because they are averages of MediaPipe landmarks and do not exactly match SMPL-X anatomical joints.

Additional constraints:

- Knee angle loss aligns SMPL-X and triangulated `hip-knee-ankle` angles, which fixes deep squat frames where joint MSE alone kept knees too straight.
- Elbow angle loss aligns `shoulder-elbow-wrist` angles, but only when target upper-arm and forearm lengths are plausible (`<= 0.40 m`).
- Hip and shoulder horizontal-axis loss keeps root orientation responsive during jump turns.
- Elbow/wrist target positions are lightly smoothed before optimization to reduce single-frame hand jitter without adding a full-frame wrist lag.
- Joint-specific temporal, velocity, and acceleration losses are tuned lower on wrists than earlier experiments so wrist joints can follow current keypoints.

## Validation

Latest full-sequence check on `sessions/mocap_test` used 200 frames and compared track against both triangulated 3D points and per-frame fit results.

```text
track 18-step realtime profile from sessions/mocap_test/logs/mocap.log:
  all 200 frames including first-frame initialization: 208.1 ms/frame, 4.80 FPS
  steady frames 1-199: loop 199.9 ms/frame, 5.00 FPS
  steady estimate_only mean/p50/p95: 53.6 / 52.8 / 59.9 ms
  steady smplx_retarget mean/p50/p95: 146.3 / 144.1 / 167.1 ms
  track body_err mean/p50/p95: 0.048 / 0.047 / 0.062 m

track 6-step realtime profile:
  loop about 95-108 ms/frame, near 9-10 FPS, but visible jitter makes it unusable for driving

fit mode wall-clock baseline, first 20 frames:
  1629.8 ms/frame, 0.61 FPS
  previous 50-frame fit report mean_body_error: 0.0416 m
  track 18-step 200-frame report mean_body_error: 0.0481 m

track mapped 3D error mean/p90/max: 0.0616 / 0.0730 / 0.1898 m
fit   mapped 3D error mean/p90/max: 0.0517 / 0.0658 / 0.1926 m
track-vs-fit body joint mean/p90/max: 0.0722 / 0.0913 / 0.1544 m
track wrist current-frame error mean/p90: left 0.0906/0.1400 m, right 0.0806/0.1297 m
track wrist acceleration p90/max: 0.2427 / 0.3742 m
hip-yaw error on frames 60-115 mean/p90/max: 9.7 / 21.6 / 28.5 deg
track retarget-only throughput: about 5.8 FPS on the earlier benchmark
```

Frame 96 deep squat angle check:

```text
knee angle target/track/fit:
  left  40.4 / 45.0 / 39.8 deg
  right 39.8 / 46.1 / 37.4 deg

elbow angle target/track/fit:
  left  116.8 / 114.4 / 119.3 deg
  right 112.1 / 112.3 / 112.8 deg
```

## Known Tradeoffs

- Track mode is not yet 10 FPS at usable quality on the test machine. The 18-step realtime run is about 5.0 FPS steady-state for full estimate+track; 6-step runs can approach 10 FPS but are too jittery for driving.
- Wrist positions now follow current-frame keypoints better than the earlier smoothed tracker, at the cost of higher wrist acceleration than fit mode.
- The visible SMPL-X hip joints are close together because `left_hip` and `right_hip` are internal pelvis joints, not MediaPipe body-surface hip landmarks. Fit mode shows the same hip width.
- The final track quality depends on triangulated 3D quality. Limb-length gating prevents obvious elbow outliers from dominating the loss, but it does not replace a full confidence-aware limb tracker.
- Save-time smoothing changes stored rotations; the implementation recomputes `smplx_joints` after smoothing so reprojection videos match the saved poses.
