# SMPL-X Track Mode: Technical Reference

## Overview

Track mode (`--retarget-mode track`) is a body-only SMPL-X tracker for realtime character driving. It keeps the same output schema as fit mode (`smplx_fit_sequence.npz`) but replaces the reference fitter with a short Adam update from the previous frame. It freezes betas, hands, face, expression, and jaw.

Fit mode remains the quality baseline. Track mode is faster and more stable for body rotations, but it is still below the 10 FPS target on the RTX 2080 Ti test sequence.

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

Adam learning rates are `0.08` for body pose, `0.05` for root orientation, and `0.03` for translation. The first frame uses 55 steps; later frames use 18 steps.

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
track mapped 3D error mean/p90/max: 0.0616 / 0.0730 / 0.1898 m
fit   mapped 3D error mean/p90/max: 0.0517 / 0.0658 / 0.1926 m
track-vs-fit body joint mean/p90/max: 0.0722 / 0.0913 / 0.1544 m
track wrist current-frame error mean/p90: left 0.0906/0.1400 m, right 0.0806/0.1297 m
track wrist acceleration p90/max: 0.2427 / 0.3742 m
hip-yaw error on frames 60-115 mean/p90/max: 9.7 / 21.6 / 28.5 deg
track retarget-only throughput: about 5.8 FPS
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

- Track mode is not yet 10 FPS on the test machine; the current full 200-frame retarget-only run is about 5.9 FPS.
- Wrist positions now follow current-frame keypoints better than the earlier smoothed tracker, at the cost of higher wrist acceleration than fit mode.
- The visible SMPL-X hip joints are close together because `left_hip` and `right_hip` are internal pelvis joints, not MediaPipe body-surface hip landmarks. Fit mode shows the same hip width.
- The final track quality depends on triangulated 3D quality. Limb-length gating prevents obvious elbow outliers from dominating the loss, but it does not replace a full confidence-aware limb tracker.
- Save-time smoothing changes stored rotations; the implementation recomputes `smplx_joints` after smoothing so reprojection videos match the saved poses.
