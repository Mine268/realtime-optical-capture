# Body-Only SMPL-X Realtime Tracking Plan

## Goal

Build a body-only SMPL-X realtime tracking path that can drive a character at 10 FPS or higher. The current SMPL-X fitting path remains the high-quality initializer, recovery path, and offline baseline.

## Current Baseline

Recent realtime fake-MVS profiling shows the bottleneck is SMPL-X fitting:

- MediaPipe pose + hands + triangulation + 3D postprocess: about 80-90 ms/frame.
- SMPL-X retarget: about 1.3 s/frame.
- Retarget bottleneck: `pose_optimize` about 1.1 s/frame, `lower_body_refine` about 0.17 s/frame.

Reducing `pose_steps` alone improves speed but makes SMPL-X rotations too jittery for driving because the current path is still frame-by-frame fitting with weak temporal constraints.

## Target Architecture

Add a separate realtime tracking mode:

```bash
--retarget-mode fit      # existing high-quality optimizer
--retarget-mode track    # new body-only realtime tracker
```

Tracking mode should:

- Use full fitting for the first frame and for recovery frames.
- Freeze `betas`, hands, face, expression, and jaw.
- Track only root, spine/head, arms, legs, and feet.
- Use previous SMPL-X pose as the primary state.
- Apply temporal, velocity, and acceleration smoothing on rotations.
- Run periodic medium/full fitting when error or motion spikes.

## Current Status

The first body-only `track` implementation is in place. It uses `RealtimeSmplxTracker` with warm-started Adam optimization, weighted body joint fitting, explicit knee and elbow angle losses, hip/shoulder horizontal-axis alignment, light elbow/wrist target smoothing, and limb-length gating for obvious arm outliers. It saves the same `smplx_fit_sequence.npz` schema as fit mode and recomputes `smplx_joints` after save-time smoothing.

Latest `sessions/mocap_test` 200-frame validation:

- Track mapped 3D error mean/p90/max: `0.0616 / 0.0730 / 0.1898 m`.
- Fit mapped 3D error mean/p90/max: `0.0517 / 0.0658 / 0.1926 m`.
- Track-vs-fit body joint mean/p90/max: `0.0722 / 0.0913 / 0.1544 m`.
- Frame 96 knee angle target/track/fit: left `40.4/45.0/39.8 deg`, right `39.8/46.1/37.4 deg`.
- Frame 96 elbow angle target/track/fit: left `116.8/114.4/119.3 deg`, right `112.1/112.3/112.8 deg`.
- Frames 60-115 hip-yaw error mean/p90/max: `9.7 / 21.6 / 28.5 deg`.
- Retarget-only throughput is about `5.8 FPS` on the RTX 2080 Ti test machine, so the 10 FPS target is not met yet.

## Implementation Phases

### Phase 1: Tracking Skeleton

- Add `RetargetMode` plumbing in CLI/config.
- Add `RealtimeSmplxTracker` beside the existing `RealtimeSmplxRetargeter`.
- Reuse current ROC 3D keypoint conversion and SMPL-X model loading.
- Save the same `smplx_fit_sequence.npz` schema so rendering and reprojection still work.

### Phase 2: Body-Only Geometric IK

- Estimate root orientation from hips and shoulders.
- Solve arms with two-bone IK: shoulder, elbow, wrist.
- Solve legs with two-bone IK: hip, knee, ankle.
- Estimate spine/neck/head from hips, trunk, neck, and head centers.
- Convert solved local rotations to SMPL-X body pose axis-angle.

### Phase 3: Stabilization

- Smooth rotations on SO(3), not raw 3D keypoints only.
- Add confidence/error gating per limb.
- Hold or blend limbs when triangulated keypoints are missing or unstable.
- Add recovery when fitting error, hip-axis turn, or root translation spikes.

### Phase 4: Hybrid Refinement

- Run fast tracking every frame.
- Run medium fitting every N frames or on error spikes.
- Blend fitting corrections back into the tracked state.
- Keep full fitting available for offline and calibration-quality validation.

## Success Criteria

- Body-only `track` mode reaches 10 FPS or higher on the RTX 2080 Ti test machine.
- SMPL-X root/body rotations are stable enough for character driving.
- Reprojection overlays still show reasonable body alignment.
- Existing `fit` retarget behavior remains available for quality comparison.

## Immediate Exploration Tasks

- Map ROC body keypoints to SMPL-X joint names and indices.
- Inspect SMPL-X body pose joint order used by the current reference fitter.
- Prototype root orientation and limb two-bone IK on saved `sessions/mocap_test` data.
- Compare tracked joints against current full-fitting output for a short sequence.

## Exploration Notes

### ROC Body Keypoints

`sessions/mocap_test/mocap_test.npz` contains `points_3d` with shape `(frames, 75, 3)`. The first 33 landmarks are MediaPipe body landmarks:

- Shoulders, elbows, wrists: `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`.
- Hips, knees, ankles, feet: `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`, `left_heel`, `right_heel`, `left_foot_index`, `right_foot_index`.
- Existing retarget conversion derives `hips_center`, `neck_center`, `trunk_center`, and `head_center`.

### SMPL-X Body Pose Order

SMPL-X `global_orient` controls `pelvis`. `body_pose` is 63 values: 21 local axis-angle joints, 3 values each. The order corresponds to `smplx.joint_names.JOINT_NAMES[1:22]`:

```text
0 left_hip
1 right_hip
2 spine1
3 left_knee
4 right_knee
5 spine2
6 left_ankle
7 right_ankle
8 spine3
9 left_foot
10 right_foot
11 neck
12 left_collar
13 right_collar
14 head
15 left_shoulder
16 right_shoulder
17 left_elbow
18 right_elbow
19 left_wrist
20 right_wrist
```

The corresponding SMPL-X kinematic parents for joints `0..24` are:

```text
pelvis=-1
left_hip=pelvis
right_hip=pelvis
spine1=pelvis
left_knee=left_hip
right_knee=right_hip
spine2=spine1
left_ankle=left_knee
right_ankle=right_knee
spine3=spine2
left_foot=left_ankle
right_foot=right_ankle
neck=spine3
left_collar=spine3
right_collar=spine3
head=neck
left_shoulder=left_collar
right_shoulder=right_collar
left_elbow=left_shoulder
right_elbow=right_shoulder
left_wrist=left_elbow
right_wrist=right_elbow
```

### Implementation Implication

The first tracking prototype should not attempt full inverse kinematics for all SMPL-X joints at once. Start with a tracked root frame and a small differentiable body-only correction step:

1. Use geometric root initialization from hips, shoulders, and head.
2. Freeze hands, face, shape, and expression.
3. Optimize only body joints `0..20` for a small fixed number of steps.
4. Add strong temporal/velocity/acceleration terms from the previous tracked state.
5. Use full fitting only for initialization and recovery.

This is lower risk than immediately hand-writing all local two-bone rotations, and it reuses the current SMPL-X forward model while moving toward realtime tracking.
