"""Exact equivalent of roc mocap --retarget --retarget-mode track pipeline."""
import time, cv2, numpy as np
from pathlib import Path
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed

from roc.config.yaml_io import load_capture_config
from roc.tracking.mediapipe_tracker import MediapipeTracker
from roc.tracking.model_paths import pose_model_path_for_complexity, hand_model_path
from roc.triangulation.cameras import load_camera_group_from_toml, camera_order_indices
from roc.triangulation.triangulate import triangulate_sequence
from roc.mocap.track import RealtimeSmplxTracker
from roc.mocap.retarget import RetargetConfig, RetargetMode

prepare_session = Path("sessions/prepare_20260603_102602")
calib_session = Path("sessions/calib_20260603_102832")
video_dir = Path("sessions/mocap_live_rkw_20260603_test01/videos")

capture_config = load_capture_config(prepare_session / "capture_config.yaml")
camera_group = load_camera_group_from_toml(calib_session / "calibration.toml")
ordered_serials = [s for s in capture_config.camera_serials if (video_dir / f"{s}.mp4").exists()]
reorder_indices = camera_order_indices(ordered_serials, [c.get_name() for c in camera_group.cameras])

pose_model = pose_model_path_for_complexity(1)
hand_model = hand_model_path()
n_frames = 100
print(f"Running roc-equivalent pipeline: {n_frames} frames, {len(ordered_serials)} cameras")

# Per-frame accumulators (matching roc mocap realtime.py)
pose_2d = []; pose_conf = []; lh_2d = []; lh_conf = []; rh_2d = []; rh_conf = []

# Tracker (EXACT CLI defaults from cli.py)
config = RetargetConfig(
    mode=RetargetMode.TRACK, device="cpu", model_dir=Path("models/smplx"),
    track_pose_steps=25,
    track_recovery_pose_steps=60,
    track_temporal_weight=0.0,    # CLI --retarget-temporal-weight default
    track_velocity_weight=0.0,    # CLI --retarget-velocity-weight default
    track_acceleration_weight=0.002,
)
tracker = RealtimeSmplxTracker(config, Path("/tmp/roc_equivalent"))

with ExitStack() as stack:
    caps = {s: cv2.VideoCapture(str(video_dir / f"{s}.mp4")) for s in ordered_serials}
    trackers_mp = {s: stack.enter_context(MediapipeTracker(
        pose_model_path=pose_model, hand_model_path=hand_model,
        model_complexity=1, hands_enabled=True, delegate="cpu"))
        for s in ordered_serials}

    for fi in range(n_frames):
        frame_set = {}
        for s in ordered_serials:
            ret, frame = caps[s].read()
            if ret: frame_set[s] = frame

        ts = fi * 33  # matching --fps 30: 1000/30 ≈ 33ms
        fps_l = []; pconf_l = []; lhp_l = []; rhp_l = []; lc_l = []; rc_l = []

        def detect_one(s):
            f = frame_set[s]; t = trackers_mp[s]
            return s, t.detect_pose(f, timestamp_ms=ts), t.detect_hands(f, timestamp_ms=ts)

        results = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(detect_one, s): s for s in ordered_serials if s in frame_set}
            for fut in as_completed(futures):
                s, pose, hand = fut.result()
                results[s] = (pose, hand)

        for s in ordered_serials:
            if s in results:
                pose, hand = results[s]
                fps_l.append(pose.xy); pconf_l.append(pose.confidence)
                lhp_l.append(hand.left_xy); rhp_l.append(hand.right_xy)
                lc_l.append(hand.left_confidence); rc_l.append(hand.right_confidence)

        pose_2d.append(np.stack(fps_l)); pose_conf.append(np.stack(pconf_l))
        lh_2d.append(np.stack(lhp_l)); lh_conf.append(np.stack(lc_l))
        rh_2d.append(np.stack(rhp_l)); rh_conf.append(np.stack(rc_l))

        # Per-frame triangulation + track (EXACTLY matching roc mocap realtime.py)
        frame_pose_np = np.stack(fps_l, axis=0).astype(np.float32)[reorder_indices]
        frame_pose_conf_np = np.stack(pconf_l, axis=0).astype(np.float32)[reorder_indices]
        frame_left_np = np.stack(lhp_l, axis=0).astype(np.float32)[reorder_indices]
        frame_left_conf_np = np.stack(lc_l, axis=0).astype(np.float32)[reorder_indices]
        frame_right_np = np.stack(rhp_l, axis=0).astype(np.float32)[reorder_indices]
        frame_right_conf_np = np.stack(rc_l, axis=0).astype(np.float32)[reorder_indices]
        frame_landmarks_2d = np.concatenate([frame_pose_np, frame_left_np, frame_right_np], axis=1)
        frame_conf = np.concatenate([frame_pose_conf_np, frame_left_conf_np, frame_right_conf_np], axis=1)
        frame_landmarks_2d = np.where(frame_conf[..., None] <= 0.1, np.nan, frame_landmarks_2d)
        frame_points_3d, _ = triangulate_sequence(camera_group, frame_landmarks_2d[:, None, :, :])
        tracker.update(fi, frame_points_3d[0])

    for c in caps.values(): c.release()

npz_path = tracker.save()
print(f"Saved: {npz_path}")

import subprocess
subprocess.run(["python3", "test/check_quality.py", str(npz_path)], check=False)
