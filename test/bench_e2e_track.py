"""End-to-end track pipeline test with offline video source."""
import time, sys, cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from contextlib import ExitStack

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
serials = capture_config.camera_serials
ordered_serials = [s for s in serials if (video_dir / f"{s}.mp4").exists()]
reorder_indices = camera_order_indices(ordered_serials, [c.get_name() for c in camera_group.cameras])

pose_model = pose_model_path_for_complexity(1)
hand_model = hand_model_path()

n_frames = 100
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 9
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

print(f"Testing: {n_frames} frames, {len(ordered_serials)} cameras, steps={steps}, device={device}")

pose_2d = []; pose_conf = []; lh_2d = []; lh_conf = []; rh_2d = []; rh_conf = []

t_all_start = time.perf_counter()
t_read = 0.0; t_mp = 0.0

with ExitStack() as stack:
    caps = {}
    for serial in ordered_serials:
        caps[serial] = cv2.VideoCapture(str(video_dir / f"{serial}.mp4"))

    trackers = {
        serial: stack.enter_context(
            MediapipeTracker(
                pose_model_path=pose_model, hand_model_path=hand_model,
                model_complexity=1, hands_enabled=True, delegate="cpu",
            )
        )
        for serial in ordered_serials
    }

    for fi in range(n_frames):
        t0 = time.perf_counter()
        frame_set = {}
        for serial in ordered_serials:
            ret, frame = caps[serial].read()
            if ret:
                frame_set[serial] = frame
        t_read += time.perf_counter() - t0

        t1 = time.perf_counter()
        timestamp_ms = fi * 33
        fps = []; pconf = []; lhp = []; rhp = []; lc = []; rc = []

        def _detect_one(serial: str):
            frame = frame_set[serial]
            trk = trackers[serial]
            pose = trk.detect_pose(frame, timestamp_ms=timestamp_ms)
            hand = trk.detect_hands(frame, timestamp_ms=timestamp_ms)
            return serial, pose, hand

        serial_to_result = {}
        with ThreadPoolExecutor(max_workers=len(ordered_serials)) as pool:
            futures = {pool.submit(_detect_one, s): s for s in ordered_serials if s in frame_set}
            for future in as_completed(futures):
                s, pose, hand = future.result()
                serial_to_result[s] = (pose, hand)

        for serial in ordered_serials:
            if serial in serial_to_result:
                pose, hand = serial_to_result[serial]
                fps.append(pose.xy); pconf.append(pose.confidence)
                lhp.append(hand.left_xy); rhp.append(hand.right_xy)
                lc.append(hand.left_confidence); rc.append(hand.right_confidence)
        t_mp += time.perf_counter() - t1

        pose_2d.append(np.stack(fps)); pose_conf.append(np.stack(pconf))
        lh_2d.append(np.stack(lhp)); lh_conf.append(np.stack(lc))
        rh_2d.append(np.stack(rhp)); rh_conf.append(np.stack(rc))

    for cap in caps.values():
        cap.release()

# Triangulate
t_tri_start = time.perf_counter()
pose_np = np.stack(pose_2d, axis=1).astype(np.float32)
lh_np = np.stack(lh_2d, axis=1).astype(np.float32)
rh_np = np.stack(rh_2d, axis=1).astype(np.float32)
pc = np.stack(pose_conf, axis=1).astype(np.float32)
lc = np.stack(lh_conf, axis=1).astype(np.float32)
rc = np.stack(rh_conf, axis=1).astype(np.float32)
pose_np = pose_np[reorder_indices]; lh_np = lh_np[reorder_indices]; rh_np = rh_np[reorder_indices]
pc = pc[reorder_indices]; lc = lc[reorder_indices]; rc = rc[reorder_indices]
all_2d = np.concatenate([pose_np, lh_np, rh_np], axis=2)
all_conf = np.concatenate([pc, lc, rc], axis=2)
all_2d = np.where(all_conf[..., None] <= 0.1, np.nan, all_2d)
pts_3d, _ = triangulate_sequence(camera_group, all_2d)
t_tri = time.perf_counter() - t_tri_start

# Track (Adam on SMPL-X)
retarget_config = RetargetConfig(
    mode=RetargetMode.TRACK, device=device,
    model_dir=Path("models/smplx"), profile=False,
)
retarget_config.track_pose_steps = steps

t_track_start = time.perf_counter()
tracker = RealtimeSmplxTracker(retarget_config, Path(f"/tmp/test_e2e_{device}"))
for i in range(pts_3d.shape[0]):
    tracker.update(i, pts_3d[i])
t_track = time.perf_counter() - t_track_start

t_total = time.perf_counter() - t_all_start

print(f"\n{'='*60}")
print(f"End-to-end ({n_frames} frames, {len(ordered_serials)} cams, steps={steps}, {device})")
print(f"{'='*60}")
print(f"  Video read:       {t_read*1000/n_frames:5.0f}ms")
print(f"  MediaPipe (4cam):  {t_mp*1000/n_frames:5.0f}ms  ({n_frames/t_mp:.1f} FPS)")
print(f"  Triangulate:       {t_tri*1000/n_frames:5.0f}ms  ({n_frames/t_tri:.0f} FPS)")
print(f"  Track (Adam):      {t_track*1000/n_frames:5.0f}ms  ({n_frames/t_track:.1f} FPS)")
print(f"  {'─'*40}")
print(f"  TOTAL:             {t_total*1000/n_frames:5.0f}ms  ({n_frames/t_total:.2f} FPS)")
