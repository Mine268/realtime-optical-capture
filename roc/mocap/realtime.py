from __future__ import annotations

from datetime import datetime
from pathlib import Path
from contextlib import ExitStack
import time
import traceback

import cv2
import numpy as np

from roc.config.models import MocapConfig
from roc.config.yaml_io import load_capture_config, save_capture_config, save_mocap_config
from roc.io.sessions import get_existing_mocap_session
from roc.mocap.common import build_temp_video_writer, finalize_videos_with_actual_fps, save_mocap_outputs
from roc.mocap.logging_utils import tee_to_log
from roc.mocap.postprocess import RealtimePostprocessor
from roc.mocap.render_2d_overlays import render_all_overlays
from roc.mocap.render_npz import render_npz_to_video
from roc.mocap.render_reprojection_overlays import render_reprojection_overlays, render_smplx_reprojection_overlays
from roc.mocap.retarget import RealtimeSmplxRetargeter, RetargetConfig, RetargetMode
from roc.mocap.track import RealtimeSmplxTracker
from roc.mvs import MvsSystem, OfflineMvsSystem
from roc.mvs import ParallelCapture
from roc.tracking.mediapipe_tracker import MediapipeTracker
from roc.tracking.model_paths import hand_model_path, pose_model_path_for_complexity
from roc.triangulation.cameras import camera_group_names, camera_order_indices, load_camera_group_from_toml
from roc.triangulation.triangulate import triangulate_sequence


def _format_estimate_profile_line(frame_index: int, timings: dict[str, float]) -> str:
    retarget_s = timings.get("retarget_s", 0.0)
    estimate_only_s = max(0.0, timings.get("frame_total_s", 0.0) - retarget_s)
    fields = (
        ("mocap_loop_total", timings.get("frame_total_s", 0.0)),
        ("estimate_only", estimate_only_s),
        ("mvs_capture", timings.get("capture_s", 0.0)),
        ("video_write", timings.get("video_write_s", 0.0)),
        ("mediapipe_pose", timings.get("pose_s", 0.0)),
        ("mediapipe_hands", timings.get("hands_s", 0.0)),
        ("buffer_append", timings.get("append_s", 0.0)),
        ("triangulate_3d", timings.get("triangulate_s", 0.0)),
        ("postprocess_3d", timings.get("postprocess_s", 0.0)),
        ("smplx_retarget", retarget_s),
    )
    parts = [f"{name}={seconds * 1000.0:.1f}ms" for name, seconds in fields]
    return f"[mocap-profile] frame={frame_index} stage=estimate " + " ".join(parts)


def run_mocap_realtime(
    prepare_session: Path,
    calib_session: Path,
    mocap_session: Path,
    fps: float,
    max_frames: int,
    hands_enabled: bool,
    model_complexity: int,
    show_preview: bool,
    delegate: str = "cpu",
    offline_source_dir: Path | None = None,
    retarget_config: RetargetConfig | None = None,
    record_videos: bool = True,
    profile: bool = False,
) -> None:
    prepare_session = prepare_session.resolve()
    calib_session = calib_session.resolve()

    capture_config_path = prepare_session / "capture_config.yaml"
    calibration_yaml_path = calib_session / "calibration.yaml"
    calibration_toml_path = calib_session / "calibration.toml"
    if not capture_config_path.is_file():
        raise RuntimeError(f"Prepare capture_config.yaml not found: {capture_config_path}")
    if not calibration_yaml_path.is_file():
        raise RuntimeError(f"Calibration yaml not found: {calibration_yaml_path}")
    if not calibration_toml_path.is_file():
        raise RuntimeError(f"Calibration toml not found: {calibration_toml_path}")

    if offline_source_dir is None:
        mocap_session.mkdir(parents=True, exist_ok=True)

    capture_config = load_capture_config(capture_config_path)
    session_paths = get_existing_mocap_session(mocap_session)
    save_capture_config(session_paths.capture_config_path, capture_config)
    session_paths.calibration_yaml_path.write_text(calibration_yaml_path.read_text(encoding="utf-8"), encoding="utf-8")

    mocap_config = MocapConfig(
        schema_version=1,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        prepare_session=str(prepare_session),
        calib_session=str(calib_session),
        mode="realtime",
        fps=fps,
        max_frames=max_frames,
        hands_enabled=hands_enabled,
        model_complexity=model_complexity,
        video_format="mp4",
        lossless=False,
    )
    save_mocap_config(session_paths.mocap_config_path, mocap_config)
    with tee_to_log(session_paths.logs_dir / "mocap.log"):
        try:
            print(f"Prepare session: {prepare_session}")
            print(f"Calibration session: {calib_session}")
            print(f"MediaPipe delegate: {delegate}")
            if offline_source_dir is not None:
                print(f"Offline source dir: {offline_source_dir.resolve()}")
            camera_group = load_camera_group_from_toml(calibration_toml_path)
            calibrated_serials = camera_group_names(camera_group)

            pose_model_path = pose_model_path_for_complexity(model_complexity)
            hand_model_path_value = hand_model_path() if hands_enabled else None
            if not pose_model_path.is_file():
                raise RuntimeError(f"Pose model not found: {pose_model_path}")
            if hands_enabled and (hand_model_path_value is None or not hand_model_path_value.is_file()):
                raise RuntimeError(f"Hand model not found: {hand_model_path_value}")

            serial_to_cfg = {camera.serial: camera for camera in capture_config.cameras if camera.enabled}
            timestamps = []
            pose_2d = []
            pose_conf = []
            left_hand_2d = []
            left_hand_conf = []
            right_hand_2d = []
            right_hand_conf = []
            bboxes = []
            record_session_videos = record_videos and (
                offline_source_dir is None
                or offline_source_dir.resolve() != session_paths.videos_dir.resolve()
            )

            system = OfflineMvsSystem(offline_source_dir, serials=capture_config.camera_serials) if offline_source_dir else MvsSystem()
            with system as mvs, ExitStack() as stack:
                devices = mvs.enumerate_devices()
                serial_to_device = {device.serial: device for device in devices}
                ordered_serials = [serial for serial in capture_config.camera_serials if serial in serial_to_cfg]
                missing = [serial for serial in ordered_serials if serial not in serial_to_device]
                if missing:
                    raise RuntimeError(f"Cameras from prepare session not currently available: {missing}")
                print(f"Using cameras: {ordered_serials}")
                reorder_indices = camera_order_indices(ordered_serials, calibrated_serials)
                realtime_postprocessor = (
                    RealtimePostprocessor(num_landmarks=75, fps=fps, cutoff_hz=1.2, max_hold_frames=3)
                    if retarget_config is not None
                    else None
                )
                realtime_retargeter = None
                if retarget_config is not None:
                    if retarget_config.mode == RetargetMode.TRACK:
                        realtime_retargeter = RealtimeSmplxTracker(
                            retarget_config, session_paths.session_dir / "smplx_retarget"
                        )
                    else:
                        realtime_retargeter = RealtimeSmplxRetargeter(
                            retarget_config, session_paths.session_dir / "smplx_retarget"
                        )
                trackers = {
                    serial: stack.enter_context(
                        MediapipeTracker(
                            pose_model_path=pose_model_path,
                            hand_model_path=hand_model_path_value,
                            model_complexity=model_complexity,
                            hands_enabled=hands_enabled,
                            delegate=delegate,
                        )
                    )
                    for serial in ordered_serials
                }

                cameras = []
                writers = {}
                temp_video_paths = {}
                frame_timestamps_ns = []
                try:
                    for serial in ordered_serials:
                        cfg = serial_to_cfg[serial]
                        device = serial_to_device[serial]
                        camera = mvs.open_camera(device.index)
                        camera.apply_manual_capture(
                            exposure_us=cfg.exposure_us,
                            gain_db=cfg.gain_db,
                            pixel_format=capture_config.pixel_format,
                        )
                        camera.start_grabbing()
                        cameras.append((serial, camera))

                    preview_window = "ROC Mocap"
                    if show_preview:
                        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

                    frame_index = 0
                    _last_postprocess_time = 0.0
                    with ParallelCapture(cameras) as parallel_cap:
                        while True:
                            frame_start = time.perf_counter()
                            profile_times = {
                                "capture_s": 0.0,
                                "video_write_s": 0.0,
                                "pose_s": 0.0,
                                "hands_s": 0.0,
                                "append_s": 0.0,
                                "triangulate_s": 0.0,
                                "postprocess_s": 0.0,
                                "retarget_s": 0.0,
                            }
                            frame_set_timestamp_ns = time.time_ns()
                            frame_set_pose = []
                            frame_set_pose_conf = []
                            frame_set_left = []
                            frame_set_left_conf = []
                            frame_set_right = []
                            frame_set_right_conf = []
                            frame_set_bbox = []
                            preview_frames = []
                            timestamp_ms = frame_index * int(1000 / max(fps, 1.0))

                            stage_start = time.perf_counter()
                            serial_to_frame = parallel_cap.snapshot_all()
                            profile_times["capture_s"] += time.perf_counter() - stage_start

                            for serial, camera in cameras:
                                frame = serial_to_frame.get(serial)
                                if frame is None:
                                    raise RuntimeError(f"No frame received for camera {serial}")

                                if record_session_videos and serial not in writers:
                                    temp_path = session_paths.videos_dir / f"{serial}.capture_tmp.avi"
                                    temp_video_paths[serial] = temp_path
                                    writers[serial] = build_temp_video_writer(
                                        temp_path,
                                        frame.shape[1],
                                        frame.shape[0],
                                        fps,
                                    )
                                if record_session_videos:
                                    stage_start = time.perf_counter()
                                    writers[serial].write(frame)
                                    profile_times["video_write_s"] += time.perf_counter() - stage_start

                                tracker = trackers[serial]
                                stage_start = time.perf_counter()
                                pose_result = tracker.detect_pose(frame, timestamp_ms=timestamp_ms)
                                profile_times["pose_s"] += time.perf_counter() - stage_start
                                stage_start = time.perf_counter()
                                hand_result = tracker.detect_hands(frame, timestamp_ms=timestamp_ms)
                                profile_times["hands_s"] += time.perf_counter() - stage_start
                                valid = ~np.isnan(pose_result.xy[:, 0])
                                if np.any(valid):
                                    xy = pose_result.xy[valid]
                                    x0, y0 = np.min(xy, axis=0)
                                    x1, y1 = np.max(xy, axis=0)
                                    bbox = np.array([x0, y0, x1, y1], dtype=np.float32)
                                else:
                                    bbox = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)

                                frame_set_pose.append(pose_result.xy)
                                frame_set_pose_conf.append(pose_result.confidence)
                                frame_set_left.append(hand_result.left_xy)
                                frame_set_left_conf.append(hand_result.left_confidence)
                                frame_set_right.append(hand_result.right_xy)
                                frame_set_right_conf.append(hand_result.right_confidence)
                                frame_set_bbox.append(bbox)

                                if show_preview:
                                    overlay = frame.copy()
                                    for point in pose_result.xy:
                                        if not np.isnan(point[0]):
                                            cv2.circle(overlay, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
                                    preview_frames.append(overlay)

                            stage_start = time.perf_counter()
                            timestamps.append(timestamp_ms)
                            pose_2d.append(np.stack(frame_set_pose, axis=0))
                            pose_conf.append(np.stack(frame_set_pose_conf, axis=0))
                            left_hand_2d.append(np.stack(frame_set_left, axis=0))
                            left_hand_conf.append(np.stack(frame_set_left_conf, axis=0))
                            right_hand_2d.append(np.stack(frame_set_right, axis=0))
                            right_hand_conf.append(np.stack(frame_set_right_conf, axis=0))
                            bboxes.append(np.stack(frame_set_bbox, axis=0))
                            frame_timestamps_ns.append(frame_set_timestamp_ns)
                            profile_times["append_s"] += time.perf_counter() - stage_start

                            if realtime_postprocessor is not None and realtime_retargeter is not None:
                                stage_start = time.perf_counter()
                                frame_pose_np = np.stack(frame_set_pose, axis=0).astype(np.float32)[reorder_indices]
                                frame_pose_conf_np = np.stack(frame_set_pose_conf, axis=0).astype(np.float32)[reorder_indices]
                                frame_left_np = np.stack(frame_set_left, axis=0).astype(np.float32)[reorder_indices]
                                frame_left_conf_np = np.stack(frame_set_left_conf, axis=0).astype(np.float32)[reorder_indices]
                                frame_right_np = np.stack(frame_set_right, axis=0).astype(np.float32)[reorder_indices]
                                frame_right_conf_np = np.stack(frame_set_right_conf, axis=0).astype(np.float32)[reorder_indices]
                                frame_landmarks_2d = np.concatenate([frame_pose_np, frame_left_np, frame_right_np], axis=1)
                                frame_conf = np.concatenate(
                                    [frame_pose_conf_np, frame_left_conf_np, frame_right_conf_np],
                                    axis=1,
                                )
                                frame_landmarks_2d = np.where(frame_conf[..., None] <= 0.1, np.nan, frame_landmarks_2d)
                                frame_points_3d, _ = triangulate_sequence(camera_group, frame_landmarks_2d[:, None, :, :])
                                profile_times["triangulate_s"] += time.perf_counter() - stage_start
                                stage_start = time.perf_counter()
                                dt_s = frame_start - _last_postprocess_time if _last_postprocess_time > 0 else None
                                processed_points_3d = realtime_postprocessor.update(frame_points_3d[0], dt_s=dt_s)
                                _last_postprocess_time = frame_start
                                profile_times["postprocess_s"] += time.perf_counter() - stage_start
                                stage_start = time.perf_counter()
                                realtime_retargeter.update(frame_index, frame_points_3d[0])
                                profile_times["retarget_s"] += time.perf_counter() - stage_start

                            if profile:
                                profile_times["frame_total_s"] = time.perf_counter() - frame_start
                                print(_format_estimate_profile_line(frame_index, profile_times), flush=True)

                            if frame_index % 25 == 0:
                                print(f"Processed frame set {frame_index}")

                            if show_preview and preview_frames:
                                preview = cv2.hconcat(preview_frames)
                                cv2.imshow(preview_window, preview)
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord("q"):
                                    break

                            frame_index += 1
                            if max_frames > 0 and frame_index >= max_frames:
                                break
                finally:
                    if show_preview:
                        cv2.destroyAllWindows()
                    for writer in writers.values():
                        writer.release()
                    for _, camera in reversed(cameras):
                        camera.close()

            if record_session_videos:
                actual_fps = finalize_videos_with_actual_fps(
                    temp_video_paths=temp_video_paths,
                    final_video_dir=session_paths.videos_dir,
                    timestamps_ns=frame_timestamps_ns,
                    fallback_fps=fps,
                )
                print(f"Finalized realtime mocap videos with actual fps={actual_fps:.3f}")
            else:
                actual_fps = fps
                print("Skipped realtime mocap video recording")

            pose_2d_np = np.stack(pose_2d, axis=1).astype(np.float32)
            pose_conf_np = np.stack(pose_conf, axis=1).astype(np.float32)
            left_hand_2d_np = np.stack(left_hand_2d, axis=1).astype(np.float32)
            left_hand_conf_np = np.stack(left_hand_conf, axis=1).astype(np.float32)
            right_hand_2d_np = np.stack(right_hand_2d, axis=1).astype(np.float32)
            right_hand_conf_np = np.stack(right_hand_conf, axis=1).astype(np.float32)
            bboxes_np = np.stack(bboxes, axis=1).astype(np.float32)

            pose_2d_np = pose_2d_np[reorder_indices]
            pose_conf_np = pose_conf_np[reorder_indices]
            left_hand_2d_np = left_hand_2d_np[reorder_indices]
            left_hand_conf_np = left_hand_conf_np[reorder_indices]
            right_hand_2d_np = right_hand_2d_np[reorder_indices]
            right_hand_conf_np = right_hand_conf_np[reorder_indices]
            bboxes_np = bboxes_np[reorder_indices]

            all_landmarks_2d = np.concatenate([pose_2d_np, left_hand_2d_np, right_hand_2d_np], axis=2)
            all_conf = np.concatenate([pose_conf_np, left_hand_conf_np, right_hand_conf_np], axis=2)
            all_landmarks_2d = np.where(all_conf[..., None] <= 0.1, np.nan, all_landmarks_2d)
            points_3d, reprojection = triangulate_sequence(camera_group, all_landmarks_2d)

            save_mocap_outputs(
                session_paths=session_paths,
                capture_config=capture_config,
                camera_serials=calibrated_serials,
                timestamps=timestamps,
                pose_2d_np=pose_2d_np,
                pose_conf_np=pose_conf_np,
                left_hand_2d_np=left_hand_2d_np,
                left_hand_conf_np=left_hand_conf_np,
                right_hand_2d_np=right_hand_2d_np,
                right_hand_conf_np=right_hand_conf_np,
                bboxes_np=bboxes_np,
                points_3d=points_3d,
                reprojection=reprojection,
                fps=actual_fps,
                hands_enabled=hands_enabled,
                model_complexity=model_complexity,
                postprocess_mode="realtime",
            )
            retarget_npz = None
            if retarget_config is not None and realtime_retargeter is not None:
                retarget_npz = realtime_retargeter.save(source_npz=session_paths.mocap_npz_path)
                print(f"Saved SMPL-X retarget sequence to: {retarget_npz}")
            if record_videos:
                render_all_overlays(
                    npz_path=session_paths.mocap_npz_path,
                    video_dir=session_paths.videos_dir,
                    output_dir=session_paths.overlay_videos_dir,
                    confidence_threshold=0.1,
                    frame_limit=0,
                )
                render_reprojection_overlays(
                    npz_path=session_paths.mocap_npz_path,
                    calibration_toml=calibration_toml_path,
                    video_dir=session_paths.videos_dir,
                    output_dir=session_paths.session_dir / "reprojection_videos",
                    confidence_threshold=0.1,
                    frame_limit=0,
                )
                if retarget_npz is not None:
                    render_smplx_reprojection_overlays(
                        mocap_npz_path=session_paths.mocap_npz_path,
                        smplx_npz_path=retarget_npz,
                        calibration_toml=calibration_toml_path,
                        video_dir=session_paths.videos_dir,
                        output_dir=session_paths.session_dir / "reprojection_videos",
                        confidence_threshold=0.1,
                        frame_limit=0,
                    )
                render_npz_to_video(
                    npz_path=session_paths.mocap_npz_path,
                    output_path=session_paths.session_dir / "pose_videos" / "mocap_3d_pose.mp4",
                    fps=actual_fps,
                    frame_limit=0,
                )
            print(f"Saved mocap session to: {session_paths.session_dir}")
        except Exception:
            traceback.print_exc()
            raise
