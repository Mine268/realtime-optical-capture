from __future__ import annotations

from pathlib import Path
from contextlib import ExitStack
import traceback
from datetime import datetime
import time

import cv2
import numpy as np

from roc.config.models import MocapConfig
from roc.config.yaml_io import load_capture_config, save_capture_config, save_mocap_config
from roc.io.sessions import get_existing_mocap_session
from roc.mocap.common import save_mocap_outputs
from roc.mocap.logging_utils import tee_to_log
from roc.mocap.render_2d_overlays import render_all_overlays
from roc.mocap.render_npz import render_npz_to_video
from roc.mocap.render_reprojection_overlays import render_reprojection_overlays, render_smplx_reprojection_overlays
from roc.mocap.retarget import RetargetConfig, RetargetMode, run_mocap_retarget
from roc.tracking.mediapipe_tracker import MediapipeTracker
from roc.tracking.model_paths import hand_model_path, pose_model_path_for_complexity
from roc.triangulation.cameras import camera_group_names, camera_order_indices, load_camera_group_from_toml
from roc.triangulation.triangulate import triangulate_sequence


def _format_estimate_profile_line(frame_index: int, timings: dict[str, float]) -> str:
    fields = (
        ("estimate_only", timings.get("frame_total_s", 0.0)),
        ("video_read", timings.get("read_s", 0.0)),
        ("mediapipe_pose", timings.get("pose_s", 0.0)),
        ("mediapipe_hands", timings.get("hands_s", 0.0)),
        ("buffer_append", timings.get("append_s", 0.0)),
    )
    parts = [f"{name}={seconds * 1000.0:.1f}ms" for name, seconds in fields]
    return f"[mocap-profile] frame={frame_index} stage=estimate " + " ".join(parts)


def run_mocap_offline(
    prepare_session: Path,
    calib_session: Path,
    video_dir: Path | None,
    mocap_session: Path,
    max_frames: int,
    hands_enabled: bool,
    model_complexity: int,
    show_preview: bool,
    postprocess_mode: str = "offline",
    delegate: str = "cpu",
    retarget_config: RetargetConfig | None = None,
    profile: bool = False,
) -> None:
    prepare_session = prepare_session.resolve()
    calib_session = calib_session.resolve()
    mocap_session = mocap_session.resolve()
    calibration_toml_path = calib_session / "calibration.toml"
    if not calibration_toml_path.is_file():
        raise RuntimeError(f"Calibration toml not found: {calibration_toml_path}")

    source_video_dir = (video_dir or mocap_session / "videos").resolve()
    if not source_video_dir.is_dir():
        raise RuntimeError(f"Video directory not found: {source_video_dir}")

    session_paths = get_existing_mocap_session(mocap_session)
    capture_config_path = prepare_session / "capture_config.yaml"
    calibration_yaml_path = calib_session / "calibration.yaml"
    if not capture_config_path.is_file():
        raise RuntimeError(f"Prepare capture_config.yaml not found: {capture_config_path}")
    if not calibration_yaml_path.is_file():
        raise RuntimeError(f"Calibration yaml not found: {calibration_yaml_path}")
    capture_config = load_capture_config(capture_config_path)
    save_capture_config(session_paths.capture_config_path, capture_config)
    session_paths.calibration_yaml_path.write_text(
        calibration_yaml_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    estimate_config = MocapConfig(
        schema_version=1,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        prepare_session=str(prepare_session),
        calib_session=str(calib_session),
        mode="capture_estimate",
        fps=_read_video_fps(source_video_dir),
        max_frames=max_frames,
        hands_enabled=hands_enabled,
        model_complexity=model_complexity,
        video_format="mp4",
        lossless=False,
    )
    save_mocap_config(session_paths.session_dir / "mocap_estimate_config.yaml", estimate_config)
    with tee_to_log(session_paths.logs_dir / "mocap.log"):
        try:
            print(f"Prepare session: {prepare_session}")
            print(f"Calibration session: {calib_session}")
            print(f"Output session: {session_paths.session_dir}")
            print(f"Postprocess mode: {postprocess_mode}")
            print(f"MediaPipe delegate: {delegate}")
            camera_group = load_camera_group_from_toml(calibration_toml_path)
            calibrated_serials = camera_group_names(camera_group)

            pose_model_path = pose_model_path_for_complexity(model_complexity)
            hand_model_path_value = hand_model_path() if hands_enabled else None
            if not pose_model_path.is_file():
                raise RuntimeError(f"Pose model not found: {pose_model_path}")
            if hands_enabled and (hand_model_path_value is None or not hand_model_path_value.is_file()):
                raise RuntimeError(f"Hand model not found: {hand_model_path_value}")

            serial_to_video = {path.stem: path for path in source_video_dir.glob("*.mp4")}
            ordered_serials = [serial for serial in capture_config.camera_serials if serial in serial_to_video]
            if not ordered_serials:
                raise RuntimeError(f"No matching mp4 files found in {source_video_dir}")
            print(f"Using videos: {ordered_serials}")
            source_fps = _read_video_fps(source_video_dir)

            caps = []
            timestamps = []
            pose_2d = []
            pose_conf = []
            left_hand_2d = []
            left_hand_conf = []
            right_hand_2d = []
            right_hand_conf = []
            bboxes = []

            try:
                for serial in ordered_serials:
                    cap = cv2.VideoCapture(str(serial_to_video[serial]))
                    if not cap.isOpened():
                        raise RuntimeError(f"Failed to open video for camera {serial}: {serial_to_video[serial]}")
                    caps.append((serial, cap))

                preview_window = "ROC Mocap Offline"
                if show_preview:
                    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

                with ExitStack() as stack:
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
                    frame_index = 0
                    while True:
                        frame_start = time.perf_counter()
                        profile_times = {
                            "read_s": 0.0,
                            "pose_s": 0.0,
                            "hands_s": 0.0,
                            "append_s": 0.0,
                        }
                        frame_set_pose = []
                        frame_set_pose_conf = []
                        frame_set_left = []
                        frame_set_left_conf = []
                        frame_set_right = []
                        frame_set_right_conf = []
                        frame_set_bbox = []
                        preview_frames = []
                        timestamp_ms = round(frame_index * 1000.0 / max(source_fps, 1.0))

                        for serial, cap in caps:
                            stage_start = time.perf_counter()
                            ret, frame = cap.read()
                            profile_times["read_s"] += time.perf_counter() - stage_start
                            if not ret:
                                frame = None
                            if frame is None:
                                return _finalize(
                                    session_paths,
                                    capture_config,
                                    timestamps,
                                    pose_2d,
                                    pose_conf,
                                    left_hand_2d,
                                    left_hand_conf,
                                    right_hand_2d,
                                    right_hand_conf,
                                    bboxes,
                                    camera_group,
                                    calibrated_serials,
                                    hands_enabled,
                                    model_complexity,
                                    source_video_dir,
                                    source_fps,
                                    postprocess_mode,
                                    calibration_toml_path,
                                    retarget_config,
                                    profile,
                                )

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
                        profile_times["append_s"] += time.perf_counter() - stage_start

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
                for _, cap in caps:
                    cap.release()

            _finalize(
                session_paths,
                capture_config,
                timestamps,
                pose_2d,
                pose_conf,
                left_hand_2d,
                left_hand_conf,
                right_hand_2d,
                right_hand_conf,
                bboxes,
                camera_group,
                calibrated_serials,
                hands_enabled,
                model_complexity,
                source_video_dir,
                source_fps,
                postprocess_mode,
                calibration_toml_path,
                retarget_config,
                profile,
            )
        except Exception:
            traceback.print_exc()
            raise


def _finalize(
    session_paths,
    capture_config,
    timestamps,
    pose_2d,
    pose_conf,
    left_hand_2d,
    left_hand_conf,
    right_hand_2d,
    right_hand_conf,
    bboxes,
    camera_group,
    calibrated_serials,
    hands_enabled,
    model_complexity,
    source_video_dir,
    source_fps,
    postprocess_mode,
    calibration_toml_path,
    retarget_config: RetargetConfig | None,
    profile: bool,
) -> None:
    if not timestamps:
        raise RuntimeError("No mocap frames were processed")

    pose_2d_np = np.stack(pose_2d, axis=1).astype(np.float32)
    pose_conf_np = np.stack(pose_conf, axis=1).astype(np.float32)
    left_hand_2d_np = np.stack(left_hand_2d, axis=1).astype(np.float32)
    left_hand_conf_np = np.stack(left_hand_conf, axis=1).astype(np.float32)
    right_hand_2d_np = np.stack(right_hand_2d, axis=1).astype(np.float32)
    right_hand_conf_np = np.stack(right_hand_conf, axis=1).astype(np.float32)
    bboxes_np = np.stack(bboxes, axis=1).astype(np.float32)

    capture_serials = list(capture_config.camera_serials)
    reorder_indices = camera_order_indices(capture_serials, calibrated_serials)
    pose_2d_np = pose_2d_np[reorder_indices]
    pose_conf_np = pose_conf_np[reorder_indices]
    left_hand_2d_np = left_hand_2d_np[reorder_indices]
    left_hand_conf_np = left_hand_conf_np[reorder_indices]
    right_hand_2d_np = right_hand_2d_np[reorder_indices]
    right_hand_conf_np = right_hand_conf_np[reorder_indices]
    bboxes_np = bboxes_np[reorder_indices]

    finalize_start = time.perf_counter()
    all_landmarks_2d = np.concatenate([pose_2d_np, left_hand_2d_np, right_hand_2d_np], axis=2)
    all_conf = np.concatenate([pose_conf_np, left_hand_conf_np, right_hand_conf_np], axis=2)
    all_landmarks_2d = np.where(all_conf[..., None] <= 0.1, np.nan, all_landmarks_2d)
    triangulate_start = time.perf_counter()
    points_3d, reprojection = triangulate_sequence(camera_group, all_landmarks_2d)
    triangulate_elapsed = time.perf_counter() - triangulate_start
    save_start = time.perf_counter()

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
        fps=source_fps,
        hands_enabled=hands_enabled,
        model_complexity=model_complexity,
        postprocess_mode=postprocess_mode,
    )
    save_elapsed = time.perf_counter() - save_start
    if profile:
        print(
            "[mocap-profile] stage=finalize "
            f"triangulate_3d_all_frames={triangulate_elapsed * 1000.0:.1f}ms "
            f"save_mocap_outputs={save_elapsed * 1000.0:.1f}ms "
            f"total={(time.perf_counter() - finalize_start) * 1000.0:.1f}ms",
            flush=True,
        )
    retarget_npz = None
    if retarget_config is not None:
        if retarget_config.mode == RetargetMode.TRACK:
            _apply_track_config_overrides(retarget_config)
            print("Retargeting 3D keypoints to SMPL-X joint rotations (track mode)...")
        else:
            print("Retargeting 3D keypoints to SMPL-X joint rotations...")
        retarget_npz = run_mocap_retarget(
            npz_path=session_paths.mocap_npz_path,
            mocap_session=session_paths.session_dir,
            config=retarget_config,
        )
        print(f"Saved SMPL-X retarget sequence to: {retarget_npz}")
    render_all_overlays(
        npz_path=session_paths.mocap_npz_path,
        video_dir=source_video_dir,
        output_dir=session_paths.overlay_videos_dir,
        confidence_threshold=0.1,
        frame_limit=0,
    )
    render_reprojection_overlays(
        npz_path=session_paths.mocap_npz_path,
        calibration_toml=calibration_toml_path,
        video_dir=source_video_dir,
        output_dir=session_paths.session_dir / "reprojection_videos",
        confidence_threshold=0.1,
        frame_limit=0,
    )
    if retarget_npz is not None:
        render_smplx_reprojection_overlays(
            mocap_npz_path=session_paths.mocap_npz_path,
            smplx_npz_path=retarget_npz,
            calibration_toml=calibration_toml_path,
            video_dir=source_video_dir,
            output_dir=session_paths.session_dir / "reprojection_videos",
            confidence_threshold=0.1,
            frame_limit=0,
        )
    render_npz_to_video(
        npz_path=session_paths.mocap_npz_path,
        output_path=session_paths.session_dir / "pose_videos" / "mocap_3d_pose.mp4",
        fps=source_fps,
        frame_limit=0,
    )
    print(f"Saved offline mocap session to: {session_paths.session_dir}")


def _read_video_fps(video_dir: Path) -> float:
    for path in sorted(video_dir.glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        try:
            if cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if fps > 0:
                    return fps
        finally:
            cap.release()
    return 5.0


def _apply_track_config_overrides(config: RetargetConfig) -> None:
    config.pose_steps = config.track_pose_steps
    config.temporal_weight = config.track_temporal_weight
    config.velocity_weight = config.track_velocity_weight
    config.acceleration_weight = config.track_acceleration_weight
    config.lower_body_refine = False
    config.optimize_hands = False
    config.use_vposer = False
