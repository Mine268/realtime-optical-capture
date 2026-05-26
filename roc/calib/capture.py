from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from roc.config.models import CalibrationConfig, CharucoConfig
from roc.config.yaml_io import load_calibration_config
from roc.config.yaml_io import load_capture_config, save_calibration_config, save_capture_config
from roc.io.video import build_mjpg_writer, encode_h264_mp4
from roc.io.sessions import create_calibration_session
from roc.calib.solve import run_calibration_solve
from roc.mvs import MvsSystem


def _build_video_writer(path: Path, frame_width: int, frame_height: int, fps: float) -> cv2.VideoWriter:
    return build_mjpg_writer(path, frame_width, frame_height, fps)


def run_calibration_capture(
    mode: str,
    prepare_session: Path,
    calib_session: Path | None,
    session_root: Path,
    fps: float,
    frames: int,
    world_mode: str,
    square_length_mm: float,
    marker_length_mm: float,
    show_preview: bool,
) -> None:
    if mode == "solve-only":
        if calib_session is None:
            raise RuntimeError("--calib-session is required for solve-only mode")
        calib_session = calib_session.resolve()
        calib_config_path = calib_session / "calib_config.yaml"
        if calib_config_path.is_file():
            calib_config = load_calibration_config(calib_config_path)
            calib_config.world_mode = world_mode
            calib_config.charuco.square_length_mm = square_length_mm
            calib_config.charuco.marker_length_mm = marker_length_mm
            save_calibration_config(calib_config_path, calib_config)
        run_calibration_solve(calib_session)
        return

    if prepare_session is None:
        raise RuntimeError("--prepare-session is required for capture modes")

    prepare_session = prepare_session.resolve()
    capture_config_path = prepare_session / "capture_config.yaml"
    if not capture_config_path.is_file():
        raise RuntimeError(f"Prepare capture_config.yaml not found: {capture_config_path}")

    capture_config = load_capture_config(capture_config_path)
    session_paths = create_calibration_session(session_root)
    save_capture_config(session_paths.capture_config_path, capture_config)

    calib_config = CalibrationConfig(
        schema_version=1,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        prepare_session=str(prepare_session),
        frames=frames,
        fps=fps,
        mode=mode,
        world_mode=world_mode,
        video_format="mp4",
        lossless=False,
        charuco=CharucoConfig(
            squares_x=7,
            squares_y=5,
            dictionary="DICT_4X4_250",
            square_length_mm=square_length_mm,
            marker_length_mm=marker_length_mm,
        ),
    )
    save_calibration_config(session_paths.calib_config_path, calib_config)

    serial_to_config = {camera.serial: camera for camera in capture_config.cameras if camera.enabled}

    with MvsSystem() as mvs:
        devices = mvs.enumerate_devices()
        serial_to_device = {device.serial: device for device in devices}

        missing = [serial for serial in serial_to_config if serial not in serial_to_device]
        if missing:
            raise RuntimeError(f"Cameras from prepare session not currently available: {missing}")

        ordered_serials = [serial for serial in capture_config.camera_serials if serial in serial_to_config]
        cameras = []
        writers = {}
        final_video_paths = {}
        trigger_sleep = 0.0 if fps <= 0 else min(0.1, 0.5 / fps)

        try:
            for serial in ordered_serials:
                camera_cfg = serial_to_config[serial]
                device = serial_to_device[serial]
                camera = mvs.open_camera(device.index)
                camera.apply_manual_capture(
                    exposure_us=camera_cfg.exposure_us,
                    gain_db=camera_cfg.gain_db,
                    pixel_format=capture_config.pixel_format,
                )
                camera.start_grabbing()
                cameras.append((serial, camera, camera_cfg))
                print(f"Camera ready for calibration: {serial}")

            preview_window = "ROC Calibration Capture"
            if show_preview:
                cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

            for frame_index in range(frames):
                preview_frames = []
                for serial, camera, camera_cfg in cameras:
                    frame = camera.snapshot(fps_sleep=trigger_sleep)
                    if frame is None:
                        print(f"[warn] no frame for camera {serial} at frame {frame_index}")
                        continue

                    if serial not in writers:
                        final_video_path = session_paths.videos_dir / f"{serial}.mp4"
                        temp_video_path = final_video_path.with_name(final_video_path.stem + ".capture_tmp.avi")
                        writer = _build_video_writer(
                            temp_video_path,
                            frame.shape[1],
                            frame.shape[0],
                            fps,
                        )
                        writers[serial] = writer
                        final_video_paths[serial] = (temp_video_path, final_video_path)
                        camera_cfg.width = frame.shape[1]
                        camera_cfg.height = frame.shape[0]

                    writers[serial].write(frame)
                    if show_preview:
                        annotated = frame.copy()
                        cv2.putText(
                            annotated,
                            f"{serial} frame={frame_index + 1}/{frames}",
                            (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        preview_frames.append(annotated)

                print(f"Captured calibration frame set {frame_index + 1}/{frames}")

                if show_preview and preview_frames:
                    preview = cv2.hconcat(preview_frames)
                    cv2.imshow(preview_window, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        raise RuntimeError("Calibration capture aborted by user")
        finally:
            if show_preview:
                cv2.destroyAllWindows()
            for writer in writers.values():
                writer.release()
            for temp_video_path, final_video_path in final_video_paths.values():
                encode_h264_mp4(temp_video_path, final_video_path, fps)
                temp_video_path.unlink(missing_ok=True)
            for _, camera, _ in reversed(cameras):
                camera.close()

    print(f"Saved calibration capture session to: {session_paths.session_dir}")
    if mode == "capture+solve":
        run_calibration_solve(session_paths.session_dir)
