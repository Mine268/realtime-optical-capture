from __future__ import annotations

from pathlib import Path
import traceback
import time

import cv2
import yaml

from roc.config.models import MocapConfig
from roc.config.yaml_io import load_capture_config, save_capture_config, save_mocap_config
from roc.io.sessions import create_mocap_session
from roc.mocap.logging_utils import tee_to_log
from roc.mvs import MvsSystem
from roc.mocap.sync_capture import SyncCaptureWorker, transcode_raw_frames_to_videos


def run_mocap_capture(
    prepare_session: Path,
    calib_session: Path,
    session_root: Path,
    fps: float,
    max_frames: int,
    show_preview: bool,
) -> None:
    prepare_session = prepare_session.resolve()
    calib_session = calib_session.resolve()
    capture_config_path = prepare_session / "capture_config.yaml"
    calibration_yaml_path = calib_session / "calibration.yaml"
    if not capture_config_path.is_file():
        raise RuntimeError(f"Prepare capture_config.yaml not found: {capture_config_path}")
    if not calibration_yaml_path.is_file():
        raise RuntimeError(f"Calibration yaml not found: {calibration_yaml_path}")

    capture_config = load_capture_config(capture_config_path)
    session_paths = create_mocap_session(session_root)
    save_capture_config(session_paths.capture_config_path, capture_config)
    session_paths.calibration_yaml_path.write_text(calibration_yaml_path.read_text(encoding="utf-8"), encoding="utf-8")

    mocap_config = MocapConfig(
        schema_version=1,
        created_at=__import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        prepare_session=str(prepare_session),
        calib_session=str(calib_session),
        mode="capture",
        fps=fps,
        max_frames=max_frames,
        hands_enabled=False,
        model_complexity=0,
        video_format="mp4",
        lossless=False,
    )
    save_mocap_config(session_paths.mocap_config_path, mocap_config)

    with tee_to_log(session_paths.logs_dir / "mocap.log"):
        try:
            serial_to_cfg = {camera.serial: camera for camera in capture_config.cameras if camera.enabled}
            with MvsSystem() as mvs:
                devices = mvs.enumerate_devices()
                serial_to_device = {device.serial: device for device in devices}
                ordered_serials = [serial for serial in capture_config.camera_serials if serial in serial_to_cfg]
                missing = [serial for serial in ordered_serials if serial not in serial_to_device]
                if missing:
                    raise RuntimeError(f"Cameras from prepare session not currently available: {missing}")
                print(f"Using cameras: {ordered_serials}")

                cameras = []
                workers = []
                raw_frame_dirs = {}
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
                        raw_dir = session_paths.raw_frames_dir / serial
                        raw_frame_dirs[serial] = raw_dir
                        worker = SyncCaptureWorker(
                            serial=serial,
                            camera=camera,
                            raw_dir=raw_dir,
                            trigger_delay_s=min(0.1, 0.5 / max(fps, 1.0)),
                        )
                        worker.start()
                        workers.append(worker)

                    preview_window = "ROC Mocap Capture"
                    if show_preview:
                        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

                    frame_index = 0
                    while True:
                        frame_set_timestamp_ns = time.time_ns()
                        preview_frames = []
                        for worker in workers:
                            worker.frame_index = frame_index
                            worker.start_sem.release()
                        for worker in workers:
                            worker.done_sem.acquire()
                        for worker in workers:
                            if worker.error is not None:
                                raise worker.error
                            frame = worker.last_frame
                            if frame is None:
                                raise RuntimeError(f"No frame stored for camera {worker.serial}")
                            if show_preview:
                                preview_frames.append(cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))

                        frame_timestamps_ns.append(frame_set_timestamp_ns)
                        if frame_index % 25 == 0:
                            print(f"Captured frame set {frame_index}")

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
                    for worker in workers:
                        worker.stop()
                    for _, camera in reversed(cameras):
                        camera.close()

            actual_fps = transcode_raw_frames_to_videos(
                raw_frame_dirs=raw_frame_dirs,
                final_video_dir=session_paths.videos_dir,
                timestamps_ns=frame_timestamps_ns,
                fallback_fps=fps,
            )
            print(f"Finalized mocap capture videos with actual fps={actual_fps:.3f}")
            with session_paths.mocap_report_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "mode": "capture",
                        "frames": len(frame_timestamps_ns),
                        "target_fps": fps,
                        "actual_fps": actual_fps,
                        "camera_serials": ordered_serials,
                    },
                    handle,
                    sort_keys=False,
                    allow_unicode=False,
                )
            print(f"Saved mocap capture session to: {session_paths.session_dir}")
        except Exception:
            traceback.print_exc()
            raise
