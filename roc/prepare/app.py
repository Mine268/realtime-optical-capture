from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from roc.config.defaults import DEFAULT_EXPOSURE_US
from roc.config.defaults import DEFAULT_GAIN_DB
from roc.config.models import CameraCaptureConfig
from roc.config.models import CaptureConfig
from roc.config.yaml_io import save_capture_config
from roc.io.sessions import PrepareSessionPaths
from roc.io.sessions import create_prepare_session
from roc.mvs import MvsCamera
from roc.mvs import MvsSystem
from roc.mvs import ParallelCapture


@dataclass(slots=True)
class PrepareState:
    selected_index: int = 0
    exposure_step: float = 500.0
    gain_step: float = 1.0
    should_save: bool = False
    should_quit: bool = False


def _build_info_panel(camera_configs: list[CameraCaptureConfig], selected_index: int) -> list[str]:
    lines = [
        "Keys: 1-9 select | [ ] exposure | - = gain | s save | q quit",
    ]
    for index, camera in enumerate(camera_configs):
        prefix = ">" if index == selected_index else " "
        lines.append(
            f"{prefix} cam {index + 1} sn={camera.serial} exp={camera.exposure_us:.0f}us gain={camera.gain_db:.1f}dB"
        )
    return lines


def _draw_info(frame, lines: Iterable[str]):
    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22


def _compose_preview(frames: list, labels: list[str], preview_scale: float):
    annotated = []
    for frame, label in zip(frames, labels):
        if frame is None:
            continue
        view = frame.copy()
        cv2.putText(
            view,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if preview_scale != 1.0:
            view = cv2.resize(view, dsize=None, fx=preview_scale, fy=preview_scale)
        annotated.append(view)
    if not annotated:
        raise RuntimeError("No preview frames available")
    return cv2.hconcat(annotated)


def _make_status_frame(message: str, width: int = 1280, height: int = 720):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (40, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _apply_key(key: int, state: PrepareState, camera_configs: list[CameraCaptureConfig]) -> None:
    if ord("1") <= key <= ord("9"):
        next_index = key - ord("1")
        if next_index < len(camera_configs):
            state.selected_index = next_index
        return

    current = camera_configs[state.selected_index]

    if key == ord("["):
        current.exposure_us = max(50.0, current.exposure_us - state.exposure_step)
    elif key == ord("]"):
        current.exposure_us += state.exposure_step
    elif key == ord("-"):
        current.gain_db = max(0.0, current.gain_db - state.gain_step)
    elif key == ord("="):
        current.gain_db += state.gain_step
    elif key == ord("s"):
        state.should_save = True
        state.should_quit = True
    elif key == ord("q"):
        state.should_quit = True


def _save_snapshots(paths: PrepareSessionPaths, frames_by_serial: dict[str, object]) -> None:
    for serial, frame in frames_by_serial.items():
        if frame is None:
            continue
        cv2.imwrite(str(paths.snapshots_dir / f"{serial}.jpg"), frame)


def run_prepare(
    session_root: Path,
    fps: float,
    serials: list[str],
    pixel_format: str,
    preview_scale: float,
    window_name: str,
) -> None:
    with MvsSystem() as mvs:
        devices = mvs.enumerate_devices()
        if not devices:
            raise RuntimeError("No MVS cameras detected")
        print(f"Detected {len(devices)} camera(s)")

        if serials:
            serial_set = set(serials)
            devices = [device for device in devices if device.serial in serial_set]
            missing = serial_set.difference(device.serial for device in devices)
            if missing:
                raise RuntimeError(f"Requested serials not found: {sorted(missing)}")
        print("Using cameras:", [device.serial for device in devices])

        camera_configs = [
            CameraCaptureConfig(
                serial=device.serial,
                index_hint=device.index,
                exposure_us=DEFAULT_EXPOSURE_US,
                gain_db=DEFAULT_GAIN_DB,
                model_name=device.model_name,
                transport_type=device.transport_name,
            )
            for device in devices
        ]

        cameras: list[MvsCamera] = []
        try:
            for camera_config in camera_configs:
                print(f"Opening camera {camera_config.serial}")
                camera = mvs.open_camera(camera_config.index_hint)
                camera.apply_manual_capture(
                    exposure_us=camera_config.exposure_us,
                    gain_db=camera_config.gain_db,
                    pixel_format=pixel_format,
                )
                camera.start_grabbing()
                cameras.append(camera)
                print(f"Camera ready: {camera_config.serial}")

            state = PrepareState()
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            latest_frames: dict[str, object] = {}
            empty_rounds = 0

            with ParallelCapture([(cfg.serial, cam) for cfg, cam in zip(camera_configs, cameras)]) as parallel_cap:
                while True:
                    selected_cfg = camera_configs[state.selected_index]
                    cameras[state.selected_index].apply_manual_capture(
                        exposure_us=selected_cfg.exposure_us,
                        gain_db=selected_cfg.gain_db,
                        pixel_format=pixel_format,
                    )

                    serial_to_frame = parallel_cap.snapshot_all()

                    frames = []
                    labels = []
                    for index, (camera, camera_config) in enumerate(zip(cameras, camera_configs)):
                        frame = serial_to_frame.get(camera_config.serial)
                        if frame is None:
                            continue
                        camera_config.width = frame.shape[1]
                        camera_config.height = frame.shape[0]
                        latest_frames[camera_config.serial] = frame
                        frames.append(frame)
                        labels.append(f"{index + 1}:{camera_config.serial}")

                    if frames:
                        empty_rounds = 0
                        preview = _compose_preview(frames, labels, preview_scale)
                    else:
                        empty_rounds += 1
                        preview = _make_status_frame(
                            f"Waiting for camera frames... round={empty_rounds}  press q to quit"
                        )
                    _draw_info(preview, _build_info_panel(camera_configs, state.selected_index))
                    cv2.imshow(window_name, preview)

                    key = cv2.waitKey(max(1, int(1000 / max(fps, 1.0)))) & 0xFF
                    if key != 255:
                        _apply_key(key, state, camera_configs)
                    if state.should_quit:
                        break

            cv2.destroyWindow(window_name)

            if state.should_save:
                session_paths = create_prepare_session(session_root)
                _save_snapshots(session_paths, latest_frames)
                config = CaptureConfig(
                    schema_version=1,
                    created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                    camera_count=len(camera_configs),
                    camera_serials=[camera.serial for camera in camera_configs],
                    sync_mode="software_trigger",
                    sync_fps=fps,
                    pixel_format=pixel_format,
                    output_format="mp4",
                    lossless=False,
                    preview_scale=preview_scale,
                    cameras=camera_configs,
                )
                save_capture_config(session_paths.capture_config_path, config)
                print(f"Saved prepare session to: {session_paths.session_dir}")
            else:
                print("Prepare session exited without saving.")
        finally:
            cv2.destroyAllWindows()
            for camera in reversed(cameras):
                camera.close()
