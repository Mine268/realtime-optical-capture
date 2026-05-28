from __future__ import annotations

import argparse
import csv
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from roc.config.yaml_io import load_capture_config
from roc.mocap.postprocess import RealtimePostprocessor
from roc.tracking.mediapipe_tracker import HAND_LANDMARK_NAMES, POSE_LANDMARK_NAMES, MediapipeTracker
from roc.tracking.model_paths import hand_model_path, pose_model_path_for_complexity
from roc.triangulation.cameras import camera_group_names, camera_order_indices, load_camera_group_from_toml
from roc.triangulation.triangulate import triangulate_sequence


@dataclass(slots=True)
class RunningStats:
    values: list[float]

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def summary(self) -> dict[str, float]:
        if not self.values:
            return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
        values = np.asarray(self.values, dtype=np.float64) * 1000.0
        return {
            "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark realtime mocap compute using recorded mp4 files")
    parser.add_argument("--prepare-session", required=True, type=Path)
    parser.add_argument("--calib-session", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--complexities", type=int, nargs="+", default=[0, 1, 2], choices=[0, 1, 2])
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--delegate", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    report = benchmark_realtime_models(
        prepare_session=args.prepare_session,
        calib_session=args.calib_session,
        video_dir=args.video_dir,
        frame_limit=args.frames,
        complexities=args.complexities,
        hands_enabled=not args.no_hands,
        delegate=args.delegate,
        output_dir=args.output_dir,
    )
    print(yaml.safe_dump(report, sort_keys=False, allow_unicode=False))


def benchmark_realtime_models(
    prepare_session: Path,
    calib_session: Path,
    video_dir: Path,
    frame_limit: int,
    complexities: list[int],
    hands_enabled: bool,
    delegate: str = "cpu",
    output_dir: Path | None = None,
) -> dict:
    prepare_session = prepare_session.resolve()
    calib_session = calib_session.resolve()
    source_video_dir = _resolve_video_dir(video_dir)
    output_dir = output_dir or source_video_dir.parent / "realtime_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_config = load_capture_config(prepare_session / "capture_config.yaml")
    camera_group = load_camera_group_from_toml(calib_session / "calibration.toml")
    calibrated_serials = camera_group_names(camera_group)
    serial_to_video = {path.stem: path for path in source_video_dir.glob("*.mp4")}
    source_serials = [serial for serial in capture_config.camera_serials if serial in serial_to_video]
    if not source_serials:
        raise RuntimeError(f"No matching mp4 files found in {source_video_dir}")
    reorder_indices = camera_order_indices(source_serials, calibrated_serials)
    source_fps = _read_video_fps(source_video_dir)

    benchmark_report = {
        "video_dir": str(source_video_dir),
        "prepare_session": str(prepare_session),
        "calib_session": str(calib_session),
        "source_fps": source_fps,
        "source_serials": source_serials,
        "calibrated_serials": calibrated_serials,
        "hands_enabled": hands_enabled,
        "delegate": delegate,
        "frame_limit": frame_limit,
        "postprocess_realtime": {
            "type": "causal_ema_hold",
            "cutoff_hz": 1.2,
            "max_hold_frames": 3,
        },
        "results": [],
    }

    csv_rows = []
    for complexity in complexities:
        result = _benchmark_one_complexity(
            complexity=complexity,
            hands_enabled=hands_enabled,
            source_serials=source_serials,
            serial_to_video=serial_to_video,
            source_fps=source_fps,
            camera_group=camera_group,
            reorder_indices=reorder_indices,
            frame_limit=frame_limit,
            delegate=delegate,
        )
        benchmark_report["results"].append(result)
        csv_rows.append(_flatten_result(result))

    yaml_path = output_dir / "realtime_benchmark_report.yaml"
    csv_path = output_dir / "realtime_benchmark_summary.csv"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(benchmark_report, handle, sort_keys=False, allow_unicode=False)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved realtime benchmark report: {yaml_path}")
    print(f"Saved realtime benchmark summary: {csv_path}")
    return benchmark_report


def _benchmark_one_complexity(
    complexity: int,
    hands_enabled: bool,
    source_serials: list[str],
    serial_to_video: dict[str, Path],
    source_fps: float,
    camera_group,
    reorder_indices: list[int],
    frame_limit: int,
    delegate: str,
) -> dict:
    pose_model_path = pose_model_path_for_complexity(complexity)
    hand_model_path_value = hand_model_path() if hands_enabled else None
    if not pose_model_path.is_file():
        return {
            "model_complexity": complexity,
            "status": "skipped",
            "delegate": delegate,
            "reason": f"Pose model not found: {pose_model_path}",
        }
    if hands_enabled and (hand_model_path_value is None or not hand_model_path_value.is_file()):
        return {
            "model_complexity": complexity,
            "status": "skipped",
            "delegate": delegate,
            "reason": f"Hand model not found: {hand_model_path_value}",
        }

    stats = {
        "read": RunningStats([]),
        "pose": RunningStats([]),
        "hands": RunningStats([]),
        "triangulate": RunningStats([]),
        "postprocess": RunningStats([]),
        "frame_set_total": RunningStats([]),
    }
    num_landmarks = len(POSE_LANDMARK_NAMES) + 2 * len(HAND_LANDMARK_NAMES)
    online_filter = RealtimePostprocessor(num_landmarks=num_landmarks, fps=source_fps, cutoff_hz=1.2, max_hold_frames=3)

    caps: list[tuple[str, cv2.VideoCapture]] = []
    try:
        for serial in source_serials:
            cap = cv2.VideoCapture(str(serial_to_video[serial]))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video for camera {serial}: {serial_to_video[serial]}")
            caps.append((serial, cap))

        with ExitStack() as stack:
            try:
                trackers = {
                    serial: stack.enter_context(
                        MediapipeTracker(
                            pose_model_path=pose_model_path,
                            hand_model_path=hand_model_path_value,
                            model_complexity=complexity,
                            hands_enabled=hands_enabled,
                            delegate=delegate,
                        )
                    )
                    for serial in source_serials
                }
            except Exception as exc:
                return {
                    "model_complexity": complexity,
                    "status": "failed",
                    "hands_enabled": hands_enabled,
                    "delegate": delegate,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            processed = 0
            while frame_limit <= 0 or processed < frame_limit:
                frame_start = time.perf_counter()
                timestamp_ms = round(processed * 1000.0 / max(source_fps, 1.0))
                frames = []
                read_start = time.perf_counter()
                for serial, cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        return _result(complexity, hands_enabled, processed, stats, online_filter, delegate)
                    frames.append((serial, frame))
                stats["read"].add(time.perf_counter() - read_start)

                pose_2d = []
                pose_conf = []
                left_2d = []
                left_conf = []
                right_2d = []
                right_conf = []
                for serial, frame in frames:
                    tracker = trackers[serial]
                    pose_start = time.perf_counter()
                    pose_result = tracker.detect_pose(frame, timestamp_ms=timestamp_ms)
                    stats["pose"].add(time.perf_counter() - pose_start)

                    hand_start = time.perf_counter()
                    hand_result = tracker.detect_hands(frame, timestamp_ms=timestamp_ms)
                    stats["hands"].add(time.perf_counter() - hand_start)

                    pose_2d.append(pose_result.xy)
                    pose_conf.append(pose_result.confidence)
                    left_2d.append(hand_result.left_xy)
                    left_conf.append(hand_result.left_confidence)
                    right_2d.append(hand_result.right_xy)
                    right_conf.append(hand_result.right_confidence)

                pose_2d_np = np.asarray(pose_2d, dtype=np.float32)[reorder_indices]
                pose_conf_np = np.asarray(pose_conf, dtype=np.float32)[reorder_indices]
                left_2d_np = np.asarray(left_2d, dtype=np.float32)[reorder_indices]
                left_conf_np = np.asarray(left_conf, dtype=np.float32)[reorder_indices]
                right_2d_np = np.asarray(right_2d, dtype=np.float32)[reorder_indices]
                right_conf_np = np.asarray(right_conf, dtype=np.float32)[reorder_indices]

                all_landmarks_2d = np.concatenate([pose_2d_np, left_2d_np, right_2d_np], axis=1)[:, None, :, :]
                all_conf = np.concatenate([pose_conf_np, left_conf_np, right_conf_np], axis=1)[:, None, :]
                all_landmarks_2d = np.where(all_conf[..., None] <= 0.1, np.nan, all_landmarks_2d)

                tri_start = time.perf_counter()
                points_3d, _ = triangulate_sequence(camera_group, all_landmarks_2d)
                stats["triangulate"].add(time.perf_counter() - tri_start)

                post_start = time.perf_counter()
                online_filter.update(points_3d[0].astype(np.float32))
                stats["postprocess"].add(time.perf_counter() - post_start)
                stats["frame_set_total"].add(time.perf_counter() - frame_start)
                processed += 1
    finally:
        for _, cap in caps:
            cap.release()

    return _result(complexity, hands_enabled, frame_limit, stats, online_filter, delegate)


def _result(
    complexity: int,
    hands_enabled: bool,
    frames_processed: int,
    stats: dict[str, RunningStats],
    online_filter: RealtimePostprocessor,
    delegate: str,
) -> dict:
    frame_set_mean_s = float(np.mean(stats["frame_set_total"].values)) if stats["frame_set_total"].values else 0.0
    result = {
        "model_complexity": complexity,
        "status": "ok",
        "hands_enabled": hands_enabled,
        "delegate": delegate,
        "frames_processed": frames_processed,
        "estimated_frame_set_fps": 1.0 / frame_set_mean_s if frame_set_mean_s > 0 else 0.0,
        "estimated_per_camera_fps": (
            len(stats["pose"].values) / sum(stats["pose"].values) if sum(stats["pose"].values) > 0 else 0.0
        ),
        "timings": {name: stat.summary() for name, stat in stats.items()},
        "postprocess_report": online_filter.report().to_dict(),
    }
    return result


def _flatten_result(result: dict) -> dict:
    if result.get("status") != "ok":
        return {
            "model_complexity": result.get("model_complexity"),
            "status": result.get("status"),
            "delegate": result.get("delegate", ""),
            "reason": result.get("reason", ""),
        }
    flattened = {
        "model_complexity": result["model_complexity"],
        "status": result["status"],
        "hands_enabled": result["hands_enabled"],
        "delegate": result["delegate"],
        "frames_processed": result["frames_processed"],
        "estimated_frame_set_fps": result["estimated_frame_set_fps"],
        "estimated_per_camera_fps": result["estimated_per_camera_fps"],
    }
    for name, values in result["timings"].items():
        for key, value in values.items():
            flattened[f"{name}_{key}"] = value
    return flattened


def _resolve_video_dir(video_dir: Path) -> Path:
    resolved = video_dir.resolve()
    if resolved.name.startswith("mocap_") and (resolved / "videos").is_dir():
        return resolved / "videos"
    return resolved


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
    return 10.0


if __name__ == "__main__":
    main()
