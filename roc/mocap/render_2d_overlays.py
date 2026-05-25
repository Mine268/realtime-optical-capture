from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


POSE_EDGES = [
    (0, 1),
    (0, 4),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
]

HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render per-camera 2D mocap overlays")
    parser.add_argument("--npz-path", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--frame-limit", type=int, default=0)
    return parser.parse_args()


def _valid_point(point: np.ndarray, confidence: float, threshold: float) -> bool:
    return bool(confidence > threshold and np.all(np.isfinite(point)))


def _draw_landmarks(
    frame: np.ndarray,
    xy: np.ndarray,
    confidence: np.ndarray,
    edges: list[tuple[int, int]],
    color: tuple[int, int, int],
    threshold: float,
    radius: int,
) -> None:
    for start, end in edges:
        if not (
            _valid_point(xy[start], float(confidence[start]), threshold)
            and _valid_point(xy[end], float(confidence[end]), threshold)
        ):
            continue
        a = tuple(np.round(xy[start]).astype(int))
        b = tuple(np.round(xy[end]).astype(int))
        cv2.line(frame, a, b, color, 2, cv2.LINE_AA)

    for point, conf in zip(xy, confidence):
        if not _valid_point(point, float(conf), threshold):
            continue
        center = tuple(np.round(point).astype(int))
        cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_bbox(frame: np.ndarray, bbox: np.ndarray) -> None:
    if bbox.shape[0] != 4 or not np.all(np.isfinite(bbox)):
        return
    x0, y0, x1, y1 = np.round(bbox).astype(int)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2, cv2.LINE_AA)


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_name in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Failed to open overlay writer: {path}")


def render_camera_overlay(
    serial: str,
    camera_index: int,
    video_path: Path,
    output_path: Path,
    pose_2d: np.ndarray,
    pose_confidence: np.ndarray,
    left_hand_2d: np.ndarray,
    left_hand_confidence: np.ndarray,
    right_hand_2d: np.ndarray,
    right_hand_confidence: np.ndarray,
    bboxes_2d: np.ndarray,
    confidence_threshold: float,
    frame_limit: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer(output_path, fps, width, height)
    max_frames = pose_2d.shape[1]
    if frame_limit > 0:
        max_frames = min(max_frames, frame_limit)

    try:
        frame_index = 0
        while frame_index < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            _draw_bbox(frame, bboxes_2d[camera_index, frame_index])
            _draw_landmarks(
                frame,
                pose_2d[camera_index, frame_index],
                pose_confidence[camera_index, frame_index],
                POSE_EDGES,
                (0, 255, 0),
                confidence_threshold,
                radius=3,
            )
            _draw_landmarks(
                frame,
                left_hand_2d[camera_index, frame_index],
                left_hand_confidence[camera_index, frame_index],
                HAND_EDGES,
                (0, 128, 255),
                confidence_threshold,
                radius=2,
            )
            _draw_landmarks(
                frame,
                right_hand_2d[camera_index, frame_index],
                right_hand_confidence[camera_index, frame_index],
                HAND_EDGES,
                (255, 0, 255),
                confidence_threshold,
                radius=2,
            )

            cv2.putText(
                frame,
                f"{serial} frame={frame_index}",
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()


def main() -> None:
    args = parse_args()
    render_all_overlays(
        npz_path=args.npz_path,
        video_dir=args.video_dir,
        output_dir=args.output_dir or args.npz_path.parent / "overlay_videos",
        confidence_threshold=args.confidence_threshold,
        frame_limit=args.frame_limit,
    )


def render_all_overlays(
    npz_path: Path,
    video_dir: Path,
    output_dir: Path,
    confidence_threshold: float = 0.1,
    frame_limit: int = 0,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    camera_serials = [str(serial) for serial in data["camera_serials"]]
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "pose_2d",
        "pose_confidence",
        "left_hand_2d",
        "left_hand_confidence",
        "right_hand_2d",
        "right_hand_confidence",
        "bboxes_2d",
    ]
    missing = [name for name in required if name not in data.files]
    if missing:
        raise RuntimeError(f"Missing arrays in npz: {missing}")

    for camera_index, serial in enumerate(camera_serials):
        video_path = video_dir / f"{serial}.mp4"
        if not video_path.is_file():
            raise RuntimeError(f"Missing video for camera {serial}: {video_path}")
        output_path = output_dir / f"{serial}_2d_overlay.mp4"
        render_camera_overlay(
            serial=serial,
            camera_index=camera_index,
            video_path=video_path,
            output_path=output_path,
            pose_2d=data["pose_2d"],
            pose_confidence=data["pose_confidence"],
            left_hand_2d=data["left_hand_2d"],
            left_hand_confidence=data["left_hand_confidence"],
            right_hand_2d=data["right_hand_2d"],
            right_hand_confidence=data["right_hand_confidence"],
            bboxes_2d=data["bboxes_2d"],
            confidence_threshold=confidence_threshold,
            frame_limit=frame_limit,
        )
        print(f"Saved 2D overlay video: {output_path}")


if __name__ == "__main__":
    main()
