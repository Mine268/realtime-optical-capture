from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from roc.io.video import H264VideoWriter
from roc.mocap.render_2d_overlays import HAND_EDGES, POSE_EDGES, render_combined_overlay
from roc.triangulation.cameras import load_camera_group_from_toml


SMPLX_BODY_EDGES = [
    ("pelvis", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "neck"),
    ("neck", "head"),
    ("spine3", "left_collar"),
    ("left_collar", "left_shoulder"),
    ("spine3", "right_collar"),
    ("right_collar", "right_shoulder"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]

SMPLX_LEFT_HAND_EDGES = [
    ("left_wrist", "left_index1"),
    ("left_index1", "left_index2"),
    ("left_index2", "left_index3"),
    ("left_wrist", "left_middle1"),
    ("left_middle1", "left_middle2"),
    ("left_middle2", "left_middle3"),
    ("left_wrist", "left_pinky1"),
    ("left_pinky1", "left_pinky2"),
    ("left_pinky2", "left_pinky3"),
    ("left_wrist", "left_ring1"),
    ("left_ring1", "left_ring2"),
    ("left_ring2", "left_ring3"),
    ("left_wrist", "left_thumb1"),
    ("left_thumb1", "left_thumb2"),
    ("left_thumb2", "left_thumb3"),
]

SMPLX_RIGHT_HAND_EDGES = [
    ("right_wrist", "right_index1"),
    ("right_index1", "right_index2"),
    ("right_index2", "right_index3"),
    ("right_wrist", "right_middle1"),
    ("right_middle1", "right_middle2"),
    ("right_middle2", "right_middle3"),
    ("right_wrist", "right_pinky1"),
    ("right_pinky1", "right_pinky2"),
    ("right_pinky2", "right_pinky3"),
    ("right_wrist", "right_ring1"),
    ("right_ring1", "right_ring2"),
    ("right_ring2", "right_ring3"),
    ("right_wrist", "right_thumb1"),
    ("right_thumb1", "right_thumb2"),
    ("right_thumb2", "right_thumb3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 3D mocap reprojections over camera videos")
    parser.add_argument("--npz-path", required=True, type=Path)
    parser.add_argument("--calibration-toml", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--points-key", default="points_3d", choices=("points_3d", "points_3d_raw"))
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument("--combined-scale", type=float, default=0.5)
    return parser.parse_args()


def _valid_xy(point: np.ndarray, max_abs: float = 1e7) -> bool:
    return bool(np.all(np.isfinite(point)) and np.all(np.abs(point) < max_abs))


def _valid_detected_xy(point: np.ndarray, confidence: float, threshold: float) -> bool:
    return bool(confidence > threshold and _valid_xy(point))


def _draw_edges(
    frame: np.ndarray,
    xy: np.ndarray,
    edges: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for start, end in edges:
        if start >= len(xy) or end >= len(xy):
            continue
        if not (_valid_xy(xy[start]) and _valid_xy(xy[end])):
            continue
        a = tuple(int(value) for value in np.round(xy[start]))
        b = tuple(int(value) for value in np.round(xy[end]))
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)


def _draw_points(
    frame: np.ndarray,
    xy: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    for point in xy:
        if not _valid_xy(point):
            continue
        center = tuple(int(value) for value in np.round(point))
        cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_detected_points(
    frame: np.ndarray,
    xy: np.ndarray,
    confidence: np.ndarray,
    edges: list[tuple[int, int]],
    color: tuple[int, int, int],
    threshold: float,
) -> None:
    filtered = xy.copy()
    invalid = np.array(
        [not _valid_detected_xy(point, float(conf), threshold) for point, conf in zip(xy, confidence)],
        dtype=bool,
    )
    filtered[invalid] = np.nan
    _draw_edges(frame, filtered, edges, color, 1)
    _draw_points(frame, filtered, color, 2)


def _draw_label(frame: np.ndarray, serial: str, frame_index: int, points_key: str) -> None:
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
    cv2.putText(
        frame,
        f"green=2D  red=3D reprojection ({points_key})",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_smplx_label(frame: np.ndarray, serial: str, source_frame_index: int) -> None:
    cv2.putText(
        frame,
        f"{serial} frame={source_frame_index}",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "green=2D  cyan=SMPL-X reprojection",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _project_points(camera_group, points_3d: np.ndarray, frame_count: int) -> np.ndarray:
    projected = camera_group.project(points_3d[:frame_count].reshape(-1, 3))
    num_cameras = projected.shape[0]
    num_landmarks = points_3d.shape[1]
    return projected.reshape(num_cameras, frame_count, num_landmarks, 2)


def render_reprojection_overlays(
    npz_path: Path,
    calibration_toml: Path,
    video_dir: Path,
    output_dir: Path,
    points_key: str = "points_3d",
    confidence_threshold: float = 0.1,
    frame_limit: int = 0,
    combined_scale: float = 0.5,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    if points_key not in data.files:
        raise RuntimeError(f"Missing {points_key} in npz: {npz_path}")

    camera_serials = [str(serial) for serial in data["camera_serials"]]
    points_3d = data[points_key]
    frame_count = points_3d.shape[0]
    if frame_limit > 0:
        frame_count = min(frame_count, frame_limit)

    camera_group = load_camera_group_from_toml(calibration_toml)
    camera_names = [camera.get_name() for camera in camera_group.cameras]
    serial_to_projection_index = {name: index for index, name in enumerate(camera_names)}
    missing = [serial for serial in camera_serials if serial not in serial_to_projection_index]
    if missing:
        raise RuntimeError(f"Camera serials missing from calibration: {missing}")

    projected_all = _project_points(camera_group, points_3d, frame_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for camera_index, serial in enumerate(camera_serials):
        video_path = video_dir / f"{serial}.mp4"
        if not video_path.is_file():
            raise RuntimeError(f"Missing video for camera {serial}: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for camera {serial}: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = output_dir / f"{serial}_3d_reprojection_overlay_{points_key}.mp4"
        writer = H264VideoWriter(output_path, width, height, fps, temp_suffix=".reprojection_tmp.avi")
        output_paths.append(output_path)
        projection_index = serial_to_projection_index[serial]

        try:
            for frame_index in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                detected_pose = data["pose_2d"][camera_index, frame_index]
                detected_pose_conf = data["pose_confidence"][camera_index, frame_index]
                detected_left = data["left_hand_2d"][camera_index, frame_index]
                detected_left_conf = data["left_hand_confidence"][camera_index, frame_index]
                detected_right = data["right_hand_2d"][camera_index, frame_index]
                detected_right_conf = data["right_hand_confidence"][camera_index, frame_index]

                reproj = projected_all[projection_index, frame_index]
                reproj_pose = reproj[:33]
                reproj_left = reproj[33:54]
                reproj_right = reproj[54:75]

                _draw_detected_points(frame, detected_pose, detected_pose_conf, POSE_EDGES, (0, 255, 0), confidence_threshold)
                _draw_detected_points(frame, detected_left, detected_left_conf, HAND_EDGES, (0, 180, 0), confidence_threshold)
                _draw_detected_points(frame, detected_right, detected_right_conf, HAND_EDGES, (0, 180, 0), confidence_threshold)

                _draw_edges(frame, reproj_pose, POSE_EDGES, (0, 0, 255), 2)
                _draw_edges(frame, reproj_left, HAND_EDGES, (0, 0, 255), 2)
                _draw_edges(frame, reproj_right, HAND_EDGES, (0, 0, 255), 2)
                _draw_points(frame, reproj_pose, (0, 0, 255), 3)
                _draw_points(frame, reproj_left, (0, 0, 255), 2)
                _draw_points(frame, reproj_right, (0, 0, 255), 2)

                _draw_label(frame, serial, frame_index, points_key)
                writer.write(frame)
        finally:
            cap.release()
            writer.close()

        print(f"Saved 3D reprojection overlay video: {output_path}")

    combined_path = output_dir / f"combined_3d_reprojection_overlay_{points_key}.mp4"
    render_combined_overlay(output_paths, combined_path, frame_limit=frame_limit, scale=combined_scale)
    print(f"Saved combined 3D reprojection overlay video: {combined_path}")


def render_smplx_reprojection_overlays(
    mocap_npz_path: Path,
    smplx_npz_path: Path,
    calibration_toml: Path,
    video_dir: Path,
    output_dir: Path,
    confidence_threshold: float = 0.1,
    frame_limit: int = 0,
    combined_scale: float = 0.5,
) -> None:
    mocap_data = np.load(mocap_npz_path, allow_pickle=True)
    smplx_data = np.load(smplx_npz_path, allow_pickle=True)
    if "smplx_joints" not in smplx_data.files:
        raise RuntimeError(f"Missing smplx_joints in npz: {smplx_npz_path}")

    camera_serials = [str(serial) for serial in mocap_data["camera_serials"]]
    frame_indices = np.asarray(smplx_data["frame_indices"], dtype=np.int32)
    smplx_joints = np.asarray(smplx_data["smplx_joints"], dtype=np.float32)
    # squeeze extra batch dim: (F, 1, J, 3) → (F, J, 3)
    if smplx_joints.ndim == 4 and smplx_joints.shape[1] == 1:
        smplx_joints = smplx_joints[:, 0, :, :]
    input_scale = float(np.asarray(smplx_data["input_scale"]).reshape(())) if "input_scale" in smplx_data.files else 0.001
    if input_scale <= 0:
        raise RuntimeError(f"Invalid SMPL-X input_scale in {smplx_npz_path}: {input_scale}")
    smplx_joints_camera_units = smplx_joints / input_scale

    frame_count = smplx_joints_camera_units.shape[0]
    if frame_limit > 0:
        frame_count = min(frame_count, frame_limit)
    frame_indices = frame_indices[:frame_count]

    camera_group = load_camera_group_from_toml(calibration_toml)
    camera_names = [camera.get_name() for camera in camera_group.cameras]
    serial_to_projection_index = {name: index for index, name in enumerate(camera_names)}
    missing = [serial for serial in camera_serials if serial not in serial_to_projection_index]
    if missing:
        raise RuntimeError(f"Camera serials missing from calibration: {missing}")

    projected_all = _project_points(camera_group, smplx_joints_camera_units, frame_count)
    smplx_edges = _smplx_edges_as_indices()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for camera_index, serial in enumerate(camera_serials):
        video_path = video_dir / f"{serial}.mp4"
        if not video_path.is_file():
            raise RuntimeError(f"Missing video for camera {serial}: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for camera {serial}: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = output_dir / f"{serial}_smplx_reprojection_overlay.mp4"
        writer = H264VideoWriter(output_path, width, height, fps, temp_suffix=".smplx_reprojection_tmp.avi")
        output_paths.append(output_path)
        projection_index = serial_to_projection_index[serial]

        try:
            for output_frame_index, source_frame_index in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame_index))
                ret, frame = cap.read()
                if not ret:
                    break

                detected_pose = mocap_data["pose_2d"][camera_index, source_frame_index]
                detected_pose_conf = mocap_data["pose_confidence"][camera_index, source_frame_index]
                detected_left = mocap_data["left_hand_2d"][camera_index, source_frame_index]
                detected_left_conf = mocap_data["left_hand_confidence"][camera_index, source_frame_index]
                detected_right = mocap_data["right_hand_2d"][camera_index, source_frame_index]
                detected_right_conf = mocap_data["right_hand_confidence"][camera_index, source_frame_index]

                _draw_detected_points(frame, detected_pose, detected_pose_conf, POSE_EDGES, (0, 255, 0), confidence_threshold)
                _draw_detected_points(frame, detected_left, detected_left_conf, HAND_EDGES, (0, 180, 0), confidence_threshold)
                _draw_detected_points(frame, detected_right, detected_right_conf, HAND_EDGES, (0, 180, 0), confidence_threshold)

                reproj = projected_all[projection_index, output_frame_index]
                _draw_edges(frame, reproj, smplx_edges["body"], (255, 255, 0), 2)
                _draw_edges(frame, reproj, smplx_edges["left_hand"], (255, 180, 0), 1)
                _draw_edges(frame, reproj, smplx_edges["right_hand"], (255, 180, 0), 1)
                _draw_points(frame, reproj[:55], (255, 255, 0), 2)

                _draw_smplx_label(frame, serial, int(source_frame_index))
                writer.write(frame)
        finally:
            cap.release()
            writer.close()

        print(f"Saved SMPL-X reprojection overlay video: {output_path}")

    combined_path = output_dir / "combined_smplx_reprojection_overlay.mp4"
    render_combined_overlay(output_paths, combined_path, frame_limit=frame_count, scale=combined_scale)
    print(f"Saved combined SMPL-X reprojection overlay video: {combined_path}")


def _smplx_edges_as_indices() -> dict[str, list[tuple[int, int]]]:
    try:
        from smplx.joint_names import JOINT_NAMES
    except ImportError as exc:
        raise RuntimeError("SMPL-X reprojection requires the smplx Python package") from exc

    name_to_idx = {name: index for index, name in enumerate(JOINT_NAMES)}

    def convert(edges: list[tuple[str, str]]) -> list[tuple[int, int]]:
        return [(name_to_idx[start], name_to_idx[end]) for start, end in edges if start in name_to_idx and end in name_to_idx]

    return {
        "body": convert(SMPLX_BODY_EDGES),
        "left_hand": convert(SMPLX_LEFT_HAND_EDGES),
        "right_hand": convert(SMPLX_RIGHT_HAND_EDGES),
    }


def main() -> None:
    args = parse_args()
    render_reprojection_overlays(
        npz_path=args.npz_path,
        calibration_toml=args.calibration_toml,
        video_dir=args.video_dir,
        output_dir=args.output_dir or args.npz_path.parent / "reprojection_videos",
        points_key=args.points_key,
        confidence_threshold=args.confidence_threshold,
        frame_limit=args.frame_limit,
        combined_scale=args.combined_scale,
    )


if __name__ == "__main__":
    main()
