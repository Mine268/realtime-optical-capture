from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


POSE_EDGES = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("mouth_left", "mouth_right"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
]

LEFT_HAND_EDGES = [
    ("left_hand_wrist", "left_hand_thumb_cmc"),
    ("left_hand_thumb_cmc", "left_hand_thumb_mcp"),
    ("left_hand_thumb_mcp", "left_hand_thumb_ip"),
    ("left_hand_thumb_ip", "left_hand_thumb_tip"),
    ("left_hand_wrist", "left_hand_index_finger_mcp"),
    ("left_hand_index_finger_mcp", "left_hand_index_finger_pip"),
    ("left_hand_index_finger_pip", "left_hand_index_finger_dip"),
    ("left_hand_index_finger_dip", "left_hand_index_finger_tip"),
    ("left_hand_wrist", "left_hand_middle_finger_mcp"),
    ("left_hand_middle_finger_mcp", "left_hand_middle_finger_pip"),
    ("left_hand_middle_finger_pip", "left_hand_middle_finger_dip"),
    ("left_hand_middle_finger_dip", "left_hand_middle_finger_tip"),
    ("left_hand_wrist", "left_hand_ring_finger_mcp"),
    ("left_hand_ring_finger_mcp", "left_hand_ring_finger_pip"),
    ("left_hand_ring_finger_pip", "left_hand_ring_finger_dip"),
    ("left_hand_ring_finger_dip", "left_hand_ring_finger_tip"),
    ("left_hand_wrist", "left_hand_pinky_mcp"),
    ("left_hand_pinky_mcp", "left_hand_pinky_pip"),
    ("left_hand_pinky_pip", "left_hand_pinky_dip"),
    ("left_hand_pinky_dip", "left_hand_pinky_tip"),
]

RIGHT_HAND_EDGES = [
    ("right_hand_wrist", "right_hand_thumb_cmc"),
    ("right_hand_thumb_cmc", "right_hand_thumb_mcp"),
    ("right_hand_thumb_mcp", "right_hand_thumb_ip"),
    ("right_hand_thumb_ip", "right_hand_thumb_tip"),
    ("right_hand_wrist", "right_hand_index_finger_mcp"),
    ("right_hand_index_finger_mcp", "right_hand_index_finger_pip"),
    ("right_hand_index_finger_pip", "right_hand_index_finger_dip"),
    ("right_hand_index_finger_dip", "right_hand_index_finger_tip"),
    ("right_hand_wrist", "right_hand_middle_finger_mcp"),
    ("right_hand_middle_finger_mcp", "right_hand_middle_finger_pip"),
    ("right_hand_middle_finger_pip", "right_hand_middle_finger_dip"),
    ("right_hand_middle_finger_dip", "right_hand_middle_finger_tip"),
    ("right_hand_wrist", "right_hand_ring_finger_mcp"),
    ("right_hand_ring_finger_mcp", "right_hand_ring_finger_pip"),
    ("right_hand_ring_finger_pip", "right_hand_ring_finger_dip"),
    ("right_hand_ring_finger_dip", "right_hand_ring_finger_tip"),
    ("right_hand_wrist", "right_hand_pinky_mcp"),
    ("right_hand_pinky_mcp", "right_hand_pinky_pip"),
    ("right_hand_pinky_pip", "right_hand_pinky_dip"),
    ("right_hand_pinky_dip", "right_hand_pinky_tip"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render mocap npz to mp4")
    parser.add_argument("--npz-path", required=True, type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--frame-limit", type=int, default=0)
    return parser.parse_args()


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if not np.isfinite(radius) or radius <= 0:
        radius = 1000.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_edges(ax, frame_points: dict[str, np.ndarray], edges, color: str) -> None:
    for start, end in edges:
        if start not in frame_points or end not in frame_points:
            continue
        a = frame_points[start]
        b = frame_points[end]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=2)


def main() -> None:
    args = parse_args()
    data = np.load(args.npz_path, allow_pickle=True)
    points_3d = data["points_3d"]
    landmark_names = [str(name) for name in data["landmark_names"]]
    reprojection = data["reprojection_error"] if "reprojection_error" in data.files else None

    frame_count = points_3d.shape[0]
    if args.frame_limit > 0:
        frame_count = min(frame_count, args.frame_limit)

    output_path = args.output_path or args.npz_path.with_name(args.npz_path.stem + "_preview.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_points = points_3d[~np.isnan(points_3d).any(axis=2)]
    if valid_points.size == 0:
        raise RuntimeError("No valid 3D points to render")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.tight_layout()

    writer = None
    try:
        for frame_index in range(frame_count):
            ax.cla()
            frame = points_3d[frame_index]
            frame_points = {name: frame[idx] for idx, name in enumerate(landmark_names)}

            valid = ~np.isnan(frame).any(axis=1)
            scatter_points = frame[valid]
            ax.scatter(scatter_points[:, 0], scatter_points[:, 1], scatter_points[:, 2], c="k", s=10)
            _draw_edges(ax, frame_points, POSE_EDGES, "tab:blue")
            _draw_edges(ax, frame_points, LEFT_HAND_EDGES, "tab:red")
            _draw_edges(ax, frame_points, RIGHT_HAND_EDGES, "tab:green")

            _set_equal_axes(ax, valid_points.reshape(-1, 3))
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            title = f"frame={frame_index}"
            if reprojection is not None:
                title += f"  reproj={np.nanmean(reprojection[frame_index]):.3f}"
            ax.set_title(title)
            ax.view_init(elev=18, azim=-70)

            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open output video writer: {output_path}")
            writer.write(image_bgr)
    finally:
        plt.close(fig)
        if writer is not None:
            writer.release()

    print(f"Saved mocap preview video to: {output_path}")


if __name__ == "__main__":
    main()
