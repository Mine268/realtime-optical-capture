from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


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


def _draw_camera(ax, position: np.ndarray, orientation: np.ndarray, name: str, scale: float = 200.0) -> None:
    origin = position
    axes = orientation
    colors = ["r", "g", "b"]
    for axis_index, color in enumerate(colors):
        direction = axes[:, axis_index] * scale
        ax.plot(
            [origin[0], origin[0] + direction[0]],
            [origin[1], origin[1] + direction[1]],
            [origin[2], origin[2] + direction[2]],
            color=color,
            linewidth=2,
        )

    forward = axes[:, 2] * scale * 1.5
    right = axes[:, 0] * scale * 0.7
    up = axes[:, 1] * scale * 0.5
    apex = origin
    base_center = origin + forward
    corners = [
        base_center + right + up,
        base_center + right - up,
        base_center - right - up,
        base_center - right + up,
    ]
    for corner in corners:
        ax.plot(
            [apex[0], corner[0]],
            [apex[1], corner[1]],
            [apex[2], corner[2]],
            color="gray",
            linewidth=1,
        )
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="gray", linewidth=1)

    ax.scatter([origin[0]], [origin[1]], [origin[2]], color="k", s=20)
    ax.text(origin[0], origin[1], origin[2], name, fontsize=8)


def _draw_charuco(ax, charuco_frame: np.ndarray) -> None:
    valid = ~np.isnan(charuco_frame).any(axis=1)
    points = charuco_frame[valid]
    if len(points) == 0:
        return
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="tab:orange", s=18, label="Charuco corners")
    for point_index, point in enumerate(points):
        if point_index >= 8:
            break
        ax.text(point[0], point[1], point[2], str(point_index), fontsize=7, color="tab:orange")


def save_calibration_visualization(
    output_path: Path,
    camera_names: list[str],
    positions: list[list[float]],
    orientations: list[list[list[float]]],
    matrices: list[list[list[float]]],
    distortions: list[list[float]],
    charuco_3d: np.ndarray | None,
    summary: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.2])
    text_ax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], projection="3d")

    text_ax.axis("off")
    lines = [
        "Calibration Summary",
        "",
        f"Mean reprojection error: {summary.get('mean_reprojection_error_px', 'n/a')}",
        f"Dictionary: {summary.get('selected_dictionary', 'n/a')}",
        f"Ground plane success: {summary.get('ground_plane_success', 'n/a')}",
        "",
        "Camera intrinsics:",
    ]
    for name, matrix, distortion in zip(camera_names, matrices, distortions):
        fx = matrix[0][0]
        fy = matrix[1][1]
        cx = matrix[0][2]
        cy = matrix[1][2]
        lines.append(f"{name}")
        lines.append(f"  fx={fx:.2f} fy={fy:.2f}")
        lines.append(f"  cx={cx:.2f} cy={cy:.2f}")
        lines.append(f"  dist={np.array(distortion)[:5]}")
    text_ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)

    world_points = []
    for name, pos, orient in zip(camera_names, positions, orientations):
        pos_np = np.array(pos, dtype=float)
        orient_np = np.array(orient, dtype=float)
        world_points.append(pos_np)
        _draw_camera(ax, pos_np, orient_np, name)

    if charuco_3d is not None and len(charuco_3d) > 0:
        valid_frames = [frame for frame in charuco_3d if np.any(~np.isnan(frame))]
        if valid_frames:
            first_frame = valid_frames[0]
            _draw_charuco(ax, first_frame)
            world_points.extend(first_frame[~np.isnan(first_frame).any(axis=1)])

    if not world_points:
        world_points = [np.zeros(3)]
    world_points_np = np.array(world_points, dtype=float)
    _set_equal_axes(ax, world_points_np)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Extrinsics and First Valid Charuco Board Pose")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
