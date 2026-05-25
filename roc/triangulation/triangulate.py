from __future__ import annotations

import numpy as np
from aniposelib.cameras import CameraGroup


def triangulate_sequence(camera_group: CameraGroup, points_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_cameras, num_frames, num_landmarks, _ = points_2d.shape
    flat = points_2d.reshape(num_cameras, -1, 2)
    points_3d_flat = camera_group.triangulate(flat, fast=True)
    reprojection = camera_group.reprojection_error(points_3d_flat, flat, mean=True)
    points_3d = points_3d_flat.reshape(num_frames, num_landmarks, 3)
    reprojection = np.asarray(reprojection, dtype=np.float32).reshape(num_frames, num_landmarks)
    return points_3d, reprojection
