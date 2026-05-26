from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt


@dataclass(slots=True)
class PostprocessReport:
    velocity_threshold_mm: float | None
    max_gap_frames: int
    butterworth_cutoff_hz: float
    butterworth_order: int
    fps: float
    velocity_outliers_removed: int
    interpolated_values: int
    butterworth_filtered: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "velocity_threshold_mm": self.velocity_threshold_mm,
            "max_gap_frames": self.max_gap_frames,
            "butterworth_cutoff_hz": self.butterworth_cutoff_hz,
            "butterworth_order": self.butterworth_order,
            "fps": self.fps,
            "velocity_outliers_removed": self.velocity_outliers_removed,
            "interpolated_values": self.interpolated_values,
            "butterworth_filtered": self.butterworth_filtered,
        }


def postprocess_points_3d(
    points_3d: np.ndarray,
    fps: float,
    velocity_threshold_mm: float | None = None,
    max_gap_frames: int = 10,
    butterworth_cutoff_hz: float = 1.2,
    butterworth_order: int = 4,
) -> tuple[np.ndarray, PostprocessReport]:
    filtered = points_3d.astype(np.float32, copy=True)
    removed = 0
    if velocity_threshold_mm is not None:
        removed = _remove_velocity_outliers(filtered, velocity_threshold_mm)
    interpolated = _interpolate_short_gaps(filtered, max_gap_frames)
    butterworth_applied = _butterworth_filter_in_place(
        filtered,
        fps=max(fps, 1.0),
        cutoff_hz=butterworth_cutoff_hz,
        order=butterworth_order,
    )
    report = PostprocessReport(
        velocity_threshold_mm=velocity_threshold_mm,
        max_gap_frames=max_gap_frames,
        butterworth_cutoff_hz=butterworth_cutoff_hz,
        butterworth_order=butterworth_order,
        fps=float(fps),
        velocity_outliers_removed=removed,
        interpolated_values=interpolated,
        butterworth_filtered=butterworth_applied,
    )
    return filtered, report


def _remove_velocity_outliers(points_3d: np.ndarray, threshold_mm: float) -> int:
    finite = np.isfinite(points_3d).all(axis=2)
    step = np.linalg.norm(np.diff(points_3d, axis=0), axis=2)
    step[~(finite[1:] & finite[:-1])] = np.nan
    jump_after = step > threshold_mm
    removed = 0
    for frame_minus_one, landmark_index in zip(*np.where(jump_after)):
        if np.isfinite(points_3d[frame_minus_one + 1, landmark_index]).all():
            points_3d[frame_minus_one + 1, landmark_index] = np.nan
            removed += 1
    return removed


def _interpolate_short_gaps(points_3d: np.ndarray, max_gap_frames: int) -> int:
    total_interpolated_values = 0
    frame_indices = np.arange(points_3d.shape[0])
    for landmark_index in range(points_3d.shape[1]):
        finite = np.isfinite(points_3d[:, landmark_index]).all(axis=1)
        if finite.sum() < 2:
            continue
        for axis in range(3):
            values = points_3d[:, landmark_index, axis]
            finite_axis = np.isfinite(values)
            filled = np.interp(frame_indices, frame_indices[finite_axis], values[finite_axis])
            gap_mask = ~finite_axis
            for start, stop in _contiguous_true_ranges(gap_mask):
                if start == 0 or stop == len(values) or (stop - start) > max_gap_frames:
                    continue
                values[start:stop] = filled[start:stop]
                total_interpolated_values += stop - start
    return total_interpolated_values


def _contiguous_true_ranges(mask: np.ndarray):
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)


def _butterworth_filter_in_place(points_3d: np.ndarray, fps: float, cutoff_hz: float, order: int) -> bool:
    nyquist = fps / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        return False

    b, a = butter(order, cutoff_hz / nyquist, btype="lowpass")
    padlen = 3 * max(len(a), len(b))
    applied = False
    for landmark_index in range(points_3d.shape[1]):
        for axis in range(3):
            values = points_3d[:, landmark_index, axis]
            finite = np.isfinite(values)
            if finite.sum() <= padlen:
                continue
            if not finite.all():
                continue
            points_3d[:, landmark_index, axis] = filtfilt(b, a, values).astype(np.float32)
            applied = True
    return applied
