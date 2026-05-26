from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze temporal jitter in a mocap npz")
    parser.add_argument("--npz-path", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jump-threshold-mm", type=float, default=500.0)
    parser.add_argument("--large-jump-threshold-mm", type=float, default=1000.0)
    return parser.parse_args()


def _safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    if not np.any(np.isfinite(values)):
        return None
    return float(np.nanpercentile(values, percentile))


def _safe_median(values: np.ndarray) -> float | None:
    if not np.any(np.isfinite(values)):
        return None
    return float(np.nanmedian(values))


def analyze_jitter(
    npz_path: Path,
    output_dir: Path | None = None,
    jump_threshold_mm: float = 500.0,
    large_jump_threshold_mm: float = 1000.0,
) -> tuple[Path, Path]:
    data = np.load(npz_path, allow_pickle=True)
    points_3d = data["points_3d"].astype(float)
    landmark_names = [str(name) for name in data["landmark_names"]]
    reprojection = data["reprojection_error"].astype(float) if "reprojection_error" in data.files else None
    timestamps = data["timestamps"].astype(float) if "timestamps" in data.files else None

    output_dir = output_dir or npz_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    finite = np.isfinite(points_3d).all(axis=2)
    step = np.linalg.norm(np.diff(points_3d, axis=0), axis=2)
    step_valid = finite[1:] & finite[:-1]
    step[~step_valid] = np.nan

    median_points = np.nanmedian(np.where(finite[..., None], points_3d, np.nan), axis=0)
    deviation = np.linalg.norm(points_3d - median_points[None, :, :], axis=2)
    deviation[~finite] = np.nan

    rows = []
    for index, name in enumerate(landmark_names):
        landmark_step = step[:, index]
        landmark_dev = deviation[:, index]
        landmark_reproj = (
            reprojection[:, index]
            if reprojection is not None and reprojection.shape[1] > index
            else np.full(points_3d.shape[0], np.nan)
        )
        valid_count = int(finite[:, index].sum())
        rows.append(
            {
                "index": index,
                "name": name,
                "valid_frames": valid_count,
                "valid_ratio": valid_count / points_3d.shape[0],
                "median_step_mm": _safe_median(landmark_step),
                "p95_step_mm": _safe_percentile(landmark_step, 95),
                "p99_step_mm": _safe_percentile(landmark_step, 99),
                "max_step_mm": _safe_percentile(landmark_step, 100),
                "p95_deviation_from_median_mm": _safe_percentile(landmark_dev, 95),
                "max_deviation_from_median_mm": _safe_percentile(landmark_dev, 100),
                "median_reprojection_error_px": _safe_median(landmark_reproj),
                f"jumps_gt_{int(jump_threshold_mm)}mm": int(np.nansum(landmark_step > jump_threshold_mm)),
                f"jumps_gt_{int(large_jump_threshold_mm)}mm": int(np.nansum(landmark_step > large_jump_threshold_mm)),
            }
        )

    frame_jump_counts = np.nansum(step > jump_threshold_mm, axis=1).astype(int)
    worst_frame_indices = np.argsort(-frame_jump_counts)
    worst_transitions = []
    for transition_index in worst_frame_indices:
        if frame_jump_counts[transition_index] <= 0:
            continue
        landmark_index = int(np.nanargmax(step[transition_index]))
        item = {
            "from_frame": int(transition_index),
            "to_frame": int(transition_index + 1),
            "jump_count": int(frame_jump_counts[transition_index]),
            "max_step_mm": float(np.nanmax(step[transition_index])),
            "max_step_landmark_index": landmark_index,
            "max_step_landmark_name": landmark_names[landmark_index],
        }
        if reprojection is not None:
            item["max_step_reprojection_after_px"] = float(reprojection[transition_index + 1, landmark_index])
        worst_transitions.append(item)
        if len(worst_transitions) >= 20:
            break

    group_ranges = {
        "pose": range(0, 33),
        "left_hand": range(33, 54),
        "right_hand": range(54, 75),
    }
    groups = {}
    for group_name, indices in group_ranges.items():
        group_step = step[:, list(indices)]
        groups[group_name] = {
            "valid_point_steps": int(np.isfinite(group_step).sum()),
            "median_step_mm": _safe_median(group_step),
            "p95_step_mm": _safe_percentile(group_step, 95),
            "p99_step_mm": _safe_percentile(group_step, 99),
            "max_step_mm": _safe_percentile(group_step, 100),
            f"jumps_gt_{int(jump_threshold_mm)}mm": int(np.nansum(group_step > jump_threshold_mm)),
            f"jumps_gt_{int(large_jump_threshold_mm)}mm": int(np.nansum(group_step > large_jump_threshold_mm)),
        }

    report = {
        "npz_path": str(npz_path),
        "frames": int(points_3d.shape[0]),
        "landmarks": int(points_3d.shape[1]),
        "jump_threshold_mm": float(jump_threshold_mm),
        "large_jump_threshold_mm": float(large_jump_threshold_mm),
        "timestamp_step_ms": None,
        "total_valid_point_steps": int(np.isfinite(step).sum()),
        f"total_jumps_gt_{int(jump_threshold_mm)}mm": int(np.nansum(step > jump_threshold_mm)),
        f"total_jumps_gt_{int(large_jump_threshold_mm)}mm": int(np.nansum(step > large_jump_threshold_mm)),
        "global_step_stats_mm": {
            "median": _safe_median(step),
            "p95": _safe_percentile(step, 95),
            "p99": _safe_percentile(step, 99),
            "max": _safe_percentile(step, 100),
        },
        "groups": groups,
        "worst_transitions": worst_transitions,
        "top_landmarks_by_jump_count": sorted(
            rows,
            key=lambda row: (
                -row[f"jumps_gt_{int(jump_threshold_mm)}mm"],
                -(row["max_step_mm"] or -1.0),
            ),
        )[:20],
    }
    if timestamps is not None and len(timestamps) > 1:
        timestamp_steps = np.diff(timestamps)
        report["timestamp_step_ms"] = {
            "median": float(np.nanmedian(timestamp_steps)),
            "min": float(np.nanmin(timestamp_steps)),
            "max": float(np.nanmax(timestamp_steps)),
        }

    yaml_path = output_dir / "jitter_report.yaml"
    csv_path = output_dir / "jitter_by_landmark.csv"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=False)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved jitter report: {yaml_path}")
    print(f"Saved landmark jitter table: {csv_path}")
    return yaml_path, csv_path


def main() -> None:
    args = parse_args()
    analyze_jitter(
        npz_path=args.npz_path,
        output_dir=args.output_dir,
        jump_threshold_mm=args.jump_threshold_mm,
        large_jump_threshold_mm=args.large_jump_threshold_mm,
    )


if __name__ == "__main__":
    main()
