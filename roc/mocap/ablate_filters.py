from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from roc.triangulation.cameras import load_camera_group_from_toml
from roc.triangulation.triangulate import triangulate_sequence


@dataclass(frozen=True)
class Experiment:
    name: str
    confidence_threshold: float
    minimum_cameras: int = 2
    robust_reprojection_threshold_px: float | None = None
    weighted_subset_reprojection_threshold_px: float | None = None
    maximum_cameras_to_drop: int = 1
    velocity_threshold_mm: float | None = None
    acceleration_threshold_mm: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation experiments for 3D mocap outlier filters")
    parser.add_argument("--npz-path", required=True, type=Path)
    parser.add_argument("--calibration-toml", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jump-threshold-mm", type=float, default=500.0)
    parser.add_argument("--large-jump-threshold-mm", type=float, default=1000.0)
    return parser.parse_args()


def _build_all_2d(data: np.lib.npyio.NpzFile, confidence_threshold: float) -> np.ndarray:
    pose_2d = data["pose_2d"].astype(np.float32)
    left_hand_2d = data["left_hand_2d"].astype(np.float32)
    right_hand_2d = data["right_hand_2d"].astype(np.float32)
    pose_conf = data["pose_confidence"].astype(np.float32)
    left_hand_conf = data["left_hand_confidence"].astype(np.float32)
    right_hand_conf = data["right_hand_confidence"].astype(np.float32)

    all_2d = np.concatenate([pose_2d, left_hand_2d, right_hand_2d], axis=2)
    all_conf = np.concatenate([pose_conf, left_hand_conf, right_hand_conf], axis=2)
    return np.where(all_conf[..., None] >= confidence_threshold, all_2d, np.nan)


def _robust_triangulate_sequence(
    camera_group,
    points_2d: np.ndarray,
    reprojection_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_cameras, num_frames, num_landmarks, _ = points_2d.shape
    points_3d = np.full((num_frames, num_landmarks, 3), np.nan, dtype=np.float32)
    reprojection = np.full((num_frames, num_landmarks), np.nan, dtype=np.float32)
    used_mask = np.zeros((num_frames, num_landmarks, num_cameras), dtype=bool)

    for frame_index in range(num_frames):
        for landmark_index in range(num_landmarks):
            candidate = points_2d[:, frame_index, landmark_index, :].copy()
            valid = np.isfinite(candidate).all(axis=1)
            if valid.sum() < 2:
                continue

            while valid.sum() >= 2:
                trial = candidate.copy()
                trial[~valid] = np.nan
                p3d = camera_group.triangulate(trial[:, None, :], fast=True)[0]
                if not np.isfinite(p3d).all():
                    break
                per_camera_error = camera_group.reprojection_error(p3d, trial, mean=False)
                per_camera_error_norm = np.linalg.norm(per_camera_error, axis=1)
                mean_error = float(np.nanmean(per_camera_error_norm))
                worst_camera = int(np.nanargmax(per_camera_error_norm))

                if mean_error <= reprojection_threshold_px or valid.sum() == 2:
                    points_3d[frame_index, landmark_index] = p3d
                    reprojection[frame_index, landmark_index] = mean_error
                    used_mask[frame_index, landmark_index] = valid
                    break

                valid[worst_camera] = False

    return points_3d, reprojection, used_mask


def _weighted_subset_triangulate_sequence(
    camera_group,
    points_2d: np.ndarray,
    target_reprojection_error_px: float,
    minimum_cameras: int = 3,
    maximum_cameras_to_drop: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_cameras, num_frames, num_landmarks, _ = points_2d.shape
    points_3d = np.full((num_frames, num_landmarks, 3), np.nan, dtype=np.float32)
    reprojection = np.full((num_frames, num_landmarks), np.nan, dtype=np.float32)
    used_mask = np.zeros((num_frames, num_landmarks, num_cameras), dtype=bool)

    for frame_index in range(num_frames):
        for landmark_index in range(num_landmarks):
            candidate = points_2d[:, frame_index, landmark_index, :].copy()
            valid = np.isfinite(candidate).all(axis=1)
            valid_indices = np.where(valid)[0]
            if len(valid_indices) < minimum_cameras:
                continue

            subset_results = []
            min_subset_size = max(minimum_cameras, len(valid_indices) - maximum_cameras_to_drop)
            for subset_size in range(len(valid_indices), min_subset_size - 1, -1):
                for subset in combinations(valid_indices, subset_size):
                    trial = np.full_like(candidate, np.nan)
                    trial[list(subset)] = candidate[list(subset)]
                    p3d = camera_group.triangulate(trial[:, None, :], fast=True)[0]
                    if not np.isfinite(p3d).all():
                        continue
                    error = float(camera_group.reprojection_error(p3d, trial, mean=True))
                    if not np.isfinite(error):
                        continue
                    weight = float(np.exp(-5.0 * error / max(target_reprojection_error_px, 1e-9)))
                    subset_results.append((p3d, error, weight, subset))

            if not subset_results:
                continue

            total_weight = sum(item[2] for item in subset_results)
            if total_weight <= 1e-12:
                best = min(subset_results, key=lambda item: item[1])
                points_3d[frame_index, landmark_index] = best[0]
                reprojection[frame_index, landmark_index] = best[1]
                used_mask[frame_index, landmark_index, list(best[3])] = True
                continue

            weighted_point = sum(item[0] * item[2] for item in subset_results) / total_weight
            normalized_camera_weights = np.zeros(num_cameras, dtype=np.float64)
            for _, _, weight, subset in subset_results:
                normalized_camera_weights[list(subset)] += weight / total_weight

            trial_all_valid = np.full_like(candidate, np.nan)
            trial_all_valid[valid] = candidate[valid]
            points_3d[frame_index, landmark_index] = weighted_point
            reprojection[frame_index, landmark_index] = camera_group.reprojection_error(
                weighted_point,
                trial_all_valid,
                mean=True,
            )
            used_mask[frame_index, landmark_index] = normalized_camera_weights > 0

    return points_3d, reprojection, used_mask


def _apply_temporal_filter(
    points_3d: np.ndarray,
    velocity_threshold_mm: float | None,
    acceleration_threshold_mm: float | None,
) -> np.ndarray:
    filtered = points_3d.copy()

    if velocity_threshold_mm is not None:
        step = np.linalg.norm(np.diff(filtered, axis=0), axis=2)
        jump_after = step > velocity_threshold_mm
        for frame_index, landmark_index in zip(*np.where(jump_after)):
            filtered[frame_index + 1, landmark_index] = np.nan

    if acceleration_threshold_mm is not None:
        finite = np.isfinite(filtered).all(axis=2)
        acceleration = np.linalg.norm(filtered[2:] - 2 * filtered[1:-1] + filtered[:-2], axis=2)
        acceleration[~(finite[2:] & finite[1:-1] & finite[:-2])] = np.nan
        bad_middle = acceleration > acceleration_threshold_mm
        for middle_minus_one, landmark_index in zip(*np.where(bad_middle)):
            filtered[middle_minus_one + 1, landmark_index] = np.nan

    return filtered


def _compute_metrics(
    points_3d: np.ndarray,
    reprojection: np.ndarray,
    jump_threshold_mm: float,
    large_jump_threshold_mm: float,
) -> dict[str, float | int | None]:
    finite = np.isfinite(points_3d).all(axis=2)
    step = np.linalg.norm(np.diff(points_3d, axis=0), axis=2)
    step[~(finite[1:] & finite[:-1])] = np.nan
    pose_step = step[:, :33]
    left_hand_step = step[:, 33:54]
    right_hand_step = step[:, 54:75]

    return {
        "valid_3d_points": int(finite.sum()),
        "valid_3d_ratio": float(finite.mean()),
        "valid_point_steps": int(np.isfinite(step).sum()),
        f"jumps_gt_{int(jump_threshold_mm)}mm": int(np.nansum(step > jump_threshold_mm)),
        f"jumps_gt_{int(large_jump_threshold_mm)}mm": int(np.nansum(step > large_jump_threshold_mm)),
        "median_step_mm": _nanmedian(step),
        "p95_step_mm": _nanpercentile(step, 95),
        "p99_step_mm": _nanpercentile(step, 99),
        "max_step_mm": _nanpercentile(step, 100),
        "pose_jumps_gt_threshold": int(np.nansum(pose_step > jump_threshold_mm)),
        "left_hand_jumps_gt_threshold": int(np.nansum(left_hand_step > jump_threshold_mm)),
        "right_hand_jumps_gt_threshold": int(np.nansum(right_hand_step > jump_threshold_mm)),
        "median_reprojection_error_px": _nanmedian(reprojection),
        "p95_reprojection_error_px": _nanpercentile(reprojection, 95),
    }


def _nanmedian(values: np.ndarray) -> float | None:
    if not np.any(np.isfinite(values)):
        return None
    return float(np.nanmedian(values))


def _nanpercentile(values: np.ndarray, percentile: float) -> float | None:
    if not np.any(np.isfinite(values)):
        return None
    return float(np.nanpercentile(values, percentile))


def run_ablation(
    npz_path: Path,
    calibration_toml: Path,
    output_dir: Path | None = None,
    jump_threshold_mm: float = 500.0,
    large_jump_threshold_mm: float = 1000.0,
) -> tuple[Path, Path]:
    output_dir = output_dir or npz_path.parent / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    camera_group = load_camera_group_from_toml(calibration_toml)

    experiments = [
        Experiment("baseline_conf_0p1", confidence_threshold=0.1),
        Experiment("confidence_0p5", confidence_threshold=0.5),
        Experiment("minimum_3_cameras_conf_0p1", confidence_threshold=0.1, minimum_cameras=3),
        Experiment("robust_reproj_50px", confidence_threshold=0.1, robust_reprojection_threshold_px=50.0),
        Experiment(
            "freemocap_weighted_min3_target50px",
            confidence_threshold=0.1,
            minimum_cameras=3,
            weighted_subset_reprojection_threshold_px=50.0,
            maximum_cameras_to_drop=1,
        ),
        Experiment("velocity_500mm", confidence_threshold=0.1, velocity_threshold_mm=500.0),
        Experiment(
            "conf_0p5_robust_50px",
            confidence_threshold=0.5,
            robust_reprojection_threshold_px=50.0,
        ),
        Experiment(
            "conf_0p5_freemocap_weighted_min3_target50px",
            confidence_threshold=0.5,
            minimum_cameras=3,
            weighted_subset_reprojection_threshold_px=50.0,
            maximum_cameras_to_drop=1,
        ),
        Experiment(
            "conf_0p5_freemocap_weighted_min3_target50px_velocity_500mm",
            confidence_threshold=0.5,
            minimum_cameras=3,
            weighted_subset_reprojection_threshold_px=50.0,
            maximum_cameras_to_drop=1,
            velocity_threshold_mm=500.0,
        ),
    ]

    rows = []
    for experiment in experiments:
        print(f"Running ablation: {experiment.name}")
        points_2d = _build_all_2d(data, experiment.confidence_threshold)
        valid_2d_points = np.isfinite(points_2d).all(axis=3)
        valid_counts = valid_2d_points.sum(axis=0)
        points_2d[:, valid_counts < experiment.minimum_cameras] = np.nan
        if experiment.weighted_subset_reprojection_threshold_px is not None:
            points_3d, reprojection, used_mask = _weighted_subset_triangulate_sequence(
                camera_group,
                points_2d,
                target_reprojection_error_px=experiment.weighted_subset_reprojection_threshold_px,
                minimum_cameras=experiment.minimum_cameras,
                maximum_cameras_to_drop=experiment.maximum_cameras_to_drop,
            )
        elif experiment.robust_reprojection_threshold_px is None:
            points_3d, reprojection = triangulate_sequence(camera_group, points_2d)
            used_mask = valid_2d_points.transpose(1, 2, 0)
        else:
            points_3d, reprojection, used_mask = _robust_triangulate_sequence(
                camera_group,
                points_2d,
                reprojection_threshold_px=experiment.robust_reprojection_threshold_px,
            )
        points_3d = _apply_temporal_filter(
            points_3d,
            velocity_threshold_mm=experiment.velocity_threshold_mm,
            acceleration_threshold_mm=experiment.acceleration_threshold_mm,
        )
        metrics = _compute_metrics(points_3d, reprojection, jump_threshold_mm, large_jump_threshold_mm)
        rows.append(
            {
                "experiment": experiment.name,
                "confidence_threshold": experiment.confidence_threshold,
                "minimum_cameras": experiment.minimum_cameras,
                "robust_reprojection_threshold_px": experiment.robust_reprojection_threshold_px,
                "weighted_subset_reprojection_threshold_px": experiment.weighted_subset_reprojection_threshold_px,
                "maximum_cameras_to_drop": experiment.maximum_cameras_to_drop,
                "velocity_threshold_mm": experiment.velocity_threshold_mm,
                "acceleration_threshold_mm": experiment.acceleration_threshold_mm,
                "valid_2d_observations": int(valid_2d_points.sum()),
                "used_2d_observations": int(used_mask.sum()),
                **metrics,
            }
        )

        np.savez_compressed(
            output_dir / f"{experiment.name}.npz",
            points_3d=points_3d.astype(np.float32),
            reprojection_error=reprojection.astype(np.float32),
            used_camera_mask=used_mask,
            landmark_names=data["landmark_names"],
            camera_serials=data["camera_serials"],
            timestamps=data["timestamps"],
        )

    report = {
        "npz_path": str(npz_path),
        "calibration_toml": str(calibration_toml),
        "jump_threshold_mm": jump_threshold_mm,
        "large_jump_threshold_mm": large_jump_threshold_mm,
        "experiments": rows,
    }
    yaml_path = output_dir / "ablation_report.yaml"
    csv_path = output_dir / "ablation_summary.csv"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=False)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved ablation report: {yaml_path}")
    print(f"Saved ablation summary: {csv_path}")
    return yaml_path, csv_path


def main() -> None:
    args = parse_args()
    run_ablation(
        npz_path=args.npz_path,
        calibration_toml=args.calibration_toml,
        output_dir=args.output_dir,
        jump_threshold_mm=args.jump_threshold_mm,
        large_jump_threshold_mm=args.large_jump_threshold_mm,
    )


if __name__ == "__main__":
    main()
