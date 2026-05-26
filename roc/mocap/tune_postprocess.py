from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from roc.mocap.postprocess import postprocess_points_3d


POSE_BONES = [
    ("left_upper_arm", "left_shoulder", "left_elbow"),
    ("left_forearm", "left_elbow", "left_wrist"),
    ("right_upper_arm", "right_shoulder", "right_elbow"),
    ("right_forearm", "right_elbow", "right_wrist"),
    ("left_thigh", "left_hip", "left_knee"),
    ("left_shank", "left_knee", "left_ankle"),
    ("right_thigh", "right_hip", "right_knee"),
    ("right_shank", "right_knee", "right_ankle"),
    ("shoulder_width", "left_shoulder", "right_shoulder"),
    ("hip_width", "left_hip", "right_hip"),
]


@dataclass(frozen=True)
class PostprocessExperiment:
    name: str
    velocity_threshold_mm: float | None
    max_gap_frames: int
    butterworth_cutoff_hz: float
    butterworth_order: int = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune temporal postprocessing for mocap 3D points")
    parser.add_argument("--npz-path", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jump-threshold-mm", type=float, default=500.0)
    return parser.parse_args()


def _safe_stat(values: np.ndarray, fn) -> float | None:
    if not np.any(np.isfinite(values)):
        return None
    return float(fn(values))


def _step_metrics(points: np.ndarray, jump_threshold_mm: float) -> dict[str, float | int | None]:
    finite = np.isfinite(points).all(axis=2)
    step = np.linalg.norm(np.diff(points, axis=0), axis=2)
    step[~(finite[1:] & finite[:-1])] = np.nan
    return {
        "valid_ratio": float(finite.mean()),
        "valid_point_steps": int(np.isfinite(step).sum()),
        "median_step_mm": _safe_stat(step, np.nanmedian),
        "p95_step_mm": _safe_stat(step, lambda values: np.nanpercentile(values, 95)),
        "p99_step_mm": _safe_stat(step, lambda values: np.nanpercentile(values, 99)),
        "max_step_mm": _safe_stat(step, lambda values: np.nanpercentile(values, 100)),
        f"jumps_gt_{int(jump_threshold_mm)}mm": int(np.nansum(step > jump_threshold_mm)),
        "pose_p95_step_mm": _safe_stat(step[:, :33], lambda values: np.nanpercentile(values, 95)),
        "hands_p95_step_mm": _safe_stat(step[:, 33:], lambda values: np.nanpercentile(values, 95)),
    }


def _reprojection_drift_metrics(
    original_points: np.ndarray,
    processed_points: np.ndarray,
    reprojection_error: np.ndarray,
) -> dict[str, float | int | None]:
    drift = np.linalg.norm(processed_points - original_points, axis=2)
    drift[~(np.isfinite(processed_points).all(axis=2) & np.isfinite(original_points).all(axis=2))] = np.nan
    return {
        "median_3d_drift_from_raw_mm": _safe_stat(drift, np.nanmedian),
        "p95_3d_drift_from_raw_mm": _safe_stat(drift, lambda values: np.nanpercentile(values, 95)),
        "max_3d_drift_from_raw_mm": _safe_stat(drift, lambda values: np.nanpercentile(values, 100)),
        "median_raw_reprojection_error_px": _safe_stat(reprojection_error, np.nanmedian),
        "p95_raw_reprojection_error_px": _safe_stat(reprojection_error, lambda values: np.nanpercentile(values, 95)),
    }


def _bone_metrics(points: np.ndarray, landmark_names: list[str]) -> dict[str, float | int | None]:
    name_to_index = {name: index for index, name in enumerate(landmark_names)}
    bone_rows = []
    all_cv = []
    all_mad = []
    for bone_name, proximal_name, distal_name in POSE_BONES:
        if proximal_name not in name_to_index or distal_name not in name_to_index:
            continue
        proximal = points[:, name_to_index[proximal_name]]
        distal = points[:, name_to_index[distal_name]]
        length = np.linalg.norm(distal - proximal, axis=1)
        length[~(np.isfinite(proximal).all(axis=1) & np.isfinite(distal).all(axis=1))] = np.nan
        median = _safe_stat(length, np.nanmedian)
        if median is None or median <= 0:
            continue
        mad = _safe_stat(np.abs(length - median), np.nanmedian)
        p95_abs_dev = _safe_stat(np.abs(length - median), lambda values: np.nanpercentile(values, 95))
        cv = _safe_stat(length, np.nanstd)
        if cv is not None:
            cv = cv / median
            all_cv.append(cv)
        if mad is not None:
            all_mad.append(mad)
        bone_rows.append(
            {
                "bone": bone_name,
                "median_length_mm": median,
                "mad_length_mm": mad,
                "p95_abs_length_dev_mm": p95_abs_dev,
                "cv_length": cv,
                "valid_frames": int(np.isfinite(length).sum()),
            }
        )

    return {
        "bone_count": len(bone_rows),
        "median_bone_cv": _safe_stat(np.array(all_cv, dtype=float), np.nanmedian) if all_cv else None,
        "max_bone_cv": _safe_stat(np.array(all_cv, dtype=float), np.nanmax) if all_cv else None,
        "median_bone_mad_mm": _safe_stat(np.array(all_mad, dtype=float), np.nanmedian) if all_mad else None,
        "bone_rows": bone_rows,
    }


def _experiments(fps: float) -> list[PostprocessExperiment]:
    nyquist = max(fps, 1.0) / 2.0
    cutoffs = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0]
    cutoffs = [cutoff for cutoff in cutoffs if cutoff < nyquist]
    experiments = [
        PostprocessExperiment("raw_no_postprocess", None, 0, 0.0),
    ]
    for cutoff in cutoffs:
        experiments.append(PostprocessExperiment(f"freemocap_like_gap10_cutoff_{cutoff:g}hz", None, 10, cutoff))
    for velocity in [300.0, 500.0, 800.0]:
        for cutoff in [1.2, 2.0, 3.0]:
            if cutoff < nyquist:
                experiments.append(
                    PostprocessExperiment(
                        f"velocity_{int(velocity)}mm_gap10_cutoff_{cutoff:g}hz",
                        velocity,
                        10,
                        cutoff,
                    )
                )
    return experiments


def tune_postprocess(
    npz_path: Path,
    output_dir: Path | None = None,
    jump_threshold_mm: float = 500.0,
) -> tuple[Path, Path]:
    data = np.load(npz_path, allow_pickle=True)
    raw_points = data["points_3d_raw"].astype(np.float32)
    reprojection = data["reprojection_error"].astype(np.float32)
    landmark_names = [str(name) for name in data["landmark_names"]]
    timestamps = data["timestamps"].astype(float)
    if len(timestamps) > 1:
        fps = 1000.0 / float(np.nanmedian(np.diff(timestamps)))
    else:
        fps = 10.0

    output_dir = output_dir or npz_path.parent / "postprocess_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    detailed_bones = {}
    for experiment in _experiments(fps):
        print(f"Running postprocess tuning: {experiment.name}")
        if experiment.name == "raw_no_postprocess":
            processed = raw_points.copy()
            report = None
        else:
            velocity = float("inf") if experiment.velocity_threshold_mm is None else experiment.velocity_threshold_mm
            processed, report = postprocess_points_3d(
                raw_points,
                fps=fps,
                velocity_threshold_mm=velocity,
                max_gap_frames=experiment.max_gap_frames,
                butterworth_cutoff_hz=experiment.butterworth_cutoff_hz,
                butterworth_order=experiment.butterworth_order,
            )

        bone_metrics = _bone_metrics(processed, landmark_names)
        row = {
            "experiment": experiment.name,
            "velocity_threshold_mm": experiment.velocity_threshold_mm,
            "max_gap_frames": experiment.max_gap_frames,
            "butterworth_cutoff_hz": experiment.butterworth_cutoff_hz,
            "butterworth_order": experiment.butterworth_order,
            "velocity_outliers_removed": 0 if report is None else report.velocity_outliers_removed,
            "interpolated_values": 0 if report is None else report.interpolated_values,
            "butterworth_filtered": False if report is None else report.butterworth_filtered,
            **_step_metrics(processed, jump_threshold_mm),
            **_reprojection_drift_metrics(raw_points, processed, reprojection),
            "bone_count": bone_metrics["bone_count"],
            "median_bone_cv": bone_metrics["median_bone_cv"],
            "max_bone_cv": bone_metrics["max_bone_cv"],
            "median_bone_mad_mm": bone_metrics["median_bone_mad_mm"],
        }
        rows.append(row)
        detailed_bones[experiment.name] = bone_metrics["bone_rows"]
        np.savez_compressed(
            output_dir / f"{experiment.name}.npz",
            points_3d=processed.astype(np.float32),
            landmark_names=data["landmark_names"],
            camera_serials=data["camera_serials"],
            timestamps=data["timestamps"],
        )

    yaml_path = output_dir / "postprocess_tuning_report.yaml"
    csv_path = output_dir / "postprocess_tuning_summary.csv"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "npz_path": str(npz_path),
                "fps": fps,
                "jump_threshold_mm": jump_threshold_mm,
                "experiments": rows,
                "bone_details": detailed_bones,
            },
            handle,
            sort_keys=False,
            allow_unicode=False,
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved postprocess tuning report: {yaml_path}")
    print(f"Saved postprocess tuning summary: {csv_path}")
    return yaml_path, csv_path


def main() -> None:
    args = parse_args()
    tune_postprocess(
        npz_path=args.npz_path,
        output_dir=args.output_dir,
        jump_threshold_mm=args.jump_threshold_mm,
    )


if __name__ == "__main__":
    main()
