from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from roc.config.yaml_io import load_calibration_config
from roc.io.sessions import CalibrationSessionPaths
from roc.calib.visualize import save_calibration_visualization
from aniposelib.boards import extract_points, extract_rtvecs, merge_rows, get_video_params  # type: ignore  # noqa: E402
from aniposelib.cameras import CameraGroup  # type: ignore  # noqa: E402
from aniposelib.boards import CharucoBoard as BaseCharucoBoard  # type: ignore  # noqa: E402
from aniposelib.cameras import get_initial_extrinsics  # type: ignore  # noqa: E402


APRILTAG_DICTS = {
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

SUPPORTED_CHARUCO_DICTS = {
    **APRILTAG_DICTS,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
}


@dataclass(slots=True)
class CalibrationSummary:
    mean_error: float
    per_camera_frames: dict[str, int]
    ground_plane_success: bool | None
    ground_plane_error: str | None
    selected_dictionary: str


@dataclass(slots=True)
class GroundPlaneResult:
    success: bool
    error: str | None = None


class AprilTagCharucoBoard(BaseCharucoBoard):
    def __init__(
        self,
        squares_x: int,
        squares_y: int,
        square_length: float,
        marker_length: float,
        dictionary_name: str,
    ) -> None:
        self.squaresX = squares_x
        self.squaresY = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        self.manually_verify = False

        dict_id = SUPPORTED_CHARUCO_DICTS[dictionary_name]
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.board = cv2.aruco.CharucoBoard(
            [squares_x, squares_y],
            square_length,
            marker_length,
            self.dictionary,
        )

        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        self.charuco_detector.setDetectorParameters(self.detector_params)

        self.total_size = (squares_x - 1) * (squares_y - 1)
        objp = np.zeros((self.total_size, 3), np.float64)
        objp[:, :2] = np.mgrid[0 : (squares_x - 1), 0 : (squares_y - 1)].T.reshape(-1, 2)
        objp *= square_length
        self.objPoints = objp
        self.empty_detection = np.zeros((self.total_size, 1, 2)) * np.nan

    def get_size(self):
        return (self.squaresX, self.squaresY)

    def get_square_length(self):
        return self.square_length

    def get_empty_detection(self):
        return np.copy(self.empty_detection)

    def draw(self, size):
        return self.board.generateImage(size)

    def fill_points(self, corners, ids):
        out = self.get_empty_detection()
        if corners is None or len(corners) == 0:
            return out
        ids = ids.ravel()
        for i, cxs in zip(ids, corners):
            out[i] = cxs
        return out

    def detect_image(self, image, camera=None):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        detected_corners, detected_ids, _, _ = self.charuco_detector.detectBoard(gray)
        if detected_corners is None or detected_ids is None or len(detected_corners) == 0:
            marker_corners, marker_ids, rejected = self.detector.detectMarkers(gray)
            if marker_ids is None or len(marker_ids) == 0:
                return np.float64([]), np.float64([])
            marker_corners, marker_ids, _, _ = cv2.aruco.refineDetectedMarkers(
                gray,
                self.board,
                marker_corners,
                marker_ids,
                rejected,
            )
            _, detected_corners, detected_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                self.board,
            )
            if detected_corners is None or detected_ids is None:
                return np.float64([]), np.float64([])
        return detected_corners, detected_ids

    def get_object_points(self):
        return self.objPoints

    def estimate_pose_points(self, camera, corners, ids):
        if corners is None or ids is None or len(corners) < 7:
            return None, None

        n_corners = corners.size // 2
        corners = np.reshape(corners, (n_corners, 1, 2))
        K = camera.get_camera_matrix()
        D = camera.get_distortions()
        obj_points = self.board.getChessboardCorners()

        detected_obj_points = []
        detected_img_points = []
        for i, corner_id in enumerate(ids.flatten()):
            if corner_id < len(obj_points):
                detected_obj_points.append(obj_points[corner_id])
                detected_img_points.append(corners[i].reshape(2))
        if len(detected_obj_points) < 7:
            return None, None

        ret, rvec, tvec = cv2.solvePnP(
            np.array(detected_obj_points, dtype=np.float32).reshape(-1, 3),
            np.array(detected_img_points, dtype=np.float32).reshape(-1, 2),
            K,
            D,
        )
        if not ret:
            return None, None
        return rvec, tvec


def _session_paths(session_dir: Path) -> CalibrationSessionPaths:
    return CalibrationSessionPaths(
        session_dir=session_dir,
        videos_dir=session_dir / "videos",
        raw_frames_dir=session_dir / "raw_frames",
        logs_dir=session_dir / "logs",
        capture_config_path=session_dir / "capture_config.yaml",
        calib_config_path=session_dir / "calib_config.yaml",
        calibration_toml_path=session_dir / "calibration.toml",
        calibration_yaml_path=session_dir / "calibration.yaml",
        charuco_2d_path=session_dir / "charuco_2d.npz",
        charuco_3d_path=session_dir / "charuco_3d.npy",
        calibration_report_path=session_dir / "calibration_report.yaml",
        calibration_visualization_path=session_dir / "calibration_visualization.png",
        charuco_overlays_dir=session_dir / "charuco_overlays",
    )


def _pin_camera_zero_to_origin(camera_group: CameraGroup) -> CameraGroup:
    rvecs = camera_group.get_rotations()
    tvecs = camera_group.get_translations()
    R0, _ = cv2.Rodrigues(rvecs[0])

    rvecs_new = np.empty_like(rvecs)
    for i in range(rvecs.shape[0]):
        Ri, _ = cv2.Rodrigues(rvecs[i])
        Ri_new, _ = cv2.Rodrigues(Ri @ R0.T)
        rvecs_new[i] = Ri_new.flatten()

    camera_group.set_rotations(rvecs_new)

    delta_to_origin_world = -R0.T @ tvecs[0, :]
    tvecs_new = np.zeros_like(tvecs)
    for cam_i in range(tvecs.shape[0]):
        Ri, _ = cv2.Rodrigues(rvecs[cam_i, :])
        delta_to_origin_camera_i = Ri @ delta_to_origin_world
        tvecs_new[cam_i, :] = tvecs[cam_i, :] + delta_to_origin_camera_i
    camera_group.set_translations(tvecs_new)
    return camera_group


def _set_world_positions(camera_group: CameraGroup) -> tuple[list[list[float]], list[list[list[float]]]]:
    rvecs = camera_group.get_rotations()
    tvecs = camera_group.get_translations()
    positions = []
    orientations = []
    for i in range(tvecs.shape[0]):
        rmat_world_to_cam_i, _ = cv2.Rodrigues(rvecs[i])
        rmat_cam_to_world = rmat_world_to_cam_i.T
        t_world = -rmat_cam_to_world @ tvecs[i]
        positions.append(t_world.astype(float).tolist())
        orientations.append(rmat_cam_to_world.astype(float).tolist())
    return positions, orientations


def _compute_basis_vectors_of_new_reference(charuco_frame: np.ndarray, squares_x: int, squares_y: int):
    num_cols = squares_x - 1
    num_rows = squares_y - 1
    idx_x = num_cols * (num_rows - 1)
    idx_y = num_cols - 1

    origin = charuco_frame[0]
    x_vec = charuco_frame[idx_x] - origin
    y_vec = charuco_frame[idx_y] - origin

    x_hat = x_vec / np.linalg.norm(x_vec)
    y_hat_raw = y_vec / np.linalg.norm(y_vec)
    z_hat = np.cross(x_hat, y_hat_raw)
    z_hat = z_hat / np.linalg.norm(z_hat)
    y_hat = np.cross(z_hat, x_hat)
    y_hat = y_hat / np.linalg.norm(y_hat)
    return x_hat, y_hat, z_hat


def _find_good_frame(charuco_3d: np.ndarray, squares_x: int, squares_y: int) -> int:
    num_cols = squares_x - 1
    num_rows = squares_y - 1
    idx_x = num_cols * (num_rows - 1)
    idx_y = num_cols - 1

    search = charuco_3d[: min(len(charuco_3d), 120)]
    candidate = search[:, [0, idx_y, idx_x]]
    visible = ~np.isnan(candidate).any(axis=(1, 2))
    if not np.any(visible):
        raise RuntimeError("No frame found where required Charuco corners are visible for ground-plane alignment")
    candidate = candidate[visible]
    if len(candidate) < 2:
        raise RuntimeError("Not enough visible Charuco frames for ground-plane alignment")
    velocity = np.linalg.norm(np.diff(candidate, axis=0), axis=2)
    max_velocity = np.nanmax(velocity, axis=1)
    best_visible_index = int(np.nanargmin(max_velocity))
    best_frame_index = np.where(visible)[0][best_visible_index + 1]
    return int(best_frame_index)


def _set_charuco_board_as_groundplane(
    camera_group: CameraGroup,
    charuco_2d: np.ndarray,
    squares_x: int,
    squares_y: int,
) -> tuple[CameraGroup, GroundPlaneResult]:
    num_cameras, num_frames, num_points, _ = charuco_2d.shape
    charuco_2d_flat = charuco_2d.reshape(num_cameras, -1, 2)
    charuco_3d_flat = camera_group.triangulate(charuco_2d_flat, fast=True)
    charuco_3d = charuco_3d_flat.reshape(num_frames, num_points, 3)

    try:
        best_frame = _find_good_frame(charuco_3d, squares_x, squares_y)
    except RuntimeError as exc:
        return camera_group, GroundPlaneResult(success=False, error=str(exc))

    charuco_frame = charuco_3d[best_frame]
    x_hat, y_hat, z_hat = _compute_basis_vectors_of_new_reference(charuco_frame, squares_x, squares_y)
    charuco_origin_in_world = charuco_frame[0]
    rmat_charuco_to_world = np.column_stack([x_hat, y_hat, z_hat])

    tvecs = camera_group.get_translations()
    rvecs = camera_group.get_rotations()
    tvecs_new = np.zeros_like(tvecs)
    rvecs_new = np.zeros_like(rvecs)
    for i in range(tvecs.shape[0]):
        rmat_world_to_cam_i, _ = cv2.Rodrigues(rvecs[i])
        t_delta = rmat_world_to_cam_i @ charuco_origin_in_world
        tvecs_new[i] = t_delta + tvecs[i]
        new_rmat = rmat_world_to_cam_i @ rmat_charuco_to_world
        new_rvec, _ = cv2.Rodrigues(new_rmat)
        rvecs_new[i] = new_rvec.flatten()

    camera_group.set_rotations(rvecs_new)
    camera_group.set_translations(tvecs_new)
    return camera_group, GroundPlaneResult(success=True)


def _build_video_writer(path: Path, width: int, height: int, fps: float) -> tuple[cv2.VideoWriter | None, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [
        (path, "mp4v"),
        (path, "avc1"),
        (path.with_suffix(".avi"), "MJPG"),
        (path.with_suffix(".avi"), "XVID"),
    ]
    for candidate_path, codec in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(candidate_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, candidate_path
    return None, path


def _detect_rows(
    video_path: Path,
    board: AprilTagCharucoBoard,
    overlay_output_path: Path | None = None,
) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open calibration video: {video_path}")

    rows: list[dict[str, Any]] = []
    frame_index = 0
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 3.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            corners, ids = board.detect_image(frame)
            if overlay_output_path is not None:
                overlay = frame.copy()
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(overlay, corners, ids)
                cv2.putText(
                    overlay,
                    f"frame={frame_index}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    f"charuco={0 if ids is None else len(ids)}",
                    (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if writer is None:
                    writer, actual_path = _build_video_writer(
                        overlay_output_path,
                        frame.shape[1],
                        frame.shape[0],
                        fps,
                    )
                    if writer is None:
                        print(f"[warn] failed to open overlay writer for {overlay_output_path}, continuing without overlay video")
                    elif actual_path != overlay_output_path:
                        print(f"[warn] overlay writer fallback in use: {actual_path.name}")
                if writer is not None:
                    writer.write(overlay)
            if corners is not None and len(corners) > 0:
                rows.append(
                    {
                        "framenum": (0, frame_index),
                        "corners": corners,
                        "ids": ids,
                    }
                )
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    rows = board.fill_points_rows(rows)
    return rows


def _sample_dictionary_score(video_paths: list[Path], dictionary_name: str, sample_frames: tuple[int, ...] = (0, 10, 20, 40, 60, 80, 100)) -> int:
    dictionary = cv2.aruco.getPredefinedDictionary(SUPPORTED_CHARUCO_DICTS[dictionary_name])
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    total_markers = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        frame_index = 0
        target_frames = set(sample_frames)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_index in target_frames:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    _, ids, _ = detector.detectMarkers(gray)
                    if ids is not None:
                        total_markers += len(ids)
                frame_index += 1
                if frame_index > max(sample_frames):
                    break
        finally:
            cap.release()
    return total_markers


def _sample_charuco_score(
    video_paths: list[Path],
    dictionary_name: str,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    sample_frames: tuple[int, ...] = (0, 10, 20, 40, 60, 80, 100),
) -> int:
    board = AprilTagCharucoBoard(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length_mm,
        marker_length=marker_length_mm,
        dictionary_name=dictionary_name,
    )
    total_corners = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        try:
            for frame_index in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue
                corners, ids = board.detect_image(frame)
                if ids is not None and len(ids) > 0:
                    total_corners += len(ids)
        finally:
            cap.release()
    return total_corners


def _select_dictionary(configured_dictionary: str, video_paths: list[Path]) -> tuple[str, dict[str, int]]:
    candidate_names = [configured_dictionary]
    for name in SUPPORTED_CHARUCO_DICTS:
        if name not in candidate_names:
            candidate_names.append(name)

    scores = {name: _sample_dictionary_score(video_paths, name) for name in candidate_names}
    best_name = max(scores, key=scores.get)
    return best_name, scores


def _select_board_orientation(
    video_paths: list[Path],
    dictionary_name: str,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
) -> tuple[tuple[int, int], dict[str, int]]:
    candidates = [
        (squares_x, squares_y),
        (squares_y, squares_x),
    ]
    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    scores = {}
    for sx, sy in unique_candidates:
        scores[f"{sx}x{sy}"] = _sample_charuco_score(
            video_paths,
            dictionary_name,
            sx,
            sy,
            square_length_mm,
            marker_length_mm,
        )
    best_key = max(scores, key=scores.get)
    best_sx, best_sy = [int(value) for value in best_key.split("x")]
    return (best_sx, best_sy), scores


def _charuco_2d_data_from_rows(all_rows: list[list[dict[str, Any]]], board: AprilTagCharucoBoard, num_frames: int) -> np.ndarray:
    num_cameras = len(all_rows)
    total_size = board.total_size
    data = np.full((num_cameras, num_frames, total_size, 2), np.nan, dtype=np.float64)
    for camera_index, rows in enumerate(all_rows):
        for row in rows:
            frame_index = row["framenum"][1]
            filled = row["filled"].reshape(total_size, 2)
            data[camera_index, frame_index] = filled
    return data


def _camera_group_to_yaml(camera_group: CameraGroup, camera_names: list[str], summary: CalibrationSummary) -> dict[str, Any]:
    output = {
        "schema_version": 1,
        "camera_order": camera_names,
        "quality": {
            "mean_reprojection_error_px": summary.mean_error,
            "per_camera_detected_frames": summary.per_camera_frames,
        },
        "world": {
            "mode": "ground" if summary.ground_plane_success is not None else "camera0",
            "ground_plane_success": summary.ground_plane_success,
            "ground_plane_error": summary.ground_plane_error,
        },
        "cameras": {},
    }

    dicts = camera_group.get_dicts()
    for name, cam_dict in zip(camera_names, dicts):
        output["cameras"][name] = {
            "size": cam_dict["size"],
            "matrix": cam_dict["matrix"],
            "distortion": cam_dict["distortions"],
            "rotation_vector": cam_dict["rotation"],
            "translation_vector": cam_dict["translation"],
        }
    return output


def run_calibration_solve(session_dir: Path) -> None:
    session_dir = session_dir.resolve()
    paths = _session_paths(session_dir)
    if not paths.calib_config_path.is_file():
        raise RuntimeError(f"Calibration config not found: {paths.calib_config_path}")
    if not paths.videos_dir.is_dir():
        raise RuntimeError(f"Calibration videos directory not found: {paths.videos_dir}")

    calib_config = load_calibration_config(paths.calib_config_path)
    video_paths = sorted(paths.videos_dir.glob("*.mp4"))
    if not video_paths:
        raise RuntimeError(f"No calibration videos found in {paths.videos_dir}")

    selected_dictionary, dictionary_scores = _select_dictionary(calib_config.charuco.dictionary, video_paths)
    print("Dictionary detection scores:", dictionary_scores)
    if dictionary_scores.get(selected_dictionary, 0) <= 0:
        raise RuntimeError(
            f"No supported ChArUco dictionary produced any marker detections. Tried: {dictionary_scores}"
        )
    if selected_dictionary != calib_config.charuco.dictionary:
        print(
            f"[warn] configured dictionary {calib_config.charuco.dictionary} produced no usable detections; "
            f"using autodetected dictionary {selected_dictionary}"
        )
        calib_config.charuco.dictionary = selected_dictionary
    (selected_squares_x, selected_squares_y), orientation_scores = _select_board_orientation(
        video_paths,
        selected_dictionary,
        calib_config.charuco.squares_x,
        calib_config.charuco.squares_y,
        calib_config.charuco.square_length_mm,
        calib_config.charuco.marker_length_mm,
    )
    print("Board orientation scores:", orientation_scores)
    if selected_squares_x != calib_config.charuco.squares_x or selected_squares_y != calib_config.charuco.squares_y:
        print(
            f"[warn] configured board orientation {calib_config.charuco.squares_x}x{calib_config.charuco.squares_y} "
            f"produced weak detections; using autodetected orientation {selected_squares_x}x{selected_squares_y}"
        )
        calib_config.charuco.squares_x = selected_squares_x
        calib_config.charuco.squares_y = selected_squares_y

    from roc.config.yaml_io import save_calibration_config

    save_calibration_config(paths.calib_config_path, calib_config)

    board = AprilTagCharucoBoard(
        squares_x=selected_squares_x,
        squares_y=selected_squares_y,
        square_length=calib_config.charuco.square_length_mm,
        marker_length=calib_config.charuco.marker_length_mm,
        dictionary_name=selected_dictionary,
    )

    camera_names = [path.stem for path in video_paths]
    all_rows = [
        _detect_rows(path, board, paths.charuco_overlays_dir / f"{path.stem}_charuco_overlay.mp4")
        for path in video_paths
    ]
    per_camera_frames = {name: len(rows) for name, rows in zip(camera_names, all_rows)}
    print("Charuco detection results:", per_camera_frames)

    camera_group = CameraGroup.from_names(camera_names)
    videos = [[str(path)] for path in video_paths]
    camera_group.set_camera_sizes_videos(videos)

    for rows, camera in zip(all_rows, camera_group.cameras):
        objp, imgp = board.get_all_calibration_points(rows)
        mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= 7]
        if not mixed:
            raise RuntimeError(f"No valid Charuco detections for camera {camera.get_name()}")
        objp, imgp = zip(*mixed)
        matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(camera.get_size()))
        camera.set_camera_matrix(matrix)

    for i, (rows, camera) in enumerate(zip(all_rows, camera_group.cameras)):
        all_rows[i] = board.estimate_pose_rows(camera, rows)

    merged = merge_rows(all_rows)
    imgp, extra = extract_points(merged, board, min_cameras=2)
    rtvecs = extract_rtvecs(merged)
    print("Merged frame count:", len(merged))

    rvecs, tvecs = get_initial_extrinsics(rtvecs)
    camera_group.set_rotations(rvecs)
    camera_group.set_translations(tvecs)
    error = camera_group.bundle_adjust_iter(imgp, extra, verbose=True, error_threshold=1)

    charuco_2d = _charuco_2d_data_from_rows(all_rows, board, max(get_video_params(str(path))["nframes"] for path in video_paths))
    np.savez_compressed(paths.charuco_2d_path, charuco_2d=charuco_2d, camera_names=np.array(camera_names, dtype=object))
    camera_group.charuco_2d_data = charuco_2d

    ground_success = None
    ground_error = None
    if calib_config.world_mode == "camera0":
        camera_group = _pin_camera_zero_to_origin(camera_group)
    else:
        camera_group, ground_result = _set_charuco_board_as_groundplane(
        camera_group,
        charuco_2d,
        selected_squares_x,
        selected_squares_y,
    )
        ground_success = ground_result.success
        ground_error = ground_result.error

    positions, orientations = _set_world_positions(camera_group)

    camera_group.dump(str(paths.calibration_toml_path))

    charuco_2d_flat = charuco_2d.reshape(len(camera_names), -1, 2)
    charuco_3d_flat = camera_group.triangulate(charuco_2d_flat, fast=True)
    charuco_3d = charuco_3d_flat.reshape(charuco_2d.shape[1], charuco_2d.shape[2], 3)
    np.save(paths.charuco_3d_path, charuco_3d)

    summary = CalibrationSummary(
        mean_error=float(error),
        per_camera_frames=per_camera_frames,
        ground_plane_success=ground_success,
        ground_plane_error=ground_error,
        selected_dictionary=selected_dictionary,
    )
    calibration_yaml = _camera_group_to_yaml(camera_group, camera_names, summary)
    for name, position, orientation in zip(camera_names, positions, orientations):
        calibration_yaml["cameras"][name]["position_world"] = position
        calibration_yaml["cameras"][name]["rotation_cam_to_world"] = orientation
    with paths.calibration_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(calibration_yaml, handle, sort_keys=False, allow_unicode=False)

    with paths.calibration_report_path.open("w", encoding="utf-8") as handle:
        report = {
            "mean_reprojection_error_px": float(error),
            "per_camera_detected_frames": per_camera_frames,
            "ground_plane_success": ground_success,
            "ground_plane_error": ground_error,
            "selected_dictionary": selected_dictionary,
            "dictionary_scores": dictionary_scores,
            "board_orientation_scores": orientation_scores,
        }
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=False)

    matrices = [calibration_yaml["cameras"][name]["matrix"] for name in camera_names]
    distortions = [calibration_yaml["cameras"][name]["distortion"] for name in camera_names]
    save_calibration_visualization(
        output_path=paths.calibration_visualization_path,
        camera_names=camera_names,
        positions=positions,
        orientations=orientations,
        matrices=matrices,
        distortions=distortions,
        charuco_3d=charuco_3d,
        summary=report,
    )

    print(f"Saved calibration solve outputs to: {session_dir}")
