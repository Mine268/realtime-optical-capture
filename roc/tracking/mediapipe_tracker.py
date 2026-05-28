from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


POSE_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


@dataclass(slots=True)
class PoseTrackResult:
    xy: np.ndarray
    confidence: np.ndarray


@dataclass(slots=True)
class HandTrackResult:
    left_xy: np.ndarray
    left_confidence: np.ndarray
    right_xy: np.ndarray
    right_confidence: np.ndarray


class MediapipeTracker:
    def __init__(
        self,
        pose_model_path: Path,
        hand_model_path: Path | None,
        model_complexity: int,
        hands_enabled: bool,
        delegate: str = "cpu",
    ) -> None:
        base_options = _base_options(pose_model_path, delegate)
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionTaskRunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
        )
        self._hands_enabled = hands_enabled and hand_model_path is not None
        self._hands = None
        if self._hands_enabled:
            self._hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=_base_options(hand_model_path, delegate),
                    running_mode=VisionTaskRunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )

    def close(self) -> None:
        self._pose.close()
        if self._hands is not None:
            self._hands.close()

    def __enter__(self) -> "MediapipeTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _as_mp_image(frame_bgr: np.ndarray) -> mp.Image:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    def detect_pose(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseTrackResult:
        mp_image = self._as_mp_image(frame_bgr)
        result = self._pose.detect_for_video(mp_image, timestamp_ms=timestamp_ms)
        xy = np.full((len(POSE_LANDMARK_NAMES), 2), np.nan, dtype=np.float32)
        confidence = np.zeros((len(POSE_LANDMARK_NAMES),), dtype=np.float32)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            height, width = frame_bgr.shape[:2]
            for index, landmark in enumerate(landmarks):
                xy[index] = np.array([landmark.x * width, landmark.y * height], dtype=np.float32)
                confidence[index] = float(getattr(landmark, "visibility", 0.0))
        return PoseTrackResult(xy=xy, confidence=confidence)

    def detect_hands(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandTrackResult:
        left_xy = np.full((len(HAND_LANDMARK_NAMES), 2), np.nan, dtype=np.float32)
        left_conf = np.zeros((len(HAND_LANDMARK_NAMES),), dtype=np.float32)
        right_xy = np.full((len(HAND_LANDMARK_NAMES), 2), np.nan, dtype=np.float32)
        right_conf = np.zeros((len(HAND_LANDMARK_NAMES),), dtype=np.float32)
        if not self._hands_enabled or self._hands is None:
            return HandTrackResult(left_xy, left_conf, right_xy, right_conf)

        mp_image = self._as_mp_image(frame_bgr)
        result = self._hands.detect_for_video(mp_image, timestamp_ms=timestamp_ms)
        height, width = frame_bgr.shape[:2]
        for handedness, landmarks in zip(result.handedness, result.hand_landmarks):
            if not handedness:
                continue
            label = handedness[0].category_name.lower()
            if label not in {"left", "right"}:
                continue
            xy_target = left_xy if label == "left" else right_xy
            conf_target = left_conf if label == "left" else right_conf
            score = float(handedness[0].score)
            for index, landmark in enumerate(landmarks):
                xy_target[index] = np.array([landmark.x * width, landmark.y * height], dtype=np.float32)
                conf_target[index] = score
        return HandTrackResult(left_xy, left_conf, right_xy, right_conf)


def _base_options(model_path: Path, delegate: str) -> BaseOptions:
    normalized = delegate.lower()
    if normalized == "cpu":
        selected_delegate = BaseOptions.Delegate.CPU
    elif normalized == "gpu":
        selected_delegate = BaseOptions.Delegate.GPU
    else:
        raise ValueError(f"Unsupported MediaPipe delegate: {delegate}")
    return BaseOptions(model_asset_path=str(model_path), delegate=selected_delegate)
