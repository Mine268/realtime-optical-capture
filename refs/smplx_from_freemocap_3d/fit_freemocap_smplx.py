from __future__ import annotations

import argparse
import json
import os
import pickle
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import smplx
import trimesh
import matplotlib.patheffects as pe
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from human_body_prior.tools.model_loader import load_vposer


def _profile_start(device: torch.device, enabled: bool) -> float:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _profile_end(device: torch.device, enabled: bool, start: float) -> float:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


BODY_SMPLX_MAP = {
    "hips_center": "pelvis",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
    "left_heel": "left_heel",
    "right_heel": "right_heel",
    "left_foot_index": "left_big_toe",
    "right_foot_index": "right_big_toe",
    "trunk_center": "spine2",
    "neck_center": "neck",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "nose": "nose",
    "left_eye": "left_eye",
    "right_eye": "right_eye",
    "left_ear": "left_ear",
    "right_ear": "right_ear",
    "head_center": "head",
}

LEFT_HAND_SMPLX_MAP = {
    "left_hand_wrist": "left_wrist",
    "left_hand_thumb_tip": "left_thumb",
    "left_hand_index_finger_tip": "left_index",
    "left_hand_middle_finger_tip": "left_middle",
    "left_hand_ring_finger_tip": "left_ring",
    "left_hand_pinky_tip": "left_pinky",
}

RIGHT_HAND_SMPLX_MAP = {
    "right_hand_wrist": "right_wrist",
    "right_hand_thumb_tip": "right_thumb",
    "right_hand_index_finger_tip": "right_index",
    "right_hand_middle_finger_tip": "right_middle",
    "right_hand_ring_finger_tip": "right_ring",
    "right_hand_pinky_tip": "right_pinky",
}

LEFT_HAND_SMPLX_MAP_FULL = {
    "left_hand_wrist": "left_wrist",
    "left_hand_thumb_cmc": "left_thumb1",
    "left_hand_thumb_mcp": "left_thumb2",
    "left_hand_thumb_ip": "left_thumb3",
    "left_hand_thumb_tip": "left_thumb",
    "left_hand_index_finger_mcp": "left_index1",
    "left_hand_index_finger_pip": "left_index2",
    "left_hand_index_finger_dip": "left_index3",
    "left_hand_index_finger_tip": "left_index",
    "left_hand_middle_finger_mcp": "left_middle1",
    "left_hand_middle_finger_pip": "left_middle2",
    "left_hand_middle_finger_dip": "left_middle3",
    "left_hand_middle_finger_tip": "left_middle",
    "left_hand_ring_finger_mcp": "left_ring1",
    "left_hand_ring_finger_pip": "left_ring2",
    "left_hand_ring_finger_dip": "left_ring3",
    "left_hand_ring_finger_tip": "left_ring",
    "left_hand_pinky_mcp": "left_pinky1",
    "left_hand_pinky_pip": "left_pinky2",
    "left_hand_pinky_dip": "left_pinky3",
    "left_hand_pinky_tip": "left_pinky",
}

RIGHT_HAND_SMPLX_MAP_FULL = {
    "right_hand_wrist": "right_wrist",
    "right_hand_thumb_cmc": "right_thumb1",
    "right_hand_thumb_mcp": "right_thumb2",
    "right_hand_thumb_ip": "right_thumb3",
    "right_hand_thumb_tip": "right_thumb",
    "right_hand_index_finger_mcp": "right_index1",
    "right_hand_index_finger_pip": "right_index2",
    "right_hand_index_finger_dip": "right_index3",
    "right_hand_index_finger_tip": "right_index",
    "right_hand_middle_finger_mcp": "right_middle1",
    "right_hand_middle_finger_pip": "right_middle2",
    "right_hand_middle_finger_dip": "right_middle3",
    "right_hand_middle_finger_tip": "right_middle",
    "right_hand_ring_finger_mcp": "right_ring1",
    "right_hand_ring_finger_pip": "right_ring2",
    "right_hand_ring_finger_dip": "right_ring3",
    "right_hand_ring_finger_tip": "right_ring",
    "right_hand_pinky_mcp": "right_pinky1",
    "right_hand_pinky_pip": "right_pinky2",
    "right_hand_pinky_dip": "right_pinky3",
    "right_hand_pinky_tip": "right_pinky",
}

LEFT_HAND_PROXIMAL_NAMES = [
    "left_hand_wrist",
    "left_hand_thumb_cmc",
    "left_hand_thumb_mcp",
    "left_hand_thumb_ip",
    "left_hand_index_finger_mcp",
    "left_hand_index_finger_pip",
    "left_hand_index_finger_dip",
    "left_hand_middle_finger_mcp",
    "left_hand_middle_finger_pip",
    "left_hand_middle_finger_dip",
    "left_hand_ring_finger_mcp",
    "left_hand_ring_finger_pip",
    "left_hand_ring_finger_dip",
    "left_hand_pinky_mcp",
    "left_hand_pinky_pip",
    "left_hand_pinky_dip",
]

RIGHT_HAND_PROXIMAL_NAMES = [
    "right_hand_wrist",
    "right_hand_thumb_cmc",
    "right_hand_thumb_mcp",
    "right_hand_thumb_ip",
    "right_hand_index_finger_mcp",
    "right_hand_index_finger_pip",
    "right_hand_index_finger_dip",
    "right_hand_middle_finger_mcp",
    "right_hand_middle_finger_pip",
    "right_hand_middle_finger_dip",
    "right_hand_ring_finger_mcp",
    "right_hand_ring_finger_pip",
    "right_hand_ring_finger_dip",
    "right_hand_pinky_mcp",
    "right_hand_pinky_pip",
    "right_hand_pinky_dip",
]

BODY_EDGES = [
    ("hips_center", "left_hip"),
    ("hips_center", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_heel"),
    ("left_heel", "left_foot_index"),
    ("right_heel", "right_foot_index"),
    ("hips_center", "trunk_center"),
    ("trunk_center", "neck_center"),
    ("neck_center", "left_shoulder"),
    ("neck_center", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("neck_center", "nose"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye_inner", "left_eye"),
    ("left_eye", "left_eye_outer"),
    ("right_eye_inner", "right_eye"),
    ("right_eye", "right_eye_outer"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("nose", "mouth_left"),
    ("nose", "mouth_right"),
    ("mouth_left", "mouth_right"),
    ("neck_center", "head_center"),
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

# Full SMPL-X skeleton edges using SMPL-X joint names (derived from model.parents)
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

BODY_LENGTH_EDGES = [
    ("hips_center", "left_hip"),
    ("hips_center", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_heel"),
    ("left_heel", "left_foot_index"),
    ("right_heel", "right_foot_index"),
    ("neck_center", "left_shoulder"),
    ("neck_center", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("hips_center", "trunk_center"),
    ("trunk_center", "neck_center"),
    ("neck_center", "head_center"),
]

BODY_SPARSE_FIT_NAMES = [
    "hips_center",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
    "neck_center",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
]

BODY_DIRECTION_EDGES = [
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
]

LOWER_BODY_DIRECTION_EDGES = [
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_heel"),
    ("left_heel", "left_foot_index"),
    ("right_heel", "right_foot_index"),
]

TORSO_DIRECTION_EDGES = [
    ("hips_center", "trunk_center"),
    ("trunk_center", "neck_center"),
    ("neck_center", "head_center"),
]

LEFT_FOOT_POINT_NAMES = ["left_ankle", "left_heel", "left_foot_index"]
RIGHT_FOOT_POINT_NAMES = ["right_ankle", "right_heel", "right_foot_index"]

LEFT_HAND_LENGTH_EDGES = LEFT_HAND_EDGES
RIGHT_HAND_LENGTH_EDGES = RIGHT_HAND_EDGES


@dataclass
class SequenceData:
    body: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    body_names: List[str]
    left_hand_names: List[str]
    right_hand_names: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit SMPL-X to FreeMoCap 3D landmarks")
    parser.add_argument("--recording-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--vposer-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--num-pca-comps", type=int, default=12)
    parser.add_argument("--betas-sample-count", type=int, default=32)
    parser.add_argument("--betas-steps", type=int, default=120)
    parser.add_argument("--pose-steps", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--betas-lr", type=float, default=0.05)
    parser.add_argument("--body-weight", type=float, default=5.0)
    parser.add_argument("--hand-weight", type=float, default=2.0)
    parser.add_argument("--foot-weight", type=float, default=8.0)
    parser.add_argument("--foot-orient-weight", type=float, default=6.0)
    parser.add_argument("--pose-prior-weight", type=float, default=0.02)
    parser.add_argument("--shape-prior-weight", type=float, default=0.001)
    parser.add_argument("--hand-prior-weight", type=float, default=0.01)
    parser.add_argument("--spine-prior-weight", type=float, default=0.0)
    parser.add_argument("--no-body-landmarks", action="store_true")
    parser.add_argument("--use-vposer", action="store_true")
    parser.add_argument("--no-mesh", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-eps", type=float, default=1e-5)
    parser.add_argument("--temporal-weight", type=float, default=0.0)
    parser.add_argument("--velocity-weight", type=float, default=0.0)
    parser.add_argument("--acceleration-weight", type=float, default=0.002)
    parser.add_argument("--disable-post-smooth", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=0)
    parser.add_argument("--smooth-sigma", type=float, default=0.0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--run-full-sequence", action="store_true")
    return parser.parse_args()


def load_sequence(recording_dir: Path) -> SequenceData:
    names = json.loads((recording_dir / "saved_data/info/trajectory_names.json").read_text())
    return SequenceData(
        body=np.load(recording_dir / "saved_data/npy/body_frame_name_xyz.npy"),
        left_hand=np.load(recording_dir / "saved_data/npy/left_hand_frame_name_xyz.npy"),
        right_hand=np.load(recording_dir / "saved_data/npy/right_hand_frame_name_xyz.npy"),
        body_names=names["body"],
        left_hand_names=names["left_hand"],
        right_hand_names=names["right_hand"],
    )


def build_name_to_index(names: List[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(names)}


def get_joint_name_to_index() -> Dict[str, int]:
    from smplx.joint_names import JOINT_NAMES

    return {name: idx for idx, name in enumerate(JOINT_NAMES)}


def gather_targets(frame_xyz: np.ndarray, name_to_idx: Dict[str, int], mapping: Dict[str, str]) -> Tuple[List[str], np.ndarray]:
    ordered_names = list(mapping.keys())
    points = np.stack([frame_xyz[name_to_idx[name]] for name in ordered_names], axis=0)
    return ordered_names, points.astype(np.float32)


def build_body_pose_from_vposer(vposer, embedding: torch.Tensor, device: torch.device) -> torch.Tensor:
    body_pose = vposer.decode(embedding, output_type="aa").view(1, -1)
    return body_pose.to(device=device)


def create_model(model_dir: Path, gender: str, num_betas: int, num_pca_comps: int, device: torch.device):
    model = smplx.create(
        str(model_dir),
        model_type="smplx",
        gender=gender,
        use_pca=True,
        num_pca_comps=num_pca_comps,
        num_betas=num_betas,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=1,
    )
    return model.to(device)


def frame_alignment_translation(frame_body: np.ndarray, body_idx: Dict[str, int]) -> np.ndarray:
    pelvis = frame_body[body_idx["hips_center"]]
    return pelvis.astype(np.float32)


def sample_frame_indices(frame_count: int, sample_count: int) -> np.ndarray:
    sample_count = min(frame_count, sample_count)
    if sample_count == frame_count:
        return np.arange(frame_count)
    return np.linspace(0, frame_count - 1, num=sample_count, dtype=int)


def build_target_sets(
    body_frame: np.ndarray,
    left_hand_frame: np.ndarray,
    right_hand_frame: np.ndarray,
    sequence: SequenceData,
) -> Dict[str, np.ndarray]:
    body_idx = build_name_to_index(sequence.body_names)
    left_idx = build_name_to_index(sequence.left_hand_names)
    right_idx = build_name_to_index(sequence.right_hand_names)
    return {
        "body_sparse": np.stack([body_frame[body_idx[name]] for name in BODY_SMPLX_MAP.keys()], axis=0).astype(np.float32),
        "body_sparse_fit": np.stack([body_frame[body_idx[name]] for name in BODY_SPARSE_FIT_NAMES], axis=0).astype(np.float32),
        "left_foot_points": np.stack([body_frame[body_idx[name]] for name in LEFT_FOOT_POINT_NAMES], axis=0).astype(np.float32),
        "right_foot_points": np.stack([body_frame[body_idx[name]] for name in RIGHT_FOOT_POINT_NAMES], axis=0).astype(np.float32),
        "left_hand_sparse": np.stack([left_hand_frame[left_idx[name]] for name in LEFT_HAND_SMPLX_MAP.keys()], axis=0).astype(np.float32),
        "right_hand_sparse": np.stack([right_hand_frame[right_idx[name]] for name in RIGHT_HAND_SMPLX_MAP.keys()], axis=0).astype(np.float32),
        "left_hand_full": np.stack([left_hand_frame[left_idx[name]] for name in LEFT_HAND_SMPLX_MAP_FULL.keys()], axis=0).astype(np.float32),
        "right_hand_full": np.stack([right_hand_frame[right_idx[name]] for name in RIGHT_HAND_SMPLX_MAP_FULL.keys()], axis=0).astype(np.float32),
        "left_hand_proximal": np.stack([left_hand_frame[left_idx[name]] for name in LEFT_HAND_PROXIMAL_NAMES], axis=0).astype(np.float32),
        "right_hand_proximal": np.stack([right_hand_frame[right_idx[name]] for name in RIGHT_HAND_PROXIMAL_NAMES], axis=0).astype(np.float32),
    }


def gather_smplx_predictions(joints: torch.Tensor, joint_name_to_idx: Dict[str, int]) -> Dict[str, torch.Tensor]:
    return {
        "body_sparse": torch.stack([joints[joint_name_to_idx[name]] for name in BODY_SMPLX_MAP.values()]),
        "body_sparse_fit": torch.stack(
            [joints[joint_name_to_idx[BODY_SMPLX_MAP[name]]] for name in BODY_SPARSE_FIT_NAMES]
        ),
        "left_foot_points": torch.stack([joints[joint_name_to_idx[BODY_SMPLX_MAP[name]]] for name in LEFT_FOOT_POINT_NAMES]),
        "right_foot_points": torch.stack([joints[joint_name_to_idx[BODY_SMPLX_MAP[name]]] for name in RIGHT_FOOT_POINT_NAMES]),
        "left_hand_sparse": torch.stack([joints[joint_name_to_idx[name]] for name in LEFT_HAND_SMPLX_MAP.values()]),
        "right_hand_sparse": torch.stack([joints[joint_name_to_idx[name]] for name in RIGHT_HAND_SMPLX_MAP.values()]),
        "left_hand_full": torch.stack([joints[joint_name_to_idx[name]] for name in LEFT_HAND_SMPLX_MAP_FULL.values()]),
        "right_hand_full": torch.stack([joints[joint_name_to_idx[name]] for name in RIGHT_HAND_SMPLX_MAP_FULL.values()]),
        "left_hand_proximal": torch.stack([joints[joint_name_to_idx[LEFT_HAND_SMPLX_MAP_FULL[name]]] for name in LEFT_HAND_PROXIMAL_NAMES]),
        "right_hand_proximal": torch.stack([joints[joint_name_to_idx[RIGHT_HAND_SMPLX_MAP_FULL[name]]] for name in RIGHT_HAND_PROXIMAL_NAMES]),
    }


def segment_lengths_from_points(points: Dict[str, np.ndarray], edges: List[Tuple[str, str]]) -> np.ndarray:
    return np.array([np.linalg.norm(points[a] - points[b]) for a, b in edges], dtype=np.float32)


def segment_lengths_from_tensor(points: Dict[str, torch.Tensor], edges: List[Tuple[str, str]]) -> torch.Tensor:
    return torch.stack([torch.norm(points[a] - points[b]) for a, b in edges])


def get_target_point_dicts(
    body_frame: np.ndarray,
    left_hand_frame: np.ndarray,
    right_hand_frame: np.ndarray,
    sequence: SequenceData,
) -> Dict[str, Dict[str, np.ndarray]]:
    body_idx = build_name_to_index(sequence.body_names)
    left_idx = build_name_to_index(sequence.left_hand_names)
    right_idx = build_name_to_index(sequence.right_hand_names)
    return {
        "body": {name: body_frame[body_idx[name]] for name in BODY_SMPLX_MAP.keys()},
        "left_hand": {name: left_hand_frame[left_idx[name]] for name in LEFT_HAND_SMPLX_MAP_FULL.keys()},
        "right_hand": {name: right_hand_frame[right_idx[name]] for name in RIGHT_HAND_SMPLX_MAP_FULL.keys()},
    }


def get_target_body_length_points(body_frame: np.ndarray, sequence: SequenceData) -> Dict[str, np.ndarray]:
    body_idx = build_name_to_index(sequence.body_names)
    names = {name for edge in BODY_LENGTH_EDGES for name in edge}
    return {name: body_frame[body_idx[name]] for name in names}


def get_smplx_point_dicts(joints: np.ndarray, joint_name_to_idx: Dict[str, int]) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        "body": {name: joints[joint_name_to_idx[target_name]] for name, target_name in BODY_SMPLX_MAP.items()},
        "left_hand": {name: joints[joint_name_to_idx[target_name]] for name, target_name in LEFT_HAND_SMPLX_MAP_FULL.items()},
        "right_hand": {name: joints[joint_name_to_idx[target_name]] for name, target_name in RIGHT_HAND_SMPLX_MAP_FULL.items()},
    }


def get_smplx_body_length_points(joints: torch.Tensor, joint_name_to_idx: Dict[str, int]) -> Dict[str, torch.Tensor]:
    name_map = {
        "hips_center": "pelvis",
        "left_hip": "left_hip",
        "right_hip": "right_hip",
        "left_knee": "left_knee",
        "right_knee": "right_knee",
        "left_ankle": "left_ankle",
        "right_ankle": "right_ankle",
        "left_heel": "left_heel",
        "right_heel": "right_heel",
        "left_foot_index": "left_big_toe",
        "right_foot_index": "right_big_toe",
        "trunk_center": "spine2",
        "neck_center": "neck",
        "left_shoulder": "left_shoulder",
        "right_shoulder": "right_shoulder",
        "left_elbow": "left_elbow",
        "right_elbow": "right_elbow",
        "left_wrist": "left_wrist",
        "right_wrist": "right_wrist",
        "head_center": "head",
    }
    return {name: joints[joint_name_to_idx[target_name]] for name, target_name in name_map.items()}


def body_bending_prior(body_pose: torch.Tensor) -> torch.Tensor:
    # Angle prior for elbows/knees, aligned with SMPL-X body pose axis-angle layout.
    idxs = torch.tensor([55, 58, 12, 15], dtype=torch.long, device=body_pose.device)
    signs = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=body_pose.dtype, device=body_pose.device)
    return torch.sum(torch.exp(body_pose[:, idxs] * signs).pow(2))


def alignment_rotation(src_a: np.ndarray, tgt_a: np.ndarray,
                       src_b: np.ndarray | None = None, tgt_b: np.ndarray | None = None) -> np.ndarray:
    """Compute axis-angle rotation that aligns src→tgt.

    If only one pair is given, aligns src_a to tgt_a (single direction).
    If two pairs are given, builds orthonormal frames and aligns both axes simultaneously.
    """
    if src_b is None or tgt_b is None:
        src_norm = np.linalg.norm(src_a)
        tgt_norm = np.linalg.norm(tgt_a)
        if src_norm < 1e-8 or tgt_norm < 1e-8:
            return np.zeros(3, dtype=np.float32)
        src = src_a / src_norm
        tgt = tgt_a / tgt_norm
        cross = np.cross(src, tgt)
        dot = np.dot(src, tgt)
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-8:
            return np.zeros(3, dtype=np.float32)
        angle = np.arctan2(cross_norm, dot)
        axis = cross / cross_norm
        return (axis * angle).astype(np.float32)

    # Two-direction alignment via Kabsch on orthonormal frames
    src_a_norm = np.linalg.norm(src_a)
    tgt_a_norm = np.linalg.norm(tgt_a)
    if src_a_norm < 1e-8 or tgt_a_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    a1 = src_a / src_a_norm
    b1 = src_b - np.dot(src_b, a1) * a1
    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    b1 = b1 / b1_norm
    c1 = np.cross(a1, b1)
    R_src = np.stack([a1, b1, c1], axis=1)  # 3×3

    a2 = tgt_a / tgt_a_norm
    b2 = tgt_b - np.dot(tgt_b, a2) * a2
    b2_norm = np.linalg.norm(b2)
    if b2_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    b2 = b2 / b2_norm
    c2 = np.cross(a2, b2)
    R_tgt = np.stack([a2, b2, c2], axis=1)

    R = R_tgt @ R_src.T
    # Convert to axis-angle
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz]) / (2.0 * np.sin(angle))
    return (axis * angle).astype(np.float32)


def blend_axis_angle(current_estimate: np.ndarray, previous_estimate: np.ndarray, current_weight: float = 0.7) -> np.ndarray:
    """Blend two small axis-angle rotations, biasing toward the current-frame geometric estimate."""
    prev_weight = 1.0 - current_weight
    return (current_weight * current_estimate + prev_weight * previous_estimate).astype(np.float32)


def spine_straightness_prior(body_pose: torch.Tensor) -> torch.Tensor:
    # Penalize axis-angle magnitudes of spine joints to keep the trunk straight.
    # spine1=[6:9], spine2=[15:18], spine3=[24:27] in the 21×3 body_pose layout.
    spine_idxs = torch.tensor([6, 7, 8, 15, 16, 17, 24, 25, 26], dtype=torch.long, device=body_pose.device)
    return torch.mean(body_pose[:, spine_idxs].pow(2))


def normalized(v: torch.Tensor) -> torch.Tensor:
    return v / torch.clamp(torch.norm(v), min=1e-6)


def axis_midpoint_width_loss(
    left_pred: torch.Tensor,
    right_pred: torch.Tensor,
    left_target: torch.Tensor,
    right_target: torch.Tensor,
    width_weight: float = 1.0,
) -> torch.Tensor:
    pred_axis = normalized(right_pred - left_pred)
    target_axis = normalized(right_target - left_target)
    pred_mid = 0.5 * (left_pred + right_pred)
    target_mid = 0.5 * (left_target + right_target)
    pred_width = torch.norm(right_pred - left_pred)
    target_width = torch.norm(right_target - left_target)
    return (
        F.mse_loss(pred_axis, target_axis)
        + F.mse_loss(pred_mid, target_mid)
        + width_weight * (pred_width - target_width).pow(2)
    )


def segment_direction_loss(
    pred_points: Dict[str, torch.Tensor],
    target_points: Dict[str, torch.Tensor],
    edges: List[Tuple[str, str]],
) -> torch.Tensor:
    losses = []
    for start_name, end_name in edges:
        pred_dir = normalized(pred_points[end_name] - pred_points[start_name])
        target_dir = normalized(target_points[end_name] - target_points[start_name])
        losses.append(F.mse_loss(pred_dir, target_dir))
    return torch.stack(losses).mean() if losses else torch.zeros((), dtype=torch.float32)


def hand_to_local_space(points: torch.Tensor, wrist_index: int = 0) -> torch.Tensor:
    wrist = points[wrist_index : wrist_index + 1]
    return points - wrist


def hand_to_palm_frame(
    points: torch.Tensor,
    wrist_index: int,
    index_mcp_index: int,
    pinky_mcp_index: int,
    middle_mcp_index: int,
) -> torch.Tensor:
    wrist = points[wrist_index]
    centered = points - wrist

    x_axis = centered[index_mcp_index] - centered[pinky_mcp_index]
    x_axis = normalized(x_axis)

    palm_forward = centered[middle_mcp_index]
    palm_forward = palm_forward - torch.dot(palm_forward, x_axis) * x_axis
    z_axis = normalized(palm_forward)

    y_axis = normalized(torch.cross(z_axis, x_axis, dim=0))
    z_axis = normalized(torch.cross(x_axis, y_axis, dim=0))

    rot = torch.stack([x_axis, y_axis, z_axis], dim=1)
    return centered @ rot


def scale_align_points(pred_points: torch.Tensor, target_points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    numerator = torch.sum(pred_points * target_points)
    denominator = torch.sum(pred_points * pred_points) + eps
    scale = numerator / denominator
    return pred_points * scale


def foot_orientation_loss(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    pred_ankle, pred_heel, pred_toe = pred_points[0], pred_points[1], pred_points[2]
    tgt_ankle, tgt_heel, tgt_toe = target_points[0], target_points[1], target_points[2]

    pred_forward = pred_toe - pred_heel
    tgt_forward = tgt_toe - tgt_heel
    pred_back = pred_heel - pred_ankle
    tgt_back = tgt_heel - tgt_ankle

    pred_forward = pred_forward / torch.clamp(torch.norm(pred_forward), min=1e-6)
    tgt_forward = tgt_forward / torch.clamp(torch.norm(tgt_forward), min=1e-6)
    pred_back = pred_back / torch.clamp(torch.norm(pred_back), min=1e-6)
    tgt_back = tgt_back / torch.clamp(torch.norm(tgt_back), min=1e-6)

    return F.mse_loss(pred_forward, tgt_forward) + F.mse_loss(pred_back, tgt_back)


def robust_length_target(length_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.median(length_matrix, axis=0).astype(np.float32)
    mad = np.median(np.abs(length_matrix - med[None, :]), axis=0).astype(np.float32)
    weights = 1.0 / np.maximum(mad, 1e-4)
    weights = weights / np.mean(weights)
    return med, weights.astype(np.float32)


def compute_length_targets(sequence: SequenceData) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]]:
    body_idx = build_name_to_index(sequence.body_names)
    left_idx = build_name_to_index(sequence.left_hand_names)
    right_idx = build_name_to_index(sequence.right_hand_names)

    body_lengths = []
    left_hand_lengths = []
    right_hand_lengths = []

    for body_frame, left_frame, right_frame in zip(sequence.body, sequence.left_hand, sequence.right_hand):
        body_points = {name: body_frame[body_idx[name]] for edge in BODY_LENGTH_EDGES for name in edge}
        left_points = {name: left_frame[left_idx[name]] for edge in LEFT_HAND_LENGTH_EDGES for name in edge}
        right_points = {name: right_frame[right_idx[name]] for edge in RIGHT_HAND_LENGTH_EDGES for name in edge}
        body_lengths.append(segment_lengths_from_points(body_points, BODY_LENGTH_EDGES))
        left_hand_lengths.append(segment_lengths_from_points(left_points, LEFT_HAND_LENGTH_EDGES))
        right_hand_lengths.append(segment_lengths_from_points(right_points, RIGHT_HAND_LENGTH_EDGES))

    body_target, body_weights = robust_length_target(np.stack(body_lengths, axis=0))
    left_target, left_weights = robust_length_target(np.stack(left_hand_lengths, axis=0))
    right_target, right_weights = robust_length_target(np.stack(right_hand_lengths, axis=0))
    return {
        "body": (body_target, body_weights, BODY_LENGTH_EDGES),
        "left_hand": (left_target, left_weights, LEFT_HAND_LENGTH_EDGES),
        "right_hand": (right_target, right_weights, RIGHT_HAND_LENGTH_EDGES),
    }


def compute_fit_errors(
    joints: np.ndarray,
    body_frame: np.ndarray,
    left_hand_frame: np.ndarray,
    right_hand_frame: np.ndarray,
    sequence: SequenceData,
    joint_name_to_idx: Dict[str, int],
) -> Dict[str, float]:
    body_idx = build_name_to_index(sequence.body_names)
    left_idx = build_name_to_index(sequence.left_hand_names)
    right_idx = build_name_to_index(sequence.right_hand_names)

    body_target = np.stack([body_frame[body_idx[name]] for name in BODY_SMPLX_MAP.keys()], axis=0)
    body_pred = np.stack([joints[joint_name_to_idx[name]] for name in BODY_SMPLX_MAP.values()], axis=0)
    left_target = np.stack([left_hand_frame[left_idx[name]] for name in LEFT_HAND_SMPLX_MAP.keys()], axis=0)
    left_pred = np.stack([joints[joint_name_to_idx[name]] for name in LEFT_HAND_SMPLX_MAP.values()], axis=0)
    right_target = np.stack([right_hand_frame[right_idx[name]] for name in RIGHT_HAND_SMPLX_MAP.keys()], axis=0)
    right_pred = np.stack([joints[joint_name_to_idx[name]] for name in RIGHT_HAND_SMPLX_MAP.values()], axis=0)

    body_err = np.linalg.norm(body_target - body_pred, axis=1)
    left_err = np.linalg.norm(left_target - left_pred, axis=1)
    right_err = np.linalg.norm(right_target - right_pred, axis=1)
    return {
        "body_mean_error_m": float(body_err.mean()),
        "body_max_error_m": float(body_err.max()),
        "left_hand_mean_error_m": float(left_err.mean()),
        "right_hand_mean_error_m": float(right_err.mean()),
        "overall_mean_error_m": float(np.concatenate([body_err, left_err, right_err]).mean()),
    }


def gaussian_kernel1d(window: int, sigma: float) -> np.ndarray:
    radius = window // 2
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def smooth_sequence_array(arr: np.ndarray, window: int, sigma: float) -> np.ndarray:
    if arr.shape[0] < 3 or window <= 1 or sigma <= 0:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    kernel = gaussian_kernel1d(window, sigma)
    radius = window // 2
    padded = np.pad(arr, [(radius, radius)] + [(0, 0)] * (arr.ndim - 1), mode="edge")
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        sl = padded[i : i + window]
        out[i] = np.tensordot(kernel, sl, axes=(0, 0))
    return out


def unwrap_axis_angle_sequence(rotvecs: np.ndarray) -> np.ndarray:
    if rotvecs.shape[0] < 2:
        return rotvecs.copy()
    out = rotvecs.copy()
    two_pi = 2.0 * np.pi
    for i in range(1, out.shape[0]):
        prev = out[i - 1]
        cur = out[i]
        prev_norm = np.linalg.norm(prev)
        cur_norm = np.linalg.norm(cur)
        if prev_norm < 1e-8 or cur_norm < 1e-8:
            continue
        axis = cur / cur_norm
        candidates = [cur, cur - axis * two_pi, cur + axis * two_pi]
        dists = [np.linalg.norm(c - prev) for c in candidates]
        out[i] = candidates[int(np.argmin(dists))]
    return out


def reproject_axis_angle_sequence(rotvecs: np.ndarray) -> np.ndarray:
    out = rotvecs.copy()
    for i in range(out.shape[0]):
        angle = np.linalg.norm(out[i])
        if angle < 1e-8:
            out[i] = np.zeros(3, dtype=np.float32)
            continue
        axis = out[i] / angle
        wrapped = ((angle + np.pi) % (2.0 * np.pi)) - np.pi
        out[i] = (axis * wrapped).astype(np.float32)
    return out.astype(np.float32)


def smooth_axis_angle_sequence(rotvecs: np.ndarray, window: int, sigma: float) -> np.ndarray:
    unwrapped = unwrap_axis_angle_sequence(rotvecs.astype(np.float32))
    smoothed = smooth_sequence_array(unwrapped, window, sigma)
    return reproject_axis_angle_sequence(smoothed)


def suppress_sequence_spikes(arr: np.ndarray, threshold_scale: float = 6.0) -> np.ndarray:
    if arr.shape[0] < 5:
        return arr.copy()
    out = arr.copy()
    vel = np.diff(out, axis=0)
    vel_norm = np.linalg.norm(vel.reshape(vel.shape[0], -1), axis=1)
    median = float(np.median(vel_norm))
    mad = float(np.median(np.abs(vel_norm - median)))
    threshold = median + threshold_scale * max(mad, 1e-6)
    spike_indices = np.where(vel_norm > threshold)[0] + 1
    for idx in spike_indices:
        if 1 <= idx < out.shape[0] - 1:
            out[idx] = 0.5 * (out[idx - 1] + out[idx + 1])
    return out


def estimate_default_smoothing_params(aggregate: List[Dict[str, np.ndarray]]) -> Tuple[int, float]:
    if len(aggregate) < 5:
        return 5, 1.0
    transl = np.stack([item["transl"] for item in aggregate], axis=0)
    body_pose = np.stack([item["body_pose"] for item in aggregate], axis=0)
    transl_acc = np.diff(transl, n=2, axis=0)
    body_acc = np.diff(body_pose, n=2, axis=0)
    transl_p90 = float(np.percentile(np.abs(transl_acc), 90)) if transl_acc.size else 0.0
    body_p90 = float(np.percentile(np.abs(body_acc), 90)) if body_acc.size else 0.0
    score = max(transl_p90 / 0.04, body_p90 / 0.07)
    if score < 0.75:
        return 5, 0.9
    if score < 1.5:
        return 7, 1.2
    return 9, 1.6


def save_mesh(result: Dict[str, np.ndarray], model, output_dir: Path, frame_index: int, device: torch.device) -> None:
    vertices = result.get("smplx_vertices")
    if vertices is None:
        output = model(
            betas=torch.tensor(result["betas"], dtype=torch.float32, device=device).unsqueeze(0),
            global_orient=torch.tensor(result["global_orient"], dtype=torch.float32, device=device).unsqueeze(0),
            body_pose=torch.tensor(result["body_pose"], dtype=torch.float32, device=device).unsqueeze(0),
            left_hand_pose=torch.tensor(result["left_hand_pose"], dtype=torch.float32, device=device).unsqueeze(0),
            right_hand_pose=torch.tensor(result["right_hand_pose"], dtype=torch.float32, device=device).unsqueeze(0),
            transl=torch.tensor(result["transl"], dtype=torch.float32, device=device).unsqueeze(0),
            expression=torch.zeros((1, model.num_expression_coeffs), dtype=torch.float32, device=device),
            jaw_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            leye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            return_verts=True,
            return_full_pose=False,
        )
        vertices = output.vertices[0].detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices, model.faces, process=False)
    mesh.export(output_dir / f"frame_{frame_index:06d}.obj")


def save_debug_plot(
    result: Dict[str, np.ndarray],
    sequence: SequenceData,
    joint_name_to_idx: Dict[str, int],
    model,
    device: torch.device,
    output_dir: Path,
    frame_index: int,
) -> None:
    body_idx = build_name_to_index(sequence.body_names)
    left_idx = build_name_to_index(sequence.left_hand_names)
    right_idx = build_name_to_index(sequence.right_hand_names)

    body_frame = sequence.body[frame_index]
    left_hand_frame = sequence.left_hand[frame_index]
    right_hand_frame = sequence.right_hand[frame_index]
    joints = result["smplx_joints"]

    body_target = np.stack([body_frame[body_idx[name]] for name in BODY_SMPLX_MAP.keys()], axis=0)
    body_pred = np.stack([joints[joint_name_to_idx[name]] for name in BODY_SMPLX_MAP.values()], axis=0)
    left_target = np.stack([left_hand_frame[left_idx[name]] for name in LEFT_HAND_SMPLX_MAP.keys()], axis=0)
    left_pred = np.stack([joints[joint_name_to_idx[name]] for name in LEFT_HAND_SMPLX_MAP.values()], axis=0)
    right_target = np.stack([right_hand_frame[right_idx[name]] for name in RIGHT_HAND_SMPLX_MAP.keys()], axis=0)
    right_pred = np.stack([joints[joint_name_to_idx[name]] for name in RIGHT_HAND_SMPLX_MAP.values()], axis=0)

    target_points = {
        **{name: body_frame[body_idx[name]] for name in sequence.body_names},
        **{name: left_hand_frame[left_idx[name]] for name in LEFT_HAND_SMPLX_MAP_FULL.keys()},
        **{name: right_hand_frame[right_idx[name]] for name in RIGHT_HAND_SMPLX_MAP_FULL.keys()},
    }
    # All body points for scatter (not just the BODY_SMPLX_MAP subset)
    body_all_names = [name for name in sequence.body_names if name in body_idx]
    body_all_pts = np.stack([body_frame[body_idx[name]] for name in body_all_names], axis=0)
    # All hand points for scatter (not just the sparse fingertip subset)
    left_all_pts = np.stack([left_hand_frame[left_idx[name]] for name in LEFT_HAND_SMPLX_MAP_FULL.keys()], axis=0)
    right_all_pts = np.stack([right_hand_frame[right_idx[name]] for name in RIGHT_HAND_SMPLX_MAP_FULL.keys()], axis=0)
    smplx_name_from_idx = {v: k for k, v in joint_name_to_idx.items()}
    fitted_points = {}
    num_joints = joints.shape[0]
    for j_idx in range(num_joints):
        j_name = smplx_name_from_idx.get(j_idx)
        if j_name is not None:
            fitted_points[j_name] = joints[j_idx]

    mesh_vertices = result.get("smplx_vertices")
    if mesh_vertices is None:
        mesh_output = model(
            betas=torch.tensor(result["betas"], dtype=torch.float32, device=device).unsqueeze(0),
            global_orient=torch.tensor(result["global_orient"], dtype=torch.float32, device=device).unsqueeze(0),
            body_pose=torch.tensor(result["body_pose"], dtype=torch.float32, device=device).unsqueeze(0),
            left_hand_pose=torch.tensor(result["left_hand_pose"], dtype=torch.float32, device=device).unsqueeze(0),
            right_hand_pose=torch.tensor(result["right_hand_pose"], dtype=torch.float32, device=device).unsqueeze(0),
            transl=torch.tensor(result["transl"], dtype=torch.float32, device=device).unsqueeze(0),
            expression=torch.zeros((1, model.num_expression_coeffs), dtype=torch.float32, device=device),
            jaw_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            leye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            return_verts=True,
            return_full_pose=False,
        )
        mesh_vertices = mesh_output.vertices[0].detach().cpu().numpy()
    mesh_faces = model.faces

    all_points = np.stack(list(target_points.values()) + list(fitted_points.values()), axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 0.25)

    def draw_edges(ax, point_dict, edges, color, alpha=1.0, linewidth=2.0):
        for start, end in edges:
            start_pt = point_dict[start]
            end_pt = point_dict[end]
            ax.plot(
                [start_pt[0], end_pt[0]],
                [start_pt[1], end_pt[1]],
                [start_pt[2], end_pt[2]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

    # Labels for FreeMoCap target points: use their trajectory indices
    target_labels = {
        **{name: str(body_idx[name]) for name in target_points.keys() if name in body_idx},
        **{name: f"L{left_idx[name]}" for name in LEFT_HAND_SMPLX_MAP_FULL.keys()},
        **{name: f"R{right_idx[name]}" for name in RIGHT_HAND_SMPLX_MAP_FULL.keys()},
    }

    # Labels for SMPL-X fitted points: show SMPL-X joint index
    smplx_labels = {
        name: str(joint_name_to_idx[name]) for name in fitted_points.keys()
    }

    def annotate_points(ax, point_dict, labels, color, fontsize=7):
        stroke = [pe.withStroke(linewidth=2.5, foreground="white")]
        for name, label in labels.items():
            if name not in point_dict:
                continue
            pt = point_dict[name]
            ax.text(
                pt[0],
                pt[1],
                pt[2],
                label,
                color=color,
                fontsize=fontsize,
                ha="center",
                va="center",
                path_effects=stroke,
            )

    def style_axis(ax, title, elev, azim):
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=elev, azim=azim)

    # Pre-split SMPL-X joints into body / left-hand / right-hand by known index ranges
    smplx_body_names = [name for name in fitted_points if joint_name_to_idx[name] <= 21]
    smplx_lhand_names = [name for name in fitted_points if 25 <= joint_name_to_idx[name] <= 39]
    smplx_rhand_names = [name for name in fitted_points if 40 <= joint_name_to_idx[name] <= 54]
    smplx_extra_names = [
        name for name in fitted_points
        if joint_name_to_idx[name] > 21 and name not in smplx_lhand_names and name not in smplx_rhand_names
    ]

    def smplx_points_array(names):
        return np.stack([fitted_points[name] for name in names], axis=0)

    def draw_panel(ax, mode: str, title: str, elev: float, azim: float, show_legend: bool = False):
        if mode in {"original", "overlay"}:
            draw_edges(ax, target_points, BODY_EDGES, "tab:blue", alpha=0.9)
            draw_edges(ax, target_points, LEFT_HAND_EDGES, "tab:green", alpha=0.9)
            draw_edges(ax, target_points, RIGHT_HAND_EDGES, "tab:red", alpha=0.9)
            # Scatter ALL body+hand points (not just the fitting subset)
            if body_all_pts.size:
                ax.scatter(body_all_pts[:, 0], body_all_pts[:, 1], body_all_pts[:, 2], c="tab:blue", s=18, label="original body")
            if left_all_pts.size:
                ax.scatter(left_all_pts[:, 0], left_all_pts[:, 1], left_all_pts[:, 2], c="tab:green", s=12, label="original left hand")
            if right_all_pts.size:
                ax.scatter(right_all_pts[:, 0], right_all_pts[:, 1], right_all_pts[:, 2], c="tab:red", s=12, label="original right hand")
            annotate_points(ax, target_points, target_labels, "black")
        if mode in {"fitted", "overlay"}:
            draw_edges(ax, fitted_points, SMPLX_BODY_EDGES, "tab:cyan", alpha=0.8)
            draw_edges(ax, fitted_points, SMPLX_LEFT_HAND_EDGES, "limegreen", alpha=0.8)
            draw_edges(ax, fitted_points, SMPLX_RIGHT_HAND_EDGES, "salmon", alpha=0.8)
            if smplx_body_names:
                pts = smplx_points_array(smplx_body_names)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="tab:cyan", s=18, label="fitted body")
            if smplx_lhand_names:
                pts = smplx_points_array(smplx_lhand_names)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="limegreen", s=12, label="fitted left hand")
            if smplx_rhand_names:
                pts = smplx_points_array(smplx_rhand_names)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="salmon", s=12, label="fitted right hand")
            if smplx_extra_names:
                pts = smplx_points_array(smplx_extra_names)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="gray", s=6, alpha=0.6, label="fitted other")
            annotate_points(ax, fitted_points, smplx_labels, "dimgray", fontsize=7)
        if mode == "mesh":
            trisample = mesh_faces[::4]
            tri_vertices = mesh_vertices[trisample]
            mesh_collection = Poly3DCollection(
                tri_vertices,
                facecolors="#b7dbe8",
                edgecolors=(0.35, 0.45, 0.5, 0.18),
                linewidths=0.08,
                alpha=0.78,
            )
            ax.add_collection3d(mesh_collection)
            # Draw full SMPL-X skeleton on mesh
            draw_edges(ax, fitted_points, SMPLX_BODY_EDGES, "tab:cyan", alpha=0.6, linewidth=1.5)
            draw_edges(ax, fitted_points, SMPLX_LEFT_HAND_EDGES, "limegreen", alpha=0.6, linewidth=1.5)
            draw_edges(ax, fitted_points, SMPLX_RIGHT_HAND_EDGES, "salmon", alpha=0.6, linewidth=1.5)
            if smplx_body_names:
                pts = smplx_points_array(smplx_body_names)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="tab:cyan", s=18)
            if smplx_lhand_names:
                pts = smplx_points_array(smplx_lhand_names)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="limegreen", s=12)
            if smplx_rhand_names:
                pts = smplx_points_array(smplx_rhand_names)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="salmon", s=12)
            if smplx_extra_names:
                pts = smplx_points_array(smplx_extra_names)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="gray", s=6, alpha=0.6)
            annotate_points(ax, fitted_points, smplx_labels, "dimgray", fontsize=7)
        style_axis(ax, title, elev, azim)
        if show_legend:
            ax.legend(loc="upper right", fontsize=7)

    viewpoints = [
        ("Front", 18, -72),
        ("Left", 18, 18),
        ("Right", 18, 162),
        ("Rear", 18, 108),
    ]

    fig = plt.figure(figsize=(24, 18))
    for row_idx, (view_name, elev, azim) in enumerate(viewpoints):
        ax_original = fig.add_subplot(len(viewpoints), 4, row_idx * 4 + 1, projection="3d")
        ax_fitted = fig.add_subplot(len(viewpoints), 4, row_idx * 4 + 2, projection="3d")
        ax_overlay = fig.add_subplot(len(viewpoints), 4, row_idx * 4 + 3, projection="3d")
        ax_mesh = fig.add_subplot(len(viewpoints), 4, row_idx * 4 + 4, projection="3d")
        draw_panel(ax_original, "original", f"{view_name} Original", elev, azim)
        draw_panel(ax_fitted, "fitted", f"{view_name} Fitted", elev, azim)
        draw_panel(ax_overlay, "overlay", f"{view_name} Overlay", elev, azim, show_legend=(row_idx == 0))
        draw_panel(ax_mesh, "mesh", f"{view_name} Mesh", elev, azim)

    fig.tight_layout()
    fig.savefig(output_dir / f"frame_{frame_index:06d}_fit.png", dpi=180)
    plt.close(fig)


def save_hand_debug_plot(
    result: Dict[str, np.ndarray],
    sequence: SequenceData,
    joint_name_to_idx: Dict[str, int],
    output_dir: Path,
    frame_index: int,
) -> None:
    left_idx = build_name_to_index(sequence.left_hand_names)
    right_idx = build_name_to_index(sequence.right_hand_names)

    left_hand_frame = sequence.left_hand[frame_index]
    right_hand_frame = sequence.right_hand[frame_index]
    joints = result["smplx_joints"]

    left_target = {name: left_hand_frame[left_idx[name]] for name in LEFT_HAND_SMPLX_MAP_FULL.keys()}
    right_target = {name: right_hand_frame[right_idx[name]] for name in RIGHT_HAND_SMPLX_MAP_FULL.keys()}
    left_fitted = {name: joints[joint_name_to_idx[target_name]] for name, target_name in LEFT_HAND_SMPLX_MAP_FULL.items()}
    right_fitted = {name: joints[joint_name_to_idx[target_name]] for name, target_name in RIGHT_HAND_SMPLX_MAP_FULL.items()}

    left_labels = {name: f"L{left_idx[name]}" for name in LEFT_HAND_SMPLX_MAP_FULL.keys()}
    right_labels = {name: f"R{right_idx[name]}" for name in RIGHT_HAND_SMPLX_MAP_FULL.keys()}

    def draw_edges(ax, point_dict, edges, color, alpha=1.0, linewidth=2.0):
        for start, end in edges:
            if start not in point_dict or end not in point_dict:
                continue
            start_pt = point_dict[start]
            end_pt = point_dict[end]
            ax.plot(
                [start_pt[0], end_pt[0]],
                [start_pt[1], end_pt[1]],
                [start_pt[2], end_pt[2]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

    def annotate_points(ax, point_dict, labels, color):
        stroke = [pe.withStroke(linewidth=2.5, foreground="white")]
        for name, label in labels.items():
            pt = point_dict[name]
            ax.text(pt[0], pt[1], pt[2], label, color=color, fontsize=7, ha="center", va="center", path_effects=stroke)

    def hand_center_radius(*point_dicts):
        pts = np.concatenate([np.stack(list(p.values()), axis=0) for p in point_dicts], axis=0)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins) / 2.0), 0.08)
        return center, radius * 1.15

    def style_axis(ax, title, center, radius):
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=20, azim=-65)

    def draw_hand_row(axs, side_name, target_dict, fitted_dict, labels, edge_defs, target_color, fitted_color):
        center, radius = hand_center_radius(target_dict, fitted_dict)
        target_pts = np.stack(list(target_dict.values()), axis=0)
        fitted_pts = np.stack(list(fitted_dict.values()), axis=0)

        draw_edges(axs[0], target_dict, edge_defs, target_color, 0.9, 2.0)
        axs[0].scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], c=target_color, s=16)
        annotate_points(axs[0], target_dict, labels, "black")
        style_axis(axs[0], f"{side_name} Original", center, radius)

        draw_edges(axs[1], fitted_dict, edge_defs, fitted_color, 0.9, 2.0)
        axs[1].scatter(fitted_pts[:, 0], fitted_pts[:, 1], fitted_pts[:, 2], c=fitted_color, s=16)
        annotate_points(axs[1], fitted_dict, labels, "dimgray")
        style_axis(axs[1], f"{side_name} Fitted", center, radius)

        draw_edges(axs[2], target_dict, edge_defs, target_color, 0.75, 2.0)
        draw_edges(axs[2], fitted_dict, edge_defs, fitted_color, 0.75, 2.0)
        axs[2].scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], c=target_color, s=14, label="original")
        axs[2].scatter(fitted_pts[:, 0], fitted_pts[:, 1], fitted_pts[:, 2], c=fitted_color, s=14, label="fitted")
        annotate_points(axs[2], target_dict, labels, "black")
        style_axis(axs[2], f"{side_name} Overlay", center, radius)
        axs[2].legend(loc="upper right", fontsize=7)

    fig = plt.figure(figsize=(16, 10))
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    draw_hand_row(axes[:3], "Left Hand", left_target, left_fitted, left_labels, LEFT_HAND_EDGES, "tab:green", "limegreen")
    draw_hand_row(axes[3:], "Right Hand", right_target, right_fitted, right_labels, RIGHT_HAND_EDGES, "tab:red", "salmon")
    fig.tight_layout()
    fig.savefig(output_dir / f"frame_{frame_index:06d}_hand_fit.png", dpi=180)
    plt.close(fig)


def optimize_shared_betas(
    model,
    vposer,
    sequence: SequenceData,
    args: argparse.Namespace,
    joint_name_to_idx: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    body_idx = build_name_to_index(sequence.body_names)
    length_targets = compute_length_targets(sequence)
    body_target_lengths, body_length_weights, body_length_edges = length_targets["body"]
    left_target_lengths, left_length_weights, left_length_edges = length_targets["left_hand"]
    right_target_lengths, right_length_weights, right_length_edges = length_targets["right_hand"]
    body_target_lengths_t = torch.tensor(body_target_lengths, dtype=torch.float32, device=device)
    left_target_lengths_t = torch.tensor(left_target_lengths, dtype=torch.float32, device=device)
    right_target_lengths_t = torch.tensor(right_target_lengths, dtype=torch.float32, device=device)
    body_length_weights_t = torch.tensor(body_length_weights, dtype=torch.float32, device=device)
    left_length_weights_t = torch.tensor(left_length_weights, dtype=torch.float32, device=device)
    right_length_weights_t = torch.tensor(right_length_weights, dtype=torch.float32, device=device)
    sample_indices = sample_frame_indices(sequence.body.shape[0], args.betas_sample_count)

    betas = torch.zeros((1, args.num_betas), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([betas], lr=args.betas_lr)
    best_loss = None
    stale_steps = 0

    for _ in range(args.betas_steps):
        optimizer.zero_grad()
        total_loss = torch.zeros((), dtype=torch.float32, device=device)

        for frame_index in sample_indices:
            body_frame = sequence.body[frame_index]

            global_orient = torch.zeros((1, 3), dtype=torch.float32, device=device)
            transl = torch.tensor(frame_alignment_translation(body_frame, body_idx), device=device).unsqueeze(0)
            if args.use_vposer:
                pose_embedding = torch.zeros((1, 32), dtype=torch.float32, device=device)
                body_pose = build_body_pose_from_vposer(vposer, pose_embedding, device)
            else:
                body_pose = torch.zeros((1, 63), dtype=torch.float32, device=device)
            left_hand_pose = torch.zeros((1, args.num_pca_comps), dtype=torch.float32, device=device)
            right_hand_pose = torch.zeros((1, args.num_pca_comps), dtype=torch.float32, device=device)

            output = model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                transl=transl,
                expression=torch.zeros((1, model.num_expression_coeffs), dtype=torch.float32, device=device),
                jaw_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
                leye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
                reye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
                return_verts=False,
                return_full_pose=False,
            )

            joints = output.joints[0]
            smplx_body_points = get_smplx_body_length_points(joints, joint_name_to_idx)
            smplx_left_hand_points = {
                name: joints[joint_name_to_idx[target_name]] for name, target_name in LEFT_HAND_SMPLX_MAP_FULL.items()
            }
            smplx_right_hand_points = {
                name: joints[joint_name_to_idx[target_name]] for name, target_name in RIGHT_HAND_SMPLX_MAP_FULL.items()
            }
            pred_body_lengths = segment_lengths_from_tensor(smplx_body_points, body_length_edges)
            pred_left_lengths = segment_lengths_from_tensor(smplx_left_hand_points, left_length_edges)
            pred_right_lengths = segment_lengths_from_tensor(smplx_right_hand_points, right_length_edges)
            total_loss = total_loss + 10.0 * torch.mean(body_length_weights_t * (pred_body_lengths - body_target_lengths_t) ** 2)
            total_loss = total_loss + 2.0 * torch.mean(left_length_weights_t * (pred_left_lengths - left_target_lengths_t) ** 2)
            total_loss = total_loss + 2.0 * torch.mean(right_length_weights_t * (pred_right_lengths - right_target_lengths_t) ** 2)

        total_loss = total_loss / len(sample_indices)
        total_loss = total_loss + args.shape_prior_weight * torch.mean(betas ** 2)
        total_loss.backward()
        optimizer.step()
        loss_value = float(total_loss.detach().cpu())
        if best_loss is None or loss_value < best_loss - args.early_stop_eps:
            best_loss = loss_value
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= args.early_stop_patience:
            break

    return betas.detach()


def fit_single_frame(
    model,
    vposer,
    frame_index: int,
    sequence: SequenceData,
    shared_betas: torch.Tensor,
    args: argparse.Namespace,
    joint_name_to_idx: Dict[str, int],
    device: torch.device,
    init_state: Dict[str, np.ndarray] | None = None,
    prev_state: Dict[str, np.ndarray] | None = None,
    prev_prev_state: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    profile_enabled = bool(getattr(args, "profile", False))
    stage_timings: Dict[str, float] = {}
    stage_start = _profile_start(device, profile_enabled)
    body_idx = build_name_to_index(sequence.body_names)
    left_hand_idx = build_name_to_index(sequence.left_hand_names)
    right_hand_idx = build_name_to_index(sequence.right_hand_names)

    body_frame = sequence.body[frame_index]
    lhand_frame = sequence.left_hand[frame_index]
    rhand_frame = sequence.right_hand[frame_index]

    target_sets = build_target_sets(body_frame, lhand_frame, rhand_frame, sequence)
    body_target_t = torch.tensor(target_sets["body_sparse_fit"], device=device)
    left_foot_t = torch.tensor(target_sets["left_foot_points"], device=device)
    right_foot_t = torch.tensor(target_sets["right_foot_points"], device=device)
    left_hand_sparse_t = torch.tensor(target_sets["left_hand_sparse"], device=device)
    right_hand_sparse_t = torch.tensor(target_sets["right_hand_sparse"], device=device)
    left_hand_full_t = torch.tensor(target_sets["left_hand_full"], device=device)
    right_hand_full_t = torch.tensor(target_sets["right_hand_full"], device=device)
    left_hand_proximal_t = torch.tensor(target_sets["left_hand_proximal"], device=device)
    right_hand_proximal_t = torch.tensor(target_sets["right_hand_proximal"], device=device)
    left_shoulder_target = torch.tensor(body_frame[body_idx["left_shoulder"]], dtype=torch.float32, device=device)
    right_shoulder_target = torch.tensor(body_frame[body_idx["right_shoulder"]], dtype=torch.float32, device=device)
    left_hip_target = torch.tensor(body_frame[body_idx["left_hip"]], dtype=torch.float32, device=device)
    right_hip_target = torch.tensor(body_frame[body_idx["right_hip"]], dtype=torch.float32, device=device)
    hips_target = torch.tensor(body_frame[body_idx["hips_center"]], dtype=torch.float32, device=device)
    trunk_target = torch.tensor(body_frame[body_idx["trunk_center"]], dtype=torch.float32, device=device)
    neck_target = torch.tensor(body_frame[body_idx["neck_center"]], dtype=torch.float32, device=device)
    head_target = torch.tensor(body_frame[body_idx["head_center"]], dtype=torch.float32, device=device)

    target_point_dicts = get_target_point_dicts(body_frame, lhand_frame, rhand_frame, sequence)
    target_body_length_points = get_target_body_length_points(body_frame, sequence)
    target_body_direction_points = {
        name: torch.tensor(point, dtype=torch.float32, device=device)
        for name, point in target_body_length_points.items()
    }
    target_left_hand_points = {
        name: torch.tensor(point, dtype=torch.float32, device=device)
        for name, point in target_point_dicts["left_hand"].items()
    }
    target_right_hand_points = {
        name: torch.tensor(point, dtype=torch.float32, device=device)
        for name, point in target_point_dicts["right_hand"].items()
    }
    body_length_targets = torch.tensor(
        segment_lengths_from_points(target_body_length_points, BODY_LENGTH_EDGES), device=device
    )
    left_hand_length_targets = segment_lengths_from_tensor(target_left_hand_points, LEFT_HAND_LENGTH_EDGES)
    right_hand_length_targets = segment_lengths_from_tensor(target_right_hand_points, RIGHT_HAND_LENGTH_EDGES)
    stage_timings["target_setup_s"] = _profile_end(device, profile_enabled, stage_start)
    stage_start = _profile_start(device, profile_enabled)

    # Forward pass at zero pose to get SMPL-X canonical spine direction
    transl_initial = torch.tensor(frame_alignment_translation(body_frame, body_idx), device=device).unsqueeze(0)
    with torch.no_grad():
        zero_output = model(
            betas=shared_betas,
            global_orient=torch.zeros((1, 3), dtype=torch.float32, device=device),
            body_pose=torch.zeros((1, 63), dtype=torch.float32, device=device),
            left_hand_pose=torch.zeros((1, args.num_pca_comps), dtype=torch.float32, device=device),
            right_hand_pose=torch.zeros((1, args.num_pca_comps), dtype=torch.float32, device=device),
            transl=transl_initial,
            expression=torch.zeros((1, model.num_expression_coeffs), dtype=torch.float32, device=device),
            jaw_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            leye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            return_verts=False,
            return_full_pose=False,
        )
        joints_zero = zero_output.joints[0].detach().cpu().numpy()

    # Compute initial global_orient aligning both spine (up) and left-right axes
    sx_spine = joints_zero[joint_name_to_idx["head"]] - joints_zero[joint_name_to_idx["pelvis"]]
    fm_spine = body_frame[body_idx["head_center"]] - body_frame[body_idx["hips_center"]]
    sx_left_right = joints_zero[joint_name_to_idx["right_hip"]] - joints_zero[joint_name_to_idx["left_hip"]]
    fm_left_right = body_frame[body_idx["right_hip"]] - body_frame[body_idx["left_hip"]]
    global_orient_np = alignment_rotation(sx_spine, fm_spine, sx_left_right, fm_left_right)
    if init_state is not None:
        global_orient_np = blend_axis_angle(global_orient_np, init_state["global_orient"].astype(np.float32))
        transl_np = init_state["transl"].astype(np.float32)
        left_hand_np = init_state["left_hand_pose"].astype(np.float32)
        right_hand_np = init_state["right_hand_pose"].astype(np.float32)
    else:
        transl_np = transl_initial.detach().cpu().numpy()[0].astype(np.float32)
        left_hand_np = np.zeros(args.num_pca_comps, dtype=np.float32)
        right_hand_np = np.zeros(args.num_pca_comps, dtype=np.float32)

    global_orient = torch.tensor(global_orient_np, dtype=torch.float32, device=device).unsqueeze(0)
    # Keep a small amount of torso freedom in spine1/spine2/neck.
    # Freeze only spine3 to avoid sharp mid-torso bending.
    # body_pose layout: spine1=[6:9], spine2=[15:18], spine3=[24:27], neck=[33:36]
    FROZEN_BP_IDX = [24, 25, 26]
    TORSO_BP_IDX = torch.tensor([6, 7, 8, 15, 16, 17, 33, 34, 35], dtype=torch.long, device=device)
    limb_mask = torch.ones(63, dtype=torch.bool, device=device)
    for i in FROZEN_BP_IDX:
        limb_mask[i] = False
    limb_idx_t = torch.where(limb_mask)[0]

    global_orient.requires_grad_(True)
    transl = torch.tensor(transl_np, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
    pose_embedding = None
    limb_pose_param = None
    if args.use_vposer:
        pose_embedding_np = init_state["pose_embedding"].astype(np.float32) if init_state and "pose_embedding" in init_state else np.zeros(32, dtype=np.float32)
        pose_embedding = torch.tensor(pose_embedding_np, dtype=torch.float32, device=device).unsqueeze(0)
        pose_embedding.requires_grad_(True)
    else:
        if init_state is not None:
            full_bp_init = init_state["body_pose"].astype(np.float32)
            limb_pose_init = full_bp_init[limb_idx_t.detach().cpu().numpy()]
        else:
            limb_pose_init = np.zeros(int(limb_mask.sum().item()), dtype=np.float32)
        limb_pose_param = torch.tensor(limb_pose_init, dtype=torch.float32, device=device, requires_grad=True)
    optimize_hands = args.hand_weight > 0.0 or args.hand_prior_weight > 0.0
    left_hand_pose = (
        torch.tensor(left_hand_np, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .requires_grad_(optimize_hands)
    )
    right_hand_pose = (
        torch.tensor(right_hand_np, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .requires_grad_(optimize_hands)
    )

    optim_params = [global_orient, transl]
    if optimize_hands:
        optim_params.extend([left_hand_pose, right_hand_pose])
    if limb_pose_param is not None:
        optim_params.append(limb_pose_param)
    if pose_embedding is not None:
        optim_params.append(pose_embedding)
    optimizer = torch.optim.Adam(optim_params, lr=args.lr)
    best_loss = None
    stale_steps = 0
    lower_body_bp_idx = torch.tensor(
        [0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 21, 22, 23],
        dtype=torch.long,
        device=device,
    )

    prev_global_orient_t = None
    prev_transl_t = None
    prev_body_pose_t = None
    prev_left_hand_t = None
    prev_right_hand_t = None
    prev_prev_global_orient_t = None
    prev_prev_transl_t = None
    prev_prev_body_pose_t = None
    if prev_state is not None:
        prev_global_orient_t = torch.tensor(prev_state["global_orient"], dtype=torch.float32, device=device).unsqueeze(0)
        prev_transl_t = torch.tensor(prev_state["transl"], dtype=torch.float32, device=device).unsqueeze(0)
        prev_body_pose_t = torch.tensor(prev_state["body_pose"], dtype=torch.float32, device=device).unsqueeze(0)
        prev_left_hand_t = torch.tensor(prev_state["left_hand_pose"], dtype=torch.float32, device=device).unsqueeze(0)
        prev_right_hand_t = torch.tensor(prev_state["right_hand_pose"], dtype=torch.float32, device=device).unsqueeze(0)
    if prev_prev_state is not None:
        prev_prev_global_orient_t = torch.tensor(prev_prev_state["global_orient"], dtype=torch.float32, device=device).unsqueeze(0)
        prev_prev_transl_t = torch.tensor(prev_prev_state["transl"], dtype=torch.float32, device=device).unsqueeze(0)
        prev_prev_body_pose_t = torch.tensor(prev_prev_state["body_pose"], dtype=torch.float32, device=device).unsqueeze(0)

    stage_timings["init_s"] = _profile_end(device, profile_enabled, stage_start)
    stage_start = _profile_start(device, profile_enabled)
    root_optimizer = torch.optim.Adam([global_orient, transl], lr=args.lr)
    root_steps = int(getattr(args, "root_steps", None) or max(12, args.pose_steps // 4))
    for _ in range(root_steps):
        root_optimizer.zero_grad()
        if pose_embedding is not None:
            body_pose = build_body_pose_from_vposer(vposer, pose_embedding.detach(), device)
        else:
            full_bp = torch.zeros(63, dtype=torch.float32, device=device)
            if limb_pose_param is not None:
                full_bp[limb_idx_t] = limb_pose_param.detach()
            body_pose = full_bp.unsqueeze(0)

        output = model(
            betas=shared_betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose.detach(),
            right_hand_pose=right_hand_pose.detach(),
            transl=transl,
            expression=torch.zeros((1, model.num_expression_coeffs), dtype=torch.float32, device=device),
            jaw_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            leye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            return_verts=False,
            return_full_pose=False,
        )

        joints = output.joints[0]
        pred_sets = gather_smplx_predictions(joints, joint_name_to_idx)
        left_shoulder_pred = joints[joint_name_to_idx["left_shoulder"]]
        right_shoulder_pred = joints[joint_name_to_idx["right_shoulder"]]
        left_hip_pred = joints[joint_name_to_idx["left_hip"]]
        right_hip_pred = joints[joint_name_to_idx["right_hip"]]
        pred_body_direction_points = get_smplx_body_length_points(joints, joint_name_to_idx)
        pelvis_pred = joints[joint_name_to_idx["pelvis"]]
        trunk_pred = joints[joint_name_to_idx["spine2"]]
        neck_pred = joints[joint_name_to_idx["neck"]]

        root_loss = torch.zeros((), dtype=torch.float32, device=device)
        root_loss = root_loss + 4.0 * axis_midpoint_width_loss(
            left_shoulder_pred, right_shoulder_pred, left_shoulder_target, right_shoulder_target, width_weight=0.5
        )
        root_loss = root_loss + 8.0 * axis_midpoint_width_loss(
            left_hip_pred, right_hip_pred, left_hip_target, right_hip_target, width_weight=0.5
        )
        root_loss = root_loss + args.foot_weight * 0.75 * F.mse_loss(pred_sets["left_foot_points"], left_foot_t)
        root_loss = root_loss + args.foot_weight * 0.75 * F.mse_loss(pred_sets["right_foot_points"], right_foot_t)
        root_loss = root_loss + args.foot_orient_weight * foot_orientation_loss(pred_sets["left_foot_points"], left_foot_t)
        root_loss = root_loss + args.foot_orient_weight * foot_orientation_loss(pred_sets["right_foot_points"], right_foot_t)
        root_loss = root_loss + 3.0 * segment_direction_loss(
            pred_body_direction_points, target_body_direction_points, BODY_DIRECTION_EDGES
        )
        root_loss = root_loss + 2.5 * segment_direction_loss(
            pred_body_direction_points, target_body_direction_points, TORSO_DIRECTION_EDGES
        )
        root_loss = root_loss + 1.5 * F.mse_loss(
            torch.stack([pelvis_pred, trunk_pred, neck_pred]),
            torch.stack([hips_target, trunk_target, neck_target]),
        )
        if prev_state is not None and args.temporal_weight > 0.0:
            root_loss = root_loss + args.temporal_weight * 0.5 * F.mse_loss(global_orient, prev_global_orient_t)
            root_loss = root_loss + args.temporal_weight * 0.5 * F.mse_loss(transl, prev_transl_t)
        if prev_state is not None and args.velocity_weight > 0.0:
            root_loss = root_loss + args.velocity_weight * torch.mean((transl - prev_transl_t).pow(2))
            root_loss = root_loss + args.velocity_weight * torch.mean((global_orient - prev_global_orient_t).pow(2))
        if (
            prev_state is not None
            and prev_prev_state is not None
            and args.acceleration_weight > 0.0
        ):
            root_loss = root_loss + args.acceleration_weight * torch.mean(
                (transl - 2.0 * prev_transl_t + prev_prev_transl_t).pow(2)
            )
            root_loss = root_loss + args.acceleration_weight * torch.mean(
                (global_orient - 2.0 * prev_global_orient_t + prev_prev_global_orient_t).pow(2)
            )
        root_loss.backward()
        root_optimizer.step()

    stage_timings["root_s"] = _profile_end(device, profile_enabled, stage_start)
    stage_start = _profile_start(device, profile_enabled)
    early_stop_check_interval = int(getattr(args, "early_stop_check_interval", 1) or 1)
    for step_index in range(args.pose_steps):
        optimizer.zero_grad()
        if pose_embedding is not None:
            body_pose = build_body_pose_from_vposer(vposer, pose_embedding, device)
        else:
            # Assemble full body_pose: zero spine + optimizable limbs
            full_bp = torch.zeros(63, dtype=torch.float32, device=device)
            full_bp[limb_idx_t] = limb_pose_param
            body_pose = full_bp.unsqueeze(0)

        output = model(
            betas=shared_betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            expression=torch.zeros((1, model.num_expression_coeffs), dtype=torch.float32, device=device),
            jaw_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            leye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
            return_verts=False,
            return_full_pose=True,
        )

        joints = output.joints[0]
        pred_sets = gather_smplx_predictions(joints, joint_name_to_idx)
        smplx_body_length_points = get_smplx_body_length_points(joints, joint_name_to_idx)
        body_length_pred = segment_lengths_from_tensor(smplx_body_length_points, BODY_LENGTH_EDGES)
        left_shoulder_pred = joints[joint_name_to_idx["left_shoulder"]]
        right_shoulder_pred = joints[joint_name_to_idx["right_shoulder"]]
        left_hip_pred = joints[joint_name_to_idx["left_hip"]]
        right_hip_pred = joints[joint_name_to_idx["right_hip"]]
        pelvis_pred = joints[joint_name_to_idx["pelvis"]]
        trunk_pred = joints[joint_name_to_idx["spine2"]]
        neck_pred = joints[joint_name_to_idx["neck"]]
        head_pred = joints[joint_name_to_idx["head"]]

        loss = torch.zeros((), dtype=torch.float32, device=device)
        if not args.no_body_landmarks:
            loss = loss + args.body_weight * F.mse_loss(pred_sets["body_sparse_fit"], body_target_t)
        loss = loss + 4.0 * axis_midpoint_width_loss(
            left_shoulder_pred, right_shoulder_pred, left_shoulder_target, right_shoulder_target, width_weight=0.5
        )
        loss = loss + 6.0 * axis_midpoint_width_loss(
            left_hip_pred, right_hip_pred, left_hip_target, right_hip_target, width_weight=0.5
        )
        loss = loss + args.foot_weight * F.mse_loss(pred_sets["left_foot_points"], left_foot_t)
        loss = loss + args.foot_weight * F.mse_loss(pred_sets["right_foot_points"], right_foot_t)
        loss = loss + args.foot_orient_weight * foot_orientation_loss(pred_sets["left_foot_points"], left_foot_t)
        loss = loss + args.foot_orient_weight * foot_orientation_loss(pred_sets["right_foot_points"], right_foot_t)
        loss = loss + 3.0 * F.mse_loss(body_length_pred, body_length_targets)
        loss = loss + 2.5 * segment_direction_loss(
            smplx_body_length_points, target_body_direction_points, BODY_DIRECTION_EDGES
        )
        loss = loss + 2.5 * segment_direction_loss(
            smplx_body_length_points, target_body_direction_points, TORSO_DIRECTION_EDGES
        )
        if optimize_hands:
            smplx_left_hand_points = {
                name: joints[joint_name_to_idx[target_name]] for name, target_name in LEFT_HAND_SMPLX_MAP_FULL.items()
            }
            smplx_right_hand_points = {
                name: joints[joint_name_to_idx[target_name]] for name, target_name in RIGHT_HAND_SMPLX_MAP_FULL.items()
            }
            left_hand_length_pred = segment_lengths_from_tensor(smplx_left_hand_points, LEFT_HAND_LENGTH_EDGES)
            right_hand_length_pred = segment_lengths_from_tensor(smplx_right_hand_points, RIGHT_HAND_LENGTH_EDGES)
            loss = loss + 0.6 * args.hand_weight * F.mse_loss(pred_sets["left_hand_sparse"][0:1], left_hand_sparse_t[0:1])
            loss = loss + 0.6 * args.hand_weight * F.mse_loss(pred_sets["right_hand_sparse"][0:1], right_hand_sparse_t[0:1])
            left_sparse_local_pred = hand_to_local_space(pred_sets["left_hand_sparse"])
            left_sparse_local_tgt = hand_to_local_space(left_hand_sparse_t)
            right_sparse_local_pred = hand_to_local_space(pred_sets["right_hand_sparse"])
            right_sparse_local_tgt = hand_to_local_space(right_hand_sparse_t)
            loss = loss + 0.35 * args.hand_weight * F.mse_loss(
                scale_align_points(left_sparse_local_pred, left_sparse_local_tgt), left_sparse_local_tgt
            )
            loss = loss + 0.35 * args.hand_weight * F.mse_loss(
                scale_align_points(right_sparse_local_pred, right_sparse_local_tgt), right_sparse_local_tgt
            )
            left_full_local_pred = hand_to_palm_frame(pred_sets["left_hand_full"], 0, 5, 17, 9)
            left_full_local_tgt = hand_to_palm_frame(left_hand_full_t, 0, 5, 17, 9)
            right_full_local_pred = hand_to_palm_frame(pred_sets["right_hand_full"], 0, 5, 17, 9)
            right_full_local_tgt = hand_to_palm_frame(right_hand_full_t, 0, 5, 17, 9)
            loss = loss + 1.0 * args.hand_weight * F.mse_loss(
                scale_align_points(left_full_local_pred, left_full_local_tgt),
                left_full_local_tgt,
            )
            loss = loss + 1.0 * args.hand_weight * F.mse_loss(
                scale_align_points(right_full_local_pred, right_full_local_tgt),
                right_full_local_tgt,
            )
            left_prox_local_pred = hand_to_palm_frame(pred_sets["left_hand_proximal"], 0, 4, 13, 7)
            left_prox_local_tgt = hand_to_palm_frame(left_hand_proximal_t, 0, 4, 13, 7)
            right_prox_local_pred = hand_to_palm_frame(pred_sets["right_hand_proximal"], 0, 4, 13, 7)
            right_prox_local_tgt = hand_to_palm_frame(right_hand_proximal_t, 0, 4, 13, 7)
            loss = loss + 0.35 * args.hand_weight * F.mse_loss(
                scale_align_points(left_prox_local_pred, left_prox_local_tgt),
                left_prox_local_tgt,
            )
            loss = loss + 0.35 * args.hand_weight * F.mse_loss(
                scale_align_points(right_prox_local_pred, right_prox_local_tgt),
                right_prox_local_tgt,
            )
            loss = loss + 0.4 * args.hand_weight * F.mse_loss(left_hand_length_pred, left_hand_length_targets)
            loss = loss + 0.4 * args.hand_weight * F.mse_loss(right_hand_length_pred, right_hand_length_targets)
            loss = loss + 0.5 * args.hand_weight * segment_direction_loss(
                smplx_left_hand_points, target_left_hand_points, LEFT_HAND_EDGES
            )
            loss = loss + 0.5 * args.hand_weight * segment_direction_loss(
                smplx_right_hand_points, target_right_hand_points, RIGHT_HAND_EDGES
            )
        loss = loss + 1.0 * F.mse_loss(
            torch.stack([pelvis_pred, trunk_pred, neck_pred, head_pred]),
            torch.stack([hips_target, trunk_target, neck_target, head_target]),
        )
        if pose_embedding is not None:
            loss = loss + args.pose_prior_weight * torch.mean(pose_embedding ** 2)
        if limb_pose_param is not None:
            loss = loss + args.pose_prior_weight * torch.mean(limb_pose_param ** 2)
            loss = loss + 0.03 * torch.mean(body_pose[:, TORSO_BP_IDX] ** 2)
        loss = loss + args.hand_prior_weight * (torch.mean(left_hand_pose ** 2) + torch.mean(right_hand_pose ** 2))
        loss = loss + 0.001 * body_bending_prior(body_pose)
        loss = loss + args.shape_prior_weight * torch.mean(shared_betas ** 2)
        loss = loss + max(args.spine_prior_weight, 0.02) * spine_straightness_prior(body_pose)
        if prev_state is not None and args.temporal_weight > 0.0:
            loss = loss + args.temporal_weight * F.mse_loss(global_orient, prev_global_orient_t)
            loss = loss + args.temporal_weight * F.mse_loss(transl, prev_transl_t)
            if optimize_hands:
                loss = loss + args.temporal_weight * F.mse_loss(left_hand_pose, prev_left_hand_t)
                loss = loss + args.temporal_weight * F.mse_loss(right_hand_pose, prev_right_hand_t)
            if limb_pose_param is not None and prev_body_pose_t is not None:
                loss = loss + args.temporal_weight * F.mse_loss(body_pose, prev_body_pose_t)
        if prev_state is not None and args.velocity_weight > 0.0:
            loss = loss + args.velocity_weight * torch.mean((transl - prev_transl_t).pow(2))
            loss = loss + args.velocity_weight * torch.mean((global_orient - prev_global_orient_t).pow(2))
        if (
            prev_state is not None
            and prev_prev_state is not None
            and args.acceleration_weight > 0.0
        ):
            loss = loss + args.acceleration_weight * torch.mean(
                (transl - 2.0 * prev_transl_t + prev_prev_transl_t).pow(2)
            )
            loss = loss + args.acceleration_weight * torch.mean(
                (global_orient - 2.0 * prev_global_orient_t + prev_prev_global_orient_t).pow(2)
            )
            if prev_body_pose_t is not None and prev_prev_body_pose_t is not None:
                loss = loss + args.acceleration_weight * torch.mean(
                    (body_pose - 2.0 * prev_body_pose_t + prev_prev_body_pose_t).pow(2)
                )
        loss.backward()
        optimizer.step()
        should_check_stop = (
            early_stop_check_interval <= 1
            or (step_index + 1) % early_stop_check_interval == 0
            or step_index + 1 == args.pose_steps
        )
        if should_check_stop:
            loss_value = float(loss.detach().cpu())
            if best_loss is None or loss_value < best_loss - args.early_stop_eps:
                best_loss = loss_value
                stale_steps = 0
            else:
                stale_steps += early_stop_check_interval
            if stale_steps >= args.early_stop_patience:
                break

    stage_timings["pose_s"] = _profile_end(device, profile_enabled, stage_start)
    stage_start = _profile_start(device, profile_enabled)
    if pose_embedding is None and limb_pose_param is not None and getattr(args, "lower_body_refine", True):
        lower_root_optimizer = torch.optim.Adam([global_orient, transl, limb_pose_param], lr=args.lr * 0.35)
        lower_steps = int(getattr(args, "lower_steps", None) or max(6, args.pose_steps // 6))
        for _ in range(lower_steps):
            lower_root_optimizer.zero_grad()
            full_bp = torch.zeros(63, dtype=torch.float32, device=device)
            full_bp[limb_idx_t] = limb_pose_param
            body_pose = full_bp.unsqueeze(0)

            output = model(
                betas=shared_betas,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose.detach(),
                right_hand_pose=right_hand_pose.detach(),
                transl=transl,
                expression=torch.zeros((1, model.num_expression_coeffs), dtype=torch.float32, device=device),
                jaw_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
                leye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
                reye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
                return_verts=False,
                return_full_pose=False,
            )

            joints = output.joints[0]
            pred_sets = gather_smplx_predictions(joints, joint_name_to_idx)
            smplx_body_length_points = get_smplx_body_length_points(joints, joint_name_to_idx)
            lower_loss = torch.zeros((), dtype=torch.float32, device=device)
            lower_loss = lower_loss + 5.0 * axis_midpoint_width_loss(
                joints[joint_name_to_idx["left_hip"]],
                joints[joint_name_to_idx["right_hip"]],
                left_hip_target,
                right_hip_target,
                width_weight=0.5,
            )
            lower_loss = lower_loss + 4.0 * segment_direction_loss(
                smplx_body_length_points, target_body_direction_points, LOWER_BODY_DIRECTION_EDGES
            )
            lower_loss = lower_loss + 1.0 * segment_direction_loss(
                smplx_body_length_points, target_body_direction_points, BODY_DIRECTION_EDGES
            )
            lower_loss = lower_loss + 0.5 * args.foot_weight * F.mse_loss(pred_sets["left_foot_points"], left_foot_t)
            lower_loss = lower_loss + 0.5 * args.foot_weight * F.mse_loss(pred_sets["right_foot_points"], right_foot_t)
            lower_loss = lower_loss + 0.5 * args.foot_orient_weight * foot_orientation_loss(pred_sets["left_foot_points"], left_foot_t)
            lower_loss = lower_loss + 0.5 * args.foot_orient_weight * foot_orientation_loss(pred_sets["right_foot_points"], right_foot_t)
            lower_loss = lower_loss + 0.05 * args.body_weight * F.mse_loss(pred_sets["body_sparse_fit"], body_target_t)
            lower_loss = lower_loss + 0.03 * torch.mean(body_pose[:, lower_body_bp_idx] ** 2)
            if prev_state is not None and args.temporal_weight > 0.0:
                lower_loss = lower_loss + args.temporal_weight * 0.5 * F.mse_loss(global_orient, prev_global_orient_t)
                lower_loss = lower_loss + args.temporal_weight * 0.5 * F.mse_loss(transl, prev_transl_t)
                if prev_body_pose_t is not None:
                    lower_loss = lower_loss + args.temporal_weight * 0.25 * F.mse_loss(
                        body_pose[:, lower_body_bp_idx], prev_body_pose_t[:, lower_body_bp_idx]
                    )
            if (
                prev_state is not None
                and prev_prev_state is not None
                and args.acceleration_weight > 0.0
            ):
                lower_loss = lower_loss + args.acceleration_weight * torch.mean(
                    (transl - 2.0 * prev_transl_t + prev_prev_transl_t).pow(2)
                )
                lower_loss = lower_loss + args.acceleration_weight * torch.mean(
                    (global_orient - 2.0 * prev_global_orient_t + prev_prev_global_orient_t).pow(2)
                )
                if prev_body_pose_t is not None and prev_prev_body_pose_t is not None:
                    lower_loss = lower_loss + args.acceleration_weight * 0.5 * torch.mean(
                        (
                            body_pose[:, lower_body_bp_idx]
                            - 2.0 * prev_body_pose_t[:, lower_body_bp_idx]
                            + prev_prev_body_pose_t[:, lower_body_bp_idx]
                        ).pow(2)
                    )
            lower_loss.backward()
            lower_root_optimizer.step()

    stage_timings["lower_s"] = _profile_end(device, profile_enabled, stage_start)
    stage_start = _profile_start(device, profile_enabled)
    if pose_embedding is not None:
        final_body_pose = build_body_pose_from_vposer(vposer, pose_embedding.detach(), device)
    else:
        final_full_bp = torch.zeros(63, dtype=torch.float32, device=device)
        final_full_bp[limb_idx_t] = limb_pose_param.detach()
        final_body_pose = final_full_bp.unsqueeze(0)
    output = model(
        betas=shared_betas,
        global_orient=global_orient.detach(),
        body_pose=final_body_pose,
        left_hand_pose=left_hand_pose.detach(),
        right_hand_pose=right_hand_pose.detach(),
        transl=transl.detach(),
        expression=torch.zeros((1, model.num_expression_coeffs), dtype=torch.float32, device=device),
        jaw_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
        leye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
        reye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
        return_verts=not args.no_mesh or not args.no_plot,
        return_full_pose=True,
    )

    joints = output.joints[0].detach().cpu().numpy()
    fit_errors = compute_fit_errors(joints, body_frame, lhand_frame, rhand_frame, sequence, joint_name_to_idx)
    result = {
        "frame_index": np.array(frame_index, dtype=np.int32),
        "betas": shared_betas.detach().cpu().numpy()[0],
        "global_orient": global_orient.detach().cpu().numpy()[0],
        "body_pose": final_body_pose.detach().cpu().numpy()[0],
        "left_hand_pose": left_hand_pose.detach().cpu().numpy()[0],
        "right_hand_pose": right_hand_pose.detach().cpu().numpy()[0],
        "transl": transl.detach().cpu().numpy()[0],
        "smplx_joints": joints,
        **{key: np.array(val, dtype=np.float32) for key, val in fit_errors.items()},
    }
    if output.vertices is not None:
        result["smplx_vertices"] = output.vertices[0].detach().cpu().numpy()
    stage_timings["final_s"] = _profile_end(device, profile_enabled, stage_start)
    if profile_enabled:
        result["stage_timings"] = stage_timings
    return result


def save_result(result: Dict[str, np.ndarray], output_dir: Path, frame_index: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / f"frame_{frame_index:06d}.npz", **result)
    with (output_dir / f"frame_{frame_index:06d}.pkl").open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    sequence = load_sequence(args.recording_dir)
    joint_name_to_idx = get_joint_name_to_index()

    model = create_model(args.model_dir, args.gender, args.num_betas, args.num_pca_comps, device)
    vposer = None
    if args.use_vposer:
        vposer, _ = load_vposer(str(args.vposer_dir), vp_model="snapshot")
        vposer = vposer.to(device=device)
        vposer.eval()

    shared_betas = optimize_shared_betas(model, vposer, sequence, args, joint_name_to_idx, device)

    if args.run_full_sequence:
        fit_indices = list(range(0, sequence.body.shape[0], max(1, args.frame_step)))
        if args.max_frames > 0:
            fit_indices = fit_indices[:args.max_frames]
    else:
        fit_indices = [args.frame_index]
    aggregate = []
    per_frame_dir = args.output_dir / "per_frame"
    prev_result = None
    prev_prev_result = None

    for frame_index in fit_indices:
        result = fit_single_frame(
            model,
            vposer,
            frame_index,
            sequence,
            shared_betas,
            args,
            joint_name_to_idx,
            device,
            init_state=prev_result if args.run_full_sequence else None,
            prev_state=prev_result if args.run_full_sequence else None,
            prev_prev_state=prev_prev_result if args.run_full_sequence else None,
        )
        save_result(result, per_frame_dir, frame_index)
        if not args.no_mesh:
            save_mesh(result, model, per_frame_dir, frame_index, device)
        if not args.no_plot:
            save_debug_plot(result, sequence, joint_name_to_idx, model, device, per_frame_dir, frame_index)
        aggregate.append(result)
        prev_prev_result = prev_result
        prev_result = result

    if args.run_full_sequence and not args.disable_post_smooth and len(aggregate) >= 3:
        if args.smooth_window > 0 and args.smooth_sigma > 0:
            smooth_window = args.smooth_window
            smooth_sigma = args.smooth_sigma
        else:
            smooth_window, smooth_sigma = estimate_default_smoothing_params(aggregate)

        transl_arr = np.stack([item["transl"] for item in aggregate], axis=0)
        body_pose_arr = np.stack([item["body_pose"] for item in aggregate], axis=0)
        global_orient_arr = np.stack([item["global_orient"] for item in aggregate], axis=0)
        transl_arr = suppress_sequence_spikes(transl_arr)
        body_pose_arr = suppress_sequence_spikes(body_pose_arr)
        global_orient_arr = suppress_sequence_spikes(global_orient_arr)
        smoothed_transl = smooth_sequence_array(transl_arr, smooth_window, smooth_sigma)
        smoothed_body_pose = smooth_sequence_array(body_pose_arr, smooth_window, smooth_sigma)
        smoothed_global_orient = smooth_axis_angle_sequence(global_orient_arr, smooth_window, smooth_sigma)

        for idx, item in enumerate(aggregate):
            item["transl"] = smoothed_transl[idx].astype(np.float32)
            item["body_pose"] = smoothed_body_pose[idx].astype(np.float32)
            item["global_orient"] = smoothed_global_orient[idx].astype(np.float32)
            output = model(
                betas=torch.tensor(item["betas"], dtype=torch.float32, device=device).unsqueeze(0),
                global_orient=torch.tensor(item["global_orient"], dtype=torch.float32, device=device).unsqueeze(0),
                body_pose=torch.tensor(item["body_pose"], dtype=torch.float32, device=device).unsqueeze(0),
                left_hand_pose=torch.tensor(item["left_hand_pose"], dtype=torch.float32, device=device).unsqueeze(0),
                right_hand_pose=torch.tensor(item["right_hand_pose"], dtype=torch.float32, device=device).unsqueeze(0),
                transl=torch.tensor(item["transl"], dtype=torch.float32, device=device).unsqueeze(0),
                expression=torch.zeros((1, model.num_expression_coeffs), dtype=torch.float32, device=device),
                jaw_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
                leye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
                reye_pose=torch.zeros((1, 3), dtype=torch.float32, device=device),
                return_verts=not args.no_mesh or not args.no_plot,
                return_full_pose=True,
            )
            joints = output.joints[0].detach().cpu().numpy()
            item["smplx_joints"] = joints
            if output.vertices is not None:
                item["smplx_vertices"] = output.vertices[0].detach().cpu().numpy()
            fit_errors = compute_fit_errors(
                joints,
                sequence.body[int(item["frame_index"])],
                sequence.left_hand[int(item["frame_index"])],
                sequence.right_hand[int(item["frame_index"])],
                sequence,
                joint_name_to_idx,
            )
            for key, val in fit_errors.items():
                item[key] = np.array(val, dtype=np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "smplx_fit_sequence.npz",
        frame_indices=np.array([item["frame_index"] for item in aggregate], dtype=np.int32),
        betas=np.stack([item["betas"] for item in aggregate], axis=0),
        global_orient=np.stack([item["global_orient"] for item in aggregate], axis=0),
        body_pose=np.stack([item["body_pose"] for item in aggregate], axis=0),
        left_hand_pose=np.stack([item["left_hand_pose"] for item in aggregate], axis=0),
        right_hand_pose=np.stack([item["right_hand_pose"] for item in aggregate], axis=0),
        transl=np.stack([item["transl"] for item in aggregate], axis=0),
        overall_mean_error_m=np.array([item["overall_mean_error_m"] for item in aggregate], dtype=np.float32),
        body_mean_error_m=np.array([item["body_mean_error_m"] for item in aggregate], dtype=np.float32),
        left_hand_mean_error_m=np.array([item["left_hand_mean_error_m"] for item in aggregate], dtype=np.float32),
        right_hand_mean_error_m=np.array([item["right_hand_mean_error_m"] for item in aggregate], dtype=np.float32),
    )


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    main()
