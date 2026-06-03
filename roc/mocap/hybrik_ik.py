"""Twist-and-swing inverse kinematics for SMPL-X, adapted from HybrIK.

Given 3D joint positions (e.g. from triangulated MediaPipe keypoints mapped to
SMPL body joints), analytically compute body_pose as axis-angle rotations,
plus global_orient and transl. Zero twist assumed (phis=0); leaf joints
(feet, hands) use identity rotations.

Reference: HybrIK (Li et al., CVPR 2021) — batch_inverse_kinematics_transform in
hybrik/models/layers/smpl/lbs.py
"""

from __future__ import annotations

import numpy as np
import torch


# SMPL-X 21 body joint indices and their parents.
# The first 22 SMPL-X joints match the SMPL 24-joint kinematic tree minus palms
# (left_hand=22, right_hand=23 in SMPL, replaced by finger chains in SMPL-X).
#
# SMPL-X body joint order (joint_name -> index):
#   0:pelvis  1:left_hip  2:right_hip  3:spine1  4:left_knee  5:right_knee
#   6:spine2  7:left_ankle  8:right_ankle  9:spine3  10:left_foot  11:right_foot
#   12:neck  13:left_collar  14:right_collar  15:head  16:left_shoulder
#   17:right_shoulder  18:left_elbow  19:right_elbow  20:left_wrist  21:right_wrist
SMPLX_BODY_PARENTS = [
    -1,   # 0: pelvis (root)
    0,    # 1: left_hip
    0,    # 2: right_hip
    0,    # 3: spine1
    1,    # 4: left_knee
    2,    # 5: right_knee
    3,    # 6: spine2
    4,    # 7: left_ankle
    5,    # 8: right_ankle
    6,    # 9: spine3
    7,    # 10: left_foot
    8,    # 11: right_foot
    9,    # 12: neck
    9,    # 13: left_collar
    9,    # 14: right_collar
    12,   # 15: head
    13,   # 16: left_shoulder
    14,   # 17: right_shoulder
    16,   # 18: left_elbow
    17,   # 19: right_elbow
    18,   # 20: left_wrist
    19,   # 21: right_wrist
]

# Joints that have exactly one child → use standard swing+twist
# Joints with 3 children (spine3=9) → use SVD-based alignment
# Root (pelvis=0) → handled via pelvis orientation
# Leaf joints (feet=10,11, wrists=20,21, head=15) → identity rotation

_NUM_BODY_JOINTS = 22  # pelvis through wrists


def _rodrigues(K: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula: R = I + sin*K + (1-cos)*K^2"""
    ident = torch.eye(3, dtype=K.dtype, device=K.device).unsqueeze(0)
    return ident + sin * K + (1.0 - cos) * torch.bmm(K, K)


def _cross_product_matrix(axis: torch.Tensor) -> torch.Tensor:
    """Build skew-symmetric cross-product matrix K from axis vectors (B, 3)."""
    B = axis.shape[0]
    zeros = torch.zeros(B, 1, dtype=axis.dtype, device=axis.device)
    rx, ry, rz = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
    return torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(B, 3, 3)


def solve_pelvis_orientation(
    rest_hips_center: torch.Tensor,
    rest_spine_dir: torch.Tensor,
    target_hips_center: torch.Tensor,
    target_spine_dir: torch.Tensor,
) -> torch.Tensor:
    """Compute pelvis rotation that aligns rest hip axis + spine to target.

    Uses SVD to find the best rotation aligning two orthonormal frames built from
    (hip_left→hip_right horizontal, pelvis→spine1 vertical).
    """
    B = rest_hips_center.shape[0]

    # Build orthonormal frames from two directions (like HybrIK's batch_get_pelvis_orient_svd)
    def _build_frame(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a / (torch.linalg.norm(a, dim=1, keepdim=True) + 1e-8)
        b = b - torch.sum(b * a, dim=1, keepdim=True) * a
        b = b / (torch.linalg.norm(b, dim=1, keepdim=True) + 1e-8)
        c = torch.cross(a, b, dim=1)
        return torch.stack([a, b, c], dim=1)  # (B, 3, 3)

    R_rest = _build_frame(rest_hips_center, rest_spine_dir)
    R_target = _build_frame(target_hips_center, target_spine_dir)

    # SVD: R_target @ R_rest^T = U @ S @ V^T → R = V @ U^T
    H = torch.bmm(R_target, R_rest.transpose(1, 2))
    U, _, Vt = torch.linalg.svd(H.float())
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))  # V @ U^T

    # Ensure det(R) = 1
    det = torch.linalg.det(R)
    flip_mask = det < 0
    if flip_mask.any():
        V = Vt.transpose(1, 2)
        R_neg = torch.bmm(V[flip_mask], U[flip_mask].transpose(1, 2))
        R_neg[:, :, 2] = -R_neg[:, :, 2]
        R[flip_mask] = R_neg
    return R


def rotation_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Shortest rotation that aligns v1 to v2 (both (B, 3))."""
    v1 = v1 / (torch.linalg.norm(v1, dim=1, keepdim=True) + 1e-8)
    v2 = v2 / (torch.linalg.norm(v2, dim=1, keepdim=True) + 1e-8)
    axis = torch.cross(v1, v2, dim=1)
    axis_norm = torch.linalg.norm(axis, dim=1, keepdim=True)
    cos = torch.sum(v1 * v2, dim=1, keepdim=True)
    sin = axis_norm

    # Handle parallel/anti-parallel cases
    parallel = axis_norm.squeeze(1) < 1e-8
    axis = axis / (axis_norm + 1e-8)
    K = _cross_product_matrix(axis)
    R = _rodrigues(K, sin, cos)

    # Anti-parallel: rotate 180° around any perpendicular axis
    antiparallel = cos.squeeze(1) < -0.9999
    if antiparallel.any():
        perp = torch.zeros_like(v1[antiparallel])
        perp[:, 0] = -v1[antiparallel, 1]
        perp[:, 1] = v1[antiparallel, 0]
        perp_norm = torch.linalg.norm(perp, dim=1, keepdim=True) + 1e-8
        perp = perp / perp_norm
        K_ap = _cross_product_matrix(perp)
        R[antiparallel] = _rodrigues(K_ap, torch.zeros_like(sin[antiparallel]),
                                      -torch.ones_like(cos[antiparallel]))
    return R


def normalize_bone_lengths(
    target_joints: torch.Tensor,
    rest_J_body: torch.Tensor,
    parents: list[int],
) -> torch.Tensor:
    """Scale bone vectors of target skeleton to match rest-pose bone lengths.

    Walks the kinematic tree from root, keeping bone directions from target
    but lengths from rest pose. This makes the skeleton kinematically consistent
    with SMPL-X bone lengths while preserving the target pose directions.
    """
    B = target_joints.shape[0]
    device = target_joints.device
    rel_target = target_joints.clone()
    rel_target[:, 1:] = rel_target[:, 1:] - rel_target[:, [parents[i] for i in range(1, 22)]]

    rel_rest = rest_J_body.clone()
    rel_rest[:, 1:] = rel_rest[:, 1:] - rel_rest[:, [parents[i] for i in range(1, 22)]]

    # Scale each bone vector to rest length while preserving direction
    for i in range(1, 22):
        target_dir = rel_target[:, i]
        rest_dir = rel_rest[:, i]
        target_norm = torch.linalg.norm(target_dir, dim=1, keepdim=True)
        rest_norm = torch.linalg.norm(rest_dir, dim=1, keepdim=True)
        # Preserve direction from target, length from rest (with minimum scale)
        scale = rest_norm / (target_norm + 1e-8)
        # Clamp scale to avoid extreme stretching (>2x or <0.5x)
        scale = torch.clamp(scale, 0.5, 2.0)
        rel_target[:, i] = target_dir * scale

    # Reconstruct absolute joint positions from scaled bone vectors
    normalized = torch.zeros_like(target_joints)
    normalized[:, 0] = target_joints[:, 0]  # root position unchanged
    for i in range(1, 22):
        normalized[:, i] = normalized[:, parents[i]] + rel_target[:, i]

    return normalized


def twist_and_swing_ik(
    target_joints: torch.Tensor,
    smplx_model: object,
    twist_phis: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SMPL-X body_pose, global_orient, transl from 3D joint positions.

    Args:
        target_joints: (B, 22, 3) — 22 SMPL-X body joint positions in meters
        smplx_model: SMPL-X model (smplx.create result) with v_template,
                     J_regressor, parents, lbs_weights, shapedirs
        twist_phis: (B, 21, 2) — twist angles as (cos, sin) pairs per joint.
                    If None, zero twist is used.

    Returns:
        body_pose: (B, 63) — axis-angle rotations for 21 SMPL-X body joints
        global_orient: (B, 3) — global root orientation
        transl: (B, 3) — global translation
    """
    B = target_joints.shape[0]
    device = target_joints.device
    dtype = target_joints.dtype

    parents = SMPLX_BODY_PARENTS

    # Get SMPL-X model data
    v_template = smplx_model.v_template.to(device)  # (V, 3)
    J_regressor = smplx_model.J_regressor.to(device)  # (J, V)

    # Compute rest-pose joints for the 22 body joints
    rest_J_all = torch.matmul(J_regressor, v_template)  # (55, 3)
    rest_J_body = rest_J_all[:22].unsqueeze(0).expand(B, -1, -1).clone()  # (B, 22, 3)

    # Normalize bone lengths to be consistent with SMPL-X rest pose
    target = normalize_bone_lengths(target_joints, rest_J_body, parents)

    # Relative rest-pose bone vectors
    rel_rest = rest_J_body.clone()
    rel_rest[:, 1:] = rel_rest[:, 1:] - rel_rest[:, [parents[i] for i in range(1, 22)]]

    # Target skeleton: center at pelvis
    pelvis_pos = target[:, 0:1].clone()
    target_centered = target - pelvis_pos

    # Relative target bone vectors
    rel_target = target.clone()
    rel_target[:, 1:] = rel_target[:, 1:] - rel_target[:, [parents[i] for i in range(1, 22)]]
    # Set root position to zero (relative)
    rel_target[:, 0] = 0.0

    # ---- 1. Pelvis orientation (global_orient) ----
    # Align hip horizontal axis + spine vertical direction
    rest_hips = rest_J_body[:, 2] - rest_J_body[:, 1]  # right_hip - left_hip
    rest_spine = rest_J_body[:, 3] - rest_J_body[:, 0]  # spine1 - pelvis
    tgt_hips = target_centered[:, 2] - target_centered[:, 1]
    tgt_spine = target_centered[:, 3] - target_centered[:, 0]

    global_orient_mat = solve_pelvis_orientation(rest_hips, rest_spine, tgt_hips, tgt_spine)

    # ---- 2. Twist angles ----
    if twist_phis is None:
        twist_phis = torch.zeros(B, 22, 2, device=device, dtype=dtype)
        twist_phis[:, :, 0] = 1.0  # cos(0) = 1

    # ---- 3. Per-joint IK in kinematic order ----
    rot_mats_local = [global_orient_mat]  # root rotation
    rot_mats_chain = [global_orient_mat]
    # Rest positions of each joint in the world frame (after applying parent rotations)
    world_rest = torch.zeros_like(rest_J_body)
    world_rest[:, 0] = rest_J_body[:, 0]

    for i in range(1, 22):
        p = parents[i]

        # Position of joint i's rest location in the current world frame
        world_rest_i = world_rest[:, p] + torch.bmm(
            rot_mats_chain[p], rel_rest[:, i:i + 1].transpose(1, 2)
        ).squeeze(-1)

        if i in (10, 11, 20, 21, 15):
            # Leaf joints: feet (10,11), wrists (20,21), head (15)
            # Use identity rotation (these are leaf endpoints with no children to align)
            rot_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            world_rest[:, i] = world_rest_i
            rot_mats_chain.append(torch.bmm(rot_mats_chain[p], rot_mat))
            rot_mats_local.append(rot_mat)
            continue

        if i == 9:
            # spine3 (joint 9): has 3 children (neck=12, left_collar=13, right_collar=14)
            # Use SVD to align 3 children simultaneously (like HybrIK's batch_get_3children_orient_svd)
            children_indices = [12, 13, 14]
            child_rest_dirs = []
            child_target_dirs = []
            for c in children_indices:
                child_rest_dirs.append(rel_rest[:, c])
                # Child target position relative to spine3
                child_target = target_centered[:, c] - world_rest_i
                child_target_dirs.append(child_target)

            # Stack children directions → SVD alignment
            R_rest = torch.stack(child_rest_dirs, dim=2)  # (B, 3, 3)
            R_target = torch.stack(child_target_dirs, dim=2)  # (B, 3, 3)

            H = torch.bmm(R_target.float(), R_rest.float().transpose(1, 2))
            U, _, Vt = torch.linalg.svd(H)
            rot_mat = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

            world_rest[:, i] = world_rest_i
            rot_mats_chain.append(torch.bmm(rot_mats_chain[p], rot_mat.float().to(dtype)))
            rot_mats_local.append(rot_mat.float().to(dtype))
            continue

        # Standard case: joint has exactly 1 child
        # Find the child
        child = None
        for c in range(1, 22):
            if parents[c] == i:
                child = c
                break

        if child is None:
            # Shouldn't happen for body joints
            rot_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            world_rest[:, i] = world_rest_i
            rot_mats_chain.append(torch.bmm(rot_mats_chain[p], rot_mat))
            rot_mats_local.append(rot_mat)
            continue

        # ---- Swing: align rest bone direction to target bone direction ----
        child_rest_dir = rel_rest[:, child]  # (B, 3) — rest pose bone direction
        child_target_dir = target_centered[:, child] - world_rest_i  # (B, 3)

        # Scale target bone to match rest bone length (HybrIK trick: prevent bone stretching)
        rest_len = torch.linalg.norm(child_rest_dir, dim=1, keepdim=True)
        target_len = torch.linalg.norm(child_target_dir, dim=1, keepdim=True)
        child_target_dir = child_target_dir * rest_len / (target_len + 1e-8)

        # Rotate target direction into parent's local frame
        child_target_local = torch.bmm(
            rot_mats_chain[p].transpose(1, 2),
            child_target_dir.unsqueeze(-1),
        ).squeeze(-1)

        child_rest_local = rel_rest[:, child].clone()

        # Compute swing rotation: rest → target in local frame
        swing_axis = torch.cross(child_rest_local, child_target_local, dim=1)
        swing_axis_norm = torch.linalg.norm(swing_axis, dim=1, keepdim=True)
        swing_cos = torch.sum(child_rest_local * child_target_local, dim=1, keepdim=True) / (
            torch.linalg.norm(child_rest_local, dim=1, keepdim=True)
            * torch.linalg.norm(child_target_local, dim=1, keepdim=True)
            + 1e-8
        )
        swing_sin = swing_axis_norm / (
            torch.linalg.norm(child_rest_local, dim=1, keepdim=True)
            * torch.linalg.norm(child_target_local, dim=1, keepdim=True)
            + 1e-8
        )

        # Guard against NaN
        swing_axis = swing_axis / (swing_axis_norm + 1e-8)
        K_swing = _cross_product_matrix(swing_axis)
        rot_swing = _rodrigues(K_swing, swing_sin, swing_cos)

        # ---- Twist: rotation around bone axis ----
        twist_axis = child_rest_local / (torch.linalg.norm(child_rest_local, dim=1, keepdim=True) + 1e-8)
        K_twist = _cross_product_matrix(twist_axis)
        phi_cos = twist_phis[:, i - 1, 0:1]  # cos of twist angle
        phi_sin = twist_phis[:, i - 1, 1:2]  # sin of twist angle
        rot_twist = _rodrigues(K_twist, phi_sin, phi_cos)

        rot_mat = torch.bmm(rot_swing, rot_twist)

        world_rest[:, i] = world_rest_i
        rot_mats_chain.append(torch.bmm(rot_mats_chain[p], rot_mat))
        rot_mats_local.append(rot_mat)

    # Stack all local rotation matrices
    all_rot_mats = torch.stack(rot_mats_local, dim=1)  # (B, 22, 3, 3)

    # ---- 4. Convert to axis-angle ----
    # global_orient from rotation matrix 0
    go_mat = all_rot_mats[:, 0]  # (B, 3, 3)
    global_orient = _rotmat_to_axis_angle(go_mat)  # (B, 3)

    # body_pose from rotation matrices 1:22 → 21 joints
    bp_mats = all_rot_mats[:, 1:22]  # (B, 21, 3, 3)
    bp_mats_flat = bp_mats.reshape(B * 21, 3, 3)
    body_pose = _rotmat_to_axis_angle(bp_mats_flat).reshape(B, 63)

    # transl: pelvis position (in world space)
    transl = pelvis_pos.squeeze(1)  # (B, 3)

    return body_pose, global_orient, transl


def _rotmat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices (B, 3, 3) to axis-angle (B, 3)."""
    B = R.shape[0]
    # trace = R00 + R11 + R22
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos = (trace - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0, 1.0)
    angle = torch.acos(cos)

    # sin(angle) * axis from off-diagonal elements
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axis = torch.stack([rx, ry, rz], dim=1)

    # Normalize: axis_angle = angle * axis / sin(angle)
    sin_angle = torch.sin(angle)
    small_angle = sin_angle.abs() < 1e-8
    sin_angle_safe = torch.where(small_angle, torch.ones_like(sin_angle), sin_angle)

    result = angle.unsqueeze(1) * axis / (2.0 * sin_angle_safe.unsqueeze(1))
    result[small_angle] = axis[small_angle] * 0.5  # small angle approximation
    return result
