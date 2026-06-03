"""Pure skeleton FK tracker — no vertices, no LBS, just rodrigues + rigid_transform."""
import time, sys, numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import smplx
from smplx.lbs import batch_rodrigues, batch_rigid_transform, vertices2joints

from roc.mocap.track import RealtimeSmplxTracker
from roc.mocap.retarget import RetargetConfig, RetargetMode
from roc.mocap.render_reprojection_overlays import render_smplx_reprojection_overlays


class SkeletonFK(nn.Module):
    """Skeleton-only FK: body joints via FK, all others via rotated offsets.

    Outputs 127 joints (same as full SMPL-X) so tracker internals work unchanged.
    """

    def __init__(self, full_model):
        super().__init__()
        with torch.no_grad():
            # Body FK: rest-pose joint positions
            J_rest = torch.einsum('bik,ji->bjk', [full_model.v_template.unsqueeze(0), full_model.J_regressor])
            self.register_buffer("_J_body", J_rest[0, :22, :])  # (22, 3)
            self.register_buffer("_parents", full_model.parents[:22])

            # Shape correction precompute
            sd = full_model.shapedirs  # (10475, 3, 10)
            sd_flat = sd.permute(0, 2, 1).reshape(10475, 30)
            J_reg_sd = full_model.J_regressor[:22] @ sd_flat.float()
            self.register_buffer("_J_reg_shapedirs", J_reg_sd.reshape(22, 3, 10))

            # Full model reference output for ALL 127 joints at rest
            ref_out = full_model(
                betas=torch.zeros(1, 10), body_pose=torch.zeros(1, 63),
                global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3),
                left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12),
                return_verts=False,
            )
            ref_joints = ref_out.joints[0]  # (127, 3)
            body_ref = ref_joints[:22]
            self.NUM_JOINTS = ref_joints.shape[0]  # 127

        # Build offset map for ALL non-body joints (22-126)
        derived = {}
        for ji in range(22, self.NUM_JOINTS):
            dists = torch.norm(body_ref - ref_joints[ji], dim=1)
            parent = int(dists.argmin())
            offset = (ref_joints[ji] - body_ref[parent]).clone()
            derived[ji] = (parent, offset)

        self._derived = derived
        self.NUM_BODY_JOINTS = 22
        self.NUM_BODY_JOINTS_ATTR = 22

    def forward(self, betas, body_pose, global_orient, transl,
                left_hand_pose=None, right_hand_pose=None,
                return_verts=False, **kwargs):
        B = body_pose.shape[0]
        pose = torch.cat([global_orient, body_pose], dim=1).reshape(B, 22, 3)
        rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(B, 22, 3, 3)

        # Body FK with optional shape correction
        J = self._J_body.unsqueeze(0).expand(B, -1, -1)
        if betas is not None and betas.abs().sum() > 0:
            delta_j = torch.einsum('ijk,bk->bij', [self._J_reg_shapedirs, betas])
            J = J + delta_j

        J_transformed, A = batch_rigid_transform(rot_mats, J, self._parents)
        body_joints = J_transformed + transl.unsqueeze(dim=1)

        # Build full 127-joint output
        all_joints = torch.zeros(B, self.NUM_JOINTS, 3, device=body_joints.device, dtype=body_joints.dtype)
        all_joints[:, :22, :] = body_joints

        for ji, (parent, offset) in self._derived.items():
            R = A[:, parent, :3, :3]
            offset_b = offset.to(body_joints.device).view(1, 3, 1)
            rotated = torch.bmm(R, offset_b).squeeze(-1)
            all_joints[:, ji, :] = body_joints[:, parent, :] + rotated

        class Out:
            pass
        out = Out()
        out.joints = all_joints
        if return_verts:
            out.vertices = torch.zeros(B, 0, 3, device=body_joints.device)
        return out


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
full_model = smplx.create('models/smplx', 'smplx', gender='neutral', num_betas=10, num_pca_comps=12)
mocap_npz = Path("sessions/mocap_live_rkw_20260603_test01/mocap_live_rkw_20260603_test01.npz")
data = np.load(mocap_npz, allow_pickle=True)
pts = data["points_3d_raw"]

# Verify FK accuracy
fk_model = SkeletonFK(full_model)
max_diff = 0
for _ in range(50):
    bp = torch.randn(1, 63) * 0.5; go = torch.randn(1, 3) * 0.8
    tr = torch.randn(1, 3) * 0.05; betas = torch.randn(1, 10) * 0.5
    with torch.no_grad():
        fj = full_model(betas=betas, body_pose=bp, global_orient=go, transl=tr,
                        left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12),
                        return_verts=False).joints[:, :22, :]
        fk_j = fk_model(betas=betas, body_pose=bp, global_orient=go, transl=tr).joints[:, :22, :]
    diff = (fj - fk_j).abs().max().item()
    if diff > max_diff:
        max_diff = diff
print(f"FK body joints accuracy: max_diff={max_diff*1000:.2f}mm")

# Benchmark
N = 200
bp_r = torch.randn(1, 63) * 0.3; go_r = torch.randn(1, 3) * 0.5; tr_r = torch.randn(1, 3) * 0.02
betas_r = torch.randn(1, 10) * 0.5

for _ in range(20):
    full_model(betas=betas_r, body_pose=bp_r, global_orient=go_r, transl=tr_r,
               left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12), return_verts=False)
t0 = time.perf_counter()
for _ in range(N):
    full_model(betas=betas_r, body_pose=bp_r, global_orient=go_r, transl=tr_r,
               left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12), return_verts=False)
fw_full = (time.perf_counter()-t0)/N*1000

for _ in range(20):
    fk_model(betas=betas_r, body_pose=bp_r, global_orient=go_r, transl=tr_r)
t0 = time.perf_counter()
for _ in range(N):
    fk_model(betas=betas_r, body_pose=bp_r, global_orient=go_r, transl=tr_r)
fw_fk = (time.perf_counter()-t0)/N*1000

for _ in range(5):
    bp_g = bp_r.clone().detach().requires_grad_(True)
    out = full_model(betas=betas_r, body_pose=bp_g, global_orient=go_r, transl=tr_r,
                     left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12), return_verts=False)
    out.joints.mean().backward()
t0 = time.perf_counter()
for _ in range(50):
    bp_g = bp_r.clone().detach().requires_grad_(True)
    out = full_model(betas=betas_r, body_pose=bp_g, global_orient=go_r, transl=tr_r,
                     left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12), return_verts=False)
    out.joints.mean().backward()
fwb_full = (time.perf_counter()-t0)/50*1000

for _ in range(5):
    bp_g2 = bp_r.clone().detach().requires_grad_(True)
    go_g2 = go_r.clone().detach().requires_grad_(True)
    tr_g2 = tr_r.clone().detach().requires_grad_(True)
    out = fk_model(betas=betas_r, body_pose=bp_g2, global_orient=go_g2, transl=tr_g2)
    out.joints.mean().backward()
t0 = time.perf_counter()
for _ in range(50):
    bp_g2 = bp_r.clone().detach().requires_grad_(True)
    go_g2 = go_r.clone().detach().requires_grad_(True)
    tr_g2 = tr_r.clone().detach().requires_grad_(True)
    out = fk_model(betas=betas_r, body_pose=bp_g2, global_orient=go_g2, transl=tr_g2)
    out.joints.mean().backward()
fwb_fk = (time.perf_counter()-t0)/50*1000

print(f"\nForward:  full={fw_full:.1f}ms  FK={fw_fk:.1f}ms  speedup={fw_full/fw_fk:.1f}x")
print(f"FW+BW:    full={fwb_full:.1f}ms  FK={fwb_fk:.1f}ms  speedup={fwb_full/fwb_fk:.1f}x")

# ---------------------------------------------------------------------------
# Run tracker with FK model
# ---------------------------------------------------------------------------
print(f"\n=== FK tracker ===")
output_dir = Path("sessions/mocap_live_rkw_20260603_test01/smplx_track_fk")
output_dir.mkdir(parents=True, exist_ok=True)

config = RetargetConfig(mode=RetargetMode.TRACK, device="cpu", model_dir=Path("models/smplx"))
config.track_pose_steps = 9

tracker = RealtimeSmplxTracker(config, output_dir)
tracker.model = fk_model

n_frames = 300
t0 = time.perf_counter()
for i in range(n_frames):
    result = tracker.update(i, pts[i])
    if i % 50 == 0:
        err = result.get("body_mean_error_m", float("nan"))
        print(f"  frame {i}: body_err={err:.4f}m")
elapsed = time.perf_counter() - t0
print(f"\nFK Track: {n_frames} frames in {elapsed:.1f}s ({n_frames/elapsed:.1f} FPS)")

npz_path = tracker.save(source_npz=mocap_npz)
print(f"Saved: {npz_path}")

# Accuracy
track_data = np.load(npz_path, allow_pickle=True)
fit_data = np.load("sessions/mocap_live_rkw_20260603_test01/smplx_retarget/smplx_fit_sequence.npz", allow_pickle=True)

tj = track_data["smplx_joints"]; fj = fit_data["smplx_joints"]
if tj.ndim == 4: tj = tj[:, 0, :, :]
if fj.ndim == 4: fj = fj[:, 0, :, :]
sc = float(fit_data["input_scale"].reshape(()))
ti = track_data["frame_indices"]; fi = fit_data["frame_indices"]
m = min(len(tj), len(fj), len(ti), len(fi))
common = np.intersect1d(ti[:m], fi[:m])
t_map = {f: i for i, f in enumerate(ti[:m])}
f_map = {f: i for i, f in enumerate(fi[:m])}
errors = [np.mean(np.linalg.norm(tj[t_map[f]][:22]-fj[f_map[f]][:22],axis=1)/sc) for f in common]
print(f"Body error vs fit: {np.mean(errors):.1f}mm")

# Spine metrics
def seg_angles(j, idx):
    pts = j[idx]; a = []
    for i in range(1, len(idx)-1):
        v1=pts[i]-pts[i-1]; v2=pts[i+1]-pts[i]
        cos=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        a.append(np.degrees(np.arccos(np.clip(cos,-1,1))))
    return np.array(a)

S = [0, 3, 6, 9, 12]
fi_f = fit_data["frame_indices"]; fj_f = fit_data["smplx_joints"]
if fj_f.ndim==4: fj_f=fj_f[:,0,:,:]
fm = {f:i for i,f in enumerate(fi_f[:m])}
t_ang = np.array([seg_angles(tj[t_map[f]]/sc,S) for f in common])
f_ang = np.array([seg_angles(fj_f[fm[f]]/sc,S) for f in common])
t_max = np.nanmax(t_ang,axis=1); f_max = np.nanmax(f_ang,axis=1)
print(f"Spine p→s1→s2: {np.nanmean(t_ang[:,0]):.1f}° (fit: {np.nanmean(f_ang[:,0]):.1f}°)")
print(f"Spine P90:      {np.nanpercentile(t_max,90):.1f}° (fit: {np.nanpercentile(f_max,90):.1f}°)")

# Video
print("\nGenerating video...")
render_smplx_reprojection_overlays(
    mocap_npz_path=mocap_npz, smplx_npz_path=npz_path,
    calibration_toml=Path("sessions/calib_20260603_102832/calibration.toml"),
    video_dir=Path("sessions/mocap_live_rkw_20260603_test01/videos"),
    output_dir=Path("sessions/mocap_live_rkw_20260603_test01/reprojection_videos/track_fk"),
    confidence_threshold=0.1, frame_limit=100, combined_scale=0.5,
)
print("Done!")
