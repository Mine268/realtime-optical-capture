"""Test body-only SMPL-X tracking with all needed joints."""
import time, sys, numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import smplx
from smplx.lbs import lbs as lbs_fn

from roc.mocap.track import RealtimeSmplxTracker
from roc.mocap.retarget import RetargetConfig, RetargetMode
from roc.mocap.render_reprojection_overlays import render_smplx_reprojection_overlays


class BodyOnlySMPLX(nn.Module):
    """Body-only SMPL-X: body LBS + derived face/foot landmarks via rotated offsets."""

    def __init__(self, full_model, needed_indices):
        super().__init__()
        J_full = full_model.J_regressor
        importance = J_full[:22, :].abs().max(dim=0).values
        keep_v = torch.where(importance > 0)[0]
        n_body = len(keep_v)

        col_idx = torch.cat([torch.arange(v * 3, v * 3 + 3) for v in keep_v])
        self.register_buffer("_v_template", full_model.v_template[keep_v].clone())
        self.register_buffer("_shapedirs", full_model.shapedirs[keep_v].clone())
        self.register_buffer("_posedirs", full_model.posedirs[:21 * 9][:, col_idx].clone())
        self.register_buffer("_lbs_weights", full_model.lbs_weights[keep_v][:, :22].clone())
        self.register_buffer("_parents", full_model.parents[:22].clone())
        self.register_buffer("_J_regressor", full_model.J_regressor[:22][:, keep_v].clone())

        # Face/foot landmark offsets from body joints at rest pose
        with torch.no_grad():
            ref = full_model(
                betas=torch.zeros(1, 10), body_pose=torch.zeros(1, 63),
                global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3),
                left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12),
                return_verts=False,
            ).joints[0]  # (127, 3)
            body_ref = ref[:22]

        # Map needed indices > 21 to (parent_body_joint, rest_offset)
        self._derived = {}
        for ji in needed_indices:
            if ji < 22:
                continue
            dists = torch.norm(body_ref - ref[ji], dim=1)
            parent = int(dists.argmin())
            offset = (ref[ji] - body_ref[parent]).clone()
            self._derived[ji] = (parent, offset)

        self.NUM_BODY_JOINTS = 22
        self.NUM_JOINTS = max(needed_indices) + 1
        self.NUM_BODY_JOINTS_ATTR = 22
        self._n_verts = n_body
        self._needed = needed_indices

    def forward(self, betas, body_pose, global_orient, transl,
                left_hand_pose=None, right_hand_pose=None,
                return_verts=False, **kwargs):
        batch_size = betas.shape[0]
        pose_body = torch.cat([global_orient, body_pose], dim=1)

        # Body LBS
        verts, body_joints = lbs_fn(
            betas, pose_body,
            self._v_template, self._shapedirs, self._posedirs,
            self._J_regressor, self._parents, self._lbs_weights,
        )
        body_joints = body_joints + transl.unsqueeze(dim=1)  # (B, 22, 3)

        # Compute joint rotation matrices for derived landmark rotation
        from smplx.lbs import batch_rodrigues, batch_rigid_transform
        rot_mats = batch_rodrigues(pose_body.view(-1, 3)).view(batch_size, 22, 3, 3)
        # We need rest joint positions for the transform chain
        with torch.no_grad():
            verts_zero, _ = lbs_fn(
                torch.zeros_like(betas), torch.zeros_like(pose_body),
                self._v_template, self._shapedirs, self._posedirs,
                self._J_regressor, self._parents, self._lbs_weights,
            )
            rest_joints = verts_zero[:, :22, :]  # body joints at rest, shaped by betas=0
        _, A = batch_rigid_transform(rot_mats, rest_joints, self._parents)
        # A: (B, 22, 4, 4) — world-space transforms

        # Build full joint array
        all_joints = torch.zeros(batch_size, self.NUM_JOINTS, 3,
                                 device=body_joints.device, dtype=body_joints.dtype)
        all_joints[:, :22, :] = body_joints

        for ji, (parent, offset) in self._derived.items():
            R = A[:, parent, :3, :3]  # (B, 3, 3) rotation of parent joint
            offset_b = offset.to(body_joints.device).view(1, 3, 1)
            rotated_offset = torch.bmm(R, offset_b).squeeze(-1)  # (B, 3)
            all_joints[:, ji, :] = body_joints[:, parent, :] + rotated_offset

        class Out:
            pass
        out = Out()
        out.joints = all_joints
        if return_verts:
            out.vertices = verts
        return out


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
full_model = smplx.create('models/smplx', 'smplx', gender='neutral', num_betas=10, num_pca_comps=12)
mocap_npz = Path("sessions/mocap_live_rkw_20260603_test01/mocap_live_rkw_20260603_test01.npz")
data = np.load(mocap_npz, allow_pickle=True)
pts = data["points_3d_raw"]

# Determine needed joints from a reference tracker
ref_cfg = RetargetConfig(mode=RetargetMode.TRACK, device="cpu", model_dir=Path("models/smplx"))
ref_tracker = RealtimeSmplxTracker(ref_cfg, Path("/tmp/ref_tracker"))
needed = ref_tracker._src.cpu().unique().tolist()
print(f"Needed joints: {sorted(needed)} ({len(needed)} joints)")

# Create body model with ALL needed joints (body + derived face/foot)
body_model = BodyOnlySMPLX(full_model, needed)
print(f"Body verts: {body_model._n_verts}/{full_model.v_template.shape[0]} "
      f"({body_model._n_verts/full_model.v_template.shape[0]*100:.0f}%)")
print(f"Output joints: {body_model.NUM_JOINTS} (includes derived face/foot)")

# _src stays as-is — body model outputs at original SMPL-X full skeleton indices
print(f"Body verts: {body_model._n_verts}/{full_model.v_template.shape[0]} "
      f"({body_model._n_verts/full_model.v_template.shape[0]*100:.0f}%)")

# Verify correctness on ALL needed joints
max_diff = 0
for _ in range(50):
    bp = torch.randn(1, 63) * 0.5; go = torch.randn(1, 3) * 0.8
    tr = torch.randn(1, 3) * 0.05; betas = torch.randn(1, 10) * 1.0
    with torch.no_grad():
        fj = full_model(betas=betas, body_pose=bp, global_orient=go, transl=tr,
                        left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12),
                        return_verts=False).joints[0, needed, :]
        bj = body_model(betas=betas, body_pose=bp, global_orient=go, transl=tr).joints[0, needed, :]
    diff = (fj - bj).abs().max().item()
    if diff > max_diff:
        max_diff = diff
print(f"Accuracy on {len(needed)} joints: max_diff={max_diff*1000:.2f}mm")
if max_diff > 0.05:
    print("WARNING: significant model mismatch!")

# Benchmark
N = 200
bp_r = torch.randn(1, 63) * 0.3; go_r = torch.randn(1, 3) * 0.5
tr_r = torch.randn(1, 3) * 0.02; betas_r = torch.randn(1, 10) * 0.5

for _ in range(20):
    full_model(betas=betas_r, body_pose=bp_r, global_orient=go_r, transl=tr_r,
               left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12),
               return_verts=False)
t0 = time.perf_counter()
for _ in range(N):
    full_model(betas=betas_r, body_pose=bp_r, global_orient=go_r, transl=tr_r,
               left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12),
               return_verts=False)
fw_full = (time.perf_counter() - t0) / N * 1000

for _ in range(20):
    body_model(betas=betas_r, body_pose=bp_r, global_orient=go_r, transl=tr_r)
t0 = time.perf_counter()
for _ in range(N):
    body_model(betas=betas_r, body_pose=bp_r, global_orient=go_r, transl=tr_r)
fw_body = (time.perf_counter() - t0) / N * 1000

# FW+BW
for _ in range(5):
    bp_g = bp_r.clone().detach().requires_grad_(True)
    out = full_model(betas=betas_r, body_pose=bp_g, global_orient=go_r, transl=tr_r,
                     left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12),
                     return_verts=False)
    out.joints.mean().backward()
t0 = time.perf_counter()
for _ in range(50):
    bp_g = bp_r.clone().detach().requires_grad_(True)
    out = full_model(betas=betas_r, body_pose=bp_g, global_orient=go_r, transl=tr_r,
                     left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12),
                     return_verts=False)
    out.joints.mean().backward()
fwb_full = (time.perf_counter() - t0) / 50 * 1000

for _ in range(5):
    bp_g2 = bp_r.clone().detach().requires_grad_(True)
    go_g2 = go_r.clone().detach().requires_grad_(True)
    betas_g = betas_r.clone().detach().requires_grad_(True)
    out = body_model(betas=betas_g, body_pose=bp_g2, global_orient=go_g2, transl=tr_r)
    out.joints.mean().backward()
t0 = time.perf_counter()
for _ in range(50):
    bp_g2 = bp_r.clone().detach().requires_grad_(True)
    go_g2 = go_r.clone().detach().requires_grad_(True)
    betas_g = betas_r.clone().detach().requires_grad_(True)
    out = body_model(betas=betas_g, body_pose=bp_g2, global_orient=go_g2, transl=tr_r)
    out.joints.mean().backward()
fwb_body = (time.perf_counter() - t0) / 50 * 1000

print(f"\nForward:  full={fw_full:.1f}ms  body={fw_body:.1f}ms  speedup={fw_full/fw_body:.1f}x")
print(f"FW+BW:    full={fwb_full:.1f}ms  body={fwb_body:.1f}ms  speedup={fwb_full/fwb_body:.1f}x")

# ---------------------------------------------------------------------------
# Run tracker
# ---------------------------------------------------------------------------
print(f"\n=== Running body-only track ===")
output_dir = Path("sessions/mocap_live_rkw_20260603_test01/smplx_track_body_only")
output_dir.mkdir(parents=True, exist_ok=True)

config = RetargetConfig(mode=RetargetMode.TRACK, device="cpu", model_dir=Path("models/smplx"))
config.track_pose_steps = 9

tracker = RealtimeSmplxTracker(config, output_dir)
tracker.model = body_model

n_frames = 300
t0 = time.perf_counter()
for i in range(n_frames):
    result = tracker.update(i, pts[i])
    if i % 50 == 0:
        err = result.get("body_mean_error_m", float("nan"))
        print(f"  frame {i}: body_err={err:.4f}m")
elapsed = time.perf_counter() - t0
print(f"\nTrack: {n_frames} frames in {elapsed:.1f}s ({n_frames/elapsed:.1f} FPS)")

npz_path = tracker.save(source_npz=mocap_npz)
print(f"Saved: {npz_path}")

# Accuracy
track_data = np.load(npz_path, allow_pickle=True)
fit_data = np.load("sessions/mocap_live_rkw_20260603_test01/smplx_retarget/smplx_fit_sequence.npz", allow_pickle=True)

tj = track_data["smplx_joints"]; fj = fit_data["smplx_joints"]
if tj.ndim == 4: tj = tj[:, 0, :, :]
if fj.ndim == 4: fj = fj[:, 0, :, :]
# Body model outputs 22 joints; compare only those
n_joints = min(22, tj.shape[1], fj.shape[1])
sc = float(fit_data["input_scale"].reshape(()))
ti = track_data["frame_indices"]; fi = fit_data["frame_indices"]
min_f = min(len(tj), len(fj), len(ti), len(fi))
common = np.intersect1d(ti[:min_f], fi[:min_f])
t_map = {f: i for i, f in enumerate(ti[:min_f])}
f_map = {f: i for i, f in enumerate(fi[:min_f])}

# body joints only (first 22)
errors = [np.mean(np.linalg.norm(tj[t_map[f]][:n_joints]-fj[f_map[f]][:n_joints],axis=1)/sc) for f in common]
print(f"Body error vs fit: {np.mean(errors):.1f}mm")

# Spine metrics
def seg_angles(j, idx):
    pts = j[idx]; a = []
    for i in range(1, len(idx)-1):
        v1=pts[i]-pts[i-1]; v2=pts[i+1]-pts[i]
        cos=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        a.append(np.degrees(np.arccos(np.clip(cos,-1,1))))
    return np.array(a)

S = [0,3,6,9,12]
fit_fj = fit_data["smplx_joints"]
if fit_fj.ndim==4: fit_fj=fit_fj[:,0,:,:]
f_map2 = {f:i for i,f in enumerate(fi[:min_f])}

t_ang = np.array([seg_angles(tj[t_map[f]]/sc,S) for f in common])
f_ang = np.array([seg_angles(fit_fj[f_map2[f]]/sc,S) for f in common])
t_max = np.nanmax(t_ang,axis=1)
f_max = np.nanmax(f_ang,axis=1)
print(f"Spine p→s1→s2: {np.nanmean(t_ang[:,0]):.1f}° (fit: {np.nanmean(f_ang[:,0]):.1f}°)")
print(f"Spine P90:      {np.nanpercentile(t_max,90):.1f}° (fit: {np.nanpercentile(f_max,90):.1f}°)")

# Generate video
print("\nGenerating video...")
render_smplx_reprojection_overlays(
    mocap_npz_path=mocap_npz,
    smplx_npz_path=npz_path,
    calibration_toml=Path("sessions/calib_20260603_102832/calibration.toml"),
    video_dir=Path("sessions/mocap_live_rkw_20260603_test01/videos"),
    output_dir=Path("sessions/mocap_live_rkw_20260603_test01/reprojection_videos/track_body_only_v2"),
    confidence_threshold=0.1,
    frame_limit=100,
    combined_scale=0.5,
)
print("Done!")
