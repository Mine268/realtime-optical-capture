"""Test pelvis LR + pelvis_frame weight increase for jump-turn hip+spine fix."""
import time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
import torch.nn as nn
from smplx.lbs import batch_rodrigues, transform_mat
import smplx
from roc.mocap.track import RealtimeSmplxTracker
from roc.mocap.retarget import RetargetConfig, RetargetMode

# Vec FK
parents_22 = torch.tensor([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
levels = [[0],[1,2,3],[4,5,6],[7,8,9],[10,11,12,13,14],[15,16,17],[18,19],[20,21]]
lc = [torch.tensor(l) for l in levels]
lp = [torch.tensor([max(0,parents_22[c].item()) for c in l]) for l in levels]

def brt_vec(rm, J, parents, dtype=torch.float32):
    B, N = rm.shape[:2]; dev = rm.device
    je = J.unsqueeze(-1); rj = je.clone(); rj[:, 1:] -= je[:, parents[1:]]
    tm = transform_mat(rm.reshape(-1, 3, 3), rj.reshape(-1, 3, 1)).reshape(-1, N, 4, 4)
    tc = torch.zeros(B, N, 4, 4, device=dev, dtype=dtype); tc[:, 0] = tm[:, 0]
    for c, p in zip(lc[1:], lp[1:]):
        if len(c) == 0: continue
        tc[:, c] = torch.matmul(tc[:, p], tm[:, c])
    pj = tc[:, :, :3, 3]
    jh = F.pad(je, [0, 0, 0, 1])
    rt = tc - F.pad(torch.matmul(tc, jh), [3, 0, 0, 0, 0, 0, 0, 0])
    return pj, rt

class SkeletonFKVec(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        with torch.no_grad():
            J_rest = torch.einsum('bik,ji->bjk', [full_model.v_template.unsqueeze(0), full_model.J_regressor])
            self.register_buffer("_J_body", J_rest[0, :22, :])
            self.register_buffer("_parents", parents_22)
            ref_out = full_model(betas=torch.zeros(1, 10), body_pose=torch.zeros(1, 63),
                                 global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3),
                                 left_hand_pose=torch.zeros(1, 12), right_hand_pose=torch.zeros(1, 12),
                                 return_verts=False)
            ref_joints = ref_out.joints[0]; body_ref = ref_joints[:22]
            self.NUM_JOINTS = ref_joints.shape[0]
        derived = {}
        for ji in range(22, self.NUM_JOINTS):
            dists = torch.norm(body_ref - ref_joints[ji], dim=1)
            parent = int(dists.argmin())
            offset = (ref_joints[ji] - body_ref[parent]).clone()
            derived[ji] = (parent, offset)
        self._derived = derived
        self.NUM_BODY_JOINTS = 22; self.NUM_BODY_JOINTS_ATTR = 22

    def forward(self, betas, body_pose, global_orient, transl, **kwargs):
        B = body_pose.shape[0]
        pose = torch.cat([global_orient, body_pose], dim=1).reshape(B, 22, 3)
        rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(B, 22, 3, 3)
        J = self._J_body.unsqueeze(0).expand(B, -1, -1)
        J_transformed, A = brt_vec(rot_mats, J, self._parents)
        body_joints = J_transformed + transl.unsqueeze(1)
        all_joints = torch.zeros(B, self.NUM_JOINTS, 3, device=body_joints.device, dtype=body_joints.dtype)
        all_joints[:, :22, :] = body_joints
        for ji, (parent, offset) in self._derived.items():
            R = A[:, parent, :3, :3]
            rotated = torch.bmm(R, offset.to(body_joints.device).view(1, 3, 1)).squeeze(-1)
            all_joints[:, ji, :] = body_joints[:, parent, :] + rotated
        class Out: pass
        out = Out(); out.joints = all_joints
        return out


full_model = smplx.create('models/smplx', 'smplx', gender='neutral', num_betas=10, num_pca_comps=12)
mocap_npz = Path("sessions/mocap_live_rkw_20260603_test01/mocap_live_rkw_20260603_test01.npz")
data = np.load(mocap_npz, allow_pickle=True)
pts = data["points_3d_raw"]

fk_m = torch.compile(SkeletonFKVec(full_model), mode="default")
config = RetargetConfig(mode=RetargetMode.TRACK, device="cpu", model_dir=Path("models/smplx"))
config.track_pose_steps = 30
tracker = RealtimeSmplxTracker(config, Path("/tmp/pelvis_fix"))
tracker.model = fk_m

for i in range(150):
    tracker.update(i, pts[i])

tj = np.stack([a["smplx_joints"].reshape(-1, 3) for a in tracker.aggregate], axis=0)
t_bp = np.stack([a["body_pose"].reshape(63) for a in tracker.aggregate], axis=0)
t_go = np.stack([a["global_orient"].reshape(3) for a in tracker.aggregate], axis=0)

fit = np.load("sessions/mocap_live_rkw_20260603_test01/smplx_retarget/smplx_fit_sequence.npz", allow_pickle=True)
fj = fit["smplx_joints"]; f_bp = fit["body_pose"]; f_go = fit["global_orient"]
if fj.ndim == 4: fj = fj[:, 0, :, :]
if f_bp.ndim == 3: f_bp = f_bp[:, 0, :]
if f_go.ndim == 3: f_go = f_go[:, 0, :]
sc = float(fit["input_scale"].reshape(()))
fi = fit["frame_indices"]
fm = {f: i for i, f in enumerate(fi)}

S = [0, 3, 6, 9, 12]
def sa(j, idx):
    pts = j[idx]; a = []
    for i in range(1, len(idx) - 1):
        v1 = pts[i] - pts[i - 1]; v2 = pts[i + 1] - pts[i]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        a.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.array(a)

print(f"{'Frame':>5s} {'tSpine':>7s} {'fSpine':>7s} {'tLH_y':>7s} {'tRH_y':>7s} {'fRH_y':>7s} {'tPelvY':>7s} {'fPelvY':>7s}")
for f in range(80, 96):
    if f >= len(tj) or f not in fm: continue
    t_sp = np.nanmax(sa(tj[f] / sc, S)); f_sp = np.nanmax(sa(fj[fm[f]] / sc, S))
    t_ly = t_bp[f, 1]; t_ry = t_bp[f, 4]; f_ry = f_bp[fm[f], 4]
    t_py = np.degrees(t_go[f, 1]); f_py = np.degrees(f_go[fm[f], 1])
    flag = " ***" if t_sp - f_sp > 10 else ""
    print(f"{f:>5d} {t_sp:>6.1f}° {f_sp:>6.1f}° {t_ly:>7.3f} {t_ry:>7.3f} {f_ry:>7.3f} {t_py:>7.1f}° {f_py:>7.1f}°{flag}")

m = min(len(tj), len(fj))
bad_sp = sum(1 for i in range(m) if i in fm and np.nanmax(sa(tj[i]/sc,S)) - np.nanmax(sa(fj[fm[i]]/sc,S)) > 15)
opp = sum(1 for i in range(m) if i in fm and abs(t_bp[i,1])>0.3 and abs(t_bp[i,4])>0.3 and t_bp[i,1]*t_bp[i,4]<0)
print(f"\nAll: spine>fit+15°={bad_sp}/{m}  opposite_hip={opp}/{m}")
