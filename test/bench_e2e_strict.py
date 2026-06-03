"""Strict end-to-end FPS measurement with warmup."""
import time, cv2, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.nn as nn
from smplx.lbs import batch_rodrigues, transform_mat
import smplx

from roc.config.yaml_io import load_capture_config
from roc.tracking.mediapipe_tracker import MediapipeTracker
from roc.tracking.model_paths import pose_model_path_for_complexity, hand_model_path
from roc.triangulation.cameras import load_camera_group_from_toml, camera_order_indices
from roc.triangulation.triangulate import triangulate_sequence
from roc.mocap.track import RealtimeSmplxTracker
from roc.mocap.retarget import RetargetConfig, RetargetMode

# ── Vec FK ──────────────────────────────────────────
parents_22 = torch.tensor([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
_LEVEL_CHILDREN = [[1,2,3],[4,5,6],[7,8,9],[10,11,12,13,14],[15,16,17],[18,19],[20,21]]
_LEVEL_PARENTS  = [[0,0,0],[1,2,3],[4,5,6],[7,8,9,9,9],[12,13,14],[16,17],[18,19]]  # parents of children at each level

def brt_vec(rm, J, parents, dtype=torch.float32):
    B, N = rm.shape[:2]; dev = rm.device
    je = J.unsqueeze(-1); rj = je.clone(); rj[:, 1:] -= je[:, parents[1:]]
    tm = transform_mat(rm.reshape(-1, 3, 3), rj.reshape(-1, 3, 1)).reshape(-1, N, 4, 4)
    tc = torch.zeros(B, N, 4, 4, device=dev, dtype=dtype); tc[:, 0] = tm[:, 0]
    for children, par in zip(_LEVEL_CHILDREN, _LEVEL_PARENTS):
        tc[:, children] = torch.matmul(tc[:, par], tm[:, children])
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


# ── Setup ───────────────────────────────────────────
STEPS = 30
full_model = smplx.create('models/smplx', 'smplx', gender='neutral', num_betas=10, num_pca_comps=12)
fk_model = torch.compile(SkeletonFKVec(full_model), mode="default")

prepare_session = Path("sessions/prepare_20260603_102602")
calib_session = Path("sessions/calib_20260603_102832")
video_dir = Path("sessions/mocap_live_rkw_20260603_test01/videos")

capture_config = load_capture_config(prepare_session / "capture_config.yaml")
camera_group = load_camera_group_from_toml(calib_session / "calibration.toml")
ordered_serials = [s for s in capture_config.camera_serials if (video_dir / f"{s}.mp4").exists()]
reorder_indices = camera_order_indices(ordered_serials, [c.get_name() for c in camera_group.cameras])

pose_model = pose_model_path_for_complexity(1)
hand_model = hand_model_path()

# ── Warmup: 50 frames ──
print("Warming up (50 frames)...")
n_warmup = 50
pose_2d_w = []; pose_conf_w = []; lh_2d_w = []; lh_conf_w = []; rh_2d_w = []; rh_conf_w = []

with ExitStack() as stack:
    caps_w = {s: cv2.VideoCapture(str(video_dir / f"{s}.mp4")) for s in ordered_serials}
    trackers_w = {s: stack.enter_context(MediapipeTracker(
        pose_model_path=pose_model, hand_model_path=hand_model,
        model_complexity=1, hands_enabled=True, delegate="cpu")) for s in ordered_serials}
    for fi in range(n_warmup):
        frame_set = {}
        for s in ordered_serials:
            ret, frame = caps_w[s].read()
            if ret: frame_set[s] = frame
        fps_l = []; pconf_l = []; lhp_l = []; rhp_l = []; lc_l = []; rc_l = []
        ts = fi * 33
        def detect_one(s):
            f = frame_set[s]; t = trackers_w[s]
            return s, t.detect_pose(f, timestamp_ms=ts), t.detect_hands(f, timestamp_ms=ts)
        results = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(detect_one, s): s for s in ordered_serials if s in frame_set}
            for fut in as_completed(futures):
                s, pose, hand = fut.result()
                results[s] = (pose, hand)
        for s in ordered_serials:
            if s in results:
                pose, hand = results[s]
                fps_l.append(pose.xy); pconf_l.append(pose.confidence)
                lhp_l.append(hand.left_xy); rhp_l.append(hand.right_xy)
                lc_l.append(hand.left_confidence); rc_l.append(hand.right_confidence)
        pose_2d_w.append(np.stack(fps_l)); pose_conf_w.append(np.stack(pconf_l))
        lh_2d_w.append(np.stack(lhp_l)); lh_conf_w.append(np.stack(lc_l))
        rh_2d_w.append(np.stack(rhp_l)); rh_conf_w.append(np.stack(rc_l))
    for c in caps_w.values(): c.release()

pose_np_w = np.stack(pose_2d_w, axis=1).astype(np.float32)
lh_np_w = np.stack(lh_2d_w, axis=1).astype(np.float32)
rh_np_w = np.stack(rh_2d_w, axis=1).astype(np.float32)
pc_w = np.stack(pose_conf_w, axis=1).astype(np.float32)
lc_w = np.stack(lh_conf_w, axis=1).astype(np.float32)
rc_w = np.stack(rh_conf_w, axis=1).astype(np.float32)
pose_np_w = pose_np_w[reorder_indices]; lh_np_w = lh_np_w[reorder_indices]; rh_np_w = rh_np_w[reorder_indices]
pc_w = pc_w[reorder_indices]; lc_w = lc_w[reorder_indices]; rc_w = rc_w[reorder_indices]
all2d_w = np.concatenate([pose_np_w, lh_np_w, rh_np_w], axis=2)
allc_w = np.concatenate([pc_w, lc_w, rc_w], axis=2)
all2d_w = np.where(allc_w[..., None] <= 0.1, np.nan, all2d_w)
pts3d_w, _ = triangulate_sequence(camera_group, all2d_w)

config = RetargetConfig(mode=RetargetMode.TRACK, device="cpu", model_dir=Path("models/smplx"))
config.track_pose_steps = STEPS
tracker = RealtimeSmplxTracker(config, Path("/tmp/e2e_strict"))
tracker.model = fk_model
for i in range(pts3d_w.shape[0]):
    tracker.update(i, pts3d_w[i])

# ── TIMED: all remaining frames ──
print("Measuring (all remaining frames)...")
n_measure = 250
t_read = 0.0; t_mp = 0.0

pose_2d = []; pose_conf = []; lh_2d = []; lh_conf = []; rh_2d = []; rh_conf = []
t_all0 = time.perf_counter()

with ExitStack() as stack:
    caps = {s: cv2.VideoCapture(str(video_dir / f"{s}.mp4")) for s in ordered_serials}
    for s in ordered_serials:
        for _ in range(n_warmup):
            caps[s].read()

    trackers_m = {s: stack.enter_context(MediapipeTracker(
        pose_model_path=pose_model, hand_model_path=hand_model,
        model_complexity=1, hands_enabled=True, delegate="cpu")) for s in ordered_serials}

    for fi in range(n_measure):
        t0 = time.perf_counter()
        frame_set = {}
        for s in ordered_serials:
            ret, frame = caps[s].read()
            if ret: frame_set[s] = frame
        t_read += time.perf_counter() - t0

        t1 = time.perf_counter()
        fps_l = []; pconf_l = []; lhp_l = []; rhp_l = []; lc_l = []; rc_l = []
        ts = fi * 33

        def detect_one_m(s):
            f = frame_set[s]; t = trackers_m[s]
            return s, t.detect_pose(f, timestamp_ms=ts), t.detect_hands(f, timestamp_ms=ts)

        results = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(detect_one_m, s): s for s in ordered_serials if s in frame_set}
            for fut in as_completed(futures):
                s, pose, hand = fut.result()
                results[s] = (pose, hand)
        t_mp += time.perf_counter() - t1

        for s in ordered_serials:
            if s in results:
                pose, hand = results[s]
                fps_l.append(pose.xy); pconf_l.append(pose.confidence)
                lhp_l.append(hand.left_xy); rhp_l.append(hand.right_xy)
                lc_l.append(hand.left_confidence); rc_l.append(hand.right_confidence)
        pose_2d.append(np.stack(fps_l)); pose_conf.append(np.stack(pconf_l))
        lh_2d.append(np.stack(lhp_l)); lh_conf.append(np.stack(lc_l))
        rh_2d.append(np.stack(rhp_l)); rh_conf.append(np.stack(rc_l))
    for c in caps.values(): c.release()

# Triangulate
t_tri0 = time.perf_counter()
pose_np = np.stack(pose_2d, axis=1).astype(np.float32)
lh_np = np.stack(lh_2d, axis=1).astype(np.float32)
rh_np = np.stack(rh_2d, axis=1).astype(np.float32)
pc = np.stack(pose_conf, axis=1).astype(np.float32)
lc = np.stack(lh_conf, axis=1).astype(np.float32)
rc = np.stack(rh_conf, axis=1).astype(np.float32)
pose_np = pose_np[reorder_indices]; lh_np = lh_np[reorder_indices]; rh_np = rh_np[reorder_indices]
pc = pc[reorder_indices]; lc = lc[reorder_indices]; rc = rc[reorder_indices]
all2d = np.concatenate([pose_np, lh_np, rh_np], axis=2)
allc = np.concatenate([pc, lc, rc], axis=2)
all2d = np.where(allc[..., None] <= 0.1, np.nan, all2d)
pts3d, _ = triangulate_sequence(camera_group, all2d)
t_tri = time.perf_counter() - t_tri0

# Track
t_track0 = time.perf_counter()
for i in range(pts3d.shape[0]):
    tracker.update(n_warmup + i, pts3d[i])
t_track = time.perf_counter() - t_track0

t_total = time.perf_counter() - t_all0

print(f"\n{'='*60}")
print(f"E2E strict ({n_measure} frames, after {n_warmup}fr warmup)")
print(f"{'='*60}")
print(f"  Video read:       {t_read*1000/n_measure:5.0f}ms")
print(f"  MediaPipe (4cam):  {t_mp*1000/n_measure:5.0f}ms  ({n_measure/t_mp:.1f} FPS)")
print(f"  Triangulate:       {t_tri*1000/n_measure:5.0f}ms  ({n_measure/t_tri:.0f} FPS)")
print(f"  Track (Adam):      {t_track*1000/n_measure:5.0f}ms  ({n_measure/t_track:.1f} FPS)")
print(f"  {'─'*40}")
print(f"  TOTAL:             {t_total*1000/n_measure:5.0f}ms  ({n_measure/t_total:.2f} FPS)")
