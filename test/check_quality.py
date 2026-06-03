"""Check two key quality metrics against fit baseline after every track change."""
import sys, numpy as np
from pathlib import Path

def check(track_npz, fit_npz):
    track = np.load(track_npz, allow_pickle=True)
    fit = np.load(fit_npz, allow_pickle=True)

    tj = track["smplx_joints"]; fj = fit["smplx_joints"]
    if tj.ndim==4: tj=tj[:,0,:,:]
    if fj.ndim==4: fj=fj[:,0,:,:]
    sc = float(fit["input_scale"].reshape(()))
    ti = track["frame_indices"]; fi = fit["frame_indices"]
    m = min(len(ti),len(fi),len(tj),len(fj))
    common = np.intersect1d(ti[:m], fi[:m])
    tm = {f:i for i,f in enumerate(ti[:m])}
    fm = {f:i for i,f in enumerate(fi[:m])}
    cf = sorted(common)

    # === SPINE ===
    S = [0,3,6,9,12]
    def seg_angles(j, idx):
        pts=j[idx]; a=[]
        for i in range(1,len(idx)-1):
            v1=pts[i]-pts[i-1]; v2=pts[i+1]-pts[i]
            cos=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
            a.append(np.degrees(np.arccos(np.clip(cos,-1,1))))
        return np.array(a)

    t_ang = np.array([seg_angles(tj[tm[f]]/sc, S) for f in cf])
    f_ang = np.array([seg_angles(fj[fm[f]]/sc, S) for f in cf])
    t_max = np.nanmax(t_ang, axis=1); f_max = np.nanmax(f_ang, axis=1)
    spine_bad = int(np.sum(t_max - f_max > 15))

    print(f"SPINE  >fit+15°: {spine_bad}/{len(cf)}")
    print(f"       p→s1→s2: track={np.nanmean(t_ang[:,0]):.1f}°  fit={np.nanmean(f_ang[:,0]):.1f}°")
    print(f"       P50/P90:  {np.nanpercentile(t_max,50):.1f}/{np.nanpercentile(t_max,90):.1f}°  "
          f"fit={np.nanpercentile(f_max,50):.1f}/{np.nanpercentile(f_max,90):.1f}°")

    # === HIP ===
    t_bp = track["body_pose"]; f_bp = fit["body_pose"]
    if t_bp.ndim==3: t_bp=t_bp[:,0,:]
    if f_bp.ndim==3: f_bp=f_bp[:,0,:]

    tly = np.array([t_bp[tm[f], 1] for f in cf])  # L hip Y
    try_ = np.array([t_bp[tm[f], 4] for f in cf])  # R hip Y
    fly = np.array([f_bp[fm[f], 1] for f in cf])
    fry = np.array([f_bp[fm[f], 4] for f in cf])

    opp = int(np.sum((np.abs(tly) > 0.3) & (np.abs(try_) > 0.3) & (tly * try_ < 0)))
    print(f"\nHIP    opposite twist: {opp}/{len(cf)}")
    print(f"       L/R corr: track={np.corrcoef(tly, try_)[0,1]:.3f}  fit={np.corrcoef(fly, fry)[0,1]:.3f}")
    print(f"       |L-R| asym: track={np.mean(np.abs(tly-try_)):.3f}  fit={np.mean(np.abs(fly-fry)):.3f}")
    print(f"       L_y/R_y mean: {np.mean(tly):.3f}/{np.mean(try_):.3f}  fit={np.mean(fly):.3f}/{np.mean(fry):.3f}")

    return spine_bad, opp

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("track_npz", type=Path)
    p.add_argument("--fit-npz", type=Path,
                   default="sessions/mocap_live_rkw_20260603_test01/smplx_retarget/smplx_fit_sequence.npz")
    args = p.parse_args()
    check(args.track_npz, args.fit_npz)
