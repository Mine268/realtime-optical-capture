"""Test RealtimeSmplxRetargeter on GPU to reproduce NaN."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from roc.mocap.retarget import RetargetConfig, RetargetMode, RealtimeSmplxRetargeter


def main() -> None:
    npz_path = Path("sessions/mocap_live_rkw_20260603_test01/mocap_live_rkw_20260603_test01.npz")
    data = np.load(npz_path, allow_pickle=True)
    points_3d = data["points_3d"]
    frames = points_3d.shape[0]
    print(f"Frames: {frames}, points shape: {points_3d.shape}")

    config = RetargetConfig(
        model_dir=Path("models/smplx"),
        mode=RetargetMode.FIT,
        output_dir=Path("/tmp/smplx_rt_diag"),
        device="cuda",
        pose_steps=10,  # fewer steps for speed
        betas_steps=5,
        lower_body_refine=True,
        optimize_hands=False,
        use_vposer=False,
        profile=True,
    )

    retargeter = RealtimeSmplxRetargeter(config, Path("/tmp/smplx_rt_diag"))

    for frame_idx in range(min(20, frames)):
        result = retargeter.update(frame_idx, points_3d[frame_idx])
        bp = result.get("body_pose")
        if bp is not None and np.any(np.isnan(bp)):
            print(f"!!! NaN detected in body_pose at frame {frame_idx}")
        if (frame_idx + 1) % 5 == 0:
            sj = result.get("smplx_joints")
            nan_count = int(np.sum(np.isnan(sj))) if sj is not None else -1
            print(f"  frame {frame_idx}: smplx_joints NaN={nan_count}")

    # Check all aggregate results
    all_nan_frames = []
    for i, item in enumerate(retargeter.aggregate):
        bp = item.get("body_pose")
        if bp is not None and np.any(np.isnan(bp)):
            all_nan_frames.append(item.get("frame_index", i))

    if all_nan_frames:
        print(f"\nNaN frames: {all_nan_frames[:10]}... ({len(all_nan_frames)} total)")
    else:
        print(f"\nAll {len(retargeter.aggregate)} frames valid (0 NaN)!")


if __name__ == "__main__":
    main()
