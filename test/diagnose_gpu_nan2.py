"""Run full GPU batch fit and trace NaN emergence."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from roc.mocap.retarget import (
    RetargetConfig, RetargetMode, run_mocap_retarget,
    _load_roc_sequence, _build_reference_args, _resolve_smplx_model_dir,
    _ensure_retarget_dependencies, _load_reference_fitter,
)


def main() -> None:
    npz_path = Path("sessions/mocap_live_rkw_20260603_test01/mocap_live_rkw_20260603_test01.npz")
    mocap_session = Path("sessions/mocap_live_rkw_20260603_test01")

    config = RetargetConfig(
        model_dir=Path("models/smplx"),
        mode=RetargetMode.FIT,
        output_dir=Path("/tmp/smplx_gpu_diag"),
        device="cuda",
        pose_steps=120,
        betas_steps=80,
        lower_body_refine=True,
        optimize_hands=False,
        use_vposer=False,
        max_frames=20,
        profile=True,
    )

    # Patch to trace NaN per frame
    _ensure_retarget_dependencies(use_vposer=False)
    fitter = _load_reference_fitter()
    T = fitter.torch

    _orig_fit = fitter.fit_single_frame
    nan_found = []

    def _traced_fit(model, vposer, frame_index, sequence, shared_betas, args, joint_name_to_idx, device, **kwargs):
        result = _orig_fit(model, vposer, frame_index, sequence, shared_betas, args, joint_name_to_idx, device, **kwargs)
        for k in ["global_orient", "body_pose", "transl"]:
            v = result.get(k)
            if v is not None and np.any(np.isnan(v)):
                nan_found.append((frame_index, k))
                print(f"  !!! NaN in {k} at frame {frame_index}")
                break
        return result

    fitter.fit_single_frame = _traced_fit

    print("Running GPU batch fit (20 frames)...")
    fit_npz = run_mocap_retarget(npz_path, mocap_session, config)

    data = np.load(fit_npz)
    for k in ["global_orient", "body_pose", "transl", "smplx_joints"]:
        v = data[k]
        print(f"  {k}: shape={v.shape}, NaN={np.sum(np.isnan(v))}/{v.size}")

    if nan_found:
        print(f"\nNaN first detected at frames: {nan_found[:5]}")
    else:
        print("\nNo NaN detected in any frame!")

    # Also check if the NPZ has NaN despite traced function not catching it
    sj = data["smplx_joints"]
    all_nan = np.all(np.isnan(sj))
    print(f"All smplx_joints NaN: {all_nan}")


if __name__ == "__main__":
    main()
