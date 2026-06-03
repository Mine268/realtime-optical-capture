"""Diagnose GPU NaN in fit mode by tracing tensor values through optimization."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


def main() -> None:
    # Minimal setup matching run_mocap_retarget
    npz_path = Path("sessions/mocap_live_rkw_20260603_test01/mocap_live_rkw_20260603_test01.npz")
    from roc.mocap.retarget import _ensure_retarget_dependencies, _load_reference_fitter
    from roc.mocap.retarget import _load_roc_sequence, _build_reference_args, RetargetConfig, RetargetMode

    _ensure_retarget_dependencies(use_vposer=False)
    fitter = _load_reference_fitter()

    config = RetargetConfig(
        model_dir=Path("models/smplx"),
        mode=RetargetMode.FIT,
        device="cuda",
        pose_steps=10,  # fewer steps for fast diagnosis
        betas_steps=5,
        lower_body_refine=True,
        optimize_hands=False,
        use_vposer=False,
    )
    output_dir = Path("/tmp/smplx_nan_diag")
    output_dir.mkdir(parents=True, exist_ok=True)
    from roc.mocap.retarget import _resolve_smplx_model_dir
    model_dir = _resolve_smplx_model_dir(Path("models/smplx"))
    print(f"Resolved model_dir: {model_dir}")
    args = _build_reference_args(config, output_dir, model_dir)
    print(f"Device arg: {args.device}")

    T = fitter.torch
    device = T.device("cuda" if args.device == "cuda" and T.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load sequence
    sequence = _load_roc_sequence(npz_path, input_scale=config.input_scale)
    print(f"Sequence body: shape={sequence.body.shape}, NaN={T.isnan(T.from_numpy(sequence.body)).sum().item()}")

    # Check input data on GPU
    body_t = T.from_numpy(sequence.body[0:1].copy()).float().to(device)
    print(f"Input body tensor (GPU): shape={body_t.shape}, NaN={T.isnan(body_t).sum().item()}, "
          f"min={body_t.min().item():.4f}, max={body_t.max().item():.4f}, "
          f"finite={T.isfinite(body_t).all().item()}")

    # Create model
    joint_name_to_idx = fitter.get_joint_name_to_index()
    model = fitter.create_model(args.model_dir, args.gender, args.num_betas, args.num_pca_comps, device)
    print(f"Model created on device: {next(model.parameters()).device}")

    # Test model forward pass with zero params
    betas = T.zeros((1, args.num_betas), device=device)
    bp = T.zeros((1, 63), device=device)
    go = T.zeros((1, 3), device=device)
    tr = T.zeros((1, 3), device=device)
    lh = T.zeros((1, 12), device=device)
    rh = T.zeros((1, 12), device=device)

    with T.no_grad():
        out = model(betas=betas, body_pose=bp, global_orient=go, transl=tr,
                     left_hand_pose=lh, right_hand_pose=rh, return_verts=False)
    joints = out.joints
    print(f"\nZero-param forward: joints shape={joints.shape}, NaN={T.isnan(joints).sum().item()}, "
          f"finite={T.isfinite(joints).all().item()}")

    # Test with small random params
    bp = T.randn(1, 63, device=device) * 0.01
    go = T.randn(1, 3, device=device) * 0.01
    tr = T.randn(1, 3, device=device) * 0.01
    with T.no_grad():
        out = model(betas=betas, body_pose=bp, global_orient=go, transl=tr,
                     left_hand_pose=lh, right_hand_pose=rh, return_verts=False)
    joints = out.joints
    print(f"Small-param forward: joints shape={joints.shape}, NaN={T.isnan(joints).sum().item()}, "
          f"finite={T.isfinite(joints).all().item()}, min={joints.min().item():.4f}, max={joints.max().item():.4f}")

    # Now trace through fit_single_frame for frame 0
    print("\n=== Running fit_single_frame on GPU (frame 0) ===")
    shared_betas = fitter.optimize_shared_betas(model, None, sequence, args, joint_name_to_idx, device)
    print(f"Shared betas: shape={shared_betas.shape}, NaN={T.isnan(shared_betas).sum().item()}")

    # Patch fit_single_frame to trace NaN
    _orig_fit_single = fitter.fit_single_frame

    def _traced_fit_single(*fargs, **fkwargs):
        print("  fit_single_frame called, tracing...")
        # Try the original
        result = _orig_fit_single(*fargs, **fkwargs)
        print(f"  Result keys: {list(result.keys())}")
        for k in ["global_orient", "body_pose", "transl", "betas"]:
            if k in result:
                val = result[k]
                nan_count = T.isnan(T.from_numpy(val.copy())).sum().item()
                print(f"  {k}: shape={val.shape}, NaN={nan_count}")
        return result

    fitter.fit_single_frame = _traced_fit_single

    result = fitter.fit_single_frame(
        model, None, 0, sequence, shared_betas, args,
        joint_name_to_idx, device, init_state=None,
        prev_state=None, prev_prev_state=None,
    )

    print(f"\n=== Final result ===")
    for k in ["global_orient", "body_pose", "transl"]:
        val = result[k]
        nan_count = int(np.sum(np.isnan(val)))
        print(f"  {k}: shape={val.shape}, NaN={nan_count}/{val.size}")


if __name__ == "__main__":
    main()
