from __future__ import annotations

# PYTHON_ARGCOMPLETE_OK

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Run camera prepare stage")
    prepare_parser.add_argument(
        "--session-root",
        type=Path,
        default=Path("sessions"),
        help="Root directory for generated sessions",
    )
    prepare_parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Software trigger preview fps",
    )
    prepare_parser.add_argument(
        "--serial",
        dest="serials",
        action="append",
        default=[],
        help="Restrict to a camera serial, may be repeated",
    )
    prepare_parser.add_argument(
        "--pixel-format",
        default="BayerRG8",
        help="Preferred pixel format",
    )
    prepare_parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help="Preview scaling factor",
    )
    prepare_parser.add_argument(
        "--window-name",
        default="ROC Prepare",
        help="OpenCV preview window name",
    )

    calib_parser = subparsers.add_parser("calib", help="Run camera calibration stage")
    calib_parser.add_argument(
        "--mode",
        default="capture+solve",
        choices=["capture-only", "solve-only", "capture+solve"],
        help="Calibration stage mode",
    )
    calib_parser.add_argument(
        "--prepare-session",
        type=Path,
        help="Path to a prepare session directory",
    )
    calib_parser.add_argument(
        "--calib-session",
        type=Path,
        help="Path to an existing calibration session directory for solve-only mode",
    )
    calib_parser.add_argument("--session-root", type=Path, default=Path("sessions"), help="Root directory for generated sessions")
    calib_parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Calibration capture fps",
    )
    calib_parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of frames to capture",
    )
    calib_parser.add_argument(
        "--world-mode",
        default="camera0",
        choices=["camera0", "ground"],
        help="Calibration world coordinate mode",
    )
    calib_parser.add_argument(
        "--square-length-mm",
        type=float,
        default=190.5,
        help="Charuco square length in millimeters",
    )
    calib_parser.add_argument(
        "--marker-length-mm",
        type=float,
        default=152.4,
        help="Charuco marker length in millimeters",
    )
    calib_parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show live preview during calibration capture",
    )

    mocap_parser = subparsers.add_parser("mocap", help="Run motion capture stage")
    mocap_parser.add_argument(
        "--mode",
        default="realtime",
        choices=["realtime", "capture", "capture_estimate"],
        help="Mocap stage mode",
    )
    mocap_parser.add_argument(
        "--prepare-session",
        type=Path,
        required=True,
        help="Path to a prepare session directory",
    )
    mocap_parser.add_argument(
        "--calib-session",
        type=Path,
        required=True,
        help="Path to a calibration session directory",
    )
    mocap_parser.add_argument(
        "--mocap-session",
        type=Path,
        required=True,
        help="Path to the mocap session directory used for all mocap inputs and outputs",
    )
    mocap_parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Motion capture fps",
    )
    mocap_parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Number of synchronized frame sets to capture, 0 means unlimited until q",
    )
    mocap_parser.add_argument(
        "--no-hands",
        action="store_true",
        help="Disable hand landmarks",
    )
    mocap_parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe pose model complexity selector: 1 uses full, 2 uses heavy",
    )
    mocap_parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show live mocap preview",
    )
    mocap_parser.add_argument(
        "--no-record-videos",
        action="store_true",
        help="Skip realtime video recording and end-of-session overlay rendering; useful for latency profiling",
    )
    mocap_parser.add_argument(
        "--video-dir",
        type=Path,
        help="Optional directory containing per-camera mp4 files for capture_estimate mode; defaults to <mocap-session>/videos/",
    )
    mocap_parser.add_argument(
        "--postprocess-mode",
        default="offline",
        choices=["offline", "realtime"],
        help="Postprocess mode for capture_estimate: offline uses zero-phase batch filtering, realtime uses causal online filtering",
    )
    mocap_parser.add_argument(
        "--inference-device",
        default="cpu",
        choices=["cpu", "gpu", "cuda"],
        help="Inference device for MediaPipe and SMPL-X retarget; gpu/cuda uses MediaPipe GPU and Torch CUDA",
    )
    mocap_parser.add_argument(
        "--offline-source-dir",
        type=Path,
        help="Use recorded files as an MVS-like camera source for realtime/capture testing",
    )
    mocap_parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-frame mocap estimate and retarget timings to stdout and the mocap log",
    )
    mocap_parser.add_argument(
        "--retarget",
        action="store_true",
        help="After 3D keypoints are saved, retarget them to SMPL-X joint rotations",
    )
    mocap_parser.add_argument(
        "--retarget-mode",
        default="fit",
        choices=["fit", "track"],
        help="SMPL-X retargeting mode: fit uses full optimization, track uses body-only fast tracking",
    )
    mocap_parser.add_argument(
        "--retarget-model-dir",
        type=Path,
        default=Path("models/smplx"),
        help="SMPL-X model directory used by --retarget",
    )
    mocap_parser.add_argument(
        "--retarget-vposer-dir",
        type=Path,
        help="Optional VPoser directory used when --retarget-use-vposer is enabled",
    )
    mocap_parser.add_argument(
        "--retarget-max-frames",
        type=int,
        default=-1,
        help="Maximum number of frames to retarget, -1 means all saved 3D frames",
    )
    mocap_parser.add_argument(
        "--retarget-frame-step",
        type=int,
        default=1,
        help="Retarget every Nth saved 3D frame",
    )
    mocap_parser.add_argument(
        "--retarget-input-scale",
        type=float,
        default=0.001,
        help="Scale applied to mocap points before SMPL-X fitting; ROC calibration is millimeters, SMPL-X uses meters",
    )
    mocap_parser.add_argument(
        "--retarget-betas-steps",
        type=int,
        default=80,
        help="SMPL-X shared shape optimization steps for --retarget",
    )
    mocap_parser.add_argument(
        "--retarget-pose-steps",
        type=int,
        default=120,
        help="Per-frame SMPL-X pose optimization steps for --retarget",
    )
    mocap_parser.add_argument(
        "--retarget-root-steps",
        type=int,
        help="Override per-frame root alignment optimization steps for --retarget",
    )
    mocap_parser.add_argument(
        "--retarget-lower-steps",
        type=int,
        help="Override lower-body refinement steps for --retarget",
    )
    mocap_parser.add_argument(
        "--retarget-no-lower-refine",
        action="store_true",
        help="Skip the extra lower-body refinement stage during --retarget",
    )
    mocap_parser.add_argument(
        "--retarget-early-stop-check-interval",
        type=int,
        default=1,
        help="Check pose-stage early stopping every N steps; higher values reduce CUDA synchronization",
    )
    mocap_parser.add_argument(
        "--retarget-temporal-weight",
        type=float,
        default=0.0,
        help="Weight for matching realtime retarget pose/root to the previous frame",
    )
    mocap_parser.add_argument(
        "--retarget-velocity-weight",
        type=float,
        default=0.0,
        help="Weight for damping realtime retarget root velocity changes",
    )
    mocap_parser.add_argument(
        "--retarget-acceleration-weight",
        type=float,
        default=0.002,
        help="Weight for damping realtime retarget acceleration changes",
    )
    mocap_parser.add_argument(
        "--retarget-no-adaptive-root",
        action="store_true",
        help="Disable realtime adaptive root steps and use --retarget-root-steps behavior for every frame",
    )
    mocap_parser.add_argument(
        "--retarget-realtime-root-steps",
        type=int,
        default=2,
        help="Root alignment steps for stable realtime retarget frames",
    )
    mocap_parser.add_argument(
        "--retarget-realtime-root-recovery-steps",
        type=int,
        help="Root alignment steps for realtime init, turn, translation, or high-error recovery frames",
    )
    mocap_parser.add_argument(
        "--retarget-realtime-root-error-threshold",
        type=float,
        default=0.12,
        help="Body mean error in meters above which realtime retarget uses recovery root steps",
    )
    mocap_parser.add_argument(
        "--retarget-realtime-root-turn-threshold",
        type=float,
        default=18.0,
        help="Hip-axis turn angle in degrees above which realtime retarget uses recovery root steps",
    )
    mocap_parser.add_argument(
        "--retarget-realtime-root-translation-threshold",
        type=float,
        default=0.20,
        help="Hip-center translation in meters above which realtime retarget uses recovery root steps",
    )
    mocap_parser.add_argument(
        "--retarget-use-vposer",
        action="store_true",
        help="Use VPoser body pose prior during --retarget",
    )
    mocap_parser.add_argument(
        "--retarget-hands",
        action="store_true",
        help="Also optimize SMPL-X hand pose during --retarget; by default only the body is optimized",
    )
    mocap_parser.add_argument(
        "--retarget-save-debug-assets",
        action="store_true",
        help="Save per-frame SMPL-X obj/png debug assets during --retarget",
    )

    return parser


def _enable_argcomplete(parser: argparse.ArgumentParser) -> None:
    try:
        import argcomplete
    except ImportError:
        return
    argcomplete.autocomplete(parser)


def _mediapipe_delegate_from_inference_device(device: str) -> str:
    return "gpu" if device in {"gpu", "cuda"} else "cpu"


def _retarget_device_from_inference_device(device: str) -> str:
    return "cuda" if device in {"gpu", "cuda"} else "cpu"


def main() -> None:
    parser = build_parser()
    _enable_argcomplete(parser)
    args = parser.parse_args()

    if args.command == "prepare":
        from roc.prepare.app import run_prepare

        run_prepare(
            session_root=args.session_root,
            fps=args.fps,
            serials=args.serials,
            pixel_format=args.pixel_format,
            preview_scale=args.preview_scale,
            window_name=args.window_name,
        )
        return

    if args.command == "calib":
        from roc.calib.capture import run_calibration_capture

        run_calibration_capture(
            mode=args.mode,
            prepare_session=args.prepare_session,
            calib_session=args.calib_session,
            session_root=args.session_root,
            fps=args.fps,
            frames=args.frames,
            world_mode=args.world_mode,
            square_length_mm=args.square_length_mm,
            marker_length_mm=args.marker_length_mm,
            show_preview=args.show_preview,
        )
        return

    if args.command == "mocap":
        mediapipe_delegate = _mediapipe_delegate_from_inference_device(args.inference_device)
        retarget_device = _retarget_device_from_inference_device(args.inference_device)
        retarget_config = None
        if args.retarget:
            if args.mode == "capture":
                parser.error("--retarget requires 3D keypoints and is only supported for realtime or capture_estimate")
            if args.retarget_frame_step < 1:
                parser.error("--retarget-frame-step must be >= 1")
            if args.retarget_pose_steps < 1:
                parser.error("--retarget-pose-steps must be >= 1")
            if args.retarget_betas_steps < 1:
                parser.error("--retarget-betas-steps must be >= 1")
            if args.retarget_root_steps is not None and args.retarget_root_steps < 1:
                parser.error("--retarget-root-steps must be >= 1")
            if args.retarget_lower_steps is not None and args.retarget_lower_steps < 1:
                parser.error("--retarget-lower-steps must be >= 1")
            if args.retarget_early_stop_check_interval < 1:
                parser.error("--retarget-early-stop-check-interval must be >= 1")
            if args.retarget_temporal_weight < 0.0:
                parser.error("--retarget-temporal-weight must be >= 0")
            if args.retarget_velocity_weight < 0.0:
                parser.error("--retarget-velocity-weight must be >= 0")
            if args.retarget_acceleration_weight < 0.0:
                parser.error("--retarget-acceleration-weight must be >= 0")
            if args.retarget_realtime_root_steps < 1:
                parser.error("--retarget-realtime-root-steps must be >= 1")
            if (
                args.retarget_realtime_root_recovery_steps is not None
                and args.retarget_realtime_root_recovery_steps < 1
            ):
                parser.error("--retarget-realtime-root-recovery-steps must be >= 1")
            if args.retarget_realtime_root_error_threshold <= 0.0:
                parser.error("--retarget-realtime-root-error-threshold must be > 0")
            if args.retarget_realtime_root_turn_threshold <= 0.0:
                parser.error("--retarget-realtime-root-turn-threshold must be > 0")
            if args.retarget_realtime_root_translation_threshold <= 0.0:
                parser.error("--retarget-realtime-root-translation-threshold must be > 0")
            from roc.mocap.retarget import RetargetConfig, RetargetMode
            retarget_mode = RetargetMode(args.retarget_mode)
            track_pose_steps = min(args.retarget_pose_steps, 20) if retarget_mode == RetargetMode.TRACK else 20
            track_recovery_pose_steps = (
                max(track_pose_steps, min(60, max(8, track_pose_steps * 4)))
                if retarget_mode == RetargetMode.TRACK
                else 60
            )

            retarget_config = RetargetConfig(
                model_dir=args.retarget_model_dir,
                mode=retarget_mode,
                vposer_dir=args.retarget_vposer_dir,
                device=retarget_device,
                betas_steps=args.retarget_betas_steps,
                pose_steps=args.retarget_pose_steps,
                root_steps=args.retarget_root_steps,
                lower_steps=args.retarget_lower_steps,
                lower_body_refine=not args.retarget_no_lower_refine,
                early_stop_check_interval=args.retarget_early_stop_check_interval,
                temporal_weight=args.retarget_temporal_weight,
                velocity_weight=args.retarget_velocity_weight,
                acceleration_weight=args.retarget_acceleration_weight,
                realtime_adaptive_root=not args.retarget_no_adaptive_root,
                realtime_root_steps=args.retarget_realtime_root_steps,
                realtime_root_recovery_steps=args.retarget_realtime_root_recovery_steps,
                realtime_root_error_threshold_m=args.retarget_realtime_root_error_threshold,
                realtime_root_turn_threshold_deg=args.retarget_realtime_root_turn_threshold,
                realtime_root_translation_threshold_m=args.retarget_realtime_root_translation_threshold,
                frame_step=args.retarget_frame_step,
                max_frames=args.retarget_max_frames,
                input_scale=args.retarget_input_scale,
                optimize_hands=args.retarget_hands,
                use_vposer=args.retarget_use_vposer,
                save_debug_assets=args.retarget_save_debug_assets,
                profile=args.profile,
                profile_interval=1,
                track_pose_steps=track_pose_steps,
                track_temporal_weight=args.retarget_temporal_weight,
                track_velocity_weight=args.retarget_velocity_weight,
                track_acceleration_weight=args.retarget_acceleration_weight,
                track_recovery_pose_steps=track_recovery_pose_steps,
            )

        if args.mode == "realtime":
            from roc.mocap.realtime import run_mocap_realtime

            run_mocap_realtime(
                prepare_session=args.prepare_session,
                calib_session=args.calib_session,
                mocap_session=args.mocap_session,
                fps=args.fps,
                max_frames=args.frames,
                hands_enabled=not args.no_hands,
                model_complexity=args.model_complexity,
                show_preview=args.show_preview,
                delegate=mediapipe_delegate,
                offline_source_dir=args.offline_source_dir,
                retarget_config=retarget_config,
                record_videos=not args.no_record_videos,
                profile=args.profile,
            )
            return

        if args.mode == "capture_estimate":
            from roc.mocap.offline import run_mocap_offline

            run_mocap_offline(
                prepare_session=args.prepare_session,
                calib_session=args.calib_session,
                video_dir=args.video_dir,
                mocap_session=args.mocap_session,
                max_frames=args.frames,
                hands_enabled=not args.no_hands,
                model_complexity=args.model_complexity,
                show_preview=args.show_preview,
                postprocess_mode=args.postprocess_mode,
                delegate=mediapipe_delegate,
                retarget_config=retarget_config,
                profile=args.profile,
            )
            return

        if args.mode == "capture":
            from roc.mocap.capture import run_mocap_capture

            run_mocap_capture(
                prepare_session=args.prepare_session,
                calib_session=args.calib_session,
                mocap_session=args.mocap_session,
                fps=args.fps,
                max_frames=args.frames,
                show_preview=args.show_preview,
                offline_source_dir=args.offline_source_dir,
            )
            return

        parser.error(f"Unsupported mocap mode: {args.mode}")
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
