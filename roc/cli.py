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
        "--session-root",
        type=Path,
        default=Path("sessions"),
        help="Root directory for generated sessions",
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
        "--video-dir",
        type=Path,
        help="Optional directory containing per-camera mp4 files for capture_estimate mode; defaults to calib session videos/",
    )
    mocap_parser.add_argument(
        "--postprocess-mode",
        default="offline",
        choices=["offline", "realtime"],
        help="Postprocess mode for capture_estimate: offline uses zero-phase batch filtering, realtime uses causal online filtering",
    )
    mocap_parser.add_argument(
        "--delegate",
        default="cpu",
        choices=["cpu", "gpu"],
        help="MediaPipe inference delegate",
    )
    mocap_parser.add_argument(
        "--offline-source-dir",
        type=Path,
        help="Use recorded files as an MVS-like camera source for realtime/capture testing",
    )

    return parser


def _enable_argcomplete(parser: argparse.ArgumentParser) -> None:
    try:
        import argcomplete
    except ImportError:
        return
    argcomplete.autocomplete(parser)


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
        if args.mode == "realtime":
            from roc.mocap.realtime import run_mocap_realtime

            run_mocap_realtime(
                prepare_session=args.prepare_session,
                calib_session=args.calib_session,
                session_root=args.session_root,
                fps=args.fps,
                max_frames=args.frames,
                hands_enabled=not args.no_hands,
                model_complexity=args.model_complexity,
                show_preview=args.show_preview,
                delegate=args.delegate,
                offline_source_dir=args.offline_source_dir,
            )
            return

        if args.mode == "capture_estimate":
            from roc.mocap.offline import run_mocap_offline

            run_mocap_offline(
                prepare_session=args.prepare_session,
                calib_session=args.calib_session,
                video_dir=args.video_dir,
                session_root=args.session_root,
                max_frames=args.frames,
                hands_enabled=not args.no_hands,
                model_complexity=args.model_complexity,
                show_preview=args.show_preview,
                postprocess_mode=args.postprocess_mode,
                delegate=args.delegate,
            )
            return

        if args.mode == "capture":
            from roc.mocap.capture import run_mocap_capture

            run_mocap_capture(
                prepare_session=args.prepare_session,
                calib_session=args.calib_session,
                session_root=args.session_root,
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
