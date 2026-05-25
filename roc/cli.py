from __future__ import annotations

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

    return parser


def main() -> None:
    parser = build_parser()
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

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
