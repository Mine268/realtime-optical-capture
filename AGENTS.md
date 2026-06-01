# Repository Guidelines

## Project Structure & Module Organization

This is a Python package for fixed multi-view optical pose capture. Source lives under `roc/`, organized by pipeline stage and shared services:

- `roc/cli.py`: `roc` command-line entry point.
- `roc/prepare/`, `roc/calib/`, `roc/mocap/`: prepare, calibration, and mocap pipeline stages.
- `roc/mvs/`: MVS SDK camera wrapper plus offline camera sources.
- `roc/tracking/`: MediaPipe tracking and model path handling.
- `roc/triangulation/`: calibration loading and 3D triangulation.
- `roc/config/` and `roc/io/`: configs, YAML, sessions, and video helpers.

Docs are in root Markdown files (`PREPARE.md`, `CALIB.md`, `MOCAP.md`) and `docs/`. MediaPipe task files belong in `models/mediapipe/`. Runtime outputs go to timestamped `sessions/<stage>_YYYYmmdd_HHMMSS/` directories.

## Build, Test, and Development Commands

- `env UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python -e .`: install the package in editable mode.
- `roc <command> ...`: run the installed CLI.
- `python -m roc.cli <command> ...`: run the CLI directly from source.
- `roc prepare --pixel-format BayerRG8`: create a prepare session.
- `roc calib --prepare sessions/prepare_YYYYmmdd_HHMMSS`: capture/solve calibration.
- `roc mocap --prepare ... --calib ...`: run mocap.

Use `--offline-source-dir` where supported to test camera workflows against recorded videos or image folders.

## Coding Style & Naming Conventions

Use Python 3.10+ idioms, 4-space indentation, type hints for public functions, and dataclasses for structured configuration. Keep module names lowercase with underscores. Prefer names such as `capture_config`, `calibration_session`, and `points_3d`. Do not introduce formatting churn; this project currently has no configured formatter, linter, or type checker.

## Testing Guidelines

There is no committed test suite yet. For pure functions, add focused tests under `tests/` using `test_<module>.py` naming. For hardware-facing changes, verify with offline sources when possible and document the command used. At minimum, run `python -m roc.cli --help` after package-level edits.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, for example `Add GPU delegate and offline MVS source` and `Implement calibration capture and solve`. Follow that style: one concise sentence, capitalized verb first, no trailing period.

Pull requests should describe the affected pipeline stage, list verification commands, and call out hardware, MVS SDK, MediaPipe model, GPU delegate, or `ffmpeg` assumptions. Include screenshots or sample output paths when previews, overlays, or generated session artifacts change.

## Security & Configuration Tips

Do not commit generated `sessions/` data, local logs, SDK binaries, or machine-specific paths beyond documented defaults. MVS SDK is expected under `/opt/MVS/`, `ffmpeg` must be on `PATH`, and MediaPipe task assets belong in `models/mediapipe/`.
