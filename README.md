# Realtime Optical Capture

## 线下测试流程

所有命令都在仓库根目录执行：

```bash
cd /home/hmmmocap/workspace/realtime_optical_capture
source .venv/bin/activate
```

### 1. Prepare：相机预览与曝光增益配置

`prepare` 用于枚举 MVS 相机、打开多相机预览、调整曝光/增益，并保存后续阶段使用的 `capture_config.yaml`。

```bash
roc prepare \
  --pixel-format BayerRG8 \
  --fps 5 \
  --preview-scale 0.5
```

窗口快捷键：

```text
1-9  选择相机
[    减小曝光
]    增大曝光
-    减小增益
=    增大增益
s    保存并退出
q    不保存退出
```

保存后会生成：

```text
sessions/prepare_YYYYmmdd_HHMMSS/
  capture_config.yaml
  preview_snapshot/
  logs/
```

### 2. Calib：采集 ChArUco 并求解标定

线下测试推荐直接使用 `capture+solve`，采集完成后立即求解标定。采集时让 ChArUco 板覆盖完整动捕空间，包括边缘、高低、远近和不同倾角。

```bash
roc calib \
  --mode capture+solve \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS \
  --fps 3 \
  --frames 120 \
  --show-preview
```

成功后会生成：

```text
sessions/calib_YYYYmmdd_HHMMSS/
  videos/<serial>.mp4
  calibration.toml
  calibration.yaml
  calibration_report.yaml
  calibration_visualization.png
  charuco_overlays/
```

如果采集完成但求解失败，可以只重跑求解：

```bash
roc calib \
  --mode solve-only \
  --calib-session sessions/calib_YYYYmmdd_HHMMSS
```

### 3. Mocap：实时姿态估计

真实 MVS 实时采集时不要传 `--offline-source-dir`。`--mocap-session` 是唯一输出目录；真 MVS `realtime` 会在目录不存在时自动创建。

基础 3D 姿态估计：

```bash
roc mocap \
  --mode realtime \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS \
  --calib-session sessions/calib_YYYYmmdd_HHMMSS \
  --mocap-session sessions/mocap_live_20260603_test01 \
  --frames 300 \
  --inference-device gpu \
  --model-complexity 2 \
  --profile
```

body-only SMPL-X realtime tracking：

```bash
roc mocap \
  --mode realtime \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS \
  --calib-session sessions/calib_YYYYmmdd_HHMMSS \
  --mocap-session sessions/mocap_live_20260603_test01 \
  --frames 300 \
  --inference-device gpu \
  --model-complexity 2 \
  --no-hands \
  --retarget \
  --retarget-mode track \
  --retarget-model-dir models/smplx \
  --retarget-pose-steps 18 \
  --profile
```

高质量 SMPL-X fitting 基线，建议只跑短序列做诊断：

```bash
roc mocap \
  --mode realtime \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS \
  --calib-session sessions/calib_YYYYmmdd_HHMMSS \
  --mocap-session sessions/mocap_live_20260603_fit_test \
  --frames 50 \
  --inference-device gpu \
  --model-complexity 2 \
  --no-hands \
  --retarget \
  --retarget-mode fit \
  --retarget-model-dir models/smplx \
  --profile
```

### 4. 结果检查

mocap 输出都在指定的 `--mocap-session` 内：

```text
sessions/mocap_live_*/
  videos/
  mocap_*.npz
  mocap_report.yaml
  overlay_videos/combined_2d_overlay.mp4
  reprojection_videos/combined_3d_reprojection_overlay_points_3d.mp4
  reprojection_videos/combined_smplx_reprojection_overlay.mp4
  pose_videos/mocap_3d_pose.mp4
  smplx_retarget/smplx_fit_sequence.npz
  logs/mocap.log
```

查看最新日志：

```bash
tail -n 80 sessions/mocap_live_20260603_test01/logs/mocap.log
```

重要 profile 字段：

```text
mocap_loop_total    单帧主循环总耗时
estimate_only       2D 检测、三角化和 3D 后处理耗时
smplx_retarget      SMPL-X track/fit 耗时
body_err            track 的 body fitting error
```

推荐上机顺序：

1. 运行 `prepare`，保存相机曝光和增益。
2. 运行 `calib capture+solve`，确认生成 `calibration.yaml`。
3. 先运行不带 retarget 的 realtime mocap，检查 2D overlay 和 3D reprojection。
4. 再运行 `--retarget-mode track --retarget-pose-steps 18`，检查 SMPL-X overlay。
5. 只有需要质量基线或诊断疑难帧时，才运行短序列 `fit`。

## 系统概览

本仓库是一个固定多视角光学姿态捕捉 Python 包。系统使用工业 MVS 相机采集多路视频，使用 MediaPipe 做 2D landmark 检测，结合多相机标定完成 3D 三角化，并可选将 3D keypoints 重定向到 SMPL-X。

主要阶段：

- `prepare`：相机发现、实时预览、曝光/增益调节、导出采集配置。
- `calib`：ChArUco 标定视频采集和多相机标定求解。
- `mocap`：实时或离线多视角姿态估计、overlay 渲染、3D 姿态视频生成，以及可选 SMPL-X retarget。

## 项目结构

```text
roc/
  cli.py                  # roc 命令行入口
  prepare/                # prepare 阶段
  calib/                  # calibration 阶段
  mocap/                  # mocap、overlay、retarget、SMPL-X tracking
  mvs/                    # MVS SDK 相机封装和离线相机源
  tracking/               # MediaPipe tracking 辅助逻辑
  triangulation/          # 标定加载和 3D 三角化
  config/, io/            # YAML、session、视频工具
models/
  mediapipe/              # MediaPipe task 文件
  smplx/                  # SMPL-X 模型文件
sessions/                 # 运行时输出
docs/                     # 技术说明
```

## 环境要求

当前默认环境：

- Python 虚拟环境位于 `.venv`
- MVS SDK 安装在 `/opt/MVS/`
- `ffmpeg` 在 `PATH` 中可用
- MediaPipe 模型文件位于 `models/mediapipe/`
- 使用 `--retarget` 时，SMPL-X 模型文件位于 `models/smplx/`

如果 `.venv` 不存在：

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv venv --python 3.10 .venv
env UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python -e .
```

## 更多文档

- `PREPARE.md`：prepare 阶段详细操作说明。
- `CALIB.md`：标定采集和求解说明。
- `MOCAP.md`：mocap 模式、输出、retarget 和 profile 说明。
- `docs/track_technical.md`：SMPL-X track 模式实现和验证记录。
- `docs/track_plan.md`：body-only realtime tracking 路线计划。
