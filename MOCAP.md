# Mocap Stage 操作说明

`roc mocap` 现在只围绕一个显式指定的 mocap session 工作。三种模式都必须传入 `--mocap-session`，并且不会自动创建新的 `sessions/mocap_YYYYmmdd_HHMMSS/` 目录。

```bash
roc mocap --mode ... \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS \
  --calib-session sessions/calib_YYYYmmdd_HHMMSS \
  --mocap-session sessions/mocap_test
```

除真 MVS `realtime` 外，`--mocap-session` 指向的目录必须已经存在。真 MVS `realtime` 没有传 `--offline-source-dir` 时，如果目录不存在会自动创建该目录。所有 mocap 输出都写在这个目录内部。

## 前提条件

先进入项目环境：

```bash
source .venv/bin/activate
```

确认 MediaPipe 模型文件存在：

```text
models/mediapipe/
  pose_landmarker_lite.task
  pose_landmarker_full.task
  pose_landmarker_heavy.task
  hand_landmarker.task
```

## 三种模式

### capture

只采集多相机视频，不做姿态估计。输出写入 `--mocap-session`：

```bash
roc mocap \
  --mode capture \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --mocap-session sessions/mocap_test \
  --frames 300 \
  --show-preview
```

主要产物：

```text
sessions/mocap_test/
  videos/<serial>.mp4
  raw_frames/<serial>/frame_*.bmp
  logs/mocap.log
  mocap_report.yaml
```

### capture_estimate

读取已有多相机 mp4，做离线姿态估计。默认读取 `--mocap-session/videos/*.mp4`：

```bash
roc mocap \
  --mode capture_estimate \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --mocap-session sessions/mocap_test \
  --frames 200 \
  --delegate gpu \
  --model-complexity 2
```

如需读取其他视频目录，可以显式传 `--video-dir`，但输出仍然写回 `--mocap-session`：

```bash
--video-dir sessions/mocap_test/videos
```

`--video-dir` 只在 `capture_estimate` 模式下使用。

### realtime

实时采集并估计姿态。使用真 MVS 时，如果 `--mocap-session` 不存在会自动创建；视频和估计结果都写入该目录：

```bash
roc mocap \
  --mode realtime \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --mocap-session sessions/mocap_test \
  --frames 200 \
  --delegate gpu \
  --model-complexity 2
```

不连接相机测试 realtime/capture 链路时，用 `--offline-source-dir` 指定输入源：

```bash
--offline-source-dir sessions/mocap_test/videos
```

该参数只是输入源，不决定输出目录。

`realtime` 模式不使用 `--video-dir`。真 MVS realtime 不需要视频输入目录；Fake MVS realtime 只使用 `--offline-source-dir` 作为输入源。

## 姿态估计输出

`realtime` 和 `capture_estimate` 都会在 `--mocap-session` 内生成或覆盖：

```text
capture_config.yaml
calibration.yaml
mocap_estimate_config.yaml      # capture_estimate
mocap_config.yaml               # realtime/capture
mocap_<session_name>.npz
mocap_report.yaml
overlay_videos/
  <serial>_2d_overlay.mp4
  combined_2d_overlay.mp4
reprojection_videos/
  <serial>_3d_reprojection_overlay_points_3d.mp4
  combined_3d_reprojection_overlay_points_3d.mp4
  <serial>_smplx_reprojection_overlay.mp4        # --retarget
  combined_smplx_reprojection_overlay.mp4        # --retarget
pose_videos/
  mocap_3d_pose.mp4
logs/mocap.log
```

`mocap_*.npz` 包含：

- `timestamps`
- `camera_serials`
- `pose_2d`, `pose_confidence`
- `left_hand_2d`, `left_hand_confidence`
- `right_hand_2d`, `right_hand_confidence`
- `bboxes_2d`
- `points_3d`, `points_3d_raw`
- `reprojection_error`
- `landmark_names`

## SMPL-X Retarget

`realtime` 和 `capture_estimate` 可以加 `--retarget`：

- `realtime`：每一帧完成 3D 三角化和 causal 后处理后，立刻进行 SMPL-X retarget。session 结束时只汇总保存已经逐帧产生的 SMPL-X 参数。
- `capture_estimate`：先处理完整视频，完成所有帧的 3D 姿态估计和离线后处理后，再对完整序列进行 retarget。

Retarget 需要 SMPL-X 依赖和模型文件。默认读取 `models/smplx`；也可以用 `--retarget-model-dir models` 指向包含 `smplx/` 子目录的父目录。只有启用 `--retarget-use-vposer` 时才需要 `--retarget-vposer-dir`。

```bash
roc mocap \
  --mode capture_estimate \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --mocap-session sessions/mocap_test \
  --frames 200 \
  --retarget \
  --retarget-model-dir models/smplx
```

输出仍在同一个 mocap session 内：

```text
sessions/mocap_test/
  smplx_retarget/
    smplx_fit_sequence.npz
    retarget_report.yaml
    trajectory_names.json
    per_frame/frame_000000.npz
    per_frame/frame_000000.pkl
```

`smplx_fit_sequence.npz` 包含 `global_orient`、`body_pose`、`left_hand_pose`、`right_hand_pose`、`transl`、`betas`、`smplx_joints` 和误差指标。默认只优化全身，手部 pose 保持零值；需要手部时再显式加 `--retarget-hands`。`capture` 模式没有 3D keypoints，因此不支持 `--retarget`。

启用 `--retarget` 时，reprojection 阶段还会把 SMPL-X skeleton 投影回原相机视频，额外生成：

```text
reprojection_videos/<serial>_smplx_reprojection_overlay.mp4
reprojection_videos/combined_smplx_reprojection_overlay.mp4
```

常用 retarget 参数：

```bash
--retarget-device cuda          # 使用 CUDA；不可用时会回退到 CPU
--retarget-max-frames 200       # 限制 retarget 帧数，-1 表示全部
--retarget-frame-step 2         # 每隔 N 帧 retarget 一帧
--retarget-input-scale 0.001    # ROC 3D 点默认从毫米转成米
--retarget-pose-steps 120       # 每帧姿态优化步数
--retarget-betas-steps 80       # 共享体型优化步数
--retarget-hands                # 同时优化 SMPL-X 手部 pose
--retarget-save-debug-assets    # 额外保存 obj/png 调试文件
```

## 常用参数

```bash
--frames 300              # 限制帧数，0 表示直到按 q
--show-preview            # 显示 OpenCV 预览
--no-hands                # 关闭手部检测
--model-complexity 0      # lite pose 模型
--model-complexity 1      # full pose 模型
--model-complexity 2      # heavy pose 模型
--delegate cpu            # CPU 推理
--delegate gpu            # GPU delegate
```

输入目录参数只在特定模式生效：

```text
--video-dir
  仅用于 capture_estimate
  逐帧读取 mp4，不模拟相机

--offline-source-dir
  仅用于 realtime/capture
  将视频或图片目录包装成 Fake MVS 相机源
```

`capture_estimate` 默认使用离线后处理：

```text
short-gap interpolation -> zero-phase Butterworth low-pass
```

如需模拟实时后处理：

```bash
--postprocess-mode realtime
```

## 手动重生成诊断视频

2D overlay：

```bash
python -m roc.mocap.render_2d_overlays \
  --npz-path sessions/mocap_test/mocap_test.npz \
  --video-dir sessions/mocap_test/videos \
  --output-dir sessions/mocap_test/overlay_videos
```

3D 重投影 overlay：

```bash
python -m roc.mocap.render_reprojection_overlays \
  --npz-path sessions/mocap_test/mocap_test.npz \
  --calibration-toml sessions/calib_20260525_163831/calibration.toml \
  --video-dir sessions/mocap_test/videos \
  --output-dir sessions/mocap_test/reprojection_videos \
  --points-key points_3d
```

3D pose 预览：

```bash
python -m roc.mocap.render_npz \
  --npz-path sessions/mocap_test/mocap_test.npz \
  --output-path sessions/mocap_test/pose_videos/mocap_3d_pose.mp4
```

## 故障排查

反馈问题时请提供：

- 完整命令
- 终端输出
- `sessions/<mocap-session>/logs/mocap.log`
- 是否生成 `mocap_*.npz`
- 相关 overlay/reprojection/3D pose 视频是否存在
