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
