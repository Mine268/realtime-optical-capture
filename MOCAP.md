# Mocap Stage 操作说明

当前 `mocap` 阶段统一使用：

```bash
roc mocap --mode ...
```

支持三种模式：

- `realtime`：实时采集 + 实时姿态估计
- `capture`：只采集多相机 mocap 视频
- `capture_estimate`：基于已录制的多相机 mp4 做离线姿态估计

## 1. 前提条件

进入项目根目录后执行：

```bash
source .venv/bin/activate
```

并确认模型文件存在：

```text
models/mediapipe/
  pose_landmarker_lite.task
  pose_landmarker_full.task
  pose_landmarker_heavy.task
  hand_landmarker.task
```

## 2. 推荐调试方式

推荐先用离线估计模式：

```bash
python -m roc.cli mocap \
  --mode capture_estimate \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS \
  --calib-session sessions/calib_YYYYmmdd_HHMMSS
```

默认会读取：

```text
sessions/calib_YYYYmmdd_HHMMSS/videos/*.mp4
```

## 3. 三种模式的命令

`capture_estimate`：

```bash
python -m roc.cli mocap \
  --mode capture_estimate \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --frames 150
```

`capture`：

```bash
python -m roc.cli mocap \
  --mode capture \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --frames 300 \
  --show-preview
```

`realtime`：

```bash
python -m roc.cli mocap \
  --mode realtime \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --frames 300 \
  --show-preview
```

## 4. 常用参数

限制处理或采集帧数：

```bash
--frames 300
```

显示预览：

```bash
--show-preview
```

关闭手部：

```bash
--no-hands
```

使用 heavy pose 模型：

```bash
--model-complexity 2
```

模型复杂度映射：

```text
--model-complexity 0 -> pose_landmarker_lite.task
--model-complexity 1 -> pose_landmarker_full.task
--model-complexity 2 -> pose_landmarker_heavy.task
```

给 `capture_estimate` 手动指定视频目录。若目录是 `sessions/mocap_*` 或 `sessions/mocap_*/videos`，结果会直接写回这个 mocap session：

```bash
--video-dir sessions/mocap_YYYYmmdd_HHMMSS
```

在 `capture_estimate` 中模拟 realtime 后处理：

```bash
--postprocess-mode realtime
```

选择 MediaPipe delegate：

```bash
--delegate cpu
--delegate gpu
```

使用离线文件作为 MVS-like 相机源测试 `realtime` / `capture`：

```bash
--offline-source-dir sessions/mocap_YYYYmmdd_HHMMSS/videos
```

## 5. 输出说明

`capture` 和 `realtime` 会生成新的：

```text
sessions/mocap_YYYYmmdd_HHMMSS/
```

`capture_estimate` 的输出规则：

- 如果 `--video-dir` 指向 `sessions/mocap_*` 或 `sessions/mocap_*/videos`，则不新建 session，直接写回该 `sessions/mocap_*`
- 如果未指定 `--video-dir` 或指定了普通视频目录，则兼容旧行为，新建 `sessions/mocap_*`

当前目录结构：

```text
sessions/mocap_YYYYmmdd_HHMMSS/
  capture_config.yaml
  calibration.yaml
  mocap_config.yaml
  mocap_estimate_config.yaml
  logs/
    mocap.log
  raw_frames/
    <serial>/
      frame_000000.bmp
  videos/
    <serial>.mp4
  mocap_YYYYmmdd_HHMMSS.npz
  mocap_report.yaml
  overlay_videos/
    <serial>_2d_overlay.mp4
    combined_2d_overlay.mp4
  reprojection_videos/
    <serial>_3d_reprojection_overlay_points_3d.mp4
    combined_3d_reprojection_overlay_points_3d.mp4
```

说明：

- `capture`：主要输出 `videos/*.mp4`
- `capture`：采集阶段先写 `raw_frames/<serial>/frame_*.bmp`，结束后再按真实 fps 转码成 `videos/*.mp4`
- `capture_estimate`：消费已有视频，输出或覆盖 `mocap_*.npz`、`mocap_report.yaml` 和 `overlay_videos/`
- `realtime`：同时输出 `videos/*.mp4` 和 `mocap_*.npz`
- 所有模式都会写 `logs/mocap.log`
- 所有生成的 mp4 都使用 `H.264 / yuv420p` 编码，便于在 VS Code 和常见播放器中直接打开

## 6. 当前 `npz` 内容

- `timestamps`
- `camera_serials`
- `pose_2d`
- `pose_confidence`
- `left_hand_2d`
- `left_hand_confidence`
- `right_hand_2d`
- `right_hand_confidence`
- `bboxes_2d`
- `points_3d`
- `points_3d_raw`
- `reprojection_error`
- `landmark_names`

`points_3d_raw` 是三角化后的原始 3D 点。`points_3d` 是主结果。

默认 `capture_estimate` 使用离线高质量后处理：

```text
raw 3D
-> short-gap interpolation, max_gap_frames=10
-> zero-phase Butterworth low-pass, cutoff=1.2 Hz, order=4
```

该模式不做全局速度删点，适合最终数据产出，但使用 `filtfilt`，需要整段数据，不能逐帧在线输出。

`realtime` 模式和 `capture_estimate --postprocess-mode realtime` 使用在线因果后处理：

```text
raw 3D
-> short hold for missing points, max_hold_frames=3
-> causal EMA low-pass, cutoff=1.2 Hz
```

该模式不使用未来帧，适合 realtime 输出；参数会写入 `mocap_report.yaml` 的 `postprocess` 字段。

## 7. Fake Realtime

可以用已经采集好的 mocap 视频模拟 realtime 姿态估计链路，用于不连接相机时测试模型速度和在线后处理效果：

```bash
python -m roc.cli mocap \
  --mode capture_estimate \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --video-dir sessions/mocap_20260526_122017 \
  --model-complexity 1 \
  --postprocess-mode realtime \
  --delegate cpu
```

这会将结果写回 `--video-dir` 对应的 mocap session。

## 8. Offline MVS-like Source

`--offline-source-dir` 会把已经录好的多相机 mp4 或图片目录伪装成 MVS SDK 相机接口，用于测试实时链路，不需要连接工业相机：

```bash
python -m roc.cli mocap \
  --mode realtime \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --offline-source-dir sessions/mocap_20260526_122017/videos \
  --model-complexity 1 \
  --delegate gpu \
  --frames 100
```

该路径使用 `OfflineMvsSystem` / `OfflineMvsCamera`，提供和 MVS 相机相近的接口：

```text
enumerate_devices()
open_camera()
apply_manual_capture()
start_grabbing()
trigger_software()
grab_frame()
snapshot()
close()
```

源目录支持：

```text
videos/
  <serial>.mp4
```

或：

```text
frames/
  <serial>/
    frame_000000.bmp
    frame_000001.bmp
```

## 9. Realtime Benchmark

使用已录制视频测量不同 MediaPipe pose 模型复杂度在当前机器上的 fake realtime 速度：

```bash
python -m roc.mocap.benchmark_realtime \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --video-dir sessions/mocap_20260526_122017/videos \
  --frames 100 \
  --complexities 0 1 2 \
  --delegate cpu
```

关闭手部测速：

```bash
python -m roc.mocap.benchmark_realtime \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --video-dir sessions/mocap_20260526_122017/videos \
  --frames 100 \
  --complexities 0 1 2 \
  --no-hands \
  --delegate cpu
```

测试 GPU delegate：

```bash
python -m roc.mocap.benchmark_realtime \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --video-dir sessions/mocap_20260526_122017/videos \
  --frames 100 \
  --complexities 1 \
  --delegate gpu
```

如果当前 MediaPipe wheel 或图形/EGL 环境不支持 GPU delegate，benchmark 会将该档记录为 `failed` 并写入错误原因。

当前环境下 `--delegate gpu` 的短测结果为失败：

```text
Unable to initialize EGL
RET_CHECK failure (mediapipe/gpu/gl_context_egl.cc:84)
```

这说明当前运行环境没有可用的 MediaPipe GPU/EGL 上下文。需要先解决 EGL/OpenGL 显示环境或运行时配置，再重新测试 GPU delegate。

在 `sessions/mocap_20260526_122017/videos` 上的 100 帧测试结果：

| 模型 | hands | 四路 frame-set fps |
|---|---:|---:|
| lite, complexity=0 | on | 6.10 |
| full, complexity=1 | on | 6.33 |
| heavy, complexity=2 | on | 4.17 |
| lite, complexity=0 | off | 13.74 |
| full, complexity=1 | off | 13.35 |
| heavy, complexity=2 | off | 6.17 |

在线后处理平均耗时小于 `0.3 ms/frame-set`，主要瓶颈是四路 MediaPipe 串行推理。

## 10. 生成 2D overlay 视频

`capture_estimate` 完成后会自动在 `overlay_videos/` 下生成每个相机视图的 2D 检测叠加视频：

```text
overlay_videos/
  <serial>_2d_overlay.mp4
```

同时还会生成一个横向拼接的合成视频：

```text
overlay_videos/combined_2d_overlay.mp4
```

如果需要手动重新生成：

```bash
python -m roc.mocap.render_2d_overlays \
  --npz-path sessions/mocap_YYYYmmdd_HHMMSS/mocap_YYYYmmdd_HHMMSS.npz \
  --video-dir sessions/mocap_YYYYmmdd_HHMMSS/videos
```

## 11. 生成 3D 可视化 mp4

姿态估计完成后，可以把 `mocap_*.npz` 里的 `points_3d` 渲染成 3D 骨架预览视频：

```bash
python -m roc.mocap.render_npz \
  --npz-path sessions/mocap_YYYYmmdd_HHMMSS/mocap_YYYYmmdd_HHMMSS.npz
```

默认输出到同目录：

```text
sessions/mocap_YYYYmmdd_HHMMSS/mocap_YYYYmmdd_HHMMSS_preview.mp4
```

也可以手动指定输出路径和帧率：

```bash
python -m roc.mocap.render_npz \
  --npz-path sessions/mocap_20260526_122017/mocap_20260526_122017.npz \
  --output-path sessions/mocap_20260526_122017/mocap_20260526_122017_preview.mp4 \
  --fps 15
```

如果只想快速检查前若干帧：

```bash
python -m roc.mocap.render_npz \
  --npz-path sessions/mocap_20260526_122017/mocap_20260526_122017.npz \
  --frame-limit 100
```

生成的视频使用 `H.264 / yuv420p` 编码，通常可以直接在 VS Code 中打开预览。

## 12. 生成 3D 重投影诊断视频

如果 3D 骨架看起来异常，可以把 3D 点重投影回四个相机视图，和原始 2D 检测直接对照：

```bash
python -m roc.mocap.render_reprojection_overlays \
  --npz-path sessions/mocap_YYYYmmdd_HHMMSS/mocap_YYYYmmdd_HHMMSS.npz \
  --calibration-toml sessions/calib_YYYYmmdd_HHMMSS/calibration.toml \
  --video-dir sessions/mocap_YYYYmmdd_HHMMSS/videos \
  --points-key points_3d
```

输出目录：

```text
sessions/mocap_YYYYmmdd_HHMMSS/reprojection_videos/
```

视频中绿色是 MediaPipe 2D 检测，红色是 3D 点重投影。横向拼接视频为：

```text
combined_3d_reprojection_overlay_points_3d.mp4
```

也可以检查原始三角化结果：

```bash
python -m roc.mocap.render_reprojection_overlays \
  --npz-path sessions/mocap_YYYYmmdd_HHMMSS/mocap_YYYYmmdd_HHMMSS.npz \
  --calibration-toml sessions/calib_YYYYmmdd_HHMMSS/calibration.toml \
  --video-dir sessions/mocap_YYYYmmdd_HHMMSS/videos \
  --points-key points_3d_raw
```

注意：三角化前会将 2D 点按 `calibration.toml` 的相机顺序重排，`mocap_*.npz` 中的 `camera_serials` 表示保存后数组实际使用的相机顺序。

## 13. 说明

- 手部固定使用 `hand_landmarker.task`
- 当前推荐优先使用 `capture_estimate` 做调试
- 当前版本是第一版最小闭环，后续还会继续补：
  - 更强的关键点筛选
  - triangulation camera mask

## 14. 如果出问题

把以下内容反馈回来：

- 完整终端输出
- 使用的命令
- 生成的 `sessions/mocap_*/logs/mocap.log`
- 是否生成了 `mocap_*.npz`
- 预览窗口或视频里的人体检测是否合理
