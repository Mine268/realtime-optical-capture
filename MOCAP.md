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

给 `capture_estimate` 手动指定视频目录。若目录是 `sessions/mocap_*/videos`，结果会直接写回这个 mocap session：

```bash
--video-dir sessions/mocap_YYYYmmdd_HHMMSS/videos
```

## 5. 输出说明

`capture` 和 `realtime` 会生成新的：

```text
sessions/mocap_YYYYmmdd_HHMMSS/
```

`capture_estimate` 的输出规则：

- 如果 `--video-dir` 指向 `sessions/mocap_*/videos`，则不新建 session，直接写回该 `sessions/mocap_*`
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
```

说明：

- `capture`：主要输出 `videos/*.mp4`
- `capture`：采集阶段先写 `raw_frames/<serial>/frame_*.bmp`，结束后再按真实 fps 转码成 `videos/*.mp4`
- `capture_estimate`：消费已有视频，输出或覆盖 `mocap_*.npz`、`mocap_report.yaml` 和 `overlay_videos/`
- `realtime`：同时输出 `videos/*.mp4` 和 `mocap_*.npz`
- 所有模式都会写 `logs/mocap.log`

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
- `reprojection_error`
- `landmark_names`

## 7. 说明

- `model-complexity 1` 使用 `pose_landmarker_full.task`
- `model-complexity 2` 使用 `pose_landmarker_heavy.task`
- 手部固定使用 `hand_landmarker.task`
- 当前推荐优先使用 `capture_estimate` 做调试
- 当前版本是第一版最小闭环，后续还会继续补：
  - 更强的关键点筛选
  - triangulation camera mask

## 8. 如果出问题

把以下内容反馈回来：

- 完整终端输出
- 使用的命令
- 生成的 `sessions/mocap_*/logs/mocap.log`
- 是否生成了 `mocap_*.npz`
- 预览窗口或视频里的人体检测是否合理
