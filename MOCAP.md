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
  --inference-device gpu \
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
  --inference-device gpu \
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

`--retarget-mode` 控制 SMPL-X 路径：

- `fit`：默认高质量 fitting，逐帧调用参考 fitter，支持 lower-body refine 和可选 hand fitting，适合离线质量基线。
- `track`：body-only realtime tracker，冻结 betas 和手部 pose，用上一帧 warm start 做轻量 Adam 优化，适合低延迟驱动。当前实现会对膝盖、肘部和髋/肩横轴朝向加入显式约束，并对异常肘/腕三角化做 limb-length gate，避免追踪明显错误的单帧手臂点。

### Fit 与 Track 的使用场景

`fit` 适合做质量基线、离线重建、算法对比、疑难帧诊断，以及后续 track 初始化/恢复策略的参考。它的输出更贴近 3D keypoints，但速度很慢；在 RTX 2080 Ti 上，当前 20 帧墙钟测试约 `0.61 FPS`。需要最高质量或需要检查 track 是否跑偏时，使用 `fit`。

```bash
roc mocap \
  --mode realtime \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --mocap-session sessions/mocap_test \
  --frames 200 \
  --offline-source-dir sessions/mocap_test/videos \
  --inference-device gpu \
  --model-complexity 2 \
  --no-hands \
  --retarget \
  --retarget-mode fit \
  --retarget-model-dir models/smplx \
  --profile
```

`track` 适合 body-only realtime SMPL-X tracking 和角色驱动。它牺牲少量拟合精度换取低延迟，当前建议使用 `--retarget-pose-steps 18`；6 steps 虽然接近 10 FPS，但抖动明显，不适合驱动。需要实时处理、后续驱动或快速预览时，使用 `track`。

```bash
roc mocap \
  --mode realtime \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --mocap-session sessions/mocap_test \
  --frames 200 \
  --offline-source-dir sessions/mocap_test/videos \
  --inference-device gpu \
  --model-complexity 2 \
  --no-hands \
  --retarget \
  --retarget-mode track \
  --retarget-model-dir models/smplx \
  --retarget-pose-steps 18 \
  --profile
```

```bash
roc mocap \
  --mode capture_estimate \
  --prepare-session sessions/prepare_20260525_162846 \
  --calib-session sessions/calib_20260525_163831 \
  --mocap-session sessions/mocap_test \
  --frames 200 \
  --retarget \
  --retarget-mode fit \
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
--inference-device gpu          # MediaPipe 使用 GPU delegate，retarget 使用 CUDA
--retarget-mode fit             # 高质量 fitting，默认
--retarget-mode track           # body-only realtime tracking
--retarget-max-frames 200       # 限制 retarget 帧数，-1 表示全部
--retarget-frame-step 2         # 每隔 N 帧 retarget 一帧
--retarget-input-scale 0.001    # ROC 3D 点默认从毫米转成米
--retarget-pose-steps 120       # 每帧姿态优化步数
--retarget-root-steps 30        # 覆盖每帧 root 对齐步数，默认 max(12, pose_steps/4)
--retarget-lower-steps 20       # 覆盖 lower-body refine 步数，默认 max(6, pose_steps/6)
--retarget-no-lower-refine      # 跳过额外 lower-body refine，适合低延迟 realtime
--retarget-early-stop-check-interval 5  # 减少 CUDA loss 同步频率
--retarget-temporal-weight 1.0          # 约束当前帧贴近上一帧，降低低步数抖动
--retarget-acceleration-weight 0.02     # 抑制二阶抖动
--retarget-realtime-root-steps 2         # realtime 稳定帧 root 对齐步数
--retarget-realtime-root-recovery-steps 12  # 初始化/转身/高误差时 root 对齐步数
--retarget-no-adaptive-root              # 关闭 realtime 自适应 root 步数
--retarget-betas-steps 80       # 共享体型优化步数
--retarget-hands                # 同时优化 SMPL-X 手部 pose
--retarget-save-debug-assets    # 额外保存 obj/png 调试文件
```

`track` 模式优先服务 realtime body-only 驱动。`--retarget-pose-steps` 会控制稳定帧的 track Adam 步数，当前上限为 20；第一帧和 recovery 帧会自动使用更高步数。实测 6 steps 能接近 10 FPS，但 SMPL-X pose 抖动明显，不适合驱动；18 steps 是当前可用质量下限。需要对比质量时，同一段数据建议分别跑 `--retarget-mode fit` 和 `--retarget-mode track`，并比较 `smplx_retarget/smplx_fit_sequence.npz`、SMPL-X reprojection overlay、膝/肘夹角和 wrist acceleration。

当前 `sessions/mocap_test` 200 帧验证结果（RTX 2080 Ti，fake MVS，body-only track）：

```text
track 18-step realtime profile from sessions/mocap_test/logs/mocap.log:
  all 200 frames including first-frame initialization: 208.1 ms/frame, 4.80 FPS
  steady frames 1-199: loop 199.9 ms/frame, 5.00 FPS
  steady estimate_only mean/p50/p95: 53.6 / 52.8 / 59.9 ms
  steady smplx_retarget mean/p50/p95: 146.3 / 144.1 / 167.1 ms
  track body_err mean/p50/p95: 0.048 / 0.047 / 0.062 m

track 6-step realtime profile:
  loop about 95-108 ms/frame, near 9-10 FPS, but visible jitter makes it unusable for driving

fit mode wall-clock baseline, first 20 frames:
  1629.8 ms/frame, 0.61 FPS
  previous 50-frame fit report mean_body_error: 0.0416 m
  track 18-step 200-frame report mean_body_error: 0.0481 m

track mapped 3D error mean/p90/max: 0.0616 / 0.0730 / 0.1898 m
fit   mapped 3D error mean/p90/max: 0.0517 / 0.0658 / 0.1926 m
track-vs-fit body joint mean/p90/max: 0.0722 / 0.0913 / 0.1544 m
frame 96 knee angle target/track/fit: L 40.4/45.0/39.8 deg, R 39.8/46.1/37.4 deg
frame 96 elbow angle target/track/fit: L 116.8/114.4/119.3 deg, R 112.1/112.3/112.8 deg
track wrist current-frame error mean/p90: L 0.0906/0.1400 m, R 0.0806/0.1297 m
track hip-yaw error on frames 60-115 mean/p90/max: 9.7 / 21.6 / 28.5 deg
track retarget-only throughput: about 5.8 FPS on the earlier 200-frame benchmark
```

`realtime` 默认也会使用高质量 fitting 参数，因此启用 `--retarget` 后会明显变慢。排查耗时时使用统一的 mocap profile：

```bash
--profile \
--inference-device gpu
```

`--profile` 会逐帧打印 `[mocap-profile]`，输出同时进入 `logs/mocap.log`。`stage=estimate` 行中 `mocap_loop_total` 是本帧主循环总耗时，`estimate_only` 是扣除 `smplx_retarget` 后的姿态估计链路耗时；`stage=retarget` 行中 `smplx_total` 是 SMPL-X fitting 总耗时，`pose_optimize` 和 `lower_body_refine` 通常是主要瓶颈。

低延迟测试可先使用较小预算：

```bash
--profile \
--inference-device gpu \
--retarget-pose-steps 60 \
--retarget-realtime-root-steps 2 \
--retarget-realtime-root-recovery-steps 8 \
--retarget-lower-steps 8 \
--retarget-temporal-weight 1.0 \
--retarget-acceleration-weight 0.02 \
--retarget-early-stop-check-interval 5
```

`realtime` 默认启用自适应 root：第一帧、高误差帧、髋部横轴转角超过阈值的帧、或髋中心位移过大的帧使用 recovery root steps；正常连续帧使用较小的 `--retarget-realtime-root-steps`。不要直接把 `--retarget-pose-steps` 降得过低；低步数需要配合 temporal/acceleration 权重，否则 SMPL-X pose 会跟随单帧 3D 噪声抖动。速度、误差和稳定性需要用 `retarget_report.yaml` 中的误差、`realtime_root_adaptation` 和阶段耗时一起评估。

## 常用参数

```bash
--frames 300              # 限制帧数，0 表示直到按 q
--show-preview            # 显示 OpenCV 预览
--no-hands                # 关闭手部检测
--model-complexity 0      # lite pose 模型
--model-complexity 1      # full pose 模型
--model-complexity 2      # heavy pose 模型
--inference-device cpu    # MediaPipe 和 retarget 都使用 CPU
--inference-device gpu    # MediaPipe 使用 GPU delegate，retarget 使用 CUDA
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
