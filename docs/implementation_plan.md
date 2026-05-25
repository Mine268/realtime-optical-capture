# 固定多视图实时光学姿态估计系统实施计划

## 1. 目标和范围

本项目实现一个基于固定多视图工业相机的实时全身 3D 姿态估计系统。系统从多台 MVS SDK 工业相机同步采集图像，结合已标定的相机内参、畸变和外参，使用 Mediapipe 估计人体 2D 关键点，并通过多视图三角化输出 3D 全身关节点。

系统分为三个阶段：

- `prepare`：相机预览和捕捉参数调整，输出 `capture_config.yaml`。
- `calib`：采集 ChArUco 标定视频并使用 aniposelib 标定相机，输出 `calibration.yaml` 和 Anipose TOML。
- `mocap`：实时或离线运行 Mediapipe 2D 检测和多视图三角化，输出 `mocap_*.npz`。

当前默认相机数量为 4，但所有配置和管线必须支持 2、4、6 等可变数量相机。相机身份以工业相机序列号作为稳定 ID。

## 2. 已确认设计决策

### 2.1 技术栈

- 主系统使用 Python 实现。
- Python 环境使用 `uv` 管理，虚拟环境放在项目目录下。
- MVS SDK 采集层可先复用和改造 `/home/hmmmocap/workspace/mvs_multiview_sync_capture` 中的 C++ 代码。
- 实时同步采集如果 Python 性能不足，采集层下沉到 C++，通过 subprocess、pybind11 或 ctypes 暴露给 Python。
- 标定使用 `aniposelib`。
- 2D 姿态估计使用 Mediapipe。
- UI 交互使用 OpenCV 窗口。

### 2.2 相机同步

- 默认同步方式为 MVS SDK 软件触发。
- 目标实时帧率为 5 FPS。
- 标定阶段默认 2-5 FPS，采集 120 帧。
- 每个阶段的图像采集数据按相机分别保存为视频文件，文件名使用相机序列号。

### 2.3 标定板

默认 ChArUco 参数：

- chessboard squares：`7 x 5`
- dictionary：`DICT_4X4_250`
- square length：`190.5 mm`
- marker length：默认 `0.8 * square_length`
- 单位：毫米

`calib` 阶段启动时询问用户是否使用默认参数；如不使用，则允许用户输入并保存到该次 calibration session 中。

### 2.4 世界坐标系

标定完成后提供两种世界坐标系：

- `camera0`：默认方式，将 0 号相机固定为世界原点。
- `ground`：先完成多相机相对 pose 标定，再基于标定板初始稳定位置重建 ChArUco 3D 点，将该板面作为地面平面并重写所有相机外参。

### 2.5 人体关键点

- 默认输出 Mediapipe pose 关键点，包含鼻子、眼睛、耳朵等头部 pose 点。
- 不输出完整 face mesh。
- hands 作为可选开关。
- 默认场景只有一个人。
- 实时版本以 5 FPS 为目标。
- 提供离线 heavy 模型版本，用于质量优先处理。

## 3. 参考实现和复用点

### 3.1 MVS 多视图采集

参考项目：

```text
/home/hmmmocap/workspace/mvs_multiview_sync_capture
```

该项目已实现：

- MVS SDK 相机枚举。
- USB3/GigE 混合相机支持。
- 软件触发同步。
- 多线程取流和帧缓冲。
- MP4、PNG、JPG、RAW 保存。
- OpenCV 实时预览。

需要改造点：

- 当前 CLI 解析了曝光和增益参数，但没有实际应用到设备。
- `SyncCaptureManager` 需要暴露按相机序列号设置参数的接口。
- 需要输出 frame metadata，至少包括：
  - camera serial
  - frame index
  - trigger index
  - device timestamp
  - host timestamp
  - exposure
  - gain
  - image width/height
  - pixel format

### 3.2 FreeMoCap 标定和三角化

参考目录：

```text
refs/freemocap/freemocap/core_processes/capture_volume_calibration
```

可参考内容：

- Anipose 标定流程。
- camera0 pin 到原点的外参变换。
- ChArUco ground plane 对齐逻辑。
- 三角化和 reprojection error 计算。

注意事项：

- FreeMoCap 当前封装中的 ArUco 字典映射主要覆盖普通 `DICT_4X4_*` 到 `DICT_7X7_*`，满足当前默认的 `DICT_4X4_250`。
- 本项目仍保留自定义 ChArUco board/detector 配置层，便于后续切换字典或做自动探测回退。

### 3.3 SMPL-X 后续对接

参考目录：

```text
refs/smplx_from_freemocap_3d
```

当前阶段只保证 `mocap_*.npz` 中的 `points_3d`、关键点名称和时间序列足够后续 retarget/IK 使用。后续可增加 FreeMoCap 风格导出器。

## 4. 项目目录规划

建议目录结构：

```text
realtime_optical_capture/
  pyproject.toml
  uv.lock
  .venv/
  docs/
    implementation_plan.md
  roc/
    __init__.py
    cli.py
    config/
      __init__.py
      models.py
      defaults.py
      yaml_io.py
    io/
      __init__.py
      sessions.py
      video.py
      npz_schema.py
    mvs/
      __init__.py
      python_api.py
      bridge/
        CMakeLists.txt
        include/
        src/
    prepare/
      __init__.py
      app.py
      controls.py
    calib/
      __init__.py
      capture.py
      charuco.py
      anipose_solver.py
      ground_plane.py
      report.py
    tracking/
      __init__.py
      mediapipe_tracker.py
      landmarks.py
    triangulation/
      __init__.py
      cameras.py
      triangulate.py
      filters.py
    mocap/
      __init__.py
      realtime.py
      offline.py
      writer.py
  sessions/
    prepare_YYYYmmdd_HHMMSS/
    calib_YYYYmmdd_HHMMSS/
    mocap_YYYYmmdd_HHMMSS/
```

Python 命令入口：

```bash
uv run roc prepare
uv run roc calib --prepare sessions/prepare_YYYYmmdd_HHMMSS
uv run roc mocap --prepare sessions/prepare_YYYYmmdd_HHMMSS --calib sessions/calib_YYYYmmdd_HHMMSS
uv run roc mocap-offline --session sessions/mocap_YYYYmmdd_HHMMSS --model heavy
```

## 5. Session 组织

每个阶段运行结果保存到独立 session 文件夹，后续阶段可直接引用前一阶段结果。

### 5.1 Prepare Session

```text
sessions/prepare_YYYYmmdd_HHMMSS/
  capture_config.yaml
  preview_snapshot/
    <serial>.jpg
  logs/
    prepare.log
```

### 5.2 Calibration Session

```text
sessions/calib_YYYYmmdd_HHMMSS/
  capture_config.yaml
  calib_config.yaml
  videos/
    <serial>.mp4
  raw_frames/              # 可选，无损保存时使用
    <serial>/
      frame_000000.png
  calibration.toml         # Anipose 原生格式
  calibration.yaml         # 本项目稳定格式
  charuco_2d.npz
  charuco_3d.npy
  calibration_report.yaml
  logs/
    calib.log
```

### 5.3 Mocap Session

```text
sessions/mocap_YYYYmmdd_HHMMSS/
  capture_config.yaml
  calibration.yaml
  videos/
    <serial>.mp4
  raw_frames/              # 可选，无损保存时使用
    <serial>/
      frame_000000.png
  mocap_YYYYmmdd_HHMMSS.npz
  overlay_videos/          # 可选
    <serial>_2d_overlay.mp4
  mocap_report.yaml
  logs/
    mocap.log
```

## 6. 配置文件设计

### 6.1 `capture_config.yaml`

示例：

```yaml
schema_version: 1
created_at: "2026-05-25T10:00:00+08:00"
camera_count: 4
camera_serials:
  - "00J78371761"
  - "00J78371906"
  - "00J78371876"
  - "00J78371888"
sync:
  mode: software_trigger
  fps: 5.0
capture:
  pixel_format: BayerRG8
  output_format: mp4
  lossless: false
  preview_scale: 0.5
cameras:
  "00J78371761":
    enabled: true
    index_hint: 0
    exposure_us: 8000.0
    gain_db: 6.0
    width: 1440
    height: 1080
  "00J78371906":
    enabled: true
    index_hint: 1
    exposure_us: 8000.0
    gain_db: 6.0
    width: 1440
    height: 1080
```

### 6.2 `calib_config.yaml`

示例：

```yaml
schema_version: 1
created_at: "2026-05-25T10:05:00+08:00"
prepare_session: "../prepare_YYYYmmdd_HHMMSS"
frames: 120
fps: 3.0
charuco:
  squares_x: 7
  squares_y: 5
  dictionary: DICT_4X4_250
  square_length_mm: 190.5
  marker_length_mm: 152.4
world:
  mode: camera0
video:
  format: mp4
  lossless: false
```

### 6.3 `calibration.yaml`

示例：

```yaml
schema_version: 1
created_at: "2026-05-25T10:10:00+08:00"
source_session: "sessions/calib_YYYYmmdd_HHMMSS"
units: mm
world:
  mode: camera0
  ground_plane_success: null
camera_order:
  - "00J78371761"
  - "00J78371906"
cameras:
  "00J78371761":
    size: [1440, 1080]
    matrix:
      - [fx, 0.0, cx]
      - [0.0, fy, cy]
      - [0.0, 0.0, 1.0]
    distortion: [k1, k2, p1, p2, k3]
    rotation_vector: [0.0, 0.0, 0.0]
    translation_vector: [0.0, 0.0, 0.0]
    position_world: [0.0, 0.0, 0.0]
    rotation_cam_to_world:
      - [1.0, 0.0, 0.0]
      - [0.0, 1.0, 0.0]
      - [0.0, 0.0, 1.0]
quality:
  mean_reprojection_error_px: 0.35
  per_camera_reprojection_error_px:
    "00J78371761": 0.31
```

## 7. Prepare 阶段计划

### 7.1 目标

通过实时预览和交互式控制，为每台相机确定曝光、增益等捕捉参数，并保存到 `capture_config.yaml`。

### 7.2 功能

- 自动枚举相机。
- 支持指定相机序列号列表。
- OpenCV 多视图预览。
- 当前相机选择。
- 曝光和增益调整。
- 保存当前帧截图用于记录。
- 保存配置并退出。

### 7.3 OpenCV 交互建议

键盘控制：

- `1`-`9`：选择相机。
- `[` / `]`：减少/增加曝光。
- `-` / `=`：减少/增加增益。
- `a`：切换自动曝光。
- `g`：切换自动增益。
- `s`：保存配置。
- `q`：退出。

如 OpenCV trackbar 在目标环境稳定，也可提供 trackbar 控制曝光和增益。

### 7.4 输出

- `capture_config.yaml`
- `preview_snapshot/<serial>.jpg`
- `logs/prepare.log`

## 8. Calibration 阶段计划

### 8.1 目标

读取 prepare 阶段保存的相机参数，采集 ChArUco 标定视频，使用 aniposelib 完成多相机内参、畸变和外参标定。

### 8.2 采集流程

1. 读取 `capture_config.yaml`。
2. 询问用户是否使用默认 ChArUco 参数。
3. 根据配置初始化相机。
4. 以 2-5 FPS 软件触发采集 120 帧。
5. 每个相机保存一个 `<serial>.mp4`。
6. 可选保存无损 PNG 序列。
7. 输出 frame metadata。

### 8.3 标定求解流程

1. 从每个相机视频中检测 ChArUco corners。
2. 保存 `charuco_2d.npz`。
3. 检查每个相机的可见帧数量。
4. 构建相机共视图，检查相机图是否连通。
5. 调用 aniposelib 标定。
6. 默认 pin camera0 到世界原点。
7. 如选择 `ground`，基于初始稳定 ChArUco 3D 点重写世界坐标。
8. 保存 Anipose TOML 和本项目 `calibration.yaml`。
9. 输出 reprojection error 报告。

### 8.4 部分视图可见处理

标定板不需要每一帧同时出现在全部相机中，但必须满足：

- 每台参与标定的相机有足够 ChArUco 检测帧。
- 任意两台相机之间可以通过一系列共同观测帧间接连接。

如果某台相机和其他相机没有共视连接，标定应失败并提示重新采集。

### 8.5 Ground Plane 处理

`--world ground` 时：

1. 先完成普通多相机标定。
2. 使用标定结果三角化 ChArUco 角点。
3. 在采集序列早期寻找稳定且可见的 ChArUco 帧。
4. 使用该帧的 ChArUco 平面定义新世界坐标：
   - 原点：ChArUco 第一个角点。
   - X/Y：标定板平面方向。
   - Z：标定板法线方向。
5. 将所有相机外参变换到新世界坐标系。
6. 保存 ground plane 成功状态和误差报告。

## 9. Mocap 阶段计划

### 9.1 目标

读取捕捉配置和标定结果，实时同步采集多视图图像，运行 Mediapipe pose/hands 2D 检测，并将有效多视图 2D 关键点三角化为 3D 关节点。

### 9.2 实时流程

1. 读取 `capture_config.yaml`。
2. 读取 `calibration.yaml`。
3. 初始化 MVS 软件触发采集。
4. 初始化 Mediapipe 模型。
5. 对每个同步 frame set：
   - 获取每个相机图像。
   - 写入每相机 MP4。
   - 运行 2D landmark 检测。
   - 保存 bbox、2D 点和置信度。
   - 对每个 landmark 选择有效相机。
   - 使用至少 2 个有效视图三角化。
   - 计算 reprojection error。
6. 实时写入缓存，结束时保存 `mocap_*.npz`。

### 9.3 离线 Heavy 流程

`mocap-offline` 读取已保存的 mocap session 视频，使用 heavy 模型重新计算 2D 检测和 3D 三角化，用于质量优先输出。

### 9.4 有效视图筛选

对每个 frame、每个 landmark：

- 坐标为 NaN 的视图剔除。
- visibility/presence 低于阈值的视图剔除。
- bbox 不可信或没有人体检测的视图剔除。
- 剩余有效视图数量必须 >= 2。
- 如果有效视图数量不足，3D 点设为 NaN。

可选高级策略：

- 对三角化后 reprojection error 过大的点做 outlier camera drop。
- 允许最多丢弃 1 个相机。
- 保存最终参与三角化的 camera mask。

### 9.5 `mocap_*.npz` Schema

```text
timestamps                   [frames]
camera_serials               [cams]
image_size                   [cams, 2]
pose_2d                      [cams, frames, pose_points, 2]
pose_confidence              [cams, frames, pose_points]
pose_bbox                    [cams, frames, 4]
hands_2d                     [cams, frames, hand_points, 2]
hands_confidence             [cams, frames, hand_points]
points_3d                    [frames, total_points, 3]
points_3d_confidence         [frames, total_points]
triangulation_camera_mask    [frames, total_points, cams]
reprojection_error           [frames, total_points]
landmark_names               [total_points]
landmark_model               string
calibration_yaml             string/object
capture_config_yaml          string/object
```

`points_3d` 的单位与 calibration 一致，默认毫米。

## 10. MVS Bridge 设计

### 10.1 分阶段接入策略

第一阶段优先使用 subprocess 方式调用 C++ bridge，保证 prepare、calib 能尽快闭环。

第二阶段为实时 mocap 增加 Python 可直接拉帧的接口，优先考虑 pybind11；如构建复杂，再考虑 ctypes C ABI。

### 10.2 Bridge 能力

Bridge 至少需要提供：

- `list-cameras`
- `preview`
- `capture-video`
- `capture-stream`
- `apply-config`

CLI 示例：

```bash
mvs_bridge list-cameras --json
mvs_bridge preview --config capture_config.yaml
mvs_bridge capture-video --config capture_config.yaml --fps 3 --frames 120 --out sessions/calib_xxx/videos
mvs_bridge capture-stream --config capture_config.yaml --fps 5
```

### 10.3 Frame Metadata

每个采集阶段应保存：

```text
frame_metadata.parquet 或 frame_metadata.jsonl
```

字段：

- frame_set_index
- camera_serial
- camera_index
- frame_number
- trigger_index
- host_timestamp_us
- device_timestamp
- exposure_us
- gain_db
- width
- height
- pixel_format
- video_frame_index

## 11. 依赖计划

`pyproject.toml` 初始依赖建议：

```toml
[project]
dependencies = [
  "numpy",
  "scipy",
  "pyyaml",
  "opencv-contrib-python",
  "mediapipe",
  "aniposelib",
  "toml",
  "tqdm",
  "pydantic",
  "rich",
]
```

如果 `opencv-contrib-python` 与系统 OpenCV 或 MVS 采集层冲突，采集层使用 C++ OpenCV，Python 环境单独使用 wheel。

## 12. 实施里程碑

### M1：项目骨架

- 创建 `pyproject.toml`。
- 创建 `roc` 包和 CLI。
- 创建 session 管理、配置读写、日志基础设施。
- 定义 pydantic/dataclass 配置模型。

### M2：MVS Bridge 最小闭环

- 将现有 MVS 采集代码迁入或以子项目方式引用。
- 修复曝光/增益解析后未应用的问题。
- 支持按序列号选择相机。
- 支持从 YAML/JSON 配置初始化相机。
- 支持按序列号输出 MP4。
- 输出 frame metadata。

### M3：Prepare 阶段

- 实现 OpenCV 多视图预览。
- 实现曝光/增益交互调整。
- 保存 `capture_config.yaml`。
- 保存 preview snapshot。

### M4：Calibration Capture

- 读取 prepare 配置。
- 询问 ChArUco 参数。
- 采集 120 帧标定视频。
- 保存 `calib_config.yaml`、视频和 metadata。

### M5：Calibration Solve

- 实现 AprilTag ChArUco 检测。
- 保存 `charuco_2d.npz`。
- 实现可见性和连通性检查。
- 使用 aniposelib 求解相机内外参。
- 实现 `camera0` 坐标系。
- 实现 `ground` 坐标系。
- 输出 `calibration.toml`、`calibration.yaml`、`calibration_report.yaml`。

### M6：Mocap Offline

- 从已保存视频读取帧。
- 运行 Mediapipe pose/hands。
- 保存 2D 检测结果。
- 实现多视图三角化。
- 保存 `mocap_*.npz`。
- 生成基础 reprojection report。

### M7：Mocap Realtime

- 将 MVS bridge 帧流接入 Python。
- 实时 5 FPS 运行 Mediapipe。
- 同步写入视频和 3D 结果缓存。
- 结束后保存完整 `mocap_*.npz`。

### M8：诊断和可视化

- 2D overlay 视频。
- 3D skeleton 预览脚本。
- 标定 reprojection error 图表。
- mocap reprojection error 和缺失率报告。

### M9：SMPL-X 对接准备

- 输出稳定 landmark names。
- 增加 FreeMoCap 风格导出器。
- 与 `refs/smplx_from_freemocap_3d` 输入格式做最小适配。

## 13. 验证标准

### 13.1 Prepare

- 能枚举所有相机。
- 能稳定预览所有相机画面。
- 能独立调整每台相机曝光和增益。
- 保存后重新加载配置，相机图像亮度符合保存参数。

### 13.2 Calibration

- 每个相机视频帧数一致或 metadata 可解释帧数差异。
- ChArUco 检测覆盖足够多帧。
- 相机共视图连通。
- 平均 reprojection error 在可接受范围内。
- `camera0` 和 `ground` 坐标系输出可复现。

### 13.3 Mocap

- 5 FPS 实时模式能持续运行。
- 每个相机视频正确保存。
- 2D pose/hands 中间结果完整保存。
- 有 >=2 个有效视图的关键点能输出 3D。
- `mocap_*.npz` 可被后续处理脚本读取。

## 14. 待确认事项

以下事项不会阻塞初始实现，但需要在真实硬件测试前确认：

- 实体 ChArUco 板是横向 `5 x 7` 还是 `7 x 5`；标定本身可配置，但会影响 ground 坐标轴方向。
- 实体板 marker length 是否确实为 `0.8 * square_length`。
- Mediapipe 是否使用 GPU 加速的具体部署方式；2080Ti 可用于后续优化，但初始实现先保证 CPU/GPU 可切换。
- 无损保存使用 PNG 序列还是原始 RAW；默认 MP4 压缩保存。
