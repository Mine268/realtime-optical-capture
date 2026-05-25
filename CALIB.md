# Calibration Stage 操作说明

当前 `calib` 阶段支持三种模式：

- `capture-only`：只采集标定视频
- `solve-only`：基于已有 `sessions/calib_*` 中的 `videos/*.mp4` 完成标定求解
- `capture+solve`：先采集，再立即求解

当前默认 ChArUco 配置：

- squares：`7 x 5`
- dictionary：`DICT_4X4_250`
- square length：`190.5 mm`
- marker length：`152.4 mm`

## 1. 启动方式

进入项目根目录后执行：

```bash
source .venv/bin/activate
```

## 2. 常用命令

只做采集：

```bash
python -m roc.cli calib \
  --mode capture-only \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS
```

采集后立即求解：

```bash
python -m roc.cli calib \
  --mode capture+solve \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS
```

对已经采集好的 session 单独求解：

```bash
python -m roc.cli calib \
  --mode solve-only \
  --calib-session sessions/calib_YYYYmmdd_HHMMSS
```

例如：

```bash
python -m roc.cli calib \
  --mode solve-only \
  --calib-session sessions/calib_20260525_170000
```

## 3. 常用参数

采集时显示实时预览：

```bash
python -m roc.cli calib \
  --mode capture-only \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS \
  --show-preview
```

修改采集帧率和帧数：

```bash
python -m roc.cli calib \
  --mode capture-only \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS \
  --fps 3 \
  --frames 120
```

修改世界坐标模式：

```bash
python -m roc.cli calib \
  --mode capture+solve \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS \
  --world-mode camera0
```

或者：

```bash
python -m roc.cli calib \
  --mode solve-only \
  --calib-session sessions/calib_YYYYmmdd_HHMMSS \
  --world-mode ground
```

修改 ChArUco 尺寸参数：

```bash
python -m roc.cli calib \
  --mode capture+solve \
  --prepare-session sessions/prepare_YYYYmmdd_HHMMSS \
  --square-length-mm 190.5 \
  --marker-length-mm 152.4
```

## 4. 当前输出

标定 session 目录结构：

```text
sessions/calib_YYYYmmdd_HHMMSS/
  capture_config.yaml
  calib_config.yaml
  videos/
    <serial>.mp4
  calibration.toml
  calibration.yaml
  charuco_2d.npz
  charuco_3d.npy
  calibration_report.yaml
  calibration_visualization.png
  charuco_overlays/
    <serial>_charuco_overlay.mp4
```

## 5. 上机建议

1. 先完成 `prepare`
2. 如果还没有标定视频，运行 `capture-only` 或 `capture+solve`
3. 让用户手持 ChArUco 板在动捕区域内运动
4. 尽量让各相机都在一段时间内看到标定板
5. 如果已经有 `sessions/calib_*`，直接运行 `solve-only`
6. 结束后确认 `calibration.toml` 和 `calibration.yaml` 是否生成

## 6. 如果出问题

把以下内容反馈回来：

- 完整终端输出
- 使用的命令
- 生成的 `sessions/calib_*/` 路径
- `videos/` 下是否每个相机都有 mp4
- 是否生成了 `calibration.toml` 和 `calibration.yaml`
