# Prepare Stage 操作说明

本文件说明如何在接有工业相机的机器上运行项目的 `prepare` 阶段，完成多相机预览、曝光/增益调节，并保存 `capture_config.yaml`。

## 1. 前提条件

运行前需要满足：

- 已安装 MVS SDK，路径为 `/opt/MVS/`
- 机器已连接工业相机
- 有图形显示环境，能够打开 OpenCV 窗口
- 当前项目目录中已经存在 `.venv`

如果 `.venv` 还不存在，先在项目根目录执行：

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv venv --python 3.10 .venv
env UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python -e .
```

## 2. 启动命令

进入项目根目录后执行：

```bash
source .venv/bin/activate
python -m roc.cli prepare
```

也可以不激活环境，直接执行：

```bash
.venv/bin/python -m roc.cli prepare
```

## 3. 常用参数

默认命令：

```bash
.venv/bin/python -m roc.cli prepare
```

指定预览帧率：

```bash
.venv/bin/python -m roc.cli prepare --fps 5
```

只使用指定序列号的相机：

```bash
.venv/bin/python -m roc.cli prepare --serial 00J78371761 --serial 00J78371906
```

修改像素格式：

```bash
.venv/bin/python -m roc.cli prepare --pixel-format BayerRG8
```

修改预览缩放比例：

```bash
.venv/bin/python -m roc.cli prepare --preview-scale 0.5
```

## 4. 运行后界面

程序会：

1. 枚举当前可见的 MVS 相机
2. 打开 OpenCV 预览窗口
3. 以软件触发方式轮询采集多相机画面
4. 在窗口顶部叠加当前控制说明和每个相机的参数

每个视图左上角会显示：

- 相机编号
- 相机序列号

窗口左上角还会显示当前选中相机的曝光和增益。

## 5. 键盘操作

- `1-9`：选择当前要调节的相机
- `[`：减小曝光时间
- `]`：增大曝光时间
- `-`：减小增益
- `=`：增大增益
- `s`：保存配置并退出
- `q`：直接退出，不保存

## 6. 保存结果

按 `s` 后，程序会在项目根目录下生成一个新的 session 目录：

```text
sessions/prepare_YYYYmmdd_HHMMSS/
```

其中包含：

```text
capture_config.yaml
preview_snapshot/<serial>.jpg
logs/
```

后续 `calib` 阶段会读取这个 `capture_config.yaml`。

## 7. 调参建议

- 先保证所有相机都能稳定看到动捕区域
- 先调曝光，再微调增益
- 曝光过长会拖影，增益过高会噪声变重
- 尽量让四路画面的亮度和清晰度接近
- 如果某个相机明显更暗，优先检查镜头、光照和相机朝向，再调参数

## 8. 常见问题

### 没有检测到相机

检查：

- 相机是否上电
- USB/GigE 连接是否正常
- MVS Client 是否能看到相机
- 当前进程是否被其他程序占用了相机访问权限

### 窗口打不开

说明当前没有图形显示环境。需要在有桌面的机器、本地终端或正确配置的 X11/显示环境中运行。

### 运行时报 MVS SDK 相关错误

检查：

- `/opt/MVS/` 是否安装完整
- 是否能找到 `/opt/MVS/lib`
- 当前机器是否与相机驱动环境匹配

### 保存后想确认结果

直接检查最新生成的：

```text
sessions/prepare_*/capture_config.yaml
```

确认里面每个序列号都存在，并且 `exposure_us`、`gain_db` 已按你保存时的数值写入。

## 9. 建议的上机流程

1. 进入项目目录
2. 激活 `.venv`
3. 运行 `python -m roc.cli prepare`
4. 确认所有相机都出图
5. 用 `1-9` 切换相机，逐个调整曝光和增益
6. 调整完成后按 `s`
7. 把报错、现象、生成的 `capture_config.yaml` 路径反馈回来
