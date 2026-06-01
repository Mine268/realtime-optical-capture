# ROC 命令行补全

`roc` 使用 `argcomplete` 提供 shell tab 补全。

## 安装依赖

如果是从当前项目环境安装，执行：

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python -e .
```

## 临时启用

当前 shell 临时启用：

```bash
source .venv/bin/activate
eval "$(register-python-argcomplete roc)"
```

之后可以补全：

```bash
roc <TAB>
roc mocap --mode <TAB>
roc mocap --mocap-session sessions/<TAB>
```

## 持久启用

追加到 `~/.bashrc`：

```bash
eval "$(register-python-argcomplete roc)"
```

打开新终端后生效。

## zsh

如果使用 zsh：

```bash
autoload -U bashcompinit
bashcompinit
eval "$(register-python-argcomplete roc)"
```
