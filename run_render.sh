#!/usr/bin/env bash
set -e

# 进入本脚本所在目录（建议把脚本放在 gaussian-splatting 项目根目录）
cd "$(dirname "$0")"

# 你只需要修改下面这一行的模型路径
MODEL_DIR="/workspace/RFGS/output/truck"

# 完全按你的命令运行
python render.py -m "$MODEL_DIR"

python metrics.py -m "$MODEL_DIR"