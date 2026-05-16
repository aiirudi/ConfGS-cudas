set -e

cd "$(dirname "$0")"

MODEL_DIR="output/playroom"

# 训练完立刻打印 FPS（默认用 test split）
python bench_fps.py -m "$MODEL_DIR" --iteration 30000 --num_frames 30000
