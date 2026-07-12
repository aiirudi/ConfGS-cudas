#!/usr/bin/env bash
# 致密化候选选择策略消融实验批量运行脚本
# 依次运行 7 种策略，每种策略输出到独立目录。
#
# Usage:
#   bash scripts/run_candidate_ablation.sh [SCENE] [SEED]
#
#   SCENE: 数据集名称（默认: garden）
#   SEED:  随机种子（默认: 0）
#
# 输出:
#   output/candidate_ablation/<strategy>_seed<SEED>/

set -e

SCENE="${1:-garden}"
SEED="${2:-0}"

DATA_DIR="data/${SCENE}"
OUTPUT_BASE="output/candidate_ablation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo " 候选选择策略消融实验"
echo " Scene: ${SCENE}"
echo " Seed:  ${SEED}"
echo " Time:  ${TIMESTAMP}"
echo "============================================"

# 检查数据集
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: 数据集目录不存在: ${DATA_DIR}"
    echo "可用数据集:"
    ls data/ 2>/dev/null || echo "  (无)"
    exit 1
fi

# 创建基础输出目录
mkdir -p "${OUTPUT_BASE}"

# 策略列表
STRATEGIES=(
    "and"
    "or"
    "abs_only"
    "conf_only"
    "weighted_score"
    "soft_fusion"
    "rfas_rank"
)

run_experiment() {
    local strategy="$1"
    local extra_args="$2"
    local output_dir="${OUTPUT_BASE}/${strategy}_seed${SEED}"

    echo ""
    echo "--- Running: ${strategy} ---"
    echo "Output: ${output_dir}"

    python train.py \
        -s "${DATA_DIR}" \
        -m "${output_dir}" \
        --candidate_selection_strategy "${strategy}" \
        --candidate_budget_mode native \
        ${extra_args} \
        --eval \
        --seed "${SEED}"

    echo "--- Done: ${strategy} ---"
}

# ---- native 模式实验 ----
echo ""
echo "===== Phase 1: Native Budget Mode ====="

for strategy in "${STRATEGIES[@]}"; do
    extra_args=""
    case "${strategy}" in
        and|or|abs_only|conf_only)
            # 布尔策略：不需要额外参数
            ;;
        weighted_score)
            extra_args="--candidate_weight_alpha 0.5 --candidate_score_normalization percentile --candidate_score_selection topk --candidate_topk_ratio 0.05"
            ;;
        soft_fusion)
            extra_args="--soft_fusion_type weighted --candidate_weight_alpha 0.5 --soft_abs_temperature 1.0 --soft_conf_temperature 1.0 --candidate_score_selection topk --candidate_topk_ratio 0.05"
            ;;
        rfas_rank)
            extra_args="--candidate_rfas_topk_ratio 0.05"
            ;;
    esac
    run_experiment "${strategy}" "${extra_args}"
done

# ---- fixed + match_and 公平对比实验 ----
echo ""
echo "===== Phase 2: Fixed Budget (match_and) Mode ====="

for strategy in "${STRATEGIES[@]}"; do
    extra_args="--candidate_budget_mode fixed --candidate_budget_reference match_and"
    case "${strategy}" in
        and|or|abs_only|conf_only)
            ;;
        weighted_score)
            extra_args="${extra_args} --candidate_weight_alpha 0.5 --candidate_score_normalization percentile"
            ;;
        soft_fusion)
            extra_args="${extra_args} --soft_fusion_type weighted --candidate_weight_alpha 0.5 --soft_abs_temperature 1.0 --soft_conf_temperature 1.0"
            ;;
        rfas_rank)
            ;;
    esac

    output_dir="${OUTPUT_BASE}/${strategy}_match_and_seed${SEED}"
    echo ""
    echo "--- Running: ${strategy} (match_and) ---"
    echo "Output: ${output_dir}"

    python train.py \
        -s "${DATA_DIR}" \
        -m "${output_dir}" \
        --candidate_selection_strategy "${strategy}" \
        ${extra_args} \
        --eval \
        --seed "${SEED}"

    echo "--- Done: ${strategy} (match_and) ---"
done

echo ""
echo "============================================"
echo " 消融实验完成!"
echo " 结果目录: ${OUTPUT_BASE}/"
echo " 输出目录列表:"
ls -d ${OUTPUT_BASE}/*_seed${SEED}/ 2>/dev/null || echo "  (无)"
echo "============================================"
