#!/usr/bin/env bash
# 致密化候选选择策略消融实验批量运行脚本
# Usage:
#   bash scripts/run_candidate_ablation.sh [SCENE] [SEED]
#   SCENE: 数据集名称 (默认: garden)
#   SEED:  随机种子 (默认: 0)
set -e

SCENE="${1:-garden}"
SEED="${2:-0}"
ALPHA="0.5"

DATA_DIR="data/${SCENE}"
OUTPUT_BASE="output/candidate_ablation"

echo "============================================"
echo " 候选选择策略消融实验"
echo " Scene: ${SCENE}  Seed: ${SEED}  Alpha: ${ALPHA}"
echo "============================================"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: 数据集不存在: ${DATA_DIR}"
    ls data/ 2>/dev/null || echo "  (无)"
    exit 1
fi

mkdir -p "${OUTPUT_BASE}"

STRATEGIES=(and or abs_only conf_only weighted_score soft_fusion rfas_rank)

run_experiment() {
    local strategy="$1"
    local extra_args="$2"
    local output_dir="$3"

    echo ""
    echo "--- ${strategy} ---"
    echo "Dir: ${output_dir}"

    python train.py \
        -s "${DATA_DIR}" \
        -m "${output_dir}" \
        --candidate_selection_strategy "${strategy}" \
        ${extra_args} \
        --eval \
        --seed "${SEED}"

    echo "--- Done: ${strategy} ---"
}

# ---- Phase 1: Native Budget Mode ----
echo ""
echo "===== Phase 1: Native Budget Mode ====="

for strategy in "${STRATEGIES[@]}"; do
    extra_args="--candidate_budget_mode native"
    suffix="native"

    case "${strategy}" in
        and|or|abs_only|conf_only)
            ;;
        weighted_score)
            extra_args="${extra_args} --candidate_weight_alpha ${ALPHA} --candidate_score_normalization percentile --candidate_score_selection threshold --candidate_score_threshold 0.5"
            suffix="native_a${ALPHA}"
            ;;
        soft_fusion)
            extra_args="${extra_args} --soft_fusion_type weighted --candidate_weight_alpha ${ALPHA} --soft_abs_temperature 1.0 --soft_conf_temperature 1.0 --candidate_score_selection threshold --soft_selection_threshold 0.5"
            suffix="native_a${ALPHA}"
            ;;
        rfas_rank)
            extra_args="${extra_args} --candidate_rfas_topk_ratio 0.05"
            ;;
    esac

    output_dir="${OUTPUT_BASE}/${strategy}_${suffix}_seed${SEED}"
    run_experiment "${strategy}" "${extra_args}" "${output_dir}"
done

# ---- Phase 2: Fixed Budget (match_and) Mode ----
echo ""
echo "===== Phase 2: Fixed Budget (match_and) Mode ====="

for strategy in "${STRATEGIES[@]}"; do
    extra_args="--candidate_budget_mode fixed --candidate_budget_reference match_and"
    suffix="match_and"

    case "${strategy}" in
        and|or|abs_only|conf_only)
            ;;
        weighted_score)
            extra_args="${extra_args} --candidate_weight_alpha ${ALPHA} --candidate_score_normalization percentile"
            suffix="match_and_a${ALPHA}"
            ;;
        soft_fusion)
            extra_args="${extra_args} --soft_fusion_type weighted --candidate_weight_alpha ${ALPHA} --soft_abs_temperature 1.0 --soft_conf_temperature 1.0"
            suffix="match_and_a${ALPHA}"
            ;;
        rfas_rank)
            ;;
    esac

    output_dir="${OUTPUT_BASE}/${strategy}_${suffix}_seed${SEED}"
    run_experiment "${strategy}" "${extra_args}" "${output_dir}"
done

echo ""
echo "============================================"
echo " 消融实验完成!"
echo " 输出目录: ${OUTPUT_BASE}/"
ls -d ${OUTPUT_BASE}/*_seed${SEED}/ 2>/dev/null || echo "  (无)"
echo "============================================"
