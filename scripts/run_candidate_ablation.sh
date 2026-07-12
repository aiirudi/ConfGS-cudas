#!/usr/bin/env bash
# 致密化候选选择策略消融实验批量运行脚本
# 支持多场景：按策略顺序，每个策略依次跑完所有场景后再跑下一个策略。
#
# Usage:
#   bash scripts/run_candidate_ablation.sh [SEED] [SCENE...]
#
#   SEED:   随机种子 (默认: 0)
#   SCENE:  数据集名称 (默认: garden)，可多个，如: train truck
#
# 示例:
#   bash scripts/run_candidate_ablation.sh 0 train truck
#   bash scripts/run_candidate_ablation.sh 0 garden bicycle flowers
set -e

SEED="${1:-0}"
shift 2>/dev/null || true
SCENES=("${@}")
if [ ${#SCENES[@]} -eq 0 ]; then
    SCENES=(garden)
fi

ALPHA="0.5"
OUTPUT_BASE="output/candidate_ablation"

echo "============================================"
echo " 候选选择策略消融实验"
echo " Scenes: ${SCENES[*]}"
echo " Seed:   ${SEED}"
echo " Alpha:  ${ALPHA}"
echo "============================================"

# 验证所有场景存在
for scene in "${SCENES[@]}"; do
    if [ ! -d "data/${scene}" ]; then
        echo "ERROR: 数据集不存在: data/${scene}"
        echo "可用数据集:"
        ls data/ 2>/dev/null || echo "  (无)"
        exit 1
    fi
done

mkdir -p "${OUTPUT_BASE}"

STRATEGIES=(and or abs_only conf_only weighted_score soft_fusion rfas_rank)

run_experiment() {
    local strategy="$1"
    local extra_args="$2"
    local output_dir="$3"
    local scene="$4"

    echo ""
    echo "--- [${scene}] ${strategy} ---"
    echo "Dir: ${output_dir}"

    python train.py \
        -s "data/${scene}" \
        -m "${output_dir}" \
        --candidate_selection_strategy "${strategy}" \
        ${extra_args} \
        --eval \
        --seed "${SEED}"

    echo "--- Done: [${scene}] ${strategy} ---"
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

    for scene in "${SCENES[@]}"; do
        output_dir="${OUTPUT_BASE}/${scene}/${strategy}_${suffix}_seed${SEED}"
        run_experiment "${strategy}" "${extra_args}" "${output_dir}" "${scene}"
    done
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

    for scene in "${SCENES[@]}"; do
        output_dir="${OUTPUT_BASE}/${scene}/${strategy}_${suffix}_seed${SEED}"
        run_experiment "${strategy}" "${extra_args}" "${output_dir}" "${scene}"
    done
done

echo ""
echo "============================================"
echo " 消融实验完成!"
echo " 输出目录: ${OUTPUT_BASE}/"
for scene in "${SCENES[@]}"; do
    echo "  ${scene}:"
    ls -d ${OUTPUT_BASE}/${scene}/*_seed${SEED}/ 2>/dev/null || echo "    (无)"
done
echo "============================================"
