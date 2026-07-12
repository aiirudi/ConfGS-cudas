#!/usr/bin/env bash
# 致密化候选选择策略消融实验批量运行脚本
# 支持多场景：按策略顺序，每个策略依次跑完所有场景后再跑下一个策略。
#
# Usage:
#   bash scripts/run_candidate_ablation.sh [SEED] [SCENE...]
#
#   SEED:  随机种子 (默认: 0)
#   SCENE: 完整数据路径，如 /workspace/dataset/tt/train
#          或简写场景名（自动从 DATA_ROOT 查找对应 group）
#          简写支持的场景: bicycle flowers garden stump treehill
#                          bonsai counter kitchen room
#                          drjohnson playroom
#                          train truck
#
# 示例:
#   bash scripts/run_candidate_ablation.sh 0 /workspace/dataset/tt/train /workspace/dataset/tt/truck
#   bash scripts/run_candidate_ablation.sh 0 train truck
set -e

SEED="${1:-0}"
shift 2>/dev/null || true
SCENES=("${@}")
if [ ${#SCENES[@]} -eq 0 ]; then
    SCENES=(garden)
fi

ALPHA="0.5"
DATA_ROOT="/workspace/dataset"
OUTPUT_BASE="output/candidate_ablation"

# 场景到 group 的映射（与 test.py paramList 一致）
declare -A SCENE_GROUP
SCENE_GROUP[bicycle]="mipnerf"
SCENE_GROUP[flowers]="mipnerf"
SCENE_GROUP[garden]="mipnerf"
SCENE_GROUP[stump]="mipnerf"
SCENE_GROUP[treehill]="mipnerf"
SCENE_GROUP[bonsai]="mipnerf"
SCENE_GROUP[counter]="mipnerf"
SCENE_GROUP[kitchen]="mipnerf"
SCENE_GROUP[room]="mipnerf"
SCENE_GROUP[drjohnson]="db"
SCENE_GROUP[playroom]="db"
SCENE_GROUP[train]="tt"
SCENE_GROUP[truck]="tt"

# 解析场景路径：完整路径直接使用，简写名拼接 DATA_ROOT
resolve_scene() {
    local input="$1"
    # 如果包含 /，当作完整路径
    if [[ "$input" == */* ]]; then
        echo "$input"
        return
    fi
    # 简写：查 group
    local group="${SCENE_GROUP[$input]}"
    if [ -z "$group" ]; then
        echo "ERROR: 未知场景 '$input'，请使用完整路径或支持的简写: ${!SCENE_GROUP[*]}" >&2
        exit 1
    fi
    echo "${DATA_ROOT}/${group}/${input}"
}

# 显示名（用于输出目录）
scene_display_name() {
    local input="$1"
    if [[ "$input" == */* ]]; then
        basename "$input"
    else
        echo "$input"
    fi
}

echo "============================================"
echo " 候选选择策略消融实验"
echo " Seed:   ${SEED}"
echo " Alpha:  ${ALPHA}"
echo " Data:   ${DATA_ROOT}"
echo " Scenes:"
for scene in "${SCENES[@]}"; do
    echo "         $(resolve_scene "$scene")"
done
echo "============================================"

# 验证所有场景存在
for scene in "${SCENES[@]}"; do
    src="$(resolve_scene "$scene")"
    if [ ! -d "$src" ]; then
        echo "ERROR: 数据集不存在: $src"
        exit 1
    fi
done

mkdir -p "${OUTPUT_BASE}"

STRATEGIES=(and or abs_only conf_only weighted_score soft_fusion rfas_rank)

run_experiment() {
    local strategy="$1"
    local extra_args="$2"
    local output_dir="$3"
    local src_path="$4"
    local display="$5"

    echo ""
    echo "--- [${display}] ${strategy} ---"
    echo "Dir: ${output_dir}"

    python train.py \
        -s "${src_path}" \
        -m "${output_dir}" \
        --candidate_selection_strategy "${strategy}" \
        ${extra_args} \
        --eval \
        --seed "${SEED}"

    echo "--- Done: [${display}] ${strategy} ---"
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
        src="$(resolve_scene "$scene")"
        display="$(scene_display_name "$scene")"
        output_dir="${OUTPUT_BASE}/${display}/${strategy}_${suffix}_seed${SEED}"
        run_experiment "${strategy}" "${extra_args}" "${output_dir}" "${src}" "${display}"
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
        src="$(resolve_scene "$scene")"
        display="$(scene_display_name "$scene")"
        output_dir="${OUTPUT_BASE}/${display}/${strategy}_${suffix}_seed${SEED}"
        run_experiment "${strategy}" "${extra_args}" "${output_dir}" "${src}" "${display}"
    done
done

echo ""
echo "============================================"
echo " 消融实验完成!"
echo " 输出目录: ${OUTPUT_BASE}/"
for scene in "${SCENES[@]}"; do
    display="$(scene_display_name "$scene")"
    echo "  ${display}:"
    ls -d ${OUTPUT_BASE}/${display}/*_seed${SEED}/ 2>/dev/null || echo "    (无)"
done
echo "============================================"
