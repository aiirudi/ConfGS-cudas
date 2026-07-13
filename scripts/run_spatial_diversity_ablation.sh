#!/bin/bash
# 空间多样性候选选择器消融实验脚本 (Tanks&Temples)
# 比较 baseline (禁用空间模块) vs spatial_voxel (启用空间模块)
#
# Usage: bash scripts/run_spatial_diversity_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_ROOT="${ROOT_DIR}/output/spatial_diversity_ablation"

# ---- 配置 ----
SEED="${SEED:-0}"
SCENES="${SCENES:-train truck}"
DATASET_DIR="${DATASET_DIR:-/workspace/dataset/tt}"
ITERATIONS="${ITERATIONS:-30000}"

# 候选选择策略 (与基线保持一致)
CANDIDATE_STRATEGY="${CANDIDATE_STRATEGY:-and}"

echo "============================================"
echo " 空间多样性消融实验"
echo " Seed:       ${SEED}"
echo " Scenes:     ${SCENES}"
echo " Iterations: ${ITERATIONS}"
echo " Strategy:   ${CANDIDATE_STRATEGY}"
echo " Output:     ${OUTPUT_ROOT}"
echo "============================================"
echo ""

mkdir -p "${OUTPUT_ROOT}"

# 汇总 JSON
AGGREGATE="${OUTPUT_ROOT}/metrics.json"
echo '{' > "${AGGREGATE}"
echo '  "config": {' >> "${AGGREGATE}"
echo "    \"seed\": ${SEED}," >> "${AGGREGATE}"
echo "    \"iterations\": ${ITERATIONS}," >> "${AGGREGATE}"
echo "    \"candidate_strategy\": \"${CANDIDATE_STRATEGY}\"" >> "${AGGREGATE}"
echo '  },' >> "${AGGREGATE}"
echo '  "scenes": {' >> "${AGGREGATE}"

FIRST_SCENE=true
for SCENE in ${SCENES}; do
    SCENE_PATH="${DATASET_DIR}/${SCENE}"
    if [ ! -d "${SCENE_PATH}" ]; then
        echo "WARNING: scene ${SCENE} not found at ${SCENE_PATH}, skipping"
        continue
    fi

    echo ""
    echo "--- Scene: ${SCENE} ---"

    # ---- Baseline (禁用空间模块) ----
    BASELINE_DIR="${OUTPUT_ROOT}/${SCENE}/baseline_seed${SEED}"
    echo "[1/2] Training baseline (spatial OFF)..."
    python train.py \
        -s "${SCENE_PATH}" \
        -m "${BASELINE_DIR}" \
        --eval \
        --seed "${SEED}" \
        --data_device cpu \
        --iterations "${ITERATIONS}" \
        --candidate_selection_strategy "${CANDIDATE_STRATEGY}" || {
        echo "ERROR: baseline training failed for ${SCENE}"
        continue
    }

    echo "  Rendering baseline..."
    python render.py -m "${BASELINE_DIR}" --skip_train || true

    echo "  Computing metrics..."
    python metrics.py -m "${BASELINE_DIR}" || true
    BASELINE_JSON="${BASELINE_DIR}/results.json"

    # ---- Spatial Voxel (启用空间模块) ----
    SPATIAL_DIR="${OUTPUT_ROOT}/${SCENE}/spatial_voxel_seed${SEED}"
    echo "[2/2] Training spatial_voxel (spatial ON)..."
    python train.py \
        -s "${SCENE_PATH}" \
        -m "${SPATIAL_DIR}" \
        --eval \
        --seed "${SEED}" \
        --data_device cpu \
        --iterations "${ITERATIONS}" \
        --candidate_selection_strategy "${CANDIDATE_STRATEGY}" \
        --enable_spatial_diversity \
        --spatial_diversity_method voxel \
        --spatial_voxel_size auto || {
        echo "ERROR: spatial_voxel training failed for ${SCENE}"
        continue
    }

    echo "  Rendering spatial_voxel..."
    python render.py -m "${SPATIAL_DIR}" --skip_train || true

    echo "  Computing metrics..."
    python metrics.py -m "${SPATIAL_DIR}" || true
    SPATIAL_JSON="${SPATIAL_DIR}/results.json"

    # ---- 收集汇总指标 ----
    echo "  Collecting aggregate metrics..."

    # 从 results.json 提取 PSNR/SSIM/LPIPS
    _extract_metric() {
        python3 -c "import json; d=json.load(open('${1}')); print(d.get('${2}', 'N/A'))" 2>/dev/null || echo "N/A"
    }
    BASELINE_PSNR=$(_extract_metric "${BASELINE_JSON}" "PSNR")
    BASELINE_SSIM=$(_extract_metric "${BASELINE_JSON}" "SSIM")
    BASELINE_LPIPS=$(_extract_metric "${BASELINE_JSON}" "LPIPS")
    SPATIAL_PSNR=$(_extract_metric "${SPATIAL_JSON}" "PSNR")
    SPATIAL_SSIM=$(_extract_metric "${SPATIAL_JSON}" "SSIM")
    SPATIAL_LPIPS=$(_extract_metric "${SPATIAL_JSON}" "LPIPS")

    # 从 candidate_selection_stats.csv 提取空间统计
    SPATIAL_CSV="${SPATIAL_DIR}/candidate_selection_stats.csv"

    _csv_mean() {
        if [ ! -f "${SPATIAL_CSV}" ]; then echo "N/A"; return; fi
        python3 -c "
import csv
with open('${SPATIAL_CSV}') as f:
    rows = [r for r in csv.DictReader(f) if r.get('${1}','').strip()]
vals = [float(r['${1}']) for r in rows]
print(f'{sum(vals)/len(vals):.1f}' if vals else 'N/A')
" 2>/dev/null || echo "N/A"
    }
    SPATIAL_RUNTIME_MEAN=$(_csv_mean "spatial_runtime_ms")
    SPATIAL_SELECTED_MEAN=$(_csv_mean "spatial_selected_count")
    SPATIAL_OCCUPIED_MEAN=$(_csv_mean "spatial_occupied_voxels")
    SPATIAL_JACCARD_MEAN=$(_csv_mean "spatial_jaccard")
    SPATIAL_REPLACED_MEAN=$(_csv_mean "spatial_replaced")

    # 最终 Gaussian 数量 (从 ply 文件推断或从 CSV)
    FINAL_GS_BASELINE=$(ls "${BASELINE_DIR}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" 2>/dev/null && echo "see_ply" || echo "N/A")
    FINAL_GS_SPATIAL=$(ls "${SPATIAL_DIR}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" 2>/dev/null && echo "see_ply" || echo "N/A")

    # 写入汇总 JSON
    if [ "${FIRST_SCENE}" = false ]; then
        echo '    ,' >> "${AGGREGATE}"
    fi
    FIRST_SCENE=false
    cat >> "${AGGREGATE}" <<EOF
    "${SCENE}": {
      "baseline": {
        "psnr": "${BASELINE_PSNR}",
        "ssim": "${BASELINE_SSIM}",
        "lpips": "${BASELINE_LPIPS}",
        "spatial_enabled": false
      },
      "spatial_voxel": {
        "psnr": "${SPATIAL_PSNR}",
        "ssim": "${SPATIAL_SSIM}",
        "lpips": "${SPATIAL_LPIPS}",
        "spatial_enabled": true,
        "mean_spatial_runtime_ms": "${SPATIAL_RUNTIME_MEAN}",
        "mean_spatial_selected_count": "${SPATIAL_SELECTED_MEAN}",
        "mean_spatial_occupied_voxels": "${SPATIAL_OCCUPIED_MEAN}",
        "mean_spatial_jaccard": "${SPATIAL_JACCARD_MEAN}",
        "mean_spatial_replaced": "${SPATIAL_REPLACED_MEAN}"
      }
    }
EOF

    echo "  Done: ${SCENE}"
done

echo ""
echo '  }' >> "${AGGREGATE}"
echo '}' >> "${AGGREGATE}"

echo ""
echo "============================================"
echo " 消融实验完成！"
echo " 汇总结果: ${AGGREGATE}"
echo "============================================"
