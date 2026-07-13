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
    BASELINE_PSNR=$(python3 -c "import json; d=json.load(open('${BASELINE_JSON}')); print(d.get('PSNR', 'N/A'))" 2>/dev/null || echo "N/A")
    SPATIAL_PSNR=$(python3 -c "import json; d=json.load(open('${SPATIAL_JSON}')); print(d.get('PSNR', 'N/A'))" 2>/dev/null || echo "N/A")

    # 从 candidate_selection_stats.csv 提取空间统计
    BASELINE_CSV="${BASELINE_DIR}/candidate_selection_stats.csv"
    SPATIAL_CSV="${SPATIAL_DIR}/candidate_selection_stats.csv"

    SPATIAL_RUNTIME_MEAN="N/A"
    SPATIAL_SELECTED_MEAN="N/A"
    SPATIAL_OCCUPIED_MEAN="N/A"
    if [ -f "${SPATIAL_CSV}" ]; then
        SPATIAL_RUNTIME_MEAN=$(python3 -c "
import csv
with open('${SPATIAL_CSV}') as f:
    rows = [r for r in csv.DictReader(f) if r.get('spatial_runtime_ms','').strip()]
vals = [float(r['spatial_runtime_ms']) for r in rows]
print(f'{sum(vals)/len(vals):.1f}' if vals else 'N/A')
" 2>/dev/null || echo "N/A")
        SPATIAL_SELECTED_MEAN=$(python3 -c "
import csv
with open('${SPATIAL_CSV}') as f:
    rows = [r for r in csv.DictReader(f) if r.get('spatial_selected_count','').strip()]
vals = [float(r['spatial_selected_count']) for r in rows]
print(f'{sum(vals)/len(vals):.1f}' if vals else 'N/A')
" 2>/dev/null || echo "N/A")
        SPATIAL_OCCUPIED_MEAN=$(python3 -c "
import csv
with open('${SPATIAL_CSV}') as f:
    rows = [r for r in csv.DictReader(f) if r.get('spatial_occupied_voxels','').strip()]
vals = [float(r['spatial_occupied_voxels']) for r in rows]
print(f'{sum(vals)/len(vals):.1f}' if vals else 'N/A')
" 2>/dev/null || echo "N/A")
    fi

    # 写入汇总 JSON
    if [ "${FIRST_SCENE}" = false ]; then
        echo '    ,' >> "${AGGREGATE}"
    fi
    FIRST_SCENE=false
    cat >> "${AGGREGATE}" <<EOF
    "${SCENE}": {
      "baseline": {
        "psnr": "${BASELINE_PSNR}",
        "spatial_enabled": false
      },
      "spatial_voxel": {
        "psnr": "${SPATIAL_PSNR}",
        "spatial_enabled": true,
        "mean_spatial_runtime_ms": "${SPATIAL_RUNTIME_MEAN}",
        "mean_spatial_selected_count": "${SPATIAL_SELECTED_MEAN}",
        "mean_spatial_occupied_voxels": "${SPATIAL_OCCUPIED_MEAN}"
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
