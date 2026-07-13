#!/bin/bash
# 空间多样性候选选择器消融实验脚本 (Tanks&Temples)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${ROOT_DIR}/output/spatial_diversity_ablation"
SEED="${SEED:-0}"
SCENES="${SCENES:-train truck}"
DATASET_DIR="${DATASET_DIR:-/workspace/dataset/tt}"
ITERATIONS="${ITERATIONS:-30000}"
CANDIDATE_STRATEGY="${CANDIDATE_STRATEGY:-and}"

echo "============================================"
echo " Spatial Diversity Ablation (TT)"
echo " Seed: ${SEED}  Iterations: ${ITERATIONS}"
echo " Scenes: ${SCENES}"
echo "============================================"

mkdir -p "${OUTPUT_ROOT}"
AGGREGATE="${OUTPUT_ROOT}/metrics.json"

cat > "${AGGREGATE}" << 'JSONHEAD'
{
  "config": {
JSONHEAD
echo "    \"seed\": ${SEED}," >> "${AGGREGATE}"
echo "    \"iterations\": ${ITERATIONS}," >> "${AGGREGATE}"
echo "    \"candidate_strategy\": \"${CANDIDATE_STRATEGY}\"" >> "${AGGREGATE}"
cat >> "${AGGREGATE}" << 'JSONHEAD2'
  },
  "scenes": {
JSONHEAD2

FIRST=true
for SCENE in ${SCENES}; do
    SCENE_PATH="${DATASET_DIR}/${SCENE}"
    [ -d "${SCENE_PATH}" ] || { echo "SKIP: ${SCENE} not found"; continue; }
    echo "--- Scene: ${SCENE} ---"

    BASELINE_DIR="${OUTPUT_ROOT}/${SCENE}/baseline_seed${SEED}"
    SPATIAL_DIR="${OUTPUT_ROOT}/${SCENE}/spatial_voxel_seed${SEED}"

    # ---- Baseline ----
    echo "[1/2] Baseline (OFF)"
    T0=$(date +%s)
    python train.py -s "${SCENE_PATH}" -m "${BASELINE_DIR}" --eval --seed "${SEED}" \
        --data_device cpu --iterations "${ITERATIONS}" \
        --candidate_selection_strategy "${CANDIDATE_STRATEGY}" || { echo "FAIL baseline ${SCENE}"; continue; }
    T1=$(date +%s)
    BTIME=$((T1 - T0))
    python render.py -m "${BASELINE_DIR}" --skip_train
    python metrics.py -m "${BASELINE_DIR}"
    # ---- Spatial Voxel ----
    echo "[2/2] Spatial Voxel (ON)"
    T0=$(date +%s)
    python train.py -s "${SCENE_PATH}" -m "${SPATIAL_DIR}" --eval --seed "${SEED}" \
        --data_device cpu --iterations "${ITERATIONS}" \
        --candidate_selection_strategy "${CANDIDATE_STRATEGY}" \
        --enable_spatial_diversity --spatial_diversity_method voxel --spatial_voxel_size auto \
        || { echo "FAIL spatial ${SCENE}"; continue; }
    T1=$(date +%s)
    STIME=$((T1 - T0))
    python render.py -m "${SPATIAL_DIR}" --skip_train
    python metrics.py -m "${SPATIAL_DIR}"
    # ---- Python aggregation ----
    python3 << PYEOF
import json, csv, os

def load(path, csv_path, train_time, spatial):
    m = {"spatial_enabled": spatial, "train_time_seconds": train_time}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        ours = sorted([k for k in data if k.startswith("ours_")],
                      key=lambda k: int(k.split("_")[1]))
        if ours:
            b = data[ours[-1]]
            m["psnr"] = b.get("PSNR", "N/A")
            m["ssim"] = b.get("SSIM", "N/A")
            m["lpips"] = b.get("LPIPS", "N/A")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            m["final_gaussians"] = int(rows[-1].get("num_gaussians_after", 0))
        if spatial:
            sv = [(float(r["spatial_runtime_ms"]), float(r["spatial_selected_count"]),
                   float(r["spatial_occupied_voxels"]), float(r.get("spatial_jaccard",0)),
                   float(r.get("spatial_replaced",0)), r.get("spatial_voxel_size",""))
                  for r in rows if r.get("spatial_runtime_ms","").strip()]
            if sv:
                n = len(sv)
                m["mean_spatial_runtime_ms"] = round(sum(v[0] for v in sv)/n, 1)
                m["mean_spatial_selected_count"] = round(sum(v[1] for v in sv)/n, 1)
                m["mean_spatial_occupied_voxels"] = round(sum(v[2] for v in sv)/n, 1)
                m["mean_spatial_jaccard"] = round(sum(v[3] for v in sv)/n, 3)
                m["mean_spatial_replaced"] = round(sum(v[4] for v in sv)/n, 1)
                lvs = [v[5] for v in sv if v[5].strip()]
                m["final_spatial_voxel_size"] = float(lvs[-1]) if lvs else "N/A"
    return m

b = load("${BASELINE_DIR}/results.json", "${BASELINE_DIR}/candidate_selection_stats.csv", ${BTIME}, False)
s = load("${SPATIAL_DIR}/results.json", "${SPATIAL_DIR}/candidate_selection_stats.csv", ${STIME}, True)
with open("${OUTPUT_ROOT}/${SCENE}_metrics.json", "w") as f:
    json.dump({"baseline": b, "spatial_voxel": s}, f, indent=2)
print(f"  Metrics written: ${OUTPUT_ROOT}/${SCENE}_metrics.json")
PYEOF

    [ "${FIRST}" = false ] && echo "    ," >> "${AGGREGATE}"
    FIRST=false
    echo -n "    \"${SCENE}\": " >> "${AGGREGATE}"
    cat "${OUTPUT_ROOT}/${SCENE}_metrics.json" >> "${AGGREGATE}"
    echo "  Done: ${SCENE}"
done

cat >> "${AGGREGATE}" << 'JSONTAIL'
  }
}
JSONTAIL

echo ""
echo "=== Ablation Complete ==="
echo "Aggregate: ${AGGREGATE}"
python3 -m json.tool "${AGGREGATE}" > /dev/null 2>&1 && echo "JSON valid" || echo "JSON may be invalid"
