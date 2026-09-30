#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-kernel}"
image="${CONF_CUDA_DOCKER_IMAGE:-pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel}"
build_root="${CONF_CUDA_BUILD_ROOT:-/tmp/conf_cuda_numerics_build_20260930}"
baseline_root="${CONF_CUDA_BASELINE_ROOT:-/tmp/conf_cuda_baseline_source/build-lib/diff_gaussian_rasterization}"
deps_root="${CONF_CUDA_DEPS_ROOT:-/tmp/conf_cuda_deps}"
garden_root="${CONF_CUDA_GARDEN_ROOT:-/home/xzh/xzh/data/3dgs/mipnerf/garden}"
baseline_source_root="${CONF_CUDA_BASELINE_SOURCE_ROOT:-/tmp/conf_cuda_baseline}"
log_root="${CONF_CUDA_LOG_ROOT:-/tmp/conf_cuda_validation_logs}"
mkdir -p "$log_root"

run_container() {
    docker run --rm --gpus all \
        -v "$repo_root:/workspace:ro" \
        -v "$build_root/lib/diff_gaussian_rasterization:/opt/diff_gaussian_rasterization:ro" \
        -v "$baseline_root:/opt/conf_cuda_baseline:ro" \
        -v "$baseline_source_root:/tmp/conf_cuda_baseline:ro" \
        -v "$deps_root:/opt/conf_cuda_deps:ro" \
        -v "$garden_root:/data/garden:ro" \
        -v "$log_root:/validation_logs" \
        -w /workspace \
        -e PYTHONPATH=/opt:/opt/conf_cuda_deps:/workspace \
        -e CONF_CUDA_BASELINE_PKG=/opt/conf_cuda_baseline \
        "$image" "$@"
}

case "$mode" in
    build)
        mkdir -p "$build_root"
        docker run --rm --gpus all \
            -v "$repo_root:/workspace:ro" \
            -v "$build_root:/build" \
            -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
            -e MAX_JOBS="${MAX_JOBS:-4}" \
            "$image" bash -lc \
            'cd /workspace/submodules/diff-gaussian-rasterization && python setup.py build_ext --build-temp /build/temp --build-lib /build/lib && cp diff_gaussian_rasterization/__init__.py /build/lib/diff_gaussian_rasterization/__init__.py' \
            2>&1 | tee "$log_root/build.log"
        ;;
    kernel)
        run_container python tests/test_conf_cuda.py TestConfCudaKernel \
            2>&1 | tee "$log_root/kernel.log"
        ;;
    raster)
        run_container python scripts/compare_conf_cuda_baseline.py \
            --baseline-wrapper /opt/conf_cuda_baseline \
            --baseline-lib /opt/conf_cuda_baseline \
            --new-wrapper /workspace/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization \
            --new-lib /opt/diff_gaussian_rasterization \
            2>&1 | tee "$log_root/raster.log"
        ;;
    lifecycle)
        run_container python scripts/validate_conf_lifecycle.py \
            2>&1 | tee "$log_root/lifecycle.log"
        ;;
    checkpoint)
        run_container python test_checkpoint.py \
            2>&1 | tee "$log_root/checkpoint.log"
        run_container python validate_conf_stats.py \
            2>&1 | tee "$log_root/stats.log"
        ;;
    sanitizer)
        run_container compute-sanitizer --tool memcheck --error-exitcode 99 \
            python scripts/compare_conf_cuda_baseline.py \
            --child new \
            --wrapper /workspace/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization \
            --binary /opt/diff_gaussian_rasterization \
            --output /validation_logs/sanitizer_new_raster.pt \
            2>&1 | tee "$log_root/compute_sanitizer.log"
        ;;
    garden-zero|garden-default)
        if [[ "$mode" == garden-zero ]]; then
            conf_threshold=0.0
            run_name=garden-conf-zero
        else
            conf_threshold=0.85
            run_name=garden-conf-default
        fi
        run_container python train.py \
            -s /data/garden \
            -m "/validation_logs/$run_name" \
            -r 128 \
            --images images \
            --iterations 20 \
            --save_iterations 5 10 15 20 \
            --test_iterations 1000000 \
            --quiet \
            --seed 117 \
            --cams 1 \
            --lambda_amp_rec 0 \
            --densify_from_iter 0 \
            --densify_until_iter 510 \
            --densification_interval 5 \
            --conf_window_size 3 \
            --conf_min_views 2 \
            --conf_thr "$conf_threshold" \
            --candidate_selection_strategy conf_only \
            --candidate_budget_mode fixed \
            --candidate_budget_reference fixed_number \
            --candidate_fixed_budget 100 \
            --candidate_stats_enabled \
            2>&1 | tee "$log_root/$run_name.log"
        ;;
    *)
        echo "usage: $0 {build|kernel|raster|sanitizer|lifecycle|checkpoint|garden-zero|garden-default}" >&2
        exit 2
        ;;
esac
