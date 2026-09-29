# CUDA Conf validation record

This records the GPU validation of the rolling per-Gaussian Conf implementation at production commit `1535241` (`fix: harden rolling Conf camera identity and CUDA edge cases`). The earlier feature commit is `372cf3e`. Tests use the rebuilt native extension below; the repository's prebuilt submodule extension is not used.

## Reproduction environment

- GPU: NVIDIA A100 80 GB PCIe.
- Container: `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel`, PyTorch 2.1.2, Python 3.10, CUDA runtime 11.8. The host driver is 525.105.17.
- New extension: `/tmp/conf_cuda_sliding_build_20260930/lib/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so`, SHA-256 `402c5245124d353edbd62bb1f32702af878b02c761e237c9d4d1f8ef92de727a`.
- Baseline extension: `/tmp/conf_cuda_baseline_source/build-lib/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so`, SHA-256 `49b56e3903b1adcf4b9c8c78178413221c3cf18b244a6ee8dad05130b36fef3f`.
- Garden data is mounted read-only from `/home/xzh/xzh/data/3dgs/mipnerf/garden`.

The review-fix extension was rebuilt in the existing build tree with CUDA architecture 8.0. The image did not contain Ninja, so PyTorch used its distutils fallback and rebuilt the extension's translation units before linking into the path above. The final wrapper was copied from the source tree into that package directory.

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace:ro" \
  -v /tmp/conf_cuda_sliding_build_20260930:/build \
  -e TORCH_CUDA_ARCH_LIST=8.0 -e MAX_JOBS=4 \
  pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel \
  bash -lc 'cd /workspace/submodules/diff-gaussian-rasterization && python setup.py build_ext --build-temp /build/temp --build-lib /build/lib'
```

The complete compiler output is in `/tmp/conf_cuda_reviewfix_build.log`. The link completed successfully. Runtime tests set the rebuilt extension and temporary dependencies ahead of the source tree on `PYTHONPATH` through `scripts/run_conf_cuda_validation.sh`. The runner accepts `CONF_CUDA_BUILD_ROOT`, `CONF_CUDA_BASELINE_ROOT`, `CONF_CUDA_BASELINE_SOURCE_ROOT`, `CONF_CUDA_DEPS_ROOT`, `CONF_CUDA_GARDEN_ROOT` and `CONF_CUDA_LOG_ROOT` overrides.

## Results

| Area | Command or entry point | Result |
|---|---|---|
| Rolling kernel | `scripts/run_conf_cuda_validation.sh kernel` | PASS. Three Gaussians were checked after each of 269 observations against an independent Python deque using float64 sums at W=2, 3 and 5. The sequence covers per-row visibility, delayed visibility, refresh recency, eviction, zero vectors, invalid markers and repeated cameras. P=0/P=1, negative camera IDs, shape rejection and accumulation on a non-default producer/consumer stream also passed. |
| Checkpoint/statistics compatibility | `validate_conf_stats.py`, then `test_checkpoint.py` | PASS. Version-2 state round-trips, legacy states reset to empty history, mismatched camera mappings reject, and a stale topology generation rejects. |
| Existing selector/lifecycle tests | `tests/test_candidate_selector.py`, `tests/test_spatial_diversity.py`, and the three functions in `tests/test_ac1_disabled_equivalence.py` | PASS: 18/18, 21/21 and 3/3. The cached image has no `pytest`; the existing direct script entry points and functions were used. The selector seed-help check passed after adding the rebuilt extension to `PYTHONPATH`. |
| Model topology and ranking helpers | `scripts/run_conf_cuda_validation.sh lifecycle` | PASS. Pruning preserves exact survivor rows; appended and split children start empty; real long-axis split removes the selected parent; zero-budget/no-split boundaries preserve history; the Conf gate still applies at iteration 14600; same-size stale samples reject; checkpoint restore preserves the next eviction; RFAS/EAS/fusion helper ASTs match the saved pre-Conf `train.py`. |
| Baseline rendering and gradients | `scripts/run_conf_cuda_validation.sh raster` | PASS. Four SH/precomputed-color × scale/rotation/covariance modes matched the baseline at 32×32. Image max error was 0 for all modes. Maximum actual-parameter gradient errors were at most 3.82e-6; most were 0. Conf-enabled versus Conf-disabled image and parameter gradients had max error 0. Debug-on versus ordinary rendering had max image error 0 and max gradient error 1.91e-6. |
| Projection VJP and visibility | Included in `raster` | PASS. Analytic stabilized projection VJP versus emitted CUDA samples differed by at most 5.17e-7. Independent float64 central differences on rotated/translated cameras differed from the analytic VJP by at most 3.69e-10. The 31×23 partial-tile case passed. Six visible zero-upstream samples remained valid zero vectors. A fully occluded rear Gaussian retained positive radius but did not advance its history. Frustum-culled and all-culled B=0 cases produced no Conf observations. |
| CUDA memory checks | `scripts/run_conf_cuda_validation.sh sanitizer` | PASS. `compute-sanitizer --tool memcheck` instrumented the rebuilt rasterizer on 31×23 partial tiles, zero upstream gradients, all-culled B=0 and the occlusion case: `ERROR SUMMARY: 0 errors`. The baseline binary was not run under the sanitizer. |
| Garden short training, `conf_thr=0` | `scripts/run_conf_cuda_validation.sh garden-zero` | PASS. Twenty iterations crossed four densification boundaries (5, 10, 15, 20). Each boundary selected the fixed cap of 100 eligible candidates and split 100 parents. Gaussian counts after the boundaries were 138866, 138966, 139066 and 139166, starting from 138766. Displayed EMA losses at iterations 10 and 20 were 0.18006 and 0.14377. |
| Garden short training, default `conf_thr=0.85` | `scripts/run_conf_cuda_validation.sh garden-default` | PASS. The same four boundaries had 254, 359, 373 and 400 Conf-eligible candidates; the fixed 100-candidate cap yielded 100 splits at each boundary and the same count progression through 139166. Displayed EMA losses at iterations 10 and 20 were 0.18410 and 0.15216. |

The Garden runs used `-r 128`, 20 iterations, `--densify_from_iter 0`, `--densify_until_iter 510`, `--densification_interval 5`, `--cams 1`, window size 3, minimum views 2 and fixed candidate budget 100. The zero threshold run is a controlled split smoke test; the default-threshold run also produced candidates and splits. CSV rows and PLY snapshots are under `/tmp/conf_cuda_validation_logs/garden-conf-zero` and `/tmp/conf_cuda_validation_logs/garden-conf-default`.

## Detailed logs

The concise command output is retained under `/tmp/conf_cuda_validation_logs/`:

- `kernel.log`, `existing_validations.log`, `existing_cpu_tests_final.log`
- `lifecycle.log`, `raster.log`, `compute_sanitizer.log`
- `garden-conf-zero.log`, `garden-conf-default.log`
- Per-boundary candidate counts: `garden-conf-zero/candidate_selection_stats.csv` and `garden-conf-default/candidate_selection_stats.csv`

These are short correctness and integration smokes, not a full quality or speed benchmark. No final PSNR, long-run stability, or throughput claim was measured. Checkpoint continuation was verified at the model-state level, including preservation of the next eviction; a full CLI training-resume run was not part of this validation.
