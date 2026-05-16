# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is the official implementation of **Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering** (Deng et al., arXiv:2508.12313). The pipeline forks the original 3DGS / TamingGS code; novel logic lives almost entirely inside `train.py` and `scene/gaussian_model.py`. The CUDA rasterizer in `submodules/diff-gaussian-rasterization` is a customized fork that returns extra per-Gaussian statistics (`accum_weights`, `accum_count`, `accum_blend`, `accum_dist`, `gaussian_centers`, `gaussian_depths`, `gaussian_radii`) and accepts a `pixel_weights` map — the densification scores depend on these outputs, so the stock 3DGS rasterizer is **not** a drop-in replacement.

## Environment / build

The conda environment is pinned in `environment.yml` (Python 3.7.13, PyTorch 1.12.1, CUDA 11.6). The CUDA submodules (`diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`) are shipped as zips and must be unpacked before `pip install`:

```bash
unzip submodules-speedy.zip          # preferred — Speedy-Splat compact-box kernel, ~15% faster
# or: unzip submodules.zip           # vanilla variant
conda env create -f environment.yml
conda activate improved_gs
```

Caveats from `README.md` worth respecting:
- `pip` must not be too new or the CUDA submodule builds break — `environment.yml` pins `pip=22.3.1`.
- `numpy` must stay on 1.x; `pip` will sometimes pull 2.x and silently break `plyfile`/`simple-knn`. Downgrade manually.
- The user is expected to have a working stock 3DGS install first; this repo reuses that toolchain.

## Common commands

```bash
# Train one scene (COLMAP layout under -s, output dir under -m)
python train.py -s data/<scene> -m output/<scene> --eval [--budget N] [--cams 10]

# Render train+test splits with a saved checkpoint
python render.py -m output/<scene> [--iteration 30000] [--skip_train] [--skip_test]

# PSNR / SSIM / LPIPS on rendered test images
python metrics.py -m output/<scene>           # test split (writes results.json, per_view.json)
python metrics-train.py -m output/<scene>     # same metrics on the train split

# FPS benchmark
python bench_fps.py -m output/<scene> --iteration 30000 --num_frames 30000

# End-to-end batch driver (Windows-style paths inside; edit before reuse)
python test.py
```

`run_train.sh`, `run_render.sh`, `run_metrics.sh`, and `cal_fps.sh` are thin wrappers around the commands above with hard-coded `MODEL_DIR` / `DATASET` paths — update the paths before running them.

`render.py`, `metrics.py`, and `bench_fps.py` use `get_combined_args` (in `arguments/__init__.py`), which **reads `<model_path>/cfg_args` and merges it with the CLI**. Train must succeed first or these tools have nothing to load.

## Key training hyperparameters

Defaults live in `arguments/__init__.py` (`OptimizationParams`). The ones the paper's techniques actually steer:

- `--budget` — target Gaussian count. Default `1_777_778` (mip-NeRF 360). Per-scene values in `budget.txt`; the "small" column is 40% of normal and reproduces on a 12 GB GPU. Note: `train.py` currently **overrides `opt.budget` based on the initial point cloud size** (`<120k init points → 1_250_000`, else `1_777_778`); CLI `--budget` is overwritten unless you remove that block.
- `--cams` (in `train.py` only, default `10`) — number of cameras sampled per densification step for EAS/RFAS scoring. `-1` uses every training camera. Inside `iteration % 3000 == 400 and iteration < 9000`, the code forces the full camera set regardless.
- `--data_device` — set to `"cpu"` for <12 GB GPUs (keeps GT images off-GPU).
- `--optimizer_type` — `"default"` (recommended) or `"sparse_adam"`. The `sparse_adam` path is functionally equivalent to enabling MU-style strided SH updates from iter 0; per the README it ships disabled because it hurts quality.
- Frequency-regularization weights: `--lambda_efre_wl`, `--lambda_efre_wh` (both 0.1), `--regulation_convert_iter` (T₀ = 7000).
- LAS knobs: `--split_distance` (0.45), `--opacity_reduction` (0.6).

`--websockets` starts the WS-based viewer at `--ip:--port` (default `127.0.0.1:6009`). The frontend is the static page in `web_viewer/` (`render.html` + `app.js`); open it in a browser while training and type a camera ID to stream that view.

## Component provenance — upstream vs. user-added

This repo forks ImprovedGS. **Knowing which components are upstream vs. user-added matters for every ablation request** — upstream pieces are baseline scaffolding and stay on; only the user-added pieces are legitimate ablation knobs.

| Component | Origin | What it is | Implementation site |
| --- | --- | --- | --- |
| **EAS** (Edge-Aware Score) | upstream ImprovedGS | edge-map weighted per-Gaussian importance | `train.py::compute_edge_score`; gradient hook in `gaussian_model.py::add_densification_stats_abs` |
| **LAS** (Long Axis Split) | upstream ImprovedGS | replaces clone+split — samples parents by score, splits only along the longest scaling axis, then prunes parents | `gaussian_model.py::long_axis_split` (algorithm) ← `densify_and_prune_Improved` ← `train.py:~220` |
| **RAP** (quantile opacity prune after reset) | upstream ImprovedGS | `only_prune(q, percent=True)` treats `q` as a quantile of the opacity distribution; called 300 iter after each opacity reset while `iter<9000` | `gaussian_model.py::only_prune` (`percent=True` branch) ← `train.py:~227` |
| **MU** (Multi-step Update) | upstream ImprovedGS | strided optimizer step schedule (every iter ≤15k, every 5 ≤22.5k, every 20 after) | `train.py:~234-256` |
| **GC** (Growth Control) | upstream ImprovedGS | per-densification budget ramp `int(sqrt(rate)·opt.budget)` | `train.py:~210-217` |
| **Conf** (gradient-direction conflict score) | **user-added** | `conf = 1 - \|Σg\| / Σ\|g\|` over xy view-space gradient; intended to gate splitting | `gaussian_model.py` (`xyz_gradient_vec_accum` / `xyz_gradient_mag_accum`, plus mask code inside `densify_and_prune_Improved`) |
| **RFAS** (Residual-Frequency Aware Score) | **user-added** | per-view `|gt−render|` → 3×3 Laplacian → per-Gaussian high-frequency residual score, fused with EAS via `fuse_importance_scores` | `train.py::compute_rf_score` / `compute_rf_score1` |
| **RFDAR** | **user-added** | residual-frequency adaptive regularization (currently toggled off in the loss block — see the `# 关闭 RFDAR` marker) | `train.py` densification-window loss block; FFT helpers in `compute_frequency_regularization` / `compute_frequency_discrepancies` |

When the user asks for an ablation ("只启用 X 和 Y"), **only toggle among {Conf, RFAS, RFDAR}**. EAS/LAS/RAP/MU/GC are part of the baseline and stay enabled.

Every component has a `# XX 实现` marker at its primary implementation site (the table's "Implementation site" column points to them), so a `grep "实现" train.py scene/gaussian_model.py` is enough to locate them.

## Architecture — what reading a single file won't tell you

### Custom rasterizer contract
`gaussian_renderer/__init__.py::render` returns a dict whose extra keys (`accum_weights`, `accum_count`, `accum_blend`, `accum_dist`, `gaussian_centers`, `gaussian_depths`, `gaussian_radii`, `visibility_filter`) are produced by the patched CUDA kernel and consumed by EAS/RFAS scoring and by `utils/taming_utils.py::compute_gaussian_score`. `pixel_weights` is forwarded into the rasterizer so per-pixel importance weighting (e.g. edge maps) directly modulates the accumulators. Anything that re-renders for scoring **must** call this `render()`, not the stock 3DGS one.

### Densification loop (`train.py::training`)
The loop runs every `densification_interval` iterations between `densify_from_iter` and `densify_until_iter`. Per step:
1. Sample `args.cams` viewpoints from the cycling `my_viewpoint_stack` (with their cached edge maps in `edges_stack`, precomputed once via `get_edges()` over all training views).
2. `compute_edge_score(...)` → **EAS**: re-renders each view with `pixel_weights = edge_map`, normalizes `accum_weights`, accumulates per-Gaussian importance over visible points.
3. `compute_rf_score(...)` → **RFAS**: per view, computes `|gt - render|`, runs a 3×3 Laplacian (`compute_high_freq_residual_log`), and aggregates into a per-Gaussian high-frequency residual score.
4. `fuse_importance_scores(eas, rfas, mode='product')` → fused score; modes `'geometric' | 'product' | 'weighted'` are all in the file.
5. **Growth Control**: budget for this densification call ramps as `int(sqrt(rate) * opt.budget)` where `rate = (iter - from) / (until - 500 - from)`, clamped to `opt.budget`.
6. `gaussians.densify_and_prune_Improved(scores, 0.005, budget, opt, iteration, opt.budget)` — calls **LAS** (`long_axis_split`) instead of the original clone+split. Note: after iter 14500, `scores` is overridden by `xyz_gradient_accum / denom` (i.e. it falls back to gradient magnitude); pruning is gated by `iteration < 14900`.

### `scene/gaussian_model.py` — what's non-stock
- Two gradient accumulators: `xyz_gradient_accum` (magnitude, populated by `add_densification_stats_abs` from `viewspace_point_tensor.grad[:, 2:]`) and `xyz_gradient_vec_accum` / `xyz_gradient_mag_accum` (xy vector + magnitude, used to compute the conflict score `conf = 1 - |Σg| / Σ|g|`). The `conf_min_views` / `conf_thr` mask is wired but currently **disabled** (the union/intersection branches are commented out inside `densify_and_prune_Improved`). If you re-enable them, also re-enable the matching reset paths in `only_prune` and `densification_postfix`.
- `long_axis_split`: samples `budget` parents proportional to scores (`torch.multinomial`), splits each along its **longest scaling axis only**, places two children at `±split_distance · 3·std_long` along that axis (rotated by the parent), rescales the long axis to `(1-rate)/√(1-rate²)` and shrinks all axes by `√(1-rate²)`, multiplies opacity by `opacity_reduction`, then prunes the parents. SH features and rotation are inherited.
- `only_prune(min_opacity, percent=False)` doubles as quantile pruning when `percent=True` (treats `min_opacity` as a quantile of the opacity distribution). This is **RAP**: called as `only_prune(0.2, True)` 300 iters after each opacity reset (only while `iter < 9000`). A separate `only_prune(0.02)` is invoked at iter 300 to prune the initial point cloud, and `reset_opacity(0.05)` is called every `opacity_reset_interval` (3000) instead of the original 0.005 floor.
- Two optimizers — `optimizer` (xyz/scaling/rotation/opacity/f_dc) and `shoptimizer` (f_rest, SH coefficients) — both stepped together. The `cat_tensors_to_optimizer` / `_prune_optimizer` helpers must keep them in sync; any new parameter must be registered in **both** during `training_setup` and handled in **both** during densification/pruning.

### Multi-step Update (MU) schedule in `train.py`
With `optimizer_type == "default"`:
- iters ≤ 15000 → step every iter
- 15000 < iter ≤ 22500 → step every 5 iters
- iter > 22500 → step every 20 iters

Both `optimizer` and `shoptimizer` follow the schedule. This is why the SH learning rate (`shfeature_lr`, default 0.005) is higher than `feature_lr` (0.0025) — fewer updates per epoch.

### Frequency Residual Regulation (FA)
Active only inside the densification window. `compute_frequency_regularization(e_image, iter, T0=7000, ...)` FFT-shifts the residual `|gt - render|`, splits magnitude/phase into a low band (`r ≤ D₀ = min(H,W)·0.15`) and a high band that anneals from `D₀` to `D_max = min(H,W)·0.5` between `T0` and `densify_until_iter`. The high-frequency term is gated on `iter > T0`.

### Initial-point heuristic in `train.py`
```python
init_point_nums = gaussians.get_xyz.shape[0]
if init_point_nums < 120000: opt.budget = 1250000
else:                        opt.budget = 1777778
```
This **silently overrides any `--budget` flag** before training starts. If the per-scene values in `budget.txt` matter, edit this block out.

## Data layout

`Scene.__init__` (`scene/__init__.py`) detects:
- COLMAP — `<source_path>/sparse` exists, with `images/` for RGB. Standard mip-NeRF 360 / Tanks&Temples / DeepBlending layout.
- Blender / NeRF synthetic — `<source_path>/transforms_train.json` exists.

`test.py` documents the expected layout for the 13-scene benchmark:
```
data/<scene>/images
data/<scene>/sparse
```
Existing models reload via `point_cloud/iteration_<N>/point_cloud.ply` plus the saved `cameras.json` and `cfg_args` files in the model dir.

## Output viewing

The trained `point_cloud.ply` files contain only the standard 3DGS attributes (no extra parameters), so any 3DGS viewer works — the README recommends [SuperSplat](https://superspl.at/editor). The bundled `web_viewer/` is for **live training preview only** (it talks to `network_gui_ws.py` over WebSockets while `train.py --websockets` is running); it is not a standalone viewer for saved scenes.
