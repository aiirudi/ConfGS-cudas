# CUDA multi-view Conf design

## Scope and invariant

Replace the training densification signal and its accumulation with CUDA Conf. Preserve rendering, photometric/PAIR losses, parameter gradients, optimizers, LAS geometry, pruning policy, EAS, RFAS, and EAS/RFAS fusion. Existing uncommitted changes in `arguments/__init__.py`, `test.py`, and `train.py` belong to the user and must be preserved.

“Change backward to Conf” means computing the densification statistic during the rasterizer backward pass. A scalar conflict score cannot replace `dL/dxyz`, `dL/dopacity`, or other derivatives: it has neither their dimensions nor their direction. The actual optimization gradient remains the derivative of the actual loss. Conf is a detached side-channel produced by backward and consumed after backward.

## Existing implementation and exact coordinates

The current chain is `train.py: loss.backward()` → `add_densification_stats_abs()` → Python `_compute_ndc_vjp_world()` → vector/norm accumulation → `densify_and_prune_Improved()` → candidate selector → LAS.

`PerGaussianRenderCUDA` accumulates signed pixel contributions in `dL_dmean2D.x/y`; the extra `.z/.w` channels sum `fabs(tmp_x/y)` for AbsGS. The x/y values **already include W/2 and H/2** and are derivatives with respect to NDC coordinates, not pixel coordinates. Applying another focal-length or image-size factor would double-scale them.

`backward.cu::preprocessCUDA` already computes the required pure projection pullback in local variable `dL_dmean`, before adding it to `dL_dmeans` and before SH derivatives. Export that local value. Do not use final `means3D.grad`: it also contains covariance and SH contributions, unlike the intended Conf statistic.

PyTorch camera matrices use row-vector convention: `q = [xyz,1] @ full_proj_transform`. CUDA reads the same memory using `transformPoint4x4`. For `r = 1 / (q.w + 1e-7)` the **existing renderer's** exact projection Jacobian is:

```
jx[k] = M[k,0] * r - M[k,3] * q.x * r*r
jy[k] = M[k,1] * r - M[k,3] * q.y * r*r
g_world[k] = jx[k] * g_ndc.x + jy[k] * g_ndc.y
```

Copying the existing CUDA intermediate is preferable to re-deriving a slightly different projection: the old Python helper omits the renderer's `+1e-7`. The forward projection itself must not change. Tests must use the renderer's actual stabilized projection when comparing exact derivatives.

## Multi-view statistic and validity

For Gaussian i and distinct eligible camera c, let `g_ic` be the signed world-space projection gradient from that camera's training backward. Maintain sufficient statistics:

```
S_i = sum_c g_ic                 # float32 (N,3)
M_i = sum_c norm(g_ic)           # float32 (N,1)
K_i = number of valid cameras   # int32   (N,1)
Conf_i = clamp(1 - norm(S_i)/M_i, 0, 1)
```

Set Conf to zero if `K < 2`, `M <= 0`, or any necessary statistic is nonfinite. Candidate validity also requires `K >= conf_min_views`. Preserve the user's `conf_min_views=2` and `conf_thr=0.85`; use `>=` for the threshold.

Do not compute `1 - norm(S)/(M+1e-6)`: this biases small coherent gradients toward Conf=1. Guard division instead of perturbing a positive denominator. Zero vectors contain no directional evidence and must not increment K. This fixes the existing zero-gradient false-positive case.

A sample is valid only for `radii > 0`, positive finite clip-w above the existing validity floor (1e-8), finite signed gradients and finite positive world norm. A frustum-visible but fully occluded Gaussian normally has zero derivative and is excluded naturally. Use `hypotf`-style stable norm calculation to avoid overflow from directly squaring large finite components. CUDA initializes all output rows to zero; invalid or skipped rows remain zero. Accumulation must reject nonfinite sample values, and score finalization must fail closed for nonfinite sums. No arbitrary gradient clipping is needed.

The formula is the existing **magnitude-weighted directional cancellation** statistic in a common coordinate frame. It is invariant to a common positive loss scaling, bounded, and gives 0 for aligned gradients, 1 for exactly cancelling equal opposite gradients, and `1 - sqrt(2)/2` for equal orthogonal gradients. It does not claim to eliminate dependence on unequal gradient magnitudes: those magnitudes intentionally represent relative optimization pressure. Unit-normalizing each view would change the method and existing threshold meaning, amplify tiny noisy observations, and is not the default.

## Distinct-camera streaming policy

Recommended default: retain the current randomized training-camera traversal, and admit **only the first backward observation from each camera within a densification interval**. A Python set of camera keys is sufficient. Every admitted observation increments K only where its per-Gaussian sample is valid. Thus K counts distinct informative cameras, never iterations, tiles, pixels, or repeated renders.

Use the training camera's stable `uid`, or an explicit dataset index assigned once, as the key. Do not use only the last camera id: repeats may be nonconsecutive. Do not use the RFAS `camlist` length as the view count. Repeated backward on the same render and repeated training-camera visits must not inflate K. RFAS/EAS/inference renders must never enter this accumulator.

This costs O(N) GPU memory plus O(V_interval) CPU set entries; no per-camera N-vector cache, N×V bitset, all-view rerender, extra loss backward, or per-Gaussian Python loop is needed. Existing training samples cameras without replacement within an epoch, so repeats mainly arise across epoch boundaries or small datasets. Deduplication also prevents conflicting gradients from different training times of one camera from falsely counting as cross-view evidence.

Tradeoffs must be documented honestly. Statistics combine observations from nearby training steps, not a frozen model snapshot; a short densification interval limits but does not eliminate temporal drift. First-observation deduplication intentionally ignores a later valid sample if that Gaussian was invisible or had zero gradient on the first visit to the same camera. Recovering those samples exactly would require per-Gaussian per-camera state. Likewise, distinct camera IDs can still have highly correlated poses. This is an efficient, well-defined default, not a proven global optimum over datasets. Frozen multi-view probing, pose clustering, reservoirs, per-camera replacement, and pairwise conflict matrices add cost or change the statistic and are deferred until measurements justify them.

Do not dynamically lower `conf_min_views` for a one-camera dataset. Such a dataset cannot provide multi-view conflict evidence; no Conf candidates is the correct result.

## CUDA and Python interfaces

1. Extend rasterizer backward with an optional auxiliary Conf sample output `conf_samples` of shape `(N,4)`, float32, CUDA, contiguous. Columns are `(g_world.x, g_world.y, g_world.z, norm(g_world))`; a zero fourth column denotes invalid/no directional evidence. An empty tensor denotes disabled collection. `preprocessCUDA` writes the sample from its existing local `dL_dmean` without changing any real gradient. Thread the pointer through `backward.h`, `rasterizer.h`, `rasterizer_impl.cu`, `rasterize_points.h/.cu`, and the Python backward unpacking consistently.

2. Remove `fabs` register sums and atomics in `PerGaussianRenderCUDA`. Preserve x/y signed accumulation exactly. Retaining a `(N,4)` temporary screen-gradient buffer with zero z/w is an acceptable ABI simplification; those columns must no longer be consumed as abs-grad. Do not place Conf into x/y or into the real means3D gradient. Compacting all means2D buffers to float2 is optional and carries a larger ABI surface.

3. Expose a pybind CUDA operation such as:

```
accumulate_conf(samples, world_sum, norm_sum, view_count, conf_out)
    -> conf_out
```

All tensors are detached. The operation validates CUDA device, float32/int32 dtype, contiguous layout, exact matching N and channel dimensions, and rejects unexpected storage aliasing. One thread per Gaussian updates S/M/K and writes current Conf; it needs no atomics because observations are submitted serially on the same stream. It does not reset the state. Alternatively split update and finalize kernels, computing Conf only at densification. Both satisfy the requirement that Python consumes CUDA-computed Conf directly. Use PyTorch current device/stream for the new operations, launch-error checks, and an early empty-N return. Explicitly test or document the inherited rasterizer's default-stream limitation; do not introduce a falsely safe asynchronous combination of default-stream producers and current-stream consumers.

4. Use an explicit auxiliary holder to cross the autograd boundary. For example `render(..., conf_stats=holder)` passes an optional non-tensor dictionary/object into `_RasterizeGaussians.apply`; its backward stores the CUDA-returned sample in `holder['samples']`. The backward returns `None` for that auxiliary input. `render_pkg['conf_stats']` points to the holder; the sample is available only after `loss.backward()`. Keep the holder on `ctx`, not as global module state. This permits two outstanding renders without one overwriting the other and avoids pretending a statistic is a derivative. Existing renderer outputs and ordinary `viewspace_points.grad` remain compatible apart from removed abs channels. Handle optional input arity consistently across direct rasterizer callers. Enable collection only for training renders within the accumulation range and, preferably, only for unseen camera IDs. Debug and non-debug paths must unpack the same backward tuple; the current debug forward incorrectly unpacks 7 of 19 outputs and should be corrected when touching that wrapper.

5. `GaussianModel.add_conf_stats(samples, camera_key)` checks deduplication and calls the CUDA accumulator, then records the key. Keep state member naming explicit (`conf_world_sum`, `conf_norm_sum`, `conf_view_count`, `conf_score`) or provide documented aliases if visualization requires old names. The production training path must not call Python `_compute_ndc_vjp_world` or read abs-grad. The helper can remain solely as a numerical reference; legacy explicitly invoked APIs must fail clearly or route to the new path rather than silently fall back to old accumulation.

## Selection and RFAS preservation

The default selection becomes `conf_only`: `base_mask = (CUDA_conf >= conf_thr) & (valid_view_count >= conf_min_views)`, with any required finite checks. Pass this mask into the existing LAS and optional spatial-diversity machinery. RFAS/EAS/fusion values remain the existing sampling/ranking scores inside eligible candidates. Preserve `compute_rf_score1`, its `pixel_weights` forward mechanism, `compute_high_freq_residual_log`, `compute_edge_score`, and `fuse_importance_scores` unchanged. `abs-grad` is unrelated to RFAS's forward pixel weighting and is not needed by it.

Remove the special `iteration > 14500` fallback to gradient magnitude and the old abs threshold relaxation. Late iterations continue using Conf eligibility and the normally supplied ranking scores. If the caller supplies no ranking score, use Conf itself as a documented fallback; do not resurrect abs-grad.

Avoid silently relabeling Conf as abs-grad in compatibility logs or candidate strategies. Keep the general selector's old strategy implementations for existing isolated tests/explicit legacy utilities if useful, but the CUDA Conf production route needs a clear Conf-only selector/branch with no `abs_score`, `abs_mask`, `densify_grad_threshold`, `match_and`, or finite-mask dependency on obsolete buffers. Reject explicitly requested incompatible abs-dependent strategies with an actionable error, or offer an explicit documented legacy mode; do not compute abs-grad to preserve them invisibly. RFAS-only ranking as a legacy ablation should not become the default or bypass the new Conf eligibility accidentally.

Fixed budgets and spatial selection must remain subsets of the Conf candidate mask. A requested budget may exceed the available candidates; never fill it with non-Conf points. Use `.reshape(-1)` or `.squeeze(-1)` rather than unconstrained `.squeeze()` so N=1 stays one-dimensional. Empty candidate sets should be a clean no-op. Keep existing split/prune/budget geometry and score semantics.

## State, topology, and checkpoints

Centralize Conf window reset. Initialize on `training_setup`; clear at **every** densification boundary, including no candidates, zero budget, and no split. The current code only resets via successful topology extension, which can leak statistics across nominal intervals. Reset the camera-key set whenever sums are reset. Capture visualization/statistics before resetting.

For pure pruning within an open interval, slice all N-dependent buffers with the same survivor mask and preserve the admitted-camera set. Alternatively deliberately reset the entire window, but never reset just the set while retaining sums. `only_prune`, split/clone append, topology replacement, and loading a new model should reset all buffers and the set together. New Gaussian rows start at zero. If a topology reset occurs after rendering but before accumulation, discard that render's sample: N equality alone is insufficient when topology identities have changed. The existing step-300 prune occurs outside the default Conf range, but custom ranges must be safe.

Recommended checkpoint extension: retain the historical 12 fields and append a versioned optional Conf-state dictionary containing S/M/K, any cached Conf score, admitted stable camera keys, and version. Restore both legacy length-12 and new payloads. Never reinterpret old `xyz_gradient_accum` or `denom` as Conf. Legacy checkpoints initialize an empty Conf window; new checkpoints preserve a mid-interval window. Validate restored shapes/dtypes/device and recompute score if it is not saved. If an implementation instead deliberately omits transient stats, explicitly reset all new state on resume and document that mid-interval resume is not selection-equivalent. PLY format remains unchanged.

## Verification and acceptance

Tests must be run using the requested testing model, and evaluated by the requested independent review model. CUDA compilation alone is not acceptance.

* **Formula/invalid cases:** CUDA vs independent float64 reference for aligned, opposite, orthogonal, unequal-magnitude, very small coherent, common loss-rescaling, zero, NaN/Inf, invisible, behind-camera, empty-N, N=1, and randomized streams. Typical float32 checks: abs/rel tolerance around 1e-5 outside degenerate scales, with separate absolute checks near zero. Verify clamp range and zero-count behavior.
* **Projection correctness:** compare emitted CUDA samples against the standalone exact stabilized projection VJP and finite differences for non-square images, translated/rotated cameras and positive depths. Validate roll/frame consistency using covariantly transformed screen gradients. Do not compare to the whole `xyz.grad`.
* **Real backward invariance:** fixed small scene, fixed upstream image tensor: forward image and all real parameter gradients match the baseline with Conf enabled/disabled within GPU float32 reduction tolerances. Include SH and precomputed-color/covariance modes supported by the existing API; separate failures due to pre-existing unsupported combinations.
* **No abs dependence:** no `fabs(tmp_x/y)` accumulation remains; train uses only CUDA Conf stats. Perturbing old dummy z/w gradients or `densify_grad_threshold` cannot change new candidates. Above iteration 14500 the same Conf gate remains active.
* **Distinct multiview:** A,A,B and A,B,A equal A,B for stats and K, including nonconsecutive repeats and cameras with only some valid Gaussian rows. Same camera with opposite gradients must not by itself qualify as multiview. Independent holders must not cross-contaminate.
* **Lifecycle:** no-split boundary reset, budget-zero reset, prune masking, split/new rows, one-camera dataset, interval reset of dedup keys, legacy checkpoint restore, and new mid-window checkpoint round trip.
* **Ranking regression:** synthetic fixed EAS/RFAS scores unchanged; final eligibility follows only Conf, rank weights preserve current score computation, fixed budget/spatial masks stay subsets, and low/high RFAS values do not modify the Conf statistic.
* **End-to-end GPU:** build the local extension, run a short real-data training job using a scene under `/home/xzh/xzh/data/3dgs` through multiple densification boundaries, observe finite losses/Conf/counts and actual eligible splits where data produces them, save/restore a checkpoint and render. Do not claim a naturally empty high-threshold split set proves split plumbing; use a controlled low threshold test separately. Run with debug enabled at least once. Record exact commands, extension source location, device, scene, iteration range, outcomes, wall time and peak memory. A short smoke test cannot establish final PSNR superiority or an optimal threshold.

Performance expectation is removal of two abs register sums and atomic adds per Gaussian/tile, elimination of Python Jacobian/masking temporaries, and one O(N) streaming kernel per unique view. State uses approximately 24 bytes/Gaussian with float32 S/M/score and int32 count, plus a transient 16-byte sample. No N×V storage or extra image backward. Measure synchronized CUDA-event timings before claiming a speedup; rasterization and RFAS passes may dominate total training time.
