# CUDA Conf: per-Gaussian rolling multi-view window

## Revised requirement

This document supersedes the original interval-wide accumulation design. The user requires `--conf_window_size W`, default **3**, to hold each Gaussian's most recent W distinct **actually visible** camera observations. The fourth new valid view at W=3 evicts that Gaussian's oldest gradient. Global training-camera visits do not advance a Gaussian's window when it does not contribute to the rendered image.

Conf replaces abs-gradient densification statistics, not real loss derivatives. Preserve rendering, photometric/PAIR losses, optimizer gradients, EAS, RFAS, EAS/RFAS fusion, LAS geometry, and pruning policy. Existing user modifications remain intact.

## Projection gradient and actual visibility

The signed `dL_dmean2D.x/y` values in `PerGaussianRenderCUDA` already include image W/2 and H/2 and therefore represent NDC derivatives. Remove the separate `fabs` sums and z/w abs atomics, but preserve signed x/y accumulation. Retaining unused zero z/w channels is an acceptable ABI simplification.

Export local `dL_dmean` from `preprocessCUDA`, before its addition to the final position derivative and before SH derivatives. Do not substitute final `means3D.grad`, which also contains covariance/SH contributions. For row-vector camera convention `q = [xyz,1] @ M`, the existing renderer uses `r=1/(q.w+1e-7)` and:

```
jx[k] = M[k,0]*r - M[k,3]*q.x*r*r
jy[k] = M[k,1]*r - M[k,3]*q.y*r*r
g_world[k] = jx[k]*g_ndc.x + jy[k]*g_ndc.y
```

Reuse this intermediate rather than re-deriving the Python Jacobian, which previously omitted stabilization. Do not double-apply focal/image scaling. All actual parameter derivatives remain unchanged.

**A visible zero gradient is a valid view.** Use actual rasterization participation independent of derivative magnitude. In backward render, after the existing valid-pixel, last-contributor, power, and alpha checks, set a register boolean indicating an accepted contributing pixel. At bucket completion, mark participation only when this boolean is true. A positive radius is not sufficient evidence: the Gaussian may be fully occluded or rejected everywhere.

Inspection confirms forward `accum_count`/`accum_blend` are updated only inside `if(pixel_weights != nullptr)` and are allocated only for weighted rendering. They cannot describe ordinary training visibility. Do not enable artificial pixel weights or alter RFAS to obtain visibility.

Keep the auxiliary output shape `(P,4)` and use its norm channel as a two-stage marker:

1. Initialize `.w=-1`, meaning invalid/unseen; xyz is ignored while w is negative.
2. Backward render receives an optional `float4* conf_samples`. Each bucket with an accepted contributing pixel performs `atomicExch(&conf_samples[i].w, 0.0f)` at its end.
3. Subsequent preprocess checks w>=0, positive finite clip-w, finite VJP and finite stable norm, then writes `(g_world.xyz, norm)`. **norm=0 is valid.** Invalid projection/VJP restores w=-1.
4. Sequential kernel ordering on the same stream completes all markers before sample conversion. This avoids an additional visibility buffer and leaves RFAS forward untouched.

Use stable `hypotf` norms. NaN/Inf derivatives cannot supply a usable sample and do not advance history even if the primitive was nominally rendered. This invalid-norm sentinel is internal API metadata, not a mathematical negative norm.

## Per-Gaussian window semantics

Each Gaussian owns an ordered bounded list, oldest to newest, of `(camera_id,g_world.xyz,norm)`. Camera IDs are stable nonnegative int64 dataset view IDs; -1 means an unused slot. They must identify the view image, not a shared COLMAP intrinsic-camera model. The implementation hashes each training image's dataset-relative path, pose, intrinsics and dimensions; it checks duplicate identities/hash collisions and stores the complete ID-to-identity mapping in the checkpoint. Restore rejects a changed mapping. A prior v2 checkpoint without this mapping explicitly starts an empty Conf history.

For a valid sample of Gaussian i from camera c:

- If c is already in its window, remove that old entry and append the new observation at the newest end. Count does not change. This updates both gradient and recency.
- If c is absent and the window is full, evict the oldest entry and append the new observation. The old gradient/norm is no longer included in either sum.
- If c is absent and there is capacity, append and increment occupied count.
- An invalid/unseen sample does nothing: no eviction, refresh, count change, or decay.

At W=3, `A,B,C,D → B,C,D`, `A,B,C,A → B,C,A`, `A,unseen-B,unseen-C,D → A,D`. Each Gaussian can retain a different history for the same global camera sequence. Repeated cameras cannot inflate distinct-view count. Visible zero vectors consume/refresh slots and can evict nonzero observations, as required by visibility-based window semantics.

Recommend a compact ordered deque instead of a circular buffer plus cursor: W=3 requires at most two slot moves and duplicate-camera refresh naturally preserves order. A literal ring is also valid if it implements identical refresh/eviction semantics. Window semantics matter more than physical layout.

No global camera set remains. Every eligible training render may refresh some Gaussian's history and must collect samples. Evaluation, EAS and RFAS renders never advance the Conf window.

## Formula and numerical stability

For the occupied window H_i:

```
S_i = sum(g_is for s in H_i)
M_i = sum(norm(g_is) for s in H_i)
K_i = occupied distinct-camera count
Conf_i = clamp(1 - norm(S_i)/M_i, 0, 1) if K_i>=2 and M_i>0
Conf_i = 0 otherwise
```

The candidate gate requires finite score and positive finite M, `K >= max(2,conf_min_views)`, and `Conf >= conf_thr`. Preserve the user's defaults `conf_min_views=2`, `conf_thr=0.85`. Validate W>=2 and `2 <= conf_min_views <= W`. Capacity and minimum evidence are different options.

Do not perturb positive M with `+1e-6`: this makes tiny aligned vectors falsely appear conflicting. Guard division instead. One nonzero vector plus visible zero vectors has Conf=0; an entirely zero window never qualifies because M must be positive. At explicitly requested `conf_thr=0`, a nonzero but coherent window may qualify: zero threshold intentionally removes the conflict-strength requirement. No extra nonzero-direction counter is necessary.

Long-running `sum -= old; sum += new` updates accumulate cancellation drift. After editing the small history, recompute S/M from its at most W active slots, optionally using double temporaries. This is exactly equivalent to removing expired gradients, numerically stable, and O(W); camera-ID search already costs O(W). Do not preserve stale aggregate errors to claim O(1) updates.

Retain the existing magnitude-weighted world-space cancellation formula. Unit-normalizing each view changes the method and threshold meaning. Aligned gradients give 0, equal opposite gradients give 1, equal orthogonal gradients give `1-sqrt(2)/2`. Common positive loss rescaling leaves the ideal score unchanged.

## State and interface

P is Gaussian count; W is window capacity.

| State | Shape | Type |
|---|---|---|
| history, ordered world xyz/norm | P,W,4 | float32 |
| camera IDs, unused=-1 | P,W | int64 |
| occupied view count | P,1 | int32 |
| world sum | P,3 | float32 |
| norm sum | P,1 | float32 |
| CUDA Conf score | P,1 | float32 |
| transient backward sample | P,4 | float32 |

The optional non-tensor holder passed into custom autograd receives the native backward output as `holder['samples']`; backward returns None for that auxiliary input. `render_pkg['conf_stats']` references that holder. Read it only after `loss.backward()`. Do not disguise a statistic as a parameter derivative or use a shared global holder.

The CUDA operation is conceptually:

```
update_conf_window(samples, camera_id, history, camera_ids,
                   world_sum, norm_sum, view_count, conf_score)
    -> conf_score
```

One thread owns one Gaussian's history, so window updates require no atomics. Validate dtype, device, contiguous shape, matching P/W, valid camera ID, and unsafe aliases. Empty P is a no-op. Preserve correct producer/consumer stream ordering; a current-stream accumulator cannot safely consume unsynchronized default-stream rasterizer outputs. Reject samples belonging to a previous topology generation even if P coincidentally matches.

The model's `add_conf_stats(samples, camera_id, producer_topology_version)` requires the version captured before rendering and rejects a mismatch before calling CUDA. The native kernel requires 16-byte alignment for its `float4` sample/history accesses; a contiguous slice alone does not prove alignment.

## Lifecycle and checkpoint

**Never reset survivor history merely because a densification interval ended**, including zero-budget/no-split boundaries. This is a continuous rolling history. A Gaussian not seen for many renders retains its own last W effective observations.

Pure pruning and `only_prune` slice every state buffer with the same survivor mask. Split/clone append initializes only new rows to zero with IDs=-1; unchanged old rows retain history. Deleted parents disappear via the same prune mask. Descendants do not inherit parent history because their geometry changes. If a split mutates an existing row's geometry in place, invalidate that row only. Initialization, a new PLY model, deliberate capacity changes, and legacy checkpoint migration start empty history. Keep a topology generation guard for stale render outputs.

Append a version-2 optional Conf-state dictionary to the historical twelve checkpoint fields. Save W/history/IDs/count, sums/score and the training-camera identity mapping, or recompute aggregates after restore. Version-2 resume at unchanged capacity and matching camera mapping preserves the exact next eviction. Legacy twelve-field and version-1 cumulative states cannot reconstruct window entries, so load model/optimizer and initialize an empty rolling history. An early version-2 checkpoint without a stable camera mapping also resets its Conf history explicitly. Never reinterpret old cumulative sums as history entries. A saved/requested W mismatch must explicitly reset with a message or reject, not silently reinterpret shapes. PLY remains unchanged.

## Selection and RFAS invariants

Python selects directly from CUDA score/count/norm validity. No production Python Jacobian or Conf recomputation; no abs mask, old gradient threshold, or >14500 fallback. Default is `conf_only`. Reject incompatible abs-dependent strategies clearly rather than restoring abs accumulation silently.

Within eligible candidates preserve existing EAS/RFAS/fused scores as LAS ranking/sampling weights. Keep `compute_rf_score1`, forward `pixel_weights`, high-frequency transforms, edge scoring, and fusion unchanged. A caller without ranking scores may use a documented Conf fallback. Fixed-budget/spatial-diversity selection remains a subset of Conf eligibility. Keep P=1 tensor dimensions and empty-set behavior valid. Logging must describe a rolling window rather than interval accumulation.

## Validation

Requested roles remain 6astra high design, 6sol xhigh implementation, 6luna xhigh testing, and 6astra medium acceptance judgment. Compilation is only one check.

- Compare CUDA state after every observation to an independent Python bounded ordered-camera reference, at W=2,3,5. Cover fourth-view eviction, duplicate refresh/recency, independently unseen rows, late visibility, and long repeated sequences.
- Verify actual-participation marker against controlled rendering: radius-positive but fully rejected/occluded Gaussian does not advance; genuinely rendered Gaussian with zero upstream image gradient does advance with zero vector. Include clipping and partial tiles.
- Check aligned/opposite/orthogonal/unequal/tiny vectors, common scaling, all-zero visible windows, invalid inputs, P=0/P=1, score bounds, long-run eviction drift, threshold-zero behavior, and positive-norm gating.
- Compare emitted projection gradient with an independent exact stabilized VJP and finite differences for rotated/translated cameras, non-square images and roll. Do not compare to the complete xyz derivative.
- Compare forward images and all actual parameter derivatives against baseline and Conf-off results within appropriate float32 GPU reduction tolerance.
- Verify no-split/budget-zero boundaries preserve history; unrelated splits preserve survivors; pruning slices exact rows; children start empty; stale topology samples are rejected; version-2 checkpoint preserves next eviction; old states migrate empty.
- Confirm RFAS/EAS/fusion values are unchanged, Conf controls eligibility throughout training, budgets never add ineligible points, and old abs thresholds have no influence.
- Build and run GPU training on a scene under `/home/xzh/xzh/data/3dgs` across multiple densification boundaries. Record finite losses, counts<=W, populated histories, splitting or controlled threshold-zero smoke, checkpoint restore and rendering. Exercise debug mode. Report commands/device/build paths and distinguish synthetic correctness, smoke success, and unmeasured final PSNR/performance.

Resident state uses approximately `(24*W+24)` bytes per Gaussian, plus transient 16-byte samples: W=3 is about 96 MB resident per million Gaussians. This bounded O(PW) memory enables exact per-Gaussian eviction independently of dataset camera count. Work is O(PW), with W=3 by default and no additional image backward or Python per-Gaussian loop. Measure performance before asserting speedup.

This implements the user's precise sliding-view definition; it is not a proof of globally optimal reconstruction quality. Samples come from different parameter-update times, so temporal drift remains possible. Rarely visible Gaussians retain old observations by definition. Distinct camera IDs may have correlated poses. Frozen-model probing, expiration by wall time, pose clustering and normalized-direction alternatives change semantics or cost and are not introduced implicitly.
