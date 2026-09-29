# CUDA Conf acceptance review

**Decision: accepted for the requested implementation and short-run validation scope.** The core requirements are met by production commit `1535241`: CUDA replaces abs-gradient densification statistics while retaining real loss derivatives; each Gaussian maintains its own configurable, most-recent distinct contributing-camera window (default 3); survivors retain history across densification, children start empty, and RFAS/EAS/fusion ranking helpers remain unchanged. No unresolved implementation blocker was found in the reviewed changes.

This is the final acceptance review requested from the GPT-6 Astra medium reviewer. It combines read-only source and test-script inspection with the build and execution evidence produced by the implementation/testing agents. The reviewer did not rerun tests. Detailed commands, environment, binary hashes and results are in [conf_cuda_validation.md](conf_cuda_validation.md).

## Basis for acceptance

- The final native extension linked successfully and ran on an A100 with PyTorch 2.1.2 / CUDA 11.8. This is actual compiled CUDA evidence, not a Python fallback or static-only judgment.
- Independent float64 deque references checked window state after every observation at W=2, 3 and 5. Coverage includes independent visibility, duplicate-camera refresh, eviction, zero observations, invalid observations and bounded counts. The kernel recomputes aggregates from occupied slots, avoiding accumulated subtraction drift.
- Isolated baseline/new rasterizer processes compared four SH/color and covariance/scale branches. Images matched exactly; maximum real-parameter gradient discrepancy was 3.82e-6. Conf on/off results matched exactly in the tested branch. The emitted projection gradient was compared to the stabilized analytic VJP and a separate float64 central-difference calculation of the projection scalar function. This correctly tests the projection contribution rather than conflating it with complete position gradients.
- Rendering checks distinguish contribution from positive radius: the fully occluded rear Gaussian had positive radius and an invalid observation marker. Visible zero-upstream samples remained valid zeros; culled observations stayed invalid. Partial tiles, the all-culled B=0 path and debug mode passed. Memcheck reported zero errors for the new rasterizer worker, including the added occlusion case.
- Real GaussianModel lifecycle checks cover pruning, append, actual long-axis split, empty child histories, unchanged survivor histories, zero-budget/no-split boundaries, stale same-size generations, the Conf gate after iteration 14500, and serialized checkpoint restoration with the same next eviction. RFAS/EAS/fusion function ASTs match the saved pre-Conf source.
- Two real Garden runs at 128-pixel resolution completed 20 iterations each, using threshold 0 and the default 0.85. Both crossed four densification boundaries, split 100 parents at each, and grew from 138766 to 139166 Gaussians. The recorded EMA losses were finite. The default-threshold CSV documents 254, 359, 373 and 400 eligible candidates before the fixed cap.

## Disposition of the earlier review

The camera-ID issue is resolved by stable image/view identities and checkpoint mapping validation, rather than shuffled enumeration IDs or shared COLMAP intrinsic IDs. The model API requires a producer topology generation. Native sample/history tensors require float4 alignment. Backward prefetch now checks tile/image bounds and the B=0 launch returns early.

The earlier ranking P1 was a false positive: the commented `tt_importance = gaussian_importance` assignment was already a user modification in the saved pre-task patch. Preserving it is correct; it must not be reverted under this review.

## Limits of this acceptance

The Garden runs are integration smokes, not full training or a reconstruction-quality benchmark. Their logs expose periodic EMA losses, not assertions over every intermediate training value. No final PSNR, speedup, peak-memory benchmark or long-duration stability claim follows from these results. The window reference test covers hundreds of observations, not an exhaustive numerical-state space.

Checkpoint continuation was tested at the serialized model-state level, including eviction order; full CLI training resume and a separate post-training evaluation render were not exercised. Projection finite differences validate the projection-only scalar function, not end-to-end finite differences of every rasterizer parameter. Memcheck covers the executed small scenes and is not a proof of all-input GPU memory safety.

The rasterizer intentionally requires the default CUDA stream; the Conf accumulator independently supports the current stream. The documented reproduction setup relies on local temporary baseline/build/dependency artifacts and the Garden dataset. Binary hashes and log paths identify the tested artifacts; another checkout must build its own compatible extension.

These limits do not invalidate the demonstrated implementation of the requested sliding-view Conf semantics, preservation of real loss gradients, or the successful real-data densification smokes. They constrain the claims that can be made about quality, performance and deployment beyond the tested environment.
