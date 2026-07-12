"""
致密化候选选择策略单元测试。Round 1 修订版。
在 CPU 上运行，不需要 GPU 或训练状态。
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.candidate_selector import (
    select_densification_candidates,
    select_by_topk,
    select_by_threshold,
    normalize_scores,
    compute_soft_mask,
    resolve_topk,
    apply_fixed_budget,
    compute_candidate_statistics,
    VALID_STRATEGIES,
    _BOOLEAN_STRATEGIES,
    _finite_valid_mask,
    EPS,
)


def _cfg(**overrides):
    defaults = {
        'candidate_selection_strategy': 'and',
        'candidate_budget_mode': 'native',
        'candidate_budget_reference': 'match_and',
        'candidate_weight_alpha': 0.5,
        'candidate_score_normalization': 'percentile',
        'candidate_score_selection': 'threshold',
        'candidate_score_threshold': 0.5,
        'candidate_topk': 0,
        'candidate_topk_ratio': 0.05,
        'soft_fusion_type': 'weighted',
        'soft_abs_temperature': 1.0,
        'soft_conf_temperature': 1.0,
        'soft_selection_threshold': 0.5,
        'densify_grad_threshold': 0.0003,
        'conf_thr': 0.8,
        'candidate_rfas_topk': 0,
        'candidate_rfas_topk_ratio': 0.05,
        'candidate_fixed_budget': 1000,
        'candidate_fixed_ratio': 0.05,
        'candidate_stats_enabled': True,
    }
    defaults.update(overrides)

    class MockConfig:
        pass
    cfg = MockConfig()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _tensors(N=100):
    abs_score = torch.rand(N, 1) * 0.001
    conf_score = torch.rand(N)
    rfas_score = torch.rand(N)
    abs_mask = abs_score.squeeze() >= 0.0003
    conf_mask = (conf_score >= 0.8) & torch.ones(N, dtype=torch.bool)
    return abs_score, conf_score, rfas_score, abs_mask, conf_mask


# ---------------------------------------------------------------------------
# 1. AND = abs_mask & conf_mask
# ---------------------------------------------------------------------------
def test_and_equals_abs_and_conf():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _tensors()
    mask, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='and', config=_cfg())
    assert torch.equal(mask, abs_m & conf_m)
    print("  PASS test_and")


# ---------------------------------------------------------------------------
# 2. OR = abs_mask | conf_mask
# ---------------------------------------------------------------------------
def test_or_equals_abs_or_conf():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _tensors()
    mask, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='or', config=_cfg())
    assert torch.equal(mask, abs_m | conf_m)
    print("  PASS test_or")


# ---------------------------------------------------------------------------
# 3. abs_only = abs_mask
# ---------------------------------------------------------------------------
def test_abs_only():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _tensors()
    mask, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='abs_only', config=_cfg())
    assert torch.equal(mask, abs_m)
    print("  PASS test_abs_only")


# ---------------------------------------------------------------------------
# 4. conf_only = conf_mask
# ---------------------------------------------------------------------------
def test_conf_only():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _tensors()
    mask, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='conf_only', config=_cfg())
    assert torch.equal(mask, conf_m)
    print("  PASS test_conf_only")


# ---------------------------------------------------------------------------
# 5. topk count
# ---------------------------------------------------------------------------
def test_topk_count():
    scores = torch.rand(100)
    for k in [1, 5, 10, 50, 100]:
        assert select_by_topk(scores, k).sum().item() == k
    print("  PASS test_topk_count")


# ---------------------------------------------------------------------------
# 6. Boolean fixed-budget NEVER backfills outside original mask
# ---------------------------------------------------------------------------
def test_boolean_fixed_budget_no_backfill():
    """布尔策略在 fixed 预算下：掩码内点数不足时不补入掩码外点。"""
    N = 100
    mask = torch.zeros(N, dtype=torch.bool)
    mask[0:3] = True  # 只有 3 个候选
    scores = torch.rand(N)
    scores[~mask] = 0.0  # 掩码外分数为 0

    for strat in _BOOLEAN_STRATEGIES:
        cfg = _cfg(candidate_budget_mode='fixed',
                   candidate_budget_reference='fixed_number',
                   candidate_fixed_budget=10)  # 目标 10，实际只有 3
        adjusted_mask, actual, resolved = apply_fixed_budget(
            mask, scores, target_budget=0,
            budget_reference='fixed_number',
            config=cfg, strategy=strat)

        assert actual <= 3, f"{strat}: should not backfill beyond mask, got {actual}"
        # 被选中的点必须全在原掩码内
        assert (adjusted_mask & ~mask).sum().item() == 0, \
            f"{strat}: selected points outside original mask"
    print("  PASS test_boolean_fixed_budget_no_backfill")


# ---------------------------------------------------------------------------
# 7. Continuous fixed-budget CAN expand
# ---------------------------------------------------------------------------
def test_continuous_fixed_budget_can_expand():
    """连续策略在 fixed 预算下可扩大候选池。"""
    N = 100
    mask = torch.zeros(N, dtype=torch.bool)
    mask[0:3] = True
    scores = torch.rand(N)
    scores[~mask] = torch.rand(97) * 0.5 + 0.5  # 掩码外也有 valid 分数

    cfg = _cfg(candidate_budget_mode='fixed',
               candidate_budget_reference='fixed_number',
               candidate_fixed_budget=10)
    adjusted, actual, _ = apply_fixed_budget(
        mask, scores, target_budget=0,
        budget_reference='fixed_number',
        config=cfg, strategy='weighted_score')

    assert actual == 10, f"continuous should expand to 10, got {actual}"
    # 掩码内 3 个点必须入选
    assert (adjusted & mask).sum().item() == 3
    print("  PASS test_continuous_fixed_budget_can_expand")


# ---------------------------------------------------------------------------
# 8. NaN/Inf excluded from candidates (strategy level)
# ---------------------------------------------------------------------------
def test_nan_inf_strategy_exclusion():
    """所有策略的 final_mask 不应包含 NaN/Inf 对应的点。"""
    abs_s, conf_s, rfas_s, abs_m, conf_m = _tensors(N=100)
    # 在 rfas_score 中注入 NaN/Inf
    rfas_s[10] = float('nan')
    rfas_s[20] = float('inf')
    rfas_s[30] = float('-inf')

    for strat in VALID_STRATEGIES:
        mask, scores, _ = select_densification_candidates(
            abs_s, conf_s, rfas_s, abs_m, conf_m,
            strategy=strat, config=_cfg())

        # 被注入 NaN/Inf 的点不应出现在最终掩码中
        assert not mask[10], f"{strat}: NaN point in mask"
        assert not mask[20], f"{strat}: +Inf point in mask"
        assert not mask[30], f"{strat}: -Inf point in mask"

        # selection_score 在这些位置应为 -inf（finite mask 将其标记为不可选）
        assert torch.isinf(scores[10]) and scores[10].item() < 0, f"{strat}: NaN score not -inf"
        assert torch.isinf(scores[20]) and scores[20].item() < 0, f"{strat}: +Inf score not -inf"
        assert torch.isinf(scores[30]) and scores[30].item() < 0, f"{strat}: -Inf score not -inf"
    print("  PASS test_nan_inf_strategy_exclusion")


# ---------------------------------------------------------------------------
# 9. Empty candidates
# ---------------------------------------------------------------------------
def test_empty_candidates():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _tensors()
    zm = torch.zeros(100, dtype=torch.bool)
    mask, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, zm, zm, strategy='and', config=_cfg())
    assert mask.sum().item() == 0
    assert select_by_topk(torch.rand(100), 0).sum().item() == 0
    assert select_by_topk(torch.zeros(0), 10).numel() == 0
    print("  PASS test_empty_candidates")


# ---------------------------------------------------------------------------
# 10. Device consistency
# ---------------------------------------------------------------------------
def test_device_consistency():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _tensors()
    for strat in VALID_STRATEGIES:
        mask, scores, _ = select_densification_candidates(
            abs_s, conf_s, rfas_s, abs_m, conf_m, strategy=strat, config=_cfg())
        assert mask.device.type == 'cpu'
        assert scores.device.type == 'cpu'
    print("  PASS test_device_consistency")


# ---------------------------------------------------------------------------
# 11. Mask dtype is bool
# ---------------------------------------------------------------------------
def test_dtype_bool():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _tensors()
    for strat in VALID_STRATEGIES:
        mask, _, _ = select_densification_candidates(
            abs_s, conf_s, rfas_s, abs_m, conf_m, strategy=strat, config=_cfg())
        assert mask.dtype == torch.bool
    print("  PASS test_dtype_bool")


# ---------------------------------------------------------------------------
# 12. Default AND matches original
# ---------------------------------------------------------------------------
def test_default_and_matches_original():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _tensors(N=500)
    mask, scores, stats = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='and', config=_cfg())
    assert torch.equal(mask, abs_m & conf_m)

    required = ['strategy', 'n_valid', 'n_abs_candidates', 'n_conf_candidates',
                'n_and_candidates', 'n_or_candidates', 'n_final_candidates',
                'candidate_ratio', 'target_budget', 'actual_budget',
                'iou_abs_conf', 'overlap_abs', 'overlap_conf']
    for k in required:
        assert k in stats, f"Missing: {k}"
    print("  PASS test_default_and_matches_original")


# ---------------------------------------------------------------------------
# 13. Fixed budget target_budget logging
# ---------------------------------------------------------------------------
def test_fixed_budget_target_logging():
    """fixed_number/fixed_ratio 的目标预算应记录为已解析值。"""
    abs_s, conf_s, rfas_s, abs_m, conf_m = _tensors(N=200)

    # fixed_number
    cfg_fn = _cfg(candidate_budget_mode='fixed',
                  candidate_budget_reference='fixed_number',
                  candidate_fixed_budget=30)
    _, _, stats = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='or', config=cfg_fn)
    assert stats['target_budget'] == 30, f"fixed_number target: {stats['target_budget']}"

    # fixed_ratio
    cfg_fr = _cfg(candidate_budget_mode='fixed',
                  candidate_budget_reference='fixed_ratio',
                  candidate_fixed_ratio=0.1)
    _, _, stats = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='or', config=cfg_fr)
    assert stats['target_budget'] == 20, f"fixed_ratio target: {stats['target_budget']}"
    print("  PASS test_fixed_budget_target_logging")


# ---------------------------------------------------------------------------
# 14. Normalization edge cases
# ---------------------------------------------------------------------------
def test_normalization_edge_cases():
    for method in ['minmax', 'zscore', 'percentile']:
        for s in [torch.ones(100), torch.zeros(100)]:
            norm = normalize_scores(s, method=method)
            assert torch.isfinite(norm).all()

    s = torch.rand(100); s[0] = float('nan')
    for method in ['minmax', 'zscore', 'percentile']:
        norm = normalize_scores(s, method=method)
        assert not torch.isnan(norm).any()
    print("  PASS test_normalization_edge_cases")


# ---------------------------------------------------------------------------
# 15. Soft mask direction
# ---------------------------------------------------------------------------
def test_soft_mask_direction():
    scores = torch.rand(100) * 2.0
    P_pos = compute_soft_mask(scores, threshold=1.0, temperature=1.0, sign=+1)
    assert (P_pos[scores > 1.0].mean() > 0.5)
    P_neg = compute_soft_mask(scores, threshold=1.0, temperature=1.0, sign=-1)
    assert (P_neg[scores > 1.0].mean() < 0.5)
    print("  PASS test_soft_mask_direction")


# ---------------------------------------------------------------------------
# 16. Invalid strategy raises
# ---------------------------------------------------------------------------
def test_invalid_strategy_raises():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _tensors()
    try:
        select_densification_candidates(
            abs_s, conf_s, rfas_s, abs_m, conf_m,
            strategy='invalid', config=_cfg())
        assert False, "Should raise"
    except ValueError:
        pass
    print("  PASS test_invalid_strategy_raises")


# ---------------------------------------------------------------------------
# 17. Parser/seed test
# ---------------------------------------------------------------------------
def test_seed_arg_accepted():
    """验证 train.py --seed 参数被接受。"""
    import subprocess
    result = subprocess.run(
        ['python3', 'train.py', '--help'],
        capture_output=True, text=True, timeout=15)
    assert '--seed' in result.stdout, "--seed not found in train.py help"
    print("  PASS test_seed_arg_accepted")


# ---------------------------------------------------------------------------
# 18. Native continuous defaults to threshold
# ---------------------------------------------------------------------------
def test_native_continuous_uses_threshold():
    """weighted_score/soft_fusion native 模式默认使用 threshold 选择。"""
    from arguments import OptimizationParams
    import argparse
    parser = argparse.ArgumentParser()
    opt = OptimizationParams(parser)
    assert opt.candidate_score_selection == 'threshold', \
        f"Default should be threshold, got {opt.candidate_score_selection}"
    print("  PASS test_native_continuous_uses_threshold")


if __name__ == '__main__':
    tests = [
        test_and_equals_abs_and_conf,
        test_or_equals_abs_or_conf,
        test_abs_only,
        test_conf_only,
        test_topk_count,
        test_boolean_fixed_budget_no_backfill,
        test_continuous_fixed_budget_can_expand,
        test_nan_inf_strategy_exclusion,
        test_empty_candidates,
        test_device_consistency,
        test_dtype_bool,
        test_default_and_matches_original,
        test_fixed_budget_target_logging,
        test_normalization_edge_cases,
        test_soft_mask_direction,
        test_invalid_strategy_raises,
        test_seed_arg_accepted,
        test_native_continuous_uses_threshold,
    ]

    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed:
        sys.exit(1)
