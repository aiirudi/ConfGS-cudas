"""
致密化候选选择策略单元测试。

在 CPU 上运行，不需要 GPU 或训练状态。
验证所有策略的正确性、边界情况和数值稳定性。
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
    EPS,
)


def _make_config(**overrides):
    """创建模拟配置对象。"""
    defaults = {
        'candidate_selection_strategy': 'and',
        'candidate_budget_mode': 'native',
        'candidate_budget_reference': 'match_and',
        'candidate_weight_alpha': 0.5,
        'candidate_score_normalization': 'percentile',
        'candidate_score_selection': 'topk',
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


def _make_tensors(N=100, device='cpu'):
    """创建测试用张量。"""
    abs_score = torch.rand(N, 1, device=device) * 0.001
    conf_score = torch.rand(N, device=device)
    rfas_score = torch.rand(N, device=device)
    abs_mask = abs_score.squeeze() >= 0.0003
    conf_mask = (conf_score >= 0.8) & torch.ones(N, dtype=torch.bool, device=device)
    return abs_score, conf_score, rfas_score, abs_mask, conf_mask


# ---------------------------------------------------------------------------
# 测试 1: AND 输出等于 abs_mask & conf_mask
# ---------------------------------------------------------------------------
def test_and_equals_abs_and_conf():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _make_tensors()
    cfg = _make_config()

    mask, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='and', config=cfg)

    expected = abs_m & conf_m
    assert torch.equal(mask, expected), "AND mask should equal abs_mask & conf_mask"
    print("  PASS test_and_equals_abs_and_conf")


# ---------------------------------------------------------------------------
# 测试 2: OR 输出等于 abs_mask | conf_mask
# ---------------------------------------------------------------------------
def test_or_equals_abs_or_conf():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _make_tensors()
    cfg = _make_config()

    mask, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='or', config=cfg)

    expected = abs_m | conf_m
    assert torch.equal(mask, expected), "OR mask should equal abs_mask | conf_mask"
    print("  PASS test_or_equals_abs_or_conf")


# ---------------------------------------------------------------------------
# 测试 3: abs_only 输出等于 abs_mask
# ---------------------------------------------------------------------------
def test_abs_only_equals_abs():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _make_tensors()
    cfg = _make_config()

    mask, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='abs_only', config=cfg)

    assert torch.equal(mask, abs_m), "abs_only mask should equal abs_mask"
    print("  PASS test_abs_only_equals_abs")


# ---------------------------------------------------------------------------
# 测试 4: conf_only 输出等于 conf_mask
# ---------------------------------------------------------------------------
def test_conf_only_equals_conf():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _make_tensors()
    cfg = _make_config()

    mask, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='conf_only', config=cfg)

    assert torch.equal(mask, conf_m), "conf_only mask should equal conf_mask"
    print("  PASS test_conf_only_equals_conf")


# ---------------------------------------------------------------------------
# 测试 5: top-k 选中数量正确
# ---------------------------------------------------------------------------
def test_topk_count():
    scores = torch.rand(100)
    for k in [1, 5, 10, 50, 100]:
        mask = select_by_topk(scores, k)
        assert mask.sum().item() == k, f"topk({k}) should select {k}, got {mask.sum().item()}"
    print("  PASS test_topk_count")


# ---------------------------------------------------------------------------
# 测试 6: fixed budget 下不同策略候选数量一致 (match_and)
# ---------------------------------------------------------------------------
def test_match_and_budget_consistency():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _make_tensors()
    cfg = _make_config(candidate_budget_mode='fixed', candidate_budget_reference='match_and')

    # 获取 AND 基线
    m_and, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='and',
        config=_make_config(candidate_budget_mode='native'))

    k_and = int(m_and.sum().item())
    assert k_and > 0, "AND baseline should be > 0 for this test"

    for strat in ['or', 'abs_only', 'conf_only', 'weighted_score', 'soft_fusion', 'rfas_rank']:
        mask, _, stats = select_densification_candidates(
            abs_s, conf_s, rfas_s, abs_m, conf_m, strategy=strat, config=cfg)
        actual = int(mask.sum().item())
        assert actual == k_and, f"{strat}: expected {k_and} candidates, got {actual}"
    print("  PASS test_match_and_budget_consistency")


# ---------------------------------------------------------------------------
# 测试 7: match_and 下其他策略候选数与 AND 一致
# ---------------------------------------------------------------------------
def test_match_and_count_match():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _make_tensors(N=200)

    cfg_native = _make_config(candidate_budget_mode='native')
    m_and, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='and', config=cfg_native)
    k_and = int(m_and.sum().item())
    assert k_and > 0, f"Need positive AND baseline, got {k_and}"

    # 确保有足够的候选来验证 match_and
    # 对于策略产生的候选少于AND的情况，fixed预算应扩大
    cfg_fixed = _make_config(candidate_budget_mode='fixed', candidate_budget_reference='match_and')
    for strat in ['or', 'abs_only', 'conf_only']:
        mask, _, _ = select_densification_candidates(
            abs_s, conf_s, rfas_s, abs_m, conf_m, strategy=strat, config=cfg_fixed)
        assert mask.sum().item() == k_and, f"{strat}: match_and should equal AND baseline"
    print("  PASS test_match_and_count_match")


# ---------------------------------------------------------------------------
# 测试 8: NaN 和 Inf 不会被选入候选点
# ---------------------------------------------------------------------------
def test_nan_inf_exclusion():
    scores = torch.rand(100)
    scores[0] = float('nan')
    scores[1] = float('inf')
    scores[2] = float('-inf')

    mask = select_by_topk(scores, 5)
    assert not mask[0], "NaN should be excluded"
    assert not mask[1], "+Inf should be excluded"
    assert not mask[2], "-Inf should be excluded"
    assert mask.sum().item() == 5

    # 所有 NaN 的情况
    all_nan = torch.full((100,), float('nan'))
    mask = select_by_topk(all_nan, 5)
    assert mask.sum().item() == 0, "All NaN should produce empty mask"
    print("  PASS test_nan_inf_exclusion")


# ---------------------------------------------------------------------------
# 测试 9: 空候选集不会报错
# ---------------------------------------------------------------------------
def test_empty_candidates():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _make_tensors()
    cfg = _make_config()
    zero_mask = torch.zeros(100, dtype=torch.bool)
    zero_score = torch.zeros(100)

    # 空掩码
    mask, _, _ = select_densification_candidates(
        abs_s, conf_s, rfas_s, zero_mask, zero_mask, strategy='and', config=cfg)
    assert mask.sum().item() == 0

    # topk k=0
    mask = select_by_topk(torch.rand(100), 0)
    assert mask.sum().item() == 0

    # N=0
    empty = torch.zeros(0)
    mask = select_by_topk(empty, 10)
    assert mask.numel() == 0
    print("  PASS test_empty_candidates")


# ---------------------------------------------------------------------------
# 测试 10: 所有输出 Tensor 均处于正确设备
# ---------------------------------------------------------------------------
def test_device_consistency():
    device = 'cpu'
    abs_s, conf_s, rfas_s, abs_m, conf_m = _make_tensors(device=device)
    cfg = _make_config()

    for strat in VALID_STRATEGIES:
        mask, scores, _ = select_densification_candidates(
            abs_s, conf_s, rfas_s, abs_m, conf_m, strategy=strat, config=cfg)
        assert mask.device.type == device, f"{strat}: mask device mismatch"
        assert scores.device.type == device, f"{strat}: scores device mismatch"
    print("  PASS test_device_consistency")


# ---------------------------------------------------------------------------
# 测试 11: 所有输出 mask 的 dtype 为 torch.bool
# ---------------------------------------------------------------------------
def test_dtype_bool():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _make_tensors()
    cfg = _make_config()

    for strat in VALID_STRATEGIES:
        mask, _, _ = select_densification_candidates(
            abs_s, conf_s, rfas_s, abs_m, conf_m, strategy=strat, config=cfg)
        assert mask.dtype == torch.bool, f"{strat}: mask dtype should be bool, got {mask.dtype}"
    print("  PASS test_dtype_bool")


# ---------------------------------------------------------------------------
# 测试 12: 默认参数下 AND 策略与原始实现逐元素一致
# ---------------------------------------------------------------------------
def test_default_and_matches_original():
    """
    验证默认 AND 策略:
    - 不使用任何新参数（或使用默认值）
    - 输出 == abs_mask & conf_mask
    - stats 包含所有必要字段
    """
    abs_s, conf_s, rfas_s, abs_m, conf_m = _make_tensors(N=500)
    cfg = _make_config()  # 全部默认值

    mask, scores, stats = select_densification_candidates(
        abs_s, conf_s, rfas_s, abs_m, conf_m, strategy='and', config=cfg)

    expected = abs_m & conf_m
    assert torch.equal(mask, expected), "Default AND should equal abs_mask & conf_mask"

    # 验证 stats 完整性
    required_keys = [
        'strategy', 'n_valid', 'n_abs_candidates', 'n_conf_candidates',
        'n_and_candidates', 'n_or_candidates', 'n_final_candidates',
        'candidate_ratio', 'target_budget', 'actual_budget',
        'abs_mean', 'abs_std', 'abs_median',
        'conf_mean', 'conf_std', 'conf_median',
        'rfas_mean', 'rfas_std', 'rfas_median',
        'sel_score_mean', 'sel_score_std',
        'iou_abs_conf', 'overlap_abs', 'overlap_conf',
        'n_nan', 'n_inf',
    ]
    for key in required_keys:
        assert key in stats, f"Missing stats key: {key}"

    # 验证 AND 逻辑一致性
    expected_and = (abs_m & conf_m).sum().item()
    assert stats['n_and_candidates'] == expected_and, \
        f"n_and should be abs_mask & conf_mask count: {stats['n_and_candidates']} vs {expected_and}"
    print("  PASS test_default_and_matches_original")


# ---------------------------------------------------------------------------
# 辅助测试：数值稳定性
# ---------------------------------------------------------------------------
def test_normalization_edge_cases():
    # 全部相同值
    s = torch.ones(100)
    for method in ['minmax', 'zscore', 'percentile']:
        norm = normalize_scores(s, method=method)
        assert torch.isfinite(norm).all(), f"{method}: should be finite"

    # 全部为零
    s = torch.zeros(100)
    for method in ['minmax', 'zscore', 'percentile']:
        norm = normalize_scores(s, method=method)
        assert torch.isfinite(norm).all(), f"{method}: all-zero should be finite"

    # 包含 NaN
    s = torch.rand(100)
    s[0] = float('nan')
    for method in ['minmax', 'zscore', 'percentile']:
        norm = normalize_scores(s, method=method)
        assert not torch.isnan(norm).any(), f"{method}: should not produce NaN"

    print("  PASS test_normalization_edge_cases")


def test_soft_mask_direction():
    scores = torch.rand(100) * 2.0
    tau = 1.0
    T = 1.0

    # sign=+1: scores > tau → P接近1
    P_pos = compute_soft_mask(scores, threshold=tau, temperature=T, sign=+1)
    assert (P_pos[scores > tau].mean() > 0.5), "sign=+1: high scores should have high P"

    # sign=-1: scores > tau → P接近0
    P_neg = compute_soft_mask(scores, threshold=tau, temperature=T, sign=-1)
    assert (P_neg[scores > tau].mean() < 0.5), "sign=-1: high scores should have low P"

    print("  PASS test_soft_mask_direction")


def test_invalid_strategy_raises():
    abs_s, conf_s, rfas_s, abs_m, conf_m = _make_tensors()
    cfg = _make_config()

    try:
        select_densification_candidates(
            abs_s, conf_s, rfas_s, abs_m, conf_m,
            strategy='invalid_strategy', config=cfg)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'invalid_strategy' in str(e)
    print("  PASS test_invalid_strategy_raises")


# ---------------------------------------------------------------------------
# 运行所有测试
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    tests = [
        test_and_equals_abs_and_conf,
        test_or_equals_abs_or_conf,
        test_abs_only_equals_abs,
        test_conf_only_equals_conf,
        test_topk_count,
        test_match_and_budget_consistency,
        test_match_and_count_match,
        test_nan_inf_exclusion,
        test_empty_candidates,
        test_device_consistency,
        test_dtype_bool,
        test_default_and_matches_original,
        test_normalization_edge_cases,
        test_soft_mask_direction,
        test_invalid_strategy_raises,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)
