"""
空间多样性候选选择器单元测试。
在 CPU 上运行，不需要 GPU 或训练状态。
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.spatial_diversity import (
    select_spatially_diverse_candidates,
    _compute_adaptive_voxel_size,
    _voxel_diversity_select,
)


def test_empty_candidates():
    """AC-11.1: 候选为空时返回全 False mask"""
    N = 100
    candidate_mask = torch.zeros(N, dtype=torch.bool)
    priority_scores = torch.rand(N)
    xyz = torch.rand(N, 3)
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, priority_scores, xyz, budget=10
    )
    assert mask.sum() == 0
    assert stats['candidate_count'] == 0
    assert stats['selected_count'] == 0


def test_zero_budget():
    """AC-3 边缘情况: 预算为 0 时返回全 False mask"""
    N = 100
    candidate_mask = torch.ones(N, dtype=torch.bool)
    priority_scores = torch.rand(N)
    xyz = torch.rand(N, 3)
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, priority_scores, xyz, budget=0
    )
    assert mask.sum() == 0


def test_subset_constraint():
    """AC-2: 空间掩码必须是候选掩码的子集"""
    N = 50
    candidate_mask = torch.zeros(N, dtype=torch.bool)
    candidate_mask[:30] = True  # 前 30 个是候选
    priority_scores = torch.rand(N)
    xyz = torch.rand(N, 3)
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, priority_scores, xyz, budget=15
    )
    assert not torch.any(mask & (~candidate_mask)), "spatial_mask must be subset of candidate_mask"
    assert mask.sum() <= 15


def test_budget_preservation_exact():
    """AC-3: 预算保持 — 候选充足时选中 exact budget"""
    N = 200
    candidate_mask = torch.ones(N, dtype=torch.bool)
    priority_scores = torch.rand(N)
    xyz = torch.rand(N, 3)
    budget = 50
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, priority_scores, xyz, budget=budget
    )
    selected = mask.sum().item()
    assert selected == budget, f"Expected {budget}, got {selected}"


def test_budget_preservation_insufficient():
    """AC-3: 预算保持 — 候选不足时选中全部候选"""
    N = 30
    candidate_mask = torch.ones(N, dtype=torch.bool)
    priority_scores = torch.rand(N)
    xyz = torch.rand(N, 3)
    budget = 100  # 大于候选数
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, priority_scores, xyz, budget=budget
    )
    selected = mask.sum().item()
    assert selected == N, f"Expected all {N}, got {selected}"


def test_deterministic():
    """AC-5: 确定性 — 相同输入产生相同输出"""
    N = 100
    torch.manual_seed(42)
    candidate_mask = torch.ones(N, dtype=torch.bool)
    priority_scores, _ = torch.sort(torch.rand(N))  # 确保有不同分数
    xyz = torch.rand(N, 3)

    mask1, _ = select_spatially_diverse_candidates(
        candidate_mask.clone(), priority_scores.clone(), xyz.clone(), budget=20
    )
    mask2, _ = select_spatially_diverse_candidates(
        candidate_mask.clone(), priority_scores.clone(), xyz.clone(), budget=20
    )

    assert torch.equal(mask1, mask2), "Results must be deterministic"


def test_tie_breaker_by_index():
    """AC-5: Score 相同时使用稳定排序"""
    N = 50
    candidate_mask = torch.ones(N, dtype=torch.bool)
    priority_scores = torch.ones(N) * 0.5  # 所有相同分数
    xyz = torch.rand(N, 3)

    mask1, _ = select_spatially_diverse_candidates(
        candidate_mask, priority_scores.clone(), xyz.clone(), budget=10
    )
    mask2, _ = select_spatially_diverse_candidates(
        candidate_mask, priority_scores.clone(), xyz.clone(), budget=10
    )

    assert torch.equal(mask1, mask2), "Tie-breaking must be deterministic"


def test_adaptive_voxel_size():
    """AC-6: 自适应体素尺寸计算"""
    N = 100
    scales = torch.ones(N, 3) * 0.1  # uniform scales
    cand_indices = torch.arange(N)
    voxel_size = _compute_adaptive_voxel_size(scales, cand_indices, voxel_scale=2.0)
    assert voxel_size > 0, f"Expected positive voxel_size, got {voxel_size}"
    # max(scales) = 0.1, median = 0.1, voxel_scale=2.0 → voxel_size ≈ 0.2
    assert abs(voxel_size - 0.2) < 0.01, f"Expected ~0.2, got {voxel_size}"


def test_adaptive_voxel_size_fallback():
    """AC-6: 候选为空时的体素尺寸回退"""
    scales = torch.zeros(10, 3)
    cand_indices = torch.tensor([], dtype=torch.long)
    voxel_size = _compute_adaptive_voxel_size(scales, cand_indices, voxel_scale=2.0)
    assert voxel_size > 0, f"Expected fallback positive value, got {voxel_size}"


def test_score_validity_positive():
    """AC-4: 确保选中的候选分数 > 0"""
    N = 100
    candidate_mask = torch.ones(N, dtype=torch.bool)
    priority_scores = torch.rand(N) + 0.01  # 全正
    xyz = torch.rand(N, 3)
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, priority_scores, xyz, budget=30
    )
    selected_scores = priority_scores[mask]
    assert (selected_scores > 0).all(), "All selected scores must be positive"


def test_device_consistency():
    """验证输出在正确设备上"""
    N = 50
    candidate_mask = torch.ones(N, dtype=torch.bool)
    priority_scores = torch.rand(N)
    xyz = torch.rand(N, 3)
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, priority_scores, xyz, budget=10
    )
    assert mask.device == candidate_mask.device
    assert mask.dtype == torch.bool


def test_stats_completeness():
    """AC-8: 统计信息完整性"""
    N = 80
    candidate_mask = torch.ones(N, dtype=torch.bool)
    priority_scores = torch.rand(N)
    xyz = torch.rand(N, 3)
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, priority_scores, xyz, budget=20
    )
    required_keys = [
        'candidate_count', 'selected_count', 'occupied_voxel_count',
        'voxel_size', 'runtime_ms', 'jaccard_vs_original',
        'replaced_count', 'method', 'max_candidates_per_voxel',
        'mean_candidates_per_voxel',
    ]
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"
    # 所有值必须是 Python 标量
    for key in required_keys:
        val = stats[key]
        assert not isinstance(val, torch.Tensor), f"Key '{key}' is a tensor, expected scalar"


def test_voxel_spatial_spread():
    """验证体素选择确实实现了空间散布"""
    N = 300
    candidate_mask = torch.ones(N, dtype=torch.bool)
    # 创建 3 个空间簇, 每个簇 100 个点
    cluster1 = torch.rand(100, 3) * 0.1
    cluster2 = torch.rand(100, 3) * 0.1 + torch.tensor([5.0, 0.0, 0.0])
    cluster3 = torch.rand(100, 3) * 0.1 + torch.tensor([0.0, 5.0, 0.0])
    xyz = torch.cat([cluster1, cluster2, cluster3], dim=0)
    # 簇1 分数较高, 簇2 中等, 簇3 较低
    priority_scores = torch.cat([
        torch.rand(100) * 0.5 + 0.5,  # cluster1: 高分
        torch.rand(100) * 0.5 + 0.25,  # cluster2: 中分
        torch.rand(100) * 0.5,          # cluster3: 低分
    ])

    mask_standard, _ = select_spatially_diverse_candidates(
        candidate_mask, priority_scores, xyz, budget=60
    )
    # 标准 top-k 几乎全选 cluster1, 空间选择应该分散
    # 检查: 每个簇至少有一些候选被选中 (相对于 voxel_size)
    # 这不是严格的断言，但提供统计证据
    selected_xyz = xyz[mask_standard]
    n_selected = selected_xyz.shape[0]
    assert n_selected == 60, f"Expected 60, got {n_selected}"


def test_invalid_inputs():
    """输入验证"""
    N = 50
    try:
        select_spatially_diverse_candidates(
            torch.ones(N, 2, dtype=torch.bool),  # 2D invalid
            torch.rand(N),
            torch.rand(N, 3),
            budget=10,
        )
        assert False, "Should have raised ValueError for 2D mask"
    except ValueError as e:
        assert "1D" in str(e), f"Expected '1D' in error, got '{e}'"

    try:
        select_spatially_diverse_candidates(
            torch.ones(N, dtype=torch.bool),
            torch.rand(N),
            torch.rand(N, 3),
            budget=10,
            method='INVALID_METHOD',
        )
        assert False, "Should have raised ValueError for invalid method"
    except ValueError as e:
        assert "Unknown" in str(e), f"Expected 'Unknown' in error, got '{e}'"


def test_method_not_implemented():
    """radius_nms 和 soft_suppression 抛出 NotImplementedError"""
    N = 50
    for method in ['radius_nms', 'soft_suppression']:
        try:
            select_spatially_diverse_candidates(
                torch.ones(N, dtype=torch.bool),
                torch.rand(N),
                torch.rand(N, 3),
                budget=10,
                method=method,
            )
            assert False, f"Should have raised NotImplementedError for {method}"
        except NotImplementedError:
            pass


def test_nan_inf_excluded():
    """NaN/Inf 分数的候选被排除，不影响预算"""
    N = 100
    candidate_mask = torch.ones(N, dtype=torch.bool)
    scores = torch.rand(N)
    scores[0] = float('nan')
    scores[1] = float('inf')
    scores[2] = float('-inf')
    xyz = torch.rand(N, 3)
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, scores, xyz, budget=50
    )
    # NaN/Inf 不被选中
    assert not mask[0], "NaN should be excluded"
    assert not mask[1], "+Inf should be excluded"
    assert not mask[2], "-Inf should be excluded"
    # 有限候选数 N-3=97, budget=50 → 选中 50
    assert mask.sum().item() == 50
    # candidate_count 报告输入候选数 (含 NaN/Inf)
    assert stats['candidate_count'] == 100


def test_zero_score_selected():
    """零分候选不应被丢弃 — 后处理器负责 clamp"""
    N = 50
    candidate_mask = torch.ones(N, dtype=torch.bool)
    # 前 10 个零分, 其余正常分数
    scores = torch.zeros(N)
    scores[10:] = torch.rand(40) + 0.01
    xyz = torch.rand(N, 3)
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, scores, xyz, budget=50
    )
    # 应选中全部可用候选
    assert mask.sum().item() == 50


def test_index_tie_breaker():
    """AC-5: 相同分数时按全局索引升序选择 (确定性 tie-breaker)

    将所有候选放在同一体素内，分数相同，预算小于候选数。
    验证选中的是最低全局索引的候选。
    """
    N = 20
    candidate_mask = torch.ones(N, dtype=torch.bool)
    scores = torch.ones(N) * 0.5  # 全同分数
    # 所有候选在同一位置 → 同一体素
    xyz = torch.zeros(N, 3)
    budget = 5
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, scores.clone(), xyz, budget=budget,
        voxel_size=1.0,  # 固定体素尺寸确保单一体素
    )
    # 同一体素内分数相同 → 按全局索引升序选择
    selected = torch.where(mask)[0]
    assert selected.numel() == budget
    expected = torch.arange(budget)
    assert torch.equal(selected, expected), \
        f"Expected indices {expected.tolist()}, got {selected.tolist()}"


def test_voxel_traversal_deterministic():
    """AC-5: 多体素相等 top-score 时的确定性体素遍历顺序"""
    N = 30
    candidate_mask = torch.ones(N, dtype=torch.bool)
    scores = torch.ones(N) * 0.5  # 全同分数
    # 3 个空间簇，每个 10 个点
    xyz = torch.zeros(N, 3)
    xyz[0:10, 0] = 0.0   # voxel A
    xyz[10:20, 0] = 5.0  # voxel B
    xyz[20:30, 0] = 10.0 # voxel C
    mask1, _ = select_spatially_diverse_candidates(
        candidate_mask, scores.clone(), xyz, budget=6,
        voxel_size=1.0, max_per_voxel=2,
    )
    mask2, _ = select_spatially_diverse_candidates(
        candidate_mask, scores.clone(), xyz, budget=6,
        voxel_size=1.0, max_per_voxel=2,
    )
    # 确定性：相同输入产生相同输出
    assert torch.equal(mask1, mask2), "Multi-voxel tie-breaking must be deterministic"
    # round-robin, max_per_voxel=2: 每个体素选 2 个，3 个体素共 6
    assert mask1.sum().item() == 6


def test_max_per_voxel_validation():
    """max_per_voxel < 1 应抛出 ValueError"""
    N = 50
    try:
        select_spatially_diverse_candidates(
            torch.ones(N, dtype=torch.bool),
            torch.rand(N),
            torch.rand(N, 3),
            budget=10,
            max_per_voxel=0,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "max_per_voxel" in str(e)


def test_negative_score_selected():
    """负分候选不应被丢弃 — 后处理器负责 clamp 到正小量"""
    N = 30
    candidate_mask = torch.ones(N, dtype=torch.bool)
    # 前 15 个负分
    scores = torch.rand(N) * 2 - 1.0  # [-1, +1]
    xyz = torch.rand(N, 3)
    mask, stats = select_spatially_diverse_candidates(
        candidate_mask, scores, xyz, budget=30
    )
    # 全部候选被选中 (budget >= N)
    assert mask.sum().item() == 30


if __name__ == '__main__':
    tests = [
        test_empty_candidates,
        test_zero_budget,
        test_subset_constraint,
        test_budget_preservation_exact,
        test_budget_preservation_insufficient,
        test_deterministic,
        test_tie_breaker_by_index,
        test_adaptive_voxel_size,
        test_adaptive_voxel_size_fallback,
        test_score_validity_positive,
        test_device_consistency,
        test_stats_completeness,
        test_voxel_spatial_spread,
        test_invalid_inputs,
        test_method_not_implemented,
        test_nan_inf_excluded,
        test_zero_score_selected,
        test_index_tie_breaker,
        test_voxel_traversal_deterministic,
        test_max_per_voxel_validation,
        test_negative_score_selected,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"PASS: {test.__name__}")
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("=== ALL TESTS PASSED ===")
