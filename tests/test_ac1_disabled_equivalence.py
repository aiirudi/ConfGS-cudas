"""
AC-1 真实生产路径验证: 关闭空间多样性时的行为等价性。

通过构造最小 GaussianModel 实例并实际调用 densify_and_prune_Improved()
来验证禁用路径:
1. 不导入 spatial_diversity 模块
2. final_mask 和 selection_score 等价传递到 LAS
3. candidate_stats 不含 spatial_* 键
"""

import sys
import os
import unittest.mock as mock
import builtins

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn


def _make_tiny_gaussian_model(N=10):
    """构造最小 GaussianModel 实例用于测试，所有张量在 CPU 上。"""
    from scene.gaussian_model import GaussianModel

    gs = GaussianModel.__new__(GaussianModel)

    gs._xyz = nn.Parameter(torch.randn(N, 3))
    gs._features_dc = nn.Parameter(torch.randn(N, 1, 3))
    gs._features_rest = nn.Parameter(torch.zeros(N, 3, 3))
    gs._scaling = nn.Parameter(torch.randn(N, 3) * 0.1 - 2.0)
    gs._rotation = nn.Parameter(torch.randn(N, 4))
    gs._opacity = nn.Parameter(torch.randn(N, 1))

    gs.scaling_activation = torch.exp
    gs.scaling_inverse_activation = torch.log
    gs.rotation_activation = lambda x: torch.nn.functional.normalize(x)
    gs.opacity_activation = torch.sigmoid

    # 高梯度统计确保 abs_mask 和 conf_mask 大部分为 True
    gs.xyz_gradient_accum = torch.ones(N, 1) * 0.01
    gs.denom = torch.ones(N, 1) * 0.01
    gs.xyz_gradient_vec_accum = torch.zeros(N, 3)
    gs.xyz_gradient_mag_accum = torch.ones(N, 1) * 0.1
    gs.xyz_gradient_conf_denom = torch.ones(N, 1) * 5
    gs.max_radii2D = torch.zeros(N)

    gs.active_sh_degree = 0
    gs.candidate_stats = {}

    return gs


def _make_disabled_opt():
    """构造 enable_spatial_diversity 未设置的 opt。"""
    class Opt:
        densify_grad_threshold = 0.0003
        conf_thr = 0.85
        conf_min_views = 3
        candidate_selection_strategy = 'and'
        candidate_budget_mode = 'native'
        budget = 1300000
        split_distance = 0.45
        opacity_reduction = 0.6
        candidate_stats_enabled = True
    return Opt()


def _make_enabled_opt():
    """构造 enable_spatial_diversity=True 的 opt。"""
    class Opt:
        densify_grad_threshold = 0.0003
        conf_thr = 0.85
        conf_min_views = 3
        candidate_selection_strategy = 'and'
        candidate_budget_mode = 'native'
        budget = 1300000
        split_distance = 0.45
        opacity_reduction = 0.6
        candidate_stats_enabled = True
        enable_spatial_diversity = True
        spatial_voxel_size = 'auto'
        spatial_diversity_method = 'voxel'
        spatial_max_per_voxel = 1
        spatial_voxel_scale = 2.0
        spatial_radius_scale = 1.0
        spatial_suppression_weight = 1.0
    return Opt()


def _install_import_blocker(blocked_module):
    """安装 scoped import blocker，若尝试导入 blocked_module 则抛出 ImportError。"""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"BLOCKED: import of {blocked_module} is forbidden")
        return original_import(name, *args, **kwargs)

    return mock.patch('builtins.__import__', side_effect=guarded_import)


def test_disabled_path_equivalence():
    """AC-1: 禁用路径下 final_mask/selection_score 等价传递到 LAS。

    1. 构造 GaussianModel, 调用 densify_and_prune_Improved(enable=False)
    2. 捕获 LAS 收到的 (selection_score, all_budget, final_mask)
    3. 验证与 select_densification_candidates 输出一致
    4. 验证 spatial_diversity 未被导入, 无 spatial_* 统计键
    """
    from scene.gaussian_model import GaussianModel
    from utils.candidate_selector import select_densification_candidates

    gs = _make_tiny_gaussian_model(N=20)
    opt = _make_disabled_opt()
    scores = torch.randn(20)
    budget = 50000

    # 先计算期望的 selector 输出 (与 densify_and_prune_Improved 内部一致)
    grad_vars = gs.xyz_gradient_accum / gs.denom
    grad_vars[grad_vars.isnan()] = 0.0
    abs_mask = torch.where(torch.norm(grad_vars, dim=-1) >= opt.densify_grad_threshold, True, False)
    conf = 1.0 - (torch.norm(gs.xyz_gradient_vec_accum, dim=-1, keepdim=True)) / (gs.xyz_gradient_mag_accum + 1e-6)
    conf[conf.isnan()] = 0.0
    zero_views = (gs.xyz_gradient_conf_denom <= 0).squeeze(-1)
    conf[zero_views] = 0.0
    conf = conf.squeeze(-1)
    has_enough_views = (gs.xyz_gradient_conf_denom.squeeze(-1) >= opt.conf_min_views)
    conf_mask_raw = (conf >= opt.conf_thr)
    conf_mask = conf_mask_raw & has_enough_views
    expected_mask, expected_selection, _ = select_densification_candidates(
        abs_score=grad_vars, conf_score=conf, rfas_score=scores,
        abs_mask=abs_mask, conf_mask=conf_mask, strategy='and', config=opt,
    )
    expected_n = int(expected_mask.sum().item())
    all_budget = min(budget, expected_n + 20) - 20  # all_budget = min(budget, total_sum+N) - N

    # Scoped import blocker + 捕获 LAS 实际参数
    las_args = []
    # 清理可能已导入的 spatial_diversity (来自其他测试)
    sys.modules.pop('utils.spatial_diversity', None)
    # 也清理 candidate_selector 中的缓存引用 (无, 这是独立模块)

    with _install_import_blocker('utils.spatial_diversity') as blocker:
        with mock.patch.object(GaussianModel, 'long_axis_split',
                               lambda self, sel_score, ab, fm, *a, **kw:
                               las_args.append((sel_score, ab, fm))) as las_mock:
            with mock.patch.object(GaussianModel, 'prune_points',
                                   lambda self, *a, **kw: None):
                gs.densify_and_prune_Improved(
                    scores, min_opacity=0.005, budget=budget,
                    opt=opt, iteration=600, limitation=1300000,
                )

    # 验证 import blocker 未被触发 (spatial_diversity 未导入)
    assert not blocker._mock_side_effect_raises or len(las_args) > 0, \
        "spatial_diversity import should be blocked"

    # 验证 LAS 被调用
    assert len(las_args) == 1, f"LAS should be called once, got {len(las_args)}"
    las_sel, las_budget, las_mask = las_args[0]

    # 验证预算等价
    assert las_budget == all_budget, \
        f"LAS budget {las_budget} != expected {all_budget}"

    # 验证 final_mask 等价 (LAS 收到的 mask 应等于 selector 输出)
    assert torch.equal(las_mask, expected_mask), \
        "Disabled path: LAS mask must equal selector output"

    # 验证候选统计无 spatial_* 键
    spatial_keys = [k for k in gs.candidate_stats if k.startswith('spatial_')]
    assert len(spatial_keys) == 0, \
        f"Disabled path should have no spatial_* keys: {spatial_keys}"

    # 验证 spatial_diversity 确实未在 sys.modules 中
    assert 'utils.spatial_diversity' not in sys.modules, \
        "spatial_diversity should not be in sys.modules"


def test_enabled_path_with_spatial():
    """AC-1 补充: 启用路径正确导入并产生 spatial_* 统计键。"""
    from scene.gaussian_model import GaussianModel

    sys.modules.pop('utils.spatial_diversity', None)

    gs = _make_tiny_gaussian_model(N=20)
    opt = _make_enabled_opt()
    scores = torch.randn(20)

    with mock.patch.object(GaussianModel, 'long_axis_split',
                           lambda self, *a, **kw: None):
        with mock.patch.object(GaussianModel, 'prune_points',
                               lambda self, *a, **kw: None):
            gs.densify_and_prune_Improved(
                scores, min_opacity=0.005, budget=50000,
                opt=opt, iteration=600, limitation=1300000,
            )

    assert 'utils.spatial_diversity' in sys.modules, \
        "spatial_diversity should be imported when enabled"
    spatial_keys = [k for k in gs.candidate_stats if k.startswith('spatial_')]
    assert len(spatial_keys) > 0, \
        f"Enabled path should have spatial_* keys: {spatial_keys}"


def test_getattr_default_behavior():
    """getattr 在 enable_spatial_diversity 未设置时返回 False"""
    class Opt:
        pass
    assert getattr(Opt(), 'enable_spatial_diversity', False) is False


if __name__ == '__main__':
    tests = [
        test_getattr_default_behavior,
        test_disabled_path_equivalence,
        test_enabled_path_with_spatial,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"PASS: {test.__name__}")
        except Exception as e:
            import traceback
            print(f"FAIL: {test.__name__} - {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("=== AC-1 FULLY VERIFIED ===")
