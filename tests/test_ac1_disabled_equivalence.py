"""
AC-1 真实生产路径验证: 关闭空间多样性时的行为等价性。

通过构造最小 GaussianModel 实例并实际调用 densify_and_prune_Improved()
来验证禁用路径不导入 spatial_diversity、不产生 spatial_* 统计键。
"""

import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

# 确保测试开始时 spatial_diversity 未被导入
assert 'utils.spatial_diversity' not in sys.modules, \
    "spatial_diversity must not be imported before disabled-path test"


def _make_tiny_gaussian_model(N=10):
    """构造最小 GaussianModel 实例用于测试，所有张量在 CPU 上。"""
    from scene.gaussian_model import GaussianModel

    gs = GaussianModel.__new__(GaussianModel)

    # 基本属性
    gs._xyz = nn.Parameter(torch.randn(N, 3))
    gs._features_dc = nn.Parameter(torch.randn(N, 1, 3))
    gs._features_rest = nn.Parameter(torch.zeros(N, 3, 3))
    gs._scaling = nn.Parameter(torch.randn(N, 3) * 0.1 - 2.0)  # log space
    gs._rotation = nn.Parameter(torch.randn(N, 4))
    gs._opacity = nn.Parameter(torch.randn(N, 1))

    gs.scaling_activation = torch.exp
    gs.scaling_inverse_activation = torch.log
    gs.rotation_activation = lambda x: torch.nn.functional.normalize(x)
    gs.opacity_activation = torch.sigmoid

    # 梯度统计 — 设置高值以确保 abs_mask 和 conf_mask 大部分为 True
    gs.xyz_gradient_accum = torch.ones(N, 1) * 0.01  # 高梯度
    gs.denom = torch.ones(N, 1) * 0.01  # 小分母 → 高 grad_var
    gs.xyz_gradient_vec_accum = torch.zeros(N, 3)  # 零向量 → |sum|=0
    gs.xyz_gradient_mag_accum = torch.ones(N, 1) * 0.1  # 正分母 → conf ≈ 1.0
    gs.xyz_gradient_conf_denom = torch.ones(N, 1) * 5  # >= min_views
    gs.max_radii2D = torch.zeros(N)

    # 属性访问
    # get_xyz, get_scaling 等是 @property，会自动从 _xyz/_scaling 读取
    # 只需确保 _xyz/_scaling 等已设置即可

    # 活跃 SH
    gs.active_sh_degree = 0

    # candidate_stats
    gs.candidate_stats = {}

    return gs


def test_disabled_path_no_spatial_import():
    """AC-1: 禁用路径中不导入 spatial_diversity 模块。

    构造真实 GaussianModel, 调用 densify_and_prune_Improved()
    并验证 spatial_diversity 未被导入。
    """
    from scene.gaussian_model import GaussianModel

    gs = _make_tiny_gaussian_model(N=20)
    assert 'utils.spatial_diversity' not in sys.modules

    # 模拟 opt
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
        # enable_spatial_diversity is NOT set → getattr returns False

    opt = Opt()
    scores = torch.randn(20)
    budget = 50000

    # Monkeypatch long_axis_split 和 prune_points 避免 CUDA 操作
    las_called = []
    with mock.patch.object(GaussianModel, 'long_axis_split',
                           lambda self, *a, **kw: las_called.append(True)):
        with mock.patch.object(GaussianModel, 'prune_points',
                               lambda self, *a, **kw: None):
            gs.densify_and_prune_Improved(
                scores, min_opacity=0.005, budget=budget,
                opt=opt, iteration=600, limitation=1300000,
            )

    # 验证 spatial_diversity 仍然未被导入
    assert 'utils.spatial_diversity' not in sys.modules, \
        "spatial_diversity was imported despite enable_spatial_diversity=False"

    # 验证 candidate_stats 不含 spatial_* 键
    spatial_keys = [k for k in gs.candidate_stats if k.startswith('spatial_')]
    assert len(spatial_keys) == 0, \
        f"candidate_stats has spatial_* keys when disabled: {spatial_keys}"

    # 验证 LAS 被调用
    assert len(las_called) > 0, "long_axis_split should have been called"


def test_enabled_path_with_spatial_import():
    """AC-1 补充: 启用路径中正确导入 spatial_diversity。

    验证 enable_spatial_diversity=True 时模块被导入且 spatial_* 统计键存在。
    """
    from scene.gaussian_model import GaussianModel

    # 先清理已导入的 spatial_diversity (如果之前有)
    sys.modules.pop('utils.spatial_diversity', None)

    gs = _make_tiny_gaussian_model(N=20)

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
        enable_spatial_diversity = True  # 启用!
        spatial_voxel_size = 'auto'
        spatial_diversity_method = 'voxel'
        spatial_max_per_voxel = 1
        spatial_voxel_scale = 2.0
        spatial_radius_scale = 1.0
        spatial_suppression_weight = 1.0

    opt = Opt()
    scores = torch.randn(20)
    budget = 50000

    with mock.patch.object(GaussianModel, 'long_axis_split',
                           lambda self, *a, **kw: None):
        with mock.patch.object(GaussianModel, 'prune_points',
                               lambda self, *a, **kw: None):
            gs.densify_and_prune_Improved(
                scores, min_opacity=0.005, budget=budget,
                opt=opt, iteration=600, limitation=1300000,
            )

    # 启用时 spatial_diversity 被导入
    assert 'utils.spatial_diversity' in sys.modules, \
        "spatial_diversity should be imported when enabled"

    # 验证 spatial_* 统计键存在
    spatial_keys = [k for k in gs.candidate_stats if k.startswith('spatial_')]
    assert len(spatial_keys) > 0, \
        f"candidate_stats should have spatial_* keys when enabled"


def test_getattr_default_behavior():
    """验证 getattr 默认值在 opt 未设置 enable_spatial_diversity 时返回 False"""
    class Opt:
        pass
    opt = Opt()
    assert getattr(opt, 'enable_spatial_diversity', False) is False


if __name__ == '__main__':
    tests = [
        test_getattr_default_behavior,
        test_disabled_path_no_spatial_import,
        test_enabled_path_with_spatial_import,
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
        print("=== AC-1 PRODUCTION-PATH VERIFIED ===")
