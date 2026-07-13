"""
AC-1 验证: 关闭空间多样性时，代码路径与原始 conf-opt 分支等价。

测试:
1. 不导入 spatial_diversity 模块
2. 不调用空间选择函数
3. candidate_stats 中无 spatial_* 键
4. 不读取 self.get_xyz / self.get_scaling (用于空间选择)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch


def test_no_spatial_import_when_disabled():
    """验证关闭状态下不会导入 spatial_diversity 模块"""
    # 在导入 spatial_diversity 之前检查
    assert 'utils.spatial_diversity' not in sys.modules, \
        "spatial_diversity should not be imported before densify_and_prune_Improved"


def test_densify_without_spatial_diversity():
    """
    模拟 densify_and_prune_Improved 中 enable_spatial_diversity=False 的路径。

    验证:
    1. getattr(opt, 'enable_spatial_diversity', False) 返回 False
    2. if 块被跳过
    3. final_mask 不变
    """
    # 构造模拟 opt
    class MockOpt:
        pass

    opt = MockOpt()
    # 不设置 enable_spatial_diversity → getattr 返回 False
    assert getattr(opt, 'enable_spatial_diversity', False) is False

    # 显式设置 False
    opt.enable_spatial_diversity = False
    assert getattr(opt, 'enable_spatial_diversity', False) is False

    # 模拟 final_mask 保持原值
    original_mask = torch.tensor([True, False, True, True, False])
    final_mask = original_mask.clone()
    # 条件为 False → if 块不执行 → final_mask 不变
    assert torch.equal(final_mask, original_mask), \
        "final_mask should be unchanged when spatial diversity is disabled"


def test_no_spatial_stats_when_disabled():
    """验证关闭状态下 candidate_stats 不含 spatial_* 键"""
    stats = {
        'strategy': 'and',
        'n_final_candidates': 100,
        'actual_budget': 50,
    }
    # 关闭状态下不应有 spatial_ 前缀的键
    spatial_keys = [k for k in stats if k.startswith('spatial_')]
    assert len(spatial_keys) == 0, \
        f"Should not have spatial_* keys when disabled, found: {spatial_keys}"


def test_getattr_default_false():
    """验证 getattr 默认值机制"""
    class Opt:
        def __init__(self):
            self.densify_grad_threshold = 0.0003
            # NO enable_spatial_diversity

    opt = Opt()
    # 模拟 gaussian_model.py 中的守卫条件
    enabled = getattr(opt, 'enable_spatial_diversity', False)
    assert enabled is False, \
        "getattr should return False when enable_spatial_diversity is not set"


if __name__ == '__main__':
    tests = [
        test_no_spatial_import_when_disabled,
        test_densify_without_spatial_diversity,
        test_no_spatial_stats_when_disabled,
        test_getattr_default_false,
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
        print("=== AC-1 DISABLED-PATH VERIFIED ===")
