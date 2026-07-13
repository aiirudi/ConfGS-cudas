"""
空间多样性候选选择模块。

在 EAS/RFAS 评分之后，对候选 Gaussian 进行空间多样性选择，
通过体素网格分区和轮转选择（round-robin selection）抑制空间冗余，
将致密化预算重新分配给空间上更多样化的候选。

该模块独立于 Conf、abs-grad、EAS 等现有组件，仅读取已有候选掩码、
优先级分数和 Gaussian 空间位置，返回新的空间选择掩码。

Methods:
  - voxel: 体素分区 + 确定性 round-robin 选择 (v1 唯一实现)
  - radius_nms: 预留接口 (未实现)
  - soft_suppression: 预留接口 (未实现)
"""

import time
import torch
from typing import Dict, Optional, Tuple

EPS = 1e-8


@torch.no_grad()
def select_spatially_diverse_candidates(
    candidate_mask: torch.Tensor,
    priority_scores: torch.Tensor,
    xyz: torch.Tensor,
    budget: int,
    method: str = 'voxel',
    voxel_size: Optional[float] = None,
    scales: Optional[torch.Tensor] = None,
    max_per_voxel: int = 1,
    radius_scale: float = 1.0,
    suppression_weight: float = 1.0,
    voxel_scale: float = 2.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    从已有候选集合中选择空间多样性子集。

    该函数不修改任何输入张量，不修改 GaussianModel，仅返回新掩码和统计信息。

    Args:
        candidate_mask: (N,) bool, 原始候选掩码
        priority_scores: (N,) float, 优先级分数 (来自 EAS/RFAS 融合)
        xyz: (N, 3) float, 世界空间坐标
        budget: int, 目标选择数量
        method: str, 选择方法 ('voxel' | 'radius_nms' | 'soft_suppression'), 默认 'voxel'
        voxel_size: float or None, 体素边长; None 时自动从 scales 计算
        scales: (N, 3) float or None, per-Gaussian 线性尺度 (用于自动体素尺寸)
        max_per_voxel: int, 每轮每体素最大选择数, 默认 1
        radius_scale: float, NMS 半径系数 (预留)
        suppression_weight: float, soft suppression 权重 (预留)

    Returns:
        spatial_mask: (N,) bool, 空间选择后的掩码 (candidate_mask 的子集)
        stats: dict, 统计信息 (全部为 Python 原生标量)
    """
    t_start = time.time()

    # ---- 输入验证 ----
    N = candidate_mask.numel()
    if candidate_mask.ndim != 1:
        raise ValueError(f"candidate_mask must be 1D, got shape {candidate_mask.shape}")
    if priority_scores.shape[0] != N:
        raise ValueError(
            f"priority_scores shape mismatch: expected ({N},), got {priority_scores.shape}"
        )
    if xyz.shape != (N, 3):
        raise ValueError(
            f"xyz shape must be ({N}, 3), got {xyz.shape}"
        )
    if method not in ('voxel', 'radius_nms', 'soft_suppression'):
        raise ValueError(
            f"Unknown method: '{method}'. Valid options: voxel, radius_nms, soft_suppression"
        )
    if max_per_voxel < 1:
        raise ValueError(
            f"max_per_voxel must be >= 1, got {max_per_voxel}"
        )

    # ---- 筛选候选 (排除 NaN/Inf, 保留零/负分数) ----
    finite_mask = candidate_mask & torch.isfinite(priority_scores)
    cand_indices = finite_mask.nonzero(as_tuple=False).squeeze(-1)  # (K,)
    n_candidates_input = int(candidate_mask.sum().item())
    n_candidates = cand_indices.numel()

    # ---- 边缘情况 ----
    if n_candidates == 0 or budget <= 0:
        runtime_ms = (time.time() - t_start) * 1000.0
        return torch.zeros(N, dtype=torch.bool, device=candidate_mask.device), {
            'candidate_count': n_candidates_input,
            'selected_count': 0,
            'occupied_voxel_count': 0,
            'max_candidates_per_voxel': 0,
            'mean_candidates_per_voxel': 0.0,
            'voxel_size': voxel_size if voxel_size is not None else 0.0,
            'runtime_ms': runtime_ms,
            'jaccard_vs_original': 0.0,
            'replaced_count': 0,
            'method': method,
        }

    # 解析目标选择数量 (有限候选范围内)
    budget = min(budget, n_candidates)

    # ---- 分流到具体方法 ----
    if method == 'voxel':
        spatial_indices, method_stats = _voxel_diversity_select(
            cand_indices=cand_indices,
            priority_scores=priority_scores,
            xyz=xyz,
            budget=budget,
            voxel_size=voxel_size,
            scales=scales,
            max_per_voxel=max_per_voxel,
            voxel_scale=voxel_scale,
        )
    elif method == 'radius_nms':
        raise NotImplementedError(
            "radius_nms method is not implemented yet. Use method='voxel'."
        )
    elif method == 'soft_suppression':
        raise NotImplementedError(
            "soft_suppression method is not implemented yet. Use method='voxel'."
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # ---- 构建输出掩码 ----
    spatial_mask = torch.zeros(N, dtype=torch.bool, device=candidate_mask.device)
    spatial_mask[spatial_indices] = True

    # ---- 计算原 EAS top-budget 集合 (用于统计 Jaccard 等) ----
    # 原始 EAS 按 priority_scores 选 top-budget (仅在 candidate_mask 内, 排除 NaN/Inf)
    work_scores = priority_scores.clone()
    work_scores[~candidate_mask] = float('-inf')
    bad = torch.isnan(work_scores) | torch.isinf(work_scores)
    work_scores[bad] = float('-inf')
    n_valid = (work_scores > float('-inf')).sum().item()
    k_orig = min(budget, max(int(n_valid), 0))
    original_set = set()
    if k_orig > 0:
        _, orig_top = torch.topk(work_scores, k_orig)
        original_set = set(orig_top.tolist())

    spatial_set = set(spatial_indices.tolist())
    intersection = len(original_set & spatial_set)
    union = len(original_set | spatial_set)
    jaccard = intersection / max(union, 1)
    replaced_count = len(original_set - spatial_set)

    runtime_ms = (time.time() - t_start) * 1000.0

    stats = {
        'candidate_count': n_candidates_input,
        'selected_count': int(spatial_mask.sum().item()),
        'occupied_voxel_count': method_stats.get('occupied_voxel_count', 0),
        'max_candidates_per_voxel': method_stats.get('max_candidates_per_voxel', 0),
        'mean_candidates_per_voxel': method_stats.get('mean_candidates_per_voxel', 0.0),
        'voxel_size': method_stats.get('voxel_size', 0.0),
        'runtime_ms': runtime_ms,
        'jaccard_vs_original': jaccard,
        'replaced_count': replaced_count,
        'method': method,
    }

    # ---- 最终断言 ----
    assert spatial_mask.dtype == torch.bool
    assert spatial_mask.shape == candidate_mask.shape
    assert not torch.any(spatial_mask & (~candidate_mask)), \
        "spatial_mask must be a subset of candidate_mask"

    return spatial_mask, stats


# ---------------------------------------------------------------------------
# 体素轮转选择实现
# ---------------------------------------------------------------------------


@torch.no_grad()
def _compute_adaptive_voxel_size(
    scales: torch.Tensor,
    candidate_indices: torch.Tensor,
    voxel_scale: float = 2.0,
) -> float:
    """
    根据候选 Gaussian 的中位尺度计算自适应体素尺寸。

    voxel_size = voxel_scale * median(max(scales[candidates], dim=1).values)

    Args:
        scales: (N, 3) float, 线性空间标准差 (get_scaling 的输出)
        candidate_indices: (K,) long, 候选索引
        voxel_scale: float, 体素尺寸系数

    Returns:
        voxel_size: float, 正体素尺寸
    """
    cand_scales = scales[candidate_indices]  # (K, 3)
    max_scale = torch.max(cand_scales, dim=1).values  # (K,)

    # 排除 NaN/Inf/零
    valid = (max_scale > EPS) & torch.isfinite(max_scale)
    if valid.sum() == 0:
        return 0.01  # 回退默认值

    median_scale = torch.median(max_scale[valid]).item()
    if median_scale <= EPS:
        return 0.01

    return voxel_scale * median_scale


def _voxel_diversity_select(
    cand_indices: torch.Tensor,
    priority_scores: torch.Tensor,
    xyz: torch.Tensor,
    budget: int,
    voxel_size: Optional[float],
    scales: Optional[torch.Tensor],
    max_per_voxel: int = 1,
    voxel_scale: float = 2.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    体素轮转选择核心逻辑。

    1. 计算体素尺寸 (自动或指定)
    2. 对候选坐标进行体素分区
    3. 每体素内按分数降序排列
    4. 轮转选取，直到达到 budget

    Args:
        cand_indices: (K,) long, 候选 Gaussian 的全局索引
        priority_scores: (N,) float, 优先级分数
        xyz: (N, 3) float, 世界坐标
        budget: int, 目标选择数量
        voxel_size: float or None
        scales: (N, 3) float or None
        max_per_voxel: int, 每轮每体素最多选几个

    Returns:
        selected_indices: (B,) long, 选中的全局索引
        stats: dict
    """
    K = cand_indices.numel()
    cand_xyz = xyz[cand_indices]  # (K, 3)
    cand_scores = priority_scores[cand_indices]  # (K,)

    # 1. 确定体素尺寸
    if voxel_size is None:
        if scales is not None:
            voxel_size = _compute_adaptive_voxel_size(scales, cand_indices, voxel_scale=voxel_scale)
        else:
            # 无 scale 信息，回退到场景范围的简单启发
            xyz_range = cand_xyz.max(dim=0).values - cand_xyz.min(dim=0).values
            voxel_size = max(xyz_range.max().item() * 0.02, 0.01)
    voxel_size = max(voxel_size, EPS)

    # 2. 体素索引编码
    xyz_min = cand_xyz.min(dim=0).values  # (3,)
    voxel_ijk = ((cand_xyz - xyz_min) / voxel_size).long()  # (K, 3)

    # 确定网格范围
    v_max = voxel_ijk.max(dim=0).values  # (3,)
    nx = int(v_max[0].item()) + 1
    ny = int(v_max[1].item()) + 1
    # 编码为 1D: v = i + j*nx + k*nx*ny
    voxel_1d = (voxel_ijk[:, 0].long()
                + voxel_ijk[:, 1].long() * nx
                + voxel_ijk[:, 2].long() * nx * ny)  # (K,)

    # 3. 对每个体素按 (score降序, global_index升序) 排序
    # 体素内使用  Python sorted (tuples 键是稳定的且支持确定性 lexicographic 排序)
    # 体素遍历也使用确定性复合键
    unique_voxels, inverse_indices, counts = torch.unique(
        voxel_1d, return_inverse=True, return_counts=True
    )
    n_voxels = unique_voxels.numel()

    # 4. 为每个体素构建有序候选列表
    occupied_voxel_count = 0
    voxel_entries = []  # list of (sorted_global_list, voxel_ijk_tuple, top_score)

    for v_idx in range(n_voxels):
        n_in = int(counts[v_idx].item())
        if n_in == 0:
            continue
        occupied_voxel_count += 1

        in_voxel = (inverse_indices == v_idx)  # (K,)
        v_global = cand_indices[in_voxel]  # 该体素中候选的全局索引
        v_scores = cand_scores[in_voxel]  # 该体素中候选的分数 (原始, 不含 NaN/Inf)

        # 按 (-score, global_index) 确定性排序
        # 使用 Python sorted: tuples 键, 词法排序
        scored_pairs = [
            (float(v_scores[i].item()), int(v_global[i].item()))
            for i in range(n_in)
        ]
        # 降序按 score, 升序按 index (tie-breaker)
        scored_pairs.sort(key=lambda x: (-x[0], x[1]))
        sorted_global_indices = torch.tensor(
            [p[1] for p in scored_pairs], dtype=torch.long, device=cand_indices.device
        )

        # 体素 3D 坐标 (用于确定性体素遍历)
        # 取体素内第一个候选的 ijk 坐标
        first_in_voxel = torch.where(in_voxel)[0][0]
        v_i = int(voxel_ijk[first_in_voxel, 0].item())
        v_j = int(voxel_ijk[first_in_voxel, 1].item())
        v_k = int(voxel_ijk[first_in_voxel, 2].item())

        top_score = scored_pairs[0][0] if scored_pairs else float('-inf')
        voxel_entries.append((sorted_global_indices, (v_i, v_j, v_k), top_score))

    # 5. 确定性体素遍历顺序: (-top_score, voxel_i, voxel_j, voxel_k)
    voxel_entries.sort(key=lambda e: (-e[2], e[1][0], e[1][1], e[1][2]))

    # 6. 轮转选择 (不丢弃零/负分数 — 后处理器负责 clamp)
    selected_global = []
    round_idx = 0

    while len(selected_global) < budget:
        any_selected_this_round = False

        for sorted_indices, _voxel_key, _top in voxel_entries:
            if len(selected_global) >= budget:
                break
            n_available = sorted_indices.numel()
            start = round_idx * max_per_voxel
            end = min(start + max_per_voxel, n_available)
            for pos in range(start, end):
                if len(selected_global) >= budget:
                    break
                selected_global.append(int(sorted_indices[pos].item()))
                any_selected_this_round = True

        if not any_selected_this_round:
            break
        round_idx += 1

    # 7. 构建输出
    device = cand_indices.device
    if len(selected_global) == 0:
        selected_indices = torch.zeros(0, dtype=torch.long, device=device)
    else:
        selected_indices = torch.tensor(selected_global, dtype=torch.long, device=device)

    # 统计体素内候选分布
    counts_list = [int(counts[i].item()) for i in range(n_voxels) if counts[i].item() > 0]
    max_candidates = max(counts_list) if counts_list else 0
    mean_candidates = sum(counts_list) / max(len(counts_list), 1)

    stats = {
        'occupied_voxel_count': occupied_voxel_count,
        'max_candidates_per_voxel': max_candidates,
        'mean_candidates_per_voxel': float(mean_candidates),
        'voxel_size': float(voxel_size),
    }

    return selected_indices, stats
