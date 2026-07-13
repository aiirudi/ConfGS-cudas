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


@torch.no_grad()
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
    体素轮转选择核心逻辑（全向量化版本）。

    1. 计算体素尺寸 (自动或指定)
    2. 对候选坐标进行体素分区
    3. CPU stable sort: 体素内按 (-score, global_index)、体素间按 (-top_score, i, j, k)
    4. 单次向量化 sort 计算 round-robin 选择顺序，取 top-budget

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
    device = cand_indices.device
    cand_xyz = xyz[cand_indices]  # (K, 3)
    cand_scores = priority_scores[cand_indices]  # (K,)

    # 1. 确定体素尺寸
    if voxel_size is None:
        if scales is not None:
            voxel_size = _compute_adaptive_voxel_size(scales, cand_indices, voxel_scale=voxel_scale)
        else:
            xyz_range = cand_xyz.max(dim=0).values - cand_xyz.min(dim=0).values
            voxel_size = max(xyz_range.max().item() * 0.02, 0.01)
    voxel_size = max(voxel_size, EPS)

    # 2. 体素索引编码
    xyz_min = cand_xyz.min(dim=0).values  # (3,)
    voxel_ijk = ((cand_xyz - xyz_min) / voxel_size).long()  # (K, 3)

    v_max = voxel_ijk.max(dim=0).values  # (3,)
    nx = int(v_max[0].item()) + 1
    ny = int(v_max[1].item()) + 1
    voxel_1d = (voxel_ijk[:, 0].long()
                + voxel_ijk[:, 1].long() * nx
                + voxel_ijk[:, 2].long() * nx * ny)  # (K,)

    # 3. CPU 多遍 stable sort: 按 (voxel_1d, -score, global_index) 排序
    #    最后一道 sort 是主键 → 先排体素, 体内按分数降序, tie-breaker 按 global_index 升序
    cand_indices_cpu = cand_indices.cpu()
    cand_scores_cpu = cand_scores.cpu()
    voxel_1d_cpu = voxel_1d.cpu()
    voxel_ijk_cpu = voxel_ijk.cpu()

    idx = torch.arange(K)  # CPU
    idx = idx[torch.sort(cand_indices_cpu[idx], stable=True).indices]    # 第三键: global_index
    idx = idx[torch.sort(-cand_scores_cpu[idx], stable=True).indices]     # 第二键: -score
    idx = idx[torch.sort(voxel_1d_cpu[idx], stable=True).indices]         # 主键: voxel_1d

    idx = idx.to(device)
    sorted_voxel_1d = voxel_1d[idx]       # (K,) 按体素分组
    sorted_global = cand_indices[idx]      # (K,) 全局索引
    sorted_scores = cand_scores[idx]       # (K,) 分数

    # 4. 体素边界与体内 rank
    is_new_voxel = torch.cat([
        torch.tensor([True], device=device),
        sorted_voxel_1d[1:] != sorted_voxel_1d[:-1],
    ])  # (K,)
    voxel_start = torch.where(is_new_voxel)[0]                    # (n_voxels,)
    n_occupied = voxel_start.numel()
    voxel_of_pos = torch.cumsum(is_new_voxel.long(), dim=0) - 1  # (K,): 每个位置属于第几个体素

    # 扩展 voxel_start 到 (K,) 用于计算体内 rank
    intra_rank = torch.arange(K, device=device) - voxel_start[voxel_of_pos]  # (K,)

    # 5. 体素遍历顺序: 按 (-top_score, i, j, k) 对体素排序
    top_scores = sorted_scores[voxel_start]                              # (n_voxels,)
    top_ijk = voxel_ijk[idx[voxel_start]]                                # (n_voxels, 3)

    top_scores_cpu = top_scores.cpu()
    top_ijk_cpu = top_ijk.cpu()
    vox_order = torch.arange(n_occupied)  # CPU
    vox_order = vox_order[torch.sort(top_ijk_cpu[vox_order, 2].float(), stable=True).indices]  # k
    vox_order = vox_order[torch.sort(top_ijk_cpu[vox_order, 1].float(), stable=True).indices]  # j
    vox_order = vox_order[torch.sort(top_ijk_cpu[vox_order, 0].float(), stable=True).indices]  # i
    vox_order = vox_order[torch.sort(-top_scores_cpu[vox_order], stable=True).indices]         # -top_score

    vox_order = vox_order.to(device)
    voxel_rank = torch.empty(n_occupied, dtype=torch.long, device=device)
    voxel_rank[vox_order] = torch.arange(n_occupied, device=device)  # 体素遍历位次

    # 6. Round-robin 选择顺序 (单次 sort, 无 while 循环)
    #    round_num = torch.div(intra_rank, max_per_voxel, rounding_mode='floor')
    #    sel_order = round_num * n_voxels * max_per_voxel + voxel_rank * max_per_voxel + pos_in_round
    round_num = torch.div(intra_rank, max_per_voxel, rounding_mode='floor')
    pos_in_round = intra_rank % max_per_voxel
    sel_order = (round_num * n_occupied * max_per_voxel
                 + voxel_rank[voxel_of_pos] * max_per_voxel
                 + pos_in_round)  # (K,)

    _, sel_pos = torch.sort(sel_order)
    selected_global = sorted_global[sel_pos[:budget]]  # (budget,)

    # 7. 统计
    voxel_sizes = torch.diff(torch.cat([
        voxel_start, torch.tensor([K], device=device)
    ]))  # (n_voxels,)
    max_candidates = int(voxel_sizes.max().item())
    mean_candidates = float(voxel_sizes.float().mean().item())

    stats = {
        'occupied_voxel_count': n_occupied,
        'max_candidates_per_voxel': max_candidates,
        'mean_candidates_per_voxel': mean_candidates,
        'voxel_size': float(voxel_size),
    }

    return selected_global, stats
