"""
致密化候选点选择策略模块。

将 densify_and_prune_Improved 中硬编码的 AND 掩码逻辑抽象为可切换的策略。
支持 7 种策略 + 2 种预算模式。
"""

import torch
from typing import Dict, Tuple, Optional

EPS = 1e-8

# 策略枚举
VALID_STRATEGIES = {
    'and', 'or', 'abs_only', 'conf_only',
    'weighted_score', 'soft_fusion', 'rfas_rank',
}
VALID_NORMALIZATIONS = {'minmax', 'zscore', 'percentile'}
VALID_SCORE_SELECTIONS = {'threshold', 'topk'}
VALID_SOFT_FUSION_TYPES = {'weighted', 'soft_and', 'soft_or'}
VALID_BUDGET_MODES = {'native', 'fixed'}
VALID_BUDGET_REFERENCES = {'fixed_number', 'fixed_ratio', 'match_and'}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _safe_tensor(t: torch.Tensor) -> torch.Tensor:
    """替换 NaN 和 Inf 为零，返回同设备同dtype张量。"""
    t = t.clone()
    t[torch.isnan(t)] = 0.0
    t[torch.isinf(t)] = 0.0
    return t


def _count(t: torch.Tensor) -> int:
    """安全计数（bool 或数值 tensor）。"""
    if t.dtype == torch.bool:
        return int(t.sum().item())
    return int((t > 0).sum().item())


def _finite_valid_mask(*tensors: torch.Tensor) -> torch.Tensor:
    """
    构造统一有限有效掩码：所有输入张量都 finite 且不为 NaN。
    返回形状与第一个输入相同的 bool Tensor。
    """
    ref = tensors[0]
    if ref.numel() == 0:
        return torch.zeros_like(ref, dtype=torch.bool)
    mask = torch.ones(ref.shape, dtype=torch.bool, device=ref.device)
    for t in tensors:
        t_flat = t.reshape(ref.shape[0], -1)
        for d in range(t_flat.shape[1]):
            col = t_flat[:, d]
            mask = mask & (~torch.isnan(col)) & (~torch.isinf(col))
    return mask


# 布尔掩码策略集合
_BOOLEAN_STRATEGIES = {'and', 'or', 'abs_only', 'conf_only'}


def normalize_scores(
    scores: torch.Tensor,
    method: str = 'percentile',
    eps: float = EPS,
) -> torch.Tensor:
    """
    将连续分数归一化到 [0, 1]，分数越大越好。

    Args:
        scores: (N,) 连续分数
        method: 'minmax' | 'zscore' | 'percentile'
        eps: 数值稳定小量

    Returns:
        normalized: (N,) 归一化分数 [0, 1]
    """
    scores = _safe_tensor(scores)
    N = scores.numel()
    if N == 0:
        return scores

    if method == 'minmax':
        s_min = scores.min()
        s_max = scores.max()
        denom = s_max - s_min
        if denom < eps:
            return torch.zeros_like(scores)
        return (scores - s_min) / denom

    elif method == 'zscore':
        s_mean = scores.mean()
        s_std = scores.std()
        if s_std < eps:
            return torch.zeros_like(scores)
        z = (scores - s_mean) / s_std
        # 用 sigmoid 映射到 [0,1]
        return torch.sigmoid(z)

    elif method == 'percentile':
        # 基于排名的归一化，对异常值稳定
        order = scores.argsort()
        ranks = torch.zeros(N, device=scores.device, dtype=torch.float32)
        ranks[order] = torch.arange(N, device=scores.device, dtype=torch.float32)
        return ranks / max(N - 1, 1)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_soft_mask(
    scores: torch.Tensor,
    threshold: float,
    temperature: float,
    sign: int,
    eps: float = EPS,
) -> torch.Tensor:
    """
    sigmoid 软掩码：P = sigma(sign * (score - threshold) / temperature)

    Args:
        scores: (N,) 连续分数
        threshold: 阈值 tau
        temperature: 温度 T (>0)
        sign: +1 表示 score >= threshold 为佳, -1 表示 score <= threshold 为佳
        eps: 防止除零

    Returns:
        P: (N,) 软掩码概率 [0, 1]
    """
    if temperature < eps:
        temperature = eps
    return torch.sigmoid(sign * (scores - threshold) / temperature)


def select_by_threshold(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """按阈值选择候选点，返回 bool mask。"""
    return scores >= threshold


def select_by_topk(
    scores: torch.Tensor,
    k: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    按分数 top-k 选择候选点。

    Args:
        scores: (N,) 连续分数
        k: 选择数量。若 k <= 0 返回全 False；若 k >= N 返回全 True。
        valid_mask: (N,) bool，可选的有效点掩码（仅在 valid 点内选 top-k）

    Returns:
        mask: (N,) bool
    """
    N = scores.numel()
    if k <= 0 or N == 0:
        return torch.zeros(N, dtype=torch.bool, device=scores.device)

    # 排除 NaN 和 Inf（设为一inf，不会被 topk 选中）
    work_scores = scores.clone()
    bad = torch.isnan(work_scores) | torch.isinf(work_scores)
    work_scores[bad] = float('-inf')

    if valid_mask is not None:
        work_scores[~valid_mask] = float('-inf')

    n_valid = (work_scores > float('-inf')).sum().item()
    k = min(k, max(int(n_valid), 0))
    if k <= 0:
        return torch.zeros(N, dtype=torch.bool, device=scores.device)

    _, top_indices = torch.topk(work_scores, k)
    mask = torch.zeros(N, dtype=torch.bool, device=scores.device)
    mask[top_indices] = True
    return mask


def resolve_topk(k_absolute: int, k_ratio: float, N: int) -> int:
    """
    解析 top-k 参数。

    优先级: k_absolute > 0 时使用 k_absolute;
           否则使用 int(k_ratio * N);
           否则返回 0。
    """
    if k_absolute > 0:
        return min(k_absolute, N)
    if k_ratio > 0:
        return max(1, int(k_ratio * N))
    return 0


# ---------------------------------------------------------------------------
# 统计信息收集
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_candidate_statistics(
    strategy: str,
    abs_score: torch.Tensor,
    conf_score: torch.Tensor,
    rfas_score: torch.Tensor,
    abs_mask: torch.Tensor,
    conf_mask: torch.Tensor,
    final_mask: torch.Tensor,
    selection_score: torch.Tensor,
    target_budget: int,
    actual_budget: int,
) -> Dict:
    """
    收集候选选择的统计信息。

    Returns:
        stats dict，所有值为 Python 标量。
    """
    abs_score_safe = _safe_tensor(abs_score).squeeze()
    conf_score_safe = _safe_tensor(conf_score)
    rfas_safe = _safe_tensor(rfas_score)

    and_mask = abs_mask & conf_mask
    or_mask = abs_mask | conf_mask

    n_valid = abs_mask.numel()
    n_abs = _count(abs_mask)
    n_conf = _count(conf_mask)
    n_and = _count(and_mask)
    n_or = _count(or_mask)
    n_final = _count(final_mask)

    # IoU 和 overlap
    union = _count(or_mask)
    iou = n_and / max(union, 1)
    overlap_abs = n_and / max(n_abs, 1)
    overlap_conf = n_and / max(n_conf, 1)

    def _stats(t: torch.Tensor):
        """计算均值、标准差、中位数、min、max。"""
        t = t[torch.isfinite(t)]  # 排除 NaN 和 Inf
        if t.numel() == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return (
            float(t.mean().item()),
            float(t.std().item()),
            float(t.median().item()),
            float(t.min().item()),
            float(t.max().item()),
        )

    abs_mean, abs_std, abs_med, abs_min, abs_max = _stats(abs_score_safe)
    conf_mean, conf_std, conf_med, conf_min, conf_max = _stats(conf_score_safe)
    rfas_mean, rfas_std, rfas_med, rfas_min, rfas_max = _stats(rfas_safe)
    sel_mean, sel_std, sel_med, sel_min, sel_max = _stats(selection_score)

    n_nan = int(torch.isnan(abs_score).sum().item() +
                torch.isnan(conf_score).sum().item() +
                torch.isnan(rfas_score).sum().item())
    n_inf = int(torch.isinf(abs_score).sum().item() +
                torch.isinf(conf_score).sum().item() +
                torch.isinf(rfas_score).sum().item())

    return {
        'strategy': strategy,
        'n_valid': n_valid,
        'n_abs_candidates': n_abs,
        'n_conf_candidates': n_conf,
        'n_and_candidates': n_and,
        'n_or_candidates': n_or,
        'n_final_candidates': n_final,
        'candidate_ratio': n_final / max(n_valid, 1),
        'target_budget': target_budget,
        'actual_budget': actual_budget,
        'abs_mean': abs_mean, 'abs_std': abs_std, 'abs_median': abs_med,
        'abs_min': abs_min, 'abs_max': abs_max,
        'conf_mean': conf_mean, 'conf_std': conf_std, 'conf_median': conf_med,
        'conf_min': conf_min, 'conf_max': conf_max,
        'rfas_mean': rfas_mean, 'rfas_std': rfas_std, 'rfas_median': rfas_med,
        'rfas_min': rfas_min, 'rfas_max': rfas_max,
        'sel_score_mean': sel_mean, 'sel_score_std': sel_std,
        'sel_score_median': sel_med, 'sel_score_min': sel_min, 'sel_score_max': sel_max,
        'iou_abs_conf': iou,
        'overlap_abs': overlap_abs,
        'overlap_conf': overlap_conf,
        'n_nan': n_nan,
        'n_inf': n_inf,
    }


# ---------------------------------------------------------------------------
# 固定预算逻辑
# ---------------------------------------------------------------------------

def apply_fixed_budget(
    mask: torch.Tensor,
    scores: torch.Tensor,
    target_budget: int,
    budget_reference: str,
    config,
    strategy: str = 'and',
) -> Tuple[torch.Tensor, int, int]:
    """
    将候选掩码约束到固定预算。

    布尔掩码策略（and/or/abs_only/conf_only）：
      仅在满足掩码条件的点内按 scores 排序选 top-k；
      若满足掩码的点数不足目标预算，决不补入掩码外的点。
    连续分数策略（weighted_score/soft_fusion/rfas_rank）：
      掩码点数不足时从所有 finite-valid 的点中按 scores 补足。

    Args:
        mask: (N,) bool，原始候选掩码
        scores: (N,) 连续排序分数（已 sanitized: 无 NaN/Inf）
        target_budget: 已解析的目标候选数量
        budget_reference: 'fixed_number' | 'fixed_ratio' | 'match_and'
        config: OptimizationParams
        strategy: 当前策略标识

    Returns:
        adjusted_mask: (N,) bool
        actual_budget: 实际选中的候选数量
        resolved_target: 解析后的目标预算（用于日志）
    """
    N = mask.numel()

    if budget_reference == 'fixed_number':
        resolved = getattr(config, 'candidate_fixed_budget', 1000)
    elif budget_reference == 'fixed_ratio':
        ratio = getattr(config, 'candidate_fixed_ratio', 0.05)
        resolved = max(1, int(ratio * N))
    elif budget_reference == 'match_and':
        resolved = target_budget
    else:
        resolved = target_budget

    k = min(resolved, N)
    is_boolean = strategy in _BOOLEAN_STRATEGIES
    n_in_mask = _count(mask)

    if k == 0:
        return torch.zeros(N, dtype=torch.bool, device=mask.device), 0, resolved

    if k <= n_in_mask:
        # 掩码内点数足够：在掩码内选 top-k
        work_scores = scores.clone()
        work_scores[~mask] = float('-inf')
        _, top_indices = torch.topk(work_scores, k)
        adjusted = torch.zeros(N, dtype=torch.bool, device=mask.device)
        adjusted[top_indices] = True
        actual_budget = k
    elif is_boolean:
        # 布尔策略：掩码不足绝不补入，返回掩码内的全部点
        adjusted = mask.clone()
        actual_budget = n_in_mask
    else:
        # 连续策略：从所有 finite-valid 的点中补足（优先保留掩码内点）
        work_scores = scores.clone()
        # 优先保留掩码内的点（加分使其在 topk 中排前）
        if n_in_mask > 0:
            offset = work_scores.max().abs() + 1.0
            work_scores[mask] = work_scores[mask] + offset
        n_valid = _count(work_scores > float('-inf')) if scores.numel() > 0 else 0
        k = min(k, max(n_valid, 0))
        if k <= 0:
            return mask.clone(), n_in_mask, resolved
        _, top_indices = torch.topk(work_scores, k)
        adjusted = torch.zeros(N, dtype=torch.bool, device=mask.device)
        adjusted[top_indices] = True
        actual_budget = k

    return adjusted, actual_budget, resolved


# ---------------------------------------------------------------------------
# 策略实现
# ---------------------------------------------------------------------------

def _strategy_and(
    abs_mask, conf_mask, abs_score, conf_score, rfas_score, config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """AND 策略：abs_mask & conf_mask。"""
    mask = abs_mask & conf_mask
    # selection_score 保持为 rfas_score（用于 multinomial）
    return mask, rfas_score.clone()


def _strategy_or(
    abs_mask, conf_mask, abs_score, conf_score, rfas_score, config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """OR 策略：abs_mask | conf_mask。"""
    mask = abs_mask | conf_mask
    return mask, rfas_score.clone()


def _strategy_abs_only(
    abs_mask, conf_mask, abs_score, conf_score, rfas_score, config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """仅 abs-grad 掩码。"""
    return abs_mask.clone(), rfas_score.clone()


def _strategy_conf_only(
    abs_mask, conf_mask, abs_score, conf_score, rfas_score, config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """仅 Conf 掩码（已包含 has_enough_views）。"""
    return conf_mask.clone(), rfas_score.clone()


def _strategy_weighted_score(
    abs_mask, conf_mask, abs_score, conf_score, rfas_score, config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加权评分融合。

    1. 对 abs_score 和 conf_score 进行方向统一归一化
    2. S = alpha * norm_abs + (1-alpha) * norm_conf
    3. 按 threshold 或 topk 选择最终掩码
    """
    alpha = getattr(config, 'candidate_weight_alpha', 0.5)
    norm_method = getattr(config, 'candidate_score_normalization', 'percentile')
    selection_mode = getattr(config, 'candidate_score_selection', 'topk')
    threshold = getattr(config, 'candidate_score_threshold', 0.5)
    topk_abs = getattr(config, 'candidate_topk', 0)
    topk_ratio = getattr(config, 'candidate_topk_ratio', 0.0)

    # 归一化（abs_score 和 conf_score 都是越大越好，无需翻转）
    norm_abs = normalize_scores(abs_score.squeeze(), method=norm_method)
    norm_conf = normalize_scores(conf_score, method=norm_method)

    # 加权融合
    fused = alpha * norm_abs + (1.0 - alpha) * norm_conf

    # 选择
    if selection_mode == 'threshold':
        mask = select_by_threshold(fused, threshold)
    elif selection_mode == 'topk':
        N = fused.numel()
        k = resolve_topk(topk_abs, topk_ratio, N)
        mask = select_by_topk(fused, k)
    else:
        raise ValueError(f"Unknown score selection: {selection_mode}")

    return mask, fused


def _strategy_soft_fusion(
    abs_mask, conf_mask, abs_score, conf_score, rfas_score, config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    软掩码融合。

    1. P_abs = sigmoid(s_abs * (abs_score - tau_g) / T_g)
    2. P_conf = sigmoid(s_conf * (conf_score - tau_c) / T_c)
    3. 融合: weighted / soft_and / soft_or
    4. 按 threshold 或 topk 选择
    """
    fusion_type = getattr(config, 'soft_fusion_type', 'weighted')
    T_abs = getattr(config, 'soft_abs_temperature', 1.0)
    T_conf = getattr(config, 'soft_conf_temperature', 1.0)
    alpha = getattr(config, 'candidate_weight_alpha', 0.5)
    selection_mode = getattr(config, 'candidate_score_selection', 'topk')
    soft_threshold = getattr(config, 'soft_selection_threshold', 0.5)
    topk_abs = getattr(config, 'candidate_topk', 0)
    topk_ratio = getattr(config, 'candidate_topk_ratio', 0.0)

    # abs-grad 阈值: densify_grad_threshold (越大越好 → sign=+1)
    tau_g = getattr(config, 'densify_grad_threshold', 0.0003)
    # Conf 阈值: conf_thr (越大越好 → sign=+1)
    tau_c = getattr(config, 'conf_thr', 0.8)

    # 软掩码
    P_abs = compute_soft_mask(abs_score.squeeze(), threshold=tau_g,
                              temperature=T_abs, sign=+1)
    P_conf = compute_soft_mask(conf_score, threshold=tau_c,
                               temperature=T_conf, sign=+1)

    # 融合
    if fusion_type == 'weighted':
        P = alpha * P_abs + (1.0 - alpha) * P_conf
    elif fusion_type == 'soft_and':
        P = P_abs * P_conf
    elif fusion_type == 'soft_or':
        P = 1.0 - (1.0 - P_abs) * (1.0 - P_conf)
    else:
        raise ValueError(f"Unknown soft fusion type: {fusion_type}")

    # 选择
    if selection_mode == 'threshold':
        mask = select_by_threshold(P, soft_threshold)
    elif selection_mode == 'topk':
        N = P.numel()
        k = resolve_topk(topk_abs, topk_ratio, N)
        mask = select_by_topk(P, k)
    else:
        raise ValueError(f"Unknown score selection: {selection_mode}")

    return mask, P


def _strategy_rfas_rank(
    abs_mask, conf_mask, abs_score, conf_score, rfas_score, config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RFAS-only ranking。

    仅按 RFAS 分数排序，选择 top-k。排除 NaN/Inf。
    """
    topk_abs = getattr(config, 'candidate_rfas_topk', 0)
    topk_ratio = getattr(config, 'candidate_rfas_topk_ratio', 0.05)

    rfas = _safe_tensor(rfas_score)
    N = rfas.numel()
    k = resolve_topk(topk_abs, topk_ratio, N)

    # 构造有效掩码：排除 NaN/Inf 且 RFAS > 0
    valid = (rfas > 0) & (~torch.isnan(rfas_score)) & (~torch.isinf(rfas_score))

    mask = select_by_topk(rfas, k, valid_mask=valid)
    return mask, rfas


# 策略分发表
STRATEGY_DISPATCH = {
    'and': _strategy_and,
    'or': _strategy_or,
    'abs_only': _strategy_abs_only,
    'conf_only': _strategy_conf_only,
    'weighted_score': _strategy_weighted_score,
    'soft_fusion': _strategy_soft_fusion,
    'rfas_rank': _strategy_rfas_rank,
}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def select_densification_candidates(
    abs_score: torch.Tensor,
    conf_score: torch.Tensor,
    rfas_score: torch.Tensor,
    abs_mask: torch.Tensor,
    conf_mask: torch.Tensor,
    strategy: str = 'and',
    config=None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    统一候选点选择入口。

    Args:
        abs_score:   (N, 1) — grad_vars = xyz_gradient_accum / denom
        conf_score:  (N,)   — conf = 1 - |vec_sum| / mag_sum
        rfas_score:  (N,)   — 原始 RFAS 分数
        abs_mask:    (N,) bool — abs_score >= densify_grad_threshold
        conf_mask:   (N,) bool — (conf >= conf_thr) & has_enough_views
        strategy:    'and'|'or'|'abs_only'|'conf_only'|'weighted_score'|'soft_fusion'|'rfas_rank'
        config:      OptimizationParams 或类似命名空间

    Returns:
        final_mask:      (N,) bool — 最终候选掩码
        selection_score: (N,) float32 — 用于 multinomial 采样的分数
        stats:           dict — 统计信息
    """
    if strategy not in STRATEGY_DISPATCH:
        raise ValueError(
            f"Unknown candidate selection strategy: '{strategy}'. "
            f"Valid options: {sorted(STRATEGY_DISPATCH.keys())}"
        )

    # 安全检查
    device = abs_mask.device
    for name, t in [('abs_score', abs_score), ('conf_score', conf_score),
                     ('rfas_score', rfas_score), ('abs_mask', abs_mask),
                     ('conf_mask', conf_mask)]:
        if t.device != device:
            raise RuntimeError(
                f"Device mismatch: '{name}' is on {t.device}, expected {device}"
            )

    strategy_fn = STRATEGY_DISPATCH[strategy]

    # 执行策略
    final_mask, selection_score = strategy_fn(
        abs_mask=abs_mask,
        conf_mask=conf_mask,
        abs_score=abs_score,
        conf_score=conf_score,
        rfas_score=rfas_score,
        config=config,
    )

    # 确保 final_mask 是 bool
    if final_mask.dtype != torch.bool:
        final_mask = final_mask.bool()

    # 确保 selection_score 在同一设备
    if selection_score.device != device:
        selection_score = selection_score.to(device)

    # 构造 finite-valid mask，统一从 final_mask 和 selection_score 中排除 NaN/Inf
    finite_mask = _finite_valid_mask(selection_score, abs_score.squeeze(), conf_score, rfas_score)
    final_mask = final_mask & finite_mask
    selection_score[~finite_mask] = float('-inf')

    # 应用固定预算（如果配置了）
    budget_mode = getattr(config, 'candidate_budget_mode', 'native')
    budget_ref = getattr(config, 'candidate_budget_reference', 'match_and')
    actual_budget = int(final_mask.sum().item())

    if budget_mode == 'fixed':
        # 确定目标预算
        if budget_ref == 'match_and':
            # match_and: 目标预算 = AND 掩码的候选数
            and_mask, _and_scores = _strategy_and(
                abs_mask=abs_mask, conf_mask=conf_mask,
                abs_score=abs_score, conf_score=conf_score,
                rfas_score=rfas_score, config=config,
            )
            # AND 基线也要排除 NaN/Inf
            and_mask = and_mask & finite_mask
            target_budget = int(and_mask.sum().item())
            resolved_target = target_budget
        elif budget_ref == 'fixed_number':
            resolved_target = getattr(config, 'candidate_fixed_budget', 1000)
            target_budget = resolved_target
        elif budget_ref == 'fixed_ratio':
            ratio = getattr(config, 'candidate_fixed_ratio', 0.05)
            resolved_target = max(1, int(ratio * final_mask.numel()))
            target_budget = resolved_target
        else:
            target_budget = actual_budget
            resolved_target = target_budget

        final_mask, actual_budget, __resolved = apply_fixed_budget(
            final_mask, selection_score,
            target_budget=target_budget,
            budget_reference=budget_ref,
            config=config,
            strategy=strategy,
        )
        resolved_target = __resolved
    else:
        target_budget = actual_budget
        resolved_target = target_budget

    # 收集统计信息（no_grad 避免污染计算图）
    stats = compute_candidate_statistics(
        strategy=strategy,
        abs_score=abs_score,
        conf_score=conf_score,
        rfas_score=rfas_score,
        abs_mask=abs_mask,
        conf_mask=conf_mask,
        final_mask=final_mask,
        selection_score=selection_score,
        target_budget=resolved_target,
        actual_budget=actual_budget,
    )

    return final_mask, selection_score, stats
