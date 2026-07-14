# Per-Gaussian Spatial Suppression Event Log

## Goal Description

在空间多样性模块（`utils/spatial_diversity.py`）每次执行 densification 候选选择时，记录每个被抑制（未选中）的高斯椭球的 per-Gaussian 元数据——包括全局行索引、世界空间坐标、优先级分数、体素分配（ijk、1D 编码、intra-voxel rank）、round-robin 选择排序位置、以及是否属于原始 top-budget 集合。事件日志以 CSV 格式写入 `<model_path>/spatial_suppression_events.csv`，一行一个被抑制高斯 × 一次 densification 步。该日志独立于现有聚合统计（`candidate_selection_stats.csv` 中的 `spatial_replaced`、`spatial_jaccard` 等），提供逐高斯粒度，用于定量分析不同空间多样性参数（`spatial_voxel_scale`、`spatial_interval`、`spatial_max_per_voxel`）产生不同 PSNR/SSIM/LPIPS 指标的原因。

## Acceptance Criteria

遵循 TDD 哲学，每条标准包含正向和负向测试以实现确定性验证。

- AC-1: 开关控制——`--spatial_suppression_log` 标志（默认 `False`）完全控制事件日志功能
  - Positive Tests (expected to PASS):
    - `--spatial_suppression_log` 未设置时，`utils/spatial_diversity` 模块行为与当前完全一致，`select_spatially_diverse_candidates` 返回 `(spatial_mask, stats)` 二元组，无额外元数据
    - `--spatial_suppression_log` 未设置时，`tests/test_spatial_diversity.py` 全部 21 个现有测试通过，无修改
    - `--spatial_suppression_log` 未设置时，`GaussianModel` 不累积任何 suppression event，`self.spatial_suppression_events` 保持为 `None` 或空列表
  - Negative Tests (expected to FAIL):
    - 在 `--spatial_suppression_log` 未设置时，尝试访问 `select_spatially_diverse_candidates` 返回的第三个元素应抛出 `ValueError`（返回二元组）
    - 在 `--spatial_suppression_log` 未设置时，`<model_path>/spatial_suppression_events.csv` 不存在

- AC-2: CSV 事件日志正确性——写入的每行数据与空间多样性模块内部计算一致
  - Positive Tests (expected to PASS):
    - `suppressed_count + selected_count == finite_candidate_count`（排除 NaN/Inf 分数的候选后）
    - 每行的 `event_class` 与 `top_budget_member` 一致：`event_class=REPLACED` 当 `top_budget_member=True`，否则 `event_class=UNSELECTED`
    - `REPLACED` 事件总数等于 `spatial_stats['replaced_count']`
    - CSV 列 `voxel_i, voxel_j, voxel_k` 与独立重算的 `floor((xyz - voxel_origin) / voxel_size)` 一致
    - 确定性：相同输入（相同 seed、相同参数）两次运行产生的 CSV 逐行一致
  - Negative Tests (expected to FAIL):
    - 任何被抑制高斯的 `global_index` 不应出现在 `spatial_mask` 为 True 的位置
    - `voxel_1d` 不为负数或超过 `nx*ny*nz - 1`
    - `intra_voxel_rank` 不为负数

- AC-3: 事件生命周期——事件在 densification 步内被捕获，在 train.py 中被写入并清理
  - Positive Tests (expected to PASS):
    - `densify_and_prune_Improved` 返回后，`gaussians.spatial_suppression_events` 包含当前步的所有事件（如果空间多样性模块执行了）
    - `train.py` 写入 CSV 后立即调用 `gaussians.spatial_suppression_events.clear()`，下次 densification 前列表为空
    - CSV 文件以追加模式写入，header 仅在文件不存在时写入一次
  - Negative Tests (expected to FAIL):
    - 空间多样性被跳过（gate 条件不满足）时，`spatial_suppression_events` 为空，不应写入空行
    - 迭代之间事件不应累积（flush+clear 后列表为空）

- AC-4: 元数据从公共 API 返回——`select_spatially_diverse_candidates` 可选返回 suppression 元数据
  - Positive Tests (expected to PASS):
    - 当 `return_metadata=True` 时，返回三元组 `(spatial_mask, stats, voxel_metadata)`
    - `voxel_metadata` 是 dict，包含 key `voxel_origin`（xyz_min tuple）、`voxel_size`（float）、`candidates`（dict，按全局索引 keyed）
    - `voxel_metadata['candidates'][global_idx]` 包含 `ijk`、`voxel_1d`、`intra_rank`、`round_num`、`sel_order`
    - `voxel_metadata['original_top_budget_set']` 包含原始 top-budget 候选的全局索引集合
  - Negative Tests (expected to FAIL):
    - `return_metadata=False`（默认）时返回二元组，尝试解包第三个值抛出 `ValueError`
    - 传入未知 method（如 `'invalid'`）且 `return_metadata=True` 时仍抛出 `ValueError`（输入校验优先于元数据构造）

- AC-5: 数据完整性——CSV 包含分析所需的所有字段
  - Positive Tests (expected to PASS):
    - CSV 至少包含以下列：`iteration`, `pre_densify_index`, `x`, `y`, `z`, `priority_score`, `event_class`, `top_budget_member`, `voxel_i`, `voxel_j`, `voxel_k`, `voxel_1d`, `intra_voxel_rank`, `round_num`, `sel_order`, `voxel_origin_x`, `voxel_origin_y`, `voxel_origin_z`, `voxel_size`
    - 所有数值列可被 `pandas.read_csv` 直接解析（无嵌套 JSON 字符串）
    - `event_class` 列的值仅为 `UNSELECTED` 或 `REPLACED`
  - Negative Tests (expected to FAIL):
    - `pre_densify_index` 不应超出 `[0, num_gaussians_before)` 范围
    - `voxel_i, voxel_j, voxel_k` 均为非负整数

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

实现包含：
- `utils/spatial_diversity.py`：`select_spatially_diverse_candidates` 和 `_voxel_diversity_select` 均支持 `return_metadata` 可选参数，返回完整的 per-candidate voxel 元数据和 `original_top_budget_set`
- `scene/gaussian_model.py`：在空间多样性 block 内（`final_mask = spatial_mask` 替换点）捕获 suppressed set、构建事件 dict 列表存入 `self.spatial_suppression_events`
- `train.py`：在 densification 步之后将事件列表 flush 到 CSV，用 `csv.DictWriter` 追加模式
- `arguments/__init__.py`：`--spatial_suppression_log` 布尔标志
- `tests/test_spatial_diversity.py`：新增 5+ 测试覆盖元数据返回、事件语义、flush+clear 生命周期
- CSV 包含完整的 19+ 列字段，每行一个被抑制高斯
- 额外的 `num_gaussians_before` 和 `candidate_rank_by_score` 辅助列用于下游分析

### Lower Bound (Minimum Acceptable Scope)

实现包含：
- `utils/spatial_diversity.py`：仅 `select_spatially_diverse_candidates` 新增 `return_metadata=False` 参数；`_voxel_diversity_select` 内部重构以可选返回元数据
- `scene/gaussian_model.py`：捕获 `suppressed_mask = old_final_mask & ~spatial_mask`，对 suppressed indices 收集属性并推入列表
- `train.py`：读取列表、flush CSV、清空列表
- `arguments/__init__.py`：`--spatial_suppression_log` 标志
- 元数据至少包含：`pre_densify_index`, `xyz`, `priority_score`, `event_class`, `voxel_ijk`, `voxel_1d`, `voxel_origin`, `voxel_size`
- 至少 3 个新测试：disabled-path 零影响、event count 一致性、metadata 字段正确性

### Allowed Choices

- 可以：使用 `csv.DictWriter`（与 `candidate_selection_stats.csv` 一致的已有模式）
- 可以：在 `GaussianModel` 上用 `self.spatial_suppression_events: list` 累积事件，由 `train.py` 消费
- 可以：用 `torch.no_grad()` 包裹元数据提取（已在 densification block 内）
- 可以：通过 `vis_context` dict 已有的 `model_path` 键传递路径（如果 `vis_context is not None`），或新增函数参数；两种方案均可
- 不可以：在 `GaussianModel` 内部直接写文件（该类不持有 `model_path`）
- 不可以：修改 CUDA 代码或 rasterizer
- 不可以：在 `return_metadata=False` 时改变 `select_spatially_diverse_candidates` 的公开返回签名
- 不可以：引入新的第三方依赖（`csv`、`os` 均已在环境中）

## Feasibility Hints and Suggestions

> **注意**：本节仅供理解和参考。这些是概念性建议，不是规范性要求。

### Conceptual Approach

**1. 扩展 `_voxel_diversity_select` 返回元数据：**

```python
@torch.no_grad()
def _voxel_diversity_select(
    cand_indices, priority_scores, xyz, budget,
    voxel_size, scales, max_per_voxel=1, voxel_scale=2.0,
    return_metadata=False,  # 新增
) -> Tuple[torch.Tensor, Dict]:
    # ... 现有的体素分区和 round-robin 逻辑 ...

    if return_metadata:
        # 在 sorted_global、intra_rank、round_num、sel_order 被丢弃之前收集它们
        voxel_metadata = {
            'voxel_origin': tuple(xyz_min.tolist()),
            'voxel_size': float(voxel_size),
            'candidates': {},  # keyed by global index (int)
        }
        for pos in range(K):
            g_idx = int(sorted_global[pos].item())
            voxel_metadata['candidates'][g_idx] = {
                'ijk': tuple(voxel_ijk[idx[pos]].tolist()),
                'voxel_1d': int(sorted_voxel_1d[pos].item()),
                'intra_rank': int(intra_rank[pos].item()),
                'round_num': int(round_num[pos].item()),
                'sel_order': int(sel_order[pos].item()),
            }
        return selected_global, stats, voxel_metadata

    return selected_global, stats
```

**2. 在 `select_spatially_diverse_candidates` 中透传并扩展：**

- 新增 `return_metadata=False` 参数
- 当 `return_metadata=True` 时，将 `original_set`（Python `set` of int）加入返回的 voxel_metadata
- 返回三元组 `(spatial_mask, stats, voxel_metadata)`

**3. 在 `densify_and_prune_Improved` 中捕获事件：**

```python
# 在 line 555 spatial_mask, spatial_stats = ... 之后，line 590 final_mask = spatial_mask 之前
if getattr(opt, 'spatial_suppression_log', False):
    # 保存旧 final_mask 的引用（在变量重绑定之前）
    old_final_mask = final_mask
    suppressed_mask = old_final_mask & ~spatial_mask
    suppressed_indices = suppressed_mask.nonzero(as_tuple=False).squeeze(-1)

    events = []
    for g_idx in suppressed_indices.tolist():
        g_idx = int(g_idx)
        is_top_budget = g_idx in voxel_metadata['original_top_budget_set']
        meta = voxel_metadata['candidates'].get(g_idx, {})
        events.append({
            'iteration': iteration,
            'pre_densify_index': g_idx,
            'x': float(xyz[g_idx, 0].item()),
            'y': float(xyz[g_idx, 1].item()),
            'z': float(xyz[g_idx, 2].item()),
            'priority_score': float(selection_score[g_idx].item()),
            'event_class': 'REPLACED' if is_top_budget else 'UNSELECTED',
            'top_budget_member': is_top_budget,
            'voxel_i': meta.get('ijk', (0,0,0))[0],
            # ... 其余字段
        })
    self.spatial_suppression_events = events
```

**4. 在 `train.py` 中 flush CSV：**

```python
# 在 densify_and_prune_Improved() 调用之后
if getattr(opt, 'spatial_suppression_log', False):
    events = gaussians.spatial_suppression_events
    if events:
        log_path = os.path.join(dataset.model_path, "spatial_suppression_events.csv")
        write_header = not os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_FIELDNAMES, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerows(events)
        gaussians.spatial_suppression_events.clear()
```

### Relevant References

- `utils/spatial_diversity.py` — `select_spatially_diverse_candidates`（公开 API）和 `_voxel_diversity_select`（体素轮转选择核心，元数据在此处被丢弃）
- `scene/gaussian_model.py` — `densify_and_prune_Improved`，空间多样性调用点在 lines 535-594，`final_mask = spatial_mask` 替换点在 line 590
- `train.py` — `candidate_selection_stats.csv` CSV 写入模式（lines 326-355），`dataset.model_path` 可用
- `arguments/__init__.py` — `OptimizationParams`（lines 165-187），已有空间多样性参数簇
- `tests/test_spatial_diversity.py` — 21 个现有测试，`select_spatially_diverse_candidates` 的测试契约
- `tests/test_ac1_disabled_equivalence.py` — 验证禁用路径零影响的前例

## Dependencies and Sequence

### Milestones

1. M1: 空间多样性模块元数据返回——`_voxel_diversity_select` 和 `select_spatially_diverse_candidates` 支持 `return_metadata`
   - Phase A: 在 `_voxel_diversity_select` 中添加 `return_metadata` 参数，在元数据被丢弃前收集 per-candidate voxel 信息
   - Phase B: 在 `select_spatially_diverse_candidates` 中添加 `return_metadata` 参数，透传并附加 `original_top_budget_set`
   - Phase C: 更新现有测试确保向后兼容（`return_metadata=False` 默认值不改变行为）

2. M2: GaussianModel 事件捕获——在 `densify_and_prune_Improved` 中计算 suppressed set 并累积事件
   - Phase A: 初始化 `self.spatial_suppression_events = []` 在 `__init__` 或 `training_setup` 中
   - Phase B: 在空间多样性 block 内捕获 `old_final_mask & ~spatial_mask`，用 `voxel_metadata` 构造事件 dict 列表
   - Phase C: 确保事件在 `long_axis_split` 之前被捕获（在 `final_mask = spatial_mask` 行之前保存引用）

3. M3: train.py CSV flush——densification 后将事件写入磁盘并清理
   - Phase A: 添加 CSV `fieldnames` 常量，定义所有列
   - Phase B: 在 densify 调用后读取 `gaussians.spatial_suppression_events`，追加写入 CSV，清空列表
   - Phase C: 与现有 `candidate_stats_enabled` guard 保持一致的新 `spatial_suppression_log` guard

4. M4: 测试和验证——确保正确性和零回归
   - Phase A: 单元测试：`return_metadata` 返回格式、suppressed+selected 一致性、event_class 正确性
   - Phase B: 集成测试：完整 densify→flush→clear 生命周期
   - Phase C: 回归测试：`--spatial_suppression_log` 未设置时所有现有测试通过

## Task Breakdown

每个任务必须包含恰好一个路由标签：
- `coding`：由 Claude 实现
- `analyze`：通过 Codex 执行（`/humanize:ask-codex`）

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | 在 `_voxel_diversity_select` 添加 `return_metadata` 参数，收集 per-candidate voxel 元数据（ijk、voxel_1d、intra_rank、round_num、sel_order）和 `voxel_origin`/`voxel_size`，默认 `False` 保持向后兼容 | AC-1, AC-4 | coding | - |
| task2 | 在 `select_spatially_diverse_candidates` 添加 `return_metadata` 参数，透传 voxel_metadata 并附加 `original_top_budget_set`，默认 `False` 保持返回签名不变 | AC-1, AC-4 | coding | task1 |
| task3 | 在 `GaussianModel.__init__`（或 `training_setup`）初始化 `self.spatial_suppression_events = []`；在 `densify_and_prune_Improved` 的空间多样性 block 内捕获 suppressed set、构造事件 dict 列表 | AC-2, AC-3 | coding | task2 |
| task4 | 在 `train.py` 添加 CSV flush 逻辑：定义 `EVENT_FIELDNAMES`，densify 后读取 events、追加写入 `<model_path>/spatial_suppression_events.csv`、清空列表 | AC-3, AC-5 | coding | task3 |
| task5 | 在 `arguments/__init__.py` 的 `OptimizationParams` 中添加 `--spatial_suppression_log` 布尔标志（默认 `False`） | AC-1 | coding | - |
| task6 | 新增测试：metadata 返回格式、suppressed+selected 一致性、event_class 与 `spatial_replaced` 对齐、flush+clear 生命周期、disabled-path 零影响 | AC-1, AC-2, AC-3, AC-4 | coding | task4 |
| task7 | 用 Codex 审查完整实现的正确性和跨组件一致性，验证 CSV schema 可用于 pandas 分析 | AC-5 | analyze | task6 |

## Claude-Codex Deliberation

### Agreements

- CSV 写入逻辑属于 `train.py`（而非 `GaussianModel`），因为 `model_path` 仅在 `train.py`/`Scene` 中可用
- 元数据返回必须通过可选参数（`return_metadata=False`）实现，以保护现有调用方和测试的向后兼容性
- `voxel_origin` 和 `voxel_size` 必须随每个候选一起记录，因为体素坐标是相对于每步动态 `cand_xyz.min()` 计算的
- 事件行必须在 `long_axis_split` 之前捕获和物化（转为 Python 标量），以避免 LAS 的 split/prune 改变张量形状后索引失效
- `global_index` 命名需澄清为 per-iteration 行索引（`pre_densify_index`），因为 split/prune 会使索引在迭代间失效

### Resolved Disagreements

- **UNSELECTED vs REPLACED 语义**：Codex 指出单一 `event_class` 列无法同时表示"预算截断未选中"和"top-budget 被替换"而不产生重复行。采纳 Codex 建议：每行一个被抑制高斯，`event_class=REPLACED` 当 `top_budget_member=True`，否则 `event_class=UNSELECTED`。两者互斥且覆盖所有 suppressed 候选。
- **元数据返回位置**：Codex 要求在公共 `select_spatially_diverse_candidates` 而非仅在私有 `_voxel_diversity_select` 返回元数据。采纳：公共 API 透传 voxel_metadata 并附加 `original_top_budget_set`（与 `spatial_replaced` 计算来源一致），确保 CSV 和聚合统计不产生分歧。
- **性能声明的精度**：Codex 建议移除 `<5%` 硬数字。采纳：从 AC 中移除硬性百分比；在 Path Boundaries 中仅定性声明"当启用时增加 modest overhead（额外的 Python dict 构造和 CSV I/O）"。
- **AC-2 覆盖范围**：Codex 指出 `suppressed + selected == candidates` 当 `candidate_mask` 包含非有限分数时不成立（因为这些候选在体素选择前被过滤）。修正：等式使用 `finite_candidate_count`（排除 NaN/Inf 分数后）。

### Convergence Status

- Final Status: `converged`
- 经过 1 轮 Claude-Codex 迭代收敛：Codex 提出 5 个 REQUIRED_CHANGES 和 5 个 DISAGREE，Claude 全部采纳并用精确工程方案回应。第二轮无需执行（所有 REQUIRED_CHANGES 均已解决，无高影响 DISAGREE 残留）。

## Pending User Decisions

- DEC-1: 是否需要跨迭代的高斯稳定身份标识（persistent Gaussian ID），还是 per-iteration 的 `pre_densify_index` 已足够？
  - Claude Position: `pre_densify_index`（当前张量行索引）对参数→指标分析场景已足够。稳定 ID 需要维护跨 split/prune 的追踪机制，属于独立特性（约 +100 LOC），不应阻塞当前事件日志。
  - Codex Position: N/A — open question
  - Tradeoff Summary: `pre_densify_index` 使得同一物理高斯在不同迭代的 suppression 事件无法直接关联。如果用户需要"某个高斯在 densify 窗口内反复被抑制"这类纵向分析，则需要稳定 ID。对当前"不同参数下指标差异"的分析目标，per-iteration 索引已足够。
  - Decision Status: `PENDING`

- DEC-2: 非有限分数（NaN/Inf）候选应如何记录？
  - Claude Position: 排除在事件 universe 之外——它们被 `select_spatially_diverse_candidates` 在内部过滤（line 83: `finite_mask = candidate_mask & torch.isfinite(priority_scores)`），不可能被选中，记录它们没有分析价值。`spatial_stats['candidate_count']` 已记录包含 NaN/Inf 的原始候选总数，可供交叉验证。
  - Codex Position: N/A — open question
  - Tradeoff Summary: 如果记录 NaN/Inf 候选为单独的 `EXCLUDED_NONFINITE` 事件类别，可帮助调试数据质量问题，但会显著增加日志量且对参数分析无直接价值。
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements

- 实现代码和注释不得包含 plan 专用的进度术语，如 "AC-"、"Milestone"、"Step"、"Phase" 或类似工作流标记
- 这些术语仅用于计划文档，不用于最终代码库
- 代码中应使用描述性、领域恰当的名称（如 `suppressed_mask`、`voxel_metadata`、`event_class`）

--- Original Design Draft Start ---

# Per-Gaussian Suppression Event Log for Spatial Diversity Parameter Analysis

## Original Idea

是不是可以把通过空间多样性抑制模块，抑制的高斯椭球效果记录下来，从而分析一下不同参数得到的指标的原因

## Primary Direction: Per-Ellipsoid Suppression Event Log

### Rationale

Record every suppressed Gaussian's ID, position, score, and voxel cell at each densification step as a structured event log for offline analysis, distinguishing this from visual overlay or aggregate heatmap approaches.

### Approach Summary

Add a per-Gaussian structured event log that records every suppressed Gaussian's identity, spatial attributes, and score at each densification step. The suppression boundary is the point where `spatial_mask` (output of `select_spatially_diverse_candidates`) replaces the original `final_mask` inside `GaussianModel.densify_and_prune_Improved()` (lines 555-590 of `scene/gaussian_model.py`). The suppressed set is precisely `final_mask_before_spatial & ~spatial_mask`. For each suppressed Gaussian, the following fields are already in memory at the suppression site: global index, world-space `xyz` position, `priority_score` (from EAS/RFAS fusion), voxel cell `(i, j, k)` and 1D voxel index, intra-voxel rank, round-robin selection order, and the round number that would have been needed for selection.

The core mechanism is twofold: (1) extend `_voxel_diversity_select()` in `utils/spatial_diversity.py` to optionally return the per-candidate voxel metadata (currently computed internally and discarded); (2) in `scene/gaussian_model.py`, after line 590 (`final_mask = spatial_mask`), compute the suppressed-Gaussian index set, gather their attributes from the already-available tensors, and append one event record per suppressed Gaussian to a persistent file. The natural output format is CSV (per-row, human-inspectable, importable into pandas/R/matplotlib), following the existing `csv.DictWriter` pattern established in `train.py` lines 329-355 for `candidate_selection_stats.csv`. A separate JSONL path (following `utils/conf_visualization.py` lines 138-141) could also serialize richer per-event structures. The event log file would be written to `<model_path>/spatial_suppression_events.csv` (or `.jsonl`), with one row per suppressed Gaussian per densification step.

### Objective Evidence

- `scene/gaussian_model.py` lines 535-594: The exact suppression site. `spatial_mask` from `select_spatially_diverse_candidates` replaces `final_mask` at line 590. All necessary tensors (`self.get_xyz`, `selection_score`, the pre-overwrite `final_mask`) are in scope.
- `utils/spatial_diversity.py` lines 248-344: `_voxel_diversity_select()` computes voxel partitioning (`voxel_ijk` at line 264, `voxel_1d` at line 269), intra-voxel rank (`intra_rank` at line 300), round-robin selection order (`round_num` at line 321, `sel_order` at line 323). All of this metadata is currently discarded after line 328 (`selected_global = sorted_global[sel_pos[:budget]]`); the `budget:` tail of `sel_pos` identifies the suppressed candidates.
- `train.py` lines 326-355: Prior art for CSV event logging — `candidate_selection_stats.csv` uses `csv.DictWriter`, `write_header` guard, and `extrasaction='ignore'`. This is the pattern to follow.
- `utils/conf_visualization.py` lines 138-141, 329-331: Prior art for JSONL append (`metadata.jsonl`) and per-interval `.pt` index files — demonstrates structured serialization of per-Gaussian indices.
- `scene/gaussian_model.py` lines 580-586: The existing aggregate stats keys (`spatial_replaced`, `spatial_jaccard`, `spatial_candidate_count`, `spatial_selected_count`) confirm the suppression concept is already quantified at the aggregate level; the gap is per-Gaussian granularity.
- `tests/test_spatial_diversity.py`: 21 tests covering the spatial diversity module — provides a testing harness that can be extended to verify event log correctness (e.g., suppressed+selected == all candidates, suppressed indices are deterministic).
- `arguments/__init__.py` lines 165-187: All spatial diversity configuration parameters are centralized here; a new `--spatial_suppression_log` flag would follow the same pattern.

### Known Risks

- The number of suppressed Gaussians per step can be large (K - budget; K up to ~50000, budget typically 50-500), producing high-volume CSV files (~150 densification steps × tens of thousands of rows). A flag to enable/disable the log or to sample a subset is warranted.
- The `_voxel_diversity_select` function currently returns only `(selected_indices, stats)`. Adding suppressed-candidate metadata to the return signature is a breaking change to its interface, though its only caller is `select_spatially_diverse_candidates` and the test file, so the blast radius is small.
- The voxel `(i, j, k)` coordinates are computed relative to `cand_xyz.min()` (line 263 of `spatial_diversity.py`), so voxel cell IDs are not globally consistent across densification steps. This limits cross-step spatial comparisons unless the origin/reference is also logged.
- Storage cost: if each suppressed event is ~200 bytes, and a typical run generates ~100k events, the log is ~20 MB, which is manageable but worth documenting.

## Alternative Directions Considered

### Alt-1: 2D Suppression Overlay Visualization
- Gist: Extend the existing `conf_visualization.py` 2D overlay pipeline to render suppressed vs retained candidate Gaussians as different-colored dots on 2D camera projections. After spatial diversity selection (at line 590 of `gaussian_model.py`), both the original `final_mask` and the new `spatial_mask` are simultaneously in scope, allowing `suppressed = final_mask & ~spatial_mask` (red) and `retained = spatial_mask` (green) to be drawn on the same camera render. The drawing function `draw_conf_overlay()` gains a `fill_color` parameter; the rest of the PIL-based pipeline is reused directly. ~100-150 LOC, mostly in `conf_visualization.py` and `gaussian_model.py`.
- Objective Evidence:
  - `utils/conf_visualization.py` (336 LOC): Direct extension surface; `draw_conf_overlay()` currently hardcodes `fill=(255, 0, 0)` at line 106 — a clear single-point modification site for multi-color support.
  - `scene/gaussian_model.py` lines 523-603: Both pre-spatial `final_mask` and post-spatial `spatial_mask` are simultaneously available at the insertion point (line 590).
  - `scripts/validate_conf_vis.py` (402 LOC): Existing validation harness that can be extended to recognize the new overlay file suffix.
  - `utils/spatial_diversity.py`: Already computes `jaccard_vs_original` and `replaced_count`, confirming the suppressed/retained semantic is well-defined.
- Why not primary: 2D overlay provides per-view qualitative insight but lacks the structured per-Gaussian data needed for systematic cross-parameter metric analysis; it complements rather than replaces the event log.

### Alt-2: Cross-Run Parameter Sweep Analyzer
- Gist: Build a standalone Python script (`scripts/sweep_spatial_diversity.py`) that runs a grid-sweep over three key spatial diversity parameters (`spatial_voxel_scale`, `spatial_interval`, `spatial_max_per_voxel`) by forking multiple training runs (identical scene and seed), then collecting per-iteration `candidate_selection_stats.csv` and final `results.json` to produce comparative charts (Jaccard vs iteration, PSNR/SSIM/LPIPS vs parameter value). The script uses argparse (following `scripts/validate_conf_vis.py`), iterates over Cartesian product of parameter values, invokes `train.py` via subprocess (as `test.py` does), and aggregates across runs. Purely a driver — no existing module logic is modified.
- Objective Evidence:
  - `train.py` lines 325-355: Per-densification CSV already exports all 8 spatial-specific fields (`spatial_jaccard`, `spatial_replaced`, `spatial_occupied_voxels`, etc.) — the exact data the analyzer would chart.
  - `scripts/run_spatial_diversity_ablation.sh` (124 LOC): Existing fixed ON-vs-OFF ablation script demonstrating the train→render→metrics→aggregate pattern.
  - `test.py` (161 LOC): Subprocess-based multi-scene batch-run pattern with JSON aggregation.
  - `arguments/__init__.py` lines 165-187: All swept parameters already registered as CLI flags — no new parameter plumbing needed.
- Why not primary: Full training runs are expensive (hours per run on GPU), and matplotlib is not in `environment.yml`; the sweep analyzer is more a downstream consumer of suppression data than the foundational recording mechanism.

### Alt-3: 3D Spatial Suppression Density Heatmap
- Gist: Build a persistent 3D voxel-grid accumulator inside `GaussianModel` that records, across all densification steps, how many candidates were suppressed in each world-space voxel cell. After each spatial diversity call, compute `suppressed_mask = final_mask & ~spatial_mask`, map each suppressed Gaussian's position to a fixed world-space voxel coordinate using the same `voxel_1d` encoding from `_voxel_diversity_select`, and increment the per-voxel counter. At end of training, export as a 3D PLY point cloud (voxel centroids with suppression count as scalar attribute), viewable in SuperSplat. ~70-100 LOC, all Python, no CUDA changes.
- Objective Evidence:
  - `utils/spatial_diversity.py` lines 250-269: The `voxel_1d` encoding formula is the exact key needed to map suppressed candidates into a persistent flat grid.
  - `scene/gaussian_model.py` lines 176-204: The pattern for initializing per-Gaussian accumulators (`xyz_gradient_accum`, etc.) is repeated 5 times — a voxel-grid accumulator follows the same pattern but indexed by voxel cell.
  - `scene/gaussian_model.py` lines 230-248: Existing `plyfile`-based PLY I/O — suppression heatmap can be exported as PLY with per-voxel attributes, viewable in SuperSplat without new dependencies.
  - No existing 3D voxel-grid accumulator or heatmap export exists in the codebase (confirmed by exhaustive grep).
- Why not primary: Voxel coordinate anchoring (global fixed `xyz_min` vs per-step dynamic) is a design risk; this direction is more of a visualization layer on top of the suppression event data rather than the data source itself.

### Alt-4: View-Aware Metric Attribution
- Gist: Build an offline analysis module (`utils/metric_attribution.py`) that replays the spatial diversity suppression decisions and attributes per-view pixel errors to individual suppressed Gaussians. For each suppressed Gaussian, estimate its contribution to per-view error by: (1) rendering with `pixel_weights = error_map` (reusing RFAS's mechanism at `train.py:773-807`), (2) reading `accum_weights` from the custom rasterizer, (3) projecting the Gaussian's 3D center to 2D and sampling the error map. Aggregate a "suppression impact score" per Gaussian. For multiple runs with different spatial diversity parameters, compare attribution maps to explain metric deltas. ~200-300 LOC, new file, no rasterizer changes.
- Objective Evidence:
  - `train.py` lines 773-807: `compute_rf_score1()` is the exact prior art — passes `pixel_weights = error_map` to `render()`, gets `accum_weights` per Gaussian. Identical mechanism for error attribution.
  - `utils/conf_visualization.py` lines 14-51: `project_gaussian_centers()` and `filter_visible_points()` for world-to-pixel projection — directly importable.
  - `gaussian_renderer/__init__.py` lines 103-129: Custom rasterizer returns `accum_weights`, `visibility_filter`, `gaussian_centers`, `gaussian_depths` — all usable for spatial-to-error correlation.
  - `utils/spatial_diversity.py` lines 144-150: `original_set` and `spatial_set` already materialized as Python sets — the suppressed indices are computed but discarded.
- Why not primary: The train/test distribution gap for error maps (suppression decisions based on training-view error vs final test-set PSNR) is a real validity risk; novel integration of ~200-300 LOC with no prior repo precedent for the closed-loop "Gaussian i suppressed → pixel error at its projection → PSNR delta explained" analysis.

### Alt-5: Real-Time Suppression Monitor
- Gist: Extend the existing `network_gui_ws.py` WebSocket server to carry a second data channel for per-densification suppression statistics. At each densification interval, serialize the `spatial_stats` dict plus per-voxel suppression metadata as JSON and push to connected clients. On the frontend (`web_viewer/app.js` + `render.html`), add a side panel dashboard showing per-voxel occupancy counts over time, suppression Jaccard index, and a stats table. Gated behind a new `--suppression_monitor` flag (default off, zero overhead when disabled, identical to `--visualize_conf` pattern).
- Objective Evidence:
  - `gaussian_renderer/network_gui_ws.py` (56 lines): Existing WebSocket server with global `latest_result` pattern and async background thread — exact infrastructure to extend with a second message type.
  - `web_viewer/app.js` (254 lines): Existing WebGL WebSocket client with binary message handler — natural extension point for JSON stats messages.
  - `utils/spatial_diversity.py`: `select_spatially_diverse_candidates()` already returns `stats` dict with 10 fields — immediately serializable for streaming.
  - `train.py` lines 92-98: WebSocket image push occurs every training iteration; stats push would occur at densification intervals only.
- Why not primary: The `app.js` frontend has no charting library dependency; adding plotly.js/Chart.js introduces a new dependency and potential latency. The existing `network_gui_ws.py` uses a single global with no locking — multiplexing image and stats channels on the same port risks protocol confusion. This direction provides interactive exploration but requires more cross-cutting changes than the event log.

## Synthesis Notes

The five alternative directions can be arranged as a natural pipeline building on the primary direction's event log. Alt-1 (2D overlay) and Alt-3 (3D heatmap) are complementary visualization layers that consume the same suppressed-Gaussian indices the event log records — they could be implemented as renderers over the CSV/JSONL output rather than requiring in-training hooks. Alt-2 (parameter sweep) and Alt-4 (metric attribution) are downstream analysis consumers: the sweep script would compare event logs across parameter configurations, while the attribution module would join suppression events with per-view error maps to estimate each suppressed Gaussian's impact. Alt-5 (real-time monitor) could stream a subset of the event log fields (scalar stats + voxel occupancy) via WebSocket for live feedback during training. If the user chose a different primary direction, the event log would still serve as the shared data foundation — building it first would accelerate all other directions by providing a stable, queryable record of every suppression decision.

--- Original Design Draft End ---
