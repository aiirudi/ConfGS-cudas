# 空间多样性候选选择器：致密化预算空间重分配方案

## Goal Description

在 RFGS 项目的 conf-opt 分支中，新增一个独立的、默认关闭的"空间多样性候选选择器"后处理模块。该模块在 EAS/RFAS 评分和 `select_densification_candidates()` 生成候选掩码之后、`long_axis_split()` 执行分裂之前运行。模块读取已有候选集合（`base_mask`）、优先级分数（`selection_score`）和 Gaussian 3D 世界坐标（`self.get_xyz`），通过体素网格分区和轮转选择（round-robin selection）抑制空间上高度重复的候选，将致密化预算重新分配给更多样化的空间区域。核心约束：不修改任何现有 Conf、abs-grad、EAS、clone、split、prune、loss、backward、optimizer、CUDA Rasterizer 的实现。关闭模块时（默认状态），训练结果与当前 conf-opt 分支逐比特完全一致。

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: **默认关闭行为等价性** — 当 `enable_spatial_diversity=False` 时，代码执行路径与当前 conf-opt 分支完全相同
  - Positive Tests (expected to PASS):
    - 设置 `--enable_spatial_diversity=False`（或不传此参数）启动训练，`densify_and_prune_Improved` 中不导入 `spatial_diversity` 模块，不调用任何空间选择函数，不读取 `self.get_xyz`，不读取 `self.get_scaling`
    - 关闭状态下 `final_mask` 和 `selection_score` 直接传递给 `long_axis_split()`，中间无任何新变量赋值
    - 关闭状态下运行时 GPU 显存和训练时间与基线无统计显著差异
  - Negative Tests (expected to FAIL):
    - 关闭状态下代码路径不经过任何 `if enable_spatial_diversity:` 分支内的 import 或函数调用
    - 关闭状态下 `candidate_stats` 字典不包含任何 `spatial_*` 前缀的键

- AC-2: **空间选择是原始候选掩码的子集** — `spatial_mask` 中的任何 True 位置在 `base_mask` 中也必须为 True
  - Positive Tests:
    - 构造已知 `base_mask`（部分 True）、已知 `xyz` 和 `scores`，调用空间选择器后，验证 `(spatial_mask & ~base_mask).sum() == 0`
    - 边界情况：`base_mask` 全 False 时，`spatial_mask` 全 False
  - Negative Tests:
    - 断言 `assert not torch.any(spatial_mask & (~base_mask))` 在单元测试中通过

- AC-3: **预算保持（Budget Preservation）** — 空间选择器选中的候选数量等于 `min(budget, base_mask.sum())`
  - Positive Tests:
    - 候选数 > 预算时，`spatial_mask.sum() == budget`
    - 候选数 < 预算时，`spatial_mask.sum() == base_mask.sum()`
    - 预算为 0 时，`spatial_mask.sum() == 0`
  - Negative Tests:
    - 空间选择器不会因候选数不足而报错
    - 不会返回比 `base_mask` 更多 True 的 mask

- AC-4: **选择分数有效性** — 空间选中的候选必须具有有限的、严格正的 `selection_score`，以满足 `torch.multinomial` 的要求
  - Positive Tests:
    - 对空间选中的候选，验证 `selection_score[spatial_mask] > 0` 全部成立
    - 对空间选中的候选，验证 `torch.isfinite(selection_score[spatial_mask]).all()` 成立
  - Negative Tests:
    - NaN/Inf score 的候选绝不被选中
    - `selection_score <= 0` 的候选绝不被选中（在空间路径中单独处理：clamp 到正小量）

- AC-5: **确定性可复现** — 相同输入（相同 xyz、scores、mask、budget、seed）产生相同输出
  - Positive Tests:
    - 固定随机种子，两次调用返回完全相同的 `spatial_mask`
    - Score 相同时使用稳定排序（Gaussian 索引作为 tie-breaker）
  - Negative Tests:
    - 不依赖 `torch.multinomial` 或任何非确定性操作进行体素选择
    - 关闭模块时不调用任何额外随机函数

- AC-6: **体素尺寸自适应** — 当 `spatial_voxel_size='auto'` 时，体素尺寸根据候选 Gaussian 的尺度自动计算
  - Positive Tests:
    - `voxel_size = spatial_voxel_scale * median(max(get_scaling, dim=1).values)` 其中 `get_scaling` 已通过 `torch.exp` 激活
    - 候选数为 0 时，`voxel_size` 回退到合理的默认值（如 0.01）
  - Negative Tests:
    - 不因 `median_scale == 0` 或 `NaN` 而除零报错
    - 不使用未激活的 `self._scaling` 内部参数

- AC-7: **现有测试不被破坏** — `tests/test_candidate_selector.py` 中的所有 18 个现有测试继续通过
  - Positive Tests:
    - `python -m pytest tests/test_candidate_selector.py -v` 返回全部 PASS
  - Negative Tests:
    - 新增的空间模块导入不会在 `test_candidate_selector.py` 中触发循环导入或 CUDA 初始化错误

- AC-8: **统计信息完整** — 空间模块返回包含所有必要统计字段的字典
  - Positive Tests:
    - `spatial_stats` 字典包含键：`candidate_count`、`selected_count`、`occupied_voxel_count`、`voxel_size`、`runtime_ms`、`jaccard_vs_original`、`replaced_count`
    - 所有值都是 Python 原生标量（可 JSON 序列化）
  - Negative Tests:
    - 不包含张量或 CUDA 设备引用（防止内存泄漏）

- AC-9: **训练烟雾测试可达致密化** — 开启空间模块的训练至少完成一次致密化间隔
  - Positive Tests:
    - `--enable_spatial_diversity` 启动训练，到达第一个 `densification_interval` 不崩溃
    - CSV 日志中 `spatial_runtime_ms` 和 `spatial_selected_count` 字段被正确写入
  - Negative Tests:
    - 空间模块代码路径不触发 CUDA 非法内存访问或 OOM

- AC-10: **不修改现有实现** — git diff 确认未修改任何现有功能文件的核心逻辑
  - Positive Tests:
    - `git diff conf-opt -- scene/gaussian_model.py` 仅显示新增的空间模块调用（后处理器模式），不修改 Conf 计算、abs-grad 计算、mask 合并、EAS 评分、clone、split、prune、loss、backward、optimizer
    - `git diff conf-opt -- utils/candidate_selector.py` 为空或仅增加可选参数（不改变现有行为）
  - Negative Tests:
    - Conf 公式（`conf = 1.0 - ||vec_sum|| / mag_sum`）完全不变
    - abs-grad 公式和阈值完全不变
    - `long_axis_split`、`densify_and_clone`、`prune_points` 函数体完全不变

- AC-11: **正确处理边缘情况**
  - AC-11.1: 候选为空时返回全 False mask
    - Positive: `candidate_mask.sum() == 0` 时返回 `torch.zeros_like(candidate_mask)`
    - Negative: 不因空候选报错
  - AC-11.2: 预算为 0 时返回全 False mask
    - Positive: `budget == 0` 时返回全 False mask
    - Negative: 不进入 round-robin 循环
  - AC-11.3: 预算大于候选数时返回全部候选
    - Positive: `budget > candidate_count` 时选中所有候选
    - Negative: 不报错，不尝试选择不存在的候选

- AC-12: **日志和可视化兼容** — 空间选择后的 mask 符合现有日志和可视化 hook 的预期
  - Positive Tests:
    - 空间选择器在 `save_conf_visualization` hook（`gaussian_model.py` line 525-531）之后运行，可视化显示原始 `final_mask`
    - `num_split` 统计值与 `long_axis_split` 实际分裂的父 Gaussian 数量一致
  - Negative Tests:
    - `candidate_stats` 中的 `n_final_candidates` 在空间选择后被正确更新

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

实现包含：新增 `utils/spatial_diversity.py` 独立模块，提供 `select_spatially_diverse_candidates()` 纯函数；支持 `voxel` 方法（体素分区 + 确定性 round-robin 选择）；自适应体素尺寸（基于候选 Gaussian 的 `max(get_scaling, dim=1).values` 中位数）；精确预算保持（选中数量 = `min(budget, available_candidates)`）；统计信息字典返回；`scene/gaussian_model.py` 中在 `all_budget` 计算后、可视化 hook 前插入 8-15 行后处理代码；`arguments/__init__.py` 中新增 6 个空间相关参数；`train.py` 中扩展 CSV 日志字段名以包含空间统计；完整单元测试覆盖 AC-1 到 AC-11（共约 15 个测试用例）；`--enable_spatial_diversity` 默认为 False。

### Lower Bound (Minimum Acceptable Scope)

最小可行实现：新增 `utils/spatial_diversity.py`，实现 `voxel` 方法的 round-robin 选择（可用 Python 循环，不强制完全向量化）；自适应体素尺寸支持 `spatial_voxel_scale` 参数（默认 `2.0`）；`scene/gaussian_model.py` 中插入不超过 12 行代码；至少覆盖 AC-1（关闭等价性）、AC-2（子集约束）、AC-3（预算保持）、AC-11（边缘情况）的单元测试。CSV 日志扩展可作为后续工作。

### Allowed Choices

- 可用：`utils/spatial_diversity.py` 作为新增独立文件；`torch.no_grad()` 装饰器；`torch.unique`、`torch.argsort`、`torch.scatter` 进行体素操作；`self.get_xyz` 和 `self.get_scaling`（仅用于体素尺寸自动计算）；Python 原生 `dict` 和 `list` 用于小规模 bookkeeping
- 不可用：修改现有策略分发表（`STRATEGY_DISPATCH`）的函数签名；修改 `select_densification_candidates` 接口；修改 `long_axis_split` 或 `densify_and_clone` 函数体；修改 CUDA 内核或 Rasterizer；引入 `scipy`、`sklearn`、`faiss` 等新依赖；在关闭路径中进行任何额外计算或导入

> 注意：草案明确指定了体素方法为首选方案，v1 仅实现 `voxel` 方法。`radius_nms` 和 `soft_suppression` 方法在代码中预留接口但保持未实现状态（`raise NotImplementedError`），待后续实验中按需添加。

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

空间多样性选择器的核心机制（体素轮转选择）：

```
输入: base_mask (N, bool), scores (N, float), xyz (N, 3), budget (int), voxel_size (float)
输出: spatial_mask (N, bool), stats (dict)

1. 筛选候选: cand_idx = base_mask.nonzero().squeeze(-1)
2. 若候选数为0或预算为0: 返回全False mask
3. 计算体素索引:
   xyz_min = xyz[cand_idx].min(dim=0)
   voxel_idx = ((xyz[cand_idx] - xyz_min) / voxel_size).long()
   # 将3D体素索引编码为1D: v = vx + vy*Nx + vz*Nx*Ny
4. 按体素分组:
   对每个非空体素，将其中的候选按 scores 降序排列
5. 轮转选择:
   selected = []
   round = 0
   while len(selected) < budget:
     for each voxel with remaining candidates:
       if round < len(voxel_candidates):
         selected.append(voxel_candidates[round])
       if len(selected) >= budget: break
     round += 1
     if 所有体素都没有第round个候选: break
6. 构建spatial_mask: 将selected中的索引对应的位置设为True
7. 确保selection_score为正: spatial_scores[spatial_mask] clamp到正小量
8. 计算统计: occupied_voxel_count, jaccard vs base_mask top-k等
```

### Relevant References

- `scene/gaussian_model.py:444-558` (`densify_and_prune_Improved`) — 后处理器插入点，紧接在 `all_budget` 计算后（line 520）、可视化 hook（line 525-531）之前
- `scene/gaussian_model.py:111-113` (`get_scaling`) — 返回 `torch.exp(self._scaling)`，即线性空间标准差，形状 (N,3)
- `scene/gaussian_model.py:121` (`get_xyz`) — 返回世界空间坐标 (N,3)
- `scene/gaussian_model.py:705-771` (`long_axis_split`) — 使用 `torch.multinomial(padded_importance, budget)` 从 `final_mask` 中采样父 Gaussian
- `utils/candidate_selector.py:65-111` (`normalize_scores`) — 分数归一化模式，可用于参考（空间模块不需要归一化）
- `utils/candidate_selector.py:144-180` (`select_by_topk`) — 现有的 top-k 选择模式
- `arguments/__init__.py:129-163` — 候选选择参数定义模式，新增参数放此处
- `train.py:319-323` — `densify_and_prune_Improved` 调用点，传入 `tt_importance`（融合分数）和 `budget`
- `train.py:340-360` — CSV 日志字段名定义，需扩展以包含空间统计

## Dependencies and Sequence

### Milestones

1. **空间多样性核心模块**：实现 `utils/spatial_diversity.py` 及其单元测试
   - Phase A: 创建 `utils/spatial_diversity.py`，实现 `select_spatially_diverse_candidates()` 函数的体素轮转选择逻辑
   - Phase B: 实现自适应体素尺寸计算（基于候选 Gaussian scale 中位数）
   - Phase C: 编写单元测试覆盖 AC-2, AC-3, AC-4, AC-5, AC-6, AC-11

2. **参数和开关集成**：将空间模块接入训练流程
   - Phase A: 在 `arguments/__init__.py` 中新增空间相关参数
   - Phase B: 在 `scene/gaussian_model.py` 中插入后处理器调用
   - Phase C: 扩展 `train.py` CSV 日志以包含空间统计

3. **端到端验证**：确保关闭等价性和开启正确性
   - Step 1: 验证关闭状态下训练结果与当前 conf-opt 分支一致（AC-1）
   - Step 2: 开启状态下烟雾测试到达致密化间隔（AC-9）
   - Step 3: 验证所有现有测试继续通过（AC-7）
   - Step 4: git diff 确认未修改现有实现（AC-10）

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| spatial-1 | 创建 `utils/spatial_diversity.py`，实现体素轮转选择核心逻辑 | AC-2, AC-3, AC-4, AC-11 | coding | - |
| spatial-2 | 实现自适应体素尺寸计算和确定性排序 | AC-5, AC-6 | coding | spatial-1 |
| spatial-3 | 编写空间模块单元测试（覆盖候选为空、预算为0、预算不足、设备一致性等） | AC-2, AC-3, AC-4, AC-5, AC-11 | coding | spatial-1 |
| spatial-4 | 在 `arguments/__init__.py` 中新增空间多样性参数 | AC-10 | coding | spatial-1 |
| spatial-5 | 在 `scene/gaussian_model.py` 中插入后处理器调用 | AC-1, AC-9, AC-12 | coding | spatial-1, spatial-4 |
| spatial-6 | 扩展 `train.py` CSV 日志字段名以包含空间统计 | AC-8, AC-12 | coding | spatial-5 |
| spatial-7 | 验证关闭等价性：确认 `--enable_spatial_diversity=False` 与基线一致 | AC-1 | coding | spatial-5 |
| spatial-8 | 审查整体修改：git diff 确认未修改现有功能 | AC-10 | analyze | spatial-1..spatial-7 |
| spatial-9 | 训练烟雾测试：开启模块完成一个完整致密化间隔 | AC-9 | coding | spatial-5 |

## Claude-Codex Deliberation

### Agreements

- **后处理器模式优于策略分发模式**：由于预算循环依赖（`all_budget` 依赖于 `final_mask.sum()`，而空间选择又需要 `all_budget`），空间多样性应作为 `densify_and_prune_Improved` 内部的后处理器，而非新增 `STRATEGY_DISPATCH` 条目
- **插入点位于 `all_budget` 计算后、可视化 hook 前**：确保可视化看到原始 `final_mask`，但 `long_axis_split` 收到空间选择后的 mask
- **默认关闭行为必须逐比特等价**：通过条件导入和早退 guard 实现零开销关闭
- **使用 `max(get_scaling, dim=1).values` 中位数作为自适应体素尺寸基础**：与 LAS 中的现有惯例一致
- **v1 仅实现 voxel 方法**：`radius_nms` 和 `soft_suppression` 预留接口但不实现，减少 v1 的范围风险
- **选择分数契约**：空间选中的候选必须具有严格正的 `selection_score`，以确保 `torch.multinomial` 不会将它们排除

### Resolved Disagreements

- **预算确切性 vs LAS 抽样的影响**: Claude 最初假设 mask.sum() == budget 即预算保持，Codex 指出 LAS 的 `torch.multinomial` 只对 `padded_importance > 0` 的点进行抽样。解决方案：在空间路径中将选中候选的 selection_score 强制 clamp 到正小量（如 1e-6），确保所有选中的候选都参与 multinomial 抽样
- **参数命名 `scene_extent`**: Claude 最初使用 scene_extent 作为参数名，Codex 指出 `densify_and_prune_Improved` 接收的 `limitation` 是预算而非场景范围。解决方案：移除 `scene_extent` 参数，仅使用 `opt.spatial_voxel_scale * median_scale` 作为体素尺寸
- **统计信息更新时序**: Codex 指出空间选择后 `candidate_stats` 会过时。解决方案：在空间选择路径中更新 `self.candidate_stats` 的 `n_final_candidates`、`actual_budget` 并添加空间统计字段

### Convergence Status

- Final Status: `partially_converged` — 架构设计完全收敛（后处理器模式、插入点、预算处理），但 3 个研究性决策需要用户确认（见下方 Pending User Decisions）

## Pending User Decisions

- DEC-1: **空间选择是精确选定分裂父节点还是仅重平衡候选池？**
  - Claude Position: 空间选择器应精确选定分裂父节点（`spatial_mask.sum() == all_budget`，所有选中点 score > 0），这样 `long_axis_split` 的 `torch.multinomial` 将确定性地分裂所有选中点（`replacement=False, budget == num_positive`）。这保证了可复现性和公平消融
  - Codex Position: 未明确反对，但指出这实质上将空间选择逻辑提升为"实际父节点选择器"，而不仅是候选池调整器
  - Tradeoff Summary: 精确父节点选定更可复现、更易分析，但偏离了原始设计的"保持 LAS 抽样"语义
  - Decision Status: `PENDING`

- DEC-2: **体素轮转顺序** — 每轮中体素应按什么顺序遍历？
  - Claude Position: 按体素的最高分候选分数降序排列体素（高分体素优先遍历）。这保证了质量（全局最高分候选优先），但可能导致高分密集区域在首轮中占据大部分选择
  - Codex Position: 确定性随机化体素顺序（固定 seed 的 hash 或按体素键排序），且需要明确
  - Tradeoff Summary: 优先级顺序偏向高分区域（更安全、质量可能更高）；确定性随机化更均匀（空间多样性更强）。效果差异需要实验评估
  - Decision Status: `PENDING`

- DEC-3: **空间多样性在 iter > 14500 的后退路径中是否生效？**
  - Claude Position: 仅在 iter <= 14500 时应用空间多样性；iter > 14500 时强制回退原始行为（包括不启用空间模块）。这尊重了代码中 `post-14500 统一回退` 的显式设计意图
  - Codex Position: 同意需要明确的决策，但未给出偏好
  - Tradeoff Summary: 仅在 iter <= 14500 应用保持了原始设计的一致性（后 14500 是纯粹的 abs-grad 回退）。全区间应用可测试空间多样性在训练后期的效果，但可能干扰原始回退逻辑
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- 实现代码和注释不得包含计划专用术语，如 "AC-"、"Milestone"、"Step"、"Phase" 或类似工作流标记
- 这些术语仅用于计划文档，不出现在最终代码库中
- 代码中使用描述性的、领域合适的命名（如 `spatial_mask`、`voxel_size`、`round_robin_select`）
- 空间模块的所有新函数必须使用 `@torch.no_grad()` 装饰器
- 遵循现有代码库的注释密度和命名惯例（参见 `utils/candidate_selector.py` 的风格）
- 新增文件头部包含简要的模块说明 docstring

### Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# Voxel-Grid Spatial Diversity Selector for Gaussian Densification Budget Reallocation

## Original Idea

你是一名熟悉3D Gaussian Splatting、PyTorch、Gaussian densification、候选点排序、空间哈希、Voxel划分和启发式搜索的研究型代码工程师。请分析项目 https://github.com/aiirudi/RFGS 的 conf-opt 分支，在现有Conf mask、abs-grad mask及EAS候选评分功能之后，研究并增加一个独立的"空间多样性抑制"模块。该模块用于在有限densification预算下抑制空间位置高度重复的候选Gaussian，使最终选中的候选点覆盖更多不同的困难区域。核心约束：不得修改任何现有功能的实现（Conf、abs-grad、EAS评分、clone/split/prune、loss/backward/optimizer、CUDA Rasterizer等）。空间多样性模块必须作为新增、独立、可关闭的后处理模块存在，关闭时运行结果与当前conf-opt分支完全一致。分析三种方案：Voxel Diversity Top-K、3D Radius NMS、Soft Spatial Suppression。优先推荐Voxel Diversity Top-B方案作为最小侵入实现。需要基于真实代码（非标准3DGS结构）定位Conf、abs-grad、EAS、clone/split的数据流和插入位置。模块默认关闭，通过--enable_spatial_diversity开关控制。在Tanks&Temples数据集上进行消融实验。

## Primary Direction: Voxel-Grid Diversity Selector

### Rationale

Partition the 3D space of candidate Gaussians into a uniform voxel grid, then perform round-robin selection from each non-empty voxel (picking the highest EAS-scored candidate per voxel per round) until the budget is exhausted. This is the most straightforward spatial diversity approach and directly extends the existing 7-strategy `STRATEGY_DISPATCH` pattern already proven in `utils/candidate_selector.py`.

### Approach Summary

Build a **Voxel-Grid Diversity Selector** as a new candidate-selection strategy inside `utils/candidate_selector.py`, following the existing `STRATEGY_DISPATCH` pattern (7 strategies already in place). The core mechanism:

1. **Voxel partitioning**: Use the 3D positions (`self.get_xyz`) of all candidate Gaussians (those passing `final_mask`). Partition them into a uniform voxel grid with configurable resolution. Assign each candidate to a voxel cell index via `q_i = floor((xyz_i - xyz_min) / voxel_size)`.

2. **Round-robin selection**: For each non-empty voxel cell, sort candidates by their EAS/RFAS fusion score (`selection_score`) in descending order. Run a round-robin loop: pick the highest-scored candidate from each non-empty voxel per round, repeating until the `all_budget` target is met.

3. **Budget preservation**: The round-robin loop keeps going until the exact budget is filled (or all candidates exhausted), ensuring the voxel-diversified selection uses the same number of Gaussians as the original strategy — critical for fair ablation.

4. **Return semantics**: The strategy returns a diversity-selected `final_mask` (bool, N,) and `selection_score` (float, N,), conforming to the existing `(mask, score)` contract. Downstream `long_axis_split` consumes them unchanged via `torch.multinomial`.

**Affected components**: `utils/candidate_selector.py` (+1 strategy function, dispatch table update, ~80 LOC), `arguments/__init__.py` (add ~4 hyperparameters: `--candidate_voxel_resolution`, `--candidate_voxel_scale`, `--candidate_max_per_voxel`), `scene/gaussian_model.py` (pass `xyz` to `select_densification_candidates` via optional parameter — zero change to existing strategies), `train.py` (no structural change needed at the call site).

### Objective Evidence

- `/home/xzh/xzh/RFGS/utils/candidate_selector.py`, lines 526-534: `STRATEGY_DISPATCH` dict maps strategy strings to functions. Seven strategies already registered (`and`, `or`, `abs_only`, `conf_only`, `weighted_score`, `soft_fusion`, `rfas_rank`). Each function receives `(abs_mask, conf_mask, abs_score, conf_score, rfas_score, config)` and returns `(mask, selection_score)`. A new `voxel_diversity` key fits this contract cleanly with an extension for `xyz`.
- `/home/xzh/xzh/RFGS/scene/gaussian_model.py`, lines 444-558: `densify_and_prune_Improved` — the exact seam where selection meets densification. `select_densification_candidates()` returns `final_mask` and `selection_score` at line 505-513. Budget is computed at lines 519-521: `all_budget = budget - curr_points`. Lines 534-541: `long_axis_split(selection_score, all_budget, final_mask, ...)` samples parents via `torch.multinomial`. **The voxel diversity selector interposes exactly here: it modifies which candidates are in `final_mask` before the multinomial draw.**
- `/home/xzh/xzh/RFGS/scene/gaussian_model.py`, line 121: `self.get_xyz` property returns `self._xyz` as `(N, 3)` float32 tensor in world-space coordinates. Available inside `densify_and_prune_Improved` at the call site.
- `/home/xzh/xzh/RFGS/train.py`, lines 262-269: Budget variable name is `budget` (Python `int`), ramped via Growth Control as `int(sqrt(rate) * opt.budget)` where `opt.budget` defaults to `1_777_778` (defined in `arguments/__init__.py`, line 118).
- `/home/xzh/xzh/RFGS/arguments/__init__.py`, lines 129-163: The `candidate_*` parameter block shows exactly how to add new strategy-specific knobs alongside existing ones (`candidate_selection_strategy`, `candidate_budget_mode`, `candidate_abs_threshold`, etc.).
- `/home/xzh/xzh/RFGS/tests/test_candidate_selector.py`: 391-line comprehensive test suite with 18 tests covering all 7 strategies, fixed-budget logic, NaN/Inf exclusion, device consistency. Provides the pattern for adding voxel-diversity tests.
- **No prior voxel or spatial-diversity code exists** in the densification pipeline: grep for `voxel`, `grid_partition`, `spatial_diversity`, `round_robin` returns zero hits across all `.py` files. The strategy dispatch infrastructure is the only prior art worth extending.

### Known Risks

- **Strategy function signature extension**: The existing `STRATEGY_DISPATCH` functions do not receive `xyz` positions. Adding spatial awareness requires either (a) extending the function signature with an optional `xyz=None` parameter (non-breaking for existing 7 strategies), (b) passing xyz through the `config` namespace, or (c) implementing voxel filtering as a post-processing step in `densify_and_prune_Improved`. Option (a) is cleanest but touches the dispatch interface.
- **Voxel resolution sensitivity**: Too coarse a grid provides no diversity benefit; too fine makes most cells single-occupant (degenerating to pure score-based selection). Adaptive voxel sizing based on scene extent or Gaussian scale is necessary.
- **Round-robin loop performance**: A Python `while` loop over potentially thousands of non-empty cells could be slow at million-Gaussian scale. A vectorized GPU implementation using `torch.scatter`/`torch.argsort` should replace the naive loop.
- **Interaction with LAS multinomial**: The voxel-diversified `final_mask` or `selection_score` must still produce a valid probability distribution for `torch.multinomial` in `long_axis_split`. The existing safety check at line 714 (`if budget > num: budget = num`) handles the edge case, but a voxel-sparse mask may underfill the budget.

## Alternative Directions Considered

### Alt-1: Gaussian-Scale-Aware Adaptive Radius Suppression
- Gist: Replace the global voxel grid with a per-Gaussian adaptive suppression radius derived from each primitive's existing scale parameter: `R_i = k * max(get_scaling[i])`. Iterate candidates sorted by EAS score descending; for each selected candidate, suppress all lower-scoring neighbors within distance `R_i`. This reuses the `max(get_scaling, dim=1).values` metric already established in three separate code paths (clone threshold at `gaussian_model_alter.py:508`, split threshold at line 542, and LAS child offset computation at `gaussian_model.py:726-731` where `3 * std_long` defines spatial extent).
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/scene/gaussian_model.py`, line 69: `self.scaling_activation = torch.exp`; `get_scaling` (lines 111-113) returns linear-space standard deviations, shape `(N, 3)`.
  - `/home/xzh/xzh/RFGS/scene/gaussian_model.py`, lines 705-771 (`long_axis_split`): Direct prior art for per-Gaussian scale-aware spatial operations. Reads `self.get_scaling[selected_pts_mask]` (line 722), identifies longest axis via `torch.max(stds, dim=1)` (line 726), uses `3 * std_long` as spatial extent for child placement.
  - `/home/xzh/xzh/RFGS/scene/gaussian_model_alter.py`, lines 542-543: Original `densify_and_clone` thresholds on `torch.max(self.get_scaling, dim=1).values` — same metric used as size criterion.
- Why not primary: The greedy NMS loop is `O(N * budget)` in runtime (distance computation against all remaining candidates per selection step), and the scale-based radius may cause large background Gaussians to over-suppress valid fine-detail candidates. The voxel grid is simpler, faster (`O(N)`), and budget-preserving by design.

### Alt-2: Morton-Code / Z-Order Curve Linearization
- Gist: Map 3D Gaussian positions to 1D Morton codes (Z-order curve) preserving spatial locality, sort candidates by Morton code, then apply a sliding-window diversity filter that limits how many candidates can be selected from each window. Morton encoding avoids grid-boundary artifacts and benefits from GPU radix-sort efficiency. The encoding itself has a verified CUDA precedent in this repo: `submodules/simple-knn/simple_knn.cu` (lines 45-70) already implements `prepMorton` (10-bit quantization, bit-interleaving via shift-OR-mask, `coord2Morton` kernel). The GLM library shipped with the rasterizer (`glm/gtc/bitfield.hpp`, lines 115-261) provides optimized `bitfieldInterleave` for 2D/3D/4D with SSE intrinsics.
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/submodules/simple-knn/simple_knn.cu`, lines 45-70: Complete CUDA Morton encoding with `cub::DeviceRadixSort::SortPairs` for Morton-code sorting.
  - `/home/xzh/xzh/RFGS/submodules/diff-gaussian-rasterization/third_party/glm/glm/gtc/bitfield.hpp`, lines 115-261: Production-quality bit-interleave primitives already in the dependency stack.
  - `/home/xzh/xzh/RFGS/utils/candidate_selector.py`, lines 144-180: `select_by_topk` — purely score-rank-based with zero spatial awareness, confirming this direction is genuinely novel for the densification path.
- Why not primary: Requires either duplicating the CUDA Morton encoding in PyTorch (risk of bit-level mismatch) or wrapping the existing CUDA kernel as a Python extension (C++ build complexity). The sliding-window boundary artifact (two spatially adjacent points falling into different windows) requires overlapping-window mitigations. Voxel hashing is simpler to implement, debug, and test.

### Alt-3: Score-Penalty Soft Spatial Suppression
- Gist: Instead of hard-suppressing candidates, apply a continuous spatial penalty to EAS scores based on proximity to already-selected points using a Gaussian decay kernel: `penalty = 1.0 - exp(-||x - x_selected||^2 / (2 * sigma^2))`. Iteratively select the highest remaining score, then multiply all remaining scores by the penalty. This preserves the budget exactly and avoids grid-boundary artifacts, at the cost of `O(N * budget)` runtime per densification step.
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/train.py`, lines 849-866 and 769-803: EAS and RFAS scores are `(N,)` float32 tensors on CUDA. Fused via `fuse_importance_scores` at line 706-728 (minmax normalization + weighted sum, alpha=0.4).
  - `/home/xzh/xzh/RFGS/utils/candidate_selector.py`, lines 65-111: `normalize_scores` normalizes `(N,)` tensors to `[0,1]` with NaN/Inf sanitization — directly reusable for penalty computation.
  - `/home/xzh/xzh/RFGS/scene/gaussian_model.py`, lines 535-541: `torch.multinomial(padded_importance, budget)` consumes the selection score — confirms scores must remain non-negative (penalty multiplication preserves this).
  - **No existing spatial proximity, soft suppression, or score-penalty code found** anywhere in the codebase.
- Why not primary: `O(N * budget)` runtime is prohibitive at scale (N up to 1.3M, budget up to several thousand). The `sigma` hyperparameter is highly sensitive — too small degenerates to top-k, too large to random selection — and has no calibration data. The voxel grid achieves similar diversity with `O(N)` complexity and fewer tunable knobs.

### Alt-4: KD-Tree Based Radius NMS
- Gist: Build a 3D KD-tree over all candidate Gaussian centers, then greedily select by EAS score while suppressing neighbors within a suppression radius. The radius can be a fixed fraction of `cameras_extent` (scene bounding sphere radius, stored at `scene/__init__.py:69` as `self.cameras_extent = scene_info.nerf_normalization["radius"]`). This is the standard NMS approach adapted from 2D object detection to 3D points.
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/scene/__init__.py`, line 69: `self.cameras_extent = scene_info.nerf_normalization["radius"]` — the canonical scene extent, computed as `diagonal * 1.1` from camera centers in `scene/dataset_readers.py:45-66`.
  - `/home/xzh/xzh/RFGS/submodules/simple-knn/simple_knn.cu`: Uses Morton-code spatial ordering for box-search approximate KNN, NOT a KD-tree. Exposes only `distCUDA2()` (mean distance to 3 nearest neighbors), not neighbor indices.
  - **No KD-tree, ball-tree, `scipy.spatial`, `sklearn.neighbors`, or `faiss` anywhere in the repo** (confirmed by grep of all `.py`, `.yml`, `.txt` files). Adding KD-tree support would require a new dependency or a custom CUDA implementation.
- Why not primary: No existing KD-tree or nearest-neighbor infrastructure in the repo. Adding `scipy` as a dependency changes the environment. The greedy NMS loop has the same `O(N * budget)` runtime issue. Radius calibration (fixed vs. adaptive) introduces another tuning dimension without precedent. The voxel grid achieves equivalent spatial diversity without external dependencies or `O(N*budget)` cost.

### Alt-5: Graph-Cut Based Spatial Partitioning
- Gist: Model candidates as a weighted graph (nodes = candidates weighted by EAS score, edges connect spatially nearby Gaussians) and solve a maximum-weight independent set (MWIS) relaxation via greedy coordinate descent. This is the most mathematically principled approach — selecting one representative per spatial neighborhood maximizes total importance under spatial constraints.
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/submodules/diff-gaussian-rasterizer/cuda_rasterizer/rasterizer_impl.cu`, lines 262, 419: CUB `DeviceRadixSort::SortPairs` and `DeviceScan::InclusiveSum` are already linked — demonstrates GPU radix-sort availability for spatial bucketing.
  - `/home/xzh/xzh/RFGS/scene/gaussian_model.py`, lines 729, 750: `torch.scatter` used in `long_axis_split` — confirms scatter primitives compile in this environment (PyTorch 1.12.1, CUDA 11.6).
  - **Zero graph libraries** (`torch_geometric`, `networkx`, `scipy.sparse`, `igraph`, `dgl`) in the entire repo. Zero `torch.cdist`, `torch.pdist`, or sparse tensor usage. No adjacency or connectivity structures exist anywhere in the non-submodule codebase.
- Why not primary: Building a full NxN adjacency matrix is infeasible at 1M+ Gaussian scale. No graph infrastructure exists anywhere in the codebase. PyTorch 1.12.1's sparse tensor support is experimental. The implementation complexity is an order of magnitude higher than the voxel approach with unproven benefit. Exploratory with no concrete precedent in this codebase.

## Synthesis Notes

The voxel-grid diversity selector (primary) benefits from being the simplest, fastest (`O(N)` complexity), and most architecturally aligned with the existing `STRATEGY_DISPATCH` pattern. However, each alternative contributes a technique that could be folded in:

- **From Alt-1 (Scale-Adaptive Radius)**: The per-Gaussian scale metric (`max(get_scaling, dim=1).values`) could replace the global voxel size as an *adaptive* voxel size — each Gaussian's voxel cell assignment could be weighted by its spatial extent, producing a multi-resolution grid where large Gaussians occupy coarser cells and small Gaussians occupy finer ones. This would marry the simplicity of voxel hashing with the physical meaningfulness of per-primitive scales.
- **From Alt-2 (Morton Code)**: The Morton-code linearization could replace explicit voxel hashing as the spatial ordering primitive. A Morton-sorted candidate list with a sliding-window filter is conceptually equivalent to voxel round-robin but avoids the hash-table overhead. The existing CUDA Morton encoder in `simple-knn/simple_knn.cu` could be wrapped as a PyTorch extension for Python-side access.
- **From Alt-3 (Soft Suppression)**: The soft penalty kernel could be applied *within* each voxel cell (instead of hard top-1 per round), allowing a voxel to contribute slightly more than `max_per_voxel` candidates when its internal EAS scores vary dramatically — preserving fine-grained priority differences within dense regions.
- **From Alt-4 (Radius NMS)** and **Alt-5 (Graph-Cut)**: These are architecturally too different to graft into the voxel approach without rearchitecting it. They serve better as separate ablation baselines than as feature donors.

If the user preferred to pivot to a scale-adaptive primary, the voxel grid infrastructure would still be useful as a fast pre-bucketing step before per-scale NMS refinement. If Morton-code were the primary, the voxel-grid round-robin logic could be recast as a post-sort window filter with minimal rewrite.

--- Original Design Draft End ---
