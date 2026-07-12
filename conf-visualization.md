# Conf 候选点标红可视化计划

## Goal Description

在训练 densification 循环中，将 Conf（梯度方向冲突分数）选中的 Gaussian 三维中心投影到当前训练相机视图，以红色实心圆标注，同时保存未标注的原始渲染图作为对照。可视化必须在完整 densification interval 的 Conf 累计统计完成后、Gaussian clone/split/prune 执行前进行，保证红点语义对应 interval 级别的候选掩码而非单步瞬时结果。整个可视化必须是旁路逻辑——开启时不改变训练行为，关闭时零开销。

## Acceptance Criteria

遵循 TDD 理念，每个验收标准包含正向测试（应通过）和负向测试（应失败）以确保可确定性验证。

- AC-1: `--visualize_conf=False`（默认）时，可视化路径零开销
  - Positive Tests:
    - 默认训练（不传 `--visualize_conf`）完成 30K iteration，loss 曲线、PSNR/SSIM/LPIPS 与基线一致
    - 训练过程中无额外 CPU-GPU 同步、无 PIL 对象创建、无文件写入
  - Negative Tests:
    - 若代码路径构造了 `vis_context` dict 或 clone 了 mask/xyz tensor → FAIL（应被外层 guard 跳过）

- AC-2: 可视化发生在 `final_mask` 确定后、`long_axis_split()` 调用前
  - Positive Tests:
    - 在 `densify_and_prune_Improved()` 中 line 512（`total_sum` 计算）之后、line 519（`self.long_axis_split(...)`）之前插入 hook
    - 快照中的 `xyz` 长度等于 `num_gaussians_before`
  - Negative Tests:
    - 在 `long_axis_split()` 返回后或 `prune_points()` 后使用 `gaussians.get_xyz[old_mask]` → FAIL（索引已变化）

- AC-3: 红点使用 interval-累计 Conf 统计生成的 mask，而非单步瞬时结果
  - Positive Tests:
    - 可视化使用的 `conf_mask` 来自 `conf = 1.0 - ||xyz_gradient_vec_accum|| / (xyz_gradient_mag_accum + 1e-6)`（整 interval 累计的 world-space 3D 梯度向量）
    - 代码注释明确标注 "# This mask is computed from accumulated Conf statistics over the complete densification interval"
  - Negative Tests:
    - 从单个 step 的 `viewspace_point_tensor.grad` 直接计算 mask → FAIL

- AC-4: 原图和 Conf 标注图使用同一次 render 结果
  - Positive Tests:
    - 通过 `vis_context` 传入 `render_image = image.detach()`（train.py 当前 step 的渲染结果）
    - `original_image = base_image.copy()` 后先保存原图，再在副本上绘制红点
  - Negative Tests:
    - 在 `base_image` 上绘制红点后保存为"原图" → FAIL（原图已被污染）
    - 为可视化额外触发一次 render 作为"原图" → FAIL（违反同一次 render 要求）

- AC-5: 投影过滤正确：排除相机后方、图像外、NaN、Inf 点
  - Positive Tests:
    - `torch.isfinite(xyz).all(dim=-1)` 和 `torch.isfinite(projected).all(dim=-1)` 过滤
    - view-space depth > 0.2（对齐 CUDA `in_frustum()` 的 `p_view.z <= 0.2f` 阈值）
    - pixel x ∈ [0, image_width), y ∈ [0, image_height)
  - Negative Tests:
    - clip-space `z <= 0` 作为后方判断 → FAIL（应用 view-space depth）
    - 未过滤 NaN 导致 `ImageDraw.ellipse()` 静默失败 → FAIL

- AC-6: 快照在 Gaussian 数量变化前保存，且形状一致
  - Positive Tests:
    - `assert conf_mask_snapshot.ndim == 1`
    - `assert conf_mask_snapshot.shape[0] == gaussians.get_xyz.shape[0]`
    - 保存 `selected_indices`（用于后续与 `gaussian_centers` 对齐验证）
  - Negative Tests:
    - `long_axis_split()` 后使用旧索引访问新 `xyz` → FAIL

- AC-7: `--conf_vis_mask_type` 正确区分三种模式
  - Positive Tests:
    - `conf` (默认): 使用 `conf_mask = conf_mask_raw & has_enough_views`，Top-K 按 `conf_score` 裁剪
    - `final_candidates`: 使用 `final_mask`（来自 `select_densification_candidates`），Top-K 按 `selection_score` 裁剪，文件名含 `final_candidates`
    - `both`: 同时生成以上两张图，各自按对应分数裁剪，只保存一张原图
  - Negative Tests:
    - `both` 模式下保存两份重复的原图 → FAIL
    - `final_candidates` 图使用 `conf_score` 做 Top-K → FAIL（应用 `selection_score`）

- AC-8: 点数超过 `--conf_vis_max_points` 上限时，按对应分数降序选择 Top-K
  - Positive Tests:
    - `conf` 类型使用 `conf_score` 的 `torch.topk(..., largest=True)`
    - `final_candidates` 类型使用 `selection_score` 的 `torch.topk(..., largest=True)`
    - 日志打印：总数、选中数、可见数、实际绘制数
  - Negative Tests:
    - 直接取索引最小的前 K 个 → FAIL

- AC-9: 不修改 CUDA rasterizer 代码
  - Positive Tests:
    - `git diff --stat` 不包含 `submodules/diff-gaussian-rasterization/` 下任何文件
    - 投影在纯 Python 中实现（复用 `geom_transform_points` + `ndc2Pix` 公式）
  - Negative Tests:
    - 修改 `forward.cu`、`backward.cu`、`auxiliary.h` 或 rasterizer 输出接口 → FAIL

- AC-10: 不改变训练行为
  - Positive Tests:
    - `--visualize_conf` 关闭时，loss、backward、optimizer step、Conf 统计、densification 逻辑与原代码完全一致
    - `--visualize_conf` 开启时，仅读取 Conf 结果、Gaussian 中心、相机矩阵和渲染图，不修改任何训练参数或 mask
    - 所有绘图代码在 `with torch.no_grad():` 中执行
  - Negative Tests:
    - 可视化代码修改 `conf_mask`、`final_mask`、`selection_score` → FAIL
    - 可视化代码改变 `gaussians.get_xyz` 数量或 clone/split/prune 结果 → FAIL

- AC-11: metadata 记录完整上下文
  - Positive Tests:
    - `metadata.jsonl` 每行记录：`iteration`, `camera_uid`, `camera_name`, `mask_type`, `strategy`, `conf_thr`, `conf_min_views`, `num_gaussians_before`, `mask_count`, `drawn_count`, `topk_score_name`, `same_render`, `trigger_reason`
    - 同时保存 `selected_indices`（mask 命中的全部候选）和 `drawn_indices`（实际画出的点）
  - Negative Tests:
    - metadata 缺少 `strategy` 字段导致 `final_candidates` 在 `abs_only` 策略下语义不明 → FAIL

- AC-12: 保存路径和文件命名规范
  - Positive Tests:
    - 目录: `<model_path>/conf_interval_visualization/`
    - 文件: `iteration_{iter:06d}_view{uid}_{type}.png`（type: `render`, `conf`, `final_candidates`）
    - Conf 为 0 时也正常保存原图和 Conf 图（两张内容一致）
  - Negative Tests:
    - Conf 图文件名包含 `conf_mask` 实际为 `final_mask` → FAIL

## Path Boundaries

路径边界定义可接受的实现质量范围。草稿对插入位置、快照时机、mask 语义、投影过滤有高度确定性的规定，以下边界反映这些硬性约束。

### Upper Bound (Maximum Acceptable Scope)

实现包括:
- `utils/conf_visualization.py` 独立模块，封装投影（`project_gaussian_centers`）、过滤（`filter_visible_points`）、绘图（`draw_conf_overlay`）、保存（`save_visualization_pair`）全部逻辑
- `utils/graphics_utils.py` 新增 `ndc_to_pixel()` 工具函数，精确复制 CUDA `ndc2Pix` 公式
- `scene/gaussian_model.py` 中 `densify_and_prune_Improved` 仅做 guard + snapshot + call 三个薄调用点
- `arguments/__init__.py` 中 `OptimizationParams` 新增 6 个 CLI 参数
- `train.py` 构造 `vis_context` dict 并传入
- 开发期 `--conf_vis_validate_projection` flag 自动比较 Python 投影与 rasterizer `gaussian_centers`，误差超阈值时 warning
- 支持 `--conf_vis_all_intervals` 和 `--conf_vis_iterations` 两种触发模式
- metadata 使用 `metadata.jsonl` append 模式

### Lower Bound (Minimum Acceptable Scope)

实现包括:
- 在 `densify_and_prune_Improved()` 中 `final_mask` 后、`long_axis_split()` 前插入可视化 hook
- 通过 `vis_context` dict 显式传入 `model_path`、当前训练 `camera`、同一次 `render_image`
- 支持 `--conf_vis_mask_type conf|final_candidates|both`，默认 `conf`
- 支持 `--conf_vis_iterations` 指定 iteration 列表；未指定时默认最后一个有效 densification interval 保存一次
- 支持 `--conf_vis_max_points` Top-K 限制，默认 5000
- NDC→pixel 映射对齐 CUDA `ndc2Pix` 公式
- view-space depth > 0.2 过滤
- `torch.no_grad()` 包裹所有可视化代码
- PIL `ImageDraw.ellipse()` 绘制红色实心圆（radius=3）
- 同时保存原图 + 标注图对
- 日志打印关键统计量

### Allowed Choices

- **必须使用**: PIL (`ImageDraw`) 或 OpenCV 绘制红点；`torch.no_grad()` 包裹；`geom_transform_points()` 做 world→NDC 投影
- **必须使用**: `vis_context` dict 模式传递可视化上下文（不通过全局变量或 `gaussian_model.py` 自行获取 `model_path`）
- **必须使用**: 快照模式（clone/detach），不依赖 densification 后的索引
- **不能使用**: CUDA kernel 修改；训练后重新计算 Conf；单步瞬时梯度生成 mask
- **不能使用**: 通过 Gaussian 的 SH/RGB/opacity/scale/rotation 修改来制造红色效果——红点必须是 2D 后处理
- **mask_type 取值固定**: `conf`（默认）、`final_candidates`、`both`——不得使用 `final` 命名（易误解为实际 split 点）
- **投影过滤方法固定**: finite check → view-space depth > 0.2 → pixel bounds
- **插入位置固定**: `densify_and_prune_Improved()` 中 `final_mask` 确定后、`long_axis_split()` 前，不得在其他位置

## Feasibility Hints and Suggestions

> **注意**: 本节仅供参考和理解，是概念性建议而非规定性要求。

### Conceptual Approach

整体流程：

```
train.py: 构造 vis_context = {model_path, camera, render_image, iteration, camlist}
    ↓
gaussian_model.py: densify_and_prune_Improved(..., vis_context=None)
    ↓
    conf = 1 - ||vec_accum|| / (mag_accum + 1e-6)    # line 465
    conf_mask = conf_raw & has_enough_views           # line 476
    final_mask, selection_score = select_densification_candidates(...)  # line 500
    ↓
    # ---- 可视化插入点 (line 512 之后) ----
    if vis_context is not None:
        utils/conf_visualization.py::save_conf_visualization(
            gaussians, conf_mask, conf, final_mask, selection_score, vis_context
        )
    ↓
    # ---- 原有流程继续 ----
    long_axis_split(selection_score, budget, final_mask, ...)  # line 519
    prune_points(prune_mask)
```

`save_conf_visualization()` 内部流程：

```
1. 快照: conf_mask, final_mask, conf_score, selection_score, xyz, selected_indices → .detach().clone()
2. 断言: mask.shape[0] == xyz.shape[0]
3. 按 mask_type 选择目标 mask 和对应 score
4. 投影: world_xyz → NDC (geom_transform_points) → pixel (ndc2Pix 公式)
5. 过滤: finite, view-space depth > 0.2, pixel bounds
6. Top-K: 按对应 score 降序取前 conf_vis_max_points
7. 保存原图: 将 render_image tensor → uint8 PIL Image → save as render.png
8. 绘制红点: PIL ImageDraw.ellipse() 在副本上画红色实心圆
9. 保存标注图: save as conf.png / final_candidates.png
10. 写入 metadata.jsonl (append 模式)
11. 日志打印统计量
```

### Relevant References

- `scene/gaussian_model.py:444-542` — `densify_and_prune_Improved()`：Conf 统计消费端。line 465 计算 `conf`，line 476 生成 `conf_mask`，line 500-508 生成 `final_mask`，line 519 调用 `long_axis_split()`
- `scene/gaussian_model.py:398-404` — `densification_postfix()`：在此间隙之后才重置 `xyz_gradient_vec_accum` 等累加器，保证快照时数据完整
- `scene/gaussian_model.py:621-651` — `add_densification_stats()`：每步将 NDC 梯度通过 `_compute_ndc_vjp_world` 拉回世界空间，累加到 `xyz_gradient_vec_accum` (N,3) 和 `xyz_gradient_mag_accum` (N,1)
- `scene/gaussian_model.py:689-755` — `long_axis_split()`：从 `final_mask` 候选中通过 `torch.multinomial` 按 `selection_score` 抽样，生成 `selected_pts_mask` 作为实际 split 的父点
- `utils/graphics_utils.py:22-29` — `geom_transform_points()`：world → NDC 投影（齐次乘 + 透视除法），需补充 NDC→pixel 映射
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h:41-44` — CUDA `ndc2Pix` 公式：`pix = ((v + 1.0) * S - 1.0) * 0.5`
- `scene/cameras.py:54-56` — Camera 暴露 `full_proj_transform`、`image_width`、`image_height`、`image_name`、`uid`，满足投影所需全部参数
- `train.py:150` — 整个 densification 块已在 `torch.no_grad()` 下运行
- `train.py:199-264` — densification interval 触发逻辑：`camlist` 构造、EAS/RFAS 评分、`densify_and_prune_Improved` 调用
- `train.py:267-292` — CSV 统计导出在 `densify_and_prune_Improved` 返回后执行，为 I/O 副作用建立了先例
- `utils/candidate_selector.py:541-640` — `select_densification_candidates()`：统一候选点选择入口，支持 `and/or/abs_only/conf_only/weighted_score/soft_fusion/rfas_rank` 策略
- `arguments/__init__.py:124-129` — 已有 `conf_min_views`、`conf_thr`、`candidate_selection_strategy` 等 Conf 相关参数，为新增可视化参数提供了模式模板

## Dependencies and Sequence

### Milestones

1. **投影工具验证**: 在 `utils/graphics_utils.py` 新增 `ndc_to_pixel()` 并验证与 CUDA rasterizer `gaussian_centers` 的像素级对齐
   - Step A: 实现 `ndc_to_pixel(ndc_xy, width, height)` 函数
   - Step B: 选取一个已训练的 checkpoint，对单帧比较 Python 投影与 rasterizer 返回的 `gaussian_centers`
   - Step C: 确认 median 误差 ≤ 0.5px，p95 误差 ≤ 1px

2. **可视化核心模块**: 创建 `utils/conf_visualization.py`，实现快照、投影、过滤、绘图、保存逻辑
   - Step A: 实现 `project_gaussian_centers(xyz, full_proj_transform, W, H)` → pixel_coords
   - Step B: 实现 `filter_visible_points(pixel_coords, view_depth, W, H)` → valid_mask
   - Step C: 实现 `draw_red_dots(image_pil, pixel_coords, radius)` → annotated_image
   - Step D: 实现 `save_visualization_pair(render_image, annotated, metadata, output_dir, filename_prefix)`
   - Step E: 实现顶层 `save_conf_visualization(gaussians, masks, scores, vis_context, opt)` 协调函数

3. **参数声明**: 在 `arguments/__init__.py` 的 `OptimizationParams` 中新增 CLI 参数
   - Step A: 添加 `visualize_conf`、`conf_vis_iterations`、`conf_vis_all_intervals`、`conf_vis_point_radius`、`conf_vis_max_points`、`conf_vis_mask_type`
   - Step B: 验证参数解析正确（默认值、类型、范围）

4. **Hook 集成**: 修改 `densify_and_prune_Improved` 签名和 `train.py` 调用点
   - Step A: `densify_and_prune_Improved` 新增 `vis_context=None` 参数
   - Step B: 在 line 512 后插入可视化 guard + call
   - Step C: `train.py` 构造 `vis_context` dict 并传入
   - Step D: 实现 `_should_visualize(iteration, opt)` 判断逻辑

5. **端到端验证**: 用实际训练验证完整流程
   - Step A: 关闭 `--visualize_conf` 训练到 7000 iter，确认 loss/metrics 与基线一致
   - Step B: 开启 `--visualize_conf --conf_vis_iterations 7000` 训练，确认生成图片对且语义正确
   - Step C: 验证 `final_candidates` 和 `both` 模式

## Task Breakdown

每个任务必须包含恰好一个路由标签：`coding`（由 Claude 实现）或 `analyze`（由 Codex 执行）。

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task-1 | 在 `utils/graphics_utils.py` 实现 `ndc_to_pixel()` 并验证与 CUDA 投影对齐 | AC-5, AC-9 | coding | - |
| task-2 | 创建 `utils/conf_visualization.py`：投影、过滤、绘图、保存模块 | AC-4, AC-5, AC-6, AC-11 | coding | task-1 |
| task-3 | 在 `arguments/__init__.py` 新增 6 个 CLI 可视化参数 | AC-1, AC-7, AC-8 | coding | - |
| task-4 | 修改 `scene/gaussian_model.py::densify_and_prune_Improved` 添加 vis_context hook | AC-2, AC-3, AC-6, AC-10 | coding | task-2, task-3 |
| task-5 | 修改 `train.py` 构造 vis_context 并传入 densification 调用 | AC-2, AC-4, AC-10 | coding | task-3, task-4 |
| task-6 | Codex 审查整体实现：验证 AC 覆盖、mask 语义、过滤正确性、零开销关闭 | AC-1~12 | analyze | task-4, task-5 |
| task-7 | 端到端验证：关闭可视化训练基线 + 开启可视化训练验证 | AC-1, AC-2, AC-3, AC-4, AC-7, AC-8, AC-10, AC-12 | coding | task-5 |
| task-8 | 元数据对齐验证：`selected_indices` 与 rasterizer `gaussian_centers` 交叉比对 | AC-5, AC-11 | analyze | task-6, task-7 |

## Claude-Codex Deliberation

> **注意**: Codex（GPT-5.5）在执行期间因 chatgpt.com 网络超时三次均不可用。以下内容基于 Claude 独立分析（deepseek-v4-pro），包括对 CUDA rasterizer 投影公式、Conf 累加器语义、mask 数据流的逐行验证。由于缺少 Codex 跨模型审查，跨审查置信度降低，计划状态为 `partially_converged`。

### Agreements

N/A — Codex 未参与，不存在双方协议。以下是 Claude 基于代码分析的独立结论：

- **插入点**: `densify_and_prune_Improved()` 中 `final_mask` 确定后（line 510）、`long_axis_split()` 调用前（line 519）是唯一正确的可视化窗口
- **独立模块架构**: 将投影/绘图逻辑放在 `utils/conf_visualization.py` 独立模块，`densify_and_prune_Improved` 中只做 thin call
- **vis_context 传递模式**: 通过 `vis_context` dict 显式传入 `model_path`、`camera`、`render_image`，不让 `gaussian_model.py` 自己耦合训练全局变量
- **mask 命名**: 使用 `final_candidates` 命名而非 `final`，避免误解为 `long_axis_split()` 的实际 split 点
- **Top-K 规则**: `conf` 类型用 `conf_score`、`final_candidates` 类型用 `selection_score`
- **投影过滤**: 使用 view-space depth > 0.2（对齐 CUDA `in_frustum()`），而非 clip-space z
- **不修改 CUDA**: 纯 Python 投影方案

### Resolved Disagreements

N/A — 无 Codex 参与，不存在分歧。

### Convergence Status

- Final Status: `partially_converged`
- 收敛轮次: 0（Codex 三次调用全部失败，未能执行收敛审查）
- Codex 失败原因: chatgpt.com 网络超时（`rmcp::transport::worker quit with fatal: Transport channel closed`）
- 风险: 缺少跨模型审查可能导致未发现的边界条件遗漏或投影精度问题

## Pending User Decisions

以下决策已由 Claude 在分析中确定，但缺少 Codex 验证。用户应确认或在启动实现前提出异议：

- DEC-1: 可视化相机选择 — 使用 `viewpoint_cam`（当前训练相机）还是 `camlist` 中的相机？
  - Claude Position: 使用 `viewpoint_cam`（可与 train.py line 120 的 `render_pkg["render"]` 匹配，满足"同一次 render"要求）
  - Tradeoff: `camlist` 多视角更全面但需要额外渲染；`viewpoint_cam` 满足 AC-4 且实现更简单
  - Decision Status: `PENDING` — Claude 推荐 `viewpoint_cam`

- DEC-2: `actual_split` mask_type 支持（`long_axis_split` 内部实际 `torch.multinomial` 抽中点）
  - Claude Position: v1 不支持（需要侵入 `long_axis_split` 内部暴露 `selected_pts_mask`），命名 `final_candidates` 明确语义
  - Tradeoff: 复杂度较高但语义更精确
  - Decision Status: `PENDING` — Claude 推荐推迟到 v2

- DEC-3: `final_candidates` 在非 Conf 策略（`abs_only`、`rfas_rank`）下是否仍输出
  - Claude Position: 允许，但 metadata 中强制记录 strategy
  - Tradeoff: 不限制策略可让用户观察不同策略下的候选分布
  - Decision Status: `PENDING` — Claude 推荐允许

## Implementation Notes

### Code Style Requirements

- 代码和注释中不得包含计划专用术语如 "AC-"、"Milestone"、"Step"、"Phase" 或类似工作流标记
- 这些术语仅用于本文档，不应出现在最终代码中
- 使用领域内描述性命名（如 `save_conf_visualization`、`project_gaussian_centers`、`filter_visible_points`）

### 关键注意事项

- Conf 梯度累加器 `xyz_gradient_vec_accum` 是 (N,3) world-space 向量，不是 2D xy 视图空间。Conf 公式 `1 - ||vec_accum|| / mag_accum` 作用于 3D world vectors
- `final_mask` 是 selector 输出的候选池，不是 `long_axis_split()` 实际 split 的点（后者内部通过 `torch.multinomial` 抽样）
- 默认行为（未指定 iterations 且未开启 all_intervals）：只在最后一个有效 densification interval 保存一次，通过 `trigger_reason: default_last_interval` 标注
- `both` 模式只保存一张原图（不重复写 `render.png`）
- `metadata.jsonl` 使用 append 模式，避免频繁读写合并 JSON

--- Original Design Draft Start ---

# Conf Candidate Red-Dot Visualization via Inline Densification Hook

## Original Idea

你是一名熟悉3D Gaussian Splatting、PyTorch、CUDA Rasterizer、Gaussian densification和多视角梯度统计的代码工程师。

请基于以下项目进行代码分析和修改：

仓库：aiirudi/RFGS
工作分支：visualization

visualization分支已经由用户创建。本次所有修改必须只在该分支上进行，不要创建任何新分支。

该分支以现有Conf实现为基础。项目中的Conf是当前方法最主要的创新点，Conf已经在Python代码中实现，并参与Gaussian densification候选点选择。

现在需要参考Hard Gaussian Splatting的周期累计候选点可视化方式，增加Conf候选Gaussian中心标红功能。

该功能的实现时机、统计范围和候选点语义应参考Hard Gaussian Splatting中PGHGS/REHGS的处理方式：

1. 候选点不是来自某一个训练step的即时结果；
2. 候选点来自一个完整densification/growth interval内的累计统计；
3. 在该interval结束、候选mask已经确定之后进行可视化；
4. 可视化必须发生在clone、split、densify、prune改变Gaussian数量之前；
5. 可视化完成后，再执行原有Gaussian growing和统计量重置。

本任务只需要把Conf选中的Gaussian三维中心投影到图像并标红，不需要生成每个像素的最大贡献Gaussian index，因此原则上不要修改CUDA Rasterizer。

========================

一、参考Hard Gaussian Splatting的正确流程
========================

Hard Gaussian Splatting中的候选Gaussian不是单次迭代临时选出的，而是在一个growth interval内完成累计统计后得到。

PGHGS流程：

多个训练视角的位置梯度累计
    ↓
在growth interval结束时排序
    ↓
检查第k大梯度
    ↓
生成PGHGS hard Gaussian mask
    ↓
执行Gaussian growing

REHGS流程：

每个训练视角检测当前视角候选
    ↓
在growth interval内累计多视角证据
    ↓
至少被多个视角确认
    ↓
生成REHGS hard Gaussian mask
    ↓
执行Gaussian growing

本项目的Conf可视化应采用同样原则：

一个完整Conf统计周期内累计Conf信息
    ↓
在densification触发时计算Conf score
    ↓
应用Conf阈值得到Conf mask
    ↓
保存当前Gaussian中心和mask快照
    ↓
将Conf选中的Gaussian中心投影并标红
    ↓
保存原始渲染图和Conf可视化图
    ↓
执行原有clone/split/densify/prune
    ↓
清空或重置下一周期统计量

禁止在训练结束后重新计算Conf。

禁止使用单个step的即时Conf结果作为最终可视化mask。

========================

二、首先分析项目现有代码
========================

请先完整阅读conf-refine分支，准确定位：

1. Conf原始梯度来自哪里；
2. Conf是否使用屏幕空间梯度或世界空间梯度；
3. 单个训练视角如何更新Conf统计量；
4. Conf累计分子；
5. Conf累计分母；
6. 每个Gaussian的有效观测次数；
7. Conf score最终计算公式；
8. Conf阈值判断位置；
9. 仅由Conf产生的mask；
10. abs-grad mask；
11. Conf mask与abs-grad mask融合后的最终densification mask；
12. densification interval的起止条件；
13. 触发clone、split和prune的位置；
14. Conf统计量清零的位置；
15. 当前训练视角viewpoint_cam；
16. 当前渲染结果render_pkg["render"]或等价变量；
17. gaussians.get_xyz；
18. 当前相机的world_view_transform、full_proj_transform、image_width和image_height。

请报告真实文件名、函数名和变量名。

不得根据名称猜测mask含义，必须根据代码数据流确认。

========================

三、默认可视化的候选点
========================

默认可视化：

仅由Conf条件选中的Gaussian。

即：

conf_selected_mask

而不是：

conf_selected_mask & abs_grad_mask

也不是最终clone/split mask。

请明确区分：

1. conf_mask：
   仅满足Conf条件的Gaussian；

2. abs_grad_mask：
   仅满足abs-grad条件的Gaussian；

3. final_mask：
   按当前项目原逻辑融合后，实际进入densification的Gaussian。

增加参数：

--conf_vis_mask_type

可选值：

conf
    只显示Conf选中的Gaussian，默认值；

final
    显示实际进入densification的最终候选Gaussian；

both
    同时生成Conf选点图和最终融合候选点图。

不得将final_mask误称为conf_mask。

========================

四、可视化的正确插入位置
========================

可视化必须插入在一个完整densification interval结束的位置。

正确时机：

1. 当前interval内的Conf统计已经累计完成；
2. Conf score已经计算；
3. Conf mask已经生成；
4. abs-grad mask和final mask如有需要也已经生成；
5. 但尚未执行clone、split、densify或prune；
6. 尚未清空Conf累计统计量。

正确顺序示例：

if densification_condition:
    conf_score = compute_conf_score(...)
    conf_mask = select_conf_mask(conf_score, ...)
    abs_grad_mask = ...
    final_mask = ...

    # 在这里生成可视化
    if should_generate_conf_visualization(...):
        save_conf_interval_visualization(...)
    
    # 可视化完成后再执行原有操作
    densify_and_prune(...)
    reset_conf_statistics(...)

禁止放置在以下位置：

- 单个训练step的backward之后立即绘图；
- Conf统计尚未完成时；
- clone或split之后；
- prune之后；
- Conf统计清零之后；
- 全部训练结束之后重新计算。

========================

五、必须使用周期累计Conf mask
========================

最终绘制的红点必须对应：

一个完整Conf/densification interval内累计统计后得到的Conf mask。

不能使用当前step的瞬时梯度生成mask。

如果现有代码存在类似变量：

current_step_conf
interval_conf
conf_accum
conf_score

必须使用最终的interval级conf_score或由其产生的conf_mask。

需要在代码中明确注释：

# This mask is computed from accumulated Conf statistics

# over the complete densification interval, not from a

# single training step.

如果当前Conf公式为：

Conf_i =
1 -
||sum_j g_{i,j}||
/
sum_j ||g_{i,j}||

则必须使用该interval内所有有效视角累计后的：

sum_j g_{i,j}

和：

sum_j ||g_{i,j}||

生成Conf mask。

不得只使用最后一个视角的g_{i,j}。

========================

六、在Gaussian数量变化前保存快照
========================

Conf mask与当前Gaussian集合严格对应。

clone、split和prune会改变Gaussian数量及索引，因此必须在这些操作之前保存快照。

推荐：

with torch.no_grad():
    conf_mask_snapshot = conf_mask.detach().bool().clone()

    assert conf_mask_snapshot.ndim == 1
    assert conf_mask_snapshot.shape[0] == gaussians.get_xyz.shape[0]
    
    conf_score_snapshot = conf_score.detach().clone()
    
    selected_ids_snapshot = torch.nonzero(
        conf_mask_snapshot,
        as_tuple=False
    ).squeeze(-1)
    
    selected_xyz_snapshot = (
        gaussians.get_xyz.detach()[conf_mask_snapshot].clone()
    )

后续绘图必须使用selected_xyz_snapshot。

禁止：

densify_and_prune(...)
selected_xyz = gaussians.get_xyz[old_conf_mask]

因为densification后Gaussian数量和索引已经变化。

========================

七、选择用于可视化的相机视角
========================

参考HGS Figure 3，候选Gaussian应投影到一个具体训练视角中进行展示。

默认使用：

当前触发densification时的viewpoint_cam。

也就是说：

- Conf mask来自整个interval的多视角累计；
- 可视化背景图来自interval结束时的当前训练视角；
- 将累计Conf选中的Gaussian投影到当前训练视角。

新增参数：

--visualize_conf
    是否启用Conf周期候选点可视化，默认False。

--conf_vis_iterations
    指定在哪些densification iteration保存图片，例如：
    --conf_vis_iterations 5000 10000 15000

--conf_vis_all_intervals
    是否在每个densification interval都保存，默认False。

--conf_vis_point_radius
    红点半径，默认3。

--conf_vis_max_points
    最多绘制点数，默认5000；
    -1表示不限制。

--conf_vis_mask_type
    conf、final或both，默认conf。

如果既未指定conf_vis_iterations，也未开启conf_vis_all_intervals，则默认只在最后一个有效densification interval保存一次。

注意：

"最后一个有效densification interval"仍然是在训练过程中、该interval候选mask生成后立即保存，不是训练全部结束后重新计算。

========================

八、原图和Conf图必须同时生成
========================

每次触发Conf可视化时，必须同时保存：

1. 当前Gaussian模型在当前视角下的原始渲染图；
2. 在同一张渲染图上加入Conf选中Gaussian中心红点后的图。

两张图必须：

- 使用同一时刻的Gaussian模型；
- 使用同一个相机视角；
- 使用同一次render结果；
- 图像尺寸完全相同；
- 除红点外，其他图像内容完全一致。

使用当前训练step已经得到的：

render_pkg["render"]

作为背景图，避免重新渲染产生随机差异。

正确流程：

base_image = convert_tensor_to_uint8(
    render_pkg["render"]
)

original_image = base_image.copy()
conf_vis_image = base_image.copy()

save_image(
    original_image,
    original_path
)

draw_red_points(
    conf_vis_image,
    projected_conf_points
)

save_image(
    conf_vis_image,
    conf_path
)

禁止先在base_image上绘制红点，再将同一个已修改对象保存为原图。

========================

九、Gaussian中心投影
========================

获取Conf选中的世界空间中心：

selected_xyz_snapshot

将其投影到当前viewpoint_cam。

优先复用项目已有的相机投影函数或与Gaussian Rasterizer一致的投影代码。

如果项目中没有公共投影函数，再根据项目实际矩阵约定实现：

world coordinates
    ↓
view transform
    ↓
projection transform
    ↓
clip coordinates
    ↓
perspective division
    ↓
NDC
    ↓
pixel coordinates

必须确认：

1. full_proj_transform是否已经转置；
2. 使用行向量还是列向量；
3. image_width和image_height字段；
4. 图像y轴是否翻转；
5. clip_w正负含义；
6. 相机前方判断方式。

必须过滤：

- NaN；
- Inf；
- 相机后方Gaussian；
- x小于0或大于等于image_width；
- y小于0或大于等于image_height。

不得修改Gaussian的SH、RGB、opacity、scale或rotation来制造红色效果。

红点必须是二维后处理绘制。

========================

十、绘制红点
========================

使用PIL.ImageDraw或OpenCV绘制红色实心圆。

推荐：

RGB = (255, 0, 0)
radius = 3

如果使用OpenCV，注意默认通道顺序为BGR，避免把红点画成蓝点。

推荐优先使用PIL：

draw.ellipse(
    [
        x - radius,
        y - radius,
        x + radius,
        y + radius
    ],
    fill=(255, 0, 0)
)

可以增加1像素白色或黑色描边，但默认应与论文Figure 3类似，使用清晰的红色实心点。

========================

十一、点数过多时的处理
========================

如果Conf选中的Gaussian数量过多：

selected_count > conf_vis_max_points

优先按照Conf score从高到低选择前K个：

selected_scores = conf_score_snapshot[
    conf_mask_snapshot
]

topk = torch.topk(
    selected_scores,
    k=conf_vis_max_points,
    largest=True
).indices

selected_xyz_snapshot = selected_xyz_snapshot[topk]
selected_ids_snapshot = selected_ids_snapshot[topk]

禁止直接使用索引最小的前K个Gaussian。

日志中打印：

- 当前iteration；
- 当前densification interval；
- 当前Gaussian总数；
- Conf选中总数；
- 当前视角可见选中点数；
- 实际绘制点数；
- Conf阈值；
- 原图路径；
- Conf图路径。

========================

十二、保存路径和文件名
========================

保存目录：

<model_path>/conf_interval_visualization/

文件示例：

iteration_005000_view_003_render.png
iteration_005000_view_003_conf.png

如果mask_type为both：

iteration_005000_view_003_render.png
iteration_005000_view_003_conf.png
iteration_005000_view_003_final.png

还可以保存：

iteration_005000_conf_score.pt
iteration_005000_conf_mask.pt
iteration_005000_conf_xyz.pt
iteration_005000_conf_ids.pt
iteration_005000_metadata.json

metadata记录：

- iteration；
- interval起始iteration；
- interval结束iteration；
- view id；
- 当前Gaussian总数；
- Conf选中数量；
- 可见点数量；
- 实际绘制数量；
- Conf阈值；
- mask类型；
- 是否使用世界空间Conf；
- 当前densification参数。

========================

十三、不得改变训练行为
========================

可视化必须是旁路逻辑。

关闭--visualize_conf时：

- 不得生成任何文件；
- 不得增加明显CPU/GPU同步；
- 不得改变随机数状态；
- 不得改变loss；
- 不得改变backward；
- 不得改变optimizer；
- 不得改变Conf；
- 不得改变densification；
- 不得改变最终指标。

开启可视化时：

- 只允许读取当前Conf结果、Gaussian中心、相机和渲染图；
- 不得修改任何训练参数；
- 不得修改mask；
- 不得修改Gaussian数量；
- 不得修改候选点选择；
- 不得改变clone、split和prune结果。

所有绘图代码放在：

with torch.no_grad():

中执行。

========================

十四、原则上不要修改CUDA
========================

本任务已有：

Conf mask
    ↓
Gaussian index
    ↓
Gaussian world-space center

因此只需要在Python中投影Gaussian中心。

不需要生成：

per-pixel maximum contribution index

不需要实现：

idx(u) = argmax_i w_i(u)

所以原则上不要修改：

- diff-gaussian-rasterization；
- CUDA forward；
- CUDA backward；
- rasterizer输出接口。

只有当项目现有Python代码无法获得相机投影矩阵时，才说明问题，但仍应优先通过现有camera对象解决，而不是修改CUDA。

========================

十五、验收标准
========================

修改完成后必须满足：

1. Conf可视化发生在一个完整densification interval结束时；

2. 红点来自该interval内累计Conf统计得到的Conf mask；

3. 红点不是来自单个训练step的临时Conf；

4. Conf mask生成后立即保存快照；

5. 可视化发生在clone、split、densify和prune之前；

6. 可视化发生在Conf统计量清零之前；

7. 可视化完成后，原有densification流程继续正常执行；

8. 每次触发可视化时，同时生成：

   - 未标记的原始渲染图；
   - Conf选点标红图。

9. 原图和Conf图使用同一次render结果；

10. 原图中不得包含红点；

11. Conf图中的红点对应当前interval累计Conf选中的Gaussian三维中心；

12. 相机后方、图像外、NaN和Inf点不得绘制；

13. Conf mask长度与可视化前gaussians.get_xyz数量严格一致；

14. 不会因densification后索引变化导致红点错位；

15. 当Conf没有选中任何Gaussian时，也正常保存原图和Conf图，两张图内容一致；

16. 当点数超过上限时，按照Conf score选择Top-K；

17. 不需要重新编译CUDA；

18. 不得改变Conf公式、Conf阈值和原有mask融合逻辑；

19. 不得改变clone、split、prune数量；

20. 关闭可视化后，训练结果与原代码一致。

========================

十六、最终输出要求
========================

完成代码修改后，请按以下顺序回答：

1. 完成修改后，还必须报告：
2. 开始修改前的分支名称；修改完成后的分支名称；
3. 确认所有代码修改是否仅位于visualization分支；
4. 确认没有创建conf-visualization或其他新分支；
5. 确认没有修改conf-refine、main或master分支；
6. 确认没有执行git reset --hard、git clean -fd或强制推送；
7. 确认是否保留了visualization分支中原有的未提交修改。
8. Conf在项目中是单步统计还是interval累计；
9. Conf累计变量的真实名称；
10. Conf score的真实公式；
11. Conf mask的真实变量；
12. abs-grad mask的真实变量；
13. final densification mask的真实变量；
14. densification interval结束的代码位置；
15. 可视化插入在哪一行之前和哪一行之后；
16. 为什么该位置与Hard Gaussian Splatting的方法一致；
17. 修改了哪些文件；
18. 给出完整unified diff；
19. 给出运行命令；
20. 给出图片保存路径；
21. 明确说明是否修改CUDA；
22. 明确说明可视化是否影响训练。

运行示例：

python train.py \
    ...原有训练参数... \
    --visualize_conf \
    --conf_vis_iterations 5000 10000 15000 \
    --conf_vis_point_radius 3 \
    --conf_vis_max_points 5000 \
    --conf_vis_mask_type conf

或者：

python train.py \
    ...原有训练参数... \
    --visualize_conf \
    --conf_vis_all_intervals \
    --conf_vis_point_radius 3 \
    --conf_vis_max_points 5000 \
    --conf_vis_mask_type conf

再次强调：

本任务必须参考Hard Gaussian Splatting的候选点处理方式。

Conf红点必须来自一个完整densification interval中的累计统计结果，并在该interval候选mask形成后、Gaussian growing执行前生成。

禁止训练完成后重新计算Conf，也禁止使用单个step的即时Conf结果。

## Primary Direction: Inline Densification Hook

### Rationale

直接在 `GaussianModel.densify_and_prune_Improved` 内嵌入可视化逻辑——这是 Conf mask 被计算和消费的唯一位置，具备最小化代码表面和间接性的优势，且与现有 inline profiling probe（`train.py:180-197`）风格一致。

### Approach Summary

在 `scene/gaussian_model.py` 的 `densify_and_prune_Improved()` 方法中，于 `final_mask` 确定后（当前 line 510）和 `long_axis_split()` 调用前（当前 line 519）之间插入可视化 hook。具体实现：

1. **新增可选参数** `vis_cams: Optional[List[Camera]] = None` 到 `densify_and_prune_Improved` 签名。
2. **快照候选点**：在 `select_densification_candidates` 返回后，对 `conf_mask` 和 `final_mask` 做 `.detach().bool().clone()`，同时取出 `self.get_xyz[conf_mask]`。
3. **投影到 2D**：复用项目已有的 `utils/graphics_utils.py::geom_transform_points()`，对每个 `vis_cams` 中的相机，使用 `cam.full_proj_transform` 将选中的世界空间中心投影到 NDC，再映射到像素坐标。
4. **过滤**：排除相机后方（z ≤ 0）、NaN、Inf、以及超出图像边界的点。
5. **绘制红点**：通过 `torchvision.transforms.ToPILImage()` 转换渲染图为 PIL Image，使用 `ImageDraw.ellipse()` 在投影位置画红色实心圆（radius=3）。
6. **保存**：同时写出原图和标红图到 `{model_path}/conf_interval_visualization/`。
7. **触发点**：在 `train.py` line 264 的调用处，传入 `camlist` 作为 `vis_cams`；由新 CLI 参数 `--visualize_conf` 控制是否启用。

受影响的文件：`scene/gaussian_model.py`（~100 LOC）、`train.py`（~5 LOC）、`arguments/__init__.py`（~10 LOC 参数声明）。不修改 CUDA 代码。

### Objective Evidence

- `/home/xzh/xzh/RFGS/scene/gaussian_model.py:444-542` — `densify_and_prune_Improved` 是 Conf 统计的消费端：`conf` 在 line 465 从 interval 累计量计算，`conf_mask` 在 line 476 生成，`final_mask` 在 line 500-508 由 `select_densification_candidates` 组装，`long_axis_split` 在 line 519 消费 mask。Line 510-518 的 8 行间隙是唯一确定的插入窗口，无副作用。
- `/home/xzh/xzh/RFGS/scene/gaussian_model.py:398-404` — `densification_postfix()` 在此间隙之后才重置 `xyz_gradient_vec_accum` 等 accumulator，保证快照时数据完整。
- `/home/xzh/xzh/RFGS/utils/graphics_utils.py:22-29` — `geom_transform_points()` 已实现世界到 NDC 的投影（齐次乘 + 透视除法），可直接复用。
- `/home/xzh/xzh/RFGS/scene/cameras.py:54-57` — Camera 暴露 `full_proj_transform`、`image_width`、`image_height`，满足投影所需的全部矩阵参数。
- `/home/xzh/xzh/RFGS/train.py:150` — 整个 densification 块已在 `torch.no_grad()` 下运行，可视化代码不会污染梯度。
- `/home/xzh/xzh/RFGS/train.py:180-197` — 已有的 Conf timing probe 是同一位置的内联测量 hook，为可视化 hook 建立了结构先例。
- `/home/xzh/xzh/RFGS/train.py:266-292` — CSV 统计导出在 `densify_and_prune_Improved` 返回后执行，确认了 densification 时 I/O 副作用的惯用模式。
- `/home/xzh/xzh/RFGS/render.py:34` — `torchvision.utils.save_image` 已被项目用于保存渲染图。
- `/home/xzh/xzh/RFGS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:207-210` 和 `auxiliary.h:41-44` — CUDA 正向投影和 `ndc2Pix` 逻辑可在 Python 中精确复制，无需修改 CUDA 代码。
- `/home/xzh/xzh/RFGS/validate_jacobian.py` — 已有的投影正确性验证脚本，其 `full_proj_transform` 构造模式可直接复用。

### Known Risks

- **函数签名变化**：`gaussian_model_alter.py` 中可能存在旧版 `densify_and_prune_Improved`，如果同时维护两份需要同步或确认 visualization 分支只用一份。
- **camlist 已被消费**：`train.py` 中 `camlist` 经过 `my_viewpoint_stack.pop()` 后仍然持有有效的 Camera 对象（tensor 在 GPU 上），但需注意 pop 后的语义。
- **渲染开销**：每次 densification 间隔（每 100 iter）为可视化多做一个 render pass（~1 次前向），total ~145 次额外渲染，仅在 `--visualize_conf` 开启时生效。
- **投影精度**：`geom_transform_points` 的 NDC→像素映射可能与 CUDA rasterizer 的 `ndc2Pix` 有 1-2 像素偏差。可通过对比 rasterizer 返回的 `gaussian_centers` 来验证。

## Alternative Directions Considered

### Alt-1: Standalone Visualization Module
- Gist: 将所有可视化逻辑抽取到新模块 `utils/conf_visualization.py` 中，遵循项目现有的 `utils/candidate_selector.py` 模式（该模块 664 行，封装完整子系统）。`train.py` 和 `gaussian_model.py` 仅添加薄调用点。`conf_mask` 需通过 `self._last_conf_mask` 实例属性暴露给外部。
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/utils/candidate_selector.py` (664 lines) 是精确的架构先例：新工具模块封装子系统，由 `scene/gaussian_model.py:16` 导入。
  - `/home/xzh/xzh/RFGS/train.py:16,21,24` 展示了 `from utils.xxx import yyy` 的惯用导入模式。
  - `/home/xzh/xzh/RFGS/utils/` 目录已有 9 个工具模块，添加第 10 个风格一致。
- Why not primary: 增加了文件数和模块边界复杂度；conf_mask 需要额外暴露（实例属性或返回值），不如 inline hook 直接在数据生产地消费数据简洁。

### Alt-2: Arguments-First Integration
- Gist: 先在 `arguments/__init__.py` 的 `OptimizationParams` 中添加所有 CLI 参数（`--visualize_conf`、`--conf_vis_iterations` 等），遵循 `candidate_stats_enabled` 和 `profile_components` 的布尔参数模式，再将参数传递到训练循环和 densification 逻辑中。
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/arguments/__init__.py:123-125` — 已有 `conf_min_views`、`conf_thr` 等 Conf 相关参数。
  - `/home/xzh/xzh/RFGS/arguments/__init__.py:164,167` — `candidate_stats_enabled` 和 `profile_components` 是精确的 flag-gating 模板。
- Why not primary: 参数先行会增加初始修改的文件数（arguments + train + gaussian_model），而 inline hook 可以在一个文件中完成核心逻辑后再参数化。

### Alt-3: Snapshot-then-Visualize Two-Phase
- Gist: 训练期间仅保存轻量 `.pt` 快照（`xyz`、`conf_mask`、`conf_score`、相机矩阵、渲染图），训练后通过独立脚本 `vis_conf_candidates.py` 异步渲染可视化图片。训练时开销极小（仅 `.cpu()` + `torch.save()`）。
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/train.py:267-292` — 已有的 `candidate_selection_stats.csv` 导出是精确的 per-interval 数据落盘先例。
  - `/home/xzh/xzh/RFGS/train.py:371-379` — `profiler_results.json` 展示了训练期导出结构化数据供下游消费的模式。
- Why not primary: 需求明确要求"在训练过程中生成可视化"，两阶段方案将可视化延迟到训练后，与需求中"在该 interval 候选 mask 生成后立即保存"的即时性要求存在语义张力。

### Alt-4: Projection-as-Utility First
- Gist: 优先构建通用 `project_gaussian_centers(xyz, full_proj_transform, W, H)` 工具函数，通过与 CUDA rasterizer 的 `gaussian_centers` 输出对比验证像素级对齐。可视化成为该工具的轻薄消费者。
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/scene/gaussian_model.py:544-619` — `_compute_ndc_vjp_world()` 已完整实现行向量约定的投影 Jacobian 回传，正向投影是同一数学链的更简单半程。
  - `/home/xzh/xzh/RFGS/gaussian_renderer/__init__.py:123` — rasterizer 已返回 `gaussian_centers`，可直接用于对比验证。
  - `/home/xzh/xzh/RFGS/validate_jacobian.py` — 已有的投影验证脚本，可直接扩展。
- Why not primary: 投影工具是基础设施而非核心交付物；inline hook 可以先实现基本投影，后续再提取为通用工具。此方向更适合作为 inline hook 完成后的重构步骤。

### Alt-5: Minimal-Path Correctness-First
- Gist: 先实现最简版本——硬编码一个 densification iteration（如 7000）、使用当前 `viewpoint_cam`、PIL `ImageDraw` 绘图、一个函数内完成所有逻辑。通过形状断言和边界检查优先验证正确性，配置参数延后添加。
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/train.py:27-28` — PIL (`ImageFilter`) 和 `torchvision.transforms` 已导入，无需新依赖。
  - `/home/xzh/xzh/RFGS/utils/taming_utils.py:9-13` — 已有 `ToPILImage()` + PIL 操作的先例。
- Why not primary: 硬编码迭代需要在后续立即重构为可配置版本，相当于把 inline hook 的正确性验证和参数化拆成两步。Inline hook 可以直接在正确位置实现带参数控制的版本，减少往返。

## Synthesis Notes

所有方向汇聚于同一个核心认知：可视化插入点必须在 `densify_and_prune_Improved()` 中 `final_mask` 形成后、`long_axis_split()` 调用前的 8 行间隙（gaussian_model.py:510-518）。各方向的差异在于代码组织风格而非功能正确性。

如果选择 Alt-1（独立模块），可以将 inline hook 的核心逻辑在首次实现后立即提取为 `utils/conf_visualization.py`，保留 `densify_and_prune_Improved` 中的 snapshot + call 薄层。如果选择 Alt-4（投影工具优先），可在 `utils/graphics_utils.py` 中扩展 `geom_transform_points` 增加 `ndc_to_pixel()` 工具，然后用 inline hook 消费。Alt-2（参数先行）和 Alt-5（最简路径）本质上是 inline hook 的前置步骤和后置迭代——可以在同一轮修改中依次完成：先加参数 → 实现 inline hook → 后续提取为独立模块。

建议实施路径：以 primary direction（inline hook）为骨架，吸收 Alt-4 的投影验证步骤（对比 rasterizer `gaussian_centers`），参考 Alt-2 的参数声明模式，最终达到与 Alt-1 同等整洁的模块边界——但始终以 `densify_and_prune_Improved` 内的插入点为唯一真实源。

## Synthesis Notes

所有方向汇聚于同一个核心认知：可视化插入点必须在 `densify_and_prune_Improved()` 中 `final_mask` 形成后、`long_axis_split()` 调用前的 8 行间隙（gaussian_model.py:510-518）。各方向的差异在于代码组织风格而非功能正确性。

建议实施路径：以 primary direction（inline hook）为骨架，吸收 Alt-4 的投影验证步骤（对比 rasterizer `gaussian_centers`），参考 Alt-2 的参数声明模式，最终达到与 Alt-1 同等整洁的模块边界——但始终以 `densify_and_prune_Improved` 内的插入点为唯一真实源。

--- Original Design Draft End ---
