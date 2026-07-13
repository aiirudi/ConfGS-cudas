# Standardized Pure-Render FPS Benchmark for 3D Gaussian Splatting

## Goal Description

在 visualization 分支上为 RFGS 项目增加规范的纯渲染 FPS 基准测试功能。该功能测量最终 Gaussian 模型执行 `gaussian_renderer.render()` 的 GPU 吞吐量（不含图片保存、CPU传输、指标计算、Conf可视化），用于论文中的渲染效率对比。FPS benchmark 入口为独立的 `bench_fps.py`（重构现有脚本），batch 测试通过 `test.py` 收集每个场景的 FPS 到 `metrics.json`。所有修改仅发生在 visualization 分支，不修改 CUDA kernel，不改变原始渲染结果。

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: `bench_fps.py` 使用 `torch.cuda.Event` 进行 batch GPU 计时，正确测量纯渲染吞吐量
  - Positive Tests (expected to PASS):
    - 运行 `python bench_fps.py -m output/<scene>` 后终端打印 FPS、平均延迟(ms/frame)、总帧数、Gaussian数量、GPU名称、图像分辨率
    - 同一场景同一参数重复运行 3 次，FPS 波动在 ±5% 以内
    - 计时区间仅包含 `gaussian_renderer.render()` 调用，不含 warmup 帧
  - Negative Tests (expected to FAIL):
    - 若 CUDA 不可用，抛出 `RuntimeError("CUDA is required for FPS benchmarking")` 而非静默退回 CPU
    - 若 `fps_repeat < 1`，抛出 `ValueError`
    - 若 `fps_warmup < 0`，抛出 `ValueError`
    - 若 `fps_max_views == 0`，抛出 `ValueError`
    - 若 cameras 列表为空，抛出 `ValueError("No cameras available for FPS benchmark")`
    - 若 elapsed time 为 0，抛出 `RuntimeError("Elapsed time is zero")`

- AC-2: 正式计时前执行 GPU 预热，预热帧不计入 FPS 统计
  - Positive Tests (expected to PASS):
    - 预热帧数由 `--fps_warmup`（默认20）控制
    - 预热阶段使用 `torch.no_grad()`，渲染 `cameras[i % len(cameras)]`
    - 预热结束后调用 `torch.cuda.synchronize()` 再开始正式计时
  - Negative Tests (expected to FAIL):
    - 预热帧的渲染时间不出现在 FPS 计算中

- AC-3: FPS benchmark 结果保存为 JSON 文件
  - Positive Tests (expected to PASS):
    - 默认保存到 `<model_path>/fps_benchmark.json`
    - `--fps_output <path>` 可覆盖保存路径
    - JSON 包含：`split`, `iteration`, `warmup_frames`, `repeat`, `camera_count`, `total_frames`, `elapsed_seconds`, `average_ms_per_frame`, `fps`, `gaussian_count`, `device`, `timing_method`, `image_width`, `image_height`, `cuda_version`, `pytorch_version`
    - JSON 的写入发生在 FPS 计时结束之后（不在计时间隔内）
  - Negative Tests (expected to FAIL):
    - 计时过程中不执行任何 `json.dump()` 或文件 I/O

- AC-4: `--benchmark_fps` 关闭时，`bench_fps.py` 的原有行为完全不变
  - Positive Tests (expected to PASS):
    - `python bench_fps.py -m output/<scene>`（不加任何新参数）产生与修改前相同的终端输出
    - `cal_fps.sh` 脚本无需修改即可正常运行
  - Negative Tests (expected to FAIL):
    - 不传 `--benchmark_fps` 时不创建 `fps_benchmark.json`

- AC-5: `test.py` 批量测试时收集每个场景的 FPS 指标到 `metrics.json`
  - Positive Tests (expected to PASS):
    - 每个场景处理完后，`ALL_METRICS[data]` 包含 `fps` 和 `fps_latency_ms` 键
    - `metrics.json` 在每场景完成后增量写入（保持现有 crash-resilience 模式）
    - `test.py` 使用 `subprocess.run(..., check=True)` 调用 `bench_fps.py`
  - Negative Tests (expected to FAIL):
    - 若 `fps_benchmark.json` 不存在，打印 WARN 但不中断 batch 流程
    - 不读取属于其他场景/iteration 的过期 `fps_benchmark.json`

- AC-6: benchmark 过程中不修改 Gaussian 参数，不影响渲染结果
  - Positive Tests (expected to PASS):
    - benchmark 在 `torch.no_grad()` 下运行
    - benchmark 前后 `gaussians.get_xyz` 的 shape 和值完全相同
  - Negative Tests (expected to FAIL):
    - 不执行 backward、optimizer.step、densification、clone、split、prune、opacity reset

- AC-7: 不修改 CUDA kernel，不需要重新编译 CUDA 扩展
  - Positive Tests (expected to PASS):
    - `submodules/diff-gaussian-rasterization/` 下所有文件未被修改
    - `gaussian_renderer/__init__.py` 中的 `render()` 函数签名和返回结构未被修改
  - Negative Tests (expected to FAIL):
    - 不需要运行 `pip install` 或 `python setup.py install` 即可使用 benchmark

- AC-8: FPS 只统计纯 `gaussian_renderer.render()` 调用
  - Positive Tests (expected to PASS):
    - 计时间隔内不包含：图片保存、tensor.cpu()、tensor.numpy()、PIL/OpenCV 处理、PSNR/SSIM/LPIPS 计算、Conf 可视化、日志写盘
  - Negative Tests (expected to FAIL):
    - 计时间隔内不包含任何 `torch.cuda.synchronize()` 调用（只在 start 前和 end 后各调用一次）

- AC-9: 命令行参数正确注册并可通过 CLI 控制
  - Positive Tests (expected to PASS):
    - `--fps_split`（默认 `test`，可选 `train`），若选 test 但无 test 相机则报错
    - `--fps_warmup`（默认 `20`，int）
    - `--fps_repeat`（默认 `3`，int，≥1）
    - `--fps_max_views`（默认 `-1` 表全部，≥1 时用 `fps_camera_seed` 确定性抽样）
    - `--fps_camera_seed`（默认 `0`，int）
    - `--fps_output`（默认 `None` → `<model_path>/fps_benchmark.json`）
    - `--fps_timing_mode`（默认 `batch`，可选 `per_frame`）
    - 以上参数在 `bench_fps.py` 的 ArgumentParser 中注册
  - Negative Tests (expected to FAIL):
    - 传入未知 `--fps_split` 值时 argparse 报错
    - 传入 `--fps_repeat 0` 时 argparse 报错

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

实现包含：
- `utils/fps_benchmark.py` 可复用模块，导出 `benchmark_render_fps(cameras, gaussians, pipeline, background, ...)` 函数和 `save_fps_json(result, path)` 辅助函数
- `bench_fps.py` 重构为使用 `utils/fps_benchmark.py` 的 CLI 封装，保留全部现有统计功能（per-frame 模式含 mean/median/percentile/stable FPS），新增 batch timing 模式、JSON 输出和全部新 CLI 参数
- `test.py` 在每个场景的 `render.py` + `metrics.py` 之后调用 `bench_fps.py`，读取 `fps_benchmark.json` 合并到 `ALL_METRICS`
- batch timing 模式下支持跨 repeat 的 mean/std 报告
- JSON schema 包含版本号、camera 元数据、per-repeat 明细
- `test.py` 使用 `subprocess.run(check=True)` 替代 `os.system()`，检测子进程失败
- 对无效参数（空 cameras、负 warmup、零 repeat、零 max_views）的参数验证函数

### Lower Bound (Minimum Acceptable Scope)

实现包含：
- `bench_fps.py` 增加 batch timing 模式（单 CUDA event pair）、`--fps_warmup`、`--fps_repeat`、`--fps_split`、`--fps_max_views`、`--fps_camera_seed`、`--fps_output`、`--fps_timing_mode` 参数
- batch 计时结束后写入 `fps_benchmark.json`（含 fps、latency、gaussian_count、device、resolution）
- 终端打印 FPS、平均延迟、Gaussian 数量、GPU 名称、分辨率
- `test.py` 在 `render.py` 后增加 `os.system('python bench_fps.py ...')` 调用，从 `fps_benchmark.json` 读取 fps 和 latency 写入 `metrics.json`
- 参数验证：空 cameras 报错、无 CUDA 报错、repeat<1 报错
- 预热阶段循环使用 cameras，结束后 `torch.cuda.synchronize()`
- `torch.no_grad()` 包裹全部 benchmark 代码
- 零 CUDA kernel 修改

### Allowed Choices

- 可以：在 `bench_fps.py` 内新增辅助函数（不强制拆分到 `utils/`）
- 可以：batch timing 使用单个 start/end event pair（遵循现有 commented-out 实现模式）
- 可以：per-frame 模式复用现有 active 实现（每帧一个 event pair），通过 `--fps_timing_mode per_frame` 启用
- 可以：`test.py` 中读取 `fps_benchmark.json` 使用 `os.path.exists` guard（与现有 `profiler_results.json` 模式一致）
- 必须：使用 `torch.cuda.Event(enable_timing=True)` 进行 GPU 计时
- 必须：`--fps_split test` 且 test 相机为空时直接报错（不 fallback）
- 禁止：修改 `gaussian_renderer/__init__.py` 中的 `render()` 函数
- 禁止：修改 `submodules/` 下任何文件
- 禁止：修改 `render.py`

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

```
# utils/fps_benchmark.py (概念结构)

@torch.no_grad()
def benchmark_render_fps(cameras, gaussians, pipeline, background,
                         warmup=20, repeat=3, timing_mode='batch'):
    # 1. 参数验证
    if len(cameras) == 0:
        raise ValueError("No cameras available")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    # 2. 动态检测 render() 的额外 kwargs
    render_kwargs = detect_render_kwargs(gaussians, pipeline)

    # 3. 预热
    for i in range(warmup):
        _ = render(cameras[i % len(cameras)], gaussians, pipeline,
                    background, **render_kwargs)["render"]
    torch.cuda.synchronize()

    # 4. 正式计时 (batch mode)
    total_frames = len(cameras) * repeat
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(repeat):
        for cam in cameras:
            render_pkg = render(cam, gaussians, pipeline, background, **render_kwargs)
            _ = render_pkg["render"]  # 保持引用防止提前释放
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_s = elapsed_ms / 1000.0
    fps = total_frames / elapsed_s
    avg_ms = elapsed_ms / total_frames

    return {
        'fps': fps, 'avg_ms': avg_ms,
        'total_frames': total_frames, 'elapsed_seconds': elapsed_s,
        'gaussian_count': gaussians.get_xyz.shape[0],
        'device': torch.cuda.get_device_name(0)
    }
```

### Relevant References

- `bench_fps.py` — 现有 FPS benchmark，含两种 timing 实现（active per-frame lines 1-243，commented-out batch lines 245-416）；需重构以整合新参数和 JSON 输出
- `gaussian_renderer/__init__.py` — `render()` 函数签名和 14-key 返回 dict；benchmark 只访问 `["render"]` 键
- `test.py` — batch 测试驱动，`os.system()` 调用 train/render/metrics，读取 `profiler_results.json` 和 `results.json` 到 `ALL_METRICS` → `metrics.json`
- `arguments/__init__.py` — `get_combined_args()` 合并 CLI 和 `cfg_args`；`ParamGroup` 类用于结构化参数注册（FPS 参数不放入 ParamGroup，直接在 bench_fps.py parser 中注册）
- `scene/__init__.py` — `Scene` 构造和 `getTrainCameras()`/`getTestCameras()`；`shuffle=False` 保证 camera 顺序确定性
- `scene/cameras.py` — `Camera` 类含 `image_width`、`image_height` 属性
- `train.py` — 多处 `torch.cuda.Event` profiling 先例（lines 53-54, 182-184, 220-221）；`profiler_results.json` JSON 输出模式（lines 400-430）
- `cal_fps.sh` — 现有 shell wrapper，直接调用 `bench_fps.py`；需确认修改后兼容

## Dependencies and Sequence

### Milestones

1. **Core Benchmark Module**: 创建/重构 FPS 计时核心逻辑
   - Phase A: 在 `bench_fps.py` 中实现 batch timing 函数（单 CUDA event pair，warmup，`torch.no_grad()`）
   - Phase B: 新增 CLI 参数注册（`--fps_warmup`, `--fps_repeat`, `--fps_split`, `--fps_max_views`, `--fps_camera_seed`, `--fps_output`, `--fps_timing_mode`）
   - Phase C: 实现 JSON 输出（`fps_benchmark.json`）和终端格式化打印
   - Phase D: 参数验证（空 cameras、CUDA 检查、数值边界）

2. **Backward Compatibility**: 确保原有功能不受影响
   - Step 1: 默认行为（不加任何新参数）与修改前完全一致
   - Step 2: `cal_fps.sh` 无需修改即可运行
   - Step 3: 现有 per-frame 统计模式通过 `--fps_timing_mode per_frame` 保持可用

3. **test.py Integration**: 将 FPS 指标集成到 batch 流程
   - Step 1: 在每个场景的 `render.py` + `metrics.py` 之后调用 `bench_fps.py`
   - Step 2: 读取 `fps_benchmark.json`，提取 `fps` 和 `average_ms_per_frame` 到 `ALL_METRICS`
   - Step 3: 遵循现有 error handling 模式（文件缺失时 WARN 不中断）
   - Step 4: 确认 batch 模式下 `metrics.json` 最终包含所有场景的 FPS 数据

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | 在 `bench_fps.py` 中实现 batch timing 函数：单 CUDA event pair、warmup、`torch.no_grad()`、参数验证 | AC-1, AC-2, AC-6 | coding | - |
| task2 | 在 `bench_fps.py` 中注册全部新 CLI 参数（`--fps_warmup` 等 8 个）并实现 camera 选择逻辑（含空 split 报错、`fps_max_views` 确定性抽样） | AC-4, AC-9 | coding | task1 |
| task3 | 实现 JSON 输出（`fps_benchmark.json` schema）和终端格式化打印 | AC-3 | coding | task1 |
| task4 | 重构 `bench_fps.py` 将现有 per-frame 模式作为 `--fps_timing_mode per_frame` 保留，batch 模式为默认 | AC-4 | coding | task1, task2 |
| task5 | 在 `test.py` 中集成 FPS 收集：调用 `bench_fps.py`、读取 JSON、合并到 `ALL_METRICS` | AC-5 | coding | task3 |
| task6 | 端到端验证：用 `--fps_timing_mode batch` 和 `per_frame` 分别运行，确认两种模式的输出格式、JSON 完整性和原有功能兼容性 | AC-1, AC-4, AC-8 | analyze | task4, task5 |

## Claude-Codex Deliberation

### Agreements

- Batch CUDA Event timing（单 start/end event pair）是论文 FPS 的正确默认方式，与 `bench_fps.py` commented-out 实现一致
- FPS 参数注册在 `bench_fps.py` parser 中（而非 `ParamGroup` 子类），避免 `get_combined_args()` 的非 None 默认值覆盖 `cfg_args` 问题
- `--fps_split test` 且 test 相机为空时直接报错（不 fallback），保证论文 FPS 比较的公平性
- 不修改 `render.py`，FPS benchmark 入口为独立的 `bench_fps.py`
- `test.py` 默认执行完整指标（render + metrics + fps），FPS 为增量功能
- 零 CUDA kernel 修改，`gaussian_renderer.render()` 签名和返回结构不变

### Resolved Disagreements

- **Timing mode 控制**: Codex 指出 plan 提到 per-frame 模式但未提供 CLI 参数 → 新增 `--fps_timing_mode {batch,per_frame}` 参数，batch 为默认
- **render() kwargs 检测**: Codex 指出当前 `gaussian_renderer.render()` 不接受 `use_trained_exp`/`separate_sh` → 移除不必要的 kwargs 检测，直接传标准 4 参数
- **test.py 子进程调用**: Codex 建议用 `subprocess.run(check=True)` 替代 `os.system()` → 采纳，增强错误检测
- **过期 JSON 防护**: Codex 要求在 benchmark 前删除或覆盖 `fps_benchmark.json` → 采纳，benchmark 开始时先删除旧文件
- **JSON key 命名**: `average_ms_per_frame` vs `fps_latency_ms` → 统一为 `average_ms_per_frame`（JSON 中）和 `fps`（test.py 收集中），`test.py` 直接读取 JSON 中的原始 key

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- DEC-1: 空 test split 行为
  - Claude Position: 报错（保证论文比较公平性）
  - Codex Position: 报错
  - Tradeoff Summary: 报错保证用户不会无意中用 train 相机做 FPS 比较；需要用户在无 test 相机的场景显式传 `--fps_split train`
  - Decision Status: `报错（fail on empty）`

- DEC-2: test.py 默认模式
  - Claude Position: 完整 render + metrics + FPS
  - Codex Position: 完整 render + metrics + FPS
  - Tradeoff Summary: FPS 作为增量功能加入现有流程，不改变默认行为；`--fps_only` 可选
  - Decision Status: `完整指标 + FPS`

- DEC-3: FPS benchmark 入口命令
  - Claude Position: `render.py --benchmark_fps`
  - Codex Position: `bench_fps.py` 独立
  - Tradeoff Summary: 独立 `bench_fps.py` 保持关注点分离，render.py 不增加复杂度，cal_fps.sh 无需修改
  - Decision Status: `bench_fps.py 独立入口`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead
- 代码注释使用中文（与项目现有风格一致）
- 新增函数需添加 docstring 说明用途和参数
- 遵循现有 `bench_fps.py` 的代码风格（print 分隔线、Phase 标记、对齐输出）

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# Reusable CUDA-Event FPS Benchmark Module for 3D Gaussian Splatting Rendering

## Original Idea

你是一名熟悉3D Gaussian Splatting、PyTorch、CUDA异步执行和GPU性能测试的代码工程师。

请基于以下项目进行代码分析和修改：

仓库：aiirudi/RFGS
工作分支：visualization

visualization分支已经由用户创建。本次所有修改必须只在visualization分支完成，不要创建新分支，不要修改conf-refine、main或master分支。

本次任务是在模型渲染过程中增加规范的FPS测试功能，用于论文中的渲染效率对比。

FPS必须统计最终Gaussian模型执行纯render函数的速度，而不是GUI刷新速度，也不能把图片保存、指标计算或Conf红点可视化计入渲染时间。

========================
一、首先分析现有渲染代码
========================

请先完整检查项目代码，准确定位：

1. 最终模型渲染入口，例如render.py、render_sets、render_set或同等函数；
2. 单个相机视角调用Gaussian renderer的位置；
3. 项目实际使用的render函数；
4. 训练视角和测试视角的获取方式；
5. Gaussian模型checkpoint加载方式；
6. background、pipeline和camera参数的来源；
7. 图片保存代码；
8. PSNR、SSIM、LPIPS等指标计算代码；
9. Conf选点投影和红点绘制代码；
10. 当前是否已有FPS、render_time或CUDA计时代码。

请报告真实文件名、函数名和关键变量名，不要根据常见3DGS结构直接猜测。

========================
二、FPS的定义
========================

FPS定义为：

FPS = 实际计时渲染帧数 / 纯渲染总耗时

同时输出平均单帧延迟：

Average latency (ms/frame)
= 纯渲染总耗时 / 实际计时渲染帧数 × 1000

FPS测试只允许统计以下调用：

render(
    viewpoint_camera,
    gaussians,
    pipeline,
    background
)

或者项目中等价的实际Gaussian渲染函数。

不得把以下操作计入FPS：

1. 读取图片；
2. 从磁盘加载相机；
3. 加载checkpoint；
4. 创建输出目录；
5. 保存PNG或JPG；
6. tensor.cpu()；
7. tensor.numpy()；
8. PIL或OpenCV处理；
9. Conf Gaussian中心投影；
10. 红点绘制；
11. PSNR、SSIM、LPIPS计算；
12. 日志写盘；
13. GUI刷新；
14. backward；
15. optimizer.step；
16. densification或pruning。

========================
三、必须正确处理CUDA异步执行
========================

CUDA kernel默认异步执行，因此禁止直接使用以下不准确方式：

start = time.time()
render(...)
elapsed = time.time() - start

因为CPU计时可能只统计到CUDA任务提交时间。

优先使用torch.cuda.Event进行GPU计时。

推荐方式：

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()

for camera in measured_cameras:
    render_pkg = render(
        camera,
        gaussians,
        pipeline,
        background
    )

end_event.record()
torch.cuda.synchronize()

elapsed_ms = start_event.elapsed_time(end_event)

也可以使用time.perf_counter，但必须在计时开始前和结束后调用：

torch.cuda.synchronize()

优先采用torch.cuda.Event，因为它更适合测量GPU渲染耗时。

========================
四、增加预热阶段
========================

正式计时前必须进行GPU预热，避免以下因素影响结果：

1. 第一次CUDA上下文初始化；
2. kernel首次启动；
3.显存分配；
4.缓存建立；
5.第一次render额外开销。

新增参数：

--fps_warmup
    FPS测试前的预热帧数，默认20。

预热方式：

with torch.no_grad():
    for i in range(fps_warmup):
        camera = cameras[i % len(cameras)]
        _ = render(
            camera,
            gaussians,
            pipeline,
            background
        )

预热结束后调用：

torch.cuda.synchronize()

预热帧不得计入FPS统计。

========================
五、正式计时方式
========================

新增独立函数，例如：

benchmark_render_fps(
    cameras,
    gaussians,
    pipeline,
    background,
    warmup,
    repeat
)

建议接口：

@torch.no_grad()
def benchmark_render_fps(
    cameras,
    gaussians,
    pipeline,
    background,
    warmup=20,
    repeat=3,
):
    """
    Benchmark pure Gaussian rendering speed.

    Returns:
        fps: float
        avg_ms: float
        total_frames: int
        elapsed_seconds: float
    """

正式计时应遍历完整的指定相机集合。

如果repeat大于1，则重复渲染整个相机列表：

total_frames = len(cameras) * repeat

禁止只测量一帧，因为单帧结果波动较大。

推荐默认参数：

--fps_repeat 3

正式计时的概念结构：

start_event.record()

for _ in range(repeat):
    for camera in cameras:
        _ = render(
            camera,
            gaussians,
            pipeline,
            background
        )

end_event.record()
torch.cuda.synchronize()

elapsed_ms = start_event.elapsed_time(end_event)
elapsed_seconds = elapsed_ms / 1000.0

fps = total_frames / elapsed_seconds
avg_ms = elapsed_ms / total_frames

========================
六、避免计时过程中的额外同步
========================

正式计时循环内部不要执行：

rendered_image.cpu()
rendered_image.numpy()
torch.cuda.synchronize()
save_image(...)
print(...)
compute_metrics(...)

也不要在每一帧之后调用torch.cuda.synchronize()，否则会改变正常连续渲染的吞吐量。

只需要：

1. 正式计时开始前完成同步；
2. 记录开始事件；
3. 连续执行全部render调用；
4. 记录结束事件；
5. 最后统一同步。

如果需要统计逐帧延迟，应提供单独模式，不能与默认吞吐FPS混在一起。

默认论文FPS使用批量连续渲染的总耗时计算。

========================
七、渲染结果的生命周期
========================

必须确保render调用确实完成，不能被Python优化或提前释放导致计时异常。

可以保留每次render结果的引用，或者读取render_pkg["render"]但不要传到CPU。

例如：

render_pkg = render(...)
last_render = render_pkg["render"]

不要在正式计时循环里保存图片。

如果担心显存引用累计，只保留最后一次结果，不要把所有render结果加入列表。

========================
八、增加命令行参数
========================

请增加以下命令行参数：

--benchmark_fps
    是否执行FPS测试，默认False。

--fps_split
    指定使用train或test相机，默认test。

--fps_warmup
    预热帧数，默认20。

--fps_repeat
    重复完整相机集合的次数，默认3。

--fps_max_views
    最多使用多少个相机视角，默认-1，表示使用全部视角。

--fps_camera_seed
    当需要抽样视角时使用的固定随机种子，默认0。

--fps_output
    FPS结果JSON的保存路径；
    如果未指定，默认保存到：
    <model_path>/fps_benchmark.json

--fps_only
    只运行FPS benchmark，不保存渲染图、不计算图像质量指标，默认False。

使用示例：

python render.py \
    -m <model_path> \
    --iteration 30000 \
    --benchmark_fps \
    --fps_split test \
    --fps_warmup 20 \
    --fps_repeat 3 \
    --fps_only

========================
九、与现有渲染流程的关系
========================

优先在最终模型加载完成、相机列表准备完成之后执行FPS测试。

推荐结构：

gaussians = load_final_model(...)
cameras = get_test_cameras(...)

if args.benchmark_fps:
    fps_result = benchmark_render_fps(
        cameras=cameras,
        gaussians=gaussians,
        pipeline=pipeline,
        background=background,
        warmup=args.fps_warmup,
        repeat=args.fps_repeat,
    )

if not args.fps_only:
    run_original_render_and_save_pipeline(...)

当--fps_only开启时：

1. 只加载模型；
2. 只准备相机；
3. 执行预热；
4. 执行纯render计时；
5. 输出和保存FPS结果；
6. 不保存渲染图；
7. 不计算PSNR、SSIM、LPIPS；
8. 不运行Conf选点可视化。

当--fps_only关闭时，可以在FPS benchmark结束后继续执行原有渲染和图片保存流程，但这些后续操作不得计入FPS。

========================
十、Conf与FPS的关系
========================

Conf是训练和densification阶段的候选Gaussian选择方法。

最终模型渲染时通常不再计算Conf，因此：

1. Conf计算时间不得计入最终Rendering FPS；
2. Conf中心投影和红点绘制不得计入FPS；
3. Conf可视化图片保存不得计入FPS；
4. FPS只反映最终Gaussian模型的纯渲染速度；
5. 如果Conf使最终Gaussian数量增加，FPS可能因最终点数增加而下降，这是正常现象。

请在代码注释和最终说明中明确区分：

- Rendering FPS；
- Conf training overhead；
- Conf visualization overhead。

========================
十一、输出内容
========================

终端必须打印：

================ FPS Benchmark ================
Split: test
Warmup frames: 20
Measured views: <camera_count>
Repeat: 3
Total measured frames: <total_frames>
Total pure render time: <seconds> s
Average render latency: <avg_ms> ms/frame
Rendering FPS: <fps>
Gaussian count: <gaussian_count>
CUDA device: <gpu_name>
Image resolution: <width>x<height>
================================================

如果相机图像分辨率不一致，需要输出：

- 最小分辨率；
- 最大分辨率；
- 或者明确说明测试集包含不同分辨率。

不要只输出FPS而不输出平均单帧耗时。

========================
十二、保存JSON结果
========================

将结果保存为JSON，例如：

{
    "split": "test",
    "iteration": 30000,
    "warmup_frames": 20,
    "camera_count": 200,
    "repeat": 3,
    "total_frames": 600,
    "elapsed_seconds": 4.8,
    "average_ms_per_frame": 8.0,
    "fps": 125.0,
    "gaussian_count": 1234567,
    "device": "NVIDIA RTX 3090",
    "timing_method": "torch.cuda.Event",
    "included_operations": [
        "gaussian_renderer.render"
    ],
    "excluded_operations": [
        "image_save",
        "cpu_transfer",
        "metric_computation",
        "conf_visualization",
        "gui_refresh"
    ]
}

保存JSON本身必须发生在FPS计时结束之后。

========================
十三、异常处理
========================

需要处理以下情况：

1. cameras为空：
   抛出明确错误，不得除以0。

2. 没有CUDA：
   如果项目本身要求CUDA，直接给出明确错误；
   不要静默退回CPU后仍将结果称为GPU FPS。

3. fps_repeat小于1：
   给出参数错误。

4. fps_warmup小于0：
   给出参数错误。

5. fps_max_views为0：
   给出参数错误。

6. elapsed time为0：
   给出明确异常，不得返回inf。

7. 相机数量少于warmup帧数：
   循环使用相机进行预热。

========================
十四、不要修改CUDA
========================

本任务只是在Python层对现有render调用计时。

原则上不得修改：

- CUDA rasterizer；
- diff-gaussian-rasterization；
- forward kernel；
- backward kernel；
- render返回接口。

不需要重新编译CUDA扩展。

========================
十五、不得改变模型渲染结果
========================

FPS benchmark必须是只读测试。

使用：

with torch.no_grad():

禁止执行：

- backward；
- optimizer.step；
- densification；
- clone；
- split；
- prune；
- opacity reset；
-参数更新。

benchmark前后Gaussian参数必须保持不变。

关闭--benchmark_fps时，原有代码行为必须完全不变。

========================
十六、公平对比要求
========================

为了用于论文方法间FPS对比，请确保不同方法使用完全相同的：

1. GPU；
2. CUDA环境；
3. 图像分辨率；
4. 相机集合；
5. 相机顺序；
6. warmup帧数；
7. repeat次数；
8. render参数；
9. background设置；
10. 是否开启antialiasing；
11. checkpoint迭代数。

不得在不同方法之间使用不同测试视角数量。

请在JSON中保存这些关键条件，确保结果可复现。

========================
十七、验收标准
========================

修改后必须满足：

1. 所有代码修改仅发生在visualization分支；

2. 不创建新分支；

3. 开启--benchmark_fps后可以正常加载最终模型并测试；

4. FPS只统计纯render调用；

5. 正式计时前执行预热；

6. 使用torch.cuda.Event或正确的CUDA同步方式；

7. 预热帧不计入FPS；

8. 图片保存、CPU传输、指标计算和Conf可视化不计入FPS；

9. 输出FPS和平均ms/frame；

10. 输出实际计时帧数和总渲染时间；

11. 输出最终Gaussian数量；

12. 输出GPU名称；

13. FPS结果保存为JSON；

14. 开启--fps_only时不保存图片、不计算指标、不运行Conf可视化；

15. benchmark过程中不修改Gaussian参数；

16. 不修改CUDA，不需要重新编译扩展；

17. 关闭--benchmark_fps时，原渲染流程和结果保持不变；

18. 相同参数重复运行时，FPS结果应处于合理波动范围；

19. 不得使用GUI显示帧率代替纯render FPS；

20. 不得把Conf可视化耗时计入Rendering FPS。

========================
十八、最终回答要求
========================

完成代码分析和修改后，请按以下内容回答：

1. 当前渲染入口文件和函数；
2. 实际render调用位置；
3. FPS benchmark插入位置；
4. 采用torch.cuda.Event还是time.perf_counter；
5. 为什么该计时方式能够正确处理CUDA异步执行；
6. 预热如何实现；
7. 哪些操作被排除在FPS之外；
8. 修改了哪些文件；
9. 给出完整unified diff；
10. 给出运行命令；
11. 给出终端输出示例；
12. 给出JSON保存路径；
13. 明确说明是否修改CUDA；
14. 明确说明是否影响原始渲染结果；
15. 明确说明全部修改是否只发生在visualization分支。

再次强调：

本任务测量的是最终Gaussian模型的纯渲染FPS，不是GUI刷新FPS，不是包含图片保存的端到端速度，也不是Conf可视化生成速度。同时将通过 @test.py 进行批量训练统计指标时，为每个场景加入fps 的指标保存到metrics.json中也就是保存指标的文件

## Primary Direction: Standalone fps module + test.py integration

### Rationale

Create a reusable module under `utils/` that both `render.py` and `test.py` import, following the existing `utils/taming_utils.py` pattern, giving clean separation of concerns with no code duplication across the two call sites.

### Approach Summary

Create `utils/fps_benchmark.py` as a reusable module that encapsulates all CUDA Event-based timing logic currently scattered in `bench_fps.py`. The module exports a primary function `benchmark_render_fps(cameras, gaussians, pipeline, background, warmup=20, repeat=3)` taking explicit parameters for all configurable knobs. Two call sites consume it:

1. **`render.py`**: When `--benchmark_fps` is passed, after (or instead of) the normal render loop, call the module, print the summary to stdout, and write `fps_benchmark.json` into the model directory. Support `--fps_only` to skip image saving entirely. The existing `--skip_train` / `--skip_test` logic is untouched; `--benchmark_fps` is an orthogonal addition. All new flags (`--benchmark_fps`, `--fps_split`, `--fps_warmup`, `--fps_repeat`, `--fps_max_views`, `--fps_camera_seed`, `--fps_output`, `--fps_only`) are registered in the `render.py` argument parser using the same `parser.add_argument()` pattern already present.

2. **`test.py`**: After each scene's `render.py` + `metrics.py` calls, read `fps_benchmark.json` from the model directory and merge into `scene_metrics` under keys like `fps`, `fps_latency_ms`, `n_gaussians`. These are written into `ALL_METRICS[data]` and persisted to `metrics.json` using the existing incremental write pattern. Alternatively, call `bench_fps.py` as a subprocess via `os.system()` and read its JSON output.

The core mechanism inside `fps_benchmark.py`: `torch.cuda.Event(enable_timing=True)` paired with `torch.cuda.synchronize()` before `elapsed_time()`, warmup frames rendered first, FPS computed as `num_timed_frames / total_elapsed_seconds`. Only `gaussian_renderer.render()` calls are timed (no image save, no CPU transfer). The function runs under `torch.no_grad()`. The existing standalone `bench_fps.py` is preserved and optionally refactored to a thin CLI wrapper around the new module.

Zero CUDA changes, zero recompilation. `--benchmark_fps` defaults to `False`, so the original render path is completely unchanged when the flag is absent.

### Objective Evidence

- `/home/xzh/xzh/RFGS/bench_fps.py` (417 lines): Prior art containing two separate implementations of CUDA Event-based FPS measurement. Lines 18-243 are the active block with warmup (lines 86-96), frame-by-frame timing (lines 112-120 with `torch.cuda.Event(enable_timing=True)` + `record()` + `synchronize()` + `elapsed_time()`), statistical analysis (lines 129-156), and per-view analysis (lines 212-239). Lines 245-416 are a commented-out second implementation using batch timing across all frames. This file is not importable as a module (it is a `__main__` script with its own `main()`) and saves no JSON output. It proves the `torch.cuda.Event` approach already works in this exact environment.
- `/home/xzh/xzh/RFGS/render.py` (65 lines): The primary render entry point. `render_sets()` (line 37) calls `gaussian_renderer.render(view, gaussians, pipeline, background)["render"]` inside `torch.no_grad()`. Has no FPS capability. Uses `from utils.general_utils import safe_state` (absolute import from utils package).
- `/home/xzh/xzh/RFGS/test.py` (108 lines): Batch test driver. Lines 62-106 show the pattern of collecting per-scene metrics into `ALL_METRICS[data] = scene_metrics`, then writing to `metrics.json` with `json.dump(ALL_METRICS, fp, indent=True)` after each scene. Lines 71-98 demonstrate reading from `profiler_results.json` and `results.json` as subprocess outputs. Has zero FPS-related logic.
- `/home/xzh/xzh/RFGS/metrics.py` (103 lines): Image quality evaluation. Lines 88-91 show JSON output pattern: `json.dump(full_dict[scene_dir], fp, indent=True)` to `results.json`.
- `/home/xzh/xzh/RFGS/arguments/__init__.py` (203 lines): `get_combined_args()` (lines 183-203) merges CLI args with `<model_path>/cfg_args`. The `ParamGroup` class (lines 21-49) with its `extract()` method is the standard way to register and resolve CLI arguments. New `--fps_*` flags are registered via `parser.add_argument()` in render.py's `__main__` block, not inside `ParamGroup`.
- `/home/xzh/xzh/RFGS/utils/taming_utils.py` (117 lines): Example of a `utils/` utility module imported by `train.py` using `from utils.taming_utils import ...`. Demonstrates the pattern of putting shared logic in utils and importing it from multiple call sites.
- `/home/xzh/xzh/RFGS/gaussian_renderer/__init__.py` (129 lines): The `render()` function signature at line 18: `render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None, pixel_weights=None)` returns dict with key `"render"`.
- `/home/xzh/xzh/RFGS/train.py` lines 53-54, 182-184, 219-221, 239-241: Additional prior art for `torch.cuda.Event(enable_timing=True)` usage inside the training loop for profiling component times. Lines 400-430 show `json.dump(prof_data, fp, indent=True)` writing `profiler_results.json`.
- `/home/xzh/xzh/RFGS/cal_fps.sh` (8 lines): Thin shell wrapper calling `bench_fps.py`, confirming the existing benchmark is an established workflow.
- `/home/xzh/xzh/RFGS/scene/__init__.py` lines 89-93: `Scene` class provides `getTrainCameras()` and `getTestCameras()`, confirming the view set selection mechanism needed for `--fps_split`.
- `/home/xzh/xzh/RFGS/scene/cameras.py` lines 40-41: `Camera` class has `image_width` and `image_height` attributes for resolution reporting.

### Known Risks

- The existing `bench_fps.py` is 417 lines with two variant implementations; the refactored module must capture both behavioral variants or pick one canonical approach. The commented-out second implementation (lines 245-416) uses batch timing across all frames with single start/end events — this is actually closer to the idea's requested "paper FPS" mode.
- `test.py` currently invokes `render.py` via `os.system()` as a subprocess. Integrating FPS collection requires either: (a) adding an `os.system('python bench_fps.py ...')` call (simplest, follows existing pattern), (b) switching to direct Python imports (cleaner but breaks the subprocess isolation pattern), or (c) having `render.py` write `fps_benchmark.json` and `test.py` read it after the subprocess exits. Option (c) is most consistent with how `test.py` already reads `profiler_results.json` and `results.json`.
- The `gaussian_renderer.render()` function has optional kwargs (`use_trained_exp`, `separate_sh`) that vary by codebase version; the module uses `inspect.signature(render)` to detect them (as `bench_fps.py` lines 58-65 do), but this detection could silently break if the render function signature changes.
- `torch.cuda.Event` timing requires CUDA to be available; the module must fail with a clear error (not silently CPU-fallback) on CPU-only environments.
- The existing `bench_fps.py` has no JSON output format; the new module must define a schema for `fps_benchmark.json`, and `test.py` must agree on key names, creating a coupling contract.

## Alternative Directions Considered

### Alt-1: Extend existing bench_fps.py
- Gist: Add JSON output, CLI flags (`--fps_split`, `--fps_warmup`, `--fps_repeat`, `--fps_max_views`, `--fps_camera_seed`, `--fps_output`), and GPU name reporting directly to the existing `bench_fps.py` (416 LOC). The core `torch.cuda.Event` timing, warmup, `torch.no_grad()`, and render-only isolation already exist there. Then hook `test.py` to call `bench_fps.py` as a subprocess and read `fps_benchmark.json` into `metrics.json`.
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/bench_fps.py` lines 112-120: exact `torch.cuda.Event` pattern already working
  - `/home/xzh/xzh/RFGS/bench_fps.py` lines 86-96: warmup loop already implemented
  - `/home/xzh/xzh/RFGS/train.py` lines 400-430: `json.dump(prof_data, fp, indent=True)` pattern for `profiler_results.json`
  - `/home/xzh/xzh/RFGS/test.py` lines 71-79: existing per-scene JSON collection pattern
- Why not primary: `bench_fps.py` is a standalone `__main__` script (not importable as a module), which forces subprocess-based integration with `test.py` (model reload overhead) and prevents shared use with `render.py` without code duplication.

### Alt-2: Inline in render.py with argparse hooks
- Gist: Add `benchmark_render_fps()` directly in `render.py` (~110 lines), register all FPS CLI flags in its `__main__` block (~10 lines), and gate the benchmark behind `--benchmark_fps`. Single-file change minimizes architectural disruption and keeps the benchmark co-located with the rendering logic it measures.
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/render.py` line 32: exact render call `render(view, gaussians, pipeline, background)["render"]`
  - `/home/xzh/xzh/RFGS/render.py` lines 51-66: existing `parser.add_argument()` pattern for `--iteration`, `--skip_train`, `--skip_test`, `--quiet`
  - `/home/xzh/xzh/RFGS/bench_fps.py` entire file: all timing logic is directly portable into render.py
- Why not primary: Couples benchmark logic to render.py, making it harder to reuse from `test.py` without subprocess overhead. Also adds ~135 lines to an otherwise simple 65-line file.

### Alt-3: Training-checkpoint FPS profiling
- Gist: Hook FPS measurement into `train.py`'s `training()` function at the checkpoint-save hook (`if iteration in saving_iterations:` at line 170). After each checkpoint save, render a configurable number of frames with `torch.cuda.Event` timing and append results to a cumulative `fps_benchmark.json`. This reveals how densification decisions affect rendering performance over time, not just final model speed.
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/train.py` lines 170-172: existing checkpoint save hook
  - `/home/xzh/xzh/RFGS/train.py` lines 53-54, 182-195: 7+ existing `torch.cuda.Event` profiling sites in the same function
  - `/home/xzh/xzh/RFGS/train.py` lines 400-430: end-of-training JSON export pattern
  - `/home/xzh/xzh/RFGS/bench_fps.py` entire file: portable timing/warmup/statistics logic
- Why not primary: Different use case — measures training-time FPS trends rather than final-model FPS for paper comparison tables. The user's idea explicitly targets "最终Gaussian模型的纯渲染FPS" (final model pure render FPS). Significant risk of training slowdown and CUDA memory pressure from extra renders during training.

### Alt-4: Post-render JSON-only collection script
- Gist: Extend `bench_fps.py` to function as a fully decoupled standalone script that loads a saved model, runs only the benchmark, writes `fps_benchmark.json`, and exits. `render.py`'s image-saving pipeline is left completely untouched. `test.py` calls this script as an additional `os.system()` step per scene and reads the JSON into `metrics.json`.
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/bench_fps.py` lines 34-42: model/scene loading identical to render.py
  - `/home/xzh/xzh/RFGS/scene/__init__.py` lines 77-83: `Scene` constructor with `load_iteration` and `shuffle` parameters
  - `/home/xzh/xzh/RFGS/test.py` lines 58-63: `os.system()` subprocess pattern for train/render/metrics
  - `/home/xzh/xzh/RFGS/scripts/run_candidate_ablation.sh` lines 133-178: identical inline-Python JSON collection pattern
- Why not primary: Requires separate model loading (GPU memory peak from loading the full checkpoint a second time). Less integrated with `render.py` — users must run two commands instead of adding one flag. The standalone approach is best reserved as a fallback for when `render.py` integration isn't desired.

### Alt-5: Comparative benchmark harness for method A/B testing
- Gist: Build `bench_fps_compare.py` accepting multiple `--model_dirs`, loading each model, and rendering the exact same deterministically-seeded camera set against every model. Outputs a side-by-side comparison table and JSON for direct paper-table use. This is the most rigorous approach for ensuring fair cross-method comparisons.
- Objective Evidence:
  - `/home/xzh/xzh/RFGS/bench_fps.py`: single-model benchmark with proven timing infrastructure
  - `/home/xzh/xzh/RFGS/metrics.py` line 100-103: already accepts `--model_paths` as `nargs="+"`, providing precedent for multi-directory CLI arguments
  - `/home/xzh/xzh/RFGS/arguments/__init__.py` lines 183-203: `get_combined_args()` recovers full model config from `cfg_args` — essential for multi-model loading
  - `/home/xzh/xzh/RFGS/scripts/validate_conf_vis.py` lines 200-217: established pattern for programmatic model loading from cfg_args
- Why not primary: Adds significant complexity (~300-400 LOC new file) for a comparison feature that can be achieved by running the single-model benchmark twice with identical parameters and comparing the JSON outputs manually. Best reserved as a follow-up after the core single-model benchmark is stable.

## Synthesis Notes

The standalone module approach (primary) naturally absorbs elements from several alternatives. The batch-timing mode from Alt-2's single-event approach (commented-out `bench_fps.py` lines 245-416) should be adopted as the default paper-FPS mode since it avoids per-frame `synchronize()` overhead. The JSON schema and terminal output format from Alt-1's `bench_fps.py` extension can be adopted directly. The `test.py` integration from Alt-4 (read `fps_benchmark.json` after subprocess exits) is the cleanest pattern since it matches how `profiler_results.json` and `results.json` are already collected. If the user later needs cross-method comparison tables, the standalone module can be wrapped by Alt-5's comparative harness without modifying the core timing logic. The training-checkpoint profiling (Alt-3) is orthogonal and could be added later as a separate `--benchmark_fps` flag in `train.py` that imports the same `utils/fps_benchmark.py` module — the module design deliberately supports this by accepting explicit `(cameras, gaussians, pipeline, background)` parameters rather than baking in model-loading logic.

--- Original Design Draft End ---
