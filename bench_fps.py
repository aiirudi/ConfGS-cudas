

import torch
from argparse import ArgumentParser
import numpy as np

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
import inspect

try:
    from utils.general_utils import SPARSE_ADAM_AVAILABLE
except Exception:
    SPARSE_ADAM_AVAILABLE = False


@torch.no_grad()
def main():
    parser = ArgumentParser()
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--num_frames", type=int, default=200, help="Number of frames to render for FPS measurement")
    args = get_combined_args(parser)

    dataset = mp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    # 初始化模型
    try:
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    except TypeError:
        gaussians = GaussianModel(dataset.sh_degree)

    try:
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    except TypeError:
        scene = Scene(dataset, gaussians)

    views = scene.getTrainCameras() if hasattr(args, 'use_train') and args.use_train else scene.getTestCameras()
    if len(views) == 0:
        print("No test cameras found, using training cameras...")
        views = scene.getTrainCameras()
    if len(views) == 0:
        raise RuntimeError("No cameras found at all!")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        device="cuda",
        dtype=torch.float32
    )

    # 动态检查render参数
    render_sig = inspect.signature(render)
    render_params = set(render_sig.parameters.keys())
    
    render_kwargs = {"scaling_modifier": 1.0}
    if "use_trained_exp" in render_params:
        render_kwargs["use_trained_exp"] = getattr(dataset, "train_test_exp", False)
    if "separate_sh" in render_params:
        render_kwargs["separate_sh"] = SPARSE_ADAM_AVAILABLE

    # 打印场景信息
    num_gaussians = len(gaussians.get_xyz)
    img_width = views[0].image_width
    img_height = views[0].image_height
    
    print(f"\n{'='*80}")
    print(f"FPS Benchmark Configuration")
    print(f"{'='*80}")
    print(f"Scene Path       : {dataset.model_path}")
    print(f"Iteration        : {args.iteration}")
    print(f"Num Gaussians    : {num_gaussians:,}")
    print(f"Image Resolution : {img_width} x {img_height}")
    print(f"Num Views        : {len(views)}")
    print(f"Test Frames      : {args.num_frames}")
    print(f"{'='*80}\n")

    # ============================================================
    # Warmup Phase (重要！GPU需要预热)
    # ============================================================
    print("Phase 1: Warming up GPU...")
    warmup_frames = min(50, args.num_frames // 4)
    test_view = views[0]
    
    for i in range(warmup_frames):
        _ = render(test_view, gaussians, pipe, background, **render_kwargs)["render"]
        if (i + 1) % 10 == 0:
            print(f"  Warmup progress: {i+1}/{warmup_frames}")
    
    torch.cuda.synchronize()
    print(f"✓ Warmup complete ({warmup_frames} frames)\n")

    # ============================================================
    # Method 1: Frame-by-Frame Timing (最精确的单帧FPS)
    # ============================================================
    print("Phase 2: Measuring frame-by-frame rendering time...")
    
    frame_times_ms = []
    view_idx = 0
    
    for i in range(args.num_frames):
        # 循环使用所有视角
        current_view = views[view_idx % len(views)]
        view_idx += 1
        
        # 使用CUDA Events精确计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        _ = render(current_view, gaussians, pipe, background, **render_kwargs)["render"]
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        frame_times_ms.append(elapsed_ms)
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{args.num_frames} frames")
    
    print(f"✓ Rendered {args.num_frames} frames\n")

    # ============================================================
    # Statistical Analysis (统计分析)
    # ============================================================
    frame_times_ms = np.array(frame_times_ms)
    
    # 基本统计
    mean_time = np.mean(frame_times_ms)
    median_time = np.median(frame_times_ms)
    std_time = np.std(frame_times_ms)
    min_time = np.min(frame_times_ms)
    max_time = np.max(frame_times_ms)
    
    # 计算FPS (1000ms / 每帧时间ms)
    fps_values = 1000.0 / frame_times_ms
    mean_fps = 1000.0 / mean_time
    median_fps = 1000.0 / median_time
    min_fps = 1000.0 / max_time  # 最长时间 = 最低FPS
    max_fps = 1000.0 / min_time  # 最短时间 = 最高FPS
    
    # 计算百分位数 (移除异常值)
    p5_time = np.percentile(frame_times_ms, 5)
    p95_time = np.percentile(frame_times_ms, 95)
    p5_fps = 1000.0 / p95_time
    p95_fps = 1000.0 / p5_time
    
    # 稳定帧时间（去除前后5%异常值）
    trimmed_times = frame_times_ms[(frame_times_ms >= p5_time) & (frame_times_ms <= p95_time)]
    stable_mean_time = np.mean(trimmed_times)
    stable_fps = 1000.0 / stable_mean_time
    
    # ============================================================
    # Results Output (结果输出)
    # ============================================================
    print(f"{'='*80}")
    print(f"FPS Benchmark Results")
    print(f"{'='*80}")
    print(f"\n📊 Frame Timing Statistics:")
    print(f"  Mean frame time    : {mean_time:.3f} ms")
    print(f"  Median frame time  : {median_time:.3f} ms")
    print(f"  Std deviation      : {std_time:.3f} ms")
    print(f"  Min frame time     : {min_time:.3f} ms")
    print(f"  Max frame time     : {max_time:.3f} ms")
    print(f"  5th percentile     : {p5_time:.3f} ms")
    print(f"  95th percentile    : {p95_time:.3f} ms")
    
    print(f"\n🎯 FPS Metrics:")
    print(f"  Mean FPS           : {mean_fps:.2f}")
    print(f"  Median FPS         : {median_fps:.2f}")
    print(f"  Stable FPS (5-95%) : {stable_fps:.2f}")  # ← 这个最重要！
    print(f"  Min FPS            : {min_fps:.2f}")
    print(f"  Max FPS            : {max_fps:.2f}")
    
    print(f"\n📈 Performance Assessment:")
    if stable_fps >= 60:
        rating = "Excellent (60+ FPS) ✓"
    elif stable_fps >= 30:
        rating = "Good (30-60 FPS)"
    elif stable_fps >= 15:
        rating = "Fair (15-30 FPS)"
    else:
        rating = "Poor (<15 FPS)"
    print(f"  Rating: {rating}")
    
    print(f"\n🔧 Scene Complexity:")
    print(f"  Gaussians          : {num_gaussians:,}")
    print(f"  Resolution         : {img_width} x {img_height}")
    print(f"  Megapixels         : {(img_width * img_height) / 1e6:.2f} MP")
    print(f"  Gaussians/Megapixel: {num_gaussians / ((img_width * img_height) / 1e6):.0f}")
    
    # ============================================================
    # Quick Reference (快速参考)
    # ============================================================
    print(f"\n{'='*80}")
    print(f"📋 SUMMARY (Copy this for your paper/report):")
    print(f"{'='*80}")
    print(f"Scene: {dataset.model_path.split('/')[-1]}")
    print(f"FPS: {stable_fps:.2f} (stable), {mean_fps:.2f} (mean)")
    print(f"Frame Time: {stable_mean_time:.2f}ms (stable), {mean_time:.2f}ms (mean)")
    print(f"Resolution: {img_width}x{img_height}, Gaussians: {num_gaussians:,}")
    print(f"{'='*80}\n")
    
    # ============================================================
    # Optional: Per-view analysis (可选：每个视角的详细分析)
    # ============================================================
    if len(views) > 1:
        print(f"\n🔍 Per-View Analysis (testing all {len(views)} views):")
        view_fps_list = []
        
        for view_id, view in enumerate(views):
            view_times = []
            test_count = min(20, args.num_frames // len(views))  # 每个视角测试20次
            
            for _ in range(test_count):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                _ = render(view, gaussians, pipe, background, **render_kwargs)["render"]
                end_event.record()
                
                torch.cuda.synchronize()
                view_times.append(start_event.elapsed_time(end_event))
            
            view_mean_time = np.mean(view_times)
            view_fps = 1000.0 / view_mean_time
            view_fps_list.append(view_fps)
            
            print(f"  View {view_id+1:2d}: {view_fps:6.2f} FPS ({view_mean_time:6.2f}ms)")
        
        fps_variance = np.std(view_fps_list)
        print(f"\n  FPS variance across views: ±{fps_variance:.2f}")
        print(f"  Most consistent view would get: {np.median(view_fps_list):.2f} FPS\n")


if __name__ == "__main__":
    main()

"""
import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
import inspect

try:
    from utils.general_utils import SPARSE_ADAM_AVAILABLE
except Exception:
    SPARSE_ADAM_AVAILABLE = False


@torch.no_grad()
def main():
    parser = ArgumentParser()
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--test_frames", type=int, default=100, help="Number of frames to render")
    parser.add_argument("--use_train", action="store_true")
    args = get_combined_args(parser)

    dataset = mp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    # 初始化模型
    try:
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    except TypeError:
        gaussians = GaussianModel(dataset.sh_degree)

    try:
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    except TypeError:
        scene = Scene(dataset, gaussians)

    views = scene.getTrainCameras() if args.use_train else scene.getTestCameras()
    if len(views) == 0:
        views = scene.getTrainCameras()
    if len(views) == 0:
        raise RuntimeError("No cameras found!")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        device="cuda",
        dtype=torch.float32
    )

    # 动态检查render参数
    render_sig = inspect.signature(render)
    render_params = set(render_sig.parameters.keys())
    
    render_kwargs = {"scaling_modifier": 1.0}
    if "use_trained_exp" in render_params:
        render_kwargs["use_trained_exp"] = getattr(dataset, "train_test_exp", False)
    if "separate_sh" in render_params:
        render_kwargs["separate_sh"] = SPARSE_ADAM_AVAILABLE

    # 场景信息
    num_gaussians = len(gaussians.get_xyz)
    img_width = views[0].image_width
    img_height = views[0].image_height
    
    print(f"\n{'='*70}")
    print(f"Scene: {dataset.model_path}")
    print(f"Gaussians: {num_gaussians:,}")
    print(f"Resolution: {img_width} x {img_height}")
    print(f"Views available: {len(views)}")
    print(f"{'='*70}\n")

    # =================================================================
    # Warmup: 让GPU进入稳定状态
    # =================================================================
    print("Warming up GPU (30 frames)...")
    warmup_view = views[0]
    for _ in range(30):
        _ = render(warmup_view, gaussians, pipe, background, **render_kwargs)["render"]
    torch.cuda.synchronize()
    print("✓ Warmup done\n")

    # =================================================================
    # 实际测量：就是简单的"渲染N帧，看用了多少时间"
    # =================================================================
    print(f"Measuring FPS by rendering {args.test_frames} frames...")
    
    # 创建CUDA事件来精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 开始计时
    start_event.record()
    
    # 连续渲染N帧（循环使用所有视角）
    for i in range(args.test_frames):
        view = views[i % len(views)]
        _ = render(view, gaussians, pipe, background, **render_kwargs)["render"]
    
    # 结束计时
    end_event.record()
    torch.cuda.synchronize()
    
    # 计算总时间（毫秒）
    total_time_ms = start_event.elapsed_time(end_event)
    total_time_sec = total_time_ms / 1000.0
    
    # 计算FPS：帧数 / 秒数
    fps = args.test_frames / total_time_sec
    
    # 平均每帧时间
    avg_frame_time_ms = total_time_ms / args.test_frames
    
    # =================================================================
    # 输出结果
    # =================================================================
    print(f"\n{'='*70}")
    print(f"RESULT:")
    print(f"{'='*70}")
    print(f"Total frames rendered : {args.test_frames}")
    print(f"Total time            : {total_time_sec:.3f} seconds ({total_time_ms:.1f} ms)")
    print(f"Average frame time    : {avg_frame_time_ms:.2f} ms")
    print(f"")
    print(f"FPS = {args.test_frames} frames / {total_time_sec:.3f} sec = {fps:.2f}")
    print(f"{'='*70}")
    print(f"")
    print(f"📊 Scene Info:")
    print(f"   Gaussians : {num_gaussians:,}")
    print(f"   Resolution: {img_width} x {img_height}")
    print(f"{'='*70}\n")
    
    # =================================================================
    # 额外验证：重复3次确保结果稳定
    # =================================================================
    print("Verification: Running 3 additional tests...")
    fps_list = [fps]
    
    for run in range(3):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for i in range(args.test_frames):
            view = views[i % len(views)]
            _ = render(view, gaussians, pipe, background, **render_kwargs)["render"]
        end_event.record()
        torch.cuda.synchronize()
        
        t_ms = start_event.elapsed_time(end_event)
        t_sec = t_ms / 1000.0
        fps_run = args.test_frames / t_sec
        fps_list.append(fps_run)
        
        print(f"  Run {run+1}: {fps_run:.2f} FPS")
    
    avg_fps = sum(fps_list) / len(fps_list)
    max_diff = max(fps_list) - min(fps_list)
    
    print(f"\nAverage across 4 runs: {avg_fps:.2f} FPS")
    print(f"Variation: ±{max_diff/2:.2f} FPS")
    print(f"\n{'='*70}")
    print(f"FINAL FPS: {avg_fps:.2f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
"""
