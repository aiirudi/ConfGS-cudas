
import torch
import json
import os
import random
import numpy as np
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
import inspect

try:
    from utils.general_utils import SPARSE_ADAM_AVAILABLE
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def positive_int(value):
    """argparse type: 正整数（≥1）。"""
    ival = int(value)
    if ival < 1:
        raise ValueError(f"must be >= 1, got {ival}")
    return ival


def non_zero_int_or_neg1(value):
    """argparse type: -1（表全部）或正整数（≥1）。"""
    ival = int(value)
    if ival == 0:
        raise ValueError("fps_max_views cannot be 0 (use -1 for all)")
    if ival < -1:
        raise ValueError(f"fps_max_views must be -1 or >= 1, got {ival}")
    return ival


def validate_fps_args(warmup, repeat, max_views):
    """验证 FPS benchmark 参数合法性，不合法时抛出 ValueError。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FPS benchmarking")
    if repeat < 1:
        raise ValueError(f"fps_repeat must be >= 1, got {repeat}")
    if warmup < 0:
        raise ValueError(f"fps_warmup must be >= 0, got {warmup}")
    if max_views == 0:
        raise ValueError("fps_max_views cannot be 0")


def select_cameras(scene, fps_split, fps_max_views, fps_camera_seed):
    """选择 FPS benchmark 使用的相机列表。

    返回 (cameras, split_name)，若选定 split 为空则抛出 ValueError。
    """
    if fps_split == "train":
        cameras = scene.getTrainCameras()
    elif fps_split == "test":
        cameras = scene.getTestCameras()
    else:
        raise ValueError(f"Unknown fps_split '{fps_split}', must be 'train' or 'test'")

    if len(cameras) == 0:
        raise ValueError(
            f"No cameras found in '{fps_split}' split. "
            f"Use --fps_split train if test cameras are not available."
        )

    if fps_max_views > 0 and fps_max_views < len(cameras):
        rng = random.Random(fps_camera_seed)
        indices = rng.sample(range(len(cameras)), fps_max_views)
        cameras = [cameras[i] for i in sorted(indices)]

    return cameras, fps_split


def select_legacy_cameras(scene, args):
    """Legacy path camera 选择：匹配原始 bench_fps.py 行为。

    test 优先，有 --use_train 时用 train，空则 fallback 到 train，都空则 RuntimeError。
    """
    use_train = hasattr(args, 'use_train') and args.use_train
    views = scene.getTrainCameras() if use_train else scene.getTestCameras()
    if len(views) == 0:
        print("No test cameras found, using training cameras...")
        views = scene.getTrainCameras()
    if len(views) == 0:
        raise RuntimeError("No cameras found at all!")
    return views


def render_warmup(cameras, gaussians, pipe, background, render_kwargs, warmup_frames):
    """GPU 预热：循环渲染 warmup_frames 帧，结束后同步。"""
    for i in range(warmup_frames):
        _ = render(cameras[i % len(cameras)], gaussians, pipe, background, **render_kwargs)["render"]
    torch.cuda.synchronize()


def benchmark_batch(cameras, gaussians, pipe, background, render_kwargs, repeat):
    """Batch timing: 单个 CUDA event 对包裹全部渲染调用。

    返回 (fps, avg_ms, total_frames, elapsed_seconds)。
    """
    total_frames = len(cameras) * repeat

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        for cam in cameras:
            render_pkg = render(cam, gaussians, pipe, background, **render_kwargs)
            _ = render_pkg["render"]  # 保持引用防止提前释放
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_seconds = elapsed_ms / 1000.0

    if elapsed_ms == 0:
        raise RuntimeError("Elapsed time is zero — cannot compute FPS")

    fps = total_frames / elapsed_seconds
    avg_ms = elapsed_ms / total_frames
    return fps, avg_ms, total_frames, elapsed_seconds


def benchmark_per_frame(cameras, gaussians, pipe, background, render_kwargs, num_frames):
    """Per-frame timing: 每帧一个 CUDA event 对，保留原有统计功能。

    返回 dict 包含 frame_times_ms, fps_values, mean_fps, median_fps, stable_fps 等。
    """

    frame_times_ms = []
    for i in range(num_frames):
        cam = cameras[i % len(cameras)]
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = render(cam, gaussians, pipe, background, **render_kwargs)["render"]
        end_event.record()

        torch.cuda.synchronize()
        frame_times_ms.append(start_event.elapsed_time(end_event))

    frame_times_ms = np.array(frame_times_ms)

    mean_time = np.mean(frame_times_ms)
    median_time = np.median(frame_times_ms)
    std_time = np.std(frame_times_ms)
    min_time = np.min(frame_times_ms)
    max_time = np.max(frame_times_ms)

    fps_values = 1000.0 / frame_times_ms
    mean_fps = 1000.0 / mean_time
    median_fps = 1000.0 / median_time

    p5_time = np.percentile(frame_times_ms, 5)
    p95_time = np.percentile(frame_times_ms, 95)
    trimmed_times = frame_times_ms[(frame_times_ms >= p5_time) & (frame_times_ms <= p95_time)]
    stable_mean_time = np.mean(trimmed_times)
    stable_fps = 1000.0 / stable_mean_time

    return {
        "frame_times_ms": frame_times_ms,
        "fps_values": fps_values,
        "mean_time": mean_time,
        "median_time": median_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "p5_time": p5_time,
        "p95_time": p95_time,
        "mean_fps": mean_fps,
        "median_fps": median_fps,
        "stable_fps": stable_fps,
        "stable_mean_time": stable_mean_time,
    }


def print_batch_result(fps, avg_ms, total_frames, elapsed_seconds,
                       gaussian_count, device_name, img_width, img_height,
                       split_name, warmup_frames, camera_count, repeat):
    """打印 batch timing 的终端输出。"""
    print(f"\n{'='*60}")
    print(f"  FPS Benchmark (batch timing)")
    print(f"{'='*60}")
    print(f"  Split              : {split_name}")
    print(f"  Warmup frames      : {warmup_frames}")
    print(f"  Measured views     : {camera_count}")
    print(f"  Repeat             : {repeat}")
    print(f"  Total measured frames: {total_frames}")
    print(f"  Total pure render time: {elapsed_seconds:.3f} s")
    print(f"  Average render latency : {avg_ms:.2f} ms/frame")
    print(f"  Rendering FPS        : {fps:.2f}")
    print(f"  Gaussian count       : {gaussian_count:,}")
    print(f"  CUDA device          : {device_name}")
    print(f"  Image resolution     : {img_width}x{img_height}")
    print(f"{'='*60}\n")


def print_per_frame_result(stats, gaussian_count, device_name, img_width, img_height):
    """打印 per-frame timing 的终端输出（保留原有统计格式）。"""
    print(f"\n{'='*80}")
    print(f"FPS Benchmark Results (per-frame timing)")
    print(f"{'='*80}")
    print(f"\n  Frame Timing Statistics:")
    print(f"  Mean frame time    : {stats['mean_time']:.3f} ms")
    print(f"  Median frame time  : {stats['median_time']:.3f} ms")
    print(f"  Std deviation      : {stats['std_time']:.3f} ms")
    print(f"  Min frame time     : {stats['min_time']:.3f} ms")
    print(f"  Max frame time     : {stats['max_time']:.3f} ms")
    print(f"  5th percentile     : {stats['p5_time']:.3f} ms")
    print(f"  95th percentile    : {stats['p95_time']:.3f} ms")

    print(f"\n  FPS Metrics:")
    print(f"  Mean FPS           : {stats['mean_fps']:.2f}")
    print(f"  Median FPS         : {stats['median_fps']:.2f}")
    print(f"  Stable FPS (5-95%) : {stats['stable_fps']:.2f}")

    print(f"\n  Scene Complexity:")
    print(f"  Gaussians          : {gaussian_count:,}")
    print(f"  Resolution         : {img_width} x {img_height}")
    print(f"{'='*80}\n")


def save_fps_json(output_path, split_name, iteration, warmup_frames, repeat,
                  camera_count, total_frames, elapsed_seconds, avg_ms, fps,
                  gaussian_count, device_name, img_width, img_height,
                  timing_mode):
    """将 FPS 结果保存为 JSON 文件（在计时结束后调用）。"""
    result = {
        "split": split_name,
        "iteration": iteration,
        "warmup_frames": warmup_frames,
        "repeat": repeat,
        "camera_count": camera_count,
        "total_frames": total_frames,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "average_ms_per_frame": round(avg_ms, 4),
        "fps": round(fps, 2),
        "gaussian_count": gaussian_count,
        "device": device_name,
        "image_width": img_width,
        "image_height": img_height,
        "timing_method": "torch.cuda.Event",
        "timing_mode": timing_mode,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "included_operations": ["gaussian_renderer.render"],
        "excluded_operations": [
            "image_save",
            "cpu_transfer",
            "metric_computation",
            "conf_visualization",
            "gui_refresh"
        ]
    }
    with open(output_path, 'w') as fp:
        json.dump(result, fp, indent=True)
    print(f"  FPS result saved to: {output_path}")


@torch.no_grad()
def main():
    parser = ArgumentParser()
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)

    # 原有参数
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--num_frames", type=int, default=200,
                        help="Number of frames for per-frame timing mode")
    parser.add_argument("--use_train", action="store_true",
                        help="Use training cameras (legacy path only)")

    # 新增 FPS benchmark 参数
    parser.add_argument("--benchmark_fps", action="store_true", default=False,
                        help="Enable standardized FPS benchmark with JSON output")
    parser.add_argument("--fps_split", type=str, default="test",
                        choices=["train", "test"],
                        help="Camera split for FPS benchmark (default: test)")
    parser.add_argument("--fps_warmup", type=int, default=20,
                        help="Warmup frames before timed measurement (default: 20)")
    parser.add_argument("--fps_repeat", type=positive_int, default=3,
                        help="Number of times to repeat full camera set (default: 3, min: 1)")
    parser.add_argument("--fps_max_views", type=non_zero_int_or_neg1, default=-1,
                        help="Max camera views (-1 = all, must be != 0, default: -1)")
    parser.add_argument("--fps_camera_seed", type=int, default=0,
                        help="Random seed for camera sampling (default: 0)")
    parser.add_argument("--fps_output", type=str, default=None,
                        help="FPS JSON output path (default: <model_path>/fps_benchmark.json)")
    parser.add_argument("--fps_timing_mode", type=str, default="batch",
                        choices=["batch", "per_frame"],
                        help="Timing mode: batch (paper FPS, default) or per_frame (detailed stats)")

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

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        device="cuda",
        dtype=torch.float32
    )

    # 动态检查 render 参数（兼容不同版本的 render 签名）
    render_sig = inspect.signature(render)
    render_params = set(render_sig.parameters.keys())

    render_kwargs = {"scaling_modifier": 1.0}
    if "use_trained_exp" in render_params:
        render_kwargs["use_trained_exp"] = getattr(dataset, "train_test_exp", False)
    if "separate_sh" in render_params:
        render_kwargs["separate_sh"] = SPARSE_ADAM_AVAILABLE

    device_name = torch.cuda.get_device_name(0)
    iteration = scene.loaded_iter if scene.loaded_iter else args.iteration

    if args.benchmark_fps:
        # ============================================================
        # Standardized Benchmark Path (--benchmark_fps)
        # ============================================================
        validate_fps_args(args.fps_warmup, args.fps_repeat, args.fps_max_views)

        cameras, split_name = select_cameras(
            scene, args.fps_split, args.fps_max_views, args.fps_camera_seed
        )

        gaussian_count = gaussians.get_xyz.shape[0]
        img_width = cameras[0].image_width
        img_height = cameras[0].image_height

        print(f"\n{'='*80}")
        print(f"FPS Benchmark Configuration")
        print(f"{'='*80}")
        print(f"Scene Path       : {dataset.model_path}")
        print(f"Iteration        : {iteration}")
        print(f"Num Gaussians    : {gaussian_count:,}")
        print(f"Image Resolution : {img_width} x {img_height}")
        print(f"Available Views  : {len(cameras)} ({split_name})")
        print(f"Timing Mode      : {args.fps_timing_mode}")
        print(f"{'='*80}\n")

        print(f"Warming up GPU ({args.fps_warmup} frames)...")
        render_warmup(cameras, gaussians, pipe, background, render_kwargs, args.fps_warmup)
        print(f"Warmup complete\n")

        if args.fps_timing_mode == "batch":
            print(f"Measuring FPS (batch timing, {args.fps_repeat} repeats x {len(cameras)} views)...")
            fps, avg_ms, total_frames, elapsed_seconds = benchmark_batch(
                cameras, gaussians, pipe, background, render_kwargs, args.fps_repeat
            )
            print_batch_result(
                fps, avg_ms, total_frames, elapsed_seconds,
                gaussian_count, device_name, img_width, img_height,
                split_name, args.fps_warmup, len(cameras), args.fps_repeat
            )
        else:
            num_frames = args.num_frames
            print(f"Measuring FPS (per-frame timing, {num_frames} frames)...")
            stats = benchmark_per_frame(
                cameras, gaussians, pipe, background, render_kwargs, num_frames
            )
            print_per_frame_result(stats, gaussian_count, device_name, img_width, img_height)
            total_frames = num_frames
            fps = stats["stable_fps"]
            avg_ms = stats["stable_mean_time"]
            elapsed_seconds = (stats["mean_time"] * num_frames) / 1000.0

            if len(cameras) > 1:
                print(f"\n  Per-View Analysis (testing all {len(cameras)} views):")
                view_fps_list = []
                for view_id, view in enumerate(cameras):
                    view_times = []
                    test_count = min(20, num_frames // len(cameras))
                    for _ in range(test_count):
                        se = torch.cuda.Event(enable_timing=True)
                        ee = torch.cuda.Event(enable_timing=True)
                        se.record()
                        _ = render(view, gaussians, pipe, background, **render_kwargs)["render"]
                        ee.record()
                        torch.cuda.synchronize()
                        view_times.append(se.elapsed_time(ee))
                    view_fps = 1000.0 / np.mean(view_times)
                    view_fps_list.append(view_fps)
                    print(f"    View {view_id+1:2d}: {view_fps:6.2f} FPS")
                print(f"\n    FPS variance across views: ±{np.std(view_fps_list):.2f}\n")

        json_path = getattr(args, 'fps_output', None)
        if json_path is None:
            json_path = os.path.join(dataset.model_path, "fps_benchmark.json")
        save_fps_json(
            json_path, split_name, iteration, args.fps_warmup, args.fps_repeat,
            len(cameras), total_frames, elapsed_seconds, avg_ms, fps,
            gaussian_count, device_name, img_width, img_height,
            args.fps_timing_mode
        )

    else:
        # ============================================================
        # Legacy Path (no --benchmark_fps): 完全匹配原始 bench_fps.py 行为
        # ============================================================
        views = select_legacy_cameras(scene, args)

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

        # 原始 warmup
        print("Phase 1: Warming up GPU...")
        warmup_frames = min(50, args.num_frames // 4)
        test_view = views[0]

        for i in range(warmup_frames):
            _ = render(test_view, gaussians, pipe, background, **render_kwargs)["render"]
            if (i + 1) % 10 == 0:
                print(f"  Warmup progress: {i+1}/{warmup_frames}")

        torch.cuda.synchronize()
        print(f"  Warmup complete ({warmup_frames} frames)\n")

        # 原始 per-frame timing
        print("Phase 2: Measuring frame-by-frame rendering time...")

        frame_times_ms = []
        view_idx = 0

        for i in range(args.num_frames):
            current_view = views[view_idx % len(views)]
            view_idx += 1

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = render(current_view, gaussians, pipe, background, **render_kwargs)["render"]
            end_event.record()

            torch.cuda.synchronize()
            frame_times_ms.append(start_event.elapsed_time(end_event))

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{args.num_frames} frames")

        print(f"  Rendered {args.num_frames} frames\n")

        # 原始统计分析
        frame_times_ms = np.array(frame_times_ms)

        mean_time = np.mean(frame_times_ms)
        median_time = np.median(frame_times_ms)
        std_time = np.std(frame_times_ms)
        min_time = np.min(frame_times_ms)
        max_time = np.max(frame_times_ms)

        fps_values = 1000.0 / frame_times_ms
        mean_fps = 1000.0 / mean_time
        median_fps = 1000.0 / median_time
        min_fps = 1000.0 / max_time
        max_fps = 1000.0 / min_time

        p5_time = np.percentile(frame_times_ms, 5)
        p95_time = np.percentile(frame_times_ms, 95)
        p5_fps = 1000.0 / p95_time
        p95_fps = 1000.0 / p5_time

        trimmed_times = frame_times_ms[(frame_times_ms >= p5_time) & (frame_times_ms <= p95_time)]
        stable_mean_time = np.mean(trimmed_times)
        stable_fps = 1000.0 / stable_mean_time

        # 原始结果输出
        print(f"{'='*80}")
        print(f"FPS Benchmark Results")
        print(f"{'='*80}")
        print(f"\n  Frame Timing Statistics:")
        print(f"  Mean frame time    : {mean_time:.3f} ms")
        print(f"  Median frame time  : {median_time:.3f} ms")
        print(f"  Std deviation      : {std_time:.3f} ms")
        print(f"  Min frame time     : {min_time:.3f} ms")
        print(f"  Max frame time     : {max_time:.3f} ms")
        print(f"  5th percentile     : {p5_time:.3f} ms")
        print(f"  95th percentile    : {p95_time:.3f} ms")

        print(f"\n  FPS Metrics:")
        print(f"  Mean FPS           : {mean_fps:.2f}")
        print(f"  Median FPS         : {median_fps:.2f}")
        print(f"  Stable FPS (5-95%) : {stable_fps:.2f}")
        print(f"  Min FPS            : {min_fps:.2f}")
        print(f"  Max FPS            : {max_fps:.2f}")

        print(f"\n  Performance Assessment:")
        if stable_fps >= 60:
            rating = "Excellent (60+ FPS)"
        elif stable_fps >= 30:
            rating = "Good (30-60 FPS)"
        elif stable_fps >= 15:
            rating = "Fair (15-30 FPS)"
        else:
            rating = "Poor (<15 FPS)"
        print(f"  Rating: {rating}")

        print(f"\n  Scene Complexity:")
        print(f"  Gaussians          : {num_gaussians:,}")
        print(f"  Resolution         : {img_width} x {img_height}")
        print(f"  Megapixels         : {(img_width * img_height) / 1e6:.2f} MP")
        print(f"  Gaussians/Megapixel: {num_gaussians / ((img_width * img_height) / 1e6):.0f}")

        print(f"\n{'='*80}")
        print(f"  SUMMARY (Copy this for your paper/report):")
        print(f"{'='*80}")
        print(f"Scene: {dataset.model_path.split('/')[-1]}")
        print(f"FPS: {stable_fps:.2f} (stable), {mean_fps:.2f} (mean)")
        print(f"Frame Time: {stable_mean_time:.2f}ms (stable), {mean_time:.2f}ms (mean)")
        print(f"Resolution: {img_width}x{img_height}, Gaussians: {num_gaussians:,}")
        print(f"{'='*80}\n")

        # 原始 per-view 分析
        if len(views) > 1:
            print(f"\n  Per-View Analysis (testing all {len(views)} views):")
            view_fps_list = []

            for view_id, view in enumerate(views):
                view_times = []
                test_count = min(20, args.num_frames // len(views))

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
