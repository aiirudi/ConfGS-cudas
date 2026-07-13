import os
import json
import time
import subprocess

# ============================================================
# 空间多样性候选选择器配置 (通过代码控制, 无需命令行传参)
# 优先级: 此处的 True/False 会作为 --enable_spatial_diversity 传给 train.py
#          设置为 None 则使用 arguments/__init__.py 中的默认值 (当前默认 True)
# ============================================================
SPATIAL_DIVERSITY_ENABLED = True     # True=启用, False=关闭, None=使用默认
SPATIAL_VOXEL_SIZE = 'auto'         # 'auto' = 自适应, 数值字符串如 '0.5'
SPATIAL_VOXEL_SCALE = 10.0           # 自适应体素尺寸系数
SPATIAL_METHOD = 'voxel'            # 空间选择方法 (仅 voxel 已实现)
# ============================================================

"""
paramList = [
    # MipNeRF-360
    ['bicycle',   'mipnerf', 3000_000],
    ['flowers',   'mipnerf', 1500_000],
    ['garden',    'mipnerf', 3000_000],
    ['stump',     'mipnerf', 3000_000],
    ['treehill',  'mipnerf', 1500_000],

    ['bonsai',    'mipnerf', 1000_000],
    ['counter',   'mipnerf', 1000_000],
    ['kitchen',   'mipnerf', 1000_000],
    ['room',      'mipnerf', 1000_000],

    # Deep Blending
    ['drjohnson', 'db',      1500_000],
    ['playroom',  'db',      1000_000],

    # Tanks & Temples
    ['train',     'tt',      1000_000],
    ['truck',     'tt',      1500_000],
]"""

paramList = [
    # MipNeRF-360
    ['bicycle',   'mipnerf', 3000_000],
    ['flowers',   'mipnerf', 1500_000],
    ['garden',    'mipnerf', 3000_000],
    ['stump',     'mipnerf', 3000_000],
    ['treehill',  'mipnerf', 1500_000],
    ['bonsai',    'mipnerf', 1000_000],
    ['counter',   'mipnerf', 1000_000],
    ['kitchen',   'mipnerf', 1000_000],
    ['room',      'mipnerf', 1000_000],
    
    # Deep Blending
    ['drjohnson', 'db',      1500_000],
    ['playroom',  'db',      1000_000],

    # Tanks & Temples
    ['train',     'tt',      1250_000],
    ['truck',     'tt',      1250_000],
]

DATA_ROOT = '/workspace/dataset/'
ALL_METRICS = {}
def build_spatial_flags():
    """根据 SPATIAL_DIVERSITY_ENABLED 构建命令行参数。"""
    if SPATIAL_DIVERSITY_ENABLED is None:
        return ''  # 使用 arguments/__init__.py 默认值
    if SPATIAL_DIVERSITY_ENABLED:
        flags = ' --enable_spatial_diversity'
        flags += f' --spatial_diversity_method {SPATIAL_METHOD}'
        flags += f' --spatial_voxel_size {SPATIAL_VOXEL_SIZE}'
        flags += f' --spatial_voxel_scale {SPATIAL_VOXEL_SCALE}'
        return flags
    # 显式 False: 传入标志确保关闭 (覆盖默认 True)
    return ''  # 不传 --enable_spatial_diversity → 使用 store_true 的默认行为


SPATIAL_FLAGS = build_spatial_flags()

for data, group, budget in paramList:
    src = f'{DATA_ROOT}/{group}/{data}'
    out = f'output/{data}'

    t_start = time.time()
    cmd = f'python train.py -s {src} -m {out} --budget {budget} --profile_components{SPATIAL_FLAGS}'
    os.system(cmd)
    t_end = time.time()
    elapsed = int(t_end - t_start)
    train_time_str = f'{elapsed // 60}分{elapsed % 60}秒'
    os.system(f'python render.py -m {out}')
    os.system(f'python metrics.py -m {out}')

    # 初始化 scene_metrics（必须在所有 JSON 读取之前）
    scene_metrics = {}
    scene_metrics['train_time'] = train_time_str

    # 4. FPS benchmark (pure rendering speed, not including image save/metrics/Conf)
    fps_json_path = os.path.join(out, 'fps_benchmark.json')
    if os.path.exists(fps_json_path):
        os.remove(fps_json_path)
    fps_ok = False
    try:
        subprocess.run(
            ['python', 'bench_fps.py', '-m', out, '--benchmark_fps', '--fps_timing_mode', 'batch'],
            check=True,
            capture_output=True,
            text=True
        )
        fps_ok = True
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] bench_fps.py failed with code {e.returncode}: {e.stderr[:200] if e.stderr else ''}")
    if fps_ok and os.path.exists(fps_json_path):
        with open(fps_json_path, 'r') as fp:
            fps_data = json.load(fp)
        scene_metrics['fps'] = fps_data.get('fps', 0.0)
        scene_metrics['fps_latency_ms'] = fps_data.get('average_ms_per_frame', 0.0)
    elif fps_ok:
        print(f"  [WARN] {fps_json_path} not found after successful bench_fps.py run")
    elif not fps_ok:
        print(f"  [WARN] Skipping FPS collection due to bench_fps.py failure")

    # 1. Training PSNR (from evaluation at iteration 30000)
    # 2. Component times (from profiler)
    prof_path = os.path.join(out, 'profiler_results.json')
    if os.path.exists(prof_path):
        with open(prof_path, 'r') as fp:
            prof_data = json.load(fp)
        scene_metrics['PSNR'] = prof_data.get('psnr', 0.0)
        scene_metrics['n_gaussians'] = prof_data.get('n_gaussians', 0)
        scene_metrics['conf_time_ms'] = prof_data.get('conf_time', 0.0)
        scene_metrics['rfas_time_ms'] = prof_data.get('rfas_time', 0.0)
        scene_metrics['fusion_time_ms'] = prof_data.get('fusion_time', 0.0)
    else:
        print(f"  [WARN] {out}/profiler_results.json not found — did training use --profile_components?")

    # 3. SSIM and LPIPS (from metrics.py results.json)
    # results.json structure: {method: {"SSIM":..., "PSNR":..., "LPIPS":...}}
    # Select the ours_* method with the highest iteration (most recent render)
    results_path = os.path.join(out, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as fp:
            results = json.load(fp)
        ours_methods = [m for m in results.keys() if m.startswith('ours_')]
        if ours_methods:
            target = max(ours_methods, key=lambda m: int(m.split('_')[-1]))
            scene_metrics['SSIM'] = results[target].get('SSIM', 0.0)
            scene_metrics['LPIPS'] = results[target].get('LPIPS', 0.0)
        else:
            print(f"  [WARN] No 'ours_*' method found in {out}/results.json")
    else:
        print(f"  [WARN] {out}/results.json not found — did render.py / metrics.py run?")

    ALL_METRICS[data] = scene_metrics
    print(f"  Collected {data}: {scene_metrics}")

    # 每完成一个场景立即写入 metrics.json，防止后续场景崩溃丢失数据
    metrics_path = 'metrics.json'
    with open(metrics_path, 'w') as fp:
        json.dump(ALL_METRICS, fp, indent=True)

print(f"\nAll metrics saved to metrics.json")
