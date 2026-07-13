import os
import json
import time
import subprocess

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
    #['bicycle',   'mipnerf', 3000_000],
    #['flowers',   'mipnerf', 1500_000],
    #['garden',    'mipnerf', 3000_000],
    #['stump',     'mipnerf', 3000_000],
    #['treehill',  'mipnerf', 1500_000],

    #['bonsai',    'mipnerf', 1000_000],
    #['counter',   'mipnerf', 1000_000],
    #['kitchen',   'mipnerf', 1000_000],
    #['room',      'mipnerf', 1000_000],

    # Deep Blending
    #['drjohnson', 'db',      1500_000],
    #['playroom',  'db',      1000_000],

    # Tanks & Temples
    ['train',     'tt',      1250_000],
    ['truck',     'tt',      1250_000],
]

DATA_ROOT = '/workspace/dataset/'
ALL_METRICS = {}

for data, group, budget in paramList:
    src = f'{DATA_ROOT}/{group}/{data}'
    out = f'output/{data}'

    t_start = time.time()
    os.system(f'python train.py -s {src} -m {out} --budget {budget} --profile_components')
    t_end = time.time()
    elapsed = int(t_end - t_start)
    train_time_str = f'{elapsed // 60}分{elapsed % 60}秒'
    os.system(f'python render.py -m {out}')
    os.system(f'python metrics.py -m {out}')

    # 4. FPS benchmark (pure rendering speed, not including image save/metrics/Conf)
    fps_json_path = os.path.join(out, 'fps_benchmark.json')
    # 删除可能存在的过期 fps_benchmark.json，确保使用最新结果
    if os.path.exists(fps_json_path):
        os.remove(fps_json_path)
    fps_cmd = f'python bench_fps.py -m {out} --fps_timing_mode batch'
    result = subprocess.run(fps_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] bench_fps.py failed with code {result.returncode}: {result.stderr[:200]}")
    if os.path.exists(fps_json_path):
        with open(fps_json_path, 'r') as fp:
            fps_data = json.load(fp)
        scene_metrics['fps'] = fps_data.get('fps', 0.0)
        scene_metrics['fps_latency_ms'] = fps_data.get('average_ms_per_frame', 0.0)
    else:
        print(f"  [WARN] {out}/fps_benchmark.json not found — did bench_fps.py run?")

    # Collect metrics for this scene
    scene_metrics = {}
    scene_metrics['train_time'] = train_time_str

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
