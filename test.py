import os
import json

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
    # Tanks & Temples
    ['train',     'tt',      1250_000],
    ['truck',     'tt',      1250_000],
]

DATA_ROOT = '/workspace/dataset/'
ALL_METRICS = {}

for data, group, budget in paramList:
    src = f'{DATA_ROOT}/{group}/{data}'
    out = f'output/{data}'

    os.system(f'python train.py -s {src} -m {out} --budget {budget} --profile_components')
    os.system(f'python render.py -m {out}')
    os.system(f'python metrics.py -m {out}')

    # Collect metrics for this scene
    scene_metrics = {}

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

    # 3. SSIM and LPIPS (from metrics.py results.json)
    # results.json structure: {method: {"SSIM":..., "PSNR":..., "LPIPS":...}}
    # Select the ours_* method with the highest iteration (most recent render)
    results_path = os.path.join(out, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as fp:
            results = json.load(fp)
        ours_methods = [m for m in results.keys() if m.startswith('ours_')]
        if ours_methods:
            target = sorted(ours_methods)[-1]  # highest iteration
            scene_metrics['SSIM'] = results[target].get('SSIM', 0.0)
            scene_metrics['LPIPS'] = results[target].get('LPIPS', 0.0)

    ALL_METRICS[data] = scene_metrics
    print(f"  Collected {data}: {scene_metrics}")

# Write all scene metrics to a single JSON
metrics_path = 'metrics.json'
with open(metrics_path, 'w') as fp:
    json.dump(ALL_METRICS, fp, indent=True)
print(f"\nAll metrics saved to {metrics_path}")
