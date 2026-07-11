import os
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

for data, group, budget in paramList:
    src = f'{DATA_ROOT}/{group}/{data}'
    out = f'output/{data}'

    os.system(f'python train.py -s {src} -m {out} --budget {budget}')
    os.system(f'python render.py -m {out}')
    os.system(f'python metrics.py -m {out}')
    #os.system(f'python metrics-train.py -m {out}')
