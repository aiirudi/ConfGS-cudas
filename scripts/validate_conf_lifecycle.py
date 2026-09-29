"""GPU lifecycle checks for a real GaussianModel and rolling CUDA Conf state.

Run with the rebuilt rasterizer and simple-knn on PYTHONPATH::

    python scripts/validate_conf_lifecycle.py

This script exercises topology mutations, checkpoint continuation and the
late-training candidate gate. It also compares the three ranking helpers'
ASTs with the saved pre-Conf train.py. A failed assertion exits nonzero.
"""

import ast
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_checkpoint import make_model, training_args  # noqa: E402


CAMERAS = {11: 'camera-11', 12: 'camera-12',
           13: 'camera-13', 14: 'camera-14'}
SEED_VIEWS = (
    (11, ((1., 0., 0., 1.), (1., 0., 0., 1.), (0., 0., 0., -1.))),
    (12, ((-1., 0., 0., 1.), (1., 0., 0., 1.), (0., 1., 0., 1.))),
    (13, ((0., 0., 0., 0.), (1., 0., 0., 1.), (0., 0., 0., -1.))),
)
FOURTH_VIEW = (14, ((0., 1., 0., 1.), (1., 0., 0., 1.),
                    (1., 0., 0., 1.)))
STATE_NAMES = ('conf_history', 'conf_camera_ids', 'conf_view_count',
               'conf_world_sum', 'conf_norm_sum', 'conf_score')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def fresh_model():
    model = make_model(3)
    with torch.no_grad():
        model._xyz[:] = torch.tensor([[0., 0., 3.], [2., 0., 3.],
                                      [4., 0., 3.]], device='cuda')
        model._rotation[:, 0] = 1.  # Unit quaternion for real LAS splitting.
    model.set_conf_camera_mapping(CAMERAS)
    model.training_setup(training_args())
    return model


def reference_push(rows, camera_id, samples, window):
    for row, sample in zip(rows, samples):
        x, y, z, valid = sample
        if valid < 0:
            continue
        row[:] = [entry for entry in row if entry[0] != camera_id]
        if len(row) == window:
            del row[0]
        row.append((camera_id, (x, y, z), math.hypot(math.hypot(x, y), z)))


def add_view(model, camera_id, samples, rows=None):
    tensor = torch.tensor(samples, dtype=torch.float32, device='cuda')
    model.add_conf_stats(tensor, camera_id, model.conf_topology_version)
    if rows is not None:
        reference_push(rows, camera_id, samples, model.conf_window_size)


def seeded_model():
    model = fresh_model()
    rows = [[], [], []]
    for camera_id, samples in SEED_VIEWS:
        add_view(model, camera_id, samples, rows)
    check_reference(model, rows)
    return model, rows


def check_reference(model, rows):
    torch.cuda.synchronize()
    for index, row in enumerate(rows):
        count = int(model.conf_view_count[index, 0])
        require(count == len(row), f'row {index}: occupied count {count}')
        ids = model.conf_camera_ids[index].cpu().tolist()
        require(ids == [entry[0] for entry in row] +
                [-1] * (model.conf_window_size - len(row)),
                f'row {index}: camera order {ids}')
        for slot, (_, xyz, norm) in enumerate(row):
            actual = model.conf_history[index, slot].cpu().tolist()
            require(all(math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-6)
                        for a, b in zip(actual, (*xyz, norm))),
                    f'row {index}, slot {slot}: history {actual}')
        vector = [sum(entry[1][axis] for entry in row) for axis in range(3)]
        total_norm = sum(entry[2] for entry in row)
        actual_sum = model.conf_world_sum[index].cpu().tolist()
        require(all(math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-6)
                    for a, b in zip(actual_sum, vector)),
                f'row {index}: world sum {actual_sum}')
        require(math.isclose(model.conf_norm_sum[index, 0].item(), total_norm,
                             abs_tol=1e-6, rel_tol=1e-6),
                f'row {index}: norm sum')
        expected_score = 0.
        if len(row) >= 2 and total_norm > 0:
            expected_score = max(0., min(1., 1. - math.hypot(*vector) / total_norm))
        require(math.isclose(model.conf_score[index, 0].item(), expected_score,
                             abs_tol=1e-6, rel_tol=1e-6),
                f'row {index}: Conf score')


def snapshot(model):
    return {name: getattr(model, name).clone() for name in STATE_NAMES}


def check_survivors(model, state, indices):
    for name in STATE_NAMES:
        actual = getattr(model, name)[:len(indices)]
        expected = state[name][indices]
        require(torch.equal(actual, expected),
                f'{name}: survivor history changed or reordered')


def check_empty_new_rows(model, first):
    require(model.get_xyz.shape[0] > first, 'no child row was appended')
    for name in STATE_NAMES:
        tensor = getattr(model, name)[first:]
        expected = (-torch.ones_like(tensor) if name == 'conf_camera_ids'
                    else torch.zeros_like(tensor))
        require(torch.equal(tensor, expected),
                f'{name}: new row inherited old Conf history')


def test_prune():
    model, _ = seeded_model()
    before = snapshot(model)
    old_version = model.conf_topology_version
    model.prune_points(torch.tensor([False, True, False], device='cuda'))
    require(model.get_xyz.shape[0] == 2, 'prune did not remove one Gaussian')
    require(model.conf_topology_version > old_version,
            'prune did not advance topology generation')
    check_survivors(model, before, [0, 2])
    print('Pure prune survivor history: PASS')


def test_append():
    model, _ = seeded_model()
    before = snapshot(model)
    old_version = model.conf_topology_version
    args = (model._xyz[:1].detach().clone(),
            model._features_dc[:1].detach().clone(),
            model._features_rest[:1].detach().clone(),
            model._opacity[:1].detach().clone(),
            model._scaling[:1].detach().clone(),
            model._rotation[:1].detach().clone())
    model.densification_postfix(*args)
    require(model.get_xyz.shape[0] == 4, 'postfix did not append one Gaussian')
    require(model.conf_topology_version > old_version,
            'postfix did not advance topology generation')
    check_survivors(model, before, [0, 1, 2])
    check_empty_new_rows(model, 3)
    print('Postfix survivor and child history: PASS')


def test_real_split():
    model, _ = seeded_model()
    before = snapshot(model)
    old_version = model.conf_topology_version
    model.long_axis_split(torch.tensor([1., 0., 0.], device='cuda'), 1,
                          torch.tensor([True, False, False], device='cuda'),
                          split_distance=0.2, opacity_reduction=0.8)
    require(model.get_xyz.shape[0] == 4,
            'one parent should be replaced by two children')
    require(model.conf_topology_version > old_version,
            'split did not advance topology generation')
    check_survivors(model, before, [1, 2])
    check_empty_new_rows(model, 2)
    require(torch.equal(model.get_xyz[:2].detach(),
                        torch.tensor([[2., 0., 3.], [4., 0., 3.]], device='cuda')),
            'the selected parent survived or another Gaussian was removed')
    print('Real long-axis split parent/child lifecycle: PASS')


def options():
    return SimpleNamespace(candidate_selection_strategy='conf_only',
                           candidate_budget_mode='native', conf_min_views=2,
                           conf_thr=0.85, split_distance=0.2,
                           opacity_reduction=0.8,
                           enable_spatial_diversity=False)


def test_no_split_boundaries():
    model, _ = seeded_model()
    before = snapshot(model)
    scores = torch.tensor([0.1, 100., 100.], device='cuda')
    model.densify_and_prune_Improved(scores, min_opacity=0.,
                                      budget=3, opt=options(), iteration=1000,
                                      limitation=None)
    require(model.get_xyz.shape[0] == 3, 'zero additional budget changed count')
    check_survivors(model, before, [0, 1, 2])
    require(model.candidate_stats['num_split'] == 0,
            'zero additional budget created a split')

    # A positive budget with no finite ranking scores also leaves the rolling
    # history intact across this interval boundary.
    no_scores = torch.full((3,), float('nan'), device='cuda')
    model.densify_and_prune_Improved(no_scores, min_opacity=0.,
                                      budget=4, opt=options(), iteration=1100,
                                      limitation=None)
    require(model.get_xyz.shape[0] == 3, 'empty selection changed count')
    check_survivors(model, before, [0, 1, 2])
    require(model.candidate_stats['n_final_candidates'] == 0,
            'nonfinite rankings remained eligible')
    print('Zero-budget and no-split intervals preserve history: PASS')


def test_late_conf_gate():
    model, _ = seeded_model()
    before = snapshot(model)
    scores = torch.tensor([0.1, 100., 100.], device='cuda')
    model.densify_and_prune_Improved(scores, min_opacity=0.,
                                      budget=4, opt=options(), iteration=14600,
                                      limitation=None)
    stats = model.candidate_stats
    require(stats['n_conf_candidates'] == 1 and
            stats['n_final_candidates'] == 1 and stats['num_split'] == 1,
            f'late iteration bypassed Conf gate: {stats}')
    require(model.get_xyz.shape[0] == 4, 'late Conf selection did not split')
    check_survivors(model, before, [1, 2])
    check_empty_new_rows(model, 2)
    print('Iteration >14500 still uses Conf eligibility: PASS')


def test_stale_generation():
    model, _ = seeded_model()
    old_version = model.conf_topology_version
    model.reset_conf_window()  # Same N, new generation.
    require(model.get_xyz.shape[0] == 3, 'reset unexpectedly changed topology')
    try:
        model.add_conf_stats(torch.zeros((3, 4), device='cuda'), 11,
                             old_version)
    except ValueError as error:
        require('stale Gaussian topology' in str(error),
                f'wrong stale-sample error: {error}')
    else:
        raise AssertionError('old samples were accepted after same-N reset')
    print('Same-N stale topology samples rejected: PASS')


def test_checkpoint_continuation():
    source, rows = seeded_model()
    buffer = io.BytesIO()
    torch.save((source.capture(), 900), buffer)
    buffer.seek(0)
    checkpoint, iteration = torch.load(buffer, map_location='cuda')
    require(iteration == 900, 'checkpoint iteration changed')
    restored = make_model(3)
    restored.set_conf_camera_mapping(CAMERAS)
    restored.restore(checkpoint, training_args())
    require(restored.conf_camera_mapping == source.conf_camera_mapping,
            'restored camera mapping changed')
    check_survivors(restored, snapshot(source), [0, 1, 2])
    camera_id, samples = FOURTH_VIEW
    add_view(source, camera_id, samples, rows)
    add_view(restored, camera_id, samples)
    check_reference(source, rows)
    check_reference(restored, rows)
    check_survivors(restored, snapshot(source), [0, 1, 2])
    require(model_ids(restored, 0) == [12, 13, 14],
            'resume did not evict the oldest row-0 view')
    print('Checkpoint roundtrip and next-view eviction: PASS')


def model_ids(model, row):
    count = int(model.conf_view_count[row, 0])
    return model.conf_camera_ids[row, :count].cpu().tolist()


def test_ranking_helpers_ast():
    baseline_path = Path('/tmp/conf_cuda_baseline/train.py')
    if not baseline_path.is_file():
        raise FileNotFoundError(f'Pre-Conf train.py missing: {baseline_path}')

    def functions(path):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        return {node.name: ast.dump(node, include_attributes=False)
                for node in tree.body if isinstance(node, ast.FunctionDef)}

    baseline = functions(baseline_path)
    current = functions(ROOT / 'train.py')
    for name in ('compute_rf_score1', 'compute_edge_score',
                 'fuse_importance_scores'):
        require(name in baseline and name in current,
                f'missing ranking helper {name}')
        require(baseline[name] == current[name],
                f'{name} source AST changed from the pre-Conf baseline')
    print('RFAS, EAS, and fusion helper ASTs unchanged: PASS')


def main():
    test_ranking_helpers_ast()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Conf lifecycle validation')
    test_prune()
    test_append()
    test_real_split()
    test_no_split_boundaries()
    test_late_conf_gate()
    test_stale_generation()
    test_checkpoint_continuation()
    print('CUDA Conf lifecycle validation: PASS')


if __name__ == '__main__':
    main()
