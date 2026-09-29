"""Production Conf eligibility is independent of the obsolete abs gradient."""

from types import SimpleNamespace

import torch

from scene.gaussian_model import GaussianModel


def make_model():
    model = GaussianModel(sh_degree=0)
    model._xyz = torch.zeros(4, 3)
    model._opacity = torch.zeros(4, 1)
    model.conf_score = torch.tensor([[0.9], [0.9], [0.95], [0.1]])
    model.conf_view_count = torch.tensor([[2], [1], [3], [4]], dtype=torch.int32)
    model.conf_norm_sum = torch.ones(4, 1)
    model.xyz_gradient_accum = torch.ones(4, 1) * 999.0
    model.denom = torch.ones(4, 1)
    selected = []
    model.long_axis_split = lambda scores, budget, mask, *args: selected.append((scores.clone(), mask.clone(), budget))
    model.prune_points = lambda mask: None
    return model, selected


def options(**kwargs):
    values = dict(
        candidate_selection_strategy='conf_only', conf_thr=0.85,
        conf_min_views=2, candidate_budget_mode='native',
        enable_spatial_diversity=False, densify_grad_threshold=1e9,
        split_distance=0.5, opacity_reduction=0.6,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_conf_gate_remains_active_after_14500():
    model, selected = make_model()
    ranking = torch.tensor([0.2, 100.0, 0.8, 50.0])
    model.densify_and_prune_Improved(ranking, 0.005, 6, options(), 14501, 6)
    assert len(selected) == 1
    scores, mask, budget = selected[0]
    assert torch.equal(mask, torch.tensor([True, False, True, False]))
    assert torch.equal(scores, ranking)
    assert budget == 2


def test_fixed_budget_stays_inside_conf_mask():
    model, selected = make_model()
    opt = options(candidate_budget_mode='fixed',
                  candidate_budget_reference='fixed_number',
                  candidate_fixed_budget=1)
    model.densify_and_prune_Improved(torch.tensor([0.2, 100.0, 0.8, 50.0]),
                                    0.005, 6, opt, 1000, 6)
    assert len(selected) == 1
    assert torch.equal(selected[0][1], torch.tensor([False, False, True, False]))


def test_abs_strategy_reports_incompatibility():
    model, _ = make_model()
    try:
        model.densify_and_prune_Improved(None, 0.005, 6,
                                        options(candidate_selection_strategy='and'),
                                        1000, 6)
    except ValueError as error:
        assert 'conf_only' in str(error)
    else:
        raise AssertionError('abs-dependent strategy should be rejected')
