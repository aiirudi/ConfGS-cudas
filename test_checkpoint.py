"""Conf checkpoint round trip and legacy checkpoint compatibility."""

from types import SimpleNamespace

import torch

from scene.gaussian_model import GaussianModel


def make_model(n=3):
    model = GaussianModel(sh_degree=0)
    def parameter(*shape):
        return torch.nn.Parameter(torch.zeros(*shape, device='cuda'))
    model._xyz = parameter(n, 3)
    model._features_dc = parameter(n, 1, 3)
    model._features_rest = parameter(n, 0, 3)
    model._scaling = parameter(n, 3)
    model._rotation = parameter(n, 4)
    model._opacity = parameter(n, 1)
    model.spatial_lr_scale = 1.0
    return model


def training_args():
    return SimpleNamespace(
        percent_dense=0.01, position_lr_init=0.00016,
        position_lr_final=0.0000016, position_lr_delay_mult=0.01,
        position_lr_max_steps=30000, feature_lr=0.0025,
        shfeature_lr=0.0025, opacity_lr=0.05,
        scaling_lr=0.005, rotation_lr=0.001,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this validation')
    args = training_args()
    source = make_model()
    source.set_conf_camera_mapping({1: 'view-one', 4: 'view-four'})
    source.training_setup(args)
    source.conf_world_sum[0, 0] = 0.5
    source.conf_norm_sum[0, 0] = 1.5
    source.conf_view_count[0, 0] = 2
    source.conf_score[0, 0] = 2.0 / 3.0
    source.conf_history[0, 0] = torch.tensor([1.0, 0.0, 0.0, 1.0], device='cuda')
    source.conf_history[0, 1] = torch.tensor([-0.5, 0.0, 0.0, 0.5], device='cuda')
    source.conf_camera_ids[0, :2] = torch.tensor([1, 4], device='cuda')
    state = source.capture()
    assert len(state) == 13
    assert state[12]['version'] == 3

    restored = make_model()
    restored.set_conf_camera_mapping({1: 'view-one', 4: 'view-four'})
    restored.restore(state, args)
    assert torch.equal(restored.conf_world_sum, source.conf_world_sum)
    assert torch.equal(restored.conf_norm_sum, source.conf_norm_sum)
    assert torch.equal(restored.conf_view_count, source.conf_view_count)
    assert torch.equal(restored.conf_score, source.conf_score)
    assert torch.equal(restored.conf_history, source.conf_history)
    assert torch.equal(restored.conf_camera_ids, source.conf_camera_ids)
    assert restored.conf_camera_mapping == source.conf_camera_mapping

    # Upgrade v2 from its authoritative ordered history, including a cache
    # already corrupted by float32 overflow. Casting the cache is insufficient.
    old_conf = dict(state[12], version=2)
    old_conf['world_sum'] = source.conf_world_sum.float()
    old_conf['norm_sum'] = source.conf_norm_sum.float()
    old_conf['history'] = source.conf_history.clone()
    old_conf['history'][0, 0] = torch.tensor([2e38, 0., 0., 2e38], device='cuda')
    old_conf['history'][0, 1] = torch.tensor([-2e38, 0., 0., 2e38], device='cuda')
    old_conf['norm_sum'][0] = float('inf')
    old_conf['score'] = source.conf_score.clone()
    old_conf['score'][0] = 0.
    upgraded = make_model()
    upgraded.restore(state[:12] + (old_conf,), args)
    assert upgraded.conf_world_sum.dtype == torch.float64
    assert upgraded.conf_norm_sum.dtype == torch.float64
    assert upgraded.conf_score[0].item() == 1.
    assert torch.isfinite(upgraded.conf_norm_sum).all()
    assert torch.equal(upgraded.conf_camera_ids, source.conf_camera_ids)
    assert torch.equal(upgraded.conf_history, old_conf['history'])
    assert torch.equal(upgraded.conf_view_count, source.conf_view_count)
    # A refresh after migration must retain the same oldest/newest ordering.
    replacement = torch.zeros((3, 4), device='cuda')
    replacement[:, 3] = -1.
    replacement[0] = torch.tensor([1., 0., 0., 1.], device='cuda')
    upgraded.add_conf_stats(replacement, 1, upgraded.conf_topology_version)
    assert upgraded.conf_camera_ids[0].tolist() == [4, 1, -1]

    # Reject malformed native-kernel state before it can reach CUDA.
    corruptions = (
        ('view_count', torch.tensor([[4], [0], [0]], dtype=torch.int32, device='cuda')),
        ('camera_ids', torch.tensor([[1, 1, -1], [-1, -1, -1], [-1, -1, -1]],
                                    dtype=torch.int64, device='cuda')),
        ('camera_ids', torch.tensor([[1, 99, -1], [-1, -1, -1], [-1, -1, -1]],
                                    dtype=torch.int64, device='cuda')),
    )
    for key, value in corruptions:
        invalid = make_model()
        try:
            invalid.restore(state[:12] + (dict(state[12], **{key: value}),), args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Malformed Conf checkpoint {key} was accepted')

    wrong_scene = make_model()
    wrong_scene.set_conf_camera_mapping({1: 'another-view', 4: 'view-four'})
    try:
        wrong_scene.restore(state, args)
    except ValueError as error:
        assert 'camera mapping differs' in str(error)
    else:
        raise AssertionError('Mismatched camera mapping was accepted')

    unmapped_v2 = make_model()
    unmapped_v2.restore(state[:12] + ({key: value for key, value in state[12].items()
                                      if key != 'camera_mapping'},), args)
    assert torch.all(unmapped_v2.conf_camera_ids == -1)

    stale_version = restored.conf_topology_version
    restored.reset_conf_window()  # Same Gaussian count, different history generation.
    try:
        restored.add_conf_stats(torch.zeros((3, 4), device='cuda'), 1, stale_version)
    except ValueError as error:
        assert 'stale Gaussian topology' in str(error)
    else:
        raise AssertionError('Stale Conf samples were accepted')

    legacy = make_model()
    legacy.restore(state[:12], args)
    assert torch.all(legacy.conf_camera_ids == -1)
    assert torch.count_nonzero(legacy.conf_view_count) == 0

    v1 = make_model()
    v1.restore(state[:12] + ({'version': 1, 'world_sum': source.conf_world_sum},), args)
    assert torch.all(v1.conf_camera_ids == -1)
    assert torch.count_nonzero(v1.conf_view_count) == 0
    print('Conf checkpoint compatibility: PASS')


if __name__ == '__main__':
    main()
