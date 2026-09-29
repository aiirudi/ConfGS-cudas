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

    restored = make_model()
    restored.restore(state, args)
    assert torch.equal(restored.conf_world_sum, source.conf_world_sum)
    assert torch.equal(restored.conf_norm_sum, source.conf_norm_sum)
    assert torch.equal(restored.conf_view_count, source.conf_view_count)
    assert torch.equal(restored.conf_score, source.conf_score)
    assert torch.equal(restored.conf_history, source.conf_history)
    assert torch.equal(restored.conf_camera_ids, source.conf_camera_ids)

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
