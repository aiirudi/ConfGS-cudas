"""Small CUDA Conf accumulator validation; exits nonzero on a failed check."""

import math

import torch

from diff_gaussian_rasterization import accumulate_conf


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this validation')

    samples = torch.tensor([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [float('nan'), 0.0, 0.0, 1.0],
    ], device='cuda')
    history = torch.zeros(3, 3, 4, device='cuda')
    camera_ids = torch.full((3, 3), -1, device='cuda', dtype=torch.int64)
    world_sum = torch.zeros(3, 3, device='cuda', dtype=torch.float64)
    norm_sum = torch.zeros(3, 1, device='cuda', dtype=torch.float64)
    count = torch.zeros(3, 1, device='cuda', dtype=torch.int32)
    score = torch.zeros(3, 1, device='cuda')

    accumulate_conf(samples, 1, history, camera_ids, count, world_sum, norm_sum, score)
    assert count[:, 0].tolist() == [1, 1, 0]
    assert score[:, 0].tolist() == [0.0, 0.0, 0.0]

    opposite = samples.clone()
    opposite[0, 0] = -1.0
    accumulate_conf(opposite, 2, history, camera_ids, count, world_sum, norm_sum, score)
    assert count[:, 0].tolist() == [2, 2, 0]
    assert math.isclose(score[0, 0].item(), 1.0, abs_tol=1e-6)
    assert torch.all(score[1:] == 0)

    # One new camera evicts the oldest per-Gaussian observation at W=3.
    aligned = torch.tensor([[1.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, -1.0],
                            [0.0, 0.0, 0.0, -1.0]], device='cuda')
    accumulate_conf(aligned, 3, history, camera_ids, count, world_sum, norm_sum, score)
    accumulate_conf(aligned, 4, history, camera_ids, count, world_sum, norm_sum, score)
    assert count[:, 0].tolist() == [3, 2, 0]
    assert camera_ids[0].tolist() == [2, 3, 4]
    assert math.isclose(score[0, 0].item(), 2.0 / 3.0, abs_tol=1e-6)

    # A very small coherent gradient must remain at Conf=0. No fixed
    # denominator epsilon is permitted to push it toward one.
    tiny = torch.tensor([[1e-9, 0.0, 0.0, 1e-9]], device='cuda')
    h = torch.zeros(1, 3, 4, device='cuda')
    ids = torch.full((1, 3), -1, device='cuda', dtype=torch.int64)
    s = torch.zeros(1, 3, device='cuda', dtype=torch.float64)
    m = torch.zeros(1, 1, device='cuda', dtype=torch.float64)
    k = torch.zeros(1, 1, device='cuda', dtype=torch.int32)
    c = torch.zeros(1, 1, device='cuda')
    accumulate_conf(tiny, 1, h, ids, k, s, m, c)
    accumulate_conf(tiny, 2, h, ids, k, s, m, c)
    assert k.item() == 2
    assert math.isclose(c.item(), 0.0, abs_tol=1e-6)
    print('CUDA Conf statistics: PASS')


if __name__ == '__main__':
    main()
