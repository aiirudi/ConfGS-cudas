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
    world_sum = torch.zeros(3, 3, device='cuda')
    norm_sum = torch.zeros(3, 1, device='cuda')
    count = torch.zeros(3, 1, device='cuda', dtype=torch.int32)
    score = torch.zeros(3, 1, device='cuda')

    accumulate_conf(samples, world_sum, norm_sum, count, score)
    assert count[:, 0].tolist() == [1, 0, 0]
    assert score[:, 0].tolist() == [0.0, 0.0, 0.0]

    opposite = samples.clone()
    opposite[0, 0] = -1.0
    accumulate_conf(opposite, world_sum, norm_sum, count, score)
    assert count[:, 0].tolist() == [2, 0, 0]
    assert math.isclose(score[0, 0].item(), 1.0, abs_tol=1e-6)
    assert torch.all(score[1:] == 0)

    # A very small coherent gradient must remain at Conf=0. No fixed
    # denominator epsilon is permitted to push it toward one.
    tiny = torch.tensor([[1e-9, 0.0, 0.0, 1e-9]], device='cuda')
    s = torch.zeros(1, 3, device='cuda')
    m = torch.zeros(1, 1, device='cuda')
    k = torch.zeros(1, 1, device='cuda', dtype=torch.int32)
    c = torch.zeros(1, 1, device='cuda')
    accumulate_conf(tiny, s, m, k, c)
    accumulate_conf(tiny, s, m, k, c)
    assert k.item() == 2
    assert math.isclose(c.item(), 0.0, abs_tol=1e-6)
    print('CUDA Conf statistics: PASS')


if __name__ == '__main__':
    main()
