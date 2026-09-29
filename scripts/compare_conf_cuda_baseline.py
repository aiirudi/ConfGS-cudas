"""Compare the original rasterizer with CUDA Conf in isolated processes.

Run after rebuilding the new extension, for example::

    python scripts/compare_conf_cuda_baseline.py \
      --baseline-wrapper /tmp/conf_cuda_baseline_source/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization \
      --baseline-lib /tmp/conf_cuda_baseline_source/build-lib/diff_gaussian_rasterization \
      --new-wrapper submodules/diff-gaussian-rasterization/diff_gaussian_rasterization \
      --new-lib /tmp/conf_cuda_sliding_build_20260930/lib/diff_gaussian_rasterization

The parent compares CPU result files. Each child loads exactly one Python
wrapper and one matching _C binary, avoiding same-name extension collisions.
The original extension sees only full 32x32 tiles: its inherited backward
prefetch has out-of-bounds reads on partial tiles.
"""

import argparse
import importlib.util
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path('/tmp/conf_cuda_baseline_source')
NEW_BUILD = Path('/tmp/conf_cuda_sliding_build_20260930')
MODES = ('color_scale', 'sh_scale', 'color_cov', 'sh_cov')


def package_from(wrapper_dir, binary_dir):
    wrapper_dir = wrapper_dir.resolve()
    binary_dir = binary_dir.resolve()
    if not (wrapper_dir / '__init__.py').is_file():
        raise FileNotFoundError(f'No rasterizer wrapper at {wrapper_dir}')
    if not list(binary_dir.glob('_C*.so')):
        raise FileNotFoundError(f'No compiled rasterizer _C at {binary_dir}')
    name = 'diff_gaussian_rasterization'
    spec = importlib.util.spec_from_file_location(
        name, wrapper_dir / '__init__.py',
        submodule_search_locations=[str(binary_dir), str(wrapper_dir)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package


def camera(height, width):
    """Rotated, translated row-vector camera with the renderer's projection."""
    yaw, roll = 0.16, -0.19
    cy, sy, cr, sr = (math.cos(yaw), math.sin(yaw),
                      math.cos(roll), math.sin(roll))
    turn = torch.tensor([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    bank = torch.tensor([[cr, -sr, 0.], [sr, cr, 0.], [0., 0., 1.]])
    rotation = turn @ bank
    translation = torch.tensor([0.12, -0.09, 0.15])
    view = torch.eye(4)
    view[:3, :3] = rotation
    view[3, :3] = translation

    fovx, fovy, near, far = 1.07, 0.89, 0.01, 100.0
    projection = torch.zeros((4, 4))
    projection[0, 0] = 1. / math.tan(fovx / 2)
    projection[1, 1] = 1. / math.tan(fovy / 2)
    projection[2, 2] = far / (far - near)
    projection[2, 3] = 1.
    projection[3, 2] = -far * near / (far - near)
    full = view @ projection
    center = torch.linalg.inv(view)[3, :3]
    return (view.cuda().contiguous(), full.cuda().contiguous(),
            center.cuda().contiguous(), fovx, fovy, rotation, translation)


def trial(package, mode, height, width, collect_conf, zero_upstream=False,
          all_culled=False, debug=False):
    generator = torch.Generator(device='cpu').manual_seed(20260930)
    view, full, center, fovx, fovy, rotation, translation = camera(height, width)
    positions_camera = torch.tensor([
        [0.00, 0.00, 2.3], [0.31, 0.18, 2.5],
        [-0.38, -0.17, 2.7], [0.11, -0.34, 3.0],
        [0.48, 0.22, 3.2], [-0.73, 0.20, 2.8],
        [15.0, 0.0, 2.5],  # Frustum-culling control: unseen sample.
    ])
    if all_culled:
        positions_camera[:, 0] = 20.0
    means_cpu = (positions_camera - translation) @ rotation.T
    n = means_cpu.shape[0]

    def leaf(values):
        return values.detach().to('cuda').contiguous().requires_grad_(True)

    means = leaf(means_cpu)
    # The native backward returns (N,4), so the auxiliary leaf has that shape.
    means2d = leaf(torch.zeros((n, 4)))
    opacity = leaf(0.56 + 0.30 * torch.rand((n, 1), generator=generator))
    scales = leaf(0.15 + 0.07 * torch.rand((n, 3), generator=generator))
    quaternions = torch.zeros((n, 4))
    quaternions[:, 0] = 1.
    quaternions[:, 1:] = 0.05 * torch.randn((n, 3), generator=generator)
    rotations = leaf(quaternions)
    covariance = torch.zeros((n, 6))
    covariance[:, 0] = 0.036
    covariance[:, 3] = 0.040
    covariance[:, 5] = 0.033
    covariance[:, 1] = 0.001
    covariance = leaf(covariance)
    colors = leaf(0.15 + 0.65 * torch.rand((n, 3), generator=generator))
    dc = leaf(0.11 * torch.randn((n, 1, 3), generator=generator))
    sh = leaf(0.04 * torch.randn((n, 15, 3), generator=generator))

    use_sh, use_cov = mode.startswith('sh_'), mode.endswith('_cov')
    settings = package.GaussianRasterizationSettings(
        image_height=height, image_width=width,
        tanfovx=math.tan(fovx / 2), tanfovy=math.tan(fovy / 2),
        bg=torch.tensor([0.03, 0.05, 0.07], device='cuda'),
        scale_modifier=1., viewmatrix=view, projmatrix=full,
        sh_degree=1, campos=center, prefiltered=False, debug=debug,
        pixel_weights=torch.empty((0,), device='cuda'))
    rasterizer = package.GaussianRasterizer(settings)
    holder = {} if collect_conf else None
    kwargs = dict(means3D=means, means2D=means2d, opacities=opacity,
                  dc=dc if use_sh else None, shs=sh if use_sh else None,
                  colors_precomp=None if use_sh else colors,
                  scales=None if use_cov else scales,
                  rotations=None if use_cov else rotations,
                  cov3D_precomp=covariance if use_cov else None)
    if collect_conf:
        kwargs['conf_stats'] = holder
    outputs = rasterizer(**kwargs)
    image, radii = outputs[0], outputs[1]
    upstream = torch.randn(image.shape, generator=generator).cuda()
    if zero_upstream:
        upstream.zero_()
    (image * upstream).sum().backward()
    torch.cuda.synchronize()

    active = dict(xyz=means, opacity=opacity, means2d=means2d)
    active.update(dict(dc=dc, sh=sh) if use_sh else dict(colors=colors))
    active.update(dict(covariance=covariance) if use_cov else
                  dict(scales=scales, rotation=rotations))
    gradients = {key: (tensor.grad if tensor.grad is not None else
                       torch.zeros_like(tensor)).detach().cpu()
                 for key, tensor in active.items()}
    return dict(image=image.detach().cpu(), radii=radii.detach().cpu(),
                grads=gradients, means=means.detach().cpu(),
                projection=full.detach().cpu(),
                samples=None if holder is None else holder['samples'].detach().cpu())


def occluded_trial(package):
    """Keep the rear splat in the frustum while front splats stop every ray."""
    height, width = 23, 31
    view, full, center, fovx, fovy, rotation, translation = camera(height, width)
    camera_xyz = torch.tensor([[0., 0., 1.2], [0., 0., 1.3],
                               [0., 0., 1.4], [0., 0., 3.]])
    means = ((camera_xyz - translation) @ rotation.T).cuda().requires_grad_(True)
    means2d = torch.zeros((4, 4), device='cuda', requires_grad=True)
    opacity = torch.tensor([[1.], [1.], [1.], [0.9]], device='cuda',
                           requires_grad=True)
    scales = torch.tensor([[20., 20., 20.]] * 3 + [[0.16, 0.16, 0.16]],
                          device='cuda', requires_grad=True)
    quaternions = torch.zeros((4, 4), device='cuda')
    quaternions[:, 0] = 1.
    quaternions.requires_grad_(True)
    colors = torch.tensor([[0.4, 0.2, 0.1]] * 4, device='cuda',
                          requires_grad=True)
    settings = package.GaussianRasterizationSettings(
        image_height=height, image_width=width,
        tanfovx=math.tan(fovx / 2), tanfovy=math.tan(fovy / 2),
        bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
        viewmatrix=view, projmatrix=full, sh_degree=0, campos=center,
        prefiltered=False, debug=False,
        pixel_weights=torch.empty((0,), device='cuda'))
    holder = {}
    image, radii, *_ = package.GaussianRasterizer(settings)(
        means3D=means, means2D=means2d, opacities=opacity,
        colors_precomp=colors, scales=scales, rotations=quaternions,
        conf_stats=holder)
    image.sum().backward()
    torch.cuda.synchronize()
    return dict(radii=radii.detach().cpu(), samples=holder['samples'].detach().cpu())


def child(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for rasterizer comparison')
    package = package_from(args.wrapper, args.binary)
    cases = {}
    for mode in MODES:
        cases[mode] = trial(package, mode, 32, 32,
                            collect_conf=(args.role == 'new'))
    if args.role == 'new':
        cases['debug_color_scale'] = trial(package, 'color_scale', 32, 32,
                                            collect_conf=True, debug=True)
        cases['off_color_scale'] = trial(package, 'color_scale', 32, 32,
                                         collect_conf=False)
        cases['partial'] = trial(package, 'color_scale', 23, 31,
                                 collect_conf=True)
        cases['zero'] = trial(package, 'color_scale', 23, 31,
                              collect_conf=True, zero_upstream=True)
        cases['all_culled'] = trial(package, 'color_scale', 23, 31,
                                    collect_conf=True, all_culled=True)
        cases['occluded'] = occluded_trial(package)
    torch.save(cases, args.output)


def compare_tensor(label, old, new, atol, rtol):
    if old.shape != new.shape:
        raise AssertionError(f'{label}: shape {old.shape} != {new.shape}')
    if not torch.isfinite(old).all() or not torch.isfinite(new).all():
        raise AssertionError(f'{label}: nonfinite output')
    error = (old - new).abs().max().item() if old.numel() else 0.
    print(f'{label}: max abs error {error:.6g}')
    if not torch.allclose(old, new, atol=atol, rtol=rtol):
        raise AssertionError(f'{label}: exceeds atol={atol}, rtol={rtol}')


def check_vjp(label, result):
    samples = result['samples']
    if samples is None or samples.shape != (result['means'].shape[0], 4):
        raise AssertionError(f'{label}: missing Conf samples')
    if not torch.isfinite(samples[samples[:, 3] >= 0]).all():
        raise AssertionError(f'{label}: nonfinite valid Conf sample')
    if samples[-1, 3].item() != -1.:
        raise AssertionError(f'{label}: frustum-culled Gaussian advanced')
    if not torch.equal(result['grads']['means2d'][:, 2:],
                       torch.zeros_like(result['grads']['means2d'][:, 2:])):
        raise AssertionError(f'{label}: legacy abs z/w accumulation remains')
    matrix = result['projection'].double()
    largest = 0.
    largest_fd = 0.
    visible = 0
    for i, sample in enumerate(samples):
        if sample[3] < 0:
            continue
        visible += 1
        position = result['means'][i].double()
        q = torch.cat((position, position.new_ones(1))) @ matrix
        r = 1. / (q[3] + 1e-7)
        grad_xy = result['grads']['means2d'][i, :2].double()
        expected = (matrix[:3, :2] * r -
                    torch.outer(matrix[:3, 3], q[:2]) * r * r) @ grad_xy
        finite = torch.empty(3, dtype=torch.float64)
        step = 2e-5
        for axis in range(3):
            delta = torch.zeros(3, dtype=torch.float64)
            delta[axis] = step
            plus = torch.cat((position + delta, position.new_ones(1))) @ matrix
            minus = torch.cat((position - delta, position.new_ones(1))) @ matrix
            f_plus = torch.dot(plus[:2] / (plus[3] + 1e-7), grad_xy)
            f_minus = torch.dot(minus[:2] / (minus[3] + 1e-7), grad_xy)
            finite[axis] = (f_plus - f_minus) / (2 * step)
        fd_error = (expected - finite).abs().max().item()
        largest_fd = max(largest_fd, fd_error)
        if not torch.allclose(expected, finite, atol=1e-7, rtol=1e-5):
            raise AssertionError(f'{label}: analytic VJP/finite difference disagree at {i}')
        difference = (sample[:3].double() - expected).abs().max().item()
        largest = max(largest, difference)
        if not torch.allclose(sample[:3].double(), expected, atol=3e-4, rtol=2e-3):
            raise AssertionError(f'{label}: projection VJP mismatch at Gaussian {i}')
        magnitude = torch.linalg.vector_norm(sample[:3].double())
        if not math.isclose(sample[3].item(), magnitude.item(),
                            abs_tol=3e-4, rel_tol=2e-3):
            raise AssertionError(f'{label}: sample norm mismatch at Gaussian {i}')
    if visible == 0:
        raise AssertionError(f'{label}: no contributing Gaussian tested')
    print(f'{label}: {visible} contributing Gaussians, VJP max abs error '
          f'{largest:.6g}, analytic/FD {largest_fd:.6g}')


def check_results(baseline, updated):
    for mode in MODES:
        left, right = baseline[mode], updated[mode]
        if not torch.equal(left['radii'], right['radii']):
            raise AssertionError(f'{mode}: forward radii differ')
        compare_tensor(f'{mode}.image', left['image'], right['image'], 3e-5, 3e-5)
        if set(left['grads']) != set(right['grads']):
            raise AssertionError(f'{mode}: active gradient sets differ')
        for name in left['grads']:
            # z/w of means2d were legacy abs-stat channels; signed x/y remain.
            old = left['grads'][name]
            new = right['grads'][name]
            if name == 'means2d':
                old, new = old[:, :2], new[:, :2]
            compare_tensor(f'{mode}.{name}', old, new, 2e-4, 3e-3)
        check_vjp(mode, right)

    off, on = updated['off_color_scale'], updated['color_scale']
    compare_tensor('Conf on/off image', off['image'], on['image'], 3e-5, 3e-5)
    for name in off['grads']:
        compare_tensor(f'Conf on/off {name}', off['grads'][name],
                       on['grads'][name], 2e-4, 3e-3)

    debug = updated['debug_color_scale']
    if not torch.equal(debug['radii'], on['radii']):
        raise AssertionError('Debug mode changed forward radii')
    compare_tensor('Debug on/off image', debug['image'], on['image'], 3e-5, 3e-5)
    for name in on['grads']:
        compare_tensor(f'Debug on/off {name}', debug['grads'][name],
                       on['grads'][name], 2e-4, 3e-3)

    partial = updated['partial']
    if not torch.isfinite(partial['image']).all():
        raise AssertionError('Partial tile image is nonfinite')
    for name, grad in partial['grads'].items():
        if not torch.isfinite(grad).all():
            raise AssertionError(f'Partial tile {name} gradient is nonfinite')
    check_vjp('31x23 partial tile', partial)

    zero = updated['zero']
    seen = zero['samples'][:, 3] >= 0
    if not seen.any() or not torch.equal(zero['samples'][seen],
                                         torch.zeros_like(zero['samples'][seen])):
        raise AssertionError('Visible zero-upstream Gaussians did not consume zero samples')
    if zero['samples'][-1, 3].item() != -1.:
        raise AssertionError('Invisible zero-upstream Gaussian advanced')
    print(f'Zero upstream: {seen.sum().item()} visible zero samples')

    culled = updated['all_culled']
    if not torch.equal(culled['radii'], torch.zeros_like(culled['radii'])):
        raise AssertionError('All-culled control had positive radii')
    if not torch.equal(culled['samples'][:, 3],
                       -torch.ones_like(culled['samples'][:, 3])):
        raise AssertionError('All-culled control emitted visible Conf samples')
    for name, grad in culled['grads'].items():
        if not torch.equal(grad, torch.zeros_like(grad)):
            raise AssertionError(f'All-culled {name} gradient was nonzero')
    print('All-culled zero-bucket backward: PASS')

    occluded = updated['occluded']
    if occluded['radii'][-1].item() <= 0:
        raise AssertionError('Occluded rear Gaussian was frustum-culled')
    if occluded['samples'][-1, 3].item() != -1.:
        raise AssertionError('Radius-positive, fully occluded Gaussian advanced Conf')
    if not (occluded['samples'][:3, 3] >= 0).any():
        raise AssertionError('Occluders did not contribute to the image')
    print('Radius-positive, fully occluded rear Gaussian: PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-wrapper', type=Path, default=BASELINE /
                        'submodules/diff-gaussian-rasterization/diff_gaussian_rasterization')
    parser.add_argument('--baseline-lib', type=Path, default=BASELINE /
                        'build-lib/diff_gaussian_rasterization')
    parser.add_argument('--new-wrapper', type=Path, default=ROOT /
                        'submodules/diff-gaussian-rasterization/diff_gaussian_rasterization')
    parser.add_argument('--new-lib', type=Path, default=NEW_BUILD /
                        'lib/diff_gaussian_rasterization')
    parser.add_argument('--child', choices=('baseline', 'new'), help=argparse.SUPPRESS)
    parser.add_argument('--wrapper', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--binary', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        args.role = args.child
        if args.wrapper is None or args.binary is None or args.output is None:
            parser.error('child mode requires --wrapper, --binary and --output')
        child(args)
        return

    with tempfile.TemporaryDirectory(prefix='conf_cuda_compare_') as directory:
        directory = Path(directory)
        results = {}
        for role, wrapper, binary in (
                ('baseline', args.baseline_wrapper, args.baseline_lib),
                ('new', args.new_wrapper, args.new_lib)):
            output = directory / f'{role}.pt'
            command = [sys.executable, str(Path(__file__).resolve()),
                       '--child', role, '--wrapper', str(wrapper.resolve()),
                       '--binary', str(binary.resolve()), '--output', str(output)]
            subprocess.run(command, check=True)
            results[role] = torch.load(output, map_location='cpu')
        check_results(results['baseline'], results['new'])
    print('Baseline vs compiled CUDA Conf: PASS')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Baseline vs compiled CUDA Conf: FAIL: {error}', file=sys.stderr)
        raise
