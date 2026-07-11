"""
Implementation-level tests for _compute_ndc_vjp_world() and add_densification_stats_abs().

Tests the actual GaussianModel helper and stats methods with synthetic tensors:
  - Valid front-camera Gaussians accumulate correctly
  - Behind-camera (qw<=0) Gaussians are excluded
  - NaN/Inf gradients are excluded from Conf accumulators
  - Non-visible indices stay zero
  - EAS scalar denom unchanged (always increments for visible idx)
  - Separates EAS denom from Conf denom
Exit non-zero on failure.
"""

import sys, math, torch
from scene.gaussian_model import GaussianModel
from scene.cameras import getProjectionMatrix

P, F = 0, 0
def check(name, ok, detail=""):
    global P, F
    if ok: P += 1; print(f"  [PASS] {name}")
    else: F += 1; print(f"  [FAIL] {name}  {detail}")

def main():
    global P, F
    print("=== Implementation-Level Conf Validation ===\n")

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = torch.float32

    # Build synthetic full_proj_transform
    proj = getProjectionMatrix(0.01, 100.0, math.radians(60), math.radians(45)).to(dt).to(dev).T.contiguous()
    w2c = torch.eye(4, dtype=dt, device=dev)
    w2c[3, :3] = torch.tensor([0.0, 0.0, 0.0], device=dev)
    fpt = (w2c.T.contiguous().unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    # Create minimal GaussianModel
    gs = GaussianModel(sh_degree=0)
    N = 8  # rows: 0-2 valid front, 3 behind-cam, 4 NaN grad, 5 Inf grad, 6 non-visible, 7 valid
    gs._xyz = torch.zeros(N, 3, device=dev)
    gs._xyz[0] = torch.tensor([0.0, 0.0, 2.0], device=dev)   # front valid
    gs._xyz[1] = torch.tensor([0.5, 0.0, 3.0], device=dev)   # front valid
    gs._xyz[2] = torch.tensor([-0.5, 0.0, 1.5], device=dev) # front valid
    gs._xyz[3] = torch.tensor([0.0, 0.0, -1.0], device=dev)  # behind (z<0 -> qw<=0)
    gs._xyz[4] = torch.tensor([0.0, 0.5, 2.0], device=dev)   # NaN grad
    gs._xyz[5] = torch.tensor([0.0, -0.5, 2.0], device=dev)  # Inf grad
    gs._xyz[6] = torch.tensor([10.0, 10.0, 5.0], device=dev) # non-visible (not in idx)
    gs._xyz[7] = torch.tensor([0.3, 0.3, 2.5], device=dev)   # front valid

    gs.xyz_gradient_vec_accum = torch.zeros(N, 3, device=dev)
    gs.xyz_gradient_mag_accum = torch.zeros(N, 1, device=dev)
    gs.xyz_gradient_conf_denom = torch.zeros(N, 1, device=dev)
    gs.xyz_gradient_accum = torch.zeros(N, 1, device=dev)
    gs.denom = torch.zeros(N, 1, device=dev)

    # visible indices: 0,1,2,3,4,5,7 (6 is not in the set)
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 7], device=dev)

    # Build synthetic viewspace_point_tensor.grad
    # Channels: 0=signed x, 1=signed y, 2=abs x, 3=abs y
    grad_tensor = torch.zeros(N, 4, device=dev)
    for i in [0, 1, 2, 7]:
        grad_tensor[i, 0] = 0.1 * (i + 1)  # signed x
        grad_tensor[i, 1] = 0.05 * (i + 1)  # signed y
        grad_tensor[i, 2] = abs(grad_tensor[i, 0])  # abs x
        grad_tensor[i, 3] = abs(grad_tensor[i, 1])  # abs y
    # Row 3 (behind-cam): has grad but should be filtered
    grad_tensor[3, 0] = 0.2; grad_tensor[3, 1] = 0.1
    grad_tensor[3, 2] = 0.2; grad_tensor[3, 3] = 0.1
    # Row 4: NaN
    grad_tensor[4, 0] = float('nan'); grad_tensor[4, 1] = 0.0
    grad_tensor[4, 2] = float('nan'); grad_tensor[4, 3] = 0.0
    # Row 5: Inf
    grad_tensor[5, 0] = float('inf'); grad_tensor[5, 1] = 0.0
    grad_tensor[5, 2] = float('inf'); grad_tensor[5, 3] = 0.0

    # Mock viewspace_point_tensor
    class MockTensor:
        def __init__(self, grad):
            self.grad = grad
    mock_vpt = MockTensor(grad_tensor)

    # Mock camera
    class MockCam:
        pass
    cam = MockCam()
    cam.full_proj_transform = fpt

    # Call add_densification_stats_abs
    gs.add_densification_stats_abs(mock_vpt, idx, cam)

    # ===== Checks =====
    print("Test 1: Conf accumulation correctness")
    # Valid rows (0,1,2,7) should have non-zero Conf accum
    for i in [0, 1, 2, 7]:
        ok = gs.xyz_gradient_vec_accum[i].abs().sum() > 0
        check(f"Row {i} (valid) has Conf vec accumulation", ok)
        ok = gs.xyz_gradient_mag_accum[i].item() > 0
        check(f"Row {i} (valid) has Conf mag accumulation", ok)
        ok = gs.xyz_gradient_conf_denom[i].item() == 1.0
        check(f"Row {i} (valid) has Conf denom=1", ok)

    # Behind-camera row (3) must be zero in Conf, but EAS scalar denom must still increment
    check("Row 3 (behind) Conf vec zero",
          gs.xyz_gradient_vec_accum[3].abs().sum() == 0,
          f"got {gs.xyz_gradient_vec_accum[3].abs().sum().item():.6f}")
    check("Row 3 (behind) Conf mag zero",
          gs.xyz_gradient_mag_accum[3].item() == 0)
    check("Row 3 (behind) Conf denom zero",
          gs.xyz_gradient_conf_denom[3].item() == 0)
    check("Row 3 (behind) EAS denom increments",
          gs.denom[3].item() == 1.0,
          f"got {gs.denom[3].item()}")

    # NaN row (4)
    check("Row 4 (NaN) Conf vec zero",
          gs.xyz_gradient_vec_accum[4].abs().sum() == 0)
    check("Row 4 (NaN) Conf denom zero",
          gs.xyz_gradient_conf_denom[4].item() == 0)

    # Inf row (5)
    check("Row 5 (Inf) Conf vec zero",
          gs.xyz_gradient_vec_accum[5].abs().sum() == 0)
    check("Row 5 (Inf) Conf denom zero",
          gs.xyz_gradient_conf_denom[5].item() == 0)

    # Non-visible row (6) must be all zero
    check("Row 6 (non-visible) Conf vec zero",
          gs.xyz_gradient_vec_accum[6].abs().sum() == 0)
    check("Row 6 (non-visible) Conf mag zero",
          gs.xyz_gradient_mag_accum[6].item() == 0)
    check("Row 6 (non-visible) Conf denom zero",
          gs.xyz_gradient_conf_denom[6].item() == 0)
    check("Row 6 (non-visible) EAS denom zero",
          gs.denom[6].item() == 0)

    # ===== Direct helper test =====
    print("\nTest 2: _compute_ndc_vjp_world() direct")
    g_world, valid = gs._compute_ndc_vjp_world(
        gs._xyz, grad_tensor[idx, :2], idx, fpt,
    )
    check("Full valid mask has correct size", valid.shape[0] == N)
    check("Row 0 valid=True", valid[0].item())
    check("Row 3 (behind) valid=False", not valid[3].item())
    check("Row 4 (NaN) valid=False", not valid[4].item())
    check("Row 5 (Inf) valid=False", not valid[5].item())
    check("Row 6 (non-visible) valid=False", not valid[6].item())
    check("Row 3 g_world zero", g_world[3].abs().sum() == 0)
    check("Row 4 g_world zero", g_world[4].abs().sum() == 0)
    check("Row 6 g_world zero (not in filter)", g_world[6].abs().sum() == 0)

    # ===== Two-call accumulation test =====
    print("\nTest 3: Two-call accumulation")
    gs2 = GaussianModel(sh_degree=0)
    gs2._xyz = gs._xyz.clone()
    gs2.xyz_gradient_vec_accum = torch.zeros(N, 3, device=dev)
    gs2.xyz_gradient_mag_accum = torch.zeros(N, 1, device=dev)
    gs2.xyz_gradient_conf_denom = torch.zeros(N, 1, device=dev)
    gs2.xyz_gradient_accum = torch.zeros(N, 1, device=dev)
    gs2.denom = torch.zeros(N, 1, device=dev)

    # Call twice to verify denom counts correctly
    gs2.add_densification_stats_abs(mock_vpt, idx, cam)
    gs2.add_densification_stats_abs(mock_vpt, idx, cam)

    check("Row 0 Conf denom=2 after two calls",
          gs2.xyz_gradient_conf_denom[0].item() == 2.0)
    check("Row 3 (behind) Conf denom=0 after two calls",
          gs2.xyz_gradient_conf_denom[3].item() == 0)
    check("Row 0 EAS denom=2 after two calls",
          gs2.denom[0].item() == 2.0,
          f"got {gs2.denom[0].item()}")
    check("Row 3 EAS denom=2 after two calls (EAS unchanged)",
          gs2.denom[3].item() == 2.0,
          f"got {gs2.denom[3].item()}")

    print(f"\n=== {P} passed, {F} failed ===")
    if F > 0:
        print("EXIT: FAILURE")
        sys.exit(1)
    print("EXIT: SUCCESS")


if __name__ == '__main__':
    main()
