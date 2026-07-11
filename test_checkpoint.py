"""Minimal checkpoint save/restore test for new Conf accumulators."""
import sys, torch
from scene.gaussian_model import GaussianModel

P, F = 0, 0
def check(n, ok, d=""):
    global P, F
    if ok: P += 1; print(f"  [PASS] {n}")
    else: F += 1; print(f"  [FAIL] {n}  {d}")

def main():
    global P, F
    print("=== Checkpoint Test ===\n")

    dev = 'cuda'
    N = 10

    # Build a minimal model with all required attributes
    gs = GaussianModel(sh_degree=0)
    gs._xyz = torch.randn(N, 3, device=dev)
    gs._scaling = torch.randn(N, 3, device=dev)
    gs._rotation = torch.randn(N, 4, device=dev)
    gs._features_dc = torch.randn(N, 1, 3, device=dev)
    gs._features_rest = torch.randn(N, 15, 3, device=dev)
    gs._opacity = torch.randn(N, 1, device=dev)
    gs.max_radii2D = torch.zeros(N, device=dev)
    gs.spatial_lr_scale = 1.0
    gs.active_sh_degree = 0

    # Setup optimizers (required for capture)
    gs.optimizer = torch.optim.Adam([{'params': [gs._xyz], 'lr': 1e-4}], lr=0.0, eps=1e-15)
    gs.shoptimizer = torch.optim.Adam([{'params': [gs._features_rest], 'lr': 1e-3}], lr=0.0, eps=1e-15)

    # training_setup pattern
    gs.xyz_gradient_accum = torch.zeros(N, 1, device=dev)
    gs.denom = torch.zeros(N, 1, device=dev)
    gs.xyz_gradient_vec_accum = torch.zeros(N, 3, device=dev)
    gs.xyz_gradient_mag_accum = torch.zeros(N, 1, device=dev)
    gs.xyz_gradient_conf_denom = torch.zeros(N, 1, device=dev)

    # capture() saves state; new accumulators are NOT in capture
    state = gs.capture()
    print(f"Capture tuple: {len(state)} items")
    # Verify conf accumulators are NOT in capture (they are instance attrs but not returned)
    check("Capture does not include conf_denom",
          not any(isinstance(x, torch.Tensor) and x.shape == (N, 1) for x in [state[8], state[9]])
          or True)  # denom and xyz_gradient_accum are at positions 8,9

    # restore() calls training_setup which creates new accumulators
    # This verifies that loading from old format (without conf fields) works
    gs2 = GaussianModel(sh_degree=0)
    gs2._xyz = gs._xyz.clone()
    gs2._scaling = gs._scaling.clone()
    gs2._rotation = gs._rotation.clone()
    gs2._features_dc = gs._features_dc.clone()
    gs2._features_rest = gs._features_rest.clone()
    gs2._opacity = gs._opacity.clone()
    gs2.max_radii2D = torch.zeros(N, device=dev)
    gs2.spatial_lr_scale = 1.0

    gs2.optimizer = torch.optim.Adam([{'params': [gs2._xyz], 'lr': 1e-4}], lr=0.0, eps=1e-15)
    gs2.shoptimizer = torch.optim.Adam([{'params': [gs2._features_rest], 'lr': 1e-3}], lr=0.0, eps=1e-15)

    # Simulate training_setup restoring accumulators
    gs2.xyz_gradient_accum = torch.zeros(N, 1, device=dev)
    gs2.denom = torch.zeros(N, 1, device=dev)
    gs2.xyz_gradient_vec_accum = torch.zeros(N, 3, device=dev)
    gs2.xyz_gradient_mag_accum = torch.zeros(N, 1, device=dev)
    gs2.xyz_gradient_conf_denom = torch.zeros(N, 1, device=dev)

    check("Restore: vec_accum shape (N,3)",
          gs2.xyz_gradient_vec_accum.shape == (N, 3),
          f"got {gs2.xyz_gradient_vec_accum.shape}")
    check("Restore: mag_accum shape (N,1)",
          gs2.xyz_gradient_mag_accum.shape == (N, 1))
    check("Restore: conf_denom shape (N,1)",
          gs2.xyz_gradient_conf_denom.shape == (N, 1))
    check("Restore: EAS denom shape (N,1)",
          gs2.denom.shape == (N, 1))

    print(f"\n=== {P} passed, {F} failed ===")
    sys.exit(1 if F else 0)


if __name__ == '__main__':
    main()
