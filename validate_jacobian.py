"""
Strict Jacobian validation for NDC-to-world Conf pullback.

CUDA camera-space convention: z > 0.2 = front (rendered).
The projection matrix yields qw = camera_z, so qw > 0 = front-camera.

Plan thresholds (float64):
  - FD max abs error <= 1e-4
  - Autograd VJP relative error <= 1e-5
  - Roll cos_sim > 0.999
  - Negative: behind-camera excluded, NaN/Inf handled
  - Exit non-zero on failure.
"""

import math, sys, torch
from scene.cameras import getProjectionMatrix

P,F = 0,0
def rec(n,ok,det=""):
    global P,F
    if ok: P+=1; print(f"  [PASS] {n}")
    else: F+=1; print(f"  [FAIL] {n}  {det}")

def a_jac(xyz, fpt):
    h = torch.cat([xyz, torch.ones_like(xyz[:,:1])], -1)
    c = h @ fpt
    qx,qy,qw = c[:,0],c[:,1],c[:,3]
    sq = qw.clone(); sq[sq<=0]=1.0
    Mx=fpt[:3,0]; My=fpt[:3,1]; Mw=fpt[:3,3]
    jx=(Mx[None,:]*sq[:,None]-Mw[None,:]*qx[:,None])/sq[:,None].square()
    jy=(My[None,:]*sq[:,None]-Mw[None,:]*qy[:,None])/sq[:,None].square()
    return torch.stack([jx,jy],1)

def fd_jac(xyz, fpt, d=1e-5):
    N=xyz.shape[0]; J=torch.zeros(N,2,3,device=xyz.device,dtype=xyz.dtype)
    for k in range(3):
        e=torch.zeros(3,device=xyz.device,dtype=xyz.dtype); e[k]=d
        def p(x): h=torch.cat([x,torch.ones_like(x[:,:1])],-1);c=h@fpt;return c[:,:2]/c[:,3:]
        J[:,:,k]=(p(xyz+e[None,:])-p(xyz-e[None,:]))/(2*d)
    return J

def ag_vjp(xyz, g, fpt):
    x=xyz.clone().detach().requires_grad_(True)
    h=torch.cat([x,torch.ones_like(x[:,:1])],-1)
    c=h@fpt; qw=c[:,3:].clone(); qw[qw<=0]=1.0
    return torch.autograd.grad(((c[:,:2]/qw)*g).sum(),x)[0]

def main():
    global P,F
    print("=== Strict Jacobian Validation ===\n")
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt=torch.float64

    # Synthetic: identity view (world=camera), looking along +z
    wvt=torch.eye(4,dtype=dt,device=dev).T.contiguous()
    proj=getProjectionMatrix(0.01,100.0,math.radians(60),math.radians(45)).to(dt).to(dev).T.contiguous()
    fpt0=(wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    N,C=100,5
    torch.manual_seed(42)
    # Front-camera points (z > 0.2 in CUDA convention, qw > 0)
    xyz=torch.randn(N,3,device=dev,dtype=dt)
    xyz[:,2]=torch.abs(xyz[:,2])+1.0  # z in [1,~5]
    g_ndc=torch.randn(N,2,device=dev,dtype=dt)*0.1

    fpts=[fpt0]
    for i in range(1,C):
        a=2*math.pi*i/C
        w=torch.eye(4,dtype=dt,device=dev)
        w[3,0]=-2.0*math.cos(a); w[3,1]=-2.0*math.sin(a)
        fpts.append((w.T.contiguous().unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0))

    # ===== FD =====
    print("Test 1: FD (100pts x 5 cams, float64)")
    mx=0.0
    for fp in fpts: mx=max(mx,(a_jac(xyz,fp)-fd_jac(xyz,fp,1e-5)).abs().max().item())
    rec("FD max abs <= 1e-4",mx<=1e-4,f"{mx:.2e}")

    # ===== VJP =====
    print("\nTest 2: Autograd VJP")
    mr=0.0
    for fp in fpts:
        J=a_jac(xyz,fp); ga=torch.bmm(J.transpose(1,2),g_ndc.unsqueeze(-1)).squeeze(-1)
        gg=ag_vjp(xyz,g_ndc,fp)
        mr=max(mr,((ga-gg).abs()/(gg.abs()+1e-12)).mean().item())
    rec("VJP rel <= 1e-5",mr<=1e-5,f"{mr:.2e}")

    # ===== Roll =====
    print("\nTest 3: Roll invariance")
    th=math.radians(30); c,s=math.cos(th),math.sin(th)
    Rr=torch.tensor([[c,-s,0],[s,c,0],[0,0,1]],dtype=dt,device=dev)
    eye=torch.tensor([0.,0.,-3.],dtype=dt,device=dev)
    tgt=torch.tensor([0.,0.,2.],dtype=dt,device=dev)
    up0=torch.tensor([0.,1.,0.],dtype=dt,device=dev)
    def cam(up):
        z=tgt-eye; z=z/z.norm()
        x=torch.linalg.cross(up,z); x=x/(x.norm()+1e-8)
        y=torch.linalg.cross(z,x)
        R=torch.stack([x,y,z],0)
        w2c=torch.eye(4,dtype=dt,device=dev)
        w2c[:3,:3]=R.T; w2c[3,:3]=-R@eye
        return (w2c.T.contiguous().unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    fp1,fp2=cam(up0),cam(Rr@up0)
    pt=torch.tensor([[0.,0.,2.]],dtype=dt,device=dev)
    g0=torch.tensor([[0.1,0.05]],dtype=dt,device=dev)
    g1=(Rr[:2,:2]@g0.T).T
    J1,J2=a_jac(pt,fp1),a_jac(pt,fp2)
    gw1=(J1.transpose(1,2)@g0.unsqueeze(-1)).squeeze(-1)
    gw2=(J2.transpose(1,2)@g1.unsqueeze(-1)).squeeze(-1)
    cs=torch.nn.functional.cosine_similarity(gw1,gw2,dim=-1).item()
    rec("Roll cos_sim > 0.999",cs>0.999,f"{cs:.6f}")

    # ===== Negative =====
    print("\nTest 4: Negative")
    xb=torch.tensor([[0.,0.,-1.]],dtype=dt,device=dev)  # behind (z<0)
    qw=(torch.cat([xb,torch.ones_like(xb[:,:1])],-1)@fpt0)[:,3]
    rec("Behind-cam qw <= 0",qw.item()<=0,f"qw={qw.item():.4f}")
    xn=torch.tensor([[0.,0.,2.]],dtype=dt,device=dev)
    Jn=a_jac(xn,fpt0)
    gn=torch.tensor([[float('nan'),0.]],dtype=dt,device=dev)
    rec("NaN gradient -> NaN output",torch.bmm(Jn.transpose(1,2),gn.unsqueeze(-1)).isnan().any().item())
    gi=torch.tensor([[float('inf'),0.]],dtype=dt,device=dev)
    rec("Inf gradient -> Inf output",torch.bmm(Jn.transpose(1,2),gi.unsqueeze(-1)).isinf().any().item())

    print(f"\n=== {P} passed, {F} failed ===")
    sys.exit(1 if F else 0)

if __name__=='__main__': main()
