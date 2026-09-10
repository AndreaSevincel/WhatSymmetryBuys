#Tests for the SO(2)-equivariant backbone. Run: .venv/bin/python test_equivariant.py

#The claim is architectural, so every test runs on an UNTRAINED network. That is
#the hard case and the only one that shows the property comes from the structure
#of the weights rather than from having been trained into the model -- which is
#exactly the distinction the paper's residual r exists to measure for the
#unconstrained backbone.

#Three failure classes, deliberately separate:
#  * EQUIVARIANCE -- rotating scene and state together about the first axis
#    rotates the output by the same angle, to floating point.
#  * TIGHTNESS -- the model is NOT equivariant to rotations off that axis. A
#    network that ignored its vector inputs entirely would pass the first test
#    and fail this one, so without it "equivariant" could mean "constant".
#  * PERMUTATION -- the obstacle encoder is a set function.

import numpy as np
import torch
import torch.nn as nn

from flowmatch.equivariant import EquivVelocityField

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    PASS += ok
    FAIL += not ok


def roll_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
                        dtype=torch.float32)


def rotate_points(R, p):
    return torch.einsum("ij,...j->...i", R, p)


def rotate_scene(R, spheres, boxes):
    sph = torch.cat([rotate_points(R, spheres[..., :3]), spheres[..., 3:]], dim=-1)
    #centre is a point and the three half-edges are free VECTORS, but about the
    #origin of the reduced frame a rotation acts identically on both -- the
    #distinction that matters elsewhere is the translation, and there is none here
    bx = torch.cat([rotate_points(R, boxes[..., 3 * i:3 * i + 3]) for i in range(4)],
                   dim=-1)
    return sph, bx


def build(seed=0):
    torch.manual_seed(seed)
    net = EquivVelocityField(channels=32, vec_channels=8, n_blocks=2,
                             time_dim=32, env_hidden=32, env_dim=32, env_vec=8,
                             cond_dim=48, cond_vec=8, groups=4).eval()
    #out_s and out_v are zero-initialised so the field starts at the identity.
    #Left that way every test below would compare 0 against 0 and pass
    #vacuously -- the exact trap that made an earlier version of test_sdf.py
    #meaningless.
    nn.init.normal_(net.out_s.weight, std=0.5)
    nn.init.normal_(net.out_s.bias, std=0.5)
    nn.init.normal_(net.out_v.re.weight, std=0.5)
    nn.init.normal_(net.out_v.im.weight, std=0.5)
    return net


def main():
    torch.manual_seed(0)
    B, N, S, K = 2, 16, 20, 20
    net = build()
    x = torch.randn(B, N, 3)
    t = torch.rand(B)
    sg = torch.rand(B, 1)
    spheres = torch.randn(B, S, 4)
    boxes = torch.randn(B, K, 12)

    with torch.no_grad():
        v0 = net(x, t, spheres, boxes, sg)
    check("output is non-trivial", v0.abs().max().item() > 1e-3,
          f"max |v| {v0.abs().max().item():.3f}")

    # --- equivariance about the first axis -----------------------------------
    worst = 0.0
    for theta in (0.3, 1.0, 2.7, -1.9):
        R = roll_x(theta)
        sph_r, box_r = rotate_scene(R, spheres, boxes)
        with torch.no_grad():
            v1 = net(rotate_points(R, x), t, sph_r, box_r, sg)
        err = (rotate_points(R, v0) - v1).abs().max().item()
        worst = max(worst, err)
    check("exactly SO(2)-equivariant about the first axis, untrained",
          worst < 1e-4, f"max |dv| over 4 angles {worst:.2e}")

    # --- tightness: NOT equivariant to a general rotation --------------------
    g = torch.Generator().manual_seed(3)
    a = torch.randn(3, 3, generator=g)
    Q, RR = torch.linalg.qr(a)
    Q = Q * torch.sign(torch.diagonal(RR))[None, :]
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    sph_q, box_q = rotate_scene(Q, spheres, boxes)
    with torch.no_grad():
        vq = net(rotate_points(Q, x), t, sph_q, box_q, sg)
    off = (rotate_points(Q, v0) - vq).abs().max().item()
    check("NOT equivariant off that axis (so the test is not vacuous)",
          off > 1e-2, f"max |dv| {off:.2e}")

    # --- the x component is genuinely invariant ------------------------------
    R = roll_x(0.7)
    sph_r, box_r = rotate_scene(R, spheres, boxes)
    with torch.no_grad():
        vr = net(rotate_points(R, x), t, sph_r, box_r, sg)
    check("the first output component is invariant under the roll",
          (vr[..., 0] - v0[..., 0]).abs().max().item() < 1e-4)
    check("the (y,z) output components are not (they rotate)",
          (vr[..., 1:] - v0[..., 1:]).abs().max().item() > 1e-2)

    # --- obstacle encoder is a set function ----------------------------------
    perm = torch.randperm(S)
    permk = torch.randperm(K)
    with torch.no_grad():
        vp = net(x, t, spheres[:, perm], boxes[:, permk], sg)
    check("permuting obstacles leaves the field unchanged",
          (vp - v0).abs().max().item() < 1e-4)

    # --- equivariance survives training-shaped perturbations -----------------
    # A network is easy to make equivariant by accident when its weights are
    # small and symmetric. Scale every parameter and re-check.
    for p in net.parameters():
        p.data = p.data * 3.0 + 0.1
    with torch.no_grad():
        v0b = net(x, t, spheres, boxes, sg)
        R = roll_x(1.3)
        sph_r, box_r = rotate_scene(R, spheres, boxes)
        v1b = net(rotate_points(R, x), t, sph_r, box_r, sg)
    #RELATIVE error here: tripling every weight drives the output to ~1e4, where
    #an absolute 1e-4 threshold is below float32 resolution and would fail a
    #perfectly equivariant network. The unperturbed tests above can afford an
    #absolute bound because their outputs are O(1).
    err = (rotate_points(R, v0b) - v1b).abs().max().item()
    rel = err / max(v0b.abs().max().item(), 1e-12)
    check("still equivariant after perturbing every weight", rel < 1e-5,
          f"relative {rel:.2e} (absolute {err:.2e} at |v|~{v0b.abs().max():.1e})")

    # --- the oracle geometry channels keep the constraint --------------------
    # local_geom feeds the SDF value and gradient into the trunk. That is only
    # legal because the split is by irrep: d and g_x are invariant, g_yz is m=1.
    # Get the routing wrong -- g concatenated to the state the way the
    # unconstrained backbone does it -- and the model still runs, still trains,
    # and is quietly no longer equivariant. So check, with REAL obstacle
    # geometry: the SDF needs positive radii and orthogonal box edges.
    torch.manual_seed(1)
    netg = EquivVelocityField(channels=32, vec_channels=8, n_blocks=2,
                              time_dim=32, env_hidden=32, env_dim=32, env_vec=8,
                              cond_dim=48, cond_vec=8, groups=4,
                              local_geom=True).eval()
    nn.init.normal_(netg.out_s.weight, std=0.5)
    nn.init.normal_(netg.out_s.bias, std=0.5)
    nn.init.normal_(netg.out_v.re.weight, std=0.5)
    nn.init.normal_(netg.out_v.im.weight, std=0.5)

    sph_g = torch.cat([torch.randn(B, S, 3), torch.rand(B, S, 1) * 0.2 + 0.05], -1)
    edges = []
    for _ in range(B * K):
        q, rr = torch.linalg.qr(torch.randn(3, 3))
        q = q * torch.sign(torch.diagonal(rr))[None, :]
        edges.append(q * (torch.rand(3) * 0.2 + 0.05))
    E = torch.stack(edges).reshape(B, K, 3, 3)
    box_g = torch.cat([torch.randn(B, K, 3), E.reshape(B, K, 9)], dim=-1)

    with torch.no_grad():
        vg0 = netg(x, t, sph_g, box_g, sg)
    R = roll_x(0.9)
    sph_r, box_r = rotate_scene(R, sph_g, box_g)
    with torch.no_grad():
        vg1 = netg(rotate_points(R, x), t, sph_r, box_r, sg)
    errg = (rotate_points(R, vg0) - vg1).abs().max().item()
    check("local_geom SDF channels preserve exact equivariance",
          errg < 1e-4 and vg0.abs().max().item() > 1e-3,
          f"max |dv| {errg:.2e} at |v|~{vg0.abs().max().item():.2f}")

    # --- the hand-written norms survive half precision -----------------------
    # The training runs use --amp. torch promotes GroupNorm to fp32 by its own
    # cast policy, so the scalar stream is safe for free and a hand-written norm
    # is not. The binding failure is OVERFLOW, not underflow: fp16 tops out at
    # 65504, so squaring any activation past ~256 gives inf, the mean is inf,
    # and every channel at that waypoint is divided to zero. The weight
    # perturbation test above already reaches |v|~1e2, so this is in range.
    # (At the small end the eps floor suppresses the output in fp32 too -- that
    # is the intended "do not amplify noise" behaviour, not a precision bug.)
    from flowmatch.equivariant import vec_rms_norm
    big = (torch.randn(2, 8, 2, 16) * 400).half()
    w = torch.ones(8).half()
    out = vec_rms_norm(big, w)
    rms = (out.float() ** 2).sum(2).mean(1).sqrt()
    check("vec_rms_norm survives fp16 activations whose squares overflow",
          torch.isfinite(out).all().item() and (rms - 1).abs().max().item() < 0.05,
          f"output RMS {rms.mean().item():.4f} (want 1.0)")

    n_par = sum(p.numel() for p in net.parameters())
    print(f"\n(test net: {n_par/1e3:.0f}k parameters)")
    print(f"{PASS}/{PASS + FAIL} passed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
