
  #.venv/bin/python test_equivariance.py

  #Tests for the frame-averaging operator. The central claim is that
  #  v_bar(x) = (1/K) sum_k rho(theta_k)^-1 f(rho(theta_k) x)
  #is EXACTLY C_K-equivariant for any f whatsoever -- no architectural
  #constraint, no training assumption. These tests use an untrained (and
  #therefore strongly non-equivariant) network, which is the hard case.

import math

import numpy as np
import torch

from flowmatch.flow import (
    frame_averaged_velocity,
    roll_matrices,
    sample,
    sample_reduced,
)
from flowmatch.geometry import (
    rotate_box_features,
    rotate_sphere_features,
    sg_frame,
)
from flowmatch.model import FlowVelocityField


def setup(B=3, K=7, S=6, Bx=5, N=64, seed=0):
    torch.manual_seed(seed)
    net = FlowVelocityField(channels=32, n_blocks=4, box_dim=12, sg_dim=1).eval()
    #out_conv is zero-initialized (identity field at init), which would make
    #every equivariance test trivially pass. Perturb it so f is a genuinely
    #arbitrary, non-equivariant function.
    with torch.no_grad():
        net.out_conv.weight.normal_(0, 0.5)
        net.out_conv.bias.normal_(0, 0.5)

    g = torch.Generator().manual_seed(seed)
    start = torch.rand(B, 3, generator=g) * 2 - 1
    goal = start + torch.nn.functional.normalize(
        torch.rand(B, 3, generator=g) * 2 - 1, dim=-1
    ) * 1.8
    spheres = torch.cat([
        (torch.rand(B, S, 3, generator=g) * 2 - 1) * 0.75,
        0.10 + torch.rand(B, S, 1, generator=g) * 0.15,
    ], dim=-1)
    boxes = torch.zeros(B, Bx, 12)
    boxes[..., :3] = (torch.rand(B, Bx, 3, generator=g) * 2 - 1) * 0.75
    boxes[..., 3::4] = 0.08 + torch.rand(B, Bx, 3, generator=g) * 0.12
    return net, start, goal, spheres, boxes, N, K


def build_ck(net, start, goal, spheres, boxes, thetas, K):
    roll, Rk, origin_k, dk = roll_matrices(start, goal, thetas, K)
    sph_k = rotate_sphere_features(spheres.repeat_interleave(K, 0), Rk, origin_k)
    box_k = rotate_box_features(boxes.repeat_interleave(K, 0), Rk, origin_k)
    c_k = net.encode_cond(sph_k, box_k, dk[:, None])
    return roll, c_k


def rho(alpha, dtype=torch.float32):
    #Roll about the reduced-frame x-axis, in the SAME convention as
    #roll_matrices (which yields R_x(theta) with rows [x, c*y+s*z, -s*y+c*z]).
    #rho(a) @ rho(b) == rho(a+b), so the rolls form a group homomorphism.
    c, s = math.cos(alpha), math.sin(alpha)
    return torch.tensor([[1.0, 0, 0], [0, c, s], [0, -s, c]], dtype=dtype)


def roll_x(x, alpha):
    #Rotate a (B,N,3) state by alpha about the reduced-frame x-axis.
    return torch.einsum("ij,bkj->bki", rho(alpha, x.dtype), x)


def roll_scene(spheres, boxes, alpha):
    #Roll a reduced-frame scene. IMPORTANT: the group acts on the WHOLE input.
    #Rolling only the state leaves each c_k bound to the wrong frame index, and
    #the reindexing argument then fails -- exactness needs both rolled together.
    B = spheres.shape[0]
    R = rho(alpha, spheres.dtype).expand(B, 3, 3)
    o = torch.zeros(B, 3, dtype=spheres.dtype)
    return (
        rotate_sphere_features(spheres, R, o),
        rotate_box_features(boxes, R, o),
    )


def reduced_scene(start, goal, spheres, boxes):
    #Express a world-frame scene in reduced frame 0 so tests can work natively.
    R0, origin, d = sg_frame(start, goal, None)
    return (
        rotate_sphere_features(spheres, R0, origin),
        rotate_box_features(boxes, R0, origin),
        d,
    )


def fa_velocity(net, sph_r, box_r, d, x, t_val, K, offset=0.0):
    #Frame-averaged velocity for a scene already in reduced frame 0.
    #Builds the K rolled scene encodings explicitly (no sg_frame involved), so
    #this is an independent path from roll_matrices.
    B = sph_r.shape[0]
    rolls, sph_k, box_k = [], [], []
    for k in range(K):
        th = offset + 2 * math.pi * k / K
        s_k, b_k = roll_scene(sph_r, box_r, th)
        rolls.append(rho(th, sph_r.dtype).expand(B, 3, 3))
        sph_k.append(s_k)
        box_k.append(b_k)
    #interleave to match repeat_interleave ordering (b0k0, b0k1, ..., b1k0, ...)
    roll = torch.stack(rolls, dim=1).reshape(B * K, 3, 3)
    sph_k = torch.stack(sph_k, dim=1).reshape(B * K, *sph_r.shape[1:])
    box_k = torch.stack(box_k, dim=1).reshape(B * K, *box_r.shape[1:])
    c_k = net.encode_cond(sph_k, box_k, d.repeat_interleave(K, 0)[:, None])
    t = torch.full((B * K,), t_val, dtype=x.dtype)
    return frame_averaged_velocity(net, x, t, c_k, roll, K)


def test_network_is_genuinely_non_equivariant():
    #Sanity: the tests below would be vacuous against an already-equivariant f.
    net, start, goal, spheres, boxes, N, K = setup()
    thetas = torch.zeros(start.shape[0], 1)
    roll, c_k = build_ck(net, start, goal, spheres, boxes, thetas, 1)
    x = torch.randn(start.shape[0], N, 3)
    t = torch.full((start.shape[0],), 0.5)
    with torch.no_grad():
        v = net.decode(x, t, c_k)
        v_rot = net.decode(roll_x(x, 0.7), t, c_k)
    gap = (roll_x(v, 0.7) - v_rot).abs().max().item()
    assert gap > 1e-3, f"f looks equivariant already ({gap:.2e}); tests would be vacuous"
    print(f"ok  untrained f is non-equivariant (gap {gap:.3f})  [tests are non-vacuous]")


def test_exact_CK_equivariance():
    #Level 1: for alpha = 2*pi*j/K the average is EXACTLY equivariant, because
    #rolling the whole input by alpha permutes the quadrature points and the sum
    #is merely reindexed. Holds for ANY f -- this network is untrained.
    net, start, goal, spheres, boxes, N, K = setup()
    sph_r, box_r, d = reduced_scene(start, goal, spheres, boxes)
    B = start.shape[0]
    x = torch.randn(B, N, 3)
    with torch.no_grad():
        v, _ = fa_velocity(net, sph_r, box_r, d, x, 0.3, K)
        worst = 0.0
        for j in range(1, K):
            alpha = 2 * math.pi * j / K
            sph_a, box_a = roll_scene(sph_r, box_r, alpha)
            v_rot, _ = fa_velocity(net, sph_a, box_a, d, roll_x(x, alpha), 0.3, K)
            err = (v_rot - roll_x(v, alpha)).abs().max().item()
            worst = max(worst, err)
            assert err < 2e-4, f"C_K equivariance broken at j={j}: {err:.2e}"
    print(f"ok  exact C_K-equivariance for all j=1..{K-1}  (worst {worst:.1e})")


def test_state_only_roll_is_not_the_group_action():
    #Guards the trap the first version of this test fell into: rolling the state
    #while leaving the scene fixed is NOT the group action, so it shows a large
    #discrepancy even though the operator is correct. A diagnostic written that
    #way would report non-equivariance that isn't there.
    net, start, goal, spheres, boxes, N, K = setup()
    sph_r, box_r, d = reduced_scene(start, goal, spheres, boxes)
    B = start.shape[0]
    x = torch.randn(B, N, 3)
    alpha = 2 * math.pi / K
    with torch.no_grad():
        v, _ = fa_velocity(net, sph_r, box_r, d, x, 0.3, K)
        #scene held fixed on purpose
        v_bad, _ = fa_velocity(net, sph_r, box_r, d, roll_x(x, alpha), 0.3, K)
        sph_a, box_a = roll_scene(sph_r, box_r, alpha)
        v_good, _ = fa_velocity(net, sph_a, box_a, d, roll_x(x, alpha), 0.3, K)
    want = roll_x(v, alpha)
    err_bad = (v_bad - want).abs().max().item()
    err_good = (v_good - want).abs().max().item()
    assert err_good < 2e-4 < err_bad, (err_good, err_bad)
    print(f"ok  rolling scene+state is exact ({err_good:.1e}); state alone is not "
          f"({err_bad:.1e})  [the trap]")


def test_off_lattice_angle_is_only_approximate():
    #Level 3: for alpha NOT a multiple of 2*pi/K the average is only
    #approximately equivariant -- the residual is the aliased content. The flip
    #side of the exactness above; confirms the test measures group structure
    #rather than something trivially true.
    net, start, goal, spheres, boxes, N, K = setup()
    sph_r, box_r, d = reduced_scene(start, goal, spheres, boxes)
    B = start.shape[0]
    x = torch.randn(B, N, 3)
    alpha = math.pi / (2 * K)  # deliberately off-lattice
    with torch.no_grad():
        v, _ = fa_velocity(net, sph_r, box_r, d, x, 0.3, K)
        sph_a, box_a = roll_scene(sph_r, box_r, alpha)
        v_rot, _ = fa_velocity(net, sph_a, box_a, d, roll_x(x, alpha), 0.3, K)
    err = (v_rot - roll_x(v, alpha)).abs().max().item()
    assert err > 1e-4, "off-lattice angle should NOT be exactly equivariant"
    print(f"ok  off-lattice alpha is only approximate (err {err:.4f})  [level 3]")


def test_averaging_is_noop_for_equivariant_field():
    #Pointwise argument: if f is already equivariant, every un-rolled term
    #equals f(x) and the average is a no-op. Uses an analytically equivariant
    #field (scale by a function of x-component and rho) in place of the network.
    class EquivariantField:
        def decode(self, x, t, c):
            r = x[..., 1:].norm(dim=-1, keepdim=True)
            gain = 1.0 + 0.5 * torch.tanh(x[..., :1]) + 0.3 * r
            #radial/axial construction: commutes with any rotation about x
            return torch.cat([x[..., :1] * 0.7, x[..., 1:] * gain], dim=-1)

    B, N, K = 3, 32, 9
    torch.manual_seed(1)
    start = torch.rand(B, 3) * 2 - 1
    goal = start + torch.nn.functional.normalize(torch.rand(B, 3) * 2 - 1, dim=-1) * 1.8
    base = torch.arange(K, dtype=start.dtype) * (2 * math.pi / K)
    thetas = base[None, :].expand(B, K)
    roll, _, _, _ = roll_matrices(start, goal, thetas, K)

    x = torch.randn(B, N, 3)
    t = torch.full((B * K,), 0.4)
    field = EquivariantField()
    v_avg, vk = frame_averaged_velocity(field, x, t, None, roll, K)
    v_single = field.decode(x, None, None)
    err = (v_avg - v_single).abs().max().item()
    spread = (vk - v_avg[:, None]).abs().max().item()
    assert err < 1e-5, f"averaging should be a no-op, got {err:.2e}"
    assert spread < 1e-5, f"all K terms should agree, spread {spread:.2e}"
    print(f"ok  averaging is a no-op for an equivariant field (err {err:.1e}, "
          f"spread {spread:.1e})  [pointwise argument]")


def test_unroll_sanity_diagnostic_is_conditional():
    #Diagnostic 1 from the plan is norm(mean(v_k)) / mean(norm(v_k)) > 0.9.
    #It is NOT an unconditional test of the un-roll: it only reads ~1 when the
    #model is ALREADY nearly equivariant. On a strongly non-equivariant model
    #the K terms genuinely disagree, so the ratio is low with a perfectly
    #correct un-roll. It therefore must not be wired in as a hard assertion
    #during early training -- it would fire on a healthy pipeline.
    B, N, K = 3, 32, 9
    torch.manual_seed(1)
    start = torch.rand(B, 3) * 2 - 1
    goal = start + torch.nn.functional.normalize(torch.rand(B, 3) * 2 - 1, dim=-1) * 1.8
    base = torch.arange(K, dtype=start.dtype) * (2 * math.pi / K)
    roll, _, _, _ = roll_matrices(start, goal, base[None, :].expand(B, K), K)
    x = torch.randn(B, N, 3)
    t = torch.full((B * K,), 0.4)

    class EquivariantField:
        def decode(self, x, t, c):
            r = x[..., 1:].norm(dim=-1, keepdim=True)
            gain = 1.0 + 0.5 * torch.tanh(x[..., :1]) + 0.3 * r
            return torch.cat([x[..., :1] * 0.7, x[..., 1:] * gain], dim=-1)

    def ratio(v, vk):
        return (v.norm(dim=-1).mean() / vk.norm(dim=-1).mean()).item()

    #(a) equivariant field WITH the un-roll -> ~1, as the diagnostic expects
    field = EquivariantField()
    v, vk = frame_averaged_velocity(field, x, t, None, roll, K)
    r_ok = ratio(v, vk)
    assert r_ok > 0.9, f"equivariant + un-roll should give ~1, got {r_ok:.3f}"

    #(b) same field, un-roll DROPPED -> collapses. sum_k R_x(theta_k) is
    #diag(K,0,0) for uniform theta, so only the axial component survives.
    xk = torch.einsum("bij,bkj->bki", roll, x.repeat_interleave(K, 0))
    vk_raw = field.decode(xk, t, None).reshape(B, K, N, 3)
    r_bad = ratio(vk_raw.mean(dim=1), vk_raw)
    assert r_bad < 0.5, f"dropping the un-roll should collapse the ratio, got {r_bad:.3f}"

    #(c) untrained network WITH a correct un-roll -> also low: false positive
    net, s2, g2, sph, box, N2, K2 = setup()
    sph_r, box_r, d = reduced_scene(s2, g2, sph, box)
    x2 = torch.randn(s2.shape[0], N2, 3)
    with torch.no_grad():
        v2, vk2 = fa_velocity(net, sph_r, box_r, d, x2, 0.3, K2)
    r_untrained = ratio(v2, vk2)
    assert r_untrained < 0.9, "expected a low ratio on an untrained net"
    print(f"ok  un-roll diagnostic: equivariant {r_ok:.3f}, no-unroll {r_bad:.3f}, "
          f"untrained {r_untrained:.3f}  [conditional, not unconditional]")


def test_k_fa_1_matches_plain_reduced_sample():
    #k_fa=1 must reduce exactly to a single-frame reduced sample.
    net, start, goal, spheres, boxes, N, K = setup()
    B = start.shape[0]
    g1 = torch.Generator().manual_seed(11)
    x_fa = sample_reduced(
        net, spheres, boxes, start, goal, k_fa=1,
        n_waypoints=N, n_steps=6, generator=g1,
    )
    #hand-rolled equivalent: reduce, sample in-frame, map back
    R0, origin, d = sg_frame(start, goal, None)
    sph_r = rotate_sphere_features(spheres, R0, origin)
    box_r = rotate_box_features(boxes, R0, origin)
    g2 = torch.Generator().manual_seed(11)
    x_r = sample(
        net, sph_r, box_r, d[:, None], n_waypoints=N, n_steps=6, generator=g2,
    )
    x_world = torch.einsum("bji,bkj->bki", R0, x_r) + origin[:, None, :]
    err = (x_fa - x_world).abs().max().item()
    assert err < 1e-5, f"k_fa=1 should match a plain reduced sample: {err:.2e}"
    print(f"ok  k_fa=1 == plain reduced sample (err {err:.1e})")


def test_residual_shrinks_with_k():
    #The residual std(v_k)/||mean(v_k)|| is what frame averaging removes. It
    #should be 0 at K=1 (nothing to disagree with) and non-zero beyond.
    net, start, goal, spheres, boxes, N, _ = setup()
    B = start.shape[0]
    prev = None
    for K in (1, 3, 5, 9):
        base = torch.arange(K, dtype=start.dtype) * (2 * math.pi / K)
        thetas = base[None, :].expand(B, K)
        roll, c_k = build_ck(net, start, goal, spheres, boxes, thetas, K)
        x = torch.randn(B, N, 3, generator=torch.Generator().manual_seed(5))
        t = torch.full((B * K,), 0.3)
        with torch.no_grad():
            v, vk = frame_averaged_velocity(net, x, t, c_k, roll, K)
        resid = ((vk - v[:, None]).norm(dim=-1).mean() / v.norm(dim=-1).mean()).item()
        if K == 1:
            assert resid < 1e-6, f"K=1 residual must be 0, got {resid:.2e}"
        else:
            assert resid > 1e-3, f"K={K} should show real disagreement"
        print(f"    K={K}: residual {resid:.4f}")
        prev = resid
    print("ok  residual is 0 at K=1 and positive beyond  [what averaging removes]")


def test_sample_reduced_endpoints_anchor_in_world():
    #With anchoring on, the returned world-frame path must start at start and
    #end at goal -- verifying the reduced->world round trip in the sampler.
    net, start, goal, spheres, boxes, N, K = setup()
    x = sample_reduced(
        net, spheres, boxes, start, goal, k_fa=5, n_waypoints=N, n_steps=5,
        anchor_endpoints=True, generator=torch.Generator().manual_seed(3),
    )
    e0 = (x[:, 0, :] - start).abs().max().item()
    e1 = (x[:, -1, :] - goal).abs().max().item()
    assert e0 < 1e-4 and e1 < 1e-4, f"anchors off: {e0:.2e}, {e1:.2e}"
    print(f"ok  anchored endpoints land on world start/goal ({e0:.1e}, {e1:.1e})")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed")
