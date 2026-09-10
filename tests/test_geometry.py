
  #.venv/bin/python test_geometry.py

  #Correctness tests for the (s,g) reduction. These cover disjoint failure
  #classes and every one of them is silent if it fails in production: a wrong
  #reduction presents as slightly-worse training, not as a crash.

  #The reference implementations below are deliberately written as explicit
  #per-item numpy loops rather than reusing the batched helpers, so the tests
  #catch einsum index errors and reshape bugs instead of agreeing with them.

import numpy as np
import torch

from flowmatch.geometry import (
    aabb_edges,
    apply_points,
    apply_vectors,
    box_features,
    check_frame,
    rotate_box_features,
    rotate_sphere_features,
    sg_frame,
    split_box_features,
)


def random_problem(B=16, K=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    #match the generator: obstacle centers in [-0.75,0.75]^3, endpoints in
    #[-1,1]^3 with ||g-s|| >= 1.5
    start = torch.rand(B, 3, generator=g) * 2 - 1
    goal = torch.rand(B, 3, generator=g) * 2 - 1
    d_vec = goal - start
    too_close = torch.linalg.norm(d_vec, dim=-1) < 1.5
    goal[too_close] = start[too_close] - 1.8 * torch.nn.functional.normalize(
        d_vec[too_close] + 1e-3, dim=-1
    )
    centers = (torch.rand(B, K, 3, generator=g) * 2 - 1) * 0.75
    half = 0.08 + torch.rand(B, K, 3, generator=g) * 0.12
    radii = 0.10 + torch.rand(B, K, 1, generator=g) * 0.15
    theta = torch.rand(B, generator=g) * 2 * np.pi
    return start, goal, centers, half, radii, theta


def reference_box_features(centers, half, R, origin):
    #Independent path: rotate the geometry item-by-item in numpy, then pack.
    #Centers are POINTS (affine); half-edges are FREE VECTORS (rotation only).
    B, K = centers.shape[:2]
    Rn, on = R.numpy(), origin.numpy()
    cn, hn = centers.numpy(), half.numpy()
    out = np.zeros((B, K, 12), dtype=np.float64)
    for b in range(B):
        for k in range(K):
            out[b, k, :3] = Rn[b] @ (cn[b, k] - on[b])
            for i in range(3):
                edge = np.zeros(3)
                edge[i] = hn[b, k, i]
                out[b, k, 3 + 3 * i : 6 + 3 * i] = Rn[b] @ edge
    return out


def test_frame_lands_start_goal_on_axis():
    #Assertion 1: start/goal must land exactly on (-+d/2, 0, 0).
    #Catches translation errors, axis ordering, and R-vs-R^T.
    start, goal, _, _, _, theta = random_problem()
    for th in (None, theta):
        R, origin, d = sg_frame(start, goal, th)
        check_frame(R, origin, start, goal, d)
    print("ok  start/goal land on (-+d/2, 0, 0)  [with and without roll]")


def test_frame_is_right_handed_and_orthonormal():
    #Assertion 2: det(R) = +1. Assertion 1 is blind to reflections that fix
    #the x-axis, so this is not redundant with it.
    start, goal, _, _, _, theta = random_problem(B=64, seed=1)
    R, _, _ = sg_frame(start, goal, theta)
    det = torch.linalg.det(R)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5), det.min()
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    #the gauge bound: ||x_hat x e_i|| >= sqrt(2/3) everywhere
    x_hat = R[:, 0, :]
    smallest = x_hat.abs().min(dim=-1).values
    assert (smallest <= 1 / np.sqrt(3) + 1e-6).all()
    assert (torch.sqrt(1 - smallest**2) >= np.sqrt(2 / 3) - 1e-6).all()
    print("ok  det(R)=+1, orthonormal, gauge norm >= sqrt(2/3)")


def test_reflection_is_caught_by_det_but_not_by_start_goal():
    #Proves the two assertions cover different things: a y-z reflection fixes
    #the x-axis, so start/goal still land correctly but handedness is wrong.
    start, goal, _, _, _, theta = random_problem(B=8, seed=2)
    R, origin, d = sg_frame(start, goal, theta)
    R_bad = R.clone()
    R_bad[:, 1, :] *= -1  # flip y_hat -> left-handed
    s_r = apply_points(R_bad, origin, start[:, None, :])[:, 0, :]
    want = torch.stack([-0.5 * d, torch.zeros_like(d), torch.zeros_like(d)], -1)
    assert torch.allclose(s_r, want, atol=1e-4), "reflection should still fix start"
    assert (torch.linalg.det(R_bad) < 0).all(), "det must catch the reflection"
    try:
        check_frame(R_bad, origin, start, goal, d)
        raise AssertionError("check_frame missed a left-handed frame")
    except AssertionError as e:
        assert "left-handed" in str(e), e
    print("ok  reflection passes start/goal but is caught by det  [tests are disjoint]")


def test_featurize_rotate_equals_typed_rotate():
    #Assertion 3: featurize(rotate(S)) == typed_rotate(featurize(S)).
    #This is the one that catches the points-vs-vectors bug.
    start, goal, centers, half, _, theta = random_problem(B=12, K=7, seed=3)
    R, origin, _ = sg_frame(start, goal, theta)

    feat_world = box_features(centers, aabb_edges(half))
    got = rotate_box_features(feat_world, R, origin)
    want = torch.from_numpy(reference_box_features(centers, half, R, origin)).float()
    err = (got - want).abs().max().item()
    assert err < 1e-4, f"typed_rotate disagrees with reference: {err:.2e}"
    print(f"ok  featurize(rotate(S)) == typed_rotate(featurize(S))  [max err {err:.1e}]")


def test_affine_on_edges_is_detectably_wrong():
    #The bug this guards against, made explicit: applying the full affine to
    #half-edge vectors adds the common offset -R@origin to all three, swamping
    #extent while leaving position correct.
    #Note the offset is COMMON, so deviation-from-mean is bit-identical between
    #the two paths -- the extent information survives in the differences and is
    #merely swamped. What actually changes is the norms: |h| ~ 0.1 becomes
    #|origin| ~ 1.6. That is the signature to test.
    start, goal, centers, half, _, theta = random_problem(B=12, K=7, seed=4)
    R, origin, _ = sg_frame(start, goal, theta)
    feat_world = box_features(centers, aabb_edges(half))
    correct = rotate_box_features(feat_world, R, origin)

    #wrong version: affine applied to the edges too
    c, edges = split_box_features(feat_world)
    B, K = edges.shape[:2]
    wrong_edges = apply_points(R, origin, edges.reshape(B, K * 3, 3)).reshape(B, K, 3, 3)
    wrong = box_features(apply_points(R, origin, c), wrong_edges)

    correct_edges = split_box_features(correct)[1]
    n_correct = torch.linalg.norm(correct_edges, dim=-1)
    assert torch.allclose(n_correct, half, atol=1e-5)

    #Exact identity: wrong - correct = -R@origin for EVERY edge, so the
    #per-edge error is exactly |origin|. Note what this means -- the severity
    #scales with |(s+g)/2|, so the bug is INVISIBLE whenever start and goal
    #straddle the world origin. Since both are drawn symmetrically about it,
    #that happens often, which is a further reason it stays silent in training.
    err = torch.linalg.norm(wrong_edges - correct_edges, dim=-1)   # (B,K,3)
    o_norm = torch.linalg.norm(origin, dim=-1)                     # (B,)
    assert torch.allclose(err, o_norm[:, None, None].expand_as(err), atol=1e-5), (
        f"error should equal |origin| exactly: {(err - o_norm[:, None, None]).abs().max():.2e}"
    )

    #On the samples where the offset is non-negligible, the signature shows up:
    #norms blow up, the three edges become near-parallel, orthogonality breaks.
    big = o_norm > 0.5
    assert big.any(), "need at least one off-center problem to test the signature"
    we, ce = wrong_edges[big], correct_edges[big]
    assert (torch.linalg.norm(we, dim=-1) > 2 * half[big]).all()

    def rel_spread(e):
        dev = (e - e.mean(dim=2, keepdim=True)).norm(dim=-1)
        return (dev / e.norm(dim=-1).clamp_min(1e-9)).mean().item()

    rs_correct, rs_wrong = rel_spread(ce), rel_spread(we)
    assert rs_wrong < 0.5 * rs_correct, (rs_wrong, rs_correct)

    gram = we @ we.transpose(-1, -2)
    off = (gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))).abs().max()
    assert off > 0.1, f"expected non-orthogonal edges, got {off:.2e}"

    print(f"ok  affine-on-edges: per-edge error == |origin| exactly; on off-center "
          f"problems rel spread {rs_correct:.2f}->{rs_wrong:.2f}, orthogonality "
          f"broken ({off:.2f})  [test has teeth]")


def test_edge_norms_and_radius_are_rotation_invariant():
    #Half-edge lengths and sphere radii are m=0 under the frame: rotation
    #must not change them.
    start, goal, centers, half, radii, theta = random_problem(B=10, K=6, seed=5)
    R, origin, _ = sg_frame(start, goal, theta)
    feat = rotate_box_features(box_features(centers, aabb_edges(half)), R, origin)
    _, edges_r = split_box_features(feat)
    norms = torch.linalg.norm(edges_r, dim=-1)
    assert torch.allclose(norms, half, atol=1e-5), (norms - half).abs().max()
    #reduced-frame edges must stay mutually orthogonal (a box stays a box)
    gram = edges_r @ edges_r.transpose(-1, -2)
    off = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
    assert off.abs().max() < 1e-5, off.abs().max()

    sph = torch.cat([centers, radii], dim=-1)
    sph_r = rotate_sphere_features(sph, R, origin)
    assert torch.equal(sph_r[..., 3:], radii), "radius must be untouched"
    print("ok  edge norms + orthogonality preserved, radius invariant")


def test_roll_is_a_rotation_about_x_and_preserves_rho():
    #Rotation about x_hat preserves x and rho = sqrt(y^2+z^2) exactly, and only
    #advances phi. This is why a cylinder domain is exactly roll-invariant.
    start, goal, centers, _, _, _ = random_problem(B=8, K=9, seed=6)
    R0, origin, _ = sg_frame(start, goal, torch.zeros(8))
    p0 = apply_points(R0, origin, centers)
    for th in (0.3, 1.0, 2.5):
        Rt, _, _ = sg_frame(start, goal, torch.full((8,), th))
        pt = apply_points(Rt, origin, centers)
        assert torch.allclose(p0[..., 0], pt[..., 0], atol=1e-5), "x must be fixed"
        rho0 = torch.linalg.norm(p0[..., 1:], dim=-1)
        rhot = torch.linalg.norm(pt[..., 1:], dim=-1)
        assert torch.allclose(rho0, rhot, atol=1e-5), "rho must be fixed"
    print("ok  roll fixes x and rho, advances phi only")


def test_roll_composition_is_additive():
    #Frame at angle a+b must equal R_x(b) applied to the frame at angle a.
    #Guards the composition order in R_total = R_x(theta) @ T.
    start, goal, centers, _, _, _ = random_problem(B=8, K=4, seed=7)
    a, b = 0.7, 1.1
    Rab, origin, _ = sg_frame(start, goal, torch.full((8,), a + b))
    Ra, _, _ = sg_frame(start, goal, torch.full((8,), a))
    c, s = float(np.cos(b)), float(np.sin(b))
    Rx = torch.tensor(
        [[1.0, 0, 0], [0, c, s], [0, -s, c]], dtype=torch.float32
    ).expand(8, 3, 3)
    assert torch.allclose(Rab, Rx @ Ra, atol=1e-5), (Rab - Rx @ Ra).abs().max()
    print("ok  roll composes additively: frame(a+b) == R_x(b) @ frame(a)")


def test_inverse_recovers_world():
    #T.inverse must round-trip: needed because collision checking happens in
    #world frame after sampling in the reduced one.
    start, goal, centers, _, _, theta = random_problem(B=10, K=8, seed=8)
    R, origin, _ = sg_frame(start, goal, theta)
    p_r = apply_points(R, origin, centers)
    back = torch.einsum("bji,bkj->bki", R, p_r) + origin[:, None, :]  # R^T @ p + o
    assert torch.allclose(back, centers, atol=1e-5), (back - centers).abs().max()
    print("ok  reduced -> world round-trip exact")


def test_se3_reduces_to_a_single_roll():
    #Proposition 1 of the writeup, stated as it is actually true. Moving the
    #whole problem by a rigid T does NOT leave the reduced representation
    #unchanged: y_hat is built from a FIXED world axis e_{i*}, so a global
    #rotation changes which axis is chosen and rolls the frame. What must hold
    #-- and what the roll augmentation and frame averaging are premised on --
    #is that the entire residual is ONE roll about x_hat, shared by the
    #trajectory, the sphere centres and the box edges alike. The claim
    #"the reduced representation is bit-identical under SE(3)" is too strong;
    #this is the correct version, and it says SE(3) collapses to SO(2).
    from flowmatch.flow import reduce_batch

    rng = np.random.default_rng(21)
    B, N, K = 6, 16, 7
    traj = torch.tensor(rng.standard_normal((B, N, 3)) * 0.3)
    start, goal = traj[:, 0, :] - 1.0, traj[:, -1, :] + 1.0
    traj[:, 0, :], traj[:, -1, :] = start, goal
    spheres = torch.tensor(np.concatenate(
        [rng.standard_normal((B, K, 3)), rng.uniform(0.05, 0.2, (B, K, 1))], -1))
    boxes = box_features(torch.tensor(rng.standard_normal((B, K, 3))),
                         aabb_edges(torch.tensor(rng.uniform(0.05, 0.2, (B, K, 3)))))

    #a random rigid motion, written through apply_points as R=Q, origin=-Q^T t
    Q, _ = torch.linalg.qr(torch.tensor(rng.standard_normal((B, 3, 3))))
    Q = Q * torch.sign(torch.linalg.det(Q))[:, None, None]
    t = torch.tensor(rng.standard_normal((B, 3)) * 3.0)
    o = -torch.einsum("bji,bj->bi", Q, t)
    moved = (
        apply_points(Q, o, traj),
        apply_points(Q, o, start[:, None, :])[:, 0],
        apply_points(Q, o, goal[:, None, :])[:, 0],
        torch.cat([apply_points(Q, o, spheres[..., :3]), spheres[..., 3:]], -1),
        torch.cat([apply_points(Q, o, boxes[..., :3]),
                   apply_vectors(Q, boxes[..., 3:].reshape(B, K * 3, 3)
                                 ).reshape(B, K, 9)], -1),
    )

    a = reduce_batch(traj, start, goal, spheres, boxes, roll=False)
    b = reduce_batch(*moved, roll=False)
    Ra, _, da = sg_frame(start, goal, None)
    Rb, _, db = sg_frame(moved[1], moved[2], None)
    M = torch.einsum("bij,bjk,blk->bil", Rb, Q, Ra)          # a-frame -> b-frame

    e1 = torch.zeros(B, 3, dtype=M.dtype); e1[:, 0] = 1.0
    assert torch.allclose(M[:, 0, :], e1, atol=1e-9), "residual is not a roll about x"
    assert torch.allclose(torch.einsum("bij,bkj->bik", M, M),
                          torch.eye(3, dtype=M.dtype).expand(B, 3, 3), atol=1e-9)

    def rot(v):
        return torch.einsum("bij,bkj->bki", M, v)

    assert torch.allclose(rot(a[0]), b[0], atol=1e-9)
    assert torch.allclose(rot(a[2][..., :3]), b[2][..., :3], atol=1e-9)
    assert torch.allclose(rot(a[3][..., 3:].reshape(B, K * 3, 3)),
                          b[3][..., 3:].reshape(B, K * 3, 3), atol=1e-9)
    #and the roll-invariant coordinates are preserved outright
    assert torch.allclose(a[0][..., 0], b[0][..., 0], atol=1e-9)
    assert torch.allclose(a[0][..., 1:3].norm(dim=-1),
                          b[0][..., 1:3].norm(dim=-1), atol=1e-9)
    assert torch.allclose(da, db, atol=1e-9)
    print("ok  SE(3) reduces to exactly one roll (x, rho, d invariant)")


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed")
