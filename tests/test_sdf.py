#Tests for the torch SDF. Run: .venv/bin/python test_sdf.py

#Two independent failure classes, deliberately kept apart:
#  * AGREEMENT -- does it compute the same field as pointmass3d/env.py, which
#    is the checker every reported number is scored against? A torch SDF that
#    disagrees would silently train the model against a different world.
#  * EQUIVARIANCE -- does it give the same answer when scene and query are
#    rigidly moved together? This is what makes it usable on the reduced arm,
#    and it is the property the axis-aligned numpy version cannot even express.
#Agreement alone would pass for an implementation that only handles axis-aligned
#boxes, which is exactly the bug that matters here.

import numpy as np
import torch
import torch.nn as nn

from flowmatch.geometry import (
    apply_points,
    rotate_box_features,
    rotate_sphere_features,
)
from flowmatch.sdf import (
    container_sdf,
    obb_sdf,
    scene_sdf,
    scene_sdf_and_grad,
    sphere_sdf,
    workspace_box,
)
from pointmass3d import BoxObstacle, PointMass3DEnv, SphereObstacle

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    PASS += ok
    FAIL += not ok


def random_scene(rng, n_sph=20, n_box=20):
    sc = rng.uniform(-0.75, 0.75, (n_sph, 3))
    sr = rng.uniform(0.05, 0.18, n_sph)
    bc = rng.uniform(-0.75, 0.75, (n_box, 3))
    bh = rng.uniform(0.05, 0.18, (n_box, 3))
    return sc, sr, bc, bh


def as_tensors(sc, sr, bc, bh):
    spheres = torch.tensor(np.concatenate([sc, sr[:, None]], 1), dtype=torch.float32)[None]
    boxes = np.zeros((len(bc), 12), dtype=np.float32)
    boxes[:, :3] = bc
    boxes[:, 3::4] = bh          # diagonal of the 3x3 half-edge matrix
    return spheres, torch.from_numpy(boxes)[None]


def main():
    rng = np.random.default_rng(0)
    sc, sr, bc, bh = random_scene(rng)
    spheres, boxes = as_tensors(sc, sr, bc, bh)
    pts_np = rng.uniform(-1.0, 1.0, (256, 3))
    pts = torch.tensor(pts_np, dtype=torch.float32)[None]

    # --- agreement with the numpy checker, obstacles only --------------------
    env_obs = PointMass3DEnv(
        [SphereObstacle(c, r) for c, r in zip(sc, sr)]
        + [BoxObstacle(c, h) for c, h in zip(bc, bh)],
        lo=-1e9, hi=1e9, robot_radius=0.0,   # walls pushed away: obstacles only
    )
    ref = env_obs.sdf(pts_np)
    got = scene_sdf(pts, spheres, boxes)[0].numpy()
    err = np.abs(ref - got).max()
    check("matches pointmass3d env.sdf on obstacles", err < 1e-5, f"max |dv| {err:.2e}")

    # separately, each primitive, so a compensating pair of errors cannot hide
    ref_s = np.min([SphereObstacle(c, r).sdf(pts_np) for c, r in zip(sc, sr)], axis=0)
    got_s = sphere_sdf(pts, spheres)[0].amin(-1).numpy()
    check("sphere term", np.abs(ref_s - got_s).max() < 1e-5)
    ref_b = np.min([BoxObstacle(c, h).sdf(pts_np) for c, h in zip(bc, bh)], axis=0)
    got_b = obb_sdf(pts, boxes)[0].amin(-1).numpy()
    check("axis-aligned box term", np.abs(ref_b - got_b).max() < 1e-5)

    # --- the workspace container, against env.sdf's wall term ----------------
    env_walls = PointMass3DEnv([], lo=-1.0, hi=1.0, robot_radius=0.0)
    ref_w = env_walls.sdf(pts_np)
    got_w = container_sdf(pts, workspace_box(-1.0, 1.0))[0].numpy()
    check("workspace container matches the wall term",
          np.abs(ref_w - got_w).max() < 1e-5)

    # --- full field including walls ------------------------------------------
    env_full = PointMass3DEnv(
        [SphereObstacle(c, r) for c, r in zip(sc, sr)]
        + [BoxObstacle(c, h) for c, h in zip(bc, bh)],
        lo=-1.0, hi=1.0, robot_radius=0.0,
    )
    got_f = scene_sdf(pts, spheres, boxes, container=workspace_box(-1.0, 1.0))[0].numpy()
    check("full field with walls", np.abs(env_full.sdf(pts_np) - got_f).max() < 1e-5)

    # --- rigid-motion equivariance: the reason this module exists -------------
    # Rotating scene AND query together must leave every distance unchanged.
    # The numpy version cannot represent the rotated boxes at all, so this is
    # the property that has no reference implementation to check against and
    # must be asserted structurally instead.
    g = torch.Generator().manual_seed(7)
    a = torch.randn(3, 3, generator=g)
    Q, R_ = torch.linalg.qr(a)
    Q = Q * torch.sign(torch.diagonal(R_))[None, :]
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    Q = Q[None]                                   # (1,3,3)
    origin = torch.tensor([[0.31, -0.22, 0.17]])  # arbitrary, non-zero

    pts_r = apply_points(Q, origin, pts)
    sph_r = rotate_sphere_features(spheres, Q, origin)
    box_r = rotate_box_features(boxes, Q, origin)
    cont_r = rotate_box_features(workspace_box(-1.0, 1.0), Q, origin)

    d0 = scene_sdf(pts, spheres, boxes, container=workspace_box(-1.0, 1.0))
    d1 = scene_sdf(pts_r, sph_r, box_r, container=cont_r)
    e = (d0 - d1).abs().max().item()
    check("rigid motion of scene+query leaves the field invariant", e < 1e-4,
          f"max |dv| {e:.2e}")

    # and the ORIENTED box really is oriented: rotating the box alone must
    # change the field, or the test above would pass for a stub that ignores
    # the edge directions entirely
    d2 = scene_sdf(pts, spheres, box_r)
    moved = (scene_sdf(pts, spheres, boxes) - d2).abs().max().item()
    check("rotating boxes alone does change the field", moved > 1e-3,
          f"max |dv| {moved:.2e}")

    # --- gradient against central differences --------------------------------
    d, grad = scene_sdf_and_grad(pts, spheres, boxes)
    eps = 1e-3
    fd = torch.zeros_like(grad)
    for k in range(3):
        dp = torch.zeros(3)
        dp[k] = eps
        fd[..., k] = (scene_sdf(pts + dp, spheres, boxes)
                      - scene_sdf(pts - dp, spheres, boxes)) / (2 * eps)
    # the field is only piecewise smooth: at a ridge equidistant from two
    # obstacles the derivative jumps, and finite differences straddle it. Score
    # the median rather than the max so a handful of ridge points cannot fail a
    # correct gradient.
    rel = (grad - fd).norm(dim=-1).median().item()
    check("autograd gradient matches central differences (median)", rel < 1e-2,
          f"median |dv| {rel:.2e}")
    unit = grad.norm(dim=-1)
    check("gradient is a unit field away from ridges",
          (unit.median() - 1.0).abs().item() < 0.05,
          f"median |grad| {unit.median().item():.4f}")

    # --- the oracle wiring: are the appended channels frame-consistent? ------
    # NOT "the model is equivariant" -- the trunk is unconstrained and mixes
    # coordinate channels with free weights, so it is not, and the size of that
    # deviation is precisely what the residual r measures. An earlier version of
    # this test asserted end-to-end equivariance and passed only because
    # out_conv is zero-initialised, i.e. it compared 0 against 0.
    #
    # What must hold is that decode() sees the SDF evaluated in the SAME frame
    # as the state: the value is invariant under a rigid motion of scene+query,
    # and the gradient is a free VECTOR that rotates with it. If either failed,
    # the reduced arm would train against geometry from a different frame -- a
    # silent error, exactly the class Sec. "Implementation" is written around.
    d0, g0 = scene_sdf_and_grad(pts, spheres, boxes)
    d1, g1 = scene_sdf_and_grad(pts_r, sph_r, box_r)
    ev = (d0 - d1).abs().max().item()
    check("oracle channel: SDF value is frame-invariant", ev < 1e-4, f"max |dv| {ev:.2e}")
    eg = (torch.einsum("bji,bkj->bki", Q, g1) - g0).abs().max().item()
    check("oracle channel: SDF gradient rotates with the frame", eg < 1e-3,
          f"max |dv| {eg:.2e}")

    # and the guards: obstacles are mandatory with local_geom, optional without
    from flowmatch.model import FlowVelocityField

    torch.manual_seed(0)
    net = FlowVelocityField(channels=16, n_blocks=2, env_hidden=16, env_dim=16,
                            cond_dim=16, groups=4, sg_dim=1, local_geom=True).eval()
    plain = FlowVelocityField(channels=16, n_blocks=2, env_hidden=16, env_dim=16,
                              cond_dim=16, groups=4, sg_dim=1).eval()
    x = torch.randn(1, 12, 3)
    tt, sg = torch.rand(1), torch.rand(1, 1)
    with torch.no_grad():
        c0 = net.encode_cond(spheres, boxes, sg)
        out = net.decode(x, tt, c0, spheres, boxes)
        ok = out.shape == x.shape
        try:
            net.decode(x, tt, c0)          # obstacles omitted -> must raise
            ok = False
        except ValueError:
            pass
        plain.decode(x, tt, plain.encode_cond(spheres, boxes, sg))  # must not raise
    check("local_geom needs obstacles and preserves output shape; "
          "plain path unaffected", ok)

    print(f"\n{PASS}/{PASS + FAIL} passed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
