
  #.venv/bin/python test_se3.py

  #Correctness tests for the SE(3) rigid-body domain.

  #The load-bearing test is test_reduction_reduces_se3_to_a_single_roll: it is
  #Proposition 1 of the paper, restated for a state that is no longer a bare
  #point. Note what it does NOT assert. Rigidly moving the problem does not
  #leave the reduced representation unchanged -- the gauge is built from a
  #fixed world axis, so a global rotation rolls the frame. What must hold is
  #that the residual is exactly ONE roll about the start-goal axis, shared by
  #every reduced quantity: SE(3) collapses to SO(2), which is precisely the
  #one degree of freedom the rest of the paper is about.

  #test_affine_on_rotation_is_detected exists because the interesting bug in
  #this domain is silent. Applying the affine map to a rotation column instead
  #of the rotation-only map leaves every position correct and every path
  #plausible, while destroying orientation. That test asserts the wrong
  #implementation actually fails, so the right one is not passing by accident.

import numpy as np
import torch

from flowmatch.geometry import apply_points, apply_vectors, box_features, aabb_edges
from pointmass3d import BoxObstacle, PointMass3DEnv, SphereObstacle, make_random_env
from se3body import (
    RigidBody,
    SE3Env,
    geodesic_angle,
    matrix_to_6d,
    plan_se3,
    rand_rotation,
    reduce_batch_se3,
    sixd_to_matrix,
    slerp,
    transform_states,
    unreduce_states,
)


def _report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return bool(ok)


# ---------------------------------------------------------------- rotations

def test_rotation_roundtrip():
    rng = np.random.default_rng(0)
    R = rand_rotation(rng, size=64)
    err = np.abs(sixd_to_matrix(matrix_to_6d(R)) - R).max()
    #and rand_rotation must actually be in SO(3)
    ortho = np.abs(np.einsum("bij,bkj->bik", R, R) - np.eye(3)).max()
    dets = np.linalg.det(R)
    ok = err < 1e-10 and ortho < 1e-10 and np.allclose(dets, 1.0)
    return _report("6D <-> matrix roundtrip, rand_rotation in SO(3)", ok,
                   f"(err {err:.1e})")


def test_sixd_gram_schmidt_on_garbage():
    #The representation must map ARBITRARY six numbers to a valid rotation --
    #that is what lets the network emit unconstrained outputs.
    rng = np.random.default_rng(1)
    d6 = rng.standard_normal((256, 6)) * 5.0
    R = sixd_to_matrix(d6)
    ortho = np.abs(np.einsum("bij,bkj->bik", R, R) - np.eye(3)).max()
    ok = ortho < 1e-9 and np.allclose(np.linalg.det(R), 1.0)
    return _report("6D -> valid rotation for arbitrary input", ok,
                   f"(orthonormality err {ortho:.1e})")


def test_slerp():
    rng = np.random.default_rng(2)
    A, B = rand_rotation(rng), rand_rotation(rng)
    ok = np.abs(slerp(A, B, 0.0) - A).max() < 1e-10
    ok &= np.abs(slerp(A, B, 1.0) - B).max() < 1e-10
    #the midpoint must be equidistant, and distances must add up
    full = geodesic_angle(A, B)
    half = slerp(A, B, 0.5)
    ok &= abs(geodesic_angle(A, half) - full / 2) < 1e-8
    ok &= abs(geodesic_angle(half, B) - full / 2) < 1e-8
    return _report("slerp: endpoints exact, midpoint equidistant", ok)


# ------------------------------------------------------------ typed transform

def _brute_force_transform(x_np, R_np, origin_np):
    #Reference implementation: explicit loops, positions affine and the
    #rotation MATRIX premultiplied, then re-encoded. Deliberately written a
    #different way from the code under test so it cannot share a bug.
    B, N = x_np.shape[:2]
    out = np.empty_like(x_np)
    for b in range(B):
        for n in range(N):
            p = x_np[b, n, :3]
            A = sixd_to_matrix(x_np[b, n, 3:])       # (3,3), columns
            out[b, n, :3] = R_np[b] @ (p - origin_np[b])
            out[b, n, 3:] = matrix_to_6d(R_np[b] @ A)
    return out


def test_transform_states_matches_reference():
    rng = np.random.default_rng(3)
    B, N = 4, 8
    pos = rng.standard_normal((B, N, 3))
    rot = np.stack([matrix_to_6d(rand_rotation(rng, size=N)) for _ in range(B)])
    x = np.concatenate([pos, rot], axis=-1)
    R = rand_rotation(rng, size=B)
    origin = rng.standard_normal((B, 3))

    got = transform_states(torch.tensor(x), torch.tensor(R),
                           torch.tensor(origin)).numpy()
    want = _brute_force_transform(x, R, origin)
    err = np.abs(got - want).max()
    return _report("transform_states matches a brute-force reference",
                   err < 1e-9, f"(err {err:.1e})")


def test_affine_on_rotation_is_detected():
    #The silent bug: treat a rotation column as a point. Positions stay right,
    #orientation is destroyed. Assert that it is loudly different.
    rng = np.random.default_rng(4)
    B, N = 3, 6
    pos = rng.standard_normal((B, N, 3))
    rot = np.stack([matrix_to_6d(rand_rotation(rng, size=N)) for _ in range(B)])
    x = torch.tensor(np.concatenate([pos, rot], axis=-1))
    R = torch.tensor(rand_rotation(rng, size=B))
    origin = torch.tensor(rng.standard_normal((B, 3)) * 2.0)

    right = transform_states(x, R, origin)
    #the wrong version: affine map applied to everything
    cols = x[..., 3:].reshape(B, N * 2, 3)
    wrong_cols = apply_points(R, origin, cols).reshape(B, N, 6)
    wrong = torch.cat([apply_points(R, origin, x[..., :3]), wrong_cols], dim=-1)

    #right: columns keep unit norm. wrong: they do not.
    n_right = right[..., 3:6].norm(dim=-1)
    n_wrong = wrong[..., 3:6].norm(dim=-1)
    ok = torch.allclose(n_right, torch.ones_like(n_right), atol=1e-9)
    ok &= (n_wrong - 1.0).abs().max() > 0.1
    return _report("affine-on-rotation bug is detected, not tolerated", bool(ok),
                   f"(wrong-norm drift {float((n_wrong-1).abs().max()):.2f})")


# ------------------------------------------------------------- the reduction

def _random_problem(rng, B=4, N=12, K=5):
    pos = rng.standard_normal((B, N, 3)) * 0.3
    rot = np.stack([matrix_to_6d(rand_rotation(rng, size=N)) for _ in range(B)])
    traj = np.concatenate([pos, rot], axis=-1)
    start = traj[:, 0, :].copy()
    goal = traj[:, -1, :].copy()
    #make start/goal well separated so the frame is well conditioned
    start[:, :3] -= 1.0
    goal[:, :3] += 1.0
    traj[:, 0, :], traj[:, -1, :] = start, goal
    spheres = np.concatenate([rng.standard_normal((B, K, 3)),
                              rng.uniform(0.05, 0.2, (B, K, 1))], axis=-1)
    centers = torch.tensor(rng.standard_normal((B, K, 3)))
    edges = aabb_edges(torch.tensor(rng.uniform(0.05, 0.2, (B, K, 3))))
    boxes = box_features(centers, edges).numpy()
    return (torch.tensor(traj), torch.tensor(start), torch.tensor(goal),
            torch.tensor(spheres), torch.tensor(boxes))


def _apply_rigid(Q, t, traj, start, goal, spheres, boxes):
    #Move the WHOLE problem by T = (Q, t). apply_points(R, o, p) = R(p - o),
    #so T is R=Q with o = -Q^T t.
    B = traj.shape[0]
    origin = -torch.einsum("bji,bj->bi", Q, t)
    tr = transform_states(traj, Q, origin)
    st = transform_states(start[:, None, :], Q, origin)[:, 0, :]
    gl = transform_states(goal[:, None, :], Q, origin)[:, 0, :]
    sph = torch.cat([apply_points(Q, origin, spheres[..., :3]), spheres[..., 3:]], -1)
    K = boxes.shape[1]
    c_r = apply_points(Q, origin, boxes[..., :3])
    e_r = apply_vectors(Q, boxes[..., 3:].reshape(B, K * 3, 3)).reshape(B, K, 9)
    return tr, st, gl, sph, torch.cat([c_r, e_r], dim=-1)


def test_reduction_reduces_se3_to_a_single_roll():
    #Proposition 1, stated correctly. Moving the whole problem by T does NOT
    #leave the reduced representation unchanged: the gauge vector y_hat is
    #built from a FIXED world axis e_{i*}, so a global rotation changes which
    #axis is picked and rolls the frame. What is true -- and is the whole
    #premise of the paper -- is that the residual is exactly ONE rotation about
    #the start-goal axis, shared by every reduced quantity. SE(3) is reduced to
    #SO(2), not to nothing.
    from flowmatch.geometry import sg_frame
    rng = np.random.default_rng(5)
    traj, start, goal, spheres, boxes = _random_problem(rng)
    B = traj.shape[0]
    Q = torch.tensor(rand_rotation(rng, size=B))
    t = torch.tensor(rng.standard_normal((B, 3)) * 3.0)

    a = reduce_batch_se3(traj, start, goal, spheres, boxes, roll=False, check=True)
    moved = _apply_rigid(Q, t, traj, start, goal, spheres, boxes)
    b = reduce_batch_se3(*moved, roll=False, check=True)

    Ra, _, da = sg_frame(start[..., :3], goal[..., :3], None)
    Rb, _, db = sg_frame(moved[1][..., :3], moved[2][..., :3], None)
    M = torch.einsum("bij,bjk,blk->bil", Rb, Q, Ra)      # a-frame -> b-frame

    #(1) M is a rotation about the first axis
    eye = torch.eye(3, dtype=M.dtype).expand(B, 3, 3)
    e1 = torch.zeros(B, 3, dtype=M.dtype); e1[:, 0] = 1.0
    ok = torch.allclose(M[:, 0, :], e1, atol=1e-9)
    ok &= torch.allclose(torch.einsum("bij,bkj->bik", M, M), eye, atol=1e-9)
    ok &= torch.allclose(torch.linalg.det(M), torch.ones(B, dtype=M.dtype), atol=1e-9)

    #(2) EVERY reduced quantity maps through that same single roll
    def rot(v):
        return torch.einsum("bij,bkj->bki", M, v)
    errs = {
        "positions": float((rot(a[0][..., :3]) - b[0][..., :3]).abs().max()),
        "rot cols": float((rot(a[0][..., 3:].reshape(B, -1, 3))
                           - b[0][..., 3:].reshape(B, -1, 3)).abs().max()),
        "spheres": float((rot(a[2][..., :3]) - b[2][..., :3]).abs().max()),
    }
    #(3) the roll-invariant coordinates are preserved exactly
    errs["x coord"] = float((a[0][..., 0] - b[0][..., 0]).abs().max())
    errs["yz radius"] = float((a[0][..., 1:3].norm(dim=-1)
                               - b[0][..., 1:3].norm(dim=-1)).abs().max())
    errs["d"] = float((da - db).abs().max())
    worst = max(errs.values())
    ok &= worst < 1e-8
    return _report("SE(3) reduces to exactly one roll (Prop. 1)", bool(ok),
                   f"(worst {worst:.1e} over {', '.join(errs)})")


def test_conditioning_dimensions():
    rng = np.random.default_rng(6)
    traj, start, goal, spheres, boxes = _random_problem(rng)
    _, sg, _, _, _, _ = reduce_batch_se3(traj, start, goal, spheres, boxes,
                                         roll=False)
    world = torch.cat([start, goal], dim=-1)
    ok = sg.shape[-1] == 13 and world.shape[-1] == 18
    #and the first conditioning entry must be the start-goal distance
    d = (goal[..., :3] - start[..., :3]).norm(dim=-1)
    ok &= torch.allclose(sg[:, 0], d, atol=1e-6)
    return _report("conditioning collapses 18 -> 13, first entry is d", bool(ok))


def test_reduce_unreduce_roundtrip():
    rng = np.random.default_rng(7)
    traj, start, goal, spheres, boxes = _random_problem(rng)
    traj_r, _, _, _, R, origin = reduce_batch_se3(traj, start, goal, spheres,
                                                  boxes, roll=False)
    back = unreduce_states(traj_r, R, origin)
    err = float((back - traj).abs().max())
    return _report("reduce -> unreduce is the identity", err < 1e-9,
                   f"(err {err:.1e})")


def test_roll_changes_frame_but_not_positions():
    #Roll augmentation must move the representation without moving the
    #canonical endpoints: start/goal stay on (-+d/2,0,0) for every roll.
    rng = np.random.default_rng(8)
    traj, start, goal, spheres, boxes = _random_problem(rng)
    outs = [reduce_batch_se3(traj, start, goal, spheres, boxes, roll=True,
                             check=True) for _ in range(4)]
    d = (goal[..., :3] - start[..., :3]).norm(dim=-1)
    ok = True
    for o in outs:
        ok &= torch.allclose(o[0][:, 0, 0], -0.5 * d, atol=1e-5)
        ok &= torch.allclose(o[0][:, -1, 0], 0.5 * d, atol=1e-5)
        ok &= torch.allclose(o[0][:, 0, 1:3], torch.zeros_like(o[0][:, 0, 1:3]),
                             atol=1e-5)
    #different rolls must actually differ somewhere off-axis
    differs = float((outs[0][0][..., 1:3] - outs[1][0][..., 1:3]).abs().max())
    ok &= differs > 1e-3
    return _report("roll moves the frame, endpoints stay canonical", bool(ok),
                   f"(roll delta {differs:.2f})")


# --------------------------------------------------------------- body & env

def test_clearance_is_rigidly_invariant():
    #Physical check: moving the scene and the body together cannot change
    #clearance. This is the SE(3) symmetry of the PROBLEM, independent of any
    #learning, and it is what the reduction is allowed to exploit.
    #The walls must be moved with everything else or they are not part of the
    #rigid motion; the cleanest way to say that is to push them out of range in
    #BOTH scenes, so the quantity compared is obstacle clearance alone. An
    #earlier version of this test translated only the obstacles and measured a
    #pose whose clearance was set by a wall, which is a wrong test rather than
    #a broken invariance.
    rng = np.random.default_rng(9)
    base = make_random_env(n_spheres=8, n_boxes=8, seed=9)
    far = 50.0
    env = SE3Env(PointMass3DEnv(base.obstacles, lo=-far, hi=far), RigidBody())
    p, R = SE3Env(base, RigidBody()).sample_free_pose(rng)
    c0 = float(env.clearance(p, R))

    #Axis-aligned boxes do not stay axis-aligned under a rotation, so the
    #rotational part is exercised on a sphere-only scene and the translational
    #part on the full one.
    t = rng.standard_normal(3) * 0.7
    obs_t = [SphereObstacle(o.center + t, o.radius) if isinstance(o, SphereObstacle)
             else BoxObstacle(o.center + t, o.half_extents) for o in base.obstacles]
    env_t = SE3Env(PointMass3DEnv(obs_t, lo=-far, hi=far), RigidBody())
    c_trans = float(env_t.clearance(p + t, R))

    sph = [o for o in base.obstacles if isinstance(o, SphereObstacle)]
    Q = rand_rotation(rng)
    env_s = SE3Env(PointMass3DEnv(sph, lo=-far, hi=far), RigidBody())
    env_sq = SE3Env(PointMass3DEnv([SphereObstacle(Q @ o.center, o.radius) for o in sph],
                                   lo=-far, hi=far), RigidBody())
    c_rot0 = float(env_s.clearance(p, R))
    c_rot1 = float(env_sq.clearance(Q @ p, Q @ R))

    ok = abs(c0 - c_trans) < 1e-9 and abs(c_rot0 - c_rot1) < 1e-9
    return _report("clearance is invariant under rigid motion of the scene", ok,
                   f"(translate {c0:.4f}/{c_trans:.4f}, "
                   f"rotate {c_rot0:.4f}/{c_rot1:.4f})")


def test_orientation_matters():
    #If clearance were independent of orientation the domain would be a
    #point-mass problem in disguise. Find a pose where rotating the body
    #changes whether it collides.
    rng = np.random.default_rng(10)
    base = make_random_env(n_spheres=20, n_boxes=20, seed=3)
    env = SE3Env(base, RigidBody())
    found = False
    for _ in range(400):
        p = rng.uniform(base.lo + 0.2, base.hi - 0.2, size=3)
        cs = [float(env.clearance(p, rand_rotation(rng))) for _ in range(12)]
        if min(cs) < 0 < max(cs):
            found = True
            break
    return _report("orientation changes collision outcome at a fixed position",
                   found)


def test_planner_returns_a_free_path():
    rng = np.random.default_rng(11)
    base = make_random_env(n_spheres=10, n_boxes=10, seed=5)
    env = SE3Env(base, RigidBody())
    ok, n_try = False, 0
    for _ in range(3):
        n_try += 1
        s = env.sample_free_pose(rng)
        g = env.sample_free_pose(rng)
        if np.linalg.norm(g[0] - s[0]) < 1.0:
            continue
        pos, rot, info = plan_se3(env, s, g, n_waypoints=64, rng=rng, timeout=30.0)
        if pos is None:
            continue
        ok = (info["success"] and len(pos) == 64
              and np.allclose(pos[0], s[0], atol=1e-6)
              and np.allclose(pos[-1], g[0], atol=1e-6))
        break
    return _report("RRT-Connect returns a free 64-pose path hitting both ends", ok)


def main():
    print("SE(3) rigid-body domain: correctness tests\n")
    results = [
        test_rotation_roundtrip(),
        test_sixd_gram_schmidt_on_garbage(),
        test_slerp(),
        test_transform_states_matches_reference(),
        test_affine_on_rotation_is_detected(),
        test_reduction_reduces_se3_to_a_single_roll(),
        test_conditioning_dimensions(),
        test_reduce_unreduce_roundtrip(),
        test_roll_changes_frame_but_not_positions(),
        test_clearance_is_rigidly_invariant(),
        test_orientation_matters(),
        test_planner_returns_a_free_path(),
    ]
    n = sum(results)
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
