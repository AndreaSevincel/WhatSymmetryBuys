
  #OMP_NUM_THREADS=1 .venv/bin/python baselines_se3.py --data data_se3 \
  #    --env-start 250 --n-envs 50 --n-pairs 10 --workers 16

  #Floors and ceilings for the SE(3) rigid-body domain, on the SAME held-out
  #problems the learned arms are scored on.

  #The point-mass domain taught the lesson this exists for: a collision-free
  #rate is uninterpretable without a scale. There, the world-frame arm turned
  #out to be sitting exactly on the straight-line floor, and nothing in the
  #learned numbers revealed it. The SE(3) arms currently read 6.2% and 2.8%,
  #which could mean the reduction is 2.3x a hard floor or that both are noise
  #around a trivial baseline. Only this tells you which.

  #The trivial baseline here is the geodesic interpolation: straight in
  #position, slerp in rotation. It is the SE(3) analogue of joining start to
  #goal with a line, and it is what a planner must beat to have done anything.

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

from pointmass3d import mean_sq_accel, path_length
from sample_flow import build_env, distinct_pairs
from se3body import RigidBody, SE3Env, plan_se3, sixd_to_matrix
from se3body.rotation import interpolate

ARMS = ["geodesic interpolation", "RRT-Connect + shortcut"]


def solve_one(job):
    env_idx, pair_idx, data_dir, n_waypoints, seed, timeout = job
    z = np.load(f"{data_dir}/env_{env_idx:04d}.npz", allow_pickle=True)
    env = SE3Env(build_env(z, 0.0),
                 RigidBody(z["body_centers"], float(z["robot_radius"])))
    s, g = z["starts"][pair_idx], z["goals"][pair_idx]
    ps, Rs = s[:3], sixd_to_matrix(s[3:])
    pg, Rg = g[:3], sixd_to_matrix(g[3:])
    rng = np.random.default_rng(seed)
    out = {}

    #the trivial baseline
    t0 = time.perf_counter()
    pos, rot = interpolate(ps, Rs, pg, Rg, n_waypoints)
    out["geodesic interpolation"] = dict(
        success=bool(env.path_free(pos, rot)),
        clearance=env.min_clearance(pos, rot),
        length=path_length(pos), sq_accel=mean_sq_accel(pos),
        secs=time.perf_counter() - t0,
    )

    #the expert pipeline, which is also the supervision
    t0 = time.perf_counter()
    pos, rot, info = plan_se3(env, (ps, Rs), (pg, Rg),
                              n_waypoints=n_waypoints, rng=rng, timeout=timeout)
    if pos is None:
        out["RRT-Connect + shortcut"] = dict(
            success=False, clearance=float("nan"), length=float("nan"),
            sq_accel=float("nan"), secs=time.perf_counter() - t0)
    else:
        out["RRT-Connect + shortcut"] = dict(
            success=bool(info["success"]),
            clearance=env.min_clearance(pos, rot),
            length=path_length(pos), sq_accel=mean_sq_accel(pos),
            secs=time.perf_counter() - t0,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data_se3")
    ap.add_argument("--env-start", type=int, default=250)
    ap.add_argument("--n-envs", type=int, default=50)
    ap.add_argument("--n-pairs", type=int, default=10)
    ap.add_argument("--n-waypoints", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="baselines_se3.json")
    args = ap.parse_args()

    jobs = []
    for ei in range(args.env_start, args.env_start + args.n_envs):
        z = np.load(f"{args.data}/env_{ei:04d}.npz", allow_pickle=True)
        for pi in distinct_pairs(z["starts"], z["goals"], args.n_pairs):
            jobs.append((ei, pi, args.data, args.n_waypoints,
                         args.seed * 100003 + len(jobs), args.timeout))
    print(f"{len(jobs)} problems, {args.workers} workers")

    t0 = time.time()
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(solve_one, jobs, chunksize=1))
    print(f"done in {time.time() - t0:.0f}s")

    header = (f"\n{'arm':<26}{'success%':>9}{'clearance':>11}{'length':>9}{'s/query':>9}")
    print(header); print("-" * (len(header) - 1))
    summary = {}
    for arm in ARMS:
        rows = [r[arm] for r in results]
        succ = np.array([r["success"] for r in rows], dtype=float)
        ok = succ > 0
        def m(k):
            v = np.array([r[k] for r in rows], dtype=float)[ok]
            return float(np.nanmean(v)) if v.size else float("nan")
        summary[arm] = dict(success=100 * float(succ.mean()), clearance=m("clearance"),
                            length=m("length"),
                            secs=float(np.mean([r["secs"] for r in rows])),
                            n=len(rows))
        s_ = summary[arm]
        print(f"{arm:<26}{s_['success']:>8.1f}%{s_['clearance']:>11.4f}"
              f"{s_['length']:>9.3f}{s_['secs']:>9.3f}")

    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), summary=summary), f, indent=1)
    print(f"\nwrote {args.out}")
    print("\nRead the learned SE(3) arms against the first row: if they are not "
          "clear of it,\nthe domain is not yet showing anything, exactly as the "
          "world-frame arm was not\nin the point-mass domain.")


if __name__ == "__main__":
    main()
