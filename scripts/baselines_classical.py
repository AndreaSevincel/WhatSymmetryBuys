
  #.venv/bin/python baselines_classical.py --data data --env-start 250 --n-envs 50 \
  #    --n-pairs 10 --workers 12 --out baselines_classical.json

  #Classical-planner reference points for the learned planner, measured on the
  #SAME problems the learned models are evaluated on: the pair selection comes
  #from distinct_pairs() in sample_flow.py, so row-for-row these are the queries
  #of sweep_steps.py. Without this table a reader cannot tell whether 45.6%
  #collision-free is good, and cannot see what the learned planner buys, which
  #is wall-clock time.

  #Arms:
  #  straight-line   trivial lower bound: the segment from start to goal.
  #  RRT-Connect     +shortcut+resample to N waypoints (feasibility, jagged).
  #  CHOMP           local optimiser from the straight-line initialisation.
  #  TrajOpt         sequential convex optimisation, same initialisation.
  #  expert          the dataset pipeline: RRT -> shortcut -> resample -> CHOMP.
  #                  This is the supervision the learned models are trained on,
  #                  so it is their ceiling, not a competitor.

  #Every arm is single-threaded; --workers parallelises over problems, and the
  #reported seconds are per-problem single-core times, so the numbers are
  #comparable to the GPU sampler's per-query time only as orders of magnitude.

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

from pointmass3d import (
    chomp,
    mean_sq_accel,
    min_clearance,
    path_length,
    resample_path,
    rrt_connect,
    shortcut,
    trajopt,
)
from sample_flow import build_env, distinct_pairs

ARMS = ["straight-line", "RRT-Connect", "CHOMP", "TrajOpt", "expert (RRT+CHOMP)"]


def _record(env, path, secs, ok=None):
    if path is None:
        return dict(success=False, clearance=float("nan"), length=float("nan"),
                    sq_accel=float("nan"), secs=secs)
    return dict(
        success=bool(env.path_free(path)) if ok is None else bool(ok),
        clearance=min_clearance(env, path),
        length=path_length(path),
        sq_accel=mean_sq_accel(path),
        secs=secs,
    )


def solve_one(job):
    """One problem, every arm. Returns {arm: metrics}. Runs in a worker."""
    env_idx, pair_idx, data_dir, n_waypoints, seed = job
    npz = np.load(f"{data_dir}/env_{env_idx:04d}.npz", allow_pickle=True)
    env = build_env(npz, float(npz["robot_radius"]))
    start, goal = npz["starts"][pair_idx], npz["goals"][pair_idx]
    rng = np.random.default_rng(seed)
    out = {}

    # -- straight line ------------------------------------------------------
    t0 = time.perf_counter()
    line = np.linspace(start, goal, n_waypoints)
    out["straight-line"] = _record(env, line, time.perf_counter() - t0)

    # -- RRT-Connect + shortcut + resample ----------------------------------
    t0 = time.perf_counter()
    raw, _ = rrt_connect(env, start, goal, rng=rng)
    rrt_path = resample_path(shortcut(env, raw, rng=rng), n_waypoints) if raw is not None else None
    out["RRT-Connect"] = _record(env, rrt_path, time.perf_counter() - t0)

    # -- CHOMP / TrajOpt from the straight line -----------------------------
    #Deliberately NOT initialised from the RRT path: that would be the expert
    #pipeline, which is the arm below. The interesting comparison for a learned
    #planner is the local optimiser given no global information.
    for label, planner in (("CHOMP", chomp), ("TrajOpt", trajopt)):
        t0 = time.perf_counter()
        path, info = planner(env, start, goal, n_waypoints=n_waypoints)
        out[label] = _record(env, path, time.perf_counter() - t0)

    # -- the expert pipeline (upper bound / supervision) --------------------
    t0 = time.perf_counter()
    if rrt_path is not None:
        path, info = chomp(env, start, goal, n_waypoints=n_waypoints, init_path=rrt_path)
        #generate_dataset.py falls back to the raw RRT path if refinement collides
        if not env.path_free(path):
            path = rrt_path
    else:
        path = None
    out["expert (RRT+CHOMP)"] = _record(env, path, time.perf_counter() - t0)

    return (env_idx, pair_idx), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--env-start", type=int, default=250)
    ap.add_argument("--n-envs", type=int, default=50)
    ap.add_argument("--n-pairs", type=int, default=10)
    ap.add_argument("--n-waypoints", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="baselines_classical.json")
    args = ap.parse_args()

    jobs = []
    for ei in range(args.env_start, args.env_start + args.n_envs):
        npz = np.load(f"{args.data}/env_{ei:04d}.npz", allow_pickle=True)
        for k, pi in enumerate(distinct_pairs(npz["starts"], npz["goals"], args.n_pairs)):
            jobs.append((ei, pi, args.data, args.n_waypoints,
                         args.seed * 100003 + len(jobs)))
    print(f"{len(jobs)} problems (envs [{args.env_start}, "
          f"{args.env_start + args.n_envs}) x {args.n_pairs} pairs), "
          f"{args.workers} workers")

    t0 = time.time()
    with Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(solve_one, jobs, chunksize=1)):
            results.append(r)
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(jobs)}  {el:.0f}s elapsed, "
                      f"~{el/(i+1)*(len(jobs)-i-1):.0f}s left", flush=True)
    print(f"done in {time.time() - t0:.0f}s")

    header = (f"\n{'planner':<22}{'success%':>9}{'clearance':>11}{'length':>9}"
              f"{'sq_accel':>10}{'s/query':>9}")
    print(header)
    print("-" * len(header.strip()))
    summary = {}
    for arm in ARMS:
        rows = [r[arm] for _, r in results]
        succ = np.array([r["success"] for r in rows], dtype=float)
        #quality metrics are averaged over solved problems only; averaging a
        #failed run's clearance with a solved one is meaningless
        ok = succ > 0
        def m(key):
            v = np.array([r[key] for r in rows], dtype=float)[ok]
            return float(np.nanmean(v)) if v.size else float("nan")
        summary[arm] = dict(
            success=100 * float(succ.mean()),
            clearance=m("clearance"), length=m("length"),
            sq_accel=m("sq_accel"),
            secs=float(np.mean([r["secs"] for r in rows])),
            n=len(rows),
        )
        s = summary[arm]
        print(f"{arm:<22}{s['success']:>8.1f}%{s['clearance']:>11.4f}"
              f"{s['length']:>9.3f}{s['sq_accel']:>10.2e}{s['secs']:>9.3f}")

    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), summary=summary,
                       per_problem=[{"env": k[0], "pair": k[1], **{a: v[a] for a in ARMS}}
                                    for k, v in results]), f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
