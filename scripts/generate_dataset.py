import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from pointmass3d import (
    BoxObstacle,
    SphereObstacle,
    chomp,
    make_random_env,
    resample_path,
    rrt_connect,
    sample_start_goal,
    shortcut,
    trajopt,
)

PRESETS = {
    "tiny": dict(n_envs=1, n_pairs=1, n_trajs=20),
    "small": dict(n_envs=20, n_pairs=25, n_trajs=20),
    "medium": dict(n_envs=200, n_pairs=125, n_trajs=20),
    "big": dict(n_envs=1000, n_pairs=250, n_trajs=20),
}

_REFINERS = {"chomp": chomp, "trajopt": trajopt, "none": None}


@dataclass
class EnvConfig:
    n_pairs: int
    n_trajs: int
    n_waypoints: int
    refine: str
    reverse_mode: str  # none | flip | replan
    n_spheres: int
    n_boxes: int
    seed: int
    dtype: str
    out: str
    overwrite: bool


def plan_trajectories(env, start, goal, n_trajs, rng, n_waypoints, refiner, counts):
    trajs = []
    attempts = 0
    while len(trajs) < n_trajs and attempts < 5 * n_trajs:
        attempts += 1
        raw, _ = rrt_connect(env, start, goal, rng=rng)
        if raw is None:
            counts["failed"] += 1
            continue
        path = resample_path(shortcut(env, raw, rng=rng), n_waypoints)
        if not env.path_free(path):
            counts["failed"] += 1
            continue

        if refiner is not None:
            refined, info = refiner(env, start, goal, n_waypoints=n_waypoints, init_path=path)
            if info["success"]:
                path = refined
                counts["ok"] += 1
            else:
                counts["fallback"] += 1  # keep the validated RRT path
        else:
            counts["ok"] += 1

        trajs.append(path)
    return trajs


def generate_env(task):
    e, cfg = task
    out = Path(cfg.out)
    dst = out / f"env_{e:04d}.npz"
    if dst.exists() and not cfg.overwrite:
        return {"env": e, "status": "skipped", "n_trajs": None,
                "ok": 0, "fallback": 0, "failed": 0, "pairs": None}

    env_seed = cfg.seed + e
    env = make_random_env(n_spheres=cfg.n_spheres, n_boxes=cfg.n_boxes, seed=env_seed)
    rng = np.random.default_rng(10_000 + env_seed)
    refiner = _REFINERS[cfg.refine]

    trajs, starts, goals, pair_ids, reversed_flags = [], [], [], [], []
    counts = {"ok": 0, "fallback": 0, "failed": 0}
    n_pairs_done, pair_attempts = 0, 0

    while n_pairs_done < cfg.n_pairs and pair_attempts < 5 * cfg.n_pairs:
        pair_attempts += 1
        try:
            start, goal = sample_start_goal(env, rng)
        except RuntimeError:
            break

        # Which (from, to) directions to plan for this pair.
        directions = [(start, goal, False)]
        if cfg.reverse_mode == "replan":
            directions.append((goal, start, True))  # independent samples, 2x cost

        for s, g, is_reversed in directions:
            planned = plan_trajectories(env, s, g, cfg.n_trajs, rng, cfg.n_waypoints, refiner, counts)
            for path in planned:
                trajs.append(path)
                starts.append(s)
                goals.append(g)
                pair_ids.append(n_pairs_done)
                reversed_flags.append(is_reversed)
                if cfg.reverse_mode == "flip":
                    trajs.append(path[::-1].copy())
                    starts.append(g)
                    goals.append(s)
                    pair_ids.append(n_pairs_done)
                    reversed_flags.append(True)

        n_pairs_done += 1

    spheres = np.array(
        [[*o.center, o.radius] for o in env.obstacles if isinstance(o, SphereObstacle)]
    )
    boxes = np.array(
        [[*o.center, *o.half_extents] for o in env.obstacles if isinstance(o, BoxObstacle)]
    )
    fdt = np.float32 if cfg.dtype == "float32" else np.float64
    tmp = out / f".env_{e:04d}.tmp.npz"
    np.savez(
        tmp,
        spheres=spheres.astype(fdt),
        boxes=boxes.astype(fdt),
        trajs=np.array(trajs, dtype=fdt),
        starts=np.array(starts, dtype=fdt),
        goals=np.array(goals, dtype=fdt),
        pair_ids=np.array(pair_ids),
        reversed=np.array(reversed_flags),
        env_seed=env_seed,
        robot_radius=env.robot_radius,
    )
    os.replace(tmp, dst)  # atomic on POSIX

    return {"env": e, "status": "ok", "n_trajs": len(trajs), "pairs": n_pairs_done, **counts}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default=None,
                    help="tiered size preset; individual --n-* flags override it")
    ap.add_argument("--n-envs", type=int, default=None, help="n distinct environments")
    ap.add_argument("--n-pairs", type=int, default=None, help="start-goal pairs per environment")
    ap.add_argument("--n-trajs", type=int, default=None, help="trajectories per pair (per direction)")
    ap.add_argument("--n-waypoints", type=int, default=64)
    ap.add_argument("--n-spheres", type=int, default=20)
    ap.add_argument("--n-boxes", type=int, default=20)
    ap.add_argument("--refine", choices=list(_REFINERS), default="chomp")
    ap.add_argument("--reverse-mode", choices=["none", "flip", "replan"], default="none",
                    help="none; flip=append reversed copy of each path (free ~2x, correlated); "
                         "replan=plan the swapped pair independently (2x cost)")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                    help="float32 halves disk/RAM; recommended for the big tier")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="parallel worker processes (default: all cores)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip environments whose shard already exists (default on)")
    ap.add_argument("--overwrite", dest="resume", action="store_false",
                    help="regenerate every environment even if a shard exists")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data")
    args = ap.parse_args()

    base = PRESETS[args.preset] if args.preset else dict(n_envs=10, n_pairs=5, n_trajs=4)
    n_envs = args.n_envs if args.n_envs is not None else base["n_envs"]
    n_pairs = args.n_pairs if args.n_pairs is not None else base["n_pairs"]
    n_trajs = args.n_trajs if args.n_trajs is not None else base["n_trajs"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = EnvConfig(
        n_pairs=n_pairs, n_trajs=n_trajs, n_waypoints=args.n_waypoints,
        refine=args.refine, reverse_mode=args.reverse_mode,
        n_spheres=args.n_spheres, n_boxes=args.n_boxes,
        seed=args.seed, dtype=args.dtype, out=str(out), overwrite=not args.resume,
    )

    all_idx = list(range(n_envs))
    todo = [e for e in all_idx if cfg.overwrite or not (out / f"env_{e:04d}.npz").exists()]
    n_done_already = n_envs - len(todo)

    mult = 2 if args.reverse_mode in ("flip", "replan") else 1
    target = n_envs * n_pairs * n_trajs * mult
    workers = max(1, min(args.workers, len(todo)) if todo else 1)
    print(f"preset={args.preset} envs={n_envs} pairs={n_pairs} trajs/pair={n_trajs} "
          f"reverse={args.reverse_mode} -> target ~{target:,} paths")
    print(f"out={out}/  workers={workers}  dtype={args.dtype}  "
          f"resume={'on' if not cfg.overwrite else 'off'}  "
          f"({n_done_already} shards already present, {len(todo)} to build)")

    t0 = time.time()
    totals = {"ok": 0, "fallback": 0, "failed": 0, "trajs": 0}
    tasks = [(e, cfg) for e in todo]

    def account(r):
        if r["status"] == "skipped":
            return
        totals["ok"] += r["ok"]
        totals["fallback"] += r["fallback"]
        totals["failed"] += r["failed"]
        totals["trajs"] += r["n_trajs"]

    done = 0
    if workers == 1:
        for task in tasks:
            r = generate_env(task)
            account(r)
            done += 1
            _progress(r, done, len(todo), t0)
    else:
        with Pool(workers) as pool:
            for r in pool.imap_unordered(generate_env, tasks):
                account(r)
                done += 1
                _progress(r, done, len(todo), t0)

    meta = {
        "preset": args.preset,
        "n_envs": n_envs,
        "n_pairs_per_env": n_pairs,
        "n_trajs_per_pair": n_trajs,
        "reverse_mode": args.reverse_mode,
        "n_waypoints": args.n_waypoints,
        "refine": args.refine,
        "dtype": args.dtype,
        "n_spheres": args.n_spheres,
        "n_boxes": args.n_boxes,
        "seed": args.seed,
        "robot_radius": 0.03,
        "workspace": [-1.0, 1.0],
        "target_paths": target,
        "paths_this_run": totals["trajs"],
        "refined_ok": totals["ok"],
        "rrt_fallback": totals["fallback"],
        "failed": totals["failed"],
        "workers": workers,
        "wall_time_s": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    dt = meta["wall_time_s"]
    rate = totals["trajs"] / dt if dt > 0 else 0.0
    print(f"\ndone in {dt}s ({rate:.1f} paths/s across {workers} workers) — "
          f"{totals['trajs']:,} paths this run; refined {totals['ok']:,}, "
          f"rrt-fallback {totals['fallback']:,}, failed attempts {totals['failed']:,}")
    print(f"dataset written to {out}/  (meta.json updated)")


def _progress(r, done, total, t0):
    if r["status"] == "skipped":
        return
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done)
    print(f"[{done}/{total}] env {r['env']:04d}: {r['n_trajs']} paths "
          f"({r['pairs']} pairs)  elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m",
          flush=True)


if __name__ == "__main__":
    main()
