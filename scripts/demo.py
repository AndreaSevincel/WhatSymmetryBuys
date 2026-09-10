
#python demo.py [--seed 0] [--n-waypoints 64] [--show]


import argparse

import numpy as np

from pointmass3d import (
    chomp,
    make_random_env,
    mean_sq_accel,
    min_clearance,
    path_length,
    plot_scene,
    resample_path,
    rrt_connect,
    sample_start_goal,
    shortcut,
    trajopt,
    brownian
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-waypoints", type=int, default=64)
    ap.add_argument("--sigma-prior", type=float, default=0.15,
                     help="Brownian bridge prior noise scale")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = make_random_env(n_spheres=20, n_boxes=20, seed=args.seed)
    start, goal = sample_start_goal(env, rng)
    N = args.n_waypoints
    print(f"seed={args.seed}  start={np.round(start, 3)}  goal={np.round(goal, 3)}")

    results = {}  # label -> (path, info)

    # --- RRT-Connect (+ shortcut + resample to N waypoints)
    raw, info = rrt_connect(env, start, goal, rng=rng)
    if raw is None:
        raise SystemExit("RRT-Connect failed — try another seed")
    rrt_path = resample_path(shortcut(env, raw, rng=rng), N)
    info["success"] = env.path_free(rrt_path)
    results["RRT-Connect"] = (rrt_path, info)

    #CHOMP and TrajOpt, initialized from the straight line; if that
    #local optimization fails, retry from the RRT-Connect path.
    for label, planner in [("CHOMP", chomp), ("TrajOpt", trajopt)]:
        path, pinfo = planner(env, start, goal, n_waypoints=N)
        if not pinfo["success"]:
            path, pinfo = planner(env, start, goal, n_waypoints=N, init_path=rrt_path)
            pinfo["init"] = "rrt"
        else:
            pinfo["init"] = "straight-line"
        results[label] = (path, pinfo)

    #Brownian bridge prior sample
    brownian_path = brownian(start, goal, n_waypoints=N, sigma_prior=args.sigma_prior, rng=rng)

    # --- report
    header = f"{'planner':<14}{'success':<9}{'time [s]':<10}{'length':<9}{'msq accel':<12}{'min clear':<10}"
    print("\n" + header + "\n" + "-" * len(header))
    for label, (path, pinfo) in results.items():
        print(
            f"{label:<14}{str(pinfo['success']):<9}{pinfo['time']:<10.3f}"
            f"{path_length(path):<9.3f}{mean_sq_accel(path):<12.2e}"
            f"{min_clearance(env, path):<10.4f}"
            + (f"  (init: {pinfo['init']})" if "init" in pinfo else "")
        )

    trajs = {label: path for label, (path, _) in results.items()}
    trajs["Brownian prior"] = brownian_path
    plot_scene(env, trajs, start, goal,
               title=f"PointMass3D — seed {args.seed}",
               save="demo.png", show=args.show)
    print("\nsaved figure to demo.png")


if __name__ == "__main__":
    main()
