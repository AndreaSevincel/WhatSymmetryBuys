
  #.venv/bin/python baselines_prior.py --data data --env-start 250 --n-envs 50 \
  #    --n-pairs 10 --n-samples 20

  #The untrained stochastic baseline, and the one a sampler must actually beat.

  #A generative planner is deployed as best-of-N behind a collision check, so
  #its per-query success is the fraction of problems where ANY of N samples is
  #free -- a number that a deterministic straight line cannot be compared
  #against, because it has only one sample. The honest control is the model's
  #own prior with the learning removed: Brownian bridges pinned at start and
  #goal. Same N, same collision check, no training.

  #sigma is swept and the BEST value is reported, so the baseline is steel-manned
  #rather than strawmanned: a learned model should beat the best untrained prior,
  #not a badly-tuned one.

import argparse
import json

import numpy as np

from pointmass3d import brownian, min_clearance, path_length
from sample_flow import build_env, distinct_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--env-start", type=int, default=250)
    ap.add_argument("--n-envs", type=int, default=50)
    ap.add_argument("--n-pairs", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--n-waypoints", type=int, default=64)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="baselines_prior.json")
    args = ap.parse_args()

    problems = []
    for ei in range(args.env_start, args.env_start + args.n_envs):
        npz = np.load(f"{args.data}/env_{ei:04d}.npz", allow_pickle=True)
        env = build_env(npz, float(npz["robot_radius"]))
        for pi in distinct_pairs(npz["starts"], npz["goals"], args.n_pairs):
            problems.append((env, npz["starts"][pi], npz["goals"][pi]))
    print(f"{len(problems)} problems x {args.n_samples} samples")

    header = (f"\n{'sigma':>7} {'free% (per-sample)':>20} {'any% (best-of-N)':>18}"
              f" {'clearance':>10} {'length':>8}")
    print(header)
    print("-" * (len(header) - 1))

    rows = []
    for sigma in args.sigmas:
        rng = np.random.default_rng(args.seed)
        free, any_free, clear, length = [], [], [], []
        for env, s, g in problems:
            #sigma=0 degenerates to N copies of the straight line; kept in the
            #sweep so the deterministic bound appears in the same table
            paths = brownian(s, g, n_waypoints=args.n_waypoints,
                             sigma_prior=sigma, n_samples=args.n_samples, rng=rng)
            f = [env.path_free(p) for p in paths]
            free.extend(f)
            any_free.append(any(f))
            clear.extend(min_clearance(env, p) for p in paths)
            length.extend(path_length(p) for p in paths)
        row = dict(sigma=sigma, free=100 * float(np.mean(free)),
                   any=100 * float(np.mean(any_free)),
                   clearance=float(np.mean(clear)), length=float(np.mean(length)))
        rows.append(row)
        print(f"{sigma:>7.2f} {row['free']:>19.1f}% {row['any']:>17.1f}%"
              f" {row['clearance']:>10.4f} {row['length']:>8.3f}")

    best_s = max(rows, key=lambda r: r["free"])
    best_a = max(rows, key=lambda r: r["any"])
    print(f"\nbest per-sample: sigma={best_s['sigma']:.2f} at {best_s['free']:.1f}%")
    print(f"best best-of-{args.n_samples}: sigma={best_a['sigma']:.2f} "
          f"at {best_a['any']:.1f}%")
    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), rows=rows), f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
