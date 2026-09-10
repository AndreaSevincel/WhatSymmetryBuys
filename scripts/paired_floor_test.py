#.venv/bin/python paired_floor_test.py results/conv/ctrl-e60-conv-s0.ep0300.json [...]

#Is the world-frame arm actually below the straight-line floor?

#The paper first said "below", then hedged to "at or below" because the
#clustered bootstrap gives SE 2.2 on the straight-line rate while the gap is
#only 1.0. That hedge is too conservative, and the reason is worth stating: 2.2
#is the standard error of ONE rate resampled on its own. Both arms are evaluated
#on the identical 500 problems and both track how cluttered each environment is,
#so their outcomes are positively correlated across environments and the
#DIFFERENCE has a much smaller standard error than either rate does.
#
#This resamples environments once per replicate and recomputes the difference
#inside each replicate, which is the paired version of the same bootstrap.

import argparse
import collections
import json
import pathlib
import sys

import numpy as np


def load_straight_line(path="baselines_classical.json"):
    d = json.load(open(path))
    out = {}
    for rec in d["per_problem"]:
        out[(int(rec["env"]), int(rec["pair"]))] = bool(rec["straight-line"]["success"])
    return out


def load_model(path):
    d = json.load(open(path))
    rows = d["rows"] if "rows" in d else [d]
    for row in rows:
        if "per_problem" in row:
            return {(int(r["env"]), int(r["pair"])): (r["n_free"], r["n_samples"])
                    for r in row["per_problem"]}
    raise SystemExit(f"{path} has no per_problem block; re-score with "
                     f"--dump-per-problem")


def paired_bootstrap(keys, model, line, n_boot=10000, seed=0):
    #Cluster on ENVIRONMENT, not on problem: pairs within an environment share a
    #layout and are not independent. Resampling problems would understate the
    #spread for both arms and for their difference.
    by_env = collections.defaultdict(list)
    for k in keys:
        by_env[k[0]].append(k)
    envs = sorted(by_env)
    rng = np.random.default_rng(seed)

    def rates(sample_keys):
        m = np.mean([model[k][0] / model[k][1] for k in sample_keys])
        l = np.mean([1.0 if line[k] else 0.0 for k in sample_keys])
        return 100 * m, 100 * l

    m0, l0 = rates(keys)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(envs, size=len(envs), replace=True)
        sk = [k for e in drawn for k in by_env[e]]
        m, l = rates(sk)
        diffs[b] = m - l
    return m0, l0, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="one or more per-problem JSONs (seeds)")
    ap.add_argument("--baselines", default="baselines_classical.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    line = load_straight_line(args.baselines)
    models = [load_model(p) for p in args.results]
    keys = sorted(set(line) & set.intersection(*[set(m) for m in models]))
    if not keys:
        raise SystemExit("no (env, pair) keys in common -- was the model scored on "
                         "the same env range and pair count as the baselines?")
    print(f"{len(keys)} problems in common, {len(models)} seed(s)\n")

    #average the seeds per problem first, so the seed spread does not enter the
    #environment bootstrap as if it were problem-level noise
    merged = {k: (sum(m[k][0] for m in models), sum(m[k][1] for m in models))
              for k in keys}
    m0, l0, diffs = paired_bootstrap(keys, merged, line, args.n_boot)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_worse = float(np.mean(diffs >= 0))
    print(f"world frame        {m0:6.2f}%")
    print(f"straight line      {l0:6.2f}%")
    print(f"paired difference  {m0 - l0:+6.2f}%   95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"bootstrap SE of the difference   {diffs.std(ddof=1):.2f}")
    print(f"P(difference >= 0)               {p_worse:.4f}")
    print()
    if hi < 0:
        print('VERDICT: "below the straight line" is supported; the CI excludes zero.')
    elif lo > 0:
        print('VERDICT: the arm is ABOVE the line. Both current phrasings are wrong.')
    else:
        print('VERDICT: "at or below the floor" is the defensible phrasing; the CI '
              'includes zero.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
