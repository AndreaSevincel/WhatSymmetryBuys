
  #.venv/bin/python run_grid.py --data data --dry-run
  #.venv/bin/python run_grid.py --data data --epochs 60 --batch 1024 --amp --multi-gpu

  #The 2x4 experiment grid: {control, treatment} x {20, 60, 150, 300} envs.

  #control   -- world frame, conditioned on raw (start, goal), sg_dim=6
  #treatment -- (s,g)-reduced frame + uniform roll augmentation, sg_dim=1
  #Both arms get the 12-dim box representation so it is not a confound. The
  #cond_enc width cannot be equalized -- the control genuinely needs the raw
  #pair -- but that asymmetry is intrinsic to the treatment, not incidental.

  #The env axis separates "the reduction helped" from "180x more data helped",
  #and the interaction term answers whether exact equivariance is structural or
  #a small-data crutch. Cost is asymmetric: only the two 300-env cells are
  #expensive, the rest are nested subsets.

  #NOTE: val loss is NOT comparable across arms. The treatment regresses targets
  #in the reduced frame, where trajectories are canonicalized along +x, so its
  #target variance -- and therefore its MSE -- is mechanically smaller. Score
  #the grid on world-frame metrics (collision-free rate, clearance, length) via
  #sample_flow.py / sweep_steps.py. best_val_loss is only meaningful within arm.

  #Every cell trains on envs [0, n_envs) and the TAIL of the shard range is
  #reserved: no cell ever sees it, so all cells can be scored on identical
  #never-trained layouts. Evaluate with
  #   sweep_steps.py --env-start <holdout_start> --n-envs <holdout>
  #Within the training envs, val holds out whole (start, goal) pairs
  #(--split-by pair): a per-trajectory split leaks badly here, because the 30
  #paths per pair are near-duplicates (within-pair std 0.054 vs overall 0.426).

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

DEFAULT_ENVS = [20, 60, 150, 250]
DEFAULT_HOLDOUT = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--out-dir", type=str, default="checkpoints/grid")
    ap.add_argument("--n-envs", type=int, nargs="+", default=DEFAULT_ENVS)
    ap.add_argument("--holdout", type=int, default=DEFAULT_HOLDOUT,
                    help="env shards reserved as a never-trained test set")
    ap.add_argument("--total-envs", type=int, default=None,
                    help="shards available (default: count them in --data)")
    ap.add_argument("--split-by", choices=["traj", "pair", "env"], default="pair",
                    help="what val holds out within the training envs")
    ap.add_argument("--arms", type=str, nargs="+", default=["control", "treatment"],
                    choices=["control", "treatment"])
    ap.add_argument("--group", type=str, default="stage3-grid",
                    help="wandb group tying the cells together")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="training seeds; the grid is run once per seed and the "
                         "checkpoint name is suffixed -sN for N>0. Reported "
                         "error bars should be across THESE, not across "
                         "held-out problems -- the two measure different things")
    ap.add_argument("--objectives", type=str, nargs="+", default=["flow"],
                    choices=["flow", "ddpm"],
                    help="generative objective per cell; ddpm is the "
                         "Diffuser/MPD-style baseline sharing the backbone")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--multi-gpu", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-offline", action="store_true")
    ap.add_argument("--check-frame", action="store_true",
                    help="run the reduction assertions every batch (slow; use once)")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = args.total_envs
    if total is None:
        total = len(list(Path(args.data).glob("env_*.npz")))
    if total == 0:
        sys.exit(f"no env_*.npz shards found in {args.data}")
    train_cap = total - args.holdout
    over = [n for n in args.n_envs if n > train_cap]
    if over:
        sys.exit(
            f"{over} exceed the {train_cap} trainable envs "
            f"({total} shards - {args.holdout} held out). Lower --n-envs or "
            f"--holdout; training into the test range would void the comparison."
        )

    #Cheap cells first: if something is wrong with the pipeline, it surfaces on
    #a 20-env run in minutes rather than after hours on the largest cell.
    cells = sorted(
        itertools.product(args.arms, args.n_envs, args.seeds, args.objectives),
        key=lambda c: (c[1], c[0], c[2], c[3]),
    )
    print(f"{len(cells)} cells: {args.arms} x {args.n_envs} x seeds {args.seeds}"
          f" x {args.objectives}  group={args.group}")
    print(f"{total} shards: train from [0, {train_cap}), "
          f"held-out test [{train_cap}, {total})  split_by={args.split_by}\n")

    failures = []
    for i, (arm, n_envs, seed, objective) in enumerate(cells, 1):
        tag = f"{'treat' if arm == 'treatment' else 'ctrl'}-e{n_envs}"
        #seed 0 keeps its historic name so existing checkpoints and the
        #numbers already reported against them stay addressable
        if objective != "flow":
            tag = f"{objective}-{tag}"
        if seed != 0:
            tag = f"{tag}-s{seed}"
        cmd = [
            sys.executable, "train_flow.py",
            "--data", args.data,
            "--n-envs", str(n_envs),
            "--out", str(out_dir / f"{tag}.pt"),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--channels", str(args.channels),
            "--n-blocks", str(args.n_blocks),
            "--seed", str(seed),
            "--num-workers", str(args.num_workers),
            "--split-by", args.split_by,
            "--env-start", "0",
            "--objective", objective,
        ]
        if arm == "treatment":
            cmd.append("--reduced")
        if args.amp:
            cmd.append("--amp")
        if args.multi_gpu:
            cmd.append("--multi-gpu")
        if args.check_frame:
            cmd.append("--check-frame")
        if not args.no_wandb:
            cmd += ["--wandb", "--wandb-group", args.group,
                    "--wandb-name", tag,
                    "--wandb-tags",
                    f"arm-{arm},envs-{n_envs},seed-{seed},obj-{objective}"]
            if args.wandb_offline:
                cmd.append("--wandb-offline")

        print(f"[{i}/{len(cells)}] {tag}")
        if args.dry_run:
            print("   ", " ".join(cmd), "\n")
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            failures.append(tag)
            print(f"    FAILED (exit {r.returncode})")
            if not args.continue_on_error:
                sys.exit(r.returncode)

    if failures:
        print(f"\n{len(failures)} cell(s) failed: {', '.join(failures)}")
        sys.exit(1)
    if not args.dry_run:
        print(f"\nall {len(cells)} cells done -> {out_dir}")
        print("score them on world-frame metrics, NOT val loss (see header):")
        print(f"  python3 sweep_steps.py --ckpt {out_dir}/treat-e{max(args.n_envs)}.pt "
              f"--data {args.data} --env-start {train_cap} --n-envs {args.holdout} "
              f"--n-pairs 10 --n-samples 20")


if __name__ == "__main__":
    main()
