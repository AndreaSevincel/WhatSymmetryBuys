# What Symmetry Buys a Learned Motion Planner

Code, data and results for the paper. A start and a goal determine a frame in
closed form. Expressing the trajectory and the obstacles in that frame removes
five of the six degrees of freedom of SE(3) at initialisation, for one cross
product per query and with no constraint on the architecture. This repository
is the benchmark that measures what that is worth, the flow-matching planner it
is measured on, and every number the paper reports.

The headline: on 500 held-out problems, holding architecture, data and budget
fixed, the frame raises the collision-free rate from **14.60%** to **51.10%**,
where a straight segment from start to goal scores 15.6% and the world-frame
model does not beat it.

- Paper: [arXiv:2609.10033](https://arxiv.org/abs/2609.10033) — PDF and source also in [`paper/`](paper/)
- Videos: [`media/`](media/)

## Layout

| path | what it holds |
|---|---|
| `paper/` | the submission, its figures, and the class files needed to build it |
| `flowmatch/` | flow-matching planner: model, sampler, the frame reduction, SDF features |
| `pointmass3d/` | the benchmark environment and the classical planners (RRT-Connect, CHOMP, TrajOpt) |
| `se3body/` | the second domain, a rigid body whose state is a full pose |
| `scripts/` | dataset generation, training, evaluation, baselines, figures |
| `results/` | every measured result the paper cites, one JSON per evaluated cell |
| `tests/` | property-based tests for the geometry and equivariance claims |
| `docs/` | the pre-registered analysis, written before the deciding runs existed |
| `media/` | paper and supplementary videos |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .            # benchmark + classical planners
.venv/bin/pip install -e '.[train]'   # adds torch and wandb for the planner
```

The editable install puts `flowmatch`, `pointmass3d` and `se3body` on the path,
so the scripts run from any directory.

<details>
<summary>Pin the CUDA wheel to your driver, or training silently runs on CPU</summary>

A bare `pip install --upgrade torch` pulls whatever CUDA build is newest, and
if that runtime is newer than the installed driver, CUDA attempts *forward
compatibility*, which is supported only on data-centre GPUs and never on
GeForce. On a 4090 you get `Error 804: forward compatibility was attempted on
non supported HW`, `torch.cuda.is_available()` returns False, and training runs
on CPU without complaining. Check `nvidia-smi` and pick the wheel:

| driver | wheel |
|---|---|
| >= 580 | `cu130` |
| >= 570 | `cu128` |
| >= 560 | `cu126` |
| >= 550 | `cu124` |

```bash
pip install "torch==2.6.*" --index-url https://download.pytorch.org/whl/cu124
```

Experiment tracking is optional: `flowmatch/tracking.py` no-ops if wandb is
absent or unauthenticated, so training never dies on a logging problem. Run
`wandb login` to enable it, or pass `--wandb-offline` on air-gapped nodes and
`wandb sync` the run directories later.

</details>

## Quick start

```bash
# one problem, all three classical planners, comparison table + demo.png
.venv/bin/python scripts/demo.py --seed 0

# expert-trajectory dataset (RRT-Connect -> shortcut -> CHOMP refinement)
.venv/bin/python scripts/generate_dataset.py --n-envs 10 --n-trajs 20 --refine chomp

# train the two arms: world frame, and the (s,g) reduction
.venv/bin/python scripts/train_flow.py --data data --n-envs 250 --epochs 20 \
    --out checkpoints/ctrl.pt
.venv/bin/python scripts/train_flow.py --data data --n-envs 250 --epochs 20 \
    --reduced --out checkpoints/treat.pt

# evaluate on the held-out 500 problems (envs 250-299)
.venv/bin/python scripts/sweep_steps.py --ckpt checkpoints/treat.pt --data data \
    --env-start 250 --n-envs 50 --n-pairs 10 --n-samples 20 --steps 8
```

`--reduced` is the whole intervention. Everything else is held fixed between
the two arms.

## Reproducing the paper

Every number in the paper comes from a JSON file in `results/`, produced by
`scripts/sweep_steps.py`. The figures are regenerated from those numbers:

```bash
.venv/bin/python scripts/make_figures.py    # writes into paper/
cd paper && pdflatex ICRA.tex               # IEEEtran.cls ships here
```

The dataset (300 environments, ~17 GB) and the trained checkpoints are not in
git. Both regenerate deterministically from the per-environment seeds via
`scripts/generate_dataset.py`.

## The benchmark

`PointMass3DEnv` (`pointmass3d/env.py`): a spherical robot of radius 0.03 in
`[-1, 1]^3` with sphere and oriented-box obstacles. Collision checking goes
through an analytic signed distance field; each obstacle implements `sdf`, the
environment takes the min over obstacles and the workspace walls, and
`clearance(q) = sdf(q) - robot_radius` is positive iff `q` is free. Paths are
validated by dense resampling at spacing 0.01, not at the waypoints alone.

The benchmark was chosen for isolability, not difficulty: RRT-Connect solves
99.6% of these problems in under 0.3 s on one CPU core. A point mass in a box
is where clutter, budget, seed and representation can be varied one at a time,
which is the only reason any number here is attributable to the representation.

## Tests

```bash
.venv/bin/python -m pytest
```

The geometry and equivariance tests run on **untrained** networks, so they
check structural properties rather than learned ones, and include negative
controls: a test that cannot fail is worse than no test.

## Citation

```bibtex
@misc{sevincel2026symmetry,
  title         = {What Symmetry Buys a Learned Motion Planner},
  author        = {Sevincel, Andrea Emir},
  year          = {2026},
  eprint        = {2609.10033},
  archivePrefix = {arXiv}
}
```
