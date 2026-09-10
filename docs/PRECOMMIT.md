# Pre-registered analysis: the augmented control

Written **before** the augmented-control runs exist, so the framing is chosen on
the merits rather than under deadline pressure once the numbers are in. If the
paper's story changes after seeing the result, it should change to one of the
three below and not to something invented afterwards.

## The experiment

A world-frame arm trained with random `T in SE(3)` applied to the whole problem
each batch (`--augment 0.25`), at **all four tiers** (20 / 60 / 150 / 250
environments), so it can be plotted on the same scaling axes as Fig. 2 rather
than compared at a single point. Everything else identical: same data, same
2.16M-parameter network, same 20 epochs, same held-out 500 problems, same
per-sample collision-free metric.

This is the experiment that decides the paper. The current comparison shows that
*ignoring* the symmetry leaves the model at the straight-line floor; it does not
show that canonicalisation beats the other standard way of *using* the symmetry.

**Decision threshold.** Treat gaps below 2x the across-seed SE at 60 envs (from
the three-seed run) as "near". Do not eyeball it.

## Outcome A — augmentation lands well below the reduction

*The paper as currently written, with the strawman objection closed.*

- Keep the title, the abstract, and the mechanism decomposition as they stand.
- Add the augmented arm as a third curve in Fig. 2 and a row in Table 1.
- The sentence in Sec. V-B that currently scopes the claim ("does not establish
  that the reduction beats augmentation") is deleted and replaced by the
  measured gap.
- Claim: exact canonicalisation beats approximate symmetry, at zero cost.

## Outcome B — augmentation lands near the reduction

*Story changes; it does not weaken. Arguably becomes more interesting.*

- New framing: **canonicalisation obtains for free, and exactly, what
  augmentation pays for in capacity and data.** The reduction is not the only
  way to get the benefit; it is the cheap and certain way.
- The headline becomes the two scaling curves side by side. **If augmentation
  needs more environments to reach the same collision-free rate, that horizontal
  gap is the result** -- quote it as "augmentation needs Nx the environments to
  match", which is a data-efficiency claim and a stronger one than a level
  difference.
- Also report training cost: augmentation is applied per batch and costs wall
  time; the reduction is a change of coordinates and costs nothing.
- Retitle to something like "Canonicalisation as Free Augmentation".
- Sec. IV's obstruction propositions and the residual diagnostic are unaffected.

## Outcome C — augmentation beats the reduction

*A real finding, and the diagnostic survives intact.*

- Report it plainly and prominently. It is the more surprising result and it is
  worth more than a confirmation would have been.
- Retitle around the diagnostic, which does not depend on which arm wins: the
  residual, Prop. 4 and Cor. 5 are statements about a trained field, whichever
  training produced it.
- The reduction keeps its two defensible claims -- exactness, and zero cost --
  and loses only the claim to be the best use of the symmetry.
- Likely explanation to investigate before writing: augmentation exposes the
  model to poses the reduction never shows it, which matters if the held-out
  problems are not distributed like the training frame. Check the residual on
  the augmented arm; if it is much smaller, that is the mechanism.

**All three are publishable. None of them are publishable if the run does not
exist.**

## Added 2026-08-07: the epoch-matched control

`long-e60` (reduced arm, 60 envs, 60 epochs) scores **42.4%** against 31.3% for
the same arm at 20 epochs -- +11.1 pp from optimisation alone, at identical data
and capacity. It also lands close to treat-e250 at 20 epochs (45.6%), so epochs
and environments are substantially substitutable here.

That is good news for the level and a threat to the headline figure: the scaling
curves compare two arms at a budget where **both** are underfit, so a referee can
ask whether the 30.2 pp gap partly reflects convergence *rate* rather than data
efficiency. **One run answers it: the world-frame arm at 60 envs for 60 epochs.**

- ctrl-e60 at 60 epochs stays near the straight-line floor -> the headline is
  robust to budget, and the paper gets stronger (the control cannot be rescued
  by optimisation).
- ctrl-e60 at 60 epochs improves a lot -> the gap is partly a convergence-rate
  artefact, and Fig. 2 must be replotted at a budget where both arms have
  converged, or relabelled as a fixed-budget comparison.

Priority: this sits *behind* the augmented control (which decides the paper) but
*ahead* of the three seeds, because it can invalidate the headline figure rather
than merely widen its error bars.

### The framing this run tests: two bottlenecks, not one

Registered before `long-ctrl-e60` finishes.

The claim is not "canonicalisation improves performance". It is that this model
faces **two distinct bottlenecks**, and that each mechanism relieves exactly one:

- a **representation** bottleneck -- the network is shown the pose and must
  relearn the same skill at every one. Canonicalisation removes it exactly and
  for free. More compute does *not* remove it.
- a **capacity / optimisation** bottleneck -- the field is underfit. More epochs
  or parameters remove it. Canonicalisation does *not* remove it.

Stated that way it is an **interaction**, not two main effects: the return on
optimisation should be *conditional* on the representation being fixed first.
The 2x2 at 60 environments:

| | 20 epochs | 60 epochs |
|---|---|---|
| world frame | 13.5% | **the pending cell** |
| (s,g) reduction | 31.3% | 42.4% |

**Prediction: the pending cell stays near the straight-line floor (13-17%).**

- Confirmed (world@60 stays low) -> the paper's arc becomes: two bottlenecks,
  canonicalisation fixes one exactly and free, the residual r tells you when
  that one is exhausted, and only then does compute pay. Report the 2x2 and the
  interaction directly. Sec. V-E stops being a caveat that threatens Fig. 2 and
  becomes the second half of the result.
- Refuted (world@60 reaches ~30%) -> the bottlenecks are not separable, the
  +11.1 pp is a generic budget effect, and Fig. 2 must be relabelled as a
  fixed-budget comparison. The reduction then buys convergence *speed*, which is
  a weaker but still reportable claim.

This also unifies the diagnostic with the framing, which the paper currently
presents as two separate contributions. **r is a test of which bottleneck is
binding**: small r means the symmetry part of the representation problem is
exhausted, so the remaining error is equivariant error -- capacity, not
representation. That is why symmetrising buys nothing at r = 0.016 while three
times the epochs buys 11 points. One measurement, two bottlenecks, and it says
which one to spend on next.

```bash
python train_flow.py --data data --n-envs 60 --epochs 60 \
    --batch 512 --amp --multi-gpu --out checkpoints/long-ctrl-e60.pt
```

## Commands

```bash
# the deciding experiment: augmented control at all four tiers
for n in 20 60 150 250; do
  python train_flow.py --data data --n-envs $n --augment 0.25 \
      --epochs 20 --batch 512 --amp --multi-gpu \
      --out checkpoints/grid/ctrl-aug-e$n.pt
done

# the noise floor every "inside noise" claim depends on: 3 seeds, both arms, 60 envs
for s in 0 1 2; do for arm in "" "--reduced"; do
  tag=$([ -z "$arm" ] && echo ctrl || echo treat)
  python train_flow.py --data data --n-envs 60 $arm --seed $s \
      --epochs 20 --batch 512 --amp --multi-gpu \
      --out checkpoints/grid/$tag-e60-s$s.pt
done; done

# eval everything: per-sample, per-query, path length, and r under the RMS definition
mkdir -p results
for f in checkpoints/grid/*.pt; do
  python sweep_steps.py --ckpt "$f" --data data --env-start 250 --n-envs 50 \
      --n-pairs 10 --n-samples 20 --steps 8 --k-fa 3 --residual \
      --out-json "results/$(basename "$f" .pt).json"
done
python aggregate_runs.py results/*.json --metric free        --steps 8
python aggregate_runs.py results/*.json --metric solved_any  --steps 8
python aggregate_runs.py results/*.json --metric length      --steps 8
python aggregate_runs.py results/*.json --metric residual    --steps 8
```

`--k-fa 3 --residual` is required for the residual: it is identically zero at
K=1, so earlier runs cannot be reused for it.

## Not attempted before the deadline

A manipulator configuration space, a diffusion-objective replication, and an
SO(2)-equivariant backbone. The code for the second exists and is unit-tested
(`flowmatch/diffusion.py`, `test_diffusion.py`); the SE(3) rigid-body domain is
built and tested (`se3body/`) but has no dataset. All three are rebuttal-phase
promises, not deliverables.
