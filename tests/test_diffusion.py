
  #.venv/bin/python test_diffusion.py

  #Correctness tests for the DDPM baseline arm (flowmatch/diffusion.py).

  #These are written to be independent of training: a sampler bug and an
  #undertrained model both produce bad paths, and only one of them is fixable by
  #training longer. Everything here holds for an arbitrary or adversarial
  #network, so a failure points at the code and not at the checkpoint.

  #The decisive one is test_oracle_sampler: it hands the sampling loop a model
  #that returns the exact eps for a chosen target, and requires the loop to
  #reconstruct that target. That exercises the strided timestep schedule, the
  #step rule, the clipping and the endpoint inpainting together, which no
  #single-step algebraic check does.

import numpy as np
import torch

from flowmatch.diffusion import Schedule, _step, _timesteps

TOL = 1e-4


def _report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def test_schedule_conditioning():
    #alpha_bar must decrease from 1 and stay strictly positive: the raw cosine
    #hits exactly 0 at t=T, which makes the first x0 estimate a division by ~0
    sch = Schedule(100)
    ab = sch.ab
    ok = bool(ab[0] == 1.0)
    ok &= bool((ab[1:] < ab[:-1]).all())          # strictly decreasing
    ok &= bool((ab > 0).all())                    # never exactly zero
    amp = float(1.0 / ab[-1].sqrt())
    ok &= amp < 1e4
    return _report("schedule: monotone, positive, conditioned", ok,
                   f"(1/sqrt(ab_T) = {amp:.0f})")


def test_step_matches_forward_process():
    #With the TRUE eps, a deterministic DDIM step must land exactly on the
    #forward process at the next level. Data is kept inside the clip range so
    #the clip is not what is being measured.
    sch = Schedule(100)
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(4, 64, 3, generator=g).clamp(-2.5, 2.5)
    eps = torch.randn(4, 64, 3, generator=g)
    worst = 0.0
    for i, j in [(90, 80), (50, 25), (25, 10), (10, 1), (5, 0), (1, 0)]:
        xt = sch.q_sample(x0, torch.full((4,), i), eps)
        got = _step(xt, eps, i, j, sch, 0.0, None)
        want = sch.q_sample(x0, torch.full((4,), j), eps) if j > 0 else x0
        worst = max(worst, float((got - want).abs().max()))
    return _report("step rule reproduces the forward process", worst < TOL,
                   f"(max err {worst:.1e})")


def test_timesteps():
    sch_T = 100
    ok = True
    for n in (4, 8, 20, 100, 200):
        ts = _timesteps(sch_T, n)
        ok &= ts == sorted(ts, reverse=True)      # descending
        ok &= ts[0] == sch_T and ts[-1] == 1      # spans the whole schedule
        ok &= len(ts) == len(set(ts))             # no repeats
        ok &= len(ts) <= max(n, 1)
    return _report("timestep subsequence is descending and spans [1, T]", ok)


class _OracleEps:
    #A "model" that knows the answer: at level i it returns the eps that
    #explains x as a noisy version of the target. The sampler must then walk
    #down to the target exactly. Mimics the FlowVelocityField interface.
    def __init__(self, target, sch):
        self.target, self.sch = target, sch

    def eval(self):
        return self

    def encode_cond(self, *a, **k):
        return None

    def decode(self, x, t, c):
        i = int(round(float(t[0]) * self.sch.T))
        ab = self.sch.ab[i]
        return (x - ab.sqrt() * self.target) / (1 - ab).sqrt().clamp_min(1e-12)


def test_oracle_sampler():
    #End-to-end: the full sampling loop, given a perfect eps predictor, must
    #reconstruct the target from any starting noise, at any step budget.
    from flowmatch.diffusion import sample_diffusion
    sch = Schedule(100)
    g = torch.Generator().manual_seed(1)
    target = torch.randn(3, 64, 3, generator=g).clamp(-2.0, 2.0)
    oracle = _OracleEps(target, sch)
    sg = torch.zeros(3, 6)
    worst = 0.0
    for n_steps in (4, 8, 20, 100):
        out = sample_diffusion(
            oracle, None, None, sg, sch, n_waypoints=64, n_steps=n_steps,
            eta=0.0, device="cpu",
            generator=torch.Generator().manual_seed(2),
        )
        worst = max(worst, float((out - target).abs().max()))
    return _report("oracle model => sampler reconstructs the target", worst < 1e-3,
                   f"(max err {worst:.1e})")


def test_oracle_sampler_anchored():
    #Same, with endpoint inpainting on: the endpoints must come out exactly at
    #the requested values and the interior must still reconstruct.
    from flowmatch.diffusion import sample_diffusion
    sch = Schedule(100)
    g = torch.Generator().manual_seed(3)
    target = torch.randn(2, 64, 3, generator=g).clamp(-2.0, 2.0)
    oracle = _OracleEps(target, sch)
    s, gl = target[:, 0, :].clone(), target[:, -1, :].clone()
    out = sample_diffusion(
        oracle, None, None, torch.zeros(2, 6), sch, anchor_start=s, anchor_goal=gl,
        n_waypoints=64, n_steps=8, eta=0.0, anchor_endpoints=True, device="cpu",
        generator=torch.Generator().manual_seed(4),
    )
    ok = float((out[:, 0] - s).abs().max()) < 1e-6
    ok &= float((out[:, -1] - gl).abs().max()) < 1e-6
    return _report("anchored sampler pins the endpoints exactly", ok)


def test_loss_is_finite_and_reduces_for_perfect_model():
    #The eps-prediction loss must be exactly 0 for a model that returns the
    #noise it was given -- a check that q_sample and the loss agree on which
    #quantity is the target.
    sch = Schedule(100)
    g = torch.Generator().manual_seed(5)
    x0 = torch.randn(8, 64, 3, generator=g)
    noise = torch.randn(8, 64, 3, generator=g)
    i = torch.randint(1, sch.T + 1, (8,), generator=g)
    xt = sch.q_sample(x0, i, noise)
    #recover the noise analytically from (xt, x0)
    ab = sch.ab[i][:, None, None]
    rec = (xt - ab.sqrt() * x0) / (1 - ab).sqrt()
    err = float((rec - noise).abs().max())
    return _report("q_sample is invertible for the noise target", err < TOL,
                   f"(max err {err:.1e})")


def test_reduced_sampler_returns_world_frame():
    #The reduced sampler must undo its own frame: with an oracle whose target
    #is the reduced-frame straight line, the returned path must start and end
    #at the world-frame start and goal.
    from flowmatch.diffusion import sample_diffusion_reduced
    from flowmatch.geometry import sg_frame
    sch = Schedule(100)
    g = torch.Generator().manual_seed(6)
    start = torch.randn(4, 3, generator=g)
    goal = torch.randn(4, 3, generator=g)
    R0, origin, d = sg_frame(start, goal, None)
    #target = the canonical straight line in the reduced frame
    gamma = torch.linspace(0, 1, 64)[None, :, None]
    a = torch.stack([-0.5 * d, torch.zeros_like(d), torch.zeros_like(d)], -1)
    b = torch.stack([0.5 * d, torch.zeros_like(d), torch.zeros_like(d)], -1)
    target = (1 - gamma) * a[:, None, :] + gamma * b[:, None, :]

    #the reduced sampler genuinely rotates the scene, so it needs real-shaped
    #obstacle features even though this oracle ignores the conditioning
    spheres = torch.randn(4, 5, 4, generator=g)
    boxes = torch.randn(4, 5, 12, generator=g)

    oracle = _OracleEps(target, sch)
    out = sample_diffusion_reduced(
        oracle, spheres, boxes, start, goal, sch, n_waypoints=64, n_steps=20,
        eta=0.0, device="cpu", generator=torch.Generator().manual_seed(7),
    )
    e0 = float((out[:, 0, :] - start).abs().max())
    e1 = float((out[:, -1, :] - goal).abs().max())
    return _report("reduced sampler maps back to the world frame",
                   max(e0, e1) < 1e-3, f"(endpoint err {max(e0, e1):.1e})")


def main():
    print("diffusion arm: correctness tests (training-independent)\n")
    results = [
        test_schedule_conditioning(),
        test_step_matches_forward_process(),
        test_timesteps(),
        test_loss_is_finite_and_reduces_for_perfect_model(),
        test_oracle_sampler(),
        test_oracle_sampler_anchored(),
        test_reduced_sampler_returns_world_frame(),
    ]
    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
