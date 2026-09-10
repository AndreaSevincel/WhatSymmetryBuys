
  #.venv/bin/python eval_density.py --ckpts checkpoints/grid/ctrl-e250.pt \
  #    checkpoints/grid/treat-e250.pt

  #How do the two arms score in SIMPLER environments?

  #Both arms are trained at 20 spheres + 20 boxes and never see another density,
  #so this measures GENERALISATION across density, not the effect of training at
  #a different one. A drop at high density could be distribution shift rather
  #than difficulty and the two are not separable from this experiment alone.

  #Evaluation needs no expert trajectories -- only obstacles, starts, goals and
  #the collision checker -- which is what makes it cheap. The RRT+CHOMP pipeline
  #is the entire cost of the real dataset and is irrelevant here, so a density
  #axis costs minutes rather than the days a training sweep would.

  #The straight-line floor is recomputed at EVERY density, and that is the point.
  #As clutter thins the floor rises toward the ceiling, so a gap that shrinks in
  #sparse environments may be both arms compressing against a ceiling rather than
  #the representation mattering less. Without the floor the two readings are
  #indistinguishable, and the optimistic one is the wrong one.

import argparse

import numpy as np
import torch

from flowmatch.data import Normalizer
from flowmatch.flow import sample, sample_reduced
from flowmatch.model import build_model
from pointmass3d import (
    BoxObstacle,
    SphereObstacle,
    make_random_env,
    sample_start_goal,
)


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_config"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    #sg_dim identifies the arm, exactly as in sweep_steps.py: 1 = full
    #reduction, 6 = world frame. Reading it from the checkpoint rather than the
    #filename means a renamed file cannot silently be scored as the wrong arm.
    reduced = model.cond_enc.sg_dim == 1
    return model, Normalizer.from_dict(ck["normalizer"]), ck["n_waypoints"], reduced


def features(env, norm):
    #Mirrors flowmatch.data.env_features, but from a live env rather than a
    #shard. The normalisation must match training or the model is shown a
    #different world from the one it learned.
    sph = [o for o in env.obstacles if isinstance(o, SphereObstacle)]
    box = [o for o in env.obstacles if isinstance(o, BoxObstacle)]
    sp = np.zeros((len(sph), 4), dtype=np.float32)
    for i, o in enumerate(sph):
        sp[i, :3] = norm.norm_pts(o.center.astype(np.float32))
        sp[i, 3] = norm.norm_len(np.float32(o.radius))
    bx = np.zeros((len(box), 12), dtype=np.float32)
    for i, o in enumerate(box):
        bx[i, :3] = norm.norm_pts(o.center.astype(np.float32))
        #diagonal of the 3x3 half-edge matrix: axis-aligned in the world frame
        bx[i, 3::4] = norm.norm_len(o.half_extents.astype(np.float32))
    return torch.from_numpy(sp), torch.from_numpy(bx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--densities", type=int, nargs="+", default=[4, 8, 12, 16, 20],
                    help="obstacles of EACH type; 20 is the training density")
    ap.add_argument("--n-envs", type=int, default=30)
    ap.add_argument("--n-pairs", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=900_000,
                    help="env seeds start here, far from the training range so "
                         "no layout can coincide with one the models saw")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    arms = [(p.split("/")[-1].replace(".pt", ""), *load(p, args.device))
            for p in args.ckpts]
    names = [a[0] for a in arms]

    head = f"{'obstacles':>10}{'line':>8}"
    for n in names:
        head += f"{n[:11]:>12}"
    for n in names:
        head += f"{n[:9] + '@N':>12}"
    print(head)

    for d in args.densities:
        line, per_sample, per_query = [], {n: [] for n in names}, {n: [] for n in names}
        nq = 0
        for e in range(args.n_envs):
            env = make_random_env(n_spheres=d, n_boxes=d, seed=args.seed + e)
            rng = np.random.default_rng(args.seed + 500_000 + e)
            pairs = []
            for _ in range(args.n_pairs):
                try:
                    pairs.append(sample_start_goal(env, rng))
                except RuntimeError:
                    break
            for s, g in pairs:
                line.append(env.path_free(np.linspace(s, g, 64)))
                for name, model, norm, N, reduced in arms:
                    sp, bx = features(env, norm)
                    B = args.n_samples
                    s_b = torch.from_numpy(norm.norm_pts(s).astype(np.float32)).repeat(B, 1)
                    g_b = torch.from_numpy(norm.norm_pts(g).astype(np.float32)).repeat(B, 1)
                    sp_b = sp.unsqueeze(0).expand(B, -1, -1).to(args.device)
                    bx_b = bx.unsqueeze(0).expand(B, -1, -1).to(args.device)
                    s_b, g_b = s_b.to(args.device), g_b.to(args.device)
                    gen = torch.Generator(device=args.device).manual_seed(nq)
                    with torch.no_grad():
                        if reduced:
                            x = sample_reduced(model, sp_b, bx_b, s_b, g_b,
                                               n_waypoints=N, n_steps=args.steps,
                                               device=args.device, generator=gen)
                        else:
                            x = sample(model, sp_b, bx_b,
                                       torch.cat([s_b, g_b], dim=-1),
                                       anchor_start=s_b, anchor_goal=g_b,
                                       n_waypoints=N, n_steps=args.steps,
                                       device=args.device, generator=gen)
                    paths = norm.denorm_pts(x.cpu().numpy())
                    ok = np.array([env.path_free(p) for p in paths])
                    per_sample[name].append(ok.mean())
                    per_query[name].append(bool(ok.any()))
                nq += 1

        row = f"{2 * d:>10}{100 * np.mean(line):>7.1f}%"
        for n in names:
            row += f"{100 * np.mean(per_sample[n]):>11.1f}%"
        for n in names:
            row += f"{100 * np.mean(per_query[n]):>11.1f}%"
        print(row, flush=True)


if __name__ == "__main__":
    main()
