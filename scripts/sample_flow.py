
  #.venv/bin/python sample_flow.py --ckpt checkpoints/flow.pt --data data1 \
  #    --env-idx 0 --n-pairs 5 --n-samples 20 --steps 100 --plot out.png
import argparse

import numpy as np
import torch

from flowmatch.data import Normalizer, env_features
from flowmatch.flow import sample, sample_reduced
from flowmatch.model import build_model
from pointmass3d import (
    BoxObstacle,
    PointMass3DEnv,
    SphereObstacle,
    mean_sq_accel,
    min_clearance,
    path_length,
    plot_scene,
)


def build_env(npz, robot_radius):
    obstacles = [SphereObstacle(s[:3], s[3]) for s in npz["spheres"]]
    obstacles += [BoxObstacle(b[:3], b[3:]) for b in npz["boxes"]]
    return PointMass3DEnv(obstacles, robot_radius=robot_radius)


def distinct_pairs(starts, goals, n):
    seen, idx = set(), []
    for i in range(len(starts)):
        key = np.round(np.concatenate([starts[i], goals[i]]), 4).tobytes()
        if key not in seen:
            seen.add(key)
            idx.append(i)
        if len(idx) >= n:
            break
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/flow.pt")
    ap.add_argument("--data", type=str, default="data1")
    ap.add_argument("--env-idx", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--k-fa", type=int, default=1,
                    help="frame-averaging width (reduced-frame checkpoints only)")
    ap.add_argument("--random-phi", action="store_true",
                    help="randomize the quadrature offset per query")
    ap.add_argument("--anchor-endpoints", action="store_true")
    ap.add_argument("--plot", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    norm = Normalizer.from_dict(ckpt["normalizer"])
    N = ckpt["n_waypoints"]

    model = build_model(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    #A checkpoint trained with sg_dim=1 was reduced; it can only be sampled in
    #the reduced frame, and only that arm supports frame averaging.
    reduced = model.cond_enc.sg_dim == 1
    if args.k_fa > 1 and not reduced:
        raise SystemExit("--k-fa > 1 requires a reduced-frame checkpoint (sg_dim=1)")
    print(f"arm={'treatment (reduced)' if reduced else 'control (world)'} "
          f"box_dim={model.obstacle_enc.box_dim} k_fa={args.k_fa}")

    npz = np.load(f"{args.data}/env_{args.env_idx:04d}.npz", allow_pickle=True)
    robot_radius = float(npz["robot_radius"])
    env = build_env(npz, robot_radius)

    # Normalized, batched obstacle tensors (shared across every sample).
    sp_t, bx_t = env_features(npz, norm, box_dim=model.obstacle_enc.box_dim)
    sp_t, bx_t = sp_t.to(device), bx_t.to(device)

    starts, goals, trajs = npz["starts"], npz["goals"], npz["trajs"]
    pair_idx = distinct_pairs(starts, goals, args.n_pairs)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    all_free, all_clear, all_ep, all_len, all_acc = [], [], [], [], []
    first_samples = None

    for pi in pair_idx:
        start, goal = starts[pi], goals[pi]
        s_n = torch.from_numpy(norm.norm_pts(start).astype(np.float32))
        g_n = torch.from_numpy(norm.norm_pts(goal).astype(np.float32))
        B = args.n_samples
        s_b = s_n.repeat(B, 1).to(device)
        g_b = g_n.repeat(B, 1).to(device)
        sp_b = sp_t.unsqueeze(0).expand(B, -1, -1)
        bx_b = bx_t.unsqueeze(0).expand(B, -1, -1)

        if reduced:
            phi = None
            if args.random_phi:
                phi = torch.rand(B, device=device, generator=gen) * 2 * np.pi
            x = sample_reduced(
                model, sp_b, bx_b, s_b, g_b, k_fa=args.k_fa,
                n_waypoints=N, n_steps=args.steps,
                anchor_endpoints=args.anchor_endpoints, device=device,
                generator=gen, phi=phi,
            )
        else:
            x = sample(
                model, sp_b, bx_b, torch.cat([s_b, g_b], dim=-1),
                anchor_start=s_b, anchor_goal=g_b,
                n_waypoints=N, n_steps=args.steps,
                anchor_endpoints=args.anchor_endpoints, device=device, generator=gen,
            )
        paths = norm.denorm_pts(x.cpu().numpy())  # (B, N, 3)

        free = [env.path_free(p) for p in paths]
        all_free.extend(free)
        all_clear.extend(min_clearance(env, p) for p in paths)
        all_ep.extend(
            0.5 * (np.linalg.norm(p[0] - start) + np.linalg.norm(p[-1] - goal))
            for p in paths
        )
        all_len.extend(path_length(p) for p in paths)
        all_acc.extend(mean_sq_accel(p) for p in paths)
        if first_samples is None:
            first_samples = (start, goal, paths, trajs[pi])

    def stat(name, xs, fmt="{:.4f}"):
        xs = np.asarray(xs)
        print(f"  {name:<18} mean {fmt.format(xs.mean())}   "
              f"min {fmt.format(xs.min())}   max {fmt.format(xs.max())}")

    n = len(all_free)
    print(f"\nenv {args.env_idx}: {len(pair_idx)} pairs x {args.n_samples} samples "
          f"= {n} trajectories  (anchor={args.anchor_endpoints})")
    print(f"  collision-free      {100*np.mean(all_free):.1f}%")
    stat("min clearance", all_clear)
    stat("endpoint error", all_ep)
    stat("path length", all_len)
    stat("mean sq accel", all_acc, "{:.5f}")

    if args.plot and first_samples is not None:
        start, goal, paths, expert = first_samples
        trajs_to_plot = {f"sample {k}": paths[k] for k in range(min(6, len(paths)))}
        trajs_to_plot["expert"] = expert
        plot_scene(
            env, trajectories=trajs_to_plot, start=start, goal=goal,
            title=f"flow-matching samples (env {args.env_idx})", save=args.plot,
        )
        print(f"  plot -> {args.plot}")


if __name__ == "__main__":
    main()
