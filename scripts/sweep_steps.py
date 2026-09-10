
  #.venv/bin/python sweep_steps.py --ckpt checkpoints/flow_cpu_demo.pt --data data1 \
  #    --n-envs 5 --n-pairs 5 --n-samples 20 --steps 100 50 20 10 8 4

  #Step 0 of the equivariance plan: how few ODE steps can the trained flow
  #tolerate? Seed-matched across step counts (identical prior draws), so any
  #metric drift is attributable to integration error alone. The 100-step row is
  #the reference; the budget is the smallest count that holds its quality.

import argparse
import time

import numpy as np
import torch

from flowmatch import tracking
from flowmatch.data import Normalizer, env_features
from flowmatch.diffusion import Schedule, sample_diffusion, sample_diffusion_reduced
from flowmatch.flow import (
    sample,
    sample_reduced,
    sample_translation_reduced,
    se3_residual,
)
from flowmatch.model import build_model
from pointmass3d import (
    BoxObstacle,
    PointMass3DEnv,
    SphereObstacle,
    mean_sq_accel,
    min_clearance,
    path_length,
)
from sample_flow import build_env, distinct_pairs
from se3body import RigidBody, SE3Env, decode_poses, sample_se3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/flow_cpu_demo.pt")
    ap.add_argument("--data", type=str, default="data1")
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--env-start", type=int, default=0,
                    help="first env shard to evaluate; point this at the "
                         "held-out test range, not at envs the model trained on")
    ap.add_argument("--n-pairs", type=int, default=5, help="pairs per env")
    ap.add_argument("--n-samples", type=int, default=20, help="samples per pair")
    ap.add_argument("--steps", type=int, nargs="+", default=[100, 50, 20, 10, 8, 4])
    ap.add_argument("--k-fa", type=int, default=1,
                    help="frame-averaging width (reduced checkpoints only)")
    ap.add_argument("--residual", action="store_true",
                    help="also report std(v_k)/||mean(v_k)||, the non-equivariance "
                         "frame averaging is removing. 0 => already equivariant, "
                         "so a larger K_FA cannot help. Needs --k-fa > 1")
    ap.add_argument("--anchor-endpoints", action="store_true")
    ap.add_argument("--residual-se3", type=int, default=0,
                    help="also measure the residual under the FULL SE(3) action "
                         "with this many random rigid motions. Works for ANY "
                         "arm, including the world frame, which has no roll "
                         "gauge and so cannot be measured by --residual")
    ap.add_argument("--residual-se3-trans", type=float, default=0.0,
                    help="translation half-width for --residual-se3; 0 keeps it "
                         "rotations-only so the measurement is not confounded "
                         "with out-of-distribution translation")
    ap.add_argument("--dump-per-problem", action="store_true",
                    help="record n_free/n_samples for every query in the "
                         "output JSON. Needed for PAIRED tests against a "
                         "baseline measured on the same problems: comparing "
                         "two independently-resampled rates ignores that both "
                         "arms track the same per-environment difficulty, and "
                         "overstates the standard error of their difference")
    ap.add_argument("--out-json", type=str, default=None,
                    help="write the rows to JSON for aggregate_runs.py")
    ap.add_argument("--eta", type=float, default=0.0,
                    help="ddpm arm only: 0 = deterministic DDIM, 1 = ancestral")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    tracking.add_args(ap)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    norm = Normalizer.from_dict(ckpt["normalizer"])
    N = ckpt["n_waypoints"]
    model = build_model(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    #sg_dim identifies the arm: 6 = world frame, 1 = full reduction,
    #3 = translation-only reduction (the direction g-s is still informative)
    #A state is 3 numbers (point mass) or 9 (SE(3) pose: position + 6D
    #rotation). The conditioning width then identifies the arm: point mass uses
    #1 for the full reduction and 6 for the world frame; SE(3) uses 13 and 18,
    #because the reduction canonicalises the endpoint positions but not their
    #orientations.
    state_dim = getattr(model, "state_dim", 3)
    is_se3 = state_dim == 9
    sg = getattr(model, "sg_dim", None)
    if sg is None:                      # checkpoints predating the attribute
        sg = model.cond_enc.sg_dim
    reduced = (sg == 13) if is_se3 else (sg == 1)
    translation_only = (not is_se3) and sg == 3
    if translation_only and args.k_fa > 1:
        raise SystemExit("frame averaging needs the rotational reduction; "
                         "the translation-only arm has no roll gauge")
    #Checkpoints predating the diffusion arm have no "objective" key.
    objective = ckpt.get("objective", "flow")
    schedule = (Schedule(ckpt.get("diffusion_steps", 100), device=device)
                if objective == "ddpm" else None)
    if objective == "ddpm" and args.k_fa > 1:
        raise SystemExit("frame averaging is implemented for the flow arm only")
    if is_se3 and (args.k_fa > 1 or args.residual_se3 > 0):
        raise SystemExit(
            "the frame-averaging and SE(3) residual paths are written for "
            "3-dim states; they would silently mis-handle 9-dim poses. Run the "
            "SE(3) domain without --k-fa/--residual-se3.")
    if args.k_fa > 1 and not reduced:
        raise SystemExit("--k-fa > 1 requires a reduced-frame checkpoint (sg_dim=1)")
    box_dim = model.obstacle_enc.box_dim

    run = tracking.from_args(
        args,
        name=f"sweep-steps-{objective}-{'treat' if reduced else 'ctrl'}-k{args.k_fa}",
        config={**vars(args), "reduced": reduced, "box_dim": box_dim},
    )

    # Pre-load the eval envs once; the same problems are reused for every step count.
    problems = []  # (env, sp_t, bx_t, list_of_(start,goal))
    for ei in range(args.env_start, args.env_start + args.n_envs):
        npz = np.load(f"{args.data}/env_{ei:04d}.npz", allow_pickle=True)
        if is_se3:
            #SE3Env evaluates the obstacle SDF at the body's transformed sphere
            #centres and subtracts the sphere radius itself, so the base env is
            #built with no robot radius of its own.
            env = SE3Env(build_env(npz, 0.0),
                         RigidBody(npz["body_centers"], float(npz["robot_radius"])))
        else:
            env = build_env(npz, float(npz["robot_radius"]))
        sp_t, bx_t = env_features(npz, norm, box_dim=box_dim)
        starts, goals = npz["starts"], npz["goals"]
        idx = distinct_pairs(starts, goals, args.n_pairs)
        problems.append((
            env, sp_t.to(device), bx_t.to(device),
            #(env index, pair index) travel with each query so --dump-per-problem
            #can be joined against baselines_classical.json, which keys on those
            #two fields. Matching on coordinates instead would work but would be
            #a silent failure if either side ever reorders.
            [(starts[i], goals[i], ei, int(i)) for i in idx],
        ))

    header = (f"{'steps':>6} {'free%':>7} {'any%':>6} {'clearance':>10} {'ep_err':>8} "
              f"{'length':>8} {'sq_accel':>9} {'disp_ref':>9} {'s/query':>8}"
              + (f" {'residual':>9}" if (args.residual and reduced and args.k_fa > 1)
                 else ""))
    print(f"\nenvs [{args.env_start}, {args.env_start + args.n_envs}) x "
          f"{args.n_pairs} pairs x {args.n_samples} samples, "
          f"seed-matched (anchor={args.anchor_endpoints})")
    print(header)

    ref_paths = None  # 100-step (first entry) trajectories, for displacement
    rows = []
    for n_steps in args.steps:
        free, clear, ep, length, acc = [], [], [], [], []
        per_problem = []
        #per-sample free% is what the ablations compare, but a sampler is
        #deployed as best-of-N behind a collision check, so also track whether
        #ANY of the N samples for a query is free
        solved_any = []
        paths_all, residuals, residuals_l1, res_se3 = [], [], [], []
        #The residual is identically 0 at K=1 (nothing to disagree with), so
        #only ask for it when there are multiple frames to compare.
        want_residual = args.residual and reduced and args.k_fa > 1
        t0 = time.time()
        n_queries = 0
        for env, sp_t, bx_t, pairs in problems:
            for start, goal, env_i, pair_i in pairs:
                B = args.n_samples
                def _norm_state(v):
                    #rotation columns are unit vectors: scaling or shifting them
                    #is meaningless, so only the position is normalised
                    if not is_se3:
                        return norm.norm_pts(v).astype(np.float32)
                    out = np.asarray(v, dtype=np.float32).copy()
                    out[:3] = norm.norm_pts(v[:3])
                    return out

                s_b = torch.from_numpy(_norm_state(start)).repeat(B, 1).to(device)
                g_b = torch.from_numpy(_norm_state(goal)).repeat(B, 1).to(device)
                sp_b = sp_t.unsqueeze(0).expand(B, -1, -1)
                bx_b = bx_t.unsqueeze(0).expand(B, -1, -1)
                # Fresh generator per query: identical prior draw for every step count.
                gen = torch.Generator(device=device).manual_seed(
                    args.seed * 100003 + n_queries
                )
                if is_se3:
                    x = sample_se3(
                        model, sp_b, bx_b, s_b, g_b, n_waypoints=N,
                        n_steps=n_steps, reduced=reduced, device=device,
                        generator=gen,
                    )
                elif objective == "ddpm":
                    #Same NFE as the flow arm: n_steps network evaluations.
                    if reduced:
                        x = sample_diffusion_reduced(
                            model, sp_b, bx_b, s_b, g_b, schedule,
                            n_waypoints=N, n_steps=n_steps, eta=args.eta,
                            anchor_endpoints=args.anchor_endpoints,
                            device=device, generator=gen,
                        )
                    else:
                        x = sample_diffusion(
                            model, sp_b, bx_b, torch.cat([s_b, g_b], dim=-1),
                            schedule, anchor_start=s_b, anchor_goal=g_b,
                            n_waypoints=N, n_steps=n_steps, eta=args.eta,
                            anchor_endpoints=args.anchor_endpoints,
                            device=device, generator=gen,
                        )
                elif translation_only:
                    x = sample_translation_reduced(
                        model, sp_b, bx_b, s_b, g_b, n_waypoints=N,
                        n_steps=n_steps, anchor_endpoints=args.anchor_endpoints,
                        device=device, generator=gen,
                    )
                elif reduced:
                    out = sample_reduced(
                        model, sp_b, bx_b, s_b, g_b, k_fa=args.k_fa,
                        n_waypoints=N, n_steps=n_steps,
                        anchor_endpoints=args.anchor_endpoints,
                        device=device, generator=gen,
                        return_residual=want_residual,
                    )
                    if want_residual:
                        x, res, res_l1 = out
                        residuals.extend(res)
                        residuals_l1.extend(res_l1)
                    else:
                        x = out
                else:
                    x = sample(
                        model, sp_b, bx_b, torch.cat([s_b, g_b], dim=-1),
                        anchor_start=s_b, anchor_goal=g_b,
                        n_waypoints=N, n_steps=n_steps,
                        anchor_endpoints=args.anchor_endpoints,
                        device=device, generator=gen,
                    )
                if is_se3:
                    #decode_poses applies Gram-Schmidt to the six unconstrained
                    #rotation outputs and denormalises the positions. Collision
                    #checking is over POSES: a rigid body sweeps volume as it
                    #rotates, so the point-mass path_free would miss exactly the
                    #collisions this domain exists to test.
                    paths, rots = decode_poses(x, norm)
                    free_q = [env.path_free(paths[i], rots[i]) for i in range(len(paths))]
                    clear.extend(env.min_clearance(paths[i], rots[i])
                                 for i in range(len(paths)))
                    #endpoint error stays positional so it is comparable with
                    #the point-mass domain; orientation error is reported apart
                    ep.extend(
                        0.5 * (np.linalg.norm(p[0] - start[:3])
                               + np.linalg.norm(p[-1] - goal[:3]))
                        for p in paths
                    )
                else:
                    paths = norm.denorm_pts(x.cpu().numpy())
                    free_q = [env.path_free(p) for p in paths]
                    clear.extend(min_clearance(env, p) for p in paths)
                    ep.extend(
                        0.5 * (np.linalg.norm(p[0] - start) + np.linalg.norm(p[-1] - goal))
                        for p in paths
                    )
                paths_all.append(paths)
                free.extend(free_q)
                solved_any.append(any(free_q))
                if args.dump_per_problem:
                    #per-QUERY, not per-sample: the paired comparison against a
                    #one-shot baseline is at the query level, and the model's
                    #per-sample rate for a query is n_free/n_samples
                    per_problem.append({
                        "env": int(env_i), "pair": int(pair_i),
                        "n_free": int(sum(bool(f) for f in free_q)),
                        "n_samples": int(len(free_q)),
                    })
                length.extend(path_length(p) for p in paths)
                acc.extend(mean_sq_accel(p) for p in paths)
                if args.residual_se3 > 0:
                    res_se3.append(se3_residual(
                        model, sp_b[:1], bx_b[:1], s_b[:1], g_b[:1],
                        reduced=reduced, k=args.residual_se3,
                        n_waypoints=N, n_steps=n_steps,
                        trans=args.residual_se3_trans, device=device,
                        generator=torch.Generator(device=device).manual_seed(
                            args.seed * 7919 + n_queries),
                    ))
                n_queries += 1
        secs = (time.time() - t0) / n_queries

        paths_all = np.stack(paths_all)  # (Q, B, N, 3)
        if ref_paths is None:
            ref_paths = paths_all
            disp = 0.0
        else:
            disp = float(np.linalg.norm(paths_all - ref_paths, axis=-1).mean())

        row = dict(
            steps=n_steps, free=100 * float(np.mean(free)),
            solved_any=100 * float(np.mean(solved_any)),
            clearance=float(np.mean(clear)), ep_err=float(np.mean(ep)),
            length=float(np.mean(length)), sq_accel=float(np.mean(acc)),
            disp_ref=disp, s_per_query=secs,
        )
        if args.dump_per_problem:
            row["per_problem"] = per_problem
        if want_residual:
            #Hilbert norm, the quantity the projection bound is stated in
            row["residual"] = float(np.mean(residuals))
            #the pre-2026-08 definition, kept so older numbers stay comparable
            row["residual_meannorm"] = float(np.mean(residuals_l1))
        if res_se3:
            #defined for every arm, so the diagnostic can be reported with two
            #signs rather than one
            row["residual_se3"] = float(np.mean(res_se3))
        rows.append(row)
        print(f"{n_steps:>6} {row['free']:>6.1f}% {row['solved_any']:>5.1f}% "
              f"{row['clearance']:>10.4f} "
              f"{row['ep_err']:>8.4f} {row['length']:>8.4f} {row['sq_accel']:>9.5f} "
              f"{row['disp_ref']:>9.4f} {row['s_per_query']:>8.3f}"
              + (f" {row['residual']:>9.4f}" if want_residual else ""))
        run.log({f"sweep/{k}": v for k, v in row.items()}, step=n_steps)

    ref = rows[0]
    ok = [r for r in rows if r["free"] >= ref["free"] - 2.0]
    budget = min(ok, key=lambda r: r["steps"])
    print(f"\nreference: {ref['steps']} steps at {ref['free']:.1f}% free")
    print(f"budget: {budget['steps']} steps holds within 2% "
          f"({budget['free']:.1f}% free, {budget['s_per_query']:.3f}s/query, "
          f"{ref['s_per_query']/max(budget['s_per_query'],1e-9):.1f}x faster)")
    if args.out_json:
        import json
        from pathlib import Path as _Path
        _Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            #`reduced` means the FULL reduction (sg_dim=1); the translation-only
            #arm is also a reduced arm but has sg_dim=3, so an aggregator keying
            #off `reduced` alone mislabels it as a world-frame run. Record the
            #arm explicitly.
            json.dump({"config": vars(args), "objective": objective,
                       "reduced": reduced,
                       "arm": "treat" if (reduced or translation_only) else "ctrl",
                       "rows": rows}, f, indent=1)
        print(f"wrote {args.out_json}")

    run.summary(budget_steps=budget["steps"], budget_free=budget["free"])
    run.finish()


if __name__ == "__main__":
    main()
