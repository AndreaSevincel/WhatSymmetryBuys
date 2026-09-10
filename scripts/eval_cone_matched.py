
  #.venv/bin/python eval_cone_matched.py

  #Does the non-equivariance residual PREDICT what frame averaging is worth?

  #run_cone.sh trains four arms on progressively narrower cones of start-goal
  #directions, which drives r up as intended (0.0182 -> 0.0238). But it scored
  #them on the FULL held-out set, spanning every direction -- so the narrow-cone
  #arms were measured almost entirely out of distribution, and their collapse in
  #quality (K=1: 31.4 -> 20.6) is distribution shift rather than anything about
  #roll diversity. That confounds the experiment: r and quality then move in
  #opposite directions by construction, frame averaging is known to pay more on
  #better models, and a flat result is consistent with r mattering OR not.

  #The fix is in the EVALUATION, not the training. Score every arm on the same
  #narrow cone of held-out queries. Because 15 deg is a subset of 30, 45 and 90,
  #every arm has trained on those directions and none is extrapolating, so the
  #quality gap should largely close while the difference in roll diversity --
  #the variable the experiment is about -- remains.

  #Reported per arm: quality (K=1), the frame-averaging gain (K=3 minus K=1,
  #paired within checkpoint, hence free of seed noise), and r on the same pass.

import argparse
import json

import numpy as np
import torch

from flowmatch.data import Normalizer, env_features
from flowmatch.flow import sample_reduced
from flowmatch.model import build_model
from sample_flow import build_env, distinct_pairs

CONES = [15, 30, 45, 90]
CK = "checkpoints/grid"


def load(tag, device):
    ck = torch.load(f"{CK}/{tag}.pt", map_location=device, weights_only=False)
    m = build_model(ck["model_config"]).to(device)
    m.load_state_dict(ck["ema"])
    m.eval()
    return m, Normalizer.from_dict(ck["normalizer"]), ck["n_waypoints"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--eval-cone", type=float, default=15.0,
                    help="held-out queries within this many degrees of +x. Must "
                         "be <= the NARROWEST training cone, or that arm is "
                         "extrapolating again and the confound returns")
    ap.add_argument("--env-start", type=int, default=250)
    ap.add_argument("--n-envs", type=int, default=50)
    ap.add_argument("--n-pairs", type=int, default=40,
                    help="candidates per env BEFORE the cone filter; a 15 deg "
                         "cone keeps a few percent, so this must be generous")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--k-fa", type=int, default=3)
    ap.add_argument("--out-json", type=str, default="results/cone_matched.json")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cos_lim = np.cos(np.deg2rad(args.eval_cone))
    rows = []
    print(f"held-out queries within {args.eval_cone:g} deg of the first axis, "
          f"envs [{args.env_start}, {args.env_start + args.n_envs})")
    print(f"{'train cone':>11}{'n':>6}{'K=1':>8}{'K=3':>8}{'delta':>8}{'r':>9}")

    for c in CONES:
        tag = f"treat-e60-cone{c}"
        model, norm, N = load(tag, args.device)
        box_dim = model.obstacle_enc.box_dim
        free1, free3, resid = [], [], []
        nq = 0
        for ei in range(args.env_start, args.env_start + args.n_envs):
            z = np.load(f"{args.data}/env_{ei:04d}.npz", allow_pickle=True)
            env = build_env(z, float(z["robot_radius"]))
            sp, bx = env_features(z, norm, box_dim=box_dim)
            starts, goals = z["starts"], z["goals"]
            for i in distinct_pairs(starts, goals, args.n_pairs):
                s, g = starts[i], goals[i]
                d = g - s
                #|.| so the cone is a DOUBLE cone: a path and its reverse are
                #the same start-goal axis, exactly as in train_flow.py
                if abs(d[0]) / np.linalg.norm(d) < cos_lim:
                    continue
                B = args.n_samples
                s_b = torch.from_numpy(norm.norm_pts(s).astype(np.float32)).repeat(B, 1).to(args.device)
                g_b = torch.from_numpy(norm.norm_pts(g).astype(np.float32)).repeat(B, 1).to(args.device)
                sp_b = sp.unsqueeze(0).expand(B, -1, -1).to(args.device)
                bx_b = bx.unsqueeze(0).expand(B, -1, -1).to(args.device)
                for k in (1, args.k_fa):
                    #same seed for both K, so the two share prior draws and the
                    #difference is paired rather than a comparison of samples
                    gen = torch.Generator(device=args.device).manual_seed(nq)
                    with torch.no_grad():
                        out = sample_reduced(
                            model, sp_b, bx_b, s_b, g_b, k_fa=k,
                            n_waypoints=N, n_steps=args.steps,
                            device=args.device, generator=gen,
                            return_residual=(k > 1),
                        )
                    if k > 1:
                        x, res, _ = out
                        resid.extend(res)
                    else:
                        x = out
                    ok = np.array([env.path_free(p)
                                   for p in norm.denorm_pts(x.cpu().numpy())])
                    (free1 if k == 1 else free3).append(ok.mean())
                nq += 1
        f1, f3 = 100 * np.mean(free1), 100 * np.mean(free3)
        r = float(np.mean(resid))
        rows.append(dict(train_cone=c, n=nq, k1=f1, k3=f3, delta=f3 - f1, r=r))
        print(f"{c:>9} deg{nq:>6}{f1:>8.2f}{f3:>8.2f}{f3 - f1:>+8.2f}{r:>9.4f}", flush=True)

    with open(args.out_json, "w") as fh:
        json.dump({"config": vars(args), "rows": rows}, fh, indent=1)
    print(f"\nwrote {args.out_json}")

    #The question in one line: across arms whose quality is now comparable, does
    #the gain move with r?
    k1 = np.array([r["k1"] for r in rows])
    rr = np.array([r["r"] for r in rows])
    dd = np.array([r["delta"] for r in rows])
    print(f"quality spread (K=1): {k1.max() - k1.min():.2f} points "
          f"[{k1.min():.2f}, {k1.max():.2f}]")
    print(f"r spread: {rr.max() - rr.min():.4f} [{rr.min():.4f}, {rr.max():.4f}]")
    if len(rows) > 2:
        print(f"corr(r, delta) = {np.corrcoef(rr, dd)[0, 1]:+.3f}   "
              f"corr(K=1, delta) = {np.corrcoef(k1, dd)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
