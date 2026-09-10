
  #python check_residual_agreement.py --ckpt checkpoints/grid/treat-e250.pt

  #Does the SE(3) residual agree with the roll residual on a reduced-frame model?

  #It must. Proposition 1 says a rigid motion T acts on the reduced
  #representation only through a roll about the start-goal axis, so the K copies
  #that se3_residual() compares differ from each other by rolls and nothing
  #else. If the two measurements disagree, one of them is wrong, and since the
  #roll number is the paper's headline r -- and Corollary 5's budget is quoted
  #from it -- that has to be settled before either is used.

  #Three sources of disagreement are checked separately:
  #  * QUADRATURE. sample_reduced() uses K equally spaced rolls, which is blind
  #    to harmonics at multiples of K: a component at m = K aliases onto the
  #    mean and contributes nothing to the measured spread. Random offsets see
  #    every harmonic. Comparing K=3,9 equally spaced against random phi
  #    isolates this.
  #  * SAMPLE SIZE. K=3 estimates a variance from three points.
  #  * IMPLEMENTATION. If equally spaced and random agree with each other but
  #    both disagree with se3_residual, the bug is in se3_residual.

import argparse

import numpy as np
import torch

from flowmatch.data import Normalizer, env_features
from flowmatch.flow import sample_reduced, se3_residual
from flowmatch.model import build_model
from sample_flow import build_env, distinct_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--env-start", type=int, default=250)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    norm = Normalizer.from_dict(ckpt["normalizer"])
    N = ckpt["n_waypoints"]
    model = build_model(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    reduced = model.cond_enc.sg_dim == 1
    if not reduced:
        raise SystemExit("this check needs a reduced-frame checkpoint: the roll "
                         "residual is not defined for the world-frame arm")
    box_dim = model.obstacle_enc.box_dim

    problems = []
    for ei in range(args.env_start, args.env_start + args.n_envs):
        npz = np.load(f"{args.data}/env_{ei:04d}.npz", allow_pickle=True)
        sp_t, bx_t = env_features(npz, norm, box_dim=box_dim)
        starts, goals = npz["starts"], npz["goals"]
        for i in distinct_pairs(starts, goals, args.n_pairs):
            problems.append((sp_t.to(device), bx_t.to(device),
                             starts[i], goals[i]))
    print(f"{len(problems)} queries, {args.steps} steps\n")

    rows = []
    for label, kind, k in (("roll, K=3 equally spaced", "roll", 3),
                           ("roll, K=9 equally spaced", "roll", 9),
                           ("roll, K=9 random offset", "rollphi", 9),
                           ("roll, K=32 random offset", "rollphi", 32),
                           ("SE(3), K=9 rotations", "se3", 9),
                           ("SE(3), K=32 rotations", "se3", 32)):
        vals = []
        for qi, (sp, bx, s, g) in enumerate(problems):
            s_t = torch.from_numpy(norm.norm_pts(s).astype(np.float32))[None].to(device)
            g_t = torch.from_numpy(norm.norm_pts(g).astype(np.float32))[None].to(device)
            gen = torch.Generator(device=device).manual_seed(1234 + qi)
            if kind == "se3":
                vals.append(se3_residual(
                    model, sp[None], bx[None], s_t, g_t, reduced=True, k=k,
                    n_waypoints=N, n_steps=args.steps, trans=0.0,
                    device=device, generator=gen))
            else:
                #a random quadrature offset per query turns exact C_K
                #equivariance into equivariance in distribution, which is what
                #makes the estimator see harmonics at multiples of K
                phi = (torch.rand(1, device=device, generator=gen) * 2 * np.pi
                       if kind == "rollphi" else None)
                out = sample_reduced(
                    model, sp[None], bx[None], s_t, g_t, k_fa=k,
                    n_waypoints=N, n_steps=args.steps, device=device,
                    generator=gen, phi=phi, return_residual=True)
                vals.append(float(np.mean(out[1])))
        rows.append((label, float(np.mean(vals)), float(np.std(vals))))
        print(f"  {label:<26} r = {rows[-1][1]:.4f}  (sd over queries {rows[-1][2]:.4f})")

    roll_eq = rows[0][1]
    roll_rand = rows[3][1]
    se3 = rows[5][1]
    print("\ninterpretation:")
    if abs(roll_rand - roll_eq) > 0.5 * max(roll_eq, 1e-9):
        print("  * equally spaced != random offset -> the K-point quadrature is")
        print("    ALIASING. The paper's r is an underestimate; quote the random-")
        print("    offset value and say which estimator was used.")
    else:
        print("  * equally spaced == random offset -> quadrature is not the issue.")
    if abs(se3 - roll_rand) > 0.5 * max(roll_rand, 1e-9):
        print("  * SE(3) != roll even at matched K and random offsets -> the two")
        print("    are measuring different things, which Proposition 1 forbids.")
        print("    Suspect se3_residual(): check the reduced branch, in")
        print("    particular that v is un-reduced by R0 BEFORE being un-rotated")
        print("    by Q, and that the scene is rotated in the same order.")
    else:
        print("  * SE(3) == roll at matched K -> both are correct, and the")
        print("    earlier gap was quadrature or sample size, not a bug.")


if __name__ == "__main__":
    main()
