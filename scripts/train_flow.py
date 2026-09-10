

  #.venv/bin/python train_flow.py --data data1 --epochs 300 --batch 1024 --amp --multi-gpu

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flowmatch import tracking
from flowmatch.data import build_datasets
from flowmatch.diffusion import Schedule, diffusion_loss
from flowmatch.flow import EMA, flow_matching_loss
from flowmatch.equivariant import EquivVelocityField
from flowmatch.model import FlowVelocityField


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data1")
    ap.add_argument("--out", type=str, default="checkpoints/flow.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")
    ap.add_argument("--multi-gpu", action="store_true", help="DataParallel over all visible GPUs")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="proceed on CPU. Without it, a run that silently fell "
                         "back to CPU is aborted: at this dataset size that is "
                         "never what was wanted, and it wastes hours before the "
                         "first epoch even prints")
    ap.add_argument("--log-every", type=int, default=50)
    #Convergence runs need the curve, not the endpoint, and best-val
    #checkpointing cannot supply it: val loss is not comparable across arms
    #(the reduced arm regresses canonicalised targets, so its MSE is
    #mechanically smaller) and is not the quantity being plotted anyway. These
    #snapshots are indexed by EPOCH so a held-out collision-free rate can be
    #scored at fixed budgets on both arms.
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also write <out>.epNNNN.pt every N epochs. Carries "
                         "optimiser and scaler state, so a snapshot is also a "
                         "resume point (see --resume)")
    ap.add_argument("--resume", action="store_true",
                    help="restart from the newest <out>.epNNNN.pt if one "
                         "exists. For cluster time limits: requeue the same "
                         "command and it continues rather than restarting")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model. Often 1.3-2x on this "
                         "conv trunk, at the cost of a slow first epoch")
    # model
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=8)
    # equivariance arm
    ap.add_argument("--reduced", action="store_true",
                    help="treatment arm: (s,g) reduction + roll augmentation")
    ap.add_argument("--no-roll", action="store_true",
                    help="with --reduced, disable roll augmentation (ablation)")
    ap.add_argument("--n-envs", type=int, default=None,
                    help="use N environments starting at --env-start (scale sweep)")
    ap.add_argument("--env-start", type=int, default=0,
                    help="first env shard to use; reserve a tail range as test set")
    ap.add_argument("--split-by", choices=["traj", "pair", "env"], default="pair",
                    help="what val holds out. traj is LEAKY on this dataset "
                         "(30 near-duplicate paths per pair); pair is the default")
    ap.add_argument("--query-encoder", action="store_true",
                    help="concatenate the raw query to every obstacle before the "
                         "pool, so the scene code varies with the query WITHOUT a "
                         "change of frame. The control that separates 'let the "
                         "encoder see the query' from 'canonicalise'")
    ap.add_argument("--vec-channels", type=int, default=34,
                    help="width of the m=1 (rotating) stream in the equivariant "
                         "backbone; the scalar stream uses --channels")
    ap.add_argument("--env-vec", type=int, default=32,
                    help="width of the m=1 stream in the equivariant obstacle "
                         "encoder and conditioning (defaults differ from "
                         "--vec-channels; changing either invalidates existing "
                         "equivariant checkpoints)")
    ap.add_argument("--equivariant", action="store_true",
                    help="use the SO(2)-equivariant backbone instead of the "
                         "unconstrained one. Requires --reduced: the constraint "
                         "is about the residual roll the reduction leaves, and a "
                         "world-frame arm has no such axis. This is the third way "
                         "of handling the sixth degree of freedom -- in the "
                         "weights, rather than in the data or in the operator")
    ap.add_argument("--local-geom", action="store_true",
                    help="ORACLE DIAGNOSTIC: append the true SDF and its "
                         "gradient at each waypoint to the trunk input. "
                         "Measures how much headroom the global max-pooled "
                         "scene code is costing, WITHOUT building a better "
                         "encoder. Not a method: it supplies exact geometry "
                         "no perception-based system would have, so its score "
                         "is an upper bound. Run BOTH arms or the comparison "
                         "is meaningless")
    ap.add_argument("--check-frame", action="store_true",
                    help="assert the reduction is well-formed on every batch (slow)")
    #generative-model control: same backbone, same conditioning, same frame
    #options, different objective. See flowmatch/diffusion.py.
    #The two arms a referee asks for: augmentation is the practitioner's
    #alternative to canonicalisation, and translation-only splits the five
    #removable DOF into 3 translations vs 2 rotations.
    ap.add_argument("--augment", type=float, default=0.0,
                    help="world-frame arm: random SE(3) augmentation per batch. "
                         "The value is the translation half-width in normalised "
                         "units (0.25 is a quarter of the workspace). 0 = off")
    ap.add_argument("--reduce-mode", choices=["full", "translation"], default="full",
                    help="with --reduced: full removes 3 translations + 2 "
                         "rotations; translation removes only the translations")
    #Dial for validating the residual as a PREDICTOR rather than measuring it
    #once. r is small on this benchmark because the gauge is a deterministic
    #function of x-hat and the data supplies ~600 distinct directions per
    #environment, so the reduced-frame data already spans many rolls. Training
    #on a narrow cone of start-goal directions starves the model of that
    #diversity, which should drive r up -- and if r is a predictor, frame
    #averaging should start to pay in proportion. Sweeping the cone width turns
    #one measurement into a curve.
    ap.add_argument("--xhat-cone", type=float, default=None,
                    help="train only on pairs whose start-goal direction lies "
                         "within this many DEGREES of +x. Shrinks r's implicit "
                         "roll diversity; use with --subsample to hold the "
                         "training-set size fixed")
    ap.add_argument("--subsample", type=int, default=None,
                    help="cap the training set at N trajectories, applied AFTER "
                         "--xhat-cone. Needed to keep a cone-restricted arm "
                         "comparable to an unrestricted one")
    ap.add_argument("--domain", choices=["pointmass", "se3"], default="pointmass",
                    help="pointmass: 3-dim waypoints. se3: 9-dim poses "
                         "(position + 6D rotation), see se3body/")
    ap.add_argument("--objective", choices=["flow", "ddpm"], default="flow",
                    help="flow: conditional OT velocity regression. "
                         "ddpm: Diffuser/MPD-style eps-prediction")
    ap.add_argument("--diffusion-steps", type=int, default=100,
                    help="T for --objective ddpm (sampling NFE is set at eval)")
    tracking.add_args(ap)
    return ap.parse_args()


def default_run_name(args):
    #Arm and env count first: those are the 2x4 grid axes.
    arm = "treat" if args.reduced else "ctrl"
    if args.reduced and args.no_roll:
        arm += "-noroll"
    if args.equivariant:
        arm += "-equiv"
    if args.objective != "flow":
        arm = f"{args.objective}-{arm}"
    if args.domain != "pointmass":
        arm = f"{args.domain}-{arm}"
    if args.reduced and args.reduce_mode != "full":
        arm = f"{arm}-{args.reduce_mode}"
    if args.augment > 0:
        arm = f"{arm}-aug"
    if args.xhat_cone is not None:
        arm = f"{arm}-cone{int(args.xhat_cone)}"
    envs = f"e{args.n_envs}" if args.n_envs else f"e{Path(args.data).name}"
    return f"{arm}-{envs}-c{args.channels}-b{args.n_blocks}-s{args.seed}"


def make_model_config(args):
    return dict(
        channels=args.channels,
        n_blocks=args.n_blocks,
        dilations=(1, 2, 4, 8),
        time_dim=256,
        env_hidden=128,
        env_dim=128,
        cond_dim=256,
        groups=8,
        box_dim=12,
        #After the reduction start/goal are (-+d/2, 0, 0), so the six numbers
        #collapse to the one invariant scalar d. The control genuinely needs
        #the raw pair -- this asymmetry is intrinsic to the treatment.
        #point mass: 6 raw numbers, or the single invariant d after reduction.
        #SE(3): 18 raw (two poses), or d plus both orientations = 13, since the
        #reduction canonicalises the positions but not the orientations.
        #translation-only leaves the query DIRECTION informative, so the
        #conditioning is the full vector g-s rather than the scalar d
        sg_dim=(
            (3 if args.reduce_mode == "translation" else 1) if args.reduced else 6
        ) if args.domain == "pointmass" else (13 if args.reduced else 18),
        state_dim=3 if args.domain == "pointmass" else 9,
        local_geom=args.local_geom,
        query_cond=args.query_encoder,
    )


def compute_loss(net, batch, tables, args, schedule, check=False):
    if args.domain == "se3":
        if args.objective == "ddpm":
            raise SystemExit("the ddpm arm is implemented for the point-mass "
                             "domain only; use --objective flow with --domain se3")
        from se3body.flow import flow_matching_loss_se3
        return flow_matching_loss_se3(net, batch, tables, reduced=args.reduced,
                                      roll=not args.no_roll, check=check)
    if args.objective == "ddpm":
        return diffusion_loss(net, batch, tables, schedule, reduced=args.reduced,
                              roll=not args.no_roll, check=check)
    return flow_matching_loss(net, batch, tables, reduced=args.reduced,
                              roll=not args.no_roll, check=check,
                              mode=args.reduce_mode, augment=args.augment)


def evaluate(net, loader, tables, device, args, schedule):
    net.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(net, batch, tables, args, schedule)
            bs = batch["traj"].shape[0]
            total += loss.item() * bs
            n += bs
    return total / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    #A silent CPU fallback on a 2M-trajectory dataset looks exactly like a hang.
    #Abort unless CPU was asked for explicitly.
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit(
            "CUDA is not available, so this would train on CPU and take days.\n"
            f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}\n"
            "  Diagnose with:  nvidia-smi ; python -c \"import torch; "
            "print(torch.cuda.is_available())\"\n"
            "  A 'forward compatibility' error (804) means the driver and the "
            "CUDA runtime disagree -- reload the kernel module or drop the "
            "CUDA compat libs from LD_LIBRARY_PATH.\n"
            "  Pass --allow-cpu if you really do want CPU."
        )
    if device.type != "cuda" and (args.amp or args.multi_gpu):
        print("[warn] --amp/--multi-gpu are no-ops on CPU")
    if device.type == "cuda":
        #Shapes are fixed for the whole run, so let cuDNN autotune once instead
        #of picking a generic algorithm every call. TF32 costs nothing here:
        #the targets are noisy regression values, not something needing full
        #fp32 mantissa.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device={device}  cuda_devices={torch.cuda.device_count()}")

    train_ds, val_ds, normalizer, _ = build_datasets(
        args.data, val_frac=args.val_frac, seed=args.seed, n_envs=args.n_envs,
        env_start=args.env_start, split_by=args.split_by,
    )
    print(f"split_by={args.split_by}  envs [{args.env_start}, "
          f"{args.env_start + (args.n_envs or 0)})")
    #--xhat-cone / --subsample reshape the TRAIN split only; val is untouched
    #so every arm is scored on the same held-out queries.
    if args.xhat_cone is not None or args.subsample is not None:
        idx = np.asarray(train_ds.indices)
        if args.xhat_cone is not None:
            d = train_ds.goals[idx] - train_ds.starts[idx]
            d = np.asarray(d, dtype=np.float64)[..., :3]
            xh = d / np.linalg.norm(d, axis=-1, keepdims=True).clip(1e-12)
            cos_lim = np.cos(np.deg2rad(args.xhat_cone))
            #|.| so the cone is a double cone: a path and its reverse describe
            #the same geometry and must not be split across the filter
            keep = np.abs(xh[:, 0]) >= cos_lim
            idx = idx[keep]
            if len(idx) == 0:
                raise SystemExit(f"--xhat-cone {args.xhat_cone} kept 0 trajectories")
        if args.subsample is not None and len(idx) > args.subsample:
            idx = np.random.default_rng(args.seed).choice(
                idx, size=args.subsample, replace=False)
        train_ds = train_ds.subset(idx)
        print(f"filtered train -> {len(train_ds)} trajectories "
              f"(cone={args.xhat_cone}, subsample={args.subsample})")

    print(f"train={len(train_ds)}  val={0 if val_ds is None else len(val_ds)} trajectories")

    #obstacle lookup tables live on the device the whole time
    tables = {
        "spheres": train_ds.spheres.to(device),
        "boxes": train_ds.boxes.to(device),
        "sphere_mask": train_ds.sphere_mask.to(device),
        "box_mask": train_ds.box_mask.to(device),
    }

    #persistent_workers matters here: without it the worker processes are torn
    #down and respawned every epoch, and each respawn re-imports torch and
    #re-inherits the dataset. Over 20-60 epochs that is pure overhead.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch, num_workers=args.num_workers)
        if val_ds is not None else None
    )

    #A fixed slice of TRAINING data, scored with the same (EMA) weights and the
    #same code path as validation. Without it the printed train/val pair is not
    #a comparison: train is a running average over the epoch on the raw weights
    #while the optimiser is still moving them, and val is the EMA weights at
    #epoch end. Both differences push val below train, so "val < train" is the
    #expected reading even under mild overfitting, and cannot be used as
    #evidence of underfitting. This loader makes the gap mean what it says.
    train_eval_loader = None
    if val_ds is not None and len(train_ds) > 0:
        n_probe = min(len(val_ds), len(train_ds))
        probe_idx = np.random.default_rng(args.seed).choice(
            np.asarray(train_ds.indices), size=n_probe, replace=False
        )
        train_eval_loader = DataLoader(
            train_ds.subset(probe_idx), batch_size=args.batch,
            num_workers=args.num_workers,
        )

    cfg = make_model_config(args)
    if args.equivariant and args.query_encoder:
        raise SystemExit("--query-encoder is a world-frame control; it makes no "
                         "sense with the equivariant backbone, whose query "
                         "conditioning is the single invariant scalar d")
    if args.equivariant:
        cfg.pop("query_cond", None)
        if not args.reduced:
            raise SystemExit("--equivariant requires --reduced: the SO(2) constraint "
                             "is about the roll the (s,g) reduction leaves behind")
        #--local-geom IS supported here: the SDF gradient is split by irrep and
        #routed to both streams (d, g_x -> scalars; g_yz -> the m=1 stream)
        #rather than concatenated to the state, which is what the earlier
        #version refused to guess at.
        #
        #Recorded explicitly rather than left to the module defaults. They were
        #chosen to match the baseline at 2.15M, but make_model_config overrides
        #channels/time_dim/cond_dim from the shared CLI flags, so what actually
        #gets built is 2.58M -- 19% ABOVE the 2.16M baseline. Leaving that
        #implicit is how the run/config disagreement went unnoticed; leaving it
        #unrecorded would mean a checkpoint could not be rebuilt if a default
        #ever moved.
        #Three separate widths, NOT one. The first draft of this line set all
        #three from --vec-channels and so moved env_vec/cond_vec from 32 to 34,
        #which changes the architecture: 2,582,233 -> 2,584,649 parameters, and
        #every checkpoint already on disk stops loading. The defaults below are
        #the module's, so recording them is a no-op by construction.
        cfg = {**cfg, "vec_channels": args.vec_channels,
               "env_vec": args.env_vec, "cond_vec": args.env_vec}
        core = EquivVelocityField(**cfg).to(device)
        from flowmatch.model import FlowVelocityField as _Base
        base = sum(p.numel() for p in
                   _Base(**{k: v for k, v in cfg.items()
                            if k not in ("vec_channels", "env_vec", "cond_vec")}
                         ).parameters())
        got = sum(p.numel() for p in core.parameters())
        print(f"capacity: equivariant {got:,} vs unconstrained {base:,} "
              f"({100 * (got / base - 1):+.1f}%)")
        #recorded so build_model() rebuilds the right class at scoring time
        cfg = {**cfg, "equivariant": True}
    else:
        core = FlowVelocityField(**cfg).to(device)
    n_params = sum(p.numel() for p in core.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    ema = EMA(core, decay=args.ema_decay)
    net = core
    if args.compile:
        net = torch.compile(net)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(core)
        print(f"DataParallel over {torch.cuda.device_count()} GPUs")

    run = tracking.from_args(
        args,
        name=default_run_name(args),
        config={
            **vars(args),
            "model": cfg,
            "n_params": n_params,
            "n_train": len(train_ds),
            "n_val": 0 if val_ds is None else len(val_ds),
            "n_envs": train_ds.spheres.shape[0],
            "n_waypoints": train_ds.trajs.shape[1],
            "cuda_devices": torch.cuda.device_count(),
        },
    )
    if run.active:
        print(f"wandb: {run.url}")

    #Noise schedule for --objective ddpm; unused by the flow arm.
    schedule = Schedule(args.diffusion_steps, device=device)

    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    try:  # current API (torch >= 2.3); fall back for older builds
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    gstep = 0

    def payload(epoch, val_loss, training_state):
        p = {
            "model": core.state_dict(),
            "ema": ema.state_dict(),
            "model_config": cfg,
            "normalizer": normalizer.to_dict(),
            "n_waypoints": train_ds.trajs.shape[1],
            "epoch": epoch,
            "val_loss": val_loss,
            #eval needs to know which sampler to use; absent => "flow"
            "objective": args.objective,
            "domain": args.domain,
            "reduce_mode": args.reduce_mode,
            "augment": args.augment,
            "diffusion_steps": args.diffusion_steps,
            "reduced": args.reduced,
        }
        if training_state:
            p["opt"] = opt.state_dict()
            p["scaler"] = scaler.state_dict()
            p["best_val"] = best_val
            p["gstep"] = gstep
        return p

    def atomic_save(obj, path):
        #Write to a temp file and rename. os.replace is atomic within a
        #filesystem, so a reader either sees the previous checkpoint or the new
        #one, never a half-written mixture. Saving in place is a real hazard
        #here: evaluating a checkpoint mid-write does not raise, it silently
        #returns a much worse score, which reads as a training failure rather
        #than as a torn read.
        tmp = str(path) + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def snapshot_path(epoch):
        out = Path(args.out)
        return out.with_name(f"{out.stem}.ep{epoch + 1:04d}{out.suffix}")

    start_epoch = 0
    if args.resume:
        out = Path(args.out)
        snaps = sorted(out.parent.glob(f"{out.stem}.ep[0-9]*{out.suffix}"))
        if snaps:
            ck = torch.load(snaps[-1], map_location=device, weights_only=False)
            core.load_state_dict(ck["model"])
            ema.load_state_dict(ck["ema"])
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
                scaler.load_state_dict(ck["scaler"])
            else:
                #a plain checkpoint has no optimiser state, so Adam's moments
                #restart cold. Say so rather than silently changing the run.
                print(f"WARNING: {snaps[-1].name} carries no optimiser state; "
                      "Adam moments restart from zero")
            best_val = ck.get("best_val", float("inf"))
            gstep = ck.get("gstep", 0)
            start_epoch = ck["epoch"] + 1
            #the shuffle order is NOT restored, so a resumed run sees the same
            #data in a different order. Harmless for the curve; noted so it is
            #not later mistaken for a seed effect.
            print(f"resumed {snaps[-1].name} at epoch {start_epoch}")
        else:
            print(f"--resume: no snapshot for {out.stem}, starting from scratch")

    for epoch in range(start_epoch, args.epochs):
        net.train()
        running, seen = 0.0, 0
        t_epoch = time.time()
        for it, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = compute_loss(net, batch, tables, args, schedule,
                                    check=args.check_frame)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(core)

            bs = batch["traj"].shape[0]
            running += loss.item() * bs
            seen += bs
            gstep += 1
            if args.log_every and it % args.log_every == 0:
                print(f"  epoch {epoch:03d} it {it:04d}  loss {loss.item():.4f}")
                run.log({"train/loss_step": loss.item(), "epoch": epoch}, step=gstep)

        train_loss = running / max(seen, 1)
        msg = f"epoch {epoch:03d}  train {train_loss:.4f}"
        if val_loader is not None:
            val_loss = evaluate(ema.shadow, val_loader, tables, device,
                                args, schedule)
            #same weights, same code path, training data
            train_ema = evaluate(ema.shadow, train_eval_loader, tables, device,
                                 args, schedule)
            msg += f"  train(ema) {train_ema:.4f}"
            msg += f"  val(ema) {val_loss:.4f}"
        else:
            val_loss = train_loss
        msg += f"  [{time.time()-t0:.0f}s]"
        print(msg)
        run.log(
            {
                "train/loss": train_loss,
                "train/loss_ema": train_ema,
                "val/loss_ema": val_loss,
                #the only honest generalisation gap: like-for-like weights
                "gap/val_minus_train_ema": val_loss - train_ema,
                "lr": opt.param_groups[0]["lr"],
                "epoch": epoch,
                "time/epoch_s": time.time() - t_epoch,
                "time/elapsed_s": time.time() - t0,
            },
            step=gstep,
        )

        if val_loss < best_val:
            best_val = val_loss
            run.summary(best_val_loss=best_val, best_epoch=epoch)
            atomic_save(payload(epoch, best_val, training_state=False), args.out)

        #Snapshots are written on a fixed epoch grid and on the final epoch, so
        #the two arms are always compared at identical budgets even when their
        #best-val epochs differ.
        if args.snapshot_every and (
            (epoch + 1) % args.snapshot_every == 0 or epoch + 1 == args.epochs
        ):
            atomic_save(payload(epoch, val_loss, training_state=True),
                        snapshot_path(epoch))
    print(f"done. best val {best_val:.4f}. checkpoint -> {args.out}")
    run.summary(total_time_s=time.time() - t0)
    run.finish()


if __name__ == "__main__":
    main()
