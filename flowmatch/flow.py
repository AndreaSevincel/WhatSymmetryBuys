#Pure flow-matching objective and ODE sampler.

#Conditional OT / rectified-flow formulation with a *standard Gaussian* prior:
#target velocity u = x1 - x0
#The network regresses u; sampling integrates dx/dt = v_theta(x, t, c) from the
#Gaussian prior at t=0 to the data manifold at t=1 with explicit Euler steps.

#Two arms share this code:
#  control   -- world frame, conditioned on raw (start, goal), sg_dim=6
#  treatment -- (s,g)-reduced frame with uniform roll augmentation, sg_dim=1
#The treatment's reduction and roll are composed into ONE per-sample rotation,
#so there is a single place where the point-vs-vector typing can go wrong.

import copy
import math

import torch

from .geometry import (
    apply_points,
    apply_vectors,
    check_frame,
    rotate_box_features,
    rotate_sphere_features,
    sg_frame,
)


def build_conditioning(start, goal, reduced):
    #sg vector for ConditionEncoder. Reduced arms keep only the invariant
    #scalar d = ||g-s||; world-frame arms keep the raw pair.
    if reduced:
        return torch.linalg.norm(goal - start, dim=-1, keepdim=True)  # (B,1)
    return torch.cat([start, goal], dim=-1)                           # (B,6)


def random_rotations(B, device, dtype=torch.float32, generator=None):
    #Uniform on SO(3) by QR of a Gaussian, sign-fixed so det = +1.
    a = torch.randn(B, 3, 3, device=device, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
    #a reflection has det -1; flipping one column repairs it
    flip = torch.sign(torch.linalg.det(q))
    q[:, :, 0] = q[:, :, 0] * flip[:, None]
    return q


def augment_se3(x1, start, goal, spheres, boxes, trans=0.25, generator=None):
    """Random rigid motion of the WHOLE problem, for the augmented control arm.

    This is the honest competitor to canonicalisation: the practitioner who
    knows the problem is SE(3)-equivariant and reaches for data augmentation
    instead of a frame. It is a different mechanism -- it makes equivariance
    likely rather than certain, and it costs training signal, since the network
    must spend capacity learning what the reduction gets for free.

    Translation is drawn in a box of half-width `trans` in NORMALISED units.
    Unbounded translation would be the mathematically complete group, but it
    would also present scenes displaced far outside the workspace the model is
    evaluated in, which tests out-of-distribution behaviour rather than
    equivariance. 0.25 is a quarter of the workspace half-width.
    """
    B = start.shape[0]
    Q = random_rotations(B, start.device, start.dtype, generator)
    t = (torch.rand(B, 3, device=start.device, dtype=start.dtype,
                    generator=generator) * 2 - 1) * trans
    #apply_points(R, o, p) = R(p - o), so T = (Q, t) is R=Q with o = -Q^T t
    o = -torch.einsum("bji,bj->bi", Q, t)
    return (
        apply_points(Q, o, x1),
        apply_points(Q, o, start[:, None, :])[:, 0],
        apply_points(Q, o, goal[:, None, :])[:, 0],
        rotate_sphere_features(spheres, Q, o),
        rotate_box_features(boxes, Q, o),
    )


def reduce_batch(x1, start, goal, spheres, boxes, roll=True, check=False,
                 mode="full"):
    #World -> (s,g)-reduced frame for a whole batch, with the roll composed in.
    #Returns (x1_r, sg, spheres_r, boxes_r, R, origin).

    #mode="translation" is the ablation that splits the five removable degrees
    #of freedom: it recentres on the midpoint but does NOT rotate, so the three
    #translations are canonicalised and the two rotations are not. The query
    #direction then still carries information, so the conditioning is the full
    #vector g-s (3 numbers) rather than the invariant scalar d.
    B = start.shape[0]
    if mode == "translation":
        origin = 0.5 * (start + goal)
        R = torch.eye(3, device=start.device, dtype=start.dtype).expand(B, 3, 3)
        x1_r = apply_points(R, origin, x1)
        spheres_r = rotate_sphere_features(spheres, R, origin)
        boxes_r = rotate_box_features(boxes, R, origin)
        return x1_r, goal - start, spheres_r, boxes_r, R, origin

    theta = (
        torch.rand(B, device=start.device) * 2 * math.pi
        if roll else torch.zeros(B, device=start.device)
    )
    R, origin, d = sg_frame(start, goal, theta)
    if check:
        check_frame(R, origin, start, goal, d)

    x1_r = apply_points(R, origin, x1)                       # POINTS
    spheres_r = rotate_sphere_features(spheres, R, origin)    # center point, radius m=0
    boxes_r = rotate_box_features(boxes, R, origin)           # center point, edges vectors
    sg = d[:, None]                                          # invariant scalar
    return x1_r, sg, spheres_r, boxes_r, R, origin


def flow_matching_loss(model, batch, tables, reduced=False, roll=True,
                       check=False, mode="full", augment=0.0):
    x1 = batch["traj"]
    start, goal, env_id = batch["start"], batch["goal"], batch["env_id"]
    spheres = tables["spheres"][env_id]
    boxes = tables["boxes"][env_id]
    sphere_mask = tables["sphere_mask"][env_id]
    box_mask = tables["box_mask"][env_id]

    #Augmentation is applied to the WORLD-frame problem, before any reduction.
    #On the reduced arm it would be a no-op up to the roll gauge, which is the
    #point of the reduction; it exists for the world-frame arm.
    if augment > 0.0:
        x1, start, goal, spheres, boxes = augment_se3(
            x1, start, goal, spheres, boxes, trans=augment
        )

    if reduced:
        #One rotation for the reduction AND the roll augmentation. x0 is drawn
        #AFTER, and never rotated: an isotropic Gaussian is already invariant,
        #so rotating it would only add a redundant (and error-prone) step.
        x1, sg, spheres, boxes, _, _ = reduce_batch(
            x1, start, goal, spheres, boxes, roll=roll, check=check, mode=mode
        )
    else:
        sg = build_conditioning(start, goal, reduced=False)

    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    xt = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
    target = x1 - x0

    pred = model(xt, t, spheres, boxes, sg, sphere_mask, box_mask)
    return ((pred - target) ** 2).mean()


@torch.no_grad()
def sample(
    model,
    spheres,
    boxes,
    sg,
    anchor_start=None,
    anchor_goal=None,
    sphere_mask=None,
    box_mask=None,
    n_waypoints=64,
    n_steps=100,
    anchor_endpoints=False,
    device="cpu",
    generator=None,
    capture=None,
):

    #Draw trajectories by integrating the learned velocity field.
    #capture, if given, is a list that receives a CPU copy of the state after
    #every Euler step. It is for visualisation only: it never touches x, so a
    #run with capture set produces bit-identical samples to one without.
    #Everything here is in whatever frame the caller supplies; anchor_start /
    #anchor_goal must be in that same frame.

    model.eval()
    net = model.module if hasattr(model, "module") else model  # unwrap DataParallel
    B = sg.shape[0]
    x = torch.randn(B, n_waypoints, 3, device=device, generator=generator)

    # Conditioning is time-independent: encode once, reuse every step.
    c = net.encode_cond(spheres, boxes, sg, sphere_mask, box_mask)

    known_eps = None
    if anchor_endpoints:
        known_eps = x[:, [0, -1], :].clone()  # frozen noise for the endpoints
        known = torch.stack([anchor_start, anchor_goal], dim=1)  # (B, 2, 3)

    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t = ts[i].expand(B)
        if anchor_endpoints:
            tt = ts[i]
            x[:, [0, -1], :] = (1.0 - tt) * known_eps + tt * known
        v = net.decode(x, t, c, spheres, boxes, sphere_mask, box_mask)
        x = x + (ts[i + 1] - ts[i]) * v
        if capture is not None:
            capture.append(x.detach().cpu().clone())

    if anchor_endpoints:
        x[:, 0, :] = anchor_start
        x[:, -1, :] = anchor_goal
        if capture:
            capture[-1] = x.detach().cpu().clone()
    return x


def frame_averaged_velocity(net, x, t, c_k, roll, k_fa, sph_k=None, box_k=None):
    #v_bar(x) = (1/K) sum_k roll_k^T f(roll_k x).
    #Exact C_K-equivariance holds for ANY f: rolling x by 2*pi*j/K permutes the
    #quadrature points, and reindexing the sum is a bijection on them.
    #Returns (v_bar (B,N,3), vk (B,K,N,3)).
    B, N = x.shape[0], x.shape[1]
    xk = apply_vectors(roll, x.repeat_interleave(k_fa, 0))
    #sph_k/box_k are the K ROLLED scene tensors, i.e. already in the same
    #frame as xk. Passing the unrolled originals here would evaluate the
    #local geometry in a different frame from the state and silently break
    #the C_K equivariance the whole operator exists to provide.
    tk = t.repeat_interleave(k_fa, 0) if t.shape[0] == B else t
    #Only forward the scene when the model actually consumes it. The diagnostic
    #harnesses substitute their own minimal decode(x, t, c) stubs -- an
    #unconditional 5-argument call would break them, and a test that cannot run
    #is worse than the coupling it was meant to avoid.
    if getattr(net, "local_geom", False):
        vk = net.decode(xk, tk, c_k, sph_k, box_k)
    else:
        vk = net.decode(xk, tk, c_k)
    vk = torch.einsum("bji,bkj->bki", roll, vk)   # roll^T = un-roll, exact
    vk = vk.reshape(B, k_fa, N, 3)
    return vk.mean(dim=1), vk


def roll_matrices(start, goal, thetas, k_fa):
    #Rolls taking reduced frame 0 into each rolled frame k, as (B*K, 3, 3).
    #Equals R_x(theta_k) exactly, since frame(a) = R_x(a) @ frame(0).
    R0, _, _ = sg_frame(start, goal, None)
    flat_start = start.repeat_interleave(k_fa, 0)
    flat_goal = goal.repeat_interleave(k_fa, 0)
    Rk, origin_k, dk = sg_frame(flat_start, flat_goal, thetas.reshape(-1))
    R0f = R0.repeat_interleave(k_fa, 0)
    return torch.einsum("bij,bkj->bik", Rk, R0f), Rk, origin_k, dk


@torch.no_grad()
def se3_residual(model, spheres, boxes, start, goal, reduced, k=8,
                 sphere_mask=None, box_mask=None, n_waypoints=64, n_steps=8,
                 trans=0.0, device="cpu", generator=None):
    """Non-equivariance residual under the FULL SE(3) action, for any arm.

    The roll residual of sample_reduced() is only defined for a reduced-frame
    model: it measures disagreement over the one gauge degree of freedom the
    reduction leaves behind. It cannot be computed for a world-frame model,
    which has no reduced frame -- so on its own it can only ever be reported
    for arms that already exploit the symmetry, and a diagnostic applied once,
    with one sign, demonstrates nothing.

    This measures the same quantity over the group the PROBLEM has. K random
    rigid motions T are applied to the whole problem (scene, endpoints, state);
    the field is evaluated in each; each result is mapped back by the inverse
    rotation, since a velocity is a free vector and takes the rotation only.
    For an exactly SE(3)-equivariant model every copy agrees and r = 0.

    Returns the RMS residual of Eq. (6), averaged over integration steps.

    trans is the translation half-width in normalised units. It defaults to 0
    (rotations only) because a translated scene is also out of distribution,
    which would conflate non-equivariance with extrapolation; set it >0 to
    measure the translational part deliberately.
    """
    net = model.module if hasattr(model, "module") else model
    net.eval()
    B, N = start.shape[0], n_waypoints
    BK = B * k

    Q = random_rotations(BK, device, start.dtype, generator)
    if trans > 0:
        t = (torch.rand(BK, 3, device=device, dtype=start.dtype,
                        generator=generator) * 2 - 1) * trans
    else:
        t = torch.zeros(BK, 3, device=device, dtype=start.dtype)
    o = -torch.einsum("bji,bj->bi", Q, t)

    rep = lambda z: z.repeat_interleave(k, 0)
    s_k = apply_points(Q, o, rep(start)[:, None, :])[:, 0]
    g_k = apply_points(Q, o, rep(goal)[:, None, :])[:, 0]
    sph_k = rotate_sphere_features(rep(spheres), Q, o)
    box_k = rotate_box_features(rep(boxes), Q, o)
    sm = None if sphere_mask is None else rep(sphere_mask)
    bm = None if box_mask is None else rep(box_mask)

    if reduced:
        #the reduction is applied inside each transformed copy, exactly as at
        #sampling time, so what is measured is the residual of the whole
        #pipeline rather than of the network alone
        R0, origin, d = sg_frame(s_k, g_k, None)
        sph_k = rotate_sphere_features(sph_k, R0, origin)
        box_k = rotate_box_features(box_k, R0, origin)
        sg_k = d[:, None]
    else:
        sg_k = build_conditioning(s_k, g_k, reduced=False)
    c_k = net.encode_cond(sph_k, box_k, sg_k, sm, bm)

    #The probe must sit where the model's own sampler operates, or the
    #measurement reports off-distribution behaviour instead of
    #non-equivariance. The world-frame sampler draws N(0,I) in world
    #coordinates -- and a rigid rotation of an isotropic Gaussian is still
    #N(0,I), so drawing here is already correct for that arm. The reduced
    #sampler draws N(0,I) in the REDUCED frame; mapping a world-frame draw in
    #leaves the state offset by the query midpoint, which inflated r by ~3.4x.
    x = torch.randn(B, N, 3, device=device, generator=generator)
    if reduced:
        R00, origin00, _ = sg_frame(start, goal, None)
        x = torch.einsum("bji,bkj->bki", R00, x) + origin00[:, None, :]

    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    num = den = 0.0
    for i in range(n_steps):
        xk = apply_points(Q, o, rep(x))
        if reduced:
            xk = apply_points(R0, origin, xk)
        #sph_k/box_k are the K TRANSFORMED scenes, i.e. already in the same
        #frame as xk -- the same coupling frame_averaged_velocity relies on.
        #Only forward them when the model consumes local geometry, so the
        #diagnostic stubs with a 3-argument decode still run.
        if getattr(net, "local_geom", False):
            v = net.decode(xk, ts[i].expand(BK), c_k, sph_k, box_k)
        else:
            v = net.decode(xk, ts[i].expand(BK), c_k)
        if reduced:
            v = torch.einsum("bji,bkj->bki", R0, v)     # un-reduce (rotation only)
        v = torch.einsum("bji,bkj->bki", Q, v)          # un-rotate: Q^T v
        vk = v.reshape(B, k, N, 3)
        vbar = vk.mean(dim=1)
        num += float((vk - vbar[:, None]).pow(2).sum(-1).mean())
        den += float(vbar.pow(2).sum(-1).mean())
        x = x + (ts[i + 1] - ts[i]) * vbar
    return (num ** 0.5) / max(den ** 0.5, 1e-12)


@torch.no_grad()
def sample_translation_reduced(
    model, spheres, boxes, start, goal, sphere_mask=None, box_mask=None,
    n_waypoints=64, n_steps=100, anchor_endpoints=False, device="cpu",
    generator=None,
):
    #Sampler for the translation-only ablation. No frame averaging: that
    #mechanism exists for the roll gauge introduced by the ROTATIONAL part of
    #the reduction, which this arm does not perform.
    net = model.module if hasattr(model, "module") else model
    net.eval()
    B = start.shape[0]
    origin = 0.5 * (start + goal)
    sg = goal - start                                     # (B,3), not invariant
    R = torch.eye(3, device=device, dtype=start.dtype).expand(B, 3, 3)
    sph_r = rotate_sphere_features(spheres, R, origin)
    box_r = rotate_box_features(boxes, R, origin)

    x = sample(
        net, sph_r, box_r, sg,
        anchor_start=start - origin, anchor_goal=goal - origin,
        sphere_mask=sphere_mask, box_mask=box_mask, n_waypoints=n_waypoints,
        n_steps=n_steps, anchor_endpoints=anchor_endpoints, device=device,
        generator=generator,
    )
    return x + origin[:, None, :]


@torch.no_grad()
def sample_reduced(
    model,
    spheres,
    boxes,
    start,
    goal,
    k_fa=1,
    sphere_mask=None,
    box_mask=None,
    n_waypoints=64,
    n_steps=100,
    anchor_endpoints=False,
    device="cpu",
    generator=None,
    phi=None,
    return_residual=False,
    capture=None,
):
    #Sample in the (s,g)-reduced frame and map the result back to world.
    #k_fa > 1 turns on frame averaging over the residual roll: K rolled copies
    #of the scene are encoded once, then at every ODE step the state is rolled
    #into each frame, decoded, UN-rolled, and averaged. The un-roll is exact
    #(an orthogonal matrix), so all residual error is model or pipeline.
    net = model.module if hasattr(model, "module") else model
    net.eval()
    B = start.shape[0]

    R0, origin, d = sg_frame(start, goal, None)
    sg = d[:, None]
    #Quadrature offsets. A random phi per query trades exact C_K-equivariance
    #for equivariance in distribution, which converts systematic aliased bias
    #into zero-mean scatter -- usually the better trade for a sampler.
    base = torch.arange(k_fa, device=device, dtype=start.dtype) * (2 * math.pi / k_fa)
    if phi is None:
        phi = torch.zeros(B, device=device, dtype=start.dtype)
    thetas = phi[:, None] + base[None, :]                        # (B, K)

    #K rolled scene encodings, computed ONCE outside the ODE loop.
    roll, Rk, origin_k, dk = roll_matrices(start, goal, thetas, k_fa)
    sph_k = rotate_sphere_features(spheres.repeat_interleave(k_fa, 0), Rk, origin_k)
    box_k = rotate_box_features(boxes.repeat_interleave(k_fa, 0), Rk, origin_k)
    c_k = net.encode_cond(
        sph_k, box_k, dk[:, None],
        None if sphere_mask is None else sphere_mask.repeat_interleave(k_fa, 0),
        None if box_mask is None else box_mask.repeat_interleave(k_fa, 0),
    )

    x = torch.randn(B, n_waypoints, 3, device=device, generator=generator)
    if anchor_endpoints:
        known_eps = x[:, [0, -1], :].clone()
        zeros = torch.zeros_like(d)
        a_s = torch.stack([-0.5 * d, zeros, zeros], dim=-1)
        a_g = torch.stack([0.5 * d, zeros, zeros], dim=-1)
        known = torch.stack([a_s, a_g], dim=1)

    residual, residual_l1 = [], []
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        if anchor_endpoints:
            tt = ts[i]
            x[:, [0, -1], :] = (1.0 - tt) * known_eps + tt * known
        t = ts[i].expand(B * k_fa)
        v, vk = frame_averaged_velocity(net, x, t, c_k, roll, k_fa, sph_k, box_k)
        if return_residual:
            #The Hilbert norm, NOT a ratio of mean norms. The projection
            #identity ||f - Af|| / ||f|| = r / sqrt(1 + r^2) holds in
            #<f,g> = E_x[f(x)^T g(x)], so r must be
            #   sqrt(E_{x,k} ||v_k - vbar||^2) / sqrt(E_x ||vbar||^2).
            #The earlier version used mean(||.||) in both places; Jensen makes
            #the two differ, and the identity is then only approximate.
            num = ((vk - v[:, None]).pow(2).sum(-1).mean()).sqrt()
            den = (v.pow(2).sum(-1).mean()).sqrt().clamp_min(1e-9)
            residual.append((num / den).item())
            #kept alongside so numbers measured before the fix stay comparable
            num_l1 = (vk - v[:, None]).norm(dim=-1).mean()
            residual_l1.append(
                (num_l1 / v.norm(dim=-1).mean().clamp_min(1e-9)).item()
            )
        x = x + (ts[i + 1] - ts[i]) * v
        if capture is not None:
            #recorded in WORLD coordinates, so a viewer can compare the two arms
            #frame by frame; the reduced state itself lives in a rotated frame
            capture.append(
                (torch.einsum("bji,bkj->bki", R0, x) + origin[:, None, :])
                .detach().cpu().clone())

    if anchor_endpoints:
        x[:, 0, :] = known[:, 0]
        x[:, -1, :] = known[:, 1]
        if capture:
            capture[-1] = (
                torch.einsum("bji,bkj->bki", R0, x) + origin[:, None, :]
            ).detach().cpu().clone()

    #reduced -> world: R0^T @ x + origin
    x_world = torch.einsum("bji,bkj->bki", R0, x) + origin[:, None, :]
    if return_residual:
        #(Hilbert-norm residual, legacy mean-norm ratio)
        return x_world, residual, residual_l1
    return x_world


class EMA:
    #Exponential moving average of model parameters
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        #cached so the parameter list is not rebuilt every step
        self._shadow_params = None

    @torch.no_grad()
    def update(self, model):
        #Fused over the whole parameter list. The obvious loop issues two tiny
        #CUDA kernels per tensor, so a 2.16M-parameter model with ~100 tensors
        #costs ~200 launches EVERY optimiser step -- launch-bound overhead that
        #is a visible fraction of the step time for a model this small.
        #torch._foreach_* does the same arithmetic in two calls.
        params = list(model.parameters())
        if self._shadow_params is None or len(self._shadow_params) != len(params):
            self._shadow_params = list(self.shadow.parameters())
        torch._foreach_mul_(self._shadow_params, self.decay)
        torch._foreach_add_(self._shadow_params, params, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        #for resuming a run: the shadow weights are the ones eval uses, so
        #restarting without them silently restarts the average from the raw
        #weights and costs an EMA horizon of quality at every resume
        self.shadow.load_state_dict(sd)
        self._shadow_params = None
