
  #.venv/bin/python make_demo_figures.py

  #Presentation figures showing the benchmark and the classical planners.

  #Separate from make_figures.py, which draws the RESULTS. These draw the
  #SETTING: what an environment looks like, what the three classical planners
  #produce on the same query, and what the Brownian-bridge prior is. They exist
  #because "40 obstacles in a two-unit cube" does not convey how dense the
  #clutter is until you see it.

  #Colours match the rest of the project: the reduction's orange and the
  #control's blue, so a reader who has seen the result plots recognises them.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pointmass3d import (
    BoxObstacle,
    SphereObstacle,
    brownian,
    chomp,
    make_random_env,
    min_clearance,
    path_length,
    resample_path,
    rrt_connect,
    sample_start_goal,
    shortcut,
    trajopt,
)

C_RRT = "#3E6DA8"
C_CHOMP = "#C2560F"
C_TRAJOPT = "#1F7A5A"
C_PRIOR = "#8A94A0"
OBST = "#9AA6B2"


def _draw_env(ax, env, alpha=0.22):
    for o in env.obstacles:
        if isinstance(o, SphereObstacle):
            u = np.linspace(0, 2 * np.pi, 18)
            v = np.linspace(0, np.pi, 12)
            ax.plot_surface(
                o.center[0] + o.radius * np.outer(np.cos(u), np.sin(v)),
                o.center[1] + o.radius * np.outer(np.sin(u), np.sin(v)),
                o.center[2] + o.radius * np.outer(np.ones_like(u), np.cos(v)),
                color=OBST, alpha=alpha, linewidth=0, shade=True)
        elif isinstance(o, BoxObstacle):
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            c, h = np.asarray(o.center), np.asarray(o.half_extents)
            k = c + h * np.array([[sx, sy, sz] for sx in (-1, 1)
                                  for sy in (-1, 1) for sz in (-1, 1)])
            faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                     (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
            ax.add_collection3d(Poly3DCollection(
                [[k[i] for i in f] for f in faces], facecolor=OBST,
                edgecolor="#6B7683", linewidths=0.3, alpha=alpha))


def _style(ax, title=None, lim=1.0):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    #zoom fills the panel: a 3D axes reserves generous margins by default and
    #leaves a square box floating in a lot of white
    ax.set_box_aspect((1, 1, 1), zoom=1.32)
    ax.view_init(elev=20, azim=48)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((1, 1, 1, 0))
        pane.pane.set_edgecolor("#C9D1D9")
    if title:
        ax.set_title(title, fontsize=10.5, color="#141C24", pad=-2)


def _endpoints(ax, s, g):
    ax.scatter(*s, s=44, color="#141C24", depthshade=False, zorder=6)
    ax.scatter(*g, s=44, color="#141C24", marker="*", depthshade=False, zorder=6)


def solve_all(seed, n_waypoints=64):
    rng = np.random.default_rng(seed)
    env = make_random_env(n_spheres=20, n_boxes=20, seed=seed)
    s, g = sample_start_goal(env, rng)
    raw, _ = rrt_connect(env, s, g, rng=rng)
    rrt = resample_path(shortcut(env, raw, rng=rng), n_waypoints)
    out = {"RRT-Connect": rrt}
    for name, fn in (("CHOMP", chomp), ("TrajOpt", trajopt)):
        p, info = fn(env, s, g, n_waypoints=n_waypoints)
        if not info["success"]:
            p, info = fn(env, s, g, n_waypoints=n_waypoints, init_path=rrt)
        out[name] = p
    return env, s, g, out


def fig_planners(seed=11, out="fig_demo_planners.png"):
    """One query, four ways. The prior is what the learned model starts from;
    the three planners are what it is trained to imitate and measured against."""
    env, s, g, paths = solve_all(seed)
    rng = np.random.default_rng(0)
    prior = brownian(s, g, n_waypoints=64, sigma_prior=0.12, n_samples=6, rng=rng)

    fig = plt.figure(figsize=(11.5, 3.15))
    panels = [
        ("Brownian-bridge prior", [(p, C_PRIOR, 1.0, 0.55) for p in prior]),
        ("RRT-Connect + shortcut", [(paths["RRT-Connect"], C_RRT, 2.0, 1.0)]),
        ("CHOMP", [(paths["CHOMP"], C_CHOMP, 2.0, 1.0)]),
        ("TrajOpt", [(paths["TrajOpt"], C_TRAJOPT, 2.0, 1.0)]),
    ]
    for i, (title, curves) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        _draw_env(ax, env)
        for p, c, lw, a in curves:
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=c, linewidth=lw, alpha=a, zorder=5)
        _endpoints(ax, s, g)
        _style(ax, title)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.02, bottom=-0.06, wspace=-0.02)
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")
    for k, p in paths.items():
        print(f"   {k:<22} len {path_length(p):.2f}  min clearance {min_clearance(env, p):+.4f}")


def fig_environments(seeds=(0, 4, 11, 12), out="fig_demo_envs.png"):
    """Four environments with an expert path each -- the point is the clutter.

    Seeds are checked, not trusted: a query whose endpoints happen to land close
    together, or whose refinement collapsed, draws as an unreadable scribble and
    misrepresents the benchmark.
    """
    fig = plt.figure(figsize=(11.5, 3.15))
    for i, sd in enumerate(seeds, 1):
        env, s, g, paths = solve_all(sd)
        d = float(np.linalg.norm(g - s))
        free = env.path_free(paths["CHOMP"])
        print(f"   env {sd:>3}  |g-s| {d:.2f}  CHOMP free {free}")
        assert d >= 1.5 and free, f"seed {sd} is not a usable illustration"
        ax = fig.add_subplot(1, 4, i, projection="3d")
        _draw_env(ax, env)
        p = paths["CHOMP"]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=C_CHOMP, linewidth=2, zorder=5)
        ax.plot(*np.linspace(s, g, 2).T, color="#141C24", linewidth=0.9,
                linestyle=(0, (3, 3)), alpha=0.65, zorder=4)
        _endpoints(ax, s, g)
        _style(ax, f"environment {sd}")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.02, bottom=-0.06, wspace=-0.02)
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_planners()
    fig_environments()
