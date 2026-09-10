
  #.venv/bin/python make_figures.py

  #Regenerate the report figures from the measured numbers. Kept as a script so
  #the figures track the results rather than drifting from them.

  #Source: sweep_steps.py on held-out envs 250-299 (50 envs x 10 pairs x
  #20 samples, 8 Euler steps, K_FA=1). See main.tex Section 4 / paper.tex Section VI.

import matplotlib
matplotlib.use("Agg")
#IEEE PDF eXpress REJECTS Type 3 fonts, and matplotlib's PDF backend emits them
#by default: every figure came back flagged on pages 4, 8 and 9. fonttype 42 is
#TrueType, which is accepted and is also selectable text rather than rasterised
#outlines. Set before pyplot draws anything.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ENVS = [20, 60, 150, 250]
# Every CONTROL and TREATMENT cell is a mean over three seeds; the seed spreads
# are carried alongside so the figure can show them, because a scaling curve
# drawn without error bars from data that HAS them is a figure that overstates
# its own precision. AUGMENTED and the no-roll ablation remain single seed and
# are drawn without bars, which is the honest way to show that difference.
CONTROL = [12.86, 13.57, 14.75, 15.37]     # world frame, 3-seed means
CONTROL_SE = [0.048, 0.052, 0.204, 0.364]
AUGMENTED = [12.72, 13.84, 15.25, 17.50]   # world frame + random SE(3) aug., n=1
TREATMENT = [19.23, 34.43, 40.60, 45.45]   # (s,g) reduction, 3-seed means
TREATMENT_SE = [0.185, 1.557, 0.314, 0.111]
NOROLL_X, NOROLL_Y = 60, 33.53          # reduction without roll augmentation, 3 seeds

# mechanism decomposition, percentage points on the held-out collision-free rate
# Converged values, matching the headline setting, EXCEPT the two DOF sub-bars
# and the SE(3) augmentation arm, which were only ever measured on the 20-epoch
# grid and are labelled as such. Mixing budgets silently is how the +0.05 for the
# equivariant backbone once sat beside a 20-epoch +0.1 for frame averaging.
MECHANISMS = [
    (r"$(s,g)$ reduction" "\n" "(5 DOF, exact)", 36.5),
    (r"  of which: 3 translations (20 ep.)", 12.3),
    (r"  of which: 2 rotations (20 ep.)", 18.0),
    ("SE(3) augmentation\n(world frame, 20 ep.)", 2.1),
    ("roll augmentation\n(training)", 0.8),
    ("frame averaging\n($K{=}1\\to3$)", 0.68),
    # the third mechanism, measured 2026-08-18: seed-matched at convergence the
    # constrained architecture is worth +0.05 at the CEILING. Its real value is
    # a ~4x saving in optimisation, which this bar chart cannot show -- the
    # caption has to say so, or the figure understates the result.
    ("$\\mathrm{SO}(2)$-equivariant\nbackbone (weights)", 0.15),
]
# across-SEED standard error at 60 envs, treatment arm (n=3). This is the right
# floor for a claim about a method; the old 2.0 came from the spread over
# held-out problems, which answers a different question.
SE = 1.56          # across seeds, treatment arm at 60 envs (n=3)
SE_PAIRED = 0.32   # seed-matched, for ablations that share a seed with their base

# non-equivariance residual, RMS (Hilbert) definition of Eq. 6, K=3.
# Flat across a 12.5x range of training data -- the earlier "falls with scale"
# reading came from two points under the old mean-of-norms definition.
RESID_ENVS = [20, 60, 150, 250]
RESID = [0.0183, 0.0170, 0.0177, 0.0168]
RESID_NOROLL = (60, 0.0279)

# --- convergence curves for the constrained architecture -------------------
# Held-out collision-free % every 10 epochs at 60 environments, seed 0. A bar of
# ceiling gains cannot show what this result actually is: the two arms end in the
# same place, and the constrained one gets there roughly 4x sooner. That is the
# whole finding, so it needs a curve rather than a number.
CONV_EPOCHS = list(range(10, 301, 10))
CONV_TREAT = [22.2, 31.3, 35.5, 37.9, 40.5, 42.6, 43.8, 44.5, 45.6, 45.8,
              46.8, 47.4, 47.7, 48.1, 48.2, 48.3, 48.9, 49.2, 49.3, 49.3,
              49.5, 49.6, 50.0, 50.0, 50.1, 50.4, 50.6, 50.6, 50.5, 50.8]
CONV_EQUIV = [35.8, 40.1, 42.7, 44.6, 45.8, 46.7, 47.9, 48.1, 48.4, 48.7,
              48.9, 49.2, 49.4, 49.6, 49.5, 49.8, 49.8, 49.6, 49.7, 50.0,
              50.3, 50.1, 50.4, 50.6, 50.6, 50.7, 50.8, 50.9, 50.8, 50.9]
# the epochs each arm first reaches the other's epoch-10 and epoch-40 marks
CONV_MARKS = [(10, 40), (40, 90)]


# classical reference points, measured by baselines_classical.py on the SAME
# 500 held-out problems (envs 250-299 x 10 pairs). Times are single-core CPU.
CLASSICAL = [
    ("RRT-Connect", 99.6, 0.284),
    ("expert (RRT+CHOMP)", 100.0, 0.342),
    ("CHOMP", 88.8, 0.394),
    ("TrajOpt", 74.0, 0.352),
]
STRAIGHT_LINE = 15.6           # trivial baseline: the segment start -> goal
# learned arms at 250 training envs, 8 Euler steps, 20 samples/query on one GPU
# The learned bars must match Table 1, which calibrates against the classical
# planners using the CONVERGED 60-environment arms (the paper's headline
# setting), not the 20-epoch scaling grid. The intro used to quote one and point
# at the other.
LEARNED = [
    (r"world frame", 14.58, 0.053),
    (r"world frame + aug.", 17.50, 0.067),
    (r"$(s,g)$ reduction", 51.12, 0.063),
]

C_CTRL = "#3E6DA8"
C_TREAT = "#C2560F"

# --- the cascade, at the HEADLINE setting ---------------------------------
# 60 environments, trained to convergence, three seeds. The earlier version ran
# 15.4 -> 45.5 at 250 environments and a fixed 20-epoch budget, which is the
# scaling grid, not the headline. A teaser is the first quantitative thing anyone
# reads, and it disagreed with the abstract.
#
# The DOF split (+12.3 translations, +18.0 rotations) is NOT here: it was only
# measured on the 20-epoch grid. It lives in the mechanism table, labelled.
CASCADE = [
    ("world frame",                 14.60, C_CTRL),
    ("+ query-conditioned encoder", 15.59, "#7A93B8"),
    ("+ $(s,g)$ reduction",         51.10, C_TREAT),
]
CASCADE_DELTAS = ["$+1.0$", "$+35.5$"]
CASCADE_FLOOR = 15.6   # straight line from start to goal, same 500 problems

# The three mechanisms for the residual roll. ALTERNATIVES, not a decomposition:
# The starred row is the 20-epoch grid: roll augmentation was never run to
# convergence, and drawing it unmarked beside two converged numbers under a
# "trained to convergence" heading claims a measurement that was not made.
# each was measured against its own base and they were never composed. Drawing
# them end to end would assert an additivity that was never tested.
ROLL_MECHANISMS = [
    ("roll augmentation (data)$^{*}$", 0.80),
    ("frame averaging (operator)",     0.68),
    ("equivariant backbone (weights)", 0.15),
]

INK = "#333333"
MUTED = "#777777"


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        if side in keep:
            ax.spines[side].set_color("0.6")
        else:
            ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def fig_scaling(out="fig_scaling.pdf", size=(6.4, 3.9), fs=9.0):
    """size/fs are set so the figure is placed at 1:1 -- report figures are wide
    (single column, 0.82\\textwidth), paper figures are one column of two."""
    fig, ax = plt.subplots(figsize=size)

    # the gap between the arms is the result -- shade it
    ax.fill_between(ENVS, CONTROL, TREATMENT, color=C_TREAT, alpha=0.08, linewidth=0)

    #capsize=0: at this scale the bars are smaller than the markers on three of
    #four points, and caps would read as data rather than as uncertainty.
    ax.errorbar(ENVS, TREATMENT, yerr=TREATMENT_SE, fmt="-o", color=C_TREAT,
                linewidth=2, markersize=6, capsize=0, elinewidth=1.4,
                label=r"$(s,g)$ reduction", zorder=3)
    ax.errorbar(ENVS, CONTROL, yerr=CONTROL_SE, fmt="-s", color=C_CTRL,
                linewidth=2, markersize=5.5, capsize=0, elinewidth=1.4,
                label="World frame", zorder=3)
    #Augmentation is the same arm using the symmetry a different way, so it
    #keeps the control's hue and is distinguished by a dashed stroke.
    ax.plot(ENVS, AUGMENTED, "--^", color=C_CTRL, linewidth=1.6, markersize=5,
            markerfacecolor="white", label="World frame + SE(3) aug.", zorder=3)

    # the no-roll ablation is the same arm with one part removed: same hue, hollow
    ax.plot([NOROLL_X], [NOROLL_Y], "o", markerfacecolor="white",
            markeredgecolor=C_TREAT, markeredgewidth=1.6, markersize=7, zorder=4)
    ax.annotate("no-roll ablation", xy=(NOROLL_X, NOROLL_Y), xytext=(9, -9),
                textcoords="offset points", fontsize=fs - 1.0, color="#555555")

    for x, y, c, dy in ((250, TREATMENT[-1], C_TREAT, 6),
                        (250, AUGMENTED[-1], C_CTRL, 5),
                        (250, CONTROL[-1], C_CTRL, -12)):
        ax.annotate(f"{y:.1f}%", xy=(x, y), xytext=(-4, dy),
                    textcoords="offset points", fontsize=fs, color=c,
                    fontweight="bold", ha="right")

    ax.set_xscale("log")
    ax.set_xticks(ENVS)
    ax.set_xticklabels([str(e) for e in ENVS])
    ax.minorticks_off()
    ax.set_xlim(17, 320)
    ax.set_ylim(0, 50)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Training environments (log scale)", fontsize=fs + 0.5)
    ax.set_ylabel("Collision-free rate (%)", fontsize=fs + 0.5)
    ax.tick_params(labelsize=fs)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    _despine(ax)
    ax.legend(frameon=False, fontsize=fs, loc="upper left")

    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    print(f"wrote {out}")


def fig_mechanisms(out="fig_mechanisms.pdf"):
    """Left: what each mechanism is worth, against the noise floor.
    Right: the quantity frame averaging exists to remove, versus scale."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.9, 2.9),
                                   gridspec_kw={"width_ratios": [1.55, 1.0]})

    # -- left: one series, so no legend; direct labels on every bar --------
    labels = [m[0] for m in MECHANISMS][::-1]
    vals = [m[1] for m in MECHANISMS][::-1]
    ypos = range(len(vals))

    axl.axvspan(-2 * SE, 2 * SE, color="0.86", alpha=0.55, linewidth=0, zorder=0)
    axl.annotate("$\\pm2$ SE\n(across seeds)", xy=(2 * SE, -0.45), xytext=(5, 0),
                 textcoords="offset points", fontsize=7, color=MUTED, va="center")
    axl.barh(list(ypos), vals, height=0.52, color=C_TREAT, zorder=2)
    for y, v in zip(ypos, vals):
        axl.annotate(f"+{v:.1f} pp", xy=(v, y), xytext=(7, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=INK, fontweight="bold")

    axl.set_yticks(list(ypos))
    axl.set_yticklabels(labels, fontsize=8)
    axl.set_xlim(-5, 38)
    axl.set_xticks([0, 10, 20, 30])
    axl.set_ylim(-0.95, 5.6)
    axl.set_xlabel("Contribution to collision-free rate (pp)", fontsize=8.5)
    axl.tick_params(labelsize=8)
    axl.grid(axis="x", color="0.9", linewidth=0.7)
    _despine(axl, keep=("bottom",))
    axl.tick_params(axis="y", length=0)
    axl.set_title("(a)  What each mechanism is worth", fontsize=9, loc="left",
                  color=INK, pad=6)

    # -- right: the residual falls with scale ------------------------------
    axr.plot(RESID_ENVS, RESID, "-o", color=C_TREAT, linewidth=2, markersize=6,
             zorder=3)
    axr.plot([RESID_NOROLL[0]], [RESID_NOROLL[1]], "o", markerfacecolor="white",
             markeredgecolor=C_TREAT, markeredgewidth=1.6, markersize=7, zorder=4)
    axr.annotate("no roll aug.", xy=RESID_NOROLL, xytext=(8, -2),
                 textcoords="offset points", fontsize=7.5, color=MUTED)
    for x, y in zip(RESID_ENVS, RESID):
        axr.annotate(f"{100*y:.2f}%", xy=(x, y), xytext=(0, 9),
                     textcoords="offset points", fontsize=8, color=C_TREAT,
                     ha="center", fontweight="bold")

    axr.set_xscale("log")
    axr.set_xticks(RESID_ENVS)
    axr.set_xticklabels([str(e) for e in RESID_ENVS])
    axr.minorticks_off()
    axr.set_xlim(15, 400)
    axr.set_ylim(0, 0.034)
    axr.set_yticks([0, 0.01, 0.02, 0.03])
    axr.set_yticklabels(["0", "1%", "2%", "3%"])
    axr.set_xlabel("Training environments", fontsize=8.5)
    axr.set_ylabel("Non-equivariance residual $r$", fontsize=8.5)
    axr.tick_params(labelsize=8)
    axr.grid(axis="y", color="0.9", linewidth=0.7)
    _despine(axr)
    axr.set_title("(b)  The residual is flat", fontsize=9, loc="left",
                  color=INK, pad=6)

    fig.tight_layout(pad=0.4, w_pad=2.2)
    fig.savefig(out)
    print(f"wrote {out}")


def fig_baselines(out="fig_baselines.pdf"):
    """Where the learned planner actually sits. Success against wall-clock,
    with the straight-line floor drawn as a reference: the world-frame arm is
    on it, which is the finding, and the classical planners are far above."""
    fig, ax = plt.subplots(figsize=(3.35, 2.75))

    #the trivial baseline is a threshold, not a competitor -- draw it as a rule
    ax.axhline(STRAIGHT_LINE, color=MUTED, linewidth=1, linestyle=(0, (4, 3)),
               zorder=1)
    ax.annotate(f"straight line ({STRAIGHT_LINE}%)", xy=(1.35, STRAIGHT_LINE),
                xytext=(0, -9), textcoords="offset points", fontsize=6.5,
                color=MUTED, ha="right")

    #hand-placed so no label touches another; the points are too clustered in
    #x for any single rule to work
    PLACE = {
        "RRT-Connect":        (-9, 3, "right", "bottom"),
        "expert (RRT+CHOMP)": (9, 2, "left", "bottom"),
        "CHOMP":              (9, 0, "left", "center"),
        "TrajOpt":            (9, 0, "left", "center"),
        "world frame":        (9, -6, "left", "top"),
        "world frame + aug.": (9, 3, "left", "bottom"),
        r"$(s,g)$ reduction": (9, 0, "left", "center"),
    }

    xs = [t for _, _, t in CLASSICAL]
    ys = [s for _, s, _ in CLASSICAL]
    ax.plot(xs, ys, "s", color=C_CTRL, markersize=6, zorder=3,
            label="classical (1 CPU core)")
    xs = [t for _, _, t in LEARNED]
    ys = [s for _, s, _ in LEARNED]
    ax.plot(xs, ys, "o", color=C_TREAT, markersize=7, zorder=3,
            label="learned (1 GPU, 20 samples)")

    for name, s, t in CLASSICAL + LEARNED:
        dx, dy, ha, va = PLACE[name]
        ax.annotate(name, xy=(t, s), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color=INK, ha=ha, va=va)

    #the gap the paper closes, and the gap it does not
    ax.annotate("", xy=(0.063, 51.12), xytext=(0.053, 14.58),
                arrowprops=dict(arrowstyle="<->", color=C_TREAT, linewidth=1.1,
                                shrinkA=3, shrinkB=3))
    ax.annotate("+36.5", xy=(0.058, 33), xytext=(-4, 0),
                textcoords="offset points", fontsize=7, color=C_TREAT,
                ha="right", va="center", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(0.035, 2.6)
    ax.set_xticks([0.05, 0.1, 0.3, 1.0])
    ax.set_xticklabels(["0.05", "0.1", "0.3", "1.0"])
    ax.minorticks_off()
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Wall-clock per query (s, log scale)", fontsize=7.5)
    ax.set_ylabel("Collision-free rate (%)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    _despine(ax)
    #empty mid-left region; anywhere lower collides with the world-frame point
    ax.legend(frameon=False, fontsize=6.5, loc="center left", handletextpad=0.4,
              bbox_to_anchor=(-0.02, 0.66))

    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    print(f"wrote {out}")


def fig_convergence(path="fig_convergence.pdf"):
    #Two convergence curves, plus the horizontal read-off that makes the
    #efficiency claim visible: the constrained arm hits at epoch 10 what the
    #unconstrained one needs epoch 40 to reach.
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(CONV_EPOCHS, CONV_TREAT, color=C_CTRL, lw=1.6,
            label="unconstrained")
    ax.plot(CONV_EPOCHS, CONV_EQUIV, color=C_TREAT, lw=1.6,
            label="$\\mathrm{SO}(2)$-equivariant")
    for a, b in CONV_MARKS:
        y = CONV_EQUIV[a // 10 - 1]
        ax.plot([a, b], [y, y], color="0.55", lw=0.7, ls=":", zorder=0)
        ax.plot([a], [y], "o", ms=3, color=C_TREAT, zorder=3)
        ax.plot([b], [y], "o", ms=3, mfc="none", color=C_CTRL, zorder=3)
    ax.set_xlabel("Training epochs", fontsize=7.5)
    ax.set_ylabel("Collision-free rate (%)", fontsize=7.5)
    ax.set_xlim(0, 305)
    ax.set_ylim(0, 55)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")





def fig_scale_conv(path="fig_scale_conv.pdf"):
    #Figs. 2 and 3 as one column-width float. They make the same point from two
    #axes -- the reduction scales, and it optimises faster -- so one caption
    #serves both, and the page budget does not stretch to two floats.
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(3.4, 1.65))
    fs = 6.0

    axl.fill_between(ENVS, CONTROL, TREATMENT, color=C_TREAT, alpha=0.08, lw=0)
    axl.errorbar(ENVS, TREATMENT, yerr=TREATMENT_SE, fmt="-o", color=C_TREAT,
                 lw=1.3, ms=3, capsize=0, elinewidth=0.9, zorder=3,
                 label=r"$(s,g)$ reduction")
    axl.errorbar(ENVS, CONTROL, yerr=CONTROL_SE, fmt="-s", color=C_CTRL,
                 lw=1.3, ms=2.8, capsize=0, elinewidth=0.9, zorder=3,
                 label="world frame")
    axl.plot(ENVS, AUGMENTED, "--^", color=C_CTRL, lw=1.0, ms=2.8,
             markerfacecolor="white", zorder=3, label=r"$+\,\mathrm{SE}(3)$ aug.")
    axl.set_xscale("log"); axl.set_xticks(ENVS)
    axl.set_xticklabels([str(e) for e in ENVS]); axl.minorticks_off()
    axl.set_xlim(17, 320); axl.set_ylim(0, 50)
    axl.set_yticks([0, 10, 20, 30, 40, 50])
    axl.set_xlabel("Training environments", fontsize=fs)
    axl.set_ylabel("Collision-free rate (%)", fontsize=fs)
    axl.tick_params(labelsize=fs - 0.5)
    axl.grid(axis="y", color="0.88", lw=0.5)
    axl.legend(frameon=False, fontsize=fs - 0.6, loc="upper left")
    _despine(axl)
    axl.set_title("(a) scale", fontsize=fs + 0.8, loc="left", color=INK)

    axr.plot(CONV_EPOCHS, CONV_TREAT, color=C_CTRL, lw=1.2,
             label="unconstrained")
    axr.plot(CONV_EPOCHS, CONV_EQUIV, color=C_TREAT, lw=1.2,
             label="$\\mathrm{SO}(2)$-equiv.")
    for a, b in CONV_MARKS:
        y = CONV_EQUIV[a // 10 - 1]
        axr.plot([a, b], [y, y], color="0.55", lw=0.6, ls=":", zorder=0)
        axr.plot([a], [y], "o", ms=2.4, color=C_TREAT, zorder=3)
        axr.plot([b], [y], "o", ms=2.4, mfc="none", color=C_CTRL, zorder=3)
    axr.set_xlabel("Training epochs", fontsize=fs)
    axr.set_xlim(0, 305); axr.set_ylim(0, 55)
    axr.set_xticks([0, 100, 200, 300])
    axr.tick_params(labelsize=fs - 0.5)
    axr.legend(frameon=False, loc="lower right", fontsize=fs - 0.6)
    axr.spines[["top", "right"]].set_visible(False)
    axr.set_title("(b) budget", fontsize=fs + 0.8, loc="left", color=INK)

    fig.tight_layout(pad=0.25, w_pad=1.0)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig_cascade(path="fig_cascade.pdf"):
    #The paper's argument in one image, entirely at the headline setting.
    #Top: what the query frame is worth, with the query-conditioned control in
    #between, so a reader sees that conditioning alone does not do it.
    #Bottom: the three mechanisms for the leftover roll, all drawn from a common
    #baseline as ALTERNATIVES rather than stacked end to end.
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(6.4, 3.2),
        gridspec_kw=dict(height_ratios=[3, 1.5], hspace=0.75))

    ys = list(range(len(CASCADE)))[::-1]
    for y, (label, val, colour) in zip(ys, CASCADE):
        ax.barh(y, val, height=0.58, color=colour, zorder=2)
        ax.text(val + 0.8, y, f"{val:.2f}", va="center", ha="left",
                fontsize=8.5, fontweight="bold", color="0.15", zorder=3)
    ax.axvline(CASCADE_FLOOR, color="0.35", lw=0.9, ls="--", zorder=1)
    ax.text(CASCADE_FLOOR + 0.7, len(CASCADE) - 0.45,
            "straight line, 15.6", fontsize=7.5, color="0.35", va="bottom")
    for i, d in enumerate(CASCADE_DELTAS):
        y0, y1 = len(CASCADE) - 1 - i, len(CASCADE) - 2 - i
        x0, x1 = CASCADE[i][1], CASCADE[i + 1][1]
        ax.annotate("", xy=(x1, y1 + 0.32), xytext=(x0, y0 - 0.32),
                    arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.8,
                                    shrinkA=0, shrinkB=0))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, d, fontsize=8.5, color="0.25",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in CASCADE], fontsize=8)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Collision-free rate (%), held-out environments", fontsize=8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    my = list(range(len(ROLL_MECHANISMS)))[::-1]
    for y, (label, val) in zip(my, ROLL_MECHANISMS):
        bx.barh(y, val, height=0.5, color="#9AA3AE", zorder=2)
        bx.text(val + 0.03, y, f"$+{val:.2f}$", va="center", ha="left",
                fontsize=8, fontweight="bold", color="0.15")
    bx.set_yticks(my)
    bx.set_yticklabels([m[0] for m in ROLL_MECHANISMS], fontsize=8)
    bx.set_xlim(0, 1.35)
    bx.set_xlabel("Added to the reduced model; each measured separately, not composed",
                  fontsize=8)
    bx.tick_params(axis="x", labelsize=7.5)
    bx.spines[["top", "right", "left"]].set_visible(False)
    bx.tick_params(axis="y", length=0)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    fig_scaling()                                        # main.tex, 0.82\textwidth
    fig_scaling("fig_scaling_col.pdf", size=(3.35, 2.45), fs=7.0)   # paper.tex, one column
    fig_scale_conv()   # ICRA.tex: Figs 2+3 merged into one float
    fig_mechanisms()                                     # paper.tex, figure* (both columns)
    fig_baselines()                                      # paper.tex, one column
    fig_convergence()                                    # paper.tex, one column
    fig_cascade()                                        # paper.tex, figure* (both columns)
