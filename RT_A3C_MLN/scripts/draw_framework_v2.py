from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.patheffects as pe


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper_artifacts" / "figures"


def box(ax, x, y, w, h, text, fc, ec="#2b4057", lw=1.5, fs=10.5, weight="normal", z=3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    patch.set_path_effects([pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.12), pe.Normal()])
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
        color="#111827",
        linespacing=1.15,
        zorder=z + 1,
    )
    return patch


def chip(ax, x, y, w, h, text, fc="#ffffff", ec="#56657a", fs=8.4):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
        zorder=5,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#1f2937", zorder=6)
    return patch


def arrow(ax, start, end, text=None, color="#2b5068", lw=1.8, style="solid", rad=0.0, fs=8.8, offset=0.014):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        zorder=2,
    )
    ax.add_patch(patch)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + offset
        ax.text(
            mx,
            my,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88),
            zorder=7,
        )
    return patch


def draw_framework():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(17.0, 7.2), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    navy = "#2d4057"
    blue = "#d9ecff"
    green = "#dff2df"
    yellow = "#fff0bd"
    purple = "#eee8ff"
    red = "#fde0dc"
    tan = "#f4dfc8"
    gray = "#eef1f5"

    # Two clean lanes.
    ax.add_patch(Rectangle((0.03, 0.56), 0.94, 0.29, facecolor="#eef7ff", edgecolor="#b9d6ef", lw=1.0, zorder=0))
    ax.add_patch(Rectangle((0.03, 0.14), 0.94, 0.35, facecolor="#f4fbf3", edgecolor="#c6dec1", lw=1.0, zorder=0))
    ax.add_patch(Rectangle((0.03, 0.815), 0.94, 0.035, facecolor="#d8ebfb", edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((0.03, 0.455), 0.94, 0.035, facecolor="#e1f1df", edgecolor="none", zorder=1))
    ax.text(0.045, 0.832, "Foreground: realtime probabilistic inference", fontsize=11.5, fontweight="bold", color="#1d4f74", zorder=2)
    ax.text(0.045, 0.472, "Background: asynchronous A3C rule-structure refresh", fontsize=11.5, fontweight="bold", color="#357240", zorder=2)

    ax.text(
        0.5,
        0.935,
        "RT-A3C-MLN realtime adaptive rule-learning framework",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )

    # Foreground path.
    box(ax, 0.055, 0.645, 0.145, 0.100, "Streaming\nchunk $x_t$", green, fs=11.0, weight="bold")
    box(ax, 0.255, 0.635, 0.180, 0.120, "Realtime MLN inference\n$P(y\\mid x_t,\\mathcal{R}_t,w_t)$", blue, fs=10.2)
    box(
        ax,
        0.505,
        0.615,
        0.210,
        0.150,
        "Accepted MLN rule memory\n$\\mathcal{R}_t, w_t, b_t$\nactive / deleted / emerging rules",
        gray,
        fs=9.8,
        weight="bold",
    )
    box(ax, 0.800, 0.645, 0.145, 0.100, "Realtime output\nprobability + label", blue, fs=10.5)
    chip(ax, 0.815, 0.592, 0.115, 0.038, "blocking cycle < 10 ms", fc="#ffffff", ec="#b9d6ef", fs=8.3)

    arrow(ax, (0.200, 0.695), (0.255, 0.695), "state")
    arrow(ax, (0.435, 0.695), (0.505, 0.690), "latest rules")
    arrow(ax, (0.715, 0.690), (0.800, 0.695), "low blocking")

    # Background rule search.
    box(ax, 0.055, 0.345, 0.145, 0.080, "Concept-lattice\nrule pool", green, fs=9.8, weight="bold")
    box(ax, 0.055, 0.215, 0.145, 0.095, "Arriving labels\nrecent buffer", yellow, fs=10.0)

    box(ax, 0.250, 0.345, 0.180, 0.080, "Partitioned concept boxes", "#e8f4ea", fs=9.7, weight="bold")
    chip(ax, 0.272, 0.363, 0.044, 0.030, "$B_1$", fs=8.0)
    chip(ax, 0.322, 0.363, 0.044, 0.030, "$B_2$", fs=8.0)
    chip(ax, 0.372, 0.363, 0.044, 0.030, "$B_K$", fs=8.0)
    box(ax, 0.250, 0.205, 0.180, 0.105, "Reward / return\n$-\\mathrm{logloss}-\\lambda|\\mathcal{R}|$\n$G_t$", yellow, fs=9.4)

    actor = box(ax, 0.495, 0.215, 0.205, 0.205, "", purple, ec="#57508b")
    ax.text(0.598, 0.392, "Parallel actor-learners", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#2f2c63", zorder=6)
    chip(ax, 0.525, 0.335, 0.145, 0.032, "Actor 1 searches $B_1$", ec="#7069a8", fs=8.0)
    chip(ax, 0.525, 0.290, 0.145, 0.032, "Actor 2 searches $B_2$", ec="#7069a8", fs=8.0)
    chip(ax, 0.525, 0.245, 0.145, 0.032, "Actor K searches $B_K$", ec="#7069a8", fs=8.0)
    ax.text(0.598, 0.197, "actions: add / delete / replace / $\\Delta w$", ha="center", va="center", fontsize=8.8, color="#2f2c63")

    box(ax, 0.770, 0.335, 0.145, 0.090, "Global Critic\n$A_t=G_t-V(s_t)$", red, ec="#8b4b47", fs=10.0, weight="bold")
    box(ax, 0.770, 0.210, 0.145, 0.080, "Structure gate\naccept / trim / merge", tan, ec="#8a5f3f", fs=9.7)

    arrow(ax, (0.200, 0.385), (0.250, 0.385), "candidates")
    arrow(ax, (0.200, 0.260), (0.250, 0.257), "feedback")
    arrow(ax, (0.430, 0.385), (0.495, 0.350), "local scope")
    arrow(ax, (0.430, 0.257), (0.495, 0.262), "reward")
    arrow(ax, (0.700, 0.355), (0.770, 0.380), "async gradients", style="dashed", lw=2.0)
    arrow(ax, (0.842, 0.335), (0.842, 0.290), "value update", style="dashed", fs=8.0, offset=0.002)
    arrow(ax, (0.842, 0.290), (0.716, 0.615), "accepted refresh", style="dashed", lw=2.0, rad=-0.18, fs=8.5, offset=0.020)

    # Current state provided to the asynchronous search without crossing the whole figure.
    arrow(ax, (0.565, 0.615), (0.430, 0.425), "current MLN state", style="dashed", lw=1.6, rad=0.08, fs=8.3)

    # Evidence strip.
    chip(ax, 0.075, 0.070, 0.235, 0.044, "Experiment evidence: low blocking cycle", fc="#f9fafb", ec=navy, fs=9.4)
    chip(ax, 0.385, 0.070, 0.230, 0.044, "accepted structure refresh", fc="#f9fafb", ec=navy, fs=9.4)
    chip(ax, 0.690, 0.070, 0.235, 0.044, "emerging-rule activation", fc="#f9fafb", ec=navy, fs=9.4)

    fig.savefig(FIG_DIR / "Fig1_Method_Framework.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig1_Method_Framework.png", bbox_inches="tight")

    mirror_dirs = [
        ROOT / "kbs_full_draft" / "els-cas-templates" / "figs",
        Path(r"C:\Users\Administrator\Desktop\temp\我的论文-MLN\RT-A3C-MLN\figs"),
    ]
    for out_dir in mirror_dirs:
        if out_dir.exists():
            fig.savefig(out_dir / "Fig1_Method_Framework.pdf", bbox_inches="tight")
            fig.savefig(out_dir / "Fig1_Method_Framework.png", bbox_inches="tight")

    root_png = Path(r"C:\Users\Administrator\Desktop\temp\我的论文-MLN\Fig1_Method_Framework.png")
    if root_png.parent.exists():
        fig.savefig(root_png, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    draw_framework()
