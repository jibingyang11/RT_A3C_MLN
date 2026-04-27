"""Generate paper figures from outputs/logs/v2/*.csv.

Each figure block is annotated with the EXACT source CSV(s).

Outputs (overwritten):
  outputs/figures/v2/figure2_protocolA_scatter.pdf
  outputs/figures/v2/figure3_group_bars.pdf
  outputs/figures/v2/figure4_protocolB_per_bin.pdf
  outputs/figures/v2/figure5_baselines.pdf
  outputs/figures/v2/figure6_detection_utility.pdf
  outputs/figures/v2/figure7_ablation.pdf
  outputs/figures/v2/figure8_a3c_curve_with_std.pdf
  outputs/figures/v2/figure_runtime_breakdown.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs" / "v2"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 2: Formal-plan diagnostics -- rule-pool size vs attack ratio.
# SOURCE: outputs/logs/v2/protocolA_per_window.csv
# ---------------------------------------------------------------------------
def figure2():
    src = LOG_DIR / "protocolA_per_window.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    color_map = {"balanced": "#5B9BD5", "standard": "#70AD47", "relaxed": "#ED7D31"}
    df["color"] = df["actual_mode_used"].map(lambda m: color_map.get(str(m).split("+")[0], "#999"))
    ax.scatter(df["attack_ratio"], df["num_rules"], c=df["color"], s=80, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Window attack ratio")
    ax.set_ylabel("Candidate rule-pool size")
    ax.set_title("Formal mixed-window plan: rule-pool diagnostics")
    ax.axvline(0.30, ls="--", color="gray", lw=0.8)
    ax.axvline(0.70, ls="--", color="gray", lw=0.8)
    ax.grid(axis="y", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=m)
        for m, c in color_map.items()
    ]
    ax.legend(handles=handles, title="actual mode")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_protocolA_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Rule-pool composition by attack-ratio bin.
# SOURCE: outputs/logs/v2/protocolA_group_stats.csv
# ---------------------------------------------------------------------------
def figure3():
    src = LOG_DIR / "protocolA_per_window.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    if "attack_bin" not in df.columns:
        return

    bins = ["low", "mid", "high"]
    attack_vals = [df[df["attack_bin"] == b]["num_attack_rules"].mean() for b in bins]
    normal_vals = [df[df["attack_bin"] == b]["num_normal_rules"].mean() for b in bins]
    support_vals = [df[df["attack_bin"] == b]["used_support"].mean() for b in bins]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    x = np.arange(len(bins))
    width = 0.36
    axes[0].bar(x - width / 2, attack_vals, width, label="attack rules", color="#5B9BD5", edgecolor="k", lw=0.5)
    axes[0].bar(x + width / 2, normal_vals, width, label="normal rules", color="#F0A35E", edgecolor="k", lw=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bins)
    axes[0].set_ylabel("Mean rules per window")
    axes[0].set_title("Rule-pool composition")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(bins, support_vals, color=["#5B9BD5", "#70AD47", "#ED7D31"], edgecolor="k", lw=0.5)
    axes[1].set_ylim(0, max(0.12, max(support_vals) * 1.2))
    axes[1].set_ylabel("Mean support threshold")
    axes[1].set_title("Support fallback by bin")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Formal-plan rule-pool diagnostics")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_group_bars.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Baseline comparison.
# SOURCE: outputs/logs/v2/baselines_protocolB_high.csv
# ---------------------------------------------------------------------------
def figure4():
    src = LOG_DIR / "baselines_protocolB_high.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    if "method" not in df.columns:
        return

    df = df.set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if "selection_accuracy_mean" in df.columns:
        sa = df["selection_accuracy_mean"]
        axes[0].bar(sa.index, sa.values, color="#5B9BD5", edgecolor="k", lw=0.5)
        axes[0].set_title("Selection accuracy")
        axes[0].set_ylim(0, 1.05)
        for label in axes[0].get_xticklabels():
            label.set_rotation(20)
    if "greedy_total_reward_mean" in df.columns:
        rew = df["greedy_total_reward_mean"]
        axes[1].bar(rew.index, rew.values, color="#70AD47", edgecolor="k", lw=0.5)
        axes[1].set_title("Greedy total reward")
        for label in axes[1].get_xticklabels():
            label.set_rotation(20)

    fig.suptitle("Internal baseline comparison on high-heldout Protocol B test windows")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure5_baselines.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure4_baselines.pdf", bbox_inches="tight")  # backward-compatible alias
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Protocol B high-heldout per-bin rule-selection summary.
# SOURCE: outputs/logs/v2/protocolB_high_seed*_test_per_window.csv
# ---------------------------------------------------------------------------
def figure5():
    curves = sorted(LOG_DIR.glob("protocolB_high_seed*_test_per_window.csv"))
    if not curves:
        return
    dfs = []
    for c in curves:
        try:
            dfs.append(pd.read_csv(c))
        except Exception:
            continue
    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)
    if "stage" not in df.columns or "attack_ratio" not in df.columns:
        return
    df["attack_bin"] = pd.cut(
        df["attack_ratio"],
        bins=[-np.inf, 0.30, 0.70, np.inf],
        labels=["low", "mid", "high"],
        right=False,
    )
    stage_order = ["warmstart", "warmstart_a3c"]
    stage_label = {"warmstart": "WS", "warmstart_a3c": "WS+A3C"}
    metrics = [
        ("attack_keep_rate", "Attack keep"),
        ("normal_disable_rate", "Normal disable"),
        ("selection_accuracy", "Selection acc."),
    ]
    bins = ["low", "mid", "high"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    x = np.arange(len(bins))
    width = 0.34
    colors = {"warmstart": "#7DA7D9", "warmstart_a3c": "#F0A35E"}
    for ax, (col, title) in zip(axes, metrics):
        for offset, stage in [(-width / 2, "warmstart"), (width / 2, "warmstart_a3c")]:
            vals, errs = [], []
            for b in bins:
                g = df[(df["stage"] == stage) & (df["attack_bin"].astype(str) == b)][col]
                vals.append(float(g.mean()) if not g.empty else np.nan)
                errs.append(float(g.std()) if len(g) > 1 else 0.0)
            ax.bar(
                x + offset,
                vals,
                width,
                yerr=errs,
                capsize=3,
                label=stage_label[stage],
                color=colors[stage],
                edgecolor="k",
                linewidth=0.5,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean +/- std over held-out windows")
    axes[-1].legend(loc="lower right")
    fig.suptitle("High-heldout Protocol B by attack-ratio bin")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure4_protocolB_per_bin.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure5_protocolB_per_bin.pdf", bbox_inches="tight")  # backward-compatible alias
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Detection utility calibration.
# SOURCE: outputs/logs/v2/detection_utility_high.csv
# ---------------------------------------------------------------------------
def figure6():
    src = LOG_DIR / "detection_utility_high.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    if not {"mean", "std"}.issubset(df.columns):
        return
    key_col = df.columns[0]
    metrics = ["P", "R", "F1", "AUROC"]
    labels = ["Precision", "Recall", "F1", "AUROC"]
    series = {
        "All kept": [f"all_kept_{m}" for m in metrics],
        "Selected": [f"ours_{m}" for m in metrics],
    }
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for offset, (name, keys) in [(-width / 2, ("All kept", series["All kept"])), (width / 2, ("Selected", series["Selected"]))]:
        vals, errs = [], []
        for k in keys:
            row = df[df[key_col] == k]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["std"].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            capsize=3,
            label=name,
            edgecolor="k",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Transaction-level score")
    ax.set_title("Detection utility calibration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure6_detection_utility.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Ablation on adaptive vs forced rule-pool modes.
# SOURCE: outputs/logs/v2/ablation_modes.csv
# ---------------------------------------------------------------------------
def figure7():
    src = LOG_DIR / "ablation_modes.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    if "forced_mode" not in df.columns:
        return
    bins = [c for c in ["low", "mid", "high"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.18
    x = np.arange(len(bins))
    for i, mode in enumerate(df["forced_mode"]):
        vals = [df[df["forced_mode"] == mode][b].iloc[0] for b in bins]
        ax.bar(x + i * width, vals, width, label=mode, edgecolor="k", lw=0.5)
    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(bins)
    ax.set_ylabel("Mean selection accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Adaptive vs forced rule-pool modes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure7_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: A3C training curve with seed-level std band.
# SOURCE: outputs/logs/v2/protocolB_high_seed*_a3c_training_curve.csv
# ---------------------------------------------------------------------------
def figure8():
    curves = sorted(LOG_DIR.glob("protocolB_high_seed*_a3c_training_curve.csv"))
    if not curves:
        return
    dfs = []
    for c in curves:
        try:
            d = pd.read_csv(c)
            d["seed_file"] = c.stem
            dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return
    full = pd.concat(dfs, ignore_index=True)
    if "episode" not in full.columns:
        return

    # average across worker for each (seed, episode)
    by_ep = full.groupby(["seed", "episode"])["total_reward"].mean().reset_index()
    g = by_ep.groupby("episode")["total_reward"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(g["episode"], g["mean"], color="#5B9BD5", lw=1.5, label="mean across seeds")
    ax.fill_between(
        g["episode"],
        g["mean"] - g["std"].fillna(0.0),
        g["mean"] + g["std"].fillna(0.0),
        color="#5B9BD5",
        alpha=0.25,
        label="+/-1 std",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward (averaged across workers)")
    ax.set_title("A3C training reward curve (high-heldout Protocol B fine-tuning)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure8_a3c_curve_with_std.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Runtime figure: train-time vs inference-latency on the same plot
# but with separate y-axes (so the units are honestly displayed).
# SOURCE: outputs/logs/v2/runtime_by_method_high.csv
# ---------------------------------------------------------------------------
def figure_runtime():
    src = LOG_DIR / "runtime_by_method_high.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    if "method" not in df.columns:
        return
    df = df.sort_values("inference_ms_per_window_mean")

    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = np.arange(len(df))
    bar1 = ax1.bar(
        x - 0.2,
        df["inference_ms_per_window_mean"].values,
        0.4,
        label="Inference ms/window",
        color="#5B9BD5",
        edgecolor="k",
        lw=0.5,
    )
    ax1.set_ylabel("Inference (ms / window)", color="#5B9BD5")
    ax1.tick_params(axis="y", labelcolor="#5B9BD5")

    ax2 = ax1.twinx()
    bar2 = ax2.bar(
        x + 0.2,
        df["total_train_sec"].values,
        0.4,
        label="Total train sec",
        color="#ED7D31",
        edgecolor="k",
        lw=0.5,
    )
    ax2.set_ylabel("Training (seconds)", color="#ED7D31")
    ax2.tick_params(axis="y", labelcolor="#ED7D31")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["method"].values, rotation=30, ha="right")
    fig.suptitle("Runtime breakdown: training vs inference (separate axes)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_runtime_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    figure_runtime()
    print(f"All figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
