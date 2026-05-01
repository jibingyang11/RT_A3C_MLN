from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "core_figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "RT_A3C_Realtime_MLN": "#D55E00",
    "OnlineWeight_MLN": "#666666",
    "Rolling_MaxEnt_MLN": "#0072B2",
    "Rolling_L1_MLN": "#CC79A7",
    "Rolling_Boosted_MLN": "#56B4E9",
    "Rolling_BeamSearch_MLN": "#E69F00",
    "Static_MaxEnt_Reference": "#BBBBBB",
    "Full_Realtime_A3C": "#D55E00",
    "Single_Actor": "#E69F00",
    "No_Prior_Shift": "#999999",
    "No_Fast_Search": "#0072B2",
    "Background_Refresh": "#009E73",
    "Online_Only_No_A3C": "#666666",
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def bar(frame, label_col, metric, title, ylabel, name, log=False, ascending=False):
    frame = frame.sort_values(metric, ascending=ascending)
    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    labels = frame[label_col].astype(str)
    ax.bar(labels, frame[metric].astype(float), color=[COLORS.get(label, "#999999") for label in labels])
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    save(fig, name)


def comparison_plots(summary):
    bar(
        summary,
        "method",
        "foreground_update_time_sec",
        "Foreground Update Time under Data-Stream Drift",
        "seconds / chunk",
        "core_foreground_update_time",
        log=True,
        ascending=True,
    )
    bar(
        summary,
        "method",
        "blocking_cycle_time_sec",
        "Blocking Response Cycle Time",
        "seconds / chunk",
        "core_blocking_cycle_time",
        log=True,
        ascending=True,
    )
    bar(summary, "method", "f1", "Predictive F1 under Data-Stream Drift", "Mean F1", "core_f1")
    bar(
        summary,
        "method",
        "meets_10ms_cycle",
        "10ms Realtime Cycle Success Rate",
        "fraction of chunks",
        "core_10ms_success",
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    for _, row in summary.iterrows():
        method = row["method"]
        ax.scatter(
            row["foreground_update_time_sec"],
            row["f1"],
            s=90,
            color=COLORS.get(method, "#999999"),
            label=method,
        )
        ax.annotate(method.replace("_MLN", ""), (row["foreground_update_time_sec"], row["f1"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Foreground update time (seconds / chunk, log scale)")
    ax.set_ylabel("Mean F1")
    ax.set_title("Speed-Adaptation Trade-off", fontsize=14, pad=12)
    ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    save(fig, "core_speed_f1_tradeoff")


def ablation_plots(summary):
    bar(
        summary,
        "variant",
        "foreground_update_time_sec",
        "Ablation: Foreground Update Time",
        "seconds / chunk",
        "ablation_foreground_update_time",
        log=True,
        ascending=True,
    )
    bar(summary, "variant", "f1", "Ablation: Predictive F1", "Mean F1", "ablation_f1")
    bar(
        summary,
        "variant",
        "initial_fit_time_sec",
        "Ablation: Initial Fit Time",
        "seconds",
        "ablation_initial_fit_time",
        log=True,
        ascending=True,
    )


def speedup_table(summary):
    rt = summary[summary["method"] == "RT_A3C_Realtime_MLN"].iloc[0]
    adaptive = summary[summary["category"] == "adaptive_batch_mln"].copy()
    adaptive["speedup_vs_rt_foreground"] = adaptive["foreground_update_time_sec"] / rt["foreground_update_time_sec"]
    adaptive["cycle_speedup_vs_rt"] = adaptive["blocking_cycle_time_sec"] / rt["blocking_cycle_time_sec"]
    cols = [
        "method",
        "foreground_update_time_sec",
        "blocking_cycle_time_sec",
        "f1",
        "speedup_vs_rt_foreground",
        "cycle_speedup_vs_rt",
    ]
    adaptive[cols].sort_values("speedup_vs_rt_foreground", ascending=False).to_csv(OUT / "rt_a3c_speedup_vs_batch_mln.csv", index=False)


def main():
    comparison = pd.read_csv(ROOT / "results" / "core_claim" / "core_realtime_summary.csv")
    ablation = pd.read_csv(ROOT / "results" / "core_ablation" / "core_ablation_summary.csv")
    comparison_plots(comparison)
    ablation_plots(ablation)
    speedup_table(comparison)
    print(f"[done] wrote figures to {OUT}")


if __name__ == "__main__":
    main()
