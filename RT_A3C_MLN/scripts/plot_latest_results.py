from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "latest_figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "RT_A3C_MLN": "#D55E00",
    "MaxEnt_MLN": "#0072B2",
    "PLL_MLN": "#009E73",
    "L1_Sparse_MLN": "#CC79A7",
    "BeamSearch_MLN": "#E69F00",
    "Boosted_MLN": "#56B4E9",
    "Online_SGD_MLN": "#666666",
    "Ultra_Fast_Update": "#D55E00",
    "Full_Background_Refresh": "#0072B2",
    "No_Fast_Search": "#E69F00",
    "No_Prior_Shift": "#999999",
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def bar(frame, label_col, metric, title, ylabel, name, log=False):
    frame = frame.sort_values(metric, ascending=metric in {"log_loss", "update_time_sec", "initial_fit_time_sec"})
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    labels = frame[label_col].astype(str)
    ax.bar(labels, frame[metric].astype(float), color=[COLORS.get(label, "#999999") for label in labels])
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    save(fig, name)


def comparison_plots():
    agg = pd.read_csv(ROOT / "results" / "latest_comparison" / "large_streaming_aggregate.csv")
    if "method" not in agg.columns:
        agg = agg.rename(columns={agg.columns[0]: "method"})
    bar(agg, "method", "f1", "Latest Large-Scale F1", "Mean F1", "latest_comparison_f1")
    bar(agg, "method", "update_time_sec", "Latest Blocking Update Time", "seconds / chunk", "latest_comparison_update_time", log=True)
    bar(agg, "method", "initial_fit_time_sec", "Latest Initial Fit Time", "seconds", "latest_comparison_initial_fit", log=True)


def ablation_plots():
    abl = pd.read_csv(ROOT / "results" / "latest_ablation" / "latest_speed_ablation_summary.csv")
    if "variant" not in abl.columns:
        abl = abl.rename(columns={abl.columns[0]: "variant"})
    bar(abl, "variant", "f1", "Latest Speed Ablation: F1", "Mean F1", "latest_ablation_f1")
    bar(abl, "variant", "update_time_sec", "Latest Speed Ablation: Blocking Update", "seconds / chunk", "latest_ablation_update_time", log=True)
    bar(abl, "variant", "initial_fit_time_sec", "Latest Speed Ablation: Initial Fit", "seconds", "latest_ablation_initial_fit", log=True)


def main():
    comparison_plots()
    ablation_plots()
    print(f"[done] wrote figures to {OUT}")


if __name__ == "__main__":
    main()
