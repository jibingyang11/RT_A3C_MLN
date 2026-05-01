from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "results" / "large_scale"
OUT_DIR = IN_DIR / "figures"

ORDER = [
    "RT_A3C_MLN",
    "MaxEnt_MLN",
    "PLL_MLN",
    "L1_Sparse_MLN",
    "BeamSearch_MLN",
    "Boosted_MLN",
    "Online_SGD_MLN",
]
COLORS = {
    "RT_A3C_MLN": "#D55E00",
    "MaxEnt_MLN": "#0072B2",
    "PLL_MLN": "#009E73",
    "L1_Sparse_MLN": "#CC79A7",
    "BeamSearch_MLN": "#E69F00",
    "Boosted_MLN": "#56B4E9",
    "Online_SGD_MLN": "#666666",
}


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def bar(frame, metric, title, ylabel, lower_is_better=False):
    values = frame.groupby("method", observed=True)[metric].mean().reindex(ORDER).dropna()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(
        values.index.astype(str),
        values.to_numpy(float),
        color=[COLORS.get(method, "#999999") for method in values.index.astype(str)],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="x", rotation=25)
    if not lower_is_better and metric in {"accuracy", "f1", "roc_auc"}:
        ax.set_ylim(max(0.0, float(values.min()) - 0.05), min(1.02, float(values.max()) + 0.05))
    save(fig, f"large_{metric}")


def curve(frame, metric, title, ylabel):
    grouped = frame.groupby(["chunk", "method"], observed=True)[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    for method in ORDER:
        part = grouped[grouped["method"] == method]
        if part.empty:
            continue
        ax.plot(part["chunk"], part[metric], marker="o", label=method, color=COLORS.get(method, "#999999"))
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Stream Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, f"large_{metric}_curve")


def score_plot():
    scores = pd.read_csv(IN_DIR / "large_streaming_predictive_scores.csv", index_col=0)
    scores = scores.reindex(ORDER).dropna()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(
        scores.index.astype(str),
        scores["score"].to_numpy(float),
        color=[COLORS.get(method, "#999999") for method in scores.index.astype(str)],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title("Large-Scale Predictive Composite Score", fontsize=14, pad=12)
    ax.set_ylabel("Mean normalized predictive score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="x", rotation=25)
    save(fig, "large_predictive_score")


def sample_scale_plots():
    path = IN_DIR / "sample_scale" / "sample_scale_summary.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    for metric, ylabel, title in [
        ("f1", "Mean F1", "F1 across Sample Scales"),
        ("log_loss", "Mean Log Loss", "Log Loss across Sample Scales"),
        ("initial_fit_time_sec", "Initial Fit Time (s)", "Initial Fit Time across Sample Scales"),
    ]:
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        for method in ["RT_A3C_MLN", "MaxEnt_MLN", "L1_Sparse_MLN", "Boosted_MLN", "PLL_MLN", "Online_SGD_MLN"]:
            part = frame[frame["method"] == method].sort_values("sample_size")
            if part.empty:
                continue
            ax.plot(
                part["sample_size"],
                part[metric],
                marker="o",
                label=method,
                color=COLORS.get(method, "#999999"),
            )
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("Max Samples")
        ax.set_ylabel(ylabel)
        ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.legend(ncol=3, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14))
        save(fig, f"sample_scale_{metric}")


def main():
    global IN_DIR, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results" / "large_scale")
    args = parser.parse_args()
    IN_DIR = args.input_dir
    OUT_DIR = IN_DIR / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(IN_DIR / "large_streaming_detail.csv")
    bar(frame, "accuracy", "Large-Scale Streaming Accuracy", "Mean Accuracy")
    bar(frame, "f1", "Large-Scale Streaming F1", "Mean F1")
    bar(frame, "log_loss", "Large-Scale Streaming Log Loss", "Mean Log Loss", lower_is_better=True)
    bar(frame, "latency_ms_per_sample", "Large-Scale Inference Latency", "ms / sample", lower_is_better=True)
    bar(frame, "update_time_sec", "Large-Scale Update Time", "seconds / chunk", lower_is_better=True)
    if "background_update_time_sec" in frame.columns:
        bar(
            frame,
            "background_update_time_sec",
            "Large-Scale Background Update Cost",
            "seconds / chunk",
            lower_is_better=True,
        )
    curve(frame, "f1", "Large-Scale Streaming F1 Curve", "F1")
    curve(frame, "log_loss", "Large-Scale Streaming Log Loss Curve", "Log Loss")
    score_plot()
    sample_scale_plots()
    print(f"[done] wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
