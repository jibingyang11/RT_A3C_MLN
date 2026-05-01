from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = [
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


def _save(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_curve(frame, metric, output_dir, ylabel, title):
    curve = frame.groupby(["chunk", "method"], observed=True)[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    for method in METHOD_ORDER:
        part = curve[curve["method"] == method]
        if part.empty:
            continue
        ax.plot(
            part["chunk"],
            part[metric],
            marker="o",
            linewidth=2.0 if method == "RT_A3C_MLN" else 1.45,
            label=method,
            color=COLORS.get(method, "#999999"),
        )
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Stream Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(ncol=4, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _save(fig, output_dir, f"streaming_{metric}_curve")


def plot_bar(frame, metric, output_dir, ylabel, title, lower_is_better=False):
    avg = frame.groupby("method", observed=True)[metric].mean().reindex(METHOD_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(
        avg.index.astype(str),
        avg.to_numpy(float),
        color=[COLORS.get(method, "#999999") for method in avg.index.astype(str)],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="x", rotation=25)
    if not lower_is_better and metric in {"accuracy", "f1", "roc_auc"}:
        ax.set_ylim(max(0, float(avg.min()) - 0.05), min(1.02, float(avg.max()) + 0.05))
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, f"streaming_{metric}_bar")


def plot_drift(frame, output_dir):
    drift = frame.groupby(["dataset", "chunk"], observed=True)["observed_positive_rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    for dataset, part in drift.groupby("dataset", observed=True):
        ax.plot(part["chunk"], part["observed_positive_rate"], marker="o", label=dataset)
    ax.set_title("Simulated Label-Prior Drift", fontsize=14, pad=14)
    ax.set_xlabel("Stream Chunk")
    ax.set_ylabel("Observed Positive Rate")
    ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _save(fig, output_dir, "streaming_drift_rates")


def plot_predictive_score(frame, output_dir):
    metrics = [("accuracy", False), ("f1", False), ("roc_auc", False), ("log_loss", True)]
    rows = []
    for dataset, group in frame.groupby("dataset", observed=True):
        values = group.groupby("method", observed=True)[[metric for metric, _ in metrics]].mean()
        score = pd.Series(0.0, index=values.index)
        for metric, lower_is_better in metrics:
            col = values[metric]
            lo = float(col.min())
            hi = float(col.max())
            if hi <= lo:
                norm = col * 0 + 1
            elif lower_is_better:
                norm = (hi - col) / (hi - lo)
            else:
                norm = (col - lo) / (hi - lo)
            score = score + norm
        rows.append((score / len(metrics)).rename(dataset))

    scores = pd.concat(rows, axis=1).mean(axis=1).reindex(METHOD_ORDER).dropna()
    scores.to_csv(output_dir / "streaming_predictive_scores.csv", header=["score"])

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(
        scores.index.astype(str),
        scores.to_numpy(float),
        color=[COLORS.get(method, "#999999") for method in scores.index.astype(str)],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title("Streaming Predictive Composite Score", fontsize=14, pad=14)
    ax.set_ylabel("Mean normalized predictive score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="x", rotation=25)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, "streaming_predictive_score")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "streaming" / "streaming_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "streaming" / "figures")
    args = parser.parse_args()
    frame = pd.read_csv(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_drift(frame, args.output_dir)
    plot_metric_curve(frame, "f1", args.output_dir, "F1 Score", "Streaming F1 under Distribution Drift")
    plot_metric_curve(frame, "log_loss", args.output_dir, "Log Loss", "Streaming Log Loss under Distribution Drift")
    plot_bar(frame, "accuracy", args.output_dir, "Mean Accuracy", "Mean Streaming Accuracy")
    plot_bar(frame, "f1", args.output_dir, "Mean F1", "Mean Streaming F1")
    plot_bar(frame, "log_loss", args.output_dir, "Mean Log Loss", "Mean Streaming Log Loss", lower_is_better=True)
    plot_bar(
        frame,
        "latency_ms_per_sample",
        args.output_dir,
        "Mean Latency (ms / sample)",
        "Mean Streaming Inference Latency",
        lower_is_better=True,
    )
    plot_bar(
        frame,
        "update_time_sec",
        args.output_dir,
        "Mean Update Time (seconds / chunk)",
        "Mean Streaming Update Time",
        lower_is_better=True,
    )
    plot_predictive_score(frame, args.output_dir)
    print(f"[done] wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
