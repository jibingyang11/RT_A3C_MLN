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


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    methods = [method for method in METHOD_ORDER if method in set(frame["method"])]
    frame = frame.copy()
    frame["method"] = pd.Categorical(frame["method"], categories=methods, ordered=True)
    return frame.sort_values(["dataset", "method"])


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def grouped_bar(
    frame: pd.DataFrame,
    metric: str,
    output_dir: Path,
    ylabel: str,
    title: str,
    lower_is_better: bool = False,
    log_scale: bool = False,
) -> None:
    pivot = frame.pivot(index="dataset", columns="method", values=metric)
    pivot = pivot[[method for method in METHOD_ORDER if method in pivot.columns]]

    datasets = list(pivot.index)
    methods = list(pivot.columns)
    x = np.arange(len(datasets))
    width = min(0.11, 0.78 / max(1, len(methods)))

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for idx, method in enumerate(methods):
        values = pivot[method].to_numpy(dtype=float)
        offset = (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=method,
            color=COLORS.get(method, "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    if log_scale:
        ax.set_yscale("log")
    if not log_scale and not lower_is_better and metric in {"accuracy", "f1", "roc_auc"}:
        ymin = max(0.0, float(np.nanmin(pivot.to_numpy(dtype=float))) - 0.05)
        ax.set_ylim(ymin, 1.02)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(ncol=4, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _save(fig, output_dir, f"{metric}_by_dataset")


def selected_rules_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    grouped_bar(
        frame,
        "selected_rules",
        output_dir,
        ylabel="Selected Rules",
        title="Selected MLN Rules by Dataset",
        lower_is_better=True,
    )


def aggregate_score_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("accuracy", True),
        ("f1", True),
        ("roc_auc", True),
        ("log_loss", False),
        ("latency_ms_per_sample", False),
    ]
    rows = []
    for dataset, part in frame.groupby("dataset", observed=True):
        scored = part[["dataset", "method"]].copy()
        values = []
        for metric, higher_is_better in metrics:
            col = part[metric].astype(float)
            lo = float(col.min())
            hi = float(col.max())
            if hi - lo < 1e-12:
                norm = np.ones(len(part))
            elif higher_is_better:
                norm = (col - lo) / (hi - lo)
            else:
                norm = (hi - col) / (hi - lo)
            values.append(norm.to_numpy(dtype=float))
        scored["score"] = np.vstack(values).mean(axis=0)
        rows.append(scored)

    scores = pd.concat(rows, ignore_index=True)
    avg = scores.groupby("method", observed=True)["score"].mean().reindex(METHOD_ORDER).dropna()

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    bars = ax.bar(
        avg.index.astype(str),
        avg.to_numpy(dtype=float),
        color=[COLORS.get(method, "#999999") for method in avg.index.astype(str)],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title("Aggregate Normalized Performance Score", fontsize=14, pad=14)
    ax.set_ylabel("Mean normalized score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="x", rotation=25)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.018,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    _save(fig, output_dir, "aggregate_normalized_score")


def summary_table(frame: pd.DataFrame, output_dir: Path) -> None:
    aggregate = (
        frame.groupby("method", observed=True)[
            ["accuracy", "f1", "roc_auc", "log_loss", "latency_ms_per_sample", "fit_time_sec", "selected_rules"]
        ]
        .mean()
        .reindex(METHOD_ORDER)
        .dropna()
        .round(6)
    )
    aggregate.to_csv(output_dir / "aggregate_metrics.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _ordered(pd.read_csv(args.summary))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped_bar(frame, "accuracy", args.output_dir, "Accuracy", "Accuracy by Dataset")
    grouped_bar(frame, "f1", args.output_dir, "F1 Score", "F1 Score by Dataset")
    grouped_bar(frame, "roc_auc", args.output_dir, "ROC-AUC", "ROC-AUC by Dataset")
    grouped_bar(frame, "log_loss", args.output_dir, "Log Loss", "Log Loss by Dataset", lower_is_better=True)
    grouped_bar(
        frame,
        "latency_ms_per_sample",
        args.output_dir,
        "Latency (ms / sample)",
        "Inference Latency by Dataset",
        lower_is_better=True,
    )
    grouped_bar(
        frame,
        "fit_time_sec",
        args.output_dir,
        "Training Time (seconds, log scale)",
        "Training Time by Dataset",
        lower_is_better=True,
        log_scale=True,
    )
    selected_rules_plot(frame, args.output_dir)
    aggregate_score_plot(frame, args.output_dir)
    summary_table(frame, args.output_dir)
    print(f"[done] wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
