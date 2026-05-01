from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "additional" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "RT_A3C_MLN": "#D55E00",
    "Full_RT_A3C_MLN": "#D55E00",
    "MaxEnt_MLN": "#0072B2",
    "L1_Sparse_MLN": "#CC79A7",
    "Boosted_MLN": "#56B4E9",
    "Online_SGD_MLN": "#666666",
    "PLL_MLN": "#009E73",
    "BeamSearch_MLN": "#E69F00",
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_multiseed():
    path = ROOT / "results" / "additional" / "multiseed_streaming" / "multiseed_streaming_summary.csv"
    frame = pd.read_csv(path)
    if "method" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "method"})
    frame = frame.sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.barh(
        frame["method"],
        frame["f1"],
        xerr=frame["f1_std"],
        color=[COLORS.get(method, "#999999") for method in frame["method"]],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title("Multi-Seed Streaming F1 (mean +/- std)", fontsize=14, pad=12)
    ax.set_xlabel("F1 Score")
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
    save(fig, "multiseed_streaming_f1")


def plot_ablation():
    path = ROOT / "results" / "additional" / "ablation_streaming" / "ablation_streaming_summary.csv"
    frame = pd.read_csv(path)
    if "variant" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "variant"})
    frame = frame.sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(frame["variant"], frame["f1"], color="#D55E00", edgecolor="white", linewidth=0.5)
    ax.set_title("Ablation Study under Streaming Drift", fontsize=14, pad=12)
    ax.set_ylabel("Mean F1")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    save(fig, "ablation_streaming_f1")


def plot_drift_strength():
    path = ROOT / "results" / "additional" / "drift_strength" / "drift_strength_summary.csv"
    frame = pd.read_csv(path)
    order = ["low", "medium", "high"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for method in ["RT_A3C_MLN", "MaxEnt_MLN", "L1_Sparse_MLN", "Boosted_MLN", "Online_SGD_MLN"]:
        part = frame[frame["method"] == method].copy()
        part["strength"] = pd.Categorical(part["strength"], categories=order, ordered=True)
        part = part.sort_values("strength")
        ax.plot(part["strength"].astype(str), part["f1"], marker="o", label=method, color=COLORS.get(method, "#999999"))
    ax.set_title("F1 across Drift Strengths", fontsize=14, pad=12)
    ax.set_xlabel("Drift Strength")
    ax.set_ylabel("Mean F1")
    ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, "drift_strength_f1")


def plot_actor_realtime():
    path = ROOT / "results" / "additional" / "actor_realtime" / "actor_realtime_summary.csv"
    frame = pd.read_csv(path)
    workers = frame[frame["mode"] == "worker_sweep"].sort_values("workers")
    chunks = frame[frame["mode"] == "chunk_sweep"].sort_values("chunk_size")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(workers["workers"], workers["f1"], marker="o", color="#D55E00")
    ax.set_title("Actor Count Sensitivity", fontsize=14, pad=12)
    ax.set_xlabel("Actor Workers")
    ax.set_ylabel("Mean F1")
    ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    save(fig, "actor_count_f1")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(chunks["chunk_size"], chunks["latency_ms_per_sample"], marker="o", color="#0072B2", label="Latency")
    ax.set_title("Chunk Size vs Inference Latency", fontsize=14, pad=12)
    ax.set_xlabel("Chunk Size")
    ax.set_ylabel("Latency (ms / sample)")
    ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    save(fig, "chunk_size_latency")


def plot_rule_dynamics():
    path = ROOT / "results" / "additional" / "rule_dynamics" / "rule_weight_deltas.csv"
    frame = pd.read_csv(path)
    for dataset, part in frame.groupby("dataset", observed=True):
        top = part.sort_values("abs_delta", ascending=False).head(8).iloc[::-1]
        labels = [textwrap.shorten(rule.replace("Target(x)=1 <- ", ""), width=42, placeholder="...") for rule in top["rule"]]
        fig, ax = plt.subplots(figsize=(10.5, 5.4))
        ax.barh(labels, top["abs_delta"], color="#D55E00", edgecolor="white", linewidth=0.5)
        ax.set_title(f"Top Rule Weight Changes: {dataset}", fontsize=14, pad=12)
        ax.set_xlabel("Absolute Weight Change")
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
        save(fig, f"rule_weight_delta_{dataset}")


def main():
    plot_multiseed()
    plot_ablation()
    plot_drift_strength()
    plot_actor_realtime()
    plot_rule_dynamics()
    print(f"[done] wrote figures to {OUT}")


if __name__ == "__main__":
    main()
