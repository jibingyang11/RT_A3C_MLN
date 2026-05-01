from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "large_scale" / "large_streaming_aggregate.csv"
OPT = ROOT / "results" / "large_scale_realtime_fastsearch" / "large_streaming_aggregate.csv"
OUT = ROOT / "results" / "large_scale_realtime_fastsearch" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    base = pd.read_csv(BASE).set_index("method")
    opt = pd.read_csv(OPT).set_index("method")
    rows = []
    for label, frame in [("Original", base), ("Realtime Optimized", opt)]:
        rows.append(
            {
                "version": label,
                "f1": frame.loc["RT_A3C_MLN", "f1"],
                "log_loss": frame.loc["RT_A3C_MLN", "log_loss"],
                "latency_ms_per_sample": frame.loc["RT_A3C_MLN", "latency_ms_per_sample"],
                "blocking_update_time_sec": frame.loc["RT_A3C_MLN", "update_time_sec"],
                "background_update_time_sec": frame.loc["RT_A3C_MLN"].get("background_update_time_sec", 0.0),
                "initial_fit_time_sec": frame.loc["RT_A3C_MLN", "initial_fit_time_sec"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT.parent / "realtime_optimization_comparison.csv", index=False)

    speed_rows = summary.melt(
        id_vars="version",
        value_vars=["initial_fit_time_sec", "blocking_update_time_sec", "latency_ms_per_sample"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    metrics = ["initial_fit_time_sec", "blocking_update_time_sec", "latency_ms_per_sample"]
    x = range(len(metrics))
    width = 0.35
    original = [float(summary.loc[summary["version"] == "Original", metric].iloc[0]) for metric in metrics]
    optimized = [float(summary.loc[summary["version"] == "Realtime Optimized", metric].iloc[0]) for metric in metrics]
    ax.bar([i - width / 2 for i in x], original, width=width, label="Original", color="#999999")
    ax.bar([i + width / 2 for i in x], optimized, width=width, label="Realtime Optimized", color="#D55E00")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Initial fit (s)", "Blocking update (s)", "Latency (ms/sample)"])
    ax.set_yscale("log")
    ax.set_title("RT-A3C-MLN Realtime Optimization", fontsize=14, pad=12)
    ax.set_ylabel("Log-scale time")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "realtime_optimization_speedup.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "realtime_optimization_speedup.pdf", bbox_inches="tight")
    plt.close(fig)

    print(summary.round(6).to_string(index=False))
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
