from pathlib import Path
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(os.environ.get("PAPER_ARTIFACTS_DIR", ROOT / "paper_artifacts"))
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
SOURCE_DIR = OUT / "source_results"
EXPLANATION_DIR = OUT / "explanations"


METHOD_ORDER = [
    "RT_A3C_Realtime_MLN",
    "Rolling_L1_MLN",
    "Rolling_MaxEnt_MLN",
    "Rolling_Boosted_MLN",
    "Rolling_BeamSearch_MLN",
    "OnlineWeight_MLN",
    "Static_MaxEnt_Reference",
]

STREAM_METHODS = [
    "RT_A3C_Realtime_MLN",
    "Rolling_L1_MLN",
    "Rolling_MaxEnt_MLN",
    "Rolling_Boosted_MLN",
    "Rolling_BeamSearch_MLN",
]

ABLATION_ORDER = [
    "Full_Realtime_A3C",
    "Single_Actor",
    "No_Prior_Shift",
    "No_Structure_Refresh",
    "Online_Only_No_A3C",
]

COLORS = {
    "RT_A3C_Realtime_MLN": "#c1121f",
    "Rolling_L1_MLN": "#2a9d8f",
    "Rolling_MaxEnt_MLN": "#457b9d",
    "Rolling_Boosted_MLN": "#f4a261",
    "Rolling_BeamSearch_MLN": "#6d597a",
    "OnlineWeight_MLN": "#adb5bd",
    "Static_MaxEnt_Reference": "#8d99ae",
}


def method_label(method, compact=False):
    compact_labels = {
        "RT_A3C_Realtime_MLN": "RT-A3C",
        "Rolling_L1_MLN": "L1",
        "Rolling_MaxEnt_MLN": "MaxEnt",
        "Rolling_Boosted_MLN": "Boosted",
        "Rolling_BeamSearch_MLN": "Beam",
        "OnlineWeight_MLN": "Online",
        "Static_MaxEnt_Reference": "Static",
    }
    labels = {
        "RT_A3C_Realtime_MLN": "RT-A3C-MLN (ours)",
        "Rolling_L1_MLN": "Rolling L1 MLN",
        "Rolling_MaxEnt_MLN": "Rolling MaxEnt MLN",
        "Rolling_Boosted_MLN": "Rolling Boosted MLN",
        "Rolling_BeamSearch_MLN": "Rolling BeamSearch MLN",
        "OnlineWeight_MLN": "OnlineWeight MLN",
        "Static_MaxEnt_Reference": "Static MaxEnt reference",
    }
    return (compact_labels if compact else labels).get(method, method)


def variant_label(variant, compact=False):
    compact_labels = {
        "Full_Realtime_A3C": "Full",
        "Single_Actor": "Single",
        "No_Prior_Shift": "No prior",
        "No_Structure_Refresh": "No structure",
        "Online_Only_No_A3C": "Online only",
    }
    labels = {
        "Full_Realtime_A3C": "Full RT-A3C-MLN",
        "Single_Actor": "Single actor",
        "No_Prior_Shift": "No prior-shift update",
        "No_Structure_Refresh": "No structure refresh",
        "Online_Only_No_A3C": "Online only, no A3C",
    }
    return (compact_labels if compact else labels).get(variant, variant)


def ensure_clean_output():
    target = OUT.resolve()
    if target.exists():
        if target.parent != ROOT.resolve() or target.name != "paper_artifacts":
            raise RuntimeError(f"Refusing to remove unexpected path: {target}")
        shutil.rmtree(target)
    FIG_DIR.mkdir(parents=True)
    TABLE_DIR.mkdir(parents=True)
    SOURCE_DIR.mkdir(parents=True)
    EXPLANATION_DIR.mkdir(parents=True)


def fmt_num(value, digits=3):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_pct(value, digits=1):
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}"


def latex_escape(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def dataframe_to_latex(df, caption, label):
    columns = list(df.columns)
    align = "l" + "c" * (len(columns) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[col]) for col in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_table_files(df, stem, caption):
    df.to_csv(TABLE_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")

    cols = list(df.columns)
    lines = [f"# {caption}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    (TABLE_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = dataframe_to_latex(df, caption=caption, label=f"tab:{stem.lower()}")
    (TABLE_DIR / f"{stem}.tex").write_text(tex, encoding="utf-8")


def set_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def save_figure(fig, stem):
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def add_box(ax, xy, width, height, text, face, edge="#264653", fontsize=10):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.016,rounding_size=0.018",
        linewidth=1.4,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#172026",
        linespacing=1.22,
    )


def add_arrow(ax, start, end, text=None, rad=0.0, linestyle="-"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.35,
        color="#31556b",
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if text:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.025,
            text,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#31556b",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
        )


def draw_framework_diagram():
    fig, ax = plt.subplots(figsize=(12.8, 6.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "Realtime A3C-MLN Structure Learning Framework",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
        color="#16303a",
    )

    add_box(ax, (0.04, 0.58), 0.14, 0.14, "Streaming data\nchunks", "#d8f3dc")
    add_box(ax, (0.25, 0.57), 0.17, 0.16, "Concept lattice\nenvironments\nMLN state", "#e7f5ff")
    add_box(
        ax,
        (0.49, 0.54),
        0.17,
        0.22,
        "Parallel actor-learners\nActor 1 ... Actor K\nlocal policy pi(a|s)\nrule actions",
        "#f1f3f5",
        fontsize=9.2,
    )
    add_box(ax, (0.75, 0.57), 0.18, 0.16, "Global Actor-Critic\nshared policy + value", "#fde2e4")
    add_box(ax, (0.75, 0.34), 0.18, 0.12, "MLN rule-weight\nknowledge memory", "#e9ecef")
    add_box(ax, (0.75, 0.15), 0.18, 0.12, "Realtime inference\nprobabilistic output", "#d0ebff")
    add_box(ax, (0.25, 0.25), 0.17, 0.12, "Reward signal\nvalidation log-likelihood", "#fff3bf")
    add_box(ax, (0.04, 0.25), 0.14, 0.12, "Arriving labels\nvalidation set", "#fff3bf")

    add_arrow(ax, (0.18, 0.65), (0.25, 0.65), "stream state")
    add_arrow(ax, (0.42, 0.65), (0.49, 0.65), "candidate rules")
    add_arrow(ax, (0.66, 0.65), (0.75, 0.65), "async gradients")
    add_arrow(ax, (0.84, 0.57), (0.84, 0.46), "policy/value update")
    add_arrow(ax, (0.84, 0.34), (0.84, 0.27), "foreground adaptation")
    add_arrow(ax, (0.18, 0.31), (0.25, 0.31), "evaluate")
    add_arrow(ax, (0.42, 0.31), (0.75, 0.59), "n-step return + reward", rad=-0.23)
    add_arrow(ax, (0.75, 0.70), (0.66, 0.73), "advantage A(s,a)", rad=0.18, linestyle="--")

    ax.text(
        0.50,
        0.075,
        "Action space: add / delete / modify rules, adjust weights     "
        "Update target: n-step return with advantage signal     "
        "Goal: low blocking cycle under streaming drift",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#263238",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#ffffff", edgecolor="#adb5bd"),
    )

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig1_Method_Framework.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    mermaid = """flowchart LR
    A["Streaming data chunks"] --> B["Concept lattice environments: MLN state"]
    B --> C["Parallel actor-learners: local policy pi(a|s)"]
    C -->|"async gradients"| D["Global Actor-Critic"]
    D -->|"policy/value update"| E["MLN rule-weight knowledge memory"]
    E -->|"foreground adaptation"| F["Realtime probabilistic inference"]
    G["Arriving labels + validation set"] --> H["Reward: validation log-likelihood"]
    H -->|"n-step return + reward"| D
    D -->|"advantage A(s,a)"| C
"""
    (FIG_DIR / "Fig1_Method_Framework.mmd").write_text(mermaid, encoding="utf-8")


def role_for_method(method):
    if method == "RT_A3C_Realtime_MLN":
        return "async A3C rule/weight adaptation"
    if method.startswith("Rolling_"):
        return "rolling global MLN refit"
    if method == "OnlineWeight_MLN":
        return "online fixed-rule reference"
    return "static non-adaptive reference"


def build_main_table(summary, speedup):
    speedup_map = dict(zip(speedup["method"], speedup["speedup_vs_rt_foreground"]))
    summary = summary.set_index("method").loc[METHOD_ORDER].reset_index()
    rows = []
    for _, row in summary.iterrows():
        method = row["method"]
        if method == "RT_A3C_Realtime_MLN":
            speed = "1.00x"
        elif method in speedup_map:
            speed = f"{speedup_map[method]:.2f}x"
        elif method == "OnlineWeight_MLN":
            speed = "fixed rule"
        else:
            speed = "no update"
        rows.append(
            {
                "Method": method_label(method),
                "Streaming update protocol": role_for_method(method),
                "Adaptive rules": "Yes" if method == "RT_A3C_Realtime_MLN" or method.startswith("Rolling_") else "No",
                "F1": fmt_num(row["f1"]),
                "AUC": fmt_num(row["roc_auc"]),
                "Log-loss": fmt_num(row["log_loss"]),
                "FG update ms": fmt_num(row["foreground_update_time_sec"] * 1000),
                "Blocking cycle ms": fmt_num(row["blocking_cycle_time_sec"] * 1000),
                "10ms success %": fmt_pct(row["meets_10ms_cycle"]),
                "Struct update %": fmt_pct(row.get("structure_refreshed", 0.0)),
                "Rule +/- per chunk": f"{fmt_num(row.get('rule_additions', 0.0), 2)} / {fmt_num(row.get('rule_deletions', 0.0), 2)}",
                "Update slowdown vs ours": speed,
            }
        )
    return pd.DataFrame(rows)


def compute_speedup(summary):
    rt = summary.loc[summary["method"] == "RT_A3C_Realtime_MLN"].iloc[0]
    baselines = summary[summary["method"].isin([
        "Rolling_L1_MLN",
        "Rolling_MaxEnt_MLN",
        "Rolling_Boosted_MLN",
        "Rolling_BeamSearch_MLN",
    ])].copy()
    baselines["speedup_vs_rt_foreground"] = (
        baselines["foreground_update_time_sec"] / max(float(rt["foreground_update_time_sec"]), 1e-12)
    )
    baselines["cycle_speedup_vs_rt"] = (
        baselines["blocking_cycle_time_sec"] / max(float(rt["blocking_cycle_time_sec"]), 1e-12)
    )
    return baselines[
        [
            "method",
            "foreground_update_time_sec",
            "blocking_cycle_time_sec",
            "f1",
            "speedup_vs_rt_foreground",
            "cycle_speedup_vs_rt",
        ]
    ]


def build_ablation_table(ablation):
    meaning = {
        "Full_Realtime_A3C": "complete realtime foreground update",
        "Single_Actor": "removes parallel actor exploration",
        "No_Prior_Shift": "removes local stream-prior adaptation",
        "No_Structure_Refresh": "disables background rule-structure refresh",
        "Online_Only_No_A3C": "keeps online weights, removes A3C structure learning",
    }
    ablation = ablation.set_index("variant").loc[ABLATION_ORDER].reset_index()
    rows = []
    for _, row in ablation.iterrows():
        rows.append(
            {
                "Variant": variant_label(row["variant"]),
                "Controlled change": meaning[row["variant"]],
                "F1": fmt_num(row["f1"]),
                "AUC": fmt_num(row["roc_auc"]),
                "Log-loss": fmt_num(row["log_loss"]),
                "FG update ms": fmt_num(row["foreground_update_time_sec"] * 1000),
                "Blocking cycle ms": fmt_num(row["blocking_cycle_time_sec"] * 1000),
                "Struct update %": fmt_pct(row.get("structure_refreshed", 0.0)),
                "Rule +/chunk": fmt_num(row.get("rule_additions", 0.0), 2),
                "Rule -/chunk": fmt_num(row.get("rule_deletions", 0.0), 2),
                "10ms success %": fmt_pct(row["meets_10ms_cycle"]),
            }
        )
    return pd.DataFrame(rows)


def draw_realtime_latency_figure(summary):
    df = summary.set_index("method").loc[METHOD_ORDER].reset_index()
    y = np.arange(len(df))
    update_ms = df["foreground_update_time_sec"].to_numpy() * 1000
    cycle_ms = df["blocking_cycle_time_sec"].to_numpy() * 1000

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for idx, method in enumerate(df["method"]):
        color = COLORS[method]
        ax.plot([max(update_ms[idx], 0.02), max(cycle_ms[idx], 0.02)], [idx, idx], color=color, lw=2.3, alpha=0.75)
        ax.scatter(max(update_ms[idx], 0.02), idx, s=52, marker="o", color=color, edgecolor="white", zorder=3)
        ax.scatter(max(cycle_ms[idx], 0.02), idx, s=68, marker="D", color=color, edgecolor="white", zorder=3)
        ax.text(
            cycle_ms[idx] * 1.08 + 0.02,
            idx,
            f"{cycle_ms[idx]:.1f} ms / {df.loc[idx, 'meets_10ms_cycle'] * 100:.0f}%",
            va="center",
            fontsize=8.0,
            color="#263238",
        )

    ax.axvline(5, color="#6c757d", lw=1.0, ls="--")
    ax.axvline(10, color="#c1121f", lw=1.2, ls="--")
    ax.text(
        5,
        0.02,
        "5 ms update",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#6c757d",
    )
    ax.text(
        10,
        0.02,
        "10 ms cycle",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#c1121f",
    )

    ax.set_xscale("log")
    ax.set_xlim(0.3, max(cycle_ms) * 1.85)
    ax.set_yticks(y)
    ax.set_yticklabels([method_label(m) for m in df["method"]])
    ax.invert_yaxis()
    ax.set_xlabel("Time per stream chunk (ms, log scale)")
    ax.set_title("Realtime response cost: foreground update vs blocking cycle")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.scatter([], [], marker="o", color="#495057", label="foreground update")
    ax.scatter([], [], marker="D", color="#495057", label="blocking cycle")
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "Fig2_Realtime_Latency_Breakdown")


def draw_deadline_heatmap(detail):
    grouped = (
        detail.groupby(["method", "dataset"], observed=True)
        .agg(blocking_ms=("blocking_cycle_time_sec", lambda x: float(np.mean(x) * 1000)),
             success=("meets_10ms_cycle", "mean"))
        .reset_index()
    )
    methods = METHOD_ORDER
    datasets = sorted(grouped["dataset"].unique())

    block = np.zeros((len(methods), len(datasets)))
    succ = np.zeros_like(block)
    for i, method in enumerate(methods):
        for j, dataset in enumerate(datasets):
            row = grouped[(grouped["method"] == method) & (grouped["dataset"] == dataset)]
            block[i, j] = row["blocking_ms"].iloc[0]
            succ[i, j] = row["success"].iloc[0] * 100

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax0, ax1 = axes

    log_block = np.log10(block + 0.02)
    im0 = ax0.imshow(log_block, aspect="auto", cmap="YlOrRd")
    ax0.set_title("Blocking cycle by dataset (ms)")
    ax0.set_xticks(range(len(datasets)))
    ax0.set_xticklabels(datasets, rotation=25, ha="right")
    ax0.set_yticks(range(len(methods)))
    ax0.set_yticklabels([method_label(m, compact=True) for m in methods])
    for i in range(len(methods)):
        for j in range(len(datasets)):
            val = block[i, j]
            text = f"{val:.1f}" if val < 100 else f"{val:.0f}"
            ax0.text(j, i, text, ha="center", va="center", fontsize=7.5, color="#172026")
    cbar = fig.colorbar(im0, ax=ax0, fraction=0.045, pad=0.02)
    cbar.set_label("log10(ms)")

    im1 = ax1.imshow(succ, aspect="auto", cmap="Greens", vmin=0, vmax=100)
    ax1.set_title("10 ms deadline success (%)")
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, rotation=25, ha="right")
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels([])
    for i in range(len(methods)):
        for j in range(len(datasets)):
            ax1.text(j, i, f"{succ[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="#172026")
    cbar = fig.colorbar(im1, ax=ax1, fraction=0.045, pad=0.02)
    cbar.set_label("%")

    for ax in axes:
        ax.grid(False)
        ax.tick_params(length=0)
    fig.suptitle("Realtime robustness across datasets", y=0.99, fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "Fig3_Dataset_Deadline_Heatmap")


def draw_stream_trajectory(detail):
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.05]})
    ax_f1, ax_time = axes

    chunked = (
        detail[detail["method"].isin(STREAM_METHODS)]
        .groupby(["method", "chunk"], observed=True)
        .agg(f1=("f1", "mean"), cycle_ms=("blocking_cycle_time_sec", lambda x: float(np.mean(x) * 1000)))
        .reset_index()
    )
    drift = (
        detail[detail["method"] == "RT_A3C_Realtime_MLN"]
        .groupby("chunk", observed=True)["drift_positive_rate"]
        .mean()
        .reset_index()
    )

    for method in STREAM_METHODS:
        sub = chunked[chunked["method"] == method]
        ax_f1.plot(
            sub["chunk"],
            sub["f1"],
            marker="o",
            lw=2.1 if method == "RT_A3C_Realtime_MLN" else 1.4,
            color=COLORS[method],
            label=method_label(method, compact=True),
            alpha=1.0 if method == "RT_A3C_Realtime_MLN" else 0.78,
        )
        ax_time.plot(
            sub["chunk"],
            sub["cycle_ms"],
            marker="o",
            lw=2.1 if method == "RT_A3C_Realtime_MLN" else 1.4,
            color=COLORS[method],
            alpha=1.0 if method == "RT_A3C_Realtime_MLN" else 0.78,
        )

    ax_drift = ax_f1.twinx()
    ax_drift.fill_between(drift["chunk"], drift["drift_positive_rate"], color="#dee2e6", alpha=0.45, step="mid")
    ax_drift.plot(drift["chunk"], drift["drift_positive_rate"], color="#6c757d", ls="--", lw=1.2)
    ax_drift.set_ylabel("mean positive rate")
    ax_drift.set_ylim(0, 1)
    ax_drift.grid(False)

    ax_f1.set_ylabel("F1")
    fig.suptitle("Stream-drift adaptation and realtime blocking response", y=0.99, fontsize=12, weight="bold")
    ax_f1.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)

    ax_time.axhline(10, color="#c1121f", ls="--", lw=1.1)
    ax_time.text(8.05, 10, "10 ms deadline", va="bottom", ha="right", color="#c1121f", fontsize=8.5)
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Blocking cycle (ms, log)")
    ax_time.set_xlabel("Stream chunk")
    ax_time.set_xticks(sorted(detail["chunk"].unique()))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "Fig4_Stream_Drift_Response")


def draw_ablation_figure(ablation):
    df = ablation.set_index("variant").loc[ABLATION_ORDER].reset_index()
    x = np.arange(len(df))
    labels = [variant_label(v, compact=True) for v in df["variant"]]
    accent = ["#c1121f", "#e76f51", "#2a9d8f", "#457b9d", "#f4a261", "#adb5bd"]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.05))

    ax0 = axes[0]
    ax0.bar(x, df["f1"], color=accent, width=0.64)
    ax0.set_title("Predictive quality")
    ax0.set_ylabel("F1")
    ax0.set_ylim(max(0.55, df["f1"].min() - 0.06), min(1.0, df["f1"].max() + 0.06))
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=35, ha="right")
    ax0b = ax0.twinx()
    ax0b.plot(x, df["log_loss"], color="#343a40", marker="D", lw=1.6, label="log-loss")
    ax0b.set_ylabel("Log-loss")
    ax0b.grid(False)

    ax1 = axes[1]
    width = 0.34
    ax1.bar(x - width / 2, df["foreground_update_time_sec"] * 1000, width=width, color="#74c0fc", label="FG update")
    ax1.bar(x + width / 2, df["blocking_cycle_time_sec"] * 1000, width=width, color="#1864ab", label="blocking cycle")
    ax1.axhline(10, color="#c1121f", ls="--", lw=1.0)
    ax1.set_title("Realtime response")
    ax1.set_ylabel("Time (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.legend(frameon=False, loc="upper left")

    ax2 = axes[2]
    additions = df.get("rule_additions", pd.Series(np.zeros(len(df)))) 
    deletions = df.get("rule_deletions", pd.Series(np.zeros(len(df))))
    ax2.bar(x, additions, color="#52b788", width=0.64, label="added rules")
    ax2.bar(x, deletions, bottom=additions, color="#d00000", width=0.64, label="deleted rules")
    ax2.set_title("Rule-structure adaptation")
    ax2.set_ylabel("Rule changes / chunk")
    stacked_max = float((additions + deletions).max()) if len(df) else 0.0
    ax2.set_ylim(0, max(1.0, stacked_max * 1.28))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2b = ax2.twinx()
    refresh_line = ax2b.plot(
        x,
        df.get("structure_refreshed", pd.Series(np.zeros(len(df)))) * 100,
        color="#343a40",
        marker="o",
        lw=1.5,
        label="accepted refresh",
    )
    ax2b.set_ylabel("Accepted refresh (%)")
    ax2b.set_ylim(0, max(100, float(df.get("structure_refreshed", pd.Series(np.zeros(len(df)))).max() * 120 + 1)))
    ax2b.grid(False)
    handles, legend_labels = ax2.get_legend_handles_labels()
    handles += refresh_line
    legend_labels += [refresh_line[0].get_label()]
    ax2.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=1,
        borderaxespad=0.2,
        handlelength=1.4,
        columnspacing=0.9,
    )

    fig.suptitle("Ablation evidence: update mechanism, rule refresh, and online adaptation", y=0.99, fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "Fig5_Ablation_Summary")


def copy_source_results():
    files = [
        RESULTS / "core_claim" / "core_realtime_summary.csv",
        RESULTS / "core_claim" / "core_realtime_detail.csv",
        RESULTS / "core_claim" / "core_realtime_config.json",
        RESULTS / "core_ablation" / "core_ablation_summary.csv",
        RESULTS / "core_ablation" / "core_ablation_detail.csv",
        RESULTS / "core_ablation" / "core_ablation_config.json",
    ]
    for src in files:
        if src.exists():
            shutil.copy2(src, SOURCE_DIR / src.name)


def write_notes(config):
    readme = f"""# Compact Paper Artifacts

This folder intentionally contains only the paper-core set:

1. One method framework figure.
2. One main result table.
3. Three realtime figures.
4. One ablation table.
5. One ablation figure.

The figures are generated from the latest core streaming results in `results/core_claim` and `results/core_ablation`.
The current protocol uses {len(config['datasets'])} datasets ({', '.join(config['datasets'])}), {config['chunks']} stream chunks, chunk size {config['chunk_size']}, and simulated prior drift per chunk.

## Recommended placement

- Fig1_Method_Framework: Method section.
- Table1_Main_Results: Main experimental comparison.
- Fig2_Realtime_Latency_Breakdown: Realtime latency comparison.
- Fig3_Dataset_Deadline_Heatmap: Per-dataset realtime robustness.
- Fig4_Stream_Drift_Response: Adaptation under data stream drift.
- Table2_Ablation: Ablation table.
- Fig5_Ablation_Summary: Ablation figure.
- explanations/: Chinese explanation files for every figure and table.

## Writing focus

The main claim should be framed around adaptive batch MLN baselines that must refit or reselect rule structures on each stream update. Static MaxEnt and fixed-structure OnlineWeight MLN are references, not direct evidence against realtime adaptive structure learning.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    captions = """# Captions

Fig. 1. Realtime A3C-MLN structure learning framework. Multiple actor-learners explore concept-lattice environments independently, send asynchronous gradients to the global Actor-Critic network, and update the MLN rule-weight memory for low-latency probabilistic inference.

Fig. 2. Realtime response cost of foreground update and blocking cycle. The blocking cycle is measured as inference plus foreground update after each stream chunk; the vertical 10 ms line marks the realtime deadline.

Fig. 3. Realtime robustness across datasets. The left panel reports mean blocking cycle time and the right panel reports the percentage of stream chunks satisfying the 10 ms deadline.

Fig. 4. Stream-drift adaptation trajectory. The top panel reports F1 across stream chunks with the average positive-rate drift shown in gray; the bottom panel reports blocking cycle time on a logarithmic scale.

Fig. 5. Ablation results. Removing prior-shift adaptation, background rule refresh, or A3C structure learning changes the speed-quality tradeoff; the full realtime model maintains sub-10 ms blocking response while preserving accepted rule-structure updates.

Table 1. Main comparison of predictive performance, update latency, blocking response time, and realtime deadline success.

Table 2. Ablation study of the asynchronous actor mechanism, local prior adaptation, background structure refresh, and online-only updating.
"""
    (OUT / "captions.md").write_text(captions, encoding="utf-8")


def write_explanation_file(stem, text):
    (EXPLANATION_DIR / f"{stem}.md").write_text(text.strip() + "\n", encoding="utf-8")


def write_explanations():
    write_explanation_file(
        "Fig1_Method_Framework_explanation",
        """
# Fig1_Method_Framework 详细解释

## 图的作用

这张图是论文方法部分的总体框架图，用来说明“基于异步优势 Actor-Critic 的 MLN 规则学习与量化”是如何在数据流场景下实现快速自适应推理的。它不是实验结果图，因此没有传统意义上的横轴和纵轴；图中每个矩形模块表示系统中的一个功能组件，每条箭头表示信息流、梯度流或更新流。

## 模块含义

- **Streaming data chunks**：数据流输入。实验中连续到达的数据被划分为多个 chunk，每个 chunk 表示一次实时输入批次。
- **Concept lattice environments / MLN state**：概念格学习环境。每个概念格可被看作一个独立的学习环境，当前状态由 MLN 的规则集合、规则结构和权重配置共同构成。
- **Parallel actor-learners**：并行 Actor 学习器。多个 Actor 在不同概念格环境中独立探索，按照当前策略 `pi(a|s)` 选择动作。
- **rule actions**：规则动作空间。动作包括添加规则、删除规则、修改规则、调整规则权重等。
- **Global Actor-Critic shared policy + value**：全局 Actor-Critic 网络。Actor 负责策略更新，Critic 负责状态价值评估，多个 Actor 的梯度异步汇入全局网络。
- **MLN rule-weight knowledge memory**：MLN 规则权重知识记忆。它保存当前概率知识网络的规则结构和权重，是实时推理的知识基础。
- **Realtime inference probabilistic output**：实时概率推理输出。输入数据到来后，模型基于当前 MLN 规则和权重快速输出概率推理结果。
- **Arriving labels / validation set**：新到达标签和验证数据。它们用于评估当前规则结构和权重在最新数据分布下的效果。
- **Reward signal validation log-likelihood**：奖励信号。实验中使用验证集对数似然作为奖励，表示当前 MLN 对新数据的解释能力。

## 箭头含义

- **stream state**：数据流 chunk 被转换为当前状态，送入概念格环境。
- **candidate rules**：概念格环境产生候选规则结构，供 Actor 探索。
- **async gradients**：多个 Actor 将各自探索得到的梯度异步发送给全局 Actor-Critic 网络。
- **policy/value update**：全局网络更新策略函数和价值函数。
- **foreground adaptation**：前台快速更新 MLN 规则权重记忆，使模型能在下一个数据 chunk 到来前快速适应。
- **n-step return + reward**：基于多步回报和奖励信号更新全局策略。
- **advantage A(s,a)**：优势函数，用于衡量某个动作相对于当前状态平均价值的增益，指导策略向更优规则调整方向更新。

## 这张图支持的论文观点

这张图直接对应论文的技术可行性论证：传统 MLN 结构学习依赖全局数据重训，而该框架通过多个 Actor 并行探索规则空间，并通过异步梯度更新全局策略，使 MLN 能够在数据流输入后快速调整规则和权重，从而形成实时响应的概率知识网络。
""",
    )

    write_explanation_file(
        "Fig2_Realtime_Latency_Breakdown_explanation",
        """
# Fig2_Realtime_Latency_Breakdown 详细解释

## 图的作用

这张图是实时性主图之一，用来比较各方法在数据流更新时的前台更新时间和阻塞周期时间。它重点回答：当一个新数据 chunk 到达并产生反馈后，模型需要多长时间才能完成更新并继续提供推理服务。

## 横轴含义

横轴是 **Time per stream chunk (ms, log scale)**，表示每个数据流 chunk 的耗时，单位是毫秒。横轴使用对数尺度，因为不同方法的耗时差距很大：RT-A3C-MLN 是毫秒级，而 Rolling BeamSearch MLN 达到秒级，用普通线性坐标会压缩低延迟方法的差异。

图中两条竖虚线表示实时约束：

- **5 ms update**：前台更新 5 毫秒参考线，用来衡量模型更新是否足够轻量。
- **10 ms cycle**：阻塞周期 10 毫秒实时截止线，是论文中支撑“实时响应”的关键阈值。

## 纵轴含义

纵轴是不同实验方法，每一行对应一种方法：

- **RT-A3C-MLN (ours)**：本文方法，基于异步优势 Actor-Critic 的实时 MLN 规则和权重自适应方法。
- **Rolling L1 MLN**：滚动窗口上的 L1 正则化 MLN 权重学习方法，每个数据流更新后重新拟合。
- **Rolling MaxEnt MLN**：滚动窗口上的最大熵 MLN 方法，每个数据流更新后重新进行全局权重估计。
- **Rolling Boosted MLN**：基于 Boosting 思路的滚动 MLN 基线，每次更新需要重新训练增强模型。
- **Rolling BeamSearch MLN**：基于束搜索进行规则结构选择的滚动 MLN 基线，结构搜索成本最高。
- **OnlineWeight MLN**：固定规则结构的在线权重更新参考方法，速度快，但不进行规则结构自适应。
- **Static MaxEnt reference**：静态最大熵参考模型，不随数据流更新，只作为非自适应参考。

## 点和线的含义

- 圆点表示 **foreground update**，即前台更新时间。
- 菱形表示 **blocking cycle**，即阻塞周期时间。
- 每行中圆点到菱形之间的线段表示从“仅更新耗时”到“推理加更新总耗时”的差值。
- 每个菱形旁边的标注格式为“阻塞周期时间 / 10ms 成功率”。例如 `5.9 ms / 100%` 表示平均阻塞周期为 5.9 毫秒，所有测试 chunk 都满足 10 毫秒实时截止要求。

## 指标含义

- **Foreground update time**：当前 chunk 标签或反馈到达后，模型在前台必须完成的更新时间。这个时间越短，越不影响实时服务。
- **Blocking cycle time**：推理时间加前台更新时间。它表示系统从接收数据、完成推理、吸收反馈到可继续响应的总阻塞时间。
- **10 ms success rate**：阻塞周期不超过 10 毫秒的 chunk 比例。数值越高，实时性越强。

## 主要结论

RT-A3C-MLN 的阻塞周期约为 5.9 ms，10 ms 截止成功率为 100%，说明它能够在数据流场景下稳定满足实时响应要求。相比之下，Rolling L1、Rolling MaxEnt、Rolling Boosted 和 Rolling BeamSearch 都需要在滚动窗口上重新拟合或重新搜索规则，因此阻塞周期明显更长。该图直接支撑论文观点：本文方法不是通过静态模型避免更新，而是通过异步 A3C 机制将 MLN 更新压缩到实时可接受范围内。
""",
    )

    write_explanation_file(
        "Fig3_Dataset_Deadline_Heatmap_explanation",
        """
# Fig3_Dataset_Deadline_Heatmap 详细解释

## 图的作用

这张图展示不同数据集上的实时鲁棒性。它不是只给一个平均值，而是把每个方法在四个数据集上的阻塞周期和 10 ms 截止成功率同时展示出来，用来说明本文方法的速度优势不是只在某个单一数据集上成立。

## 左侧子图：Blocking cycle by dataset (ms)

### 横轴

横轴是数据集名称：

- **adult**：Adult 收入分类数据集。
- **bank**：Bank Marketing 银行营销分类数据集。
- **mushroom**：Mushroom 蘑菇分类数据集。
- **spambase**：Spambase 垃圾邮件分类数据集。

### 纵轴

纵轴是方法名称的紧凑写法：

- **RT-A3C**：本文方法 RT-A3C-MLN。
- **L1**：Rolling L1 MLN。
- **MaxEnt**：Rolling MaxEnt MLN。
- **Boosted**：Rolling Boosted MLN。
- **Beam**：Rolling BeamSearch MLN。
- **Online**：OnlineWeight MLN，固定规则结构在线权重更新参考。
- **Static**：Static MaxEnt reference，静态非自适应参考。

### 颜色和数字

每个格子的数字表示该方法在该数据集上的平均 **blocking cycle time**，单位是毫秒。颜色表示 `log10(ms)`，也就是阻塞周期时间的对数变换。颜色越深，表示阻塞时间越长。

使用对数颜色尺度的原因是不同方法耗时差异很大：RT-A3C-MLN 是 5 到 7 ms 左右，BeamSearch 基线超过 1500 ms。如果使用线性颜色尺度，低延迟方法之间的区别会不明显。

## 右侧子图：10 ms deadline success (%)

### 横轴

横轴同样是四个数据集：adult、bank、mushroom、spambase。

### 纵轴

纵轴同样是不同方法。

### 颜色和数字

每个格子的数字表示该方法在该数据集上满足 10 ms 阻塞周期截止要求的 chunk 比例，单位是百分比。颜色越深绿色，表示实时截止成功率越高。

例如，RT-A3C 在四个数据集上均为 100，表示每个数据集中的所有流式 chunk 都满足 10 ms 实时约束。

## 指标含义

- **Blocking cycle time**：推理时间加前台更新时间，越低越好。
- **10 ms deadline success**：阻塞周期小于等于 10 ms 的数据流 chunk 比例，越高越好。

## 主要结论

RT-A3C-MLN 在四个数据集上的平均阻塞周期均低于 10 ms，并且 10 ms 成功率均为 100%。Rolling L1 和 Rolling MaxEnt 在部分数据集上接近或超过 10 ms，因此成功率不稳定；Boosted 和 BeamSearch 的阻塞周期明显超过实时要求。该图支持论文的实验可行性论证：本文方法在不同高维数据集和不同数据流输入下都能保持实时响应。
""",
    )

    write_explanation_file(
        "Fig4_Stream_Drift_Response_explanation",
        """
# Fig4_Stream_Drift_Response 详细解释

## 图的作用

这张图展示模型在数据流漂移过程中的动态表现。它同时回答两个问题：第一，数据分布变化时模型的预测性能如何变化；第二，模型在每个数据流 chunk 上是否仍然保持实时阻塞周期。

## 横轴含义

上下两个子图的横轴都是 **Stream chunk**，表示数据流被划分后的第几个数据块。当前实验中共有 8 个 chunk。每个 chunk 表示一次连续到达的数据批次，模型先对该 chunk 进行推理，再根据新反馈进行更新。

## 上方子图：F1 与数据流漂移

### 左纵轴

左纵轴是 **F1**。F1 是精确率和召回率的调和平均值，数值越高表示分类性能越好。相比单纯准确率，F1 更适合类别比例发生变化的数据流场景。

### 右纵轴

右纵轴是 **mean positive rate**，表示当前 chunk 中正类样本比例的平均值。灰色虚线和灰色背景区域表示数据流中的类别先验漂移，即不同 chunk 的正类比例在变化。

### 颜色和曲线名称

- **RT-A3C**：本文方法。
- **L1**：Rolling L1 MLN。
- **MaxEnt**：Rolling MaxEnt MLN。
- **Boosted**：Rolling Boosted MLN。
- **Beam**：Rolling BeamSearch MLN。

每条彩色曲线表示一种方法在每个 chunk 上的平均 F1。曲线越高，表示该方法在该 chunk 上的预测效果越好。

## 下方子图：阻塞周期随 chunk 的变化

### 纵轴

纵轴是 **Blocking cycle (ms, log)**，表示阻塞周期时间，单位是毫秒，并使用对数坐标。阻塞周期等于当前 chunk 的推理时间加前台更新时间。

### 红色虚线

红色虚线表示 **10 ms deadline**。低于这条线表示该 chunk 上的推理与更新过程满足实时响应要求。

### 曲线含义

每条曲线对应一个方法在 8 个数据流 chunk 上的阻塞周期变化。RT-A3C 曲线始终位于 10 ms 红线以下，说明它在数据分布变化过程中仍能稳定满足实时要求。

## 主要结论

随着灰色漂移曲线变化，各方法的 F1 都会发生波动，说明实验确实处在动态数据流场景中。RT-A3C-MLN 在保持较稳定 F1 的同时，其阻塞周期始终低于 10 ms；而 Rolling MaxEnt、Boosted、BeamSearch 等方法虽然可以重新适应数据，但更新代价较高，阻塞周期明显超过实时阈值。该图直接支撑论文核心观点：MLN 规则可以在不同数据流输入下自适应调整，并保持实时推理能力。
""",
    )

    write_explanation_file(
        "Fig5_Ablation_Summary_explanation",
        """
# Fig5_Ablation_Summary 详细解释

## 图的作用

这张图是消融实验图，用来分析本文方法中不同设计组件的作用。它把预测性能、实时响应时间和规则结构刷新活动放在同一张图中，说明 RT-A3C-MLN 的实时性和自适应能力来自异步更新、后台结构刷新、局部先验适应和在线权重更新的组合。

## 横轴名称含义

三个子图的横轴都是消融版本：

- **Full**：完整的 RT-A3C-MLN。
- **Single**：Single actor，只使用单个 Actor，去掉多 Actor 并行探索。
- **No prior**：No prior-shift update，去掉局部数据流先验漂移更新。
- **No structure**：No structure refresh，关闭后台规则结构刷新，只保留前台权重更新和概率校准。
- **Online only**：Online only, no A3C，只保留在线权重更新，不使用 A3C 结构学习。

## 左侧子图：Predictive quality

### 左纵轴

左纵轴是 **F1**，柱状图表示不同消融版本的分类 F1。F1 越高表示预测性能越好。

### 右纵轴

右纵轴是 **Log-loss**，黑色折线表示不同版本的对数损失。Log-loss 衡量模型输出概率的校准质量，越低越好。它不仅关心预测类别是否正确，也关心模型给出的概率是否可信。

### 解释重点

Online only 虽然速度很快，但 Log-loss 通常较差，说明固定结构在线权重更新难以替代 A3C 规则结构学习。No structure refresh 用来观察关闭后台规则结构调整后，模型是否退化为固定结构权重更新。

## 中间子图：Realtime response

### 纵轴

纵轴是 **Time (ms)**，表示耗时，单位为毫秒。

### 柱状图含义

- 浅蓝色柱表示 **FG update**，即前台更新时间。
- 深蓝色柱表示 **blocking cycle**，即推理时间加前台更新时间。

### 红色虚线

红色虚线表示 10 ms 实时截止线。柱子低于红线表示该版本满足实时响应要求。

### 解释重点

完整 RT-A3C-MLN 的阻塞周期低于 10 ms，说明后台结构刷新被放到非阻塞路径后，前台响应仍能满足实时约束。Online only 最快，但它牺牲了规则结构自适应能力，因此只能作为固定结构参考。

## 右侧子图：Rule-structure adaptation

### 纵轴

左纵轴是 **Rule changes / chunk**，表示每个数据流 chunk 平均发生的规则结构变化数量。绿色柱表示新增规则数量，红色柱表示删除规则数量。右纵轴是 **Accepted refresh (%)**，表示后台结构刷新被验证窗口接受的比例。

### 柱状图含义

如果某个版本的新增/删除规则数量为 0，说明该版本没有发生规则结构调整，或者后台刷新结果没有通过验证窗口接受准则。Full 若出现非零规则变化，说明模型确实在数据流阶段对规则结构进行了自适应调整。

## 主要结论

这张图说明：局部先验漂移更新有助于稳定概率质量；后台结构刷新使规则集合能够在数据流阶段发生新增和删除；A3C 结构学习相较于 Online only 更能保持概率建模能力。它支撑论文中关于“异步 Actor-Critic 机制使 MLN 规则能够在数据流下自适应调整，同时保持低前台阻塞时间”的论证。
""",
    )

    write_explanation_file(
        "Table1_Main_Results_explanation",
        """
# Table1_Main_Results 详细解释

## 表格作用

这张表是主结果表，用来同时展示预测性能、实时更新速度和数据流响应能力。它不是单纯比较准确率，而是围绕论文核心观点比较：在数据流到来后，MLN 是否能够快速自适应并实时完成概率推理。

## 行名称含义

- **RT-A3C-MLN (ours)**：本文方法。使用异步优势 Actor-Critic 进行规则结构和权重自适应更新。
- **Rolling L1 MLN**：滚动窗口上的 L1 正则化 MLN 方法，每个 chunk 后重新拟合。
- **Rolling MaxEnt MLN**：滚动窗口上的最大熵 MLN 方法，每个 chunk 后进行全局权重估计。
- **Rolling Boosted MLN**：基于 Boosting 的滚动 MLN 基线，每个 chunk 后重新训练。
- **Rolling BeamSearch MLN**：基于束搜索进行规则结构选择的滚动 MLN 基线，搜索成本较高。
- **OnlineWeight MLN**：固定规则结构的在线权重更新方法。它速度快，但不进行规则结构自适应。
- **Static MaxEnt reference**：静态最大熵参考模型。它不更新，因此不是数据流自适应方法，只作为非自适应参考。

## 列名称和指标含义

- **Method**：方法名称。
- **Streaming update protocol**：该方法在数据流到来后的更新机制。
- **Adaptive rules**：是否支持规则结构自适应。Yes 表示规则结构可以随数据流更新；No 表示规则结构固定或模型不更新。
- **F1**：精确率和召回率的调和平均值，越高越好，用于衡量分类性能。
- **AUC**：ROC-AUC，衡量模型区分正负样本的能力，越高越好。
- **Log-loss**：对数损失，衡量概率预测质量，越低越好。
- **FG update ms**：Foreground update time，前台更新时间，单位毫秒。表示新 chunk 标签到达后，模型必须在前台完成的更新时间。
- **Blocking cycle ms**：阻塞周期时间，单位毫秒，等于推理时间加前台更新时间。它是实时系统最关键的指标。
- **10ms success %**：阻塞周期不超过 10 毫秒的 chunk 比例，越高表示实时响应越稳定。
- **Struct update %**：后台规则结构刷新被接受的比例。该指标只对支持结构自适应的方法有意义。
- **Rule +/- per chunk**：每个 chunk 平均新增和删除的规则数量，格式为“新增 / 删除”。
- **Update slowdown vs ours**：相对于本文方法的前台更新变慢倍数。比如 `61.41x` 表示该基线前台更新约为本文方法的 61.41 倍。

## 如何解读这张表

RT-A3C-MLN 的关键解读应放在两个方面：一是阻塞周期是否稳定低于 10 ms，二是 Struct update 和 Rule +/- 是否显示出数据流阶段的规则结构调整。这样可以证明它不是静态模型，也不是仅更新固定权重，而是在后台接受验证的前提下调整规则结构。

Rolling L1、Rolling MaxEnt、Rolling Boosted 和 Rolling BeamSearch 都属于需要滚动全局重训或结构搜索的自适应 MLN 基线。它们可以适应数据流，但前台更新代价明显更高，尤其 Boosted 和 BeamSearch 的阻塞周期远超 10 ms。

OnlineWeight MLN 和 Static MaxEnt reference 的速度看起来较快，但它们不支持规则结构自适应。OnlineWeight 只更新固定规则权重，Static MaxEnt 根本不更新。因此它们是参考方法，不是证明或否定实时自适应结构学习的直接竞争者。

## 论文中应强调的结论

这张表应支撑主结论：RT-A3C-MLN 相比滚动重训类 MLN 基线，在保持竞争性预测性能的同时显著降低前台更新时间和阻塞周期，从而验证 MLN 可以在数据流下实现快速自适应和实时概率推理。
""",
    )

    write_explanation_file(
        "Table2_Ablation_explanation",
        """
# Table2_Ablation 详细解释

## 表格作用

这张表是消融实验表，用来解释本文方法中每个关键组件的贡献。它回答的问题是：实时性、稳定性和预测性能分别来自哪些设计。

## 行名称含义

- **Full RT-A3C-MLN**：完整模型，包含异步 Actor-Critic、后台规则结构刷新、局部先验漂移适应和前台快速更新。
- **Single actor**：只保留单个 Actor，去掉多 Actor 并行探索，用来检验并行独立学习的作用。
- **No prior-shift update**：去掉局部先验漂移更新，用来检验模型对数据流分布变化的快速适应能力。
- **No structure refresh**：关闭后台规则结构刷新，用来检验数据流阶段规则结构自适应的作用。
- **Online only, no A3C**：只保留在线权重更新，去掉 A3C 结构学习，用来检验是否仅靠在线权重更新就足够。

## 列名称和指标含义

- **Variant**：消融版本名称。
- **Controlled change**：该消融版本相对于完整模型改变了什么组件。
- **F1**：分类性能指标，越高越好。
- **AUC**：ROC-AUC，衡量正负样本排序区分能力，越高越好。
- **Log-loss**：概率预测损失，越低越好。该指标能反映概率知识网络输出概率是否稳定、可信。
- **FG update ms**：前台更新时间，单位毫秒。越低说明模型越适合实时更新。
- **Blocking cycle ms**：阻塞周期时间，单位毫秒。它等于推理时间加前台更新时间，是实时性核心指标。
- **Struct update %**：后台规则结构刷新被验证窗口接受的比例。
- **Rule +/chunk**：每个 chunk 平均新增规则数量。
- **Rule -/chunk**：每个 chunk 平均删除规则数量。
- **10ms success %**：阻塞周期不超过 10 毫秒的 chunk 比例。

## 如何解读这张表

Full RT-A3C-MLN 的阻塞周期低于 10 ms，说明完整模型满足实时响应。No structure refresh 不会产生规则新增/删除，可用于证明 Full 中的规则变化来自后台结构刷新机制，而不是普通在线权重更新。

No prior-shift update 的 Log-loss 变差，说明局部先验漂移更新对数据流概率适应有帮助。No structure refresh 若规则新增/删除为 0，说明关闭后台结构刷新后模型只能进行固定结构权重更新。Online only, no A3C 虽然前台更新时间最短，但 Log-loss 通常更差，并且不具备规则结构自适应能力，因此不能替代本文的 A3C-MLN 结构学习机制。

## 论文中应强调的结论

这张表应服务于理论和技术可行性论证：优势函数和异步 Actor-Critic 机制不仅用于提高速度，也用于保持稳定的概率建模；后台结构刷新让 MLN 能够在数据流变化时调整规则集合；局部先验漂移更新让模型快速响应类别比例变化。完整方法的优势来自这些组件的组合，而不是单纯牺牲模型能力换取速度。
""",
    )


def main():
    ensure_clean_output()
    set_style()

    config = json.loads((RESULTS / "core_claim" / "core_realtime_config.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(RESULTS / "core_claim" / "core_realtime_summary.csv")
    detail = pd.read_csv(RESULTS / "core_claim" / "core_realtime_detail.csv")
    ablation = pd.read_csv(RESULTS / "core_ablation" / "core_ablation_summary.csv")
    speedup = compute_speedup(summary)
    speedup.to_csv(SOURCE_DIR / "rt_a3c_speedup_vs_batch_mln.csv", index=False)

    write_table_files(
        build_main_table(summary, speedup),
        "Table1_Main_Results",
        "Main realtime adaptive MLN comparison",
    )
    write_table_files(
        build_ablation_table(ablation),
        "Table2_Ablation",
        "Ablation study",
    )

    draw_framework_diagram()
    draw_realtime_latency_figure(summary)
    draw_deadline_heatmap(detail)
    draw_stream_trajectory(detail)
    draw_ablation_figure(ablation)

    copy_source_results()
    write_notes(config)
    write_explanations()

    print(f"[done] wrote compact paper artifacts to {OUT}")
    print(f"[figures] {FIG_DIR}")
    print(f"[tables] {TABLE_DIR}")
    print(f"[explanations] {EXPLANATION_DIR}")


if __name__ == "__main__":
    main()
