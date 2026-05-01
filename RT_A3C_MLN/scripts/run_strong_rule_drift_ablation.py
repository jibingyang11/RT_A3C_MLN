from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.models import _safe_proba
from scripts.run_streaming_experiments import AdaptiveA3CStreamModel
from scripts.run_strong_rule_drift_experiment import (
    EMERGING_RULES,
    make_emerging_rule_stream,
    metrics_from_prob,
)


OUT_DIR = ROOT / "results" / "strong_rule_drift_ablation"
ARTIFACT_ROOT = Path(os.environ.get("PAPER_ARTIFACTS_DIR", ROOT / "paper_artifacts"))
ARTIFACT_DIR = ARTIFACT_ROOT / "strong_rule_drift_ablation"
FIG_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"


VARIANT_ORDER = [
    "Full_RT_A3C",
    "No_Structure_Refresh",
    "No_Prior_Shift",
    "Online_Only_No_A3C",
]

LABELS = {
    "Full_RT_A3C": "Full RT-A3C",
    "No_Structure_Refresh": "No structure",
    "No_Prior_Shift": "No prior-shift",
    "Online_Only_No_A3C": "Online only",
}

COLORS = {
    "Full_RT_A3C": "#c1121f",
    "No_Structure_Refresh": "#457b9d",
    "No_Prior_Shift": "#2a9d8f",
    "Online_Only_No_A3C": "#adb5bd",
}


class OnlineOnlyFullRuleModel:
    category = "online_only_no_a3c"

    def __init__(self, random_state):
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=7e-5,
            l1_ratio=0.05,
            learning_rate="optimal",
            average=True,
            random_state=random_state,
        )
        self.classes = np.array([0, 1])
        self.active_rules = []
        self.last_background_update_sec = 0.0
        self.last_structure_refreshed = 0.0
        self.last_rule_additions = 0
        self.last_rule_deletions = 0

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.partial_fit(x_train, y_train, classes=self.classes)
        return self

    def predict_proba(self, x):
        return _safe_proba(self.model.predict_proba(x))

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.model.partial_fit(x_chunk, y_chunk)
        return time.perf_counter() - start


def make_a3c(args, *, workers=None, prior_shift_lr=None, structure_refresh=True):
    return AdaptiveA3CStreamModel(
        random_state=args.seed,
        workers=args.a3c_workers if workers is None else workers,
        episodes=args.a3c_episodes,
        steps=args.a3c_steps,
        rules=args.a3c_rules,
        ultra_fast_update=True,
        max_active_rules=args.a3c_max_active_rules,
        rolling_window=args.a3c_rolling_window,
        calibration_window=args.a3c_calibration_window,
        refit_interval=args.a3c_refit_interval,
        async_refit=True,
        prior_shift_lr=args.a3c_prior_shift_lr if prior_shift_lr is None else prior_shift_lr,
        prior_shift_clip=args.a3c_prior_shift_clip,
        structure_refresh=structure_refresh,
        structure_refresh_interval=args.a3c_refit_interval,
        structure_refresh_episodes=args.a3c_structure_episodes,
        structure_refresh_tolerance=args.a3c_structure_tolerance,
    )


def build_variants(args):
    return {
        "Full_RT_A3C": make_a3c(args, structure_refresh=True),
        "No_Structure_Refresh": make_a3c(args, structure_refresh=False),
        "No_Prior_Shift": make_a3c(args, prior_shift_lr=0.0, structure_refresh=True),
        "Online_Only_No_A3C": OnlineOnlyFullRuleModel(args.seed),
    }


def run_experiment(args):
    x_train, y_train, x_val, y_val, chunks = make_emerging_rule_stream(args)
    rows = []
    for variant, model in build_variants(args).items():
        print(f"[strong-drift-ablation] variant={variant}")
        start = time.perf_counter()
        model.fit(x_train, y_train, x_val, y_val)
        initial_fit = time.perf_counter() - start
        for chunk in chunks:
            x_chunk = chunk["x"]
            y_chunk = chunk["y"]
            start = time.perf_counter()
            prob = model.predict_proba(x_chunk)[:, 1]
            inference_time = time.perf_counter() - start
            update_time = model.update(x_chunk, y_chunk)
            cycle_time = inference_time + update_time
            active_rules = getattr(model, "active_rules", []) or []
            rows.append(
                {
                    "variant": variant,
                    "chunk": chunk["chunk"],
                    "phase": chunk["phase"],
                    "positive_rate": chunk["positive_rate"],
                    "emerging_rule_support": chunk["emerging_support"],
                    "initial_fit_time_sec": initial_fit,
                    "foreground_update_time_sec": update_time,
                    "background_update_time_sec": getattr(model, "last_background_update_sec", 0.0),
                    "blocking_cycle_time_sec": cycle_time,
                    "meets_10ms_cycle": float(cycle_time <= 0.010),
                    "structure_refreshed": getattr(model, "last_structure_refreshed", 0.0),
                    "rule_additions": getattr(model, "last_rule_additions", 0),
                    "rule_deletions": getattr(model, "last_rule_deletions", 0),
                    "active_rule_count": len(active_rules),
                    "active_emerging_rule_count": int(len(set(active_rules).intersection(set(EMERGING_RULES)))),
                    **metrics_from_prob(y_chunk, prob, inference_time),
                }
            )
    return pd.DataFrame(rows)


def summarize(detail, args):
    metrics = [
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "foreground_update_time_sec",
        "blocking_cycle_time_sec",
        "meets_10ms_cycle",
        "structure_refreshed",
        "rule_additions",
        "rule_deletions",
        "active_emerging_rule_count",
    ]
    all_summary = detail.groupby("variant", observed=True)[metrics].mean().reset_index()
    post = detail[detail["chunk"] > args.drift_chunk]
    post_summary = post.groupby("variant", observed=True)[metrics].mean().reset_index()
    post_summary = post_summary.rename(columns={col: f"post_{col}" for col in metrics})
    summary = all_summary.merge(post_summary, on="variant", how="left")

    recovery_rows = []
    for variant, sub in detail[detail["chunk"] >= args.drift_chunk].groupby("variant", observed=True):
        recovered = sub[(sub["chunk"] > args.drift_chunk) & (sub["f1"] >= args.recovery_f1)]
        recovery_chunk = int(recovered["chunk"].min()) if len(recovered) else -1
        recovery_rows.append(
            {
                "variant": variant,
                "recovery_chunk": recovery_chunk,
                "recovery_lag_chunks": recovery_chunk - args.drift_chunk if recovery_chunk > 0 else -1,
            }
        )
    summary = summary.merge(pd.DataFrame(recovery_rows), on="variant", how="left")
    summary["order"] = summary["variant"].map({v: i for i, v in enumerate(VARIANT_ORDER)})
    return summary.sort_values("order").drop(columns=["order"])


def fmt(value, digits=3):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def make_table(summary):
    rows = []
    full = summary.set_index("variant").loc["Full_RT_A3C"]
    for _, row in summary.iterrows():
        delta = row["post_f1"] - full["post_f1"]
        rows.append(
            {
                "Variant": LABELS[row["variant"]],
                "Controlled change": {
                    "Full_RT_A3C": "complete model",
                    "No_Structure_Refresh": "disables background rule-structure refresh",
                    "No_Prior_Shift": "removes local prior-shift adaptation",
                    "Online_Only_No_A3C": "removes A3C structure learning",
                }[row["variant"]],
                "Post-drift F1": fmt(row["post_f1"]),
                "ΔF1 vs Full": fmt(delta, 3),
                "Post-drift Log-loss": fmt(row["post_log_loss"]),
                "Recovery lag": "not recovered" if int(row["recovery_lag_chunks"]) < 0 else str(int(row["recovery_lag_chunks"])),
                "Blocking cycle ms": fmt(row["blocking_cycle_time_sec"] * 1000),
                "10ms success %": fmt(row["meets_10ms_cycle"] * 100, 1),
                "Struct update %": fmt(row["structure_refreshed"] * 100, 1),
                "Rule +/- per chunk": f"{fmt(row['rule_additions'], 2)} / {fmt(row['rule_deletions'], 2)}",
                "Active emerging rules": fmt(row["active_emerging_rule_count"], 2),
            }
        )
    return pd.DataFrame(rows)


def save_table(table_df):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(TABLE_DIR / "Table4_Strong_Drift_Ablation.csv", index=False)
    header = "| " + " | ".join(table_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(row[col]) for col in table_df.columns) + " |"
        for _, row in table_df.iterrows()
    )
    md = "# Strong rule-drift structure ablation\n\n" + "\n".join([header, sep, body])
    (TABLE_DIR / "Table4_Strong_Drift_Ablation.md").write_text(md, encoding="utf-8")


def plot_ablation(detail, summary, args):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    df = summary.set_index("variant").loc[VARIANT_ORDER].reset_index()
    x = np.arange(len(df))
    labels = [LABELS[v].replace(" ", "\n") for v in df["variant"]]
    colors = [COLORS[v] for v in df["variant"]]

    axes[0].bar(x, df["post_f1"], color=colors, width=0.64)
    axes[0].set_title("Post-drift F1")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)

    axes[1].bar(x, df["post_log_loss"], color=colors, width=0.64)
    axes[1].set_title("Post-drift Log-loss")
    axes[1].set_ylabel("Log-loss")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)

    for variant in ["Full_RT_A3C", "No_Structure_Refresh"]:
        sub = detail[detail["variant"] == variant].sort_values("chunk")
        axes[2].plot(
            sub["chunk"],
            sub["active_emerging_rule_count"],
            marker="o",
            lw=2.2,
            color=COLORS[variant],
            label=LABELS[variant],
        )
    axes[2].axvline(args.drift_chunk, color="#343a40", ls="--", lw=1.0)
    axes[2].set_title("Emerging-rule activation")
    axes[2].set_xlabel("Stream chunk")
    axes[2].set_ylabel("Active emerging rules")
    axes[2].legend(frameon=False, loc="upper left", fontsize=8.5)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Strong-drift ablation: removing structure refresh blocks emerging-rule adaptation", y=0.99, fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Fig8_Strong_Drift_Structure_Ablation.{ext}", dpi=260 if ext == "png" else None)
    plt.close(fig)


def write_explanation(args):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    text = f"""# 强规则漂移结构刷新消融实验说明

## 实验目的

该实验在强规则漂移压力测试数据流上进一步做消融，专门验证 No structure refresh 是否会在新规则出现后退化。实验保持数据流、漂移点、A3C 参数和前台更新机制一致，只关闭后台规则结构刷新。

## 变量设置

- **Full RT-A3C**：完整模型，允许后台结构刷新、规则新增和规则删除。
- **No structure**：关闭后台规则结构刷新，只保留前台在线权重更新、局部校准和已有规则上的适应。
- **No prior-shift**：关闭局部先验漂移更新，用于区分概率校准和结构刷新作用。
- **Online only**：移除 A3C 结构学习，仅保留在线权重更新。

## 关键指标

- **Post-drift F1**：只统计规则漂移后的 chunk，越高表示恢复越好。
- **ΔF1 vs Full**：相对于完整 RT-A3C 的漂移后 F1 差值。
- **Struct update %**：后台结构刷新被接受的比例。
- **Rule +/- per chunk**：每个 chunk 的平均规则新增和删除数量。
- **Active emerging rules**：模型当前激活的新规则数量，是证明结构自适应的核心指标。

## 论文解释

如果 No structure 的 Post-drift F1 低于 Full，并且 Struct update、Rule +/- 和 Active emerging rules 均明显不足，就说明在强规则漂移下，仅依赖已有规则权重更新不能充分适应后半段新出现的判别规则。该结果可作为消融证据，证明后台规则结构刷新不是装饰性模块，而是 RT-A3C-MLN 在强规则漂移下恢复性能的关键机制。
"""
    (ARTIFACT_DIR / "Strong_Drift_Ablation_explanation.md").write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--n-features", type=int, default=420)
    parser.add_argument("--train-size", type=int, default=3600)
    parser.add_argument("--val-size", type=int, default=900)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--drift-chunk", type=int, default=5)
    parser.add_argument("--recovery-f1", type=float, default=0.68)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=6)
    parser.add_argument("--a3c-steps", type=int, default=5)
    parser.add_argument("--a3c-rules", type=int, default=84)
    parser.add_argument("--a3c-max-active-rules", type=int, default=32)
    parser.add_argument("--a3c-rolling-window", type=int, default=2400)
    parser.add_argument("--a3c-calibration-window", type=int, default=400)
    parser.add_argument("--a3c-refit-interval", type=int, default=1)
    parser.add_argument("--a3c-structure-episodes", type=int, default=3)
    parser.add_argument("--a3c-structure-tolerance", type=float, default=0.015)
    parser.add_argument("--a3c-prior-shift-lr", type=float, default=0.24)
    parser.add_argument("--a3c-prior-shift-clip", type=float, default=1.7)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail = run_experiment(args)
    summary = summarize(detail, args)
    table = make_table(summary)

    detail.to_csv(OUT_DIR / "strong_drift_ablation_detail.csv", index=False)
    summary.to_csv(OUT_DIR / "strong_drift_ablation_summary.csv", index=False)
    (OUT_DIR / "strong_drift_ablation_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    save_table(table)
    plot_ablation(detail, summary, args)
    write_explanation(args)

    print(table.to_string(index=False))
    print(f"[done] results: {OUT_DIR}")
    print(f"[done] artifacts: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
