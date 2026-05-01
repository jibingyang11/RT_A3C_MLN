from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS, load_dataset
from mln_a3c_experiment.models import _safe_proba
from mln_a3c_experiment.rules import RuleFeatureBuilder
from scripts.run_core_realtime_experiment import OnlineWeightMLN, StaticMaxEntReference
from scripts.run_streaming_experiments import AdaptiveA3CStreamModel, make_prior_drift_chunks
from scripts.run_strong_rule_drift_experiment import (
    EMERGING_RULES,
    make_emerging_rule_stream,
)

try:
    from river import ensemble, forest, linear_model, optim, preprocessing, tree
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The modern stream baseline experiment requires river. "
        "Install it with: python -m pip install river"
    ) from exc


OUT_DIR = ROOT / "results" / "modern_baselines"
ARTIFACT_DIR = ROOT / "paper_artifacts" / "modern_baselines"
FIG_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"
EXPLAIN_DIR = ARTIFACT_DIR / "explanations"

METHOD_ORDER = [
    "RT_A3C_Realtime_MLN",
    "OnlineWeight_MLN",
    "Static_MaxEnt_Reference",
    "River_LogReg_2021",
    "Hoeffding_Adaptive_Tree_2009",
    "Adaptive_Random_Forest_2017",
    "Streaming_Random_Patches_2019",
]

METHOD_LABELS = {
    "RT_A3C_Realtime_MLN": "RT-A3C-MLN (ours)",
    "OnlineWeight_MLN": "OnlineWeight MLN",
    "Static_MaxEnt_Reference": "Static MaxEnt",
    "River_LogReg_2021": "River Online LR",
    "Hoeffding_Adaptive_Tree_2009": "HAT",
    "Adaptive_Random_Forest_2017": "ARF",
    "Streaming_Random_Patches_2019": "SRP",
}

METHOD_YEARS = {
    "RT_A3C_Realtime_MLN": "2026 / ours",
    "OnlineWeight_MLN": "fixed-rule online reference",
    "Static_MaxEnt_Reference": "static reference",
    "River_LogReg_2021": "2021 framework",
    "Hoeffding_Adaptive_Tree_2009": "2009 stream tree",
    "Adaptive_Random_Forest_2017": "2017 stream ensemble",
    "Streaming_Random_Patches_2019": "2019 stream ensemble",
}

COLORS = {
    "RT_A3C_Realtime_MLN": "#c1121f",
    "OnlineWeight_MLN": "#adb5bd",
    "Static_MaxEnt_Reference": "#8d99ae",
    "River_LogReg_2021": "#2a9d8f",
    "Hoeffding_Adaptive_Tree_2009": "#457b9d",
    "Adaptive_Random_Forest_2017": "#f4a261",
    "Streaming_Random_Patches_2019": "#6d597a",
}


def sigmoid(score: np.ndarray) -> np.ndarray:
    score = np.clip(score, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-score))


def metrics_from_prob(y_true, prob, inference_sec):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = (prob >= 0.5).astype(int)
    row = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "inference_time_sec": float(inference_sec),
        "latency_ms_per_sample": float(inference_sec * 1000.0 / max(1, len(y_true))),
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        row["roc_auc"] = np.nan
    return row


def array_row_to_dict(row):
    return {f"r{i}": float(value) for i, value in enumerate(row)}


def river_prob_one(model, row):
    proba = model.predict_proba_one(array_row_to_dict(row))
    if not proba:
        return 0.5
    return float(proba.get(1, proba.get(True, 0.0)))


class RiverStreamModel:
    category = "modern_stream_baseline"

    def __init__(self, model, random_state=13, replay_fit_passes=1, warmup_limit=None):
        self.model = model
        self.random_state = random_state
        self.replay_fit_passes = replay_fit_passes
        self.warmup_limit = warmup_limit

    def fit(self, x_train, y_train, x_val, y_val):
        if self.warmup_limit is not None and len(y_train) > self.warmup_limit:
            rng = np.random.default_rng(self.random_state)
            keep = np.sort(rng.choice(len(y_train), size=self.warmup_limit, replace=False))
            x_train = x_train[keep]
            y_train = y_train[keep]
        for _ in range(self.replay_fit_passes):
            for row, label in zip(x_train, y_train):
                self.model.learn_one(array_row_to_dict(row), int(label))
        return self

    def predict_proba(self, x):
        prob = np.array([river_prob_one(self.model, row) for row in x], dtype=float)
        return _safe_proba(prob)

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        for row, label in zip(x_chunk, y_chunk):
            self.model.learn_one(array_row_to_dict(row), int(label))
        return time.perf_counter() - start


def make_river_logreg():
    return preprocessing.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.SGD(0.025),
        l2=1e-4,
        intercept_lr=0.01,
    )


def make_hat(seed):
    return tree.HoeffdingAdaptiveTreeClassifier(
        grace_period=80,
        max_depth=12,
        leaf_prediction="nba",
        seed=seed,
    )


def make_arf(seed, n_models):
    return forest.ARFClassifier(
        n_models=n_models,
        max_features="sqrt",
        grace_period=80,
        max_depth=12,
        seed=seed,
    )


def make_srp(seed, n_models):
    base = tree.HoeffdingAdaptiveTreeClassifier(
        grace_period=80,
        max_depth=12,
        leaf_prediction="nba",
        seed=seed,
    )
    return ensemble.SRPClassifier(
        model=base,
        n_models=n_models,
        subspace_size=0.6,
        training_method="patches",
        seed=seed,
    )


def build_modern_models(args):
    return {
        "RT_A3C_Realtime_MLN": AdaptiveA3CStreamModel(
            random_state=args.seed,
            workers=args.a3c_workers,
            episodes=args.a3c_episodes,
            steps=args.a3c_steps,
            rules=args.a3c_rules,
            ultra_fast_update=True,
            max_active_rules=args.a3c_max_active_rules,
            rolling_window=args.a3c_rolling_window,
            calibration_window=args.a3c_calibration_window,
            refit_interval=args.a3c_refit_interval,
            async_refit=True,
            prior_shift_lr=args.a3c_prior_shift_lr,
            prior_shift_clip=args.a3c_prior_shift_clip,
            structure_refresh=True,
            structure_refresh_interval=args.a3c_refit_interval,
            structure_refresh_episodes=args.a3c_structure_episodes,
            structure_refresh_tolerance=args.a3c_structure_tolerance,
        ),
        "OnlineWeight_MLN": OnlineWeightMLN(args.seed),
        "Static_MaxEnt_Reference": StaticMaxEntReference(args.seed),
        "River_LogReg_2021": RiverStreamModel(make_river_logreg(), args.seed, warmup_limit=args.river_warmup_limit),
        "Hoeffding_Adaptive_Tree_2009": RiverStreamModel(make_hat(args.seed), args.seed, warmup_limit=args.river_warmup_limit),
        "Adaptive_Random_Forest_2017": RiverStreamModel(
            make_arf(args.seed, args.river_trees),
            args.seed,
            warmup_limit=args.river_warmup_limit,
        ),
        "Streaming_Random_Patches_2019": RiverStreamModel(
            make_srp(args.seed, args.river_trees),
            args.seed,
            warmup_limit=args.river_warmup_limit,
        ),
    }


def model_category(model):
    return getattr(model, "category", "rt_a3c_realtime")


def run_core_dataset(dataset, args):
    frame, target = load_dataset(dataset, ROOT / "data" / "raw", max_samples=args.max_samples)
    y = frame[target].astype(int).to_numpy()
    x = frame.drop(columns=[target])
    x_initial, x_stream, y_initial, y_stream = train_test_split(
        x, y, test_size=args.stream_fraction, random_state=args.seed, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_initial, y_initial, test_size=0.25, random_state=args.seed, stratify=y_initial
    )
    builder = RuleFeatureBuilder(
        max_rules=args.max_rules,
        max_single_literals=args.max_literals,
        numeric_bins=args.numeric_bins,
        min_support=args.min_support,
        random_state=args.seed,
    )
    x_train_rules = builder.fit_transform(x_train, y_train)
    x_val_rules = builder.transform(x_val)
    x_stream_rules = builder.transform(x_stream)
    chunks = make_prior_drift_chunks(
        x_stream_rules,
        y_stream,
        chunks=args.chunks,
        chunk_size=args.chunk_size,
        random_state=args.seed + 101,
    )

    rows = []
    for method, model in build_modern_models(args).items():
        print(f"[modern-core] dataset={dataset} method={method}")
        start = time.perf_counter()
        model.fit(x_train_rules, y_train, x_val_rules, y_val)
        fit_time = time.perf_counter() - start
        for chunk_id, drift_rate, x_chunk, y_chunk in chunks:
            start = time.perf_counter()
            prob = model.predict_proba(x_chunk)[:, 1]
            inference_time = time.perf_counter() - start
            update_time = model.update(x_chunk, y_chunk)
            cycle_time = inference_time + update_time
            rows.append(
                {
                    "experiment": "core_stream",
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "method_year": METHOD_YEARS.get(method, ""),
                    "category": model_category(model),
                    "chunk": chunk_id,
                    "phase": "stream",
                    "drift_positive_rate": drift_rate,
                    "initial_fit_time_sec": fit_time,
                    "foreground_update_time_sec": update_time,
                    "background_update_time_sec": getattr(model, "last_background_update_sec", 0.0),
                    "blocking_cycle_time_sec": cycle_time,
                    "meets_10ms_cycle": float(cycle_time <= 0.010),
                    "structure_refreshed": getattr(model, "last_structure_refreshed", 0.0),
                    "rule_additions": getattr(model, "last_rule_additions", 0),
                    "rule_deletions": getattr(model, "last_rule_deletions", 0),
                    "active_rule_count": getattr(model, "last_active_rule_count", 0),
                    "active_emerging_rule_count": np.nan,
                    **metrics_from_prob(y_chunk, prob, inference_time),
                }
            )
    return rows


def run_strong_drift(args):
    x_train, y_train, x_val, y_val, chunks = make_emerging_rule_stream(args)
    rows = []
    for method, model in build_modern_models(args).items():
        print(f"[modern-strong] method={method}")
        start = time.perf_counter()
        model.fit(x_train, y_train, x_val, y_val)
        fit_time = time.perf_counter() - start
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
                    "experiment": "strong_rule_drift",
                    "dataset": "emerging_rule_pressure",
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "method_year": METHOD_YEARS.get(method, ""),
                    "category": model_category(model),
                    "chunk": chunk["chunk"],
                    "phase": chunk["phase"],
                    "drift_positive_rate": chunk["positive_rate"],
                    "initial_fit_time_sec": fit_time,
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
    return rows


def summarize_core(detail):
    metrics = [
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "latency_ms_per_sample",
        "foreground_update_time_sec",
        "blocking_cycle_time_sec",
        "background_update_time_sec",
        "meets_10ms_cycle",
        "structure_refreshed",
        "rule_additions",
        "rule_deletions",
        "active_rule_count",
        "initial_fit_time_sec",
    ]
    core = detail[detail["experiment"] == "core_stream"].copy()
    if core.empty:
        return pd.DataFrame()
    summary = core.groupby(["method", "method_label", "method_year", "category"], observed=True)[metrics].mean().reset_index()
    summary["method_order"] = summary["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return summary.sort_values("method_order").drop(columns=["method_order"])


def summarize_strong(detail, recovery_f1):
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
    strong = detail[detail["experiment"] == "strong_rule_drift"].copy()
    if strong.empty:
        return pd.DataFrame()
    all_summary = strong.groupby(["method", "method_label", "method_year", "category"], observed=True)[metrics].mean().reset_index()
    post = strong[strong["phase"] == "post"].copy()
    post_summary = post.groupby("method", observed=True)[metrics].mean().reset_index()
    post_summary = post_summary.rename(columns={col: f"post_{col}" for col in metrics})
    summary = all_summary.merge(post_summary, on="method", how="left")
    recovery_rows = []
    drift_chunk = int(strong.loc[strong["phase"] == "transition", "chunk"].min())
    for method, sub in strong[strong["chunk"] >= drift_chunk].groupby("method", observed=True):
        recovered = sub[(sub["chunk"] > drift_chunk) & (sub["f1"] >= recovery_f1)]
        recovery_chunk = int(recovered["chunk"].min()) if len(recovered) else -1
        recovery_rows.append(
            {
                "method": method,
                "recovery_chunk": recovery_chunk,
                "recovery_lag_chunks": recovery_chunk - drift_chunk if recovery_chunk > 0 else -1,
            }
        )
    summary = summary.merge(pd.DataFrame(recovery_rows), on="method", how="left")
    summary["method_order"] = summary["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return summary.sort_values("method_order").drop(columns=["method_order"])


def fmt(value, digits=3):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_ms(value, digits=3):
    return fmt(float(value) * 1000.0, digits)


def fmt_pct(value, digits=1):
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100.0:.{digits}f}"


def write_markdown_table(df, path, title):
    cols = list(df.columns)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_modern_table(core_summary, strong_summary):
    core_map = core_summary.set_index("method") if not core_summary.empty else pd.DataFrame()
    strong_map = strong_summary.set_index("method") if not strong_summary.empty else pd.DataFrame()
    rows = []
    for method in METHOD_ORDER:
        core = core_map.loc[method] if method in core_map.index else None
        strong = strong_map.loc[method] if method in strong_map.index else None
        rows.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "Lineage/year": METHOD_YEARS.get(method, ""),
                "Rule-structure adaptation": "Yes" if method == "RT_A3C_Realtime_MLN" else "No",
                "Core F1": "-" if core is None else fmt(core["f1"]),
                "Core Log-loss": "-" if core is None else fmt(core["log_loss"]),
                "Core cycle ms": "-" if core is None else fmt_ms(core["blocking_cycle_time_sec"]),
                "Core 10ms %": "-" if core is None else fmt_pct(core["meets_10ms_cycle"]),
                "Strong post-F1": "-" if strong is None else fmt(strong["post_f1"]),
                "Strong post Log-loss": "-" if strong is None else fmt(strong["post_log_loss"]),
                "Recovery lag": "-"
                if strong is None
                else ("not recovered" if int(strong["recovery_lag_chunks"]) < 0 else str(int(strong["recovery_lag_chunks"]))),
                "Struct update %": "-" if strong is None else fmt_pct(strong["structure_refreshed"]),
                "Active emerging rules": "-" if strong is None else fmt(strong["active_emerging_rule_count"], 2),
            }
        )
    return pd.DataFrame(rows)


def plot_modern_summary(core_summary, strong_summary):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    methods = [method for method in METHOD_ORDER if method in set(core_summary["method"]).union(set(strong_summary["method"]))]
    labels = [METHOD_LABELS.get(method, method) for method in methods]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

    core_map = core_summary.set_index("method")
    core_cycle = [float(core_map.loc[m, "blocking_cycle_time_sec"]) * 1000.0 if m in core_map.index else np.nan for m in methods]
    core_f1 = [float(core_map.loc[m, "f1"]) if m in core_map.index else np.nan for m in methods]
    axes[0].bar(x, core_cycle, color=[COLORS.get(m, "#6c757d") for m in methods], width=0.62)
    axes[0].axhline(10, color="#c1121f", ls="--", lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("Core stream realtime cost")
    axes[0].set_ylabel("Blocking cycle (ms, log)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")

    ax1 = axes[1]
    ax1.bar(x - 0.17, core_f1, width=0.34, color=[COLORS.get(m, "#6c757d") for m in methods], alpha=0.88, label="Core F1")
    if not strong_summary.empty:
        strong_map = strong_summary.set_index("method")
        post_f1 = [float(strong_map.loc[m, "post_f1"]) if m in strong_map.index else np.nan for m in methods]
    else:
        post_f1 = [np.nan for _ in methods]
    ax1.bar(x + 0.17, post_f1, width=0.34, color="#495057", alpha=0.74, label="Strong post-F1")
    ax1.set_title("Predictive quality under streams")
    ax1.set_ylabel("F1")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.legend(frameon=False, fontsize=8)

    if not strong_summary.empty:
        active = [float(strong_map.loc[m, "active_emerging_rule_count"]) if m in strong_map.index else np.nan for m in methods]
        recovery = [
            np.nan if m not in strong_map.index or int(strong_map.loc[m, "recovery_lag_chunks"]) < 0 else float(strong_map.loc[m, "recovery_lag_chunks"])
            for m in methods
        ]
    else:
        active = [np.nan for _ in methods]
        recovery = [np.nan for _ in methods]
    axes[2].bar(x - 0.17, active, width=0.34, color="#52b788", label="Active emerging rules")
    axes[2].bar(x + 0.17, recovery, width=0.34, color="#457b9d", label="Recovery lag")
    axes[2].set_title("Strong-drift adaptation evidence")
    axes[2].set_ylabel("Count / chunks")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")
    fig.suptitle("Modern stream baselines: speed, drift recovery, and rule adaptation", y=1.02, fontsize=13, weight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Fig9_Modern_Stream_Baselines.{ext}", dpi=260 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def write_explanations(table):
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    text = """# Table5_Modern_Stream_Baselines 解释

## 表格作用

这张表用于回应“现有对比方法年份偏老”的问题。它不替代主结果表，而是作为补充实验：把本文方法与若干数据流学习领域常用的在线/增量基线放在同一数据流协议下比较。

## 行名称

- **RT-A3C-MLN (ours)**：本文方法，支持 MLN 规则结构与权重在数据流中异步更新。
- **OnlineWeight MLN**：固定 MLN 规则结构，只进行在线权重更新。
- **Static MaxEnt**：静态最大熵参考模型，不进行在线更新。
- **River Online LR**：River 框架中的在线逻辑回归，代表轻量级流式线性模型。
- **HAT**：Hoeffding Adaptive Tree，使用漂移检测的自适应流式决策树。
- **ARF**：Adaptive Random Forest，数据流场景中的自适应随机森林集成。
- **SRP**：Streaming Random Patches，数据流随机子空间与在线集成方法。

## 列名称

- **Lineage/year**：方法谱系或代表性年份。
- **Rule-structure adaptation**：是否支持规则结构自适应。这里的“规则结构”指是否能新增、删除或替换 MLN 规则特征，而不仅是更新已有参数。
- **Core F1 / Core Log-loss**：常规数据流实验中的平均分类质量和概率质量。
- **Core cycle ms**：常规数据流下每个 chunk 的阻塞周期，即推理时间加前台更新时间。
- **Core 10ms %**：阻塞周期不超过 10 ms 的 chunk 比例。
- **Strong post-F1 / Strong post Log-loss**：强规则漂移压力测试中，漂移后的 F1 和 Log-loss。
- **Recovery lag**：漂移后恢复到设定 F1 阈值所需的 chunk 数；not recovered 表示未恢复。
- **Struct update %**：规则结构刷新被接受的比例。
- **Active emerging rules**：强漂移后被本文方法激活的新规则数量。其他流式学习方法不是 MLN 结构学习方法，因此该指标通常为 0。

## 结果解读

如果 ARF、SRP 或 HAT 在速度上具有竞争力，但 **Rule-structure adaptation** 为 No，说明它们是强数据流分类器，却不能直接证明“MLN 规则可以在数据流下自适应调整”。如果本文方法在强规则漂移下恢复更快，同时出现 active emerging rules 和结构刷新，则说明本文优势来自规则结构层面的实时自适应，而不是单纯的在线分类。
"""
    (EXPLAIN_DIR / "Table5_Modern_Stream_Baselines_explanation.md").write_text(text, encoding="utf-8")

    fig_text = """# Fig9_Modern_Stream_Baselines 解释

## 图的作用

这张图把新增现代数据流基线的速度、预测质量和强规则漂移适应能力放在同一张图中，作为论文的补充对比图。

## 左侧子图

横轴是方法名称，纵轴是 **Blocking cycle (ms, log)**，表示每个数据流 chunk 的阻塞周期，单位为毫秒，并使用对数坐标。红色虚线是 10 ms 实时阈值。柱子越低，说明方法越适合实时响应。

## 中间子图

横轴是方法名称，纵轴是 **F1**。彩色柱表示常规数据流实验的平均 F1，灰色柱表示强规则漂移后的 post-F1。该子图用于观察方法在温和漂移和强规则漂移下的预测稳定性。

## 右侧子图

横轴是方法名称，纵轴表示数量。绿色柱是强规则漂移后被激活的 emerging rules 数量，蓝色柱是恢复延迟 chunk 数。本文方法如果绿色柱显著非零，说明模型确实进行了规则结构层面的调整；如果蓝色柱较低，说明模型能够较快恢复。

## 结论

该图的核心不是证明本文方法在所有静态分类指标上都最高，而是证明：在需要“规则结构可解释地变化”的数据流场景中，RT-A3C-MLN 同时保留实时响应和规则结构自适应，而普通数据流分类器只能更新预测器参数，不能形成可解释的 MLN 规则增删证据。
"""
    (EXPLAIN_DIR / "Fig9_Modern_Stream_Baselines_explanation.md").write_text(fig_text, encoding="utf-8")


def save_outputs(detail, core_summary, strong_summary, table, args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT_DIR / "modern_stream_detail.csv", index=False)
    core_summary.to_csv(OUT_DIR / "modern_core_summary.csv", index=False)
    strong_summary.to_csv(OUT_DIR / "modern_strong_summary.csv", index=False)
    table.to_csv(TABLE_DIR / "Table5_Modern_Stream_Baselines.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(table, TABLE_DIR / "Table5_Modern_Stream_Baselines.md", "Modern stream baselines")
    (OUT_DIR / "modern_stream_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    plot_modern_summary(core_summary, strong_summary)
    write_explanations(table)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--max-rules", type=int, default=220)
    parser.add_argument("--max-literals", type=int, default=80)
    parser.add_argument("--numeric-bins", type=int, default=5)
    parser.add_argument("--min-support", type=float, default=0.015)
    parser.add_argument("--stream-fraction", type=float, default=0.70)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--river-trees", type=int, default=3)
    parser.add_argument("--river-warmup-limit", type=int, default=2500)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=5)
    parser.add_argument("--a3c-steps", type=int, default=5)
    parser.add_argument("--a3c-rules", type=int, default=180)
    parser.add_argument("--a3c-max-active-rules", type=int, default=64)
    parser.add_argument("--a3c-rolling-window", type=int, default=600)
    parser.add_argument("--a3c-calibration-window", type=int, default=300)
    parser.add_argument("--a3c-refit-interval", type=int, default=4)
    parser.add_argument("--a3c-structure-episodes", type=int, default=2)
    parser.add_argument("--a3c-structure-tolerance", type=float, default=0.0)
    parser.add_argument("--a3c-prior-shift-lr", type=float, default=0.20)
    parser.add_argument("--a3c-prior-shift-clip", type=float, default=1.5)
    parser.add_argument("--strong-n-features", dest="n_features", type=int, default=420)
    parser.add_argument("--strong-train-size", dest="train_size", type=int, default=2600)
    parser.add_argument("--strong-val-size", dest="val_size", type=int, default=650)
    parser.add_argument("--strong-chunks", dest="strong_chunks", type=int, default=8)
    parser.add_argument("--strong-chunk-size", dest="strong_chunk_size", type=int, default=600)
    parser.add_argument("--strong-drift-chunk", dest="drift_chunk", type=int, default=5)
    parser.add_argument("--initial-rule-budget", type=int, default=22)
    parser.add_argument("--rolling-window", type=int, default=2200)
    parser.add_argument("--recovery-f1", type=float, default=0.68)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for dataset in args.datasets:
        rows.extend(run_core_dataset(dataset, args))

    strong_args = argparse.Namespace(**vars(args))
    strong_args.chunks = args.strong_chunks
    strong_args.chunk_size = args.strong_chunk_size
    rows.extend(run_strong_drift(strong_args))

    detail = pd.DataFrame(rows)
    core_summary = summarize_core(detail)
    strong_summary = summarize_strong(detail, args.recovery_f1)
    table = build_modern_table(core_summary, strong_summary)
    save_outputs(detail, core_summary, strong_summary, table, args)

    print(table.to_string(index=False))
    print(f"[done] results: {OUT_DIR}")
    print(f"[done] artifacts: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
