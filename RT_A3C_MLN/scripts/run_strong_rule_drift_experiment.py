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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.models import BeamSearchMLN, _safe_proba
from scripts.run_streaming_experiments import AdaptiveA3CStreamModel


OUT_DIR = ROOT / "results" / "strong_rule_drift"
ARTIFACT_ROOT = Path(os.environ.get("PAPER_ARTIFACTS_DIR", ROOT / "paper_artifacts"))
ARTIFACT_DIR = ARTIFACT_ROOT / "strong_rule_drift"
FIG_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"

METHOD_ORDER = [
    "RT_A3C_Realtime_MLN",
    "Rolling_L1_MLN",
    "Rolling_MaxEnt_MLN",
    "Rolling_Boosted_MLN",
    "Rolling_BeamSearch_MLN",
    "OnlineWeight_MLN_FixedStructure",
    "Static_MaxEnt_FixedStructure",
]

LABELS = {
    "RT_A3C_Realtime_MLN": "RT-A3C-MLN",
    "Rolling_L1_MLN": "Rolling L1",
    "Rolling_MaxEnt_MLN": "Rolling MaxEnt",
    "Rolling_Boosted_MLN": "Rolling Boosted",
    "Rolling_BeamSearch_MLN": "Rolling Beam",
    "OnlineWeight_MLN_FixedStructure": "OnlineWeight fixed",
    "Static_MaxEnt_FixedStructure": "Static fixed",
}

COLORS = {
    "RT_A3C_Realtime_MLN": "#c1121f",
    "Rolling_L1_MLN": "#2a9d8f",
    "Rolling_MaxEnt_MLN": "#457b9d",
    "Rolling_Boosted_MLN": "#f4a261",
    "Rolling_BeamSearch_MLN": "#6d597a",
    "OnlineWeight_MLN_FixedStructure": "#adb5bd",
    "Static_MaxEnt_FixedStructure": "#8d99ae",
}

OLD_RULES = np.arange(0, 8)
DISTRACTOR_RULES = np.arange(8, 48)
EMERGING_RULES = np.arange(48, 60)


def sigmoid(score):
    score = np.clip(score, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-score))


def metrics_from_prob(y_true, prob, inference_sec):
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
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


def make_rule_names(n_features):
    names = []
    for idx in range(n_features):
        if idx in OLD_RULES:
            names.append(f"old_rule_{idx:02d}: Target <- historical concept predicate")
        elif idx in EMERGING_RULES:
            names.append(f"emerging_rule_{idx:02d}: Target <- post-drift predicate")
        else:
            names.append(f"distractor_rule_{idx:02d}: weak/noisy predicate")
    return names


def generate_phase(n, rng, phase, n_features):
    x = np.zeros((n, n_features), dtype=np.uint8)

    old_p = 0.30 if phase != "post" else 0.34
    x[:, OLD_RULES] = rng.binomial(1, old_p, size=(n, len(OLD_RULES)))

    if phase == "pre":
        emerging_p = 0.018
    elif phase == "transition":
        emerging_p = 0.13
    else:
        emerging_p = 0.36
    x[:, EMERGING_RULES] = rng.binomial(1, emerging_p, size=(n, len(EMERGING_RULES)))

    distractor_ps = rng.uniform(0.06, 0.44, size=len(DISTRACTOR_RULES))
    x[:, DISTRACTOR_RULES] = rng.binomial(1, distractor_ps, size=(n, len(DISTRACTOR_RULES)))

    extra = np.arange(60, n_features)
    if len(extra):
        extra_ps = rng.uniform(0.04, 0.35, size=len(extra))
        x[:, extra] = rng.binomial(1, extra_ps, size=(n, len(extra)))

    old_signal = (
        1.35 * x[:, 0]
        + 1.15 * x[:, 1]
        + 1.05 * x[:, 2]
        + 0.85 * (x[:, 3] & x[:, 4])
        - 0.55 * x[:, 6]
    )
    emerging_signal = (
        1.55 * x[:, 48]
        + 1.35 * x[:, 49]
        + 1.20 * x[:, 50]
        + 1.05 * (x[:, 51] & x[:, 52])
        + 0.95 * x[:, 53]
    )
    noise_signal = 0.18 * x[:, 12] - 0.16 * x[:, 18] + 0.10 * x[:, 27]

    if phase == "pre":
        score = old_signal - 1.25 - 0.65 * emerging_signal + noise_signal
    elif phase == "transition":
        score = 0.65 * old_signal + 0.95 * emerging_signal - 1.35 + noise_signal
    else:
        score = emerging_signal - 1.15 - 0.70 * old_signal + noise_signal

    prob = sigmoid(score)
    y = rng.binomial(1, prob).astype(int)
    return x, y, prob


def make_emerging_rule_stream(args):
    rng = np.random.default_rng(args.seed)
    x_train, y_train, _ = generate_phase(args.train_size, rng, "pre", args.n_features)
    x_val, y_val, _ = generate_phase(args.val_size, rng, "pre", args.n_features)

    chunks = []
    for chunk_id in range(1, args.chunks + 1):
        if chunk_id <= args.drift_chunk - 1:
            phase = "pre"
        elif chunk_id == args.drift_chunk:
            phase = "transition"
        else:
            phase = "post"
        x_chunk, y_chunk, prob = generate_phase(args.chunk_size, rng, phase, args.n_features)
        chunks.append(
            {
                "chunk": chunk_id,
                "phase": phase,
                "x": x_chunk,
                "y": y_chunk,
                "positive_rate": float(y_chunk.mean()),
                "emerging_support": float(x_chunk[:, EMERGING_RULES].mean()),
                "old_support": float(x_chunk[:, OLD_RULES].mean()),
                "oracle_prob": prob,
            }
        )
    return x_train, y_train, x_val, y_val, chunks


def select_initial_structure(x_train, y_train, max_rules=22, random_state=13):
    selector = LogisticRegression(
        solver="liblinear",
        penalty="l1",
        C=0.22,
        max_iter=800,
        random_state=random_state,
    )
    selector.fit(x_train, y_train)
    coef = np.abs(np.ravel(selector.coef_))
    selected = [int(idx) for idx in np.flatnonzero(coef > 1e-7)]
    if len(selected) < 8:
        selected = [int(idx) for idx in np.argsort(coef)[::-1][: max_rules] if coef[idx] > 0]
    if not selected:
        selected = list(range(min(max_rules, x_train.shape[1])))
    selected = sorted(selected[:max_rules])
    return selected


class StaticFixedStructureModel:
    category = "fixed_structure_reference"

    def __init__(self, selected, random_state):
        self.selected = list(selected)
        self.model = LogisticRegression(solver="liblinear", C=1.0, max_iter=800, random_state=random_state)

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.fit(x_train[:, self.selected], y_train)
        return self

    def predict_proba(self, x):
        return _safe_proba(self.model.predict_proba(x[:, self.selected]))

    def update(self, x_chunk, y_chunk):
        return 0.0


class OnlineFixedStructureModel:
    category = "fixed_structure_reference"

    def __init__(self, selected, random_state):
        self.selected = list(selected)
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

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.partial_fit(x_train[:, self.selected], y_train, classes=self.classes)
        return self

    def predict_proba(self, x):
        return _safe_proba(self.model.predict_proba(x[:, self.selected]))

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.model.partial_fit(x_chunk[:, self.selected], y_chunk)
        return time.perf_counter() - start


class RollingRetrainModel:
    category = "adaptive_batch_mln"

    def __init__(self, estimator_factory, window=2400):
        self.estimator_factory = estimator_factory
        self.window = window
        self.model = None
        self.x_recent = None
        self.y_recent = None

    def fit(self, x_train, y_train, x_val, y_val):
        self.x_recent = x_train[-self.window :].copy()
        self.y_recent = y_train[-self.window :].copy()
        self.model = self.estimator_factory()
        self.model.fit(self.x_recent, self.y_recent)
        return self

    def predict_proba(self, x):
        return _safe_proba(self.model.predict_proba(x))

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.x_recent = np.vstack([self.x_recent, x_chunk])[-self.window :]
        self.y_recent = np.concatenate([self.y_recent, y_chunk])[-self.window :]
        self.model = self.estimator_factory()
        self.model.fit(self.x_recent, self.y_recent)
        return time.perf_counter() - start


class RollingBeamMLN:
    category = "adaptive_batch_mln"

    def __init__(self, random_state, window=1800):
        self.random_state = random_state
        self.window = window
        self.model = None
        self.x_recent = None
        self.y_recent = None
        self.x_val = None
        self.y_val = None

    def fit(self, x_train, y_train, x_val, y_val):
        self.x_recent = x_train[-self.window :].copy()
        self.y_recent = y_train[-self.window :].copy()
        self.x_val = x_val
        self.y_val = y_val
        self._fit_model()
        return self

    def _fit_model(self):
        self.model = BeamSearchMLN(max_selected=18, search_pool=70, random_state=self.random_state)
        self.model.fit(self.x_recent, self.y_recent, self.x_val, self.y_val, rules=None)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.x_recent = np.vstack([self.x_recent, x_chunk])[-self.window :]
        self.y_recent = np.concatenate([self.y_recent, y_chunk])[-self.window :]
        self._fit_model()
        return time.perf_counter() - start


def build_models(args, selected):
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
        "Rolling_L1_MLN": RollingRetrainModel(
            lambda: LogisticRegression(
                solver="liblinear",
                penalty="l1",
                C=0.25,
                max_iter=800,
                random_state=args.seed,
            ),
            window=args.rolling_window,
        ),
        "Rolling_MaxEnt_MLN": RollingRetrainModel(
            lambda: LogisticRegression(solver="liblinear", C=1.0, max_iter=800, random_state=args.seed),
            window=args.rolling_window,
        ),
        "Rolling_Boosted_MLN": RollingRetrainModel(
            lambda: GradientBoostingClassifier(
                n_estimators=90,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.85,
                random_state=args.seed,
            ),
            window=args.rolling_window,
        ),
        "Rolling_BeamSearch_MLN": RollingBeamMLN(args.seed, window=args.rolling_window),
        "OnlineWeight_MLN_FixedStructure": OnlineFixedStructureModel(selected, args.seed),
        "Static_MaxEnt_FixedStructure": StaticFixedStructureModel(selected, args.seed),
    }


def run_experiment(args):
    x_train, y_train, x_val, y_val, chunks = make_emerging_rule_stream(args)
    selected = select_initial_structure(x_train, y_train, args.initial_rule_budget, args.seed)
    rule_names = make_rule_names(args.n_features)
    rows = []
    model_rows = []
    for method, model in build_models(args, selected).items():
        print(f"[strong-drift] method={method}")
        start = time.perf_counter()
        model.fit(x_train, y_train, x_val, y_val)
        initial_fit = time.perf_counter() - start
        if method == "RT_A3C_Realtime_MLN":
            initial_rules = getattr(model, "active_rules", [])
        elif hasattr(model, "selected"):
            initial_rules = model.selected
        else:
            initial_rules = list(range(args.n_features))
        model_rows.append(
            {
                "method": method,
                "initial_fit_time_sec": initial_fit,
                "initial_rule_count": len(initial_rules),
                "initial_emerging_rule_count": int(len(set(initial_rules).intersection(set(EMERGING_RULES)))),
                "initial_rules": " ".join(map(str, initial_rules[:40])),
            }
        )
        for chunk in chunks:
            x_chunk = chunk["x"]
            y_chunk = chunk["y"]
            start = time.perf_counter()
            prob = model.predict_proba(x_chunk)[:, 1]
            inference_time = time.perf_counter() - start
            update_time = model.update(x_chunk, y_chunk)
            cycle_time = inference_time + update_time
            active_rules = getattr(model, "active_rules", [])
            rows.append(
                {
                    "dataset": "emerging_rule_pressure",
                    "method": method,
                    "category": getattr(model, "category", "rt_a3c_realtime"),
                    "chunk": chunk["chunk"],
                    "phase": chunk["phase"],
                    "positive_rate": chunk["positive_rate"],
                    "old_rule_support": chunk["old_support"],
                    "emerging_rule_support": chunk["emerging_support"],
                    "foreground_update_time_sec": update_time,
                    "background_update_time_sec": getattr(model, "last_background_update_sec", 0.0),
                    "blocking_cycle_time_sec": cycle_time,
                    "meets_10ms_cycle": float(cycle_time <= 0.010),
                    "structure_refreshed": getattr(model, "last_structure_refreshed", 0.0),
                    "rule_additions": getattr(model, "last_rule_additions", 0),
                    "rule_deletions": getattr(model, "last_rule_deletions", 0),
                    "active_rule_count": len(active_rules) if active_rules is not None else 0,
                    "active_emerging_rule_count": int(len(set(active_rules or []).intersection(set(EMERGING_RULES)))),
                    **metrics_from_prob(y_chunk, prob, inference_time),
                }
            )
    detail = pd.DataFrame(rows)
    initial = pd.DataFrame(model_rows)
    return detail, initial, selected, rule_names


def summarize(detail, args):
    metrics = [
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "foreground_update_time_sec",
        "blocking_cycle_time_sec",
        "background_update_time_sec",
        "meets_10ms_cycle",
        "structure_refreshed",
        "rule_additions",
        "rule_deletions",
        "active_emerging_rule_count",
    ]
    all_summary = detail.groupby(["method", "category"], observed=True)[metrics].mean().reset_index()
    post = detail[detail["chunk"] > args.drift_chunk].copy()
    post_summary = post.groupby(["method"], observed=True)[metrics].mean().reset_index()
    post_summary = post_summary.rename(columns={col: f"post_{col}" for col in metrics})
    summary = all_summary.merge(post_summary, on="method", how="left")
    recovery_rows = []
    for method, sub in detail[detail["chunk"] >= args.drift_chunk].groupby("method", observed=True):
        recovered = sub[(sub["chunk"] > args.drift_chunk) & (sub["f1"] >= args.recovery_f1)]
        recovery_chunk = int(recovered["chunk"].min()) if len(recovered) else -1
        recovery_rows.append(
            {
                "method": method,
                "recovery_chunk": recovery_chunk,
                "recovery_lag_chunks": recovery_chunk - args.drift_chunk if recovery_chunk > 0 else -1,
            }
        )
    summary = summary.merge(pd.DataFrame(recovery_rows), on="method", how="left")
    summary["order"] = summary["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return summary.sort_values("order").drop(columns=["order"])


def fmt(value, digits=3):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def make_summary_table(summary):
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "Method": LABELS.get(row["method"], row["method"]),
                "Adaptive structure": "Yes" if row["method"] == "RT_A3C_Realtime_MLN" or row["method"].startswith("Rolling") else "No",
                "Post-drift F1": fmt(row["post_f1"]),
                "Post-drift AUC": fmt(row["post_roc_auc"]),
                "Post-drift Log-loss": fmt(row["post_log_loss"]),
                "Recovery lag chunks": "not recovered" if int(row["recovery_lag_chunks"]) < 0 else str(int(row["recovery_lag_chunks"])),
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
    table_df.to_csv(TABLE_DIR / "Table3_Strong_Rule_Drift.csv", index=False)
    header = "| " + " | ".join(table_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(row[col]) for col in table_df.columns) + " |"
        for _, row in table_df.iterrows()
    )
    md = "# Strong rule-drift pressure test\n\n" + "\n".join([header, sep, body])
    (TABLE_DIR / "Table3_Strong_Rule_Drift.md").write_text(md, encoding="utf-8")


def plot_trajectory(detail, args):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8.7, 8.4), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.95]})
    methods = METHOD_ORDER
    for method in methods:
        sub = detail[detail["method"] == method].sort_values("chunk")
        if sub.empty:
            continue
        label = LABELS.get(method, method)
        lw = 2.3 if method == "RT_A3C_Realtime_MLN" else 1.4
        alpha = 1.0 if method == "RT_A3C_Realtime_MLN" else 0.78
        axes[0].plot(sub["chunk"], sub["f1"], marker="o", lw=lw, color=COLORS[method], label=label, alpha=alpha)
        axes[1].plot(sub["chunk"], sub["log_loss"], marker="s", lw=lw, color=COLORS[method], alpha=alpha)
        axes[2].plot(sub["chunk"], sub["blocking_cycle_time_sec"] * 1000, marker="^", lw=lw, color=COLORS[method], alpha=alpha)

    ax_drift = axes[0].twinx()
    drift = detail[detail["method"] == "RT_A3C_Realtime_MLN"].sort_values("chunk")
    ax_drift.fill_between(drift["chunk"], drift["emerging_rule_support"], color="#dee2e6", alpha=0.50, step="mid")
    ax_drift.plot(drift["chunk"], drift["positive_rate"], color="#495057", ls="--", lw=1.0, label="positive rate")
    ax_drift.set_ylabel("emerging support / positive rate")
    ax_drift.set_ylim(0, 1)
    ax_drift.grid(False)

    for ax in axes:
        ax.axvline(args.drift_chunk, color="#343a40", ls="--", lw=1.0)
        ax.grid(True, alpha=0.28)
    axes[0].text(args.drift_chunk + 0.08, axes[0].get_ylim()[1] * 0.96, "rule drift", fontsize=8.5, color="#343a40")
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("Log-loss")
    axes[2].set_ylabel("Blocking cycle (ms)")
    axes[2].set_xlabel("Stream chunk")
    axes[2].axhline(10, color="#c1121f", ls="--", lw=1.0)
    axes[2].set_yscale("log")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.26), frameon=False, fontsize=8.0)
    fig.suptitle("Strong emerging-rule drift: recovery and realtime response", y=0.995, fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Fig6_Strong_Rule_Drift_Trajectory.{ext}", dpi=260 if ext == "png" else None)
    plt.close(fig)


def plot_structure(detail, summary, args):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.05))
    post = summary.set_index("method").loc[METHOD_ORDER].reset_index()
    compact = {
        "RT_A3C_Realtime_MLN": "RT-A3C",
        "Rolling_L1_MLN": "L1",
        "Rolling_MaxEnt_MLN": "MaxEnt",
        "Rolling_Boosted_MLN": "Boosted",
        "Rolling_BeamSearch_MLN": "Beam",
        "OnlineWeight_MLN_FixedStructure": "Online\nfixed",
        "Static_MaxEnt_FixedStructure": "Static\nfixed",
    }
    labels = [compact.get(m, LABELS[m]) for m in post["method"]]
    x = np.arange(len(post))

    axes[0].bar(x, post["post_f1"], color=[COLORS[m] for m in post["method"]], width=0.64)
    axes[0].set_title("Post-drift F1")
    axes[0].set_ylabel("F1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=0, fontsize=7.3)
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, post["post_log_loss"], color=[COLORS[m] for m in post["method"]], width=0.64)
    axes[1].set_title("Post-drift probability loss")
    axes[1].set_ylabel("Log-loss")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=0, fontsize=7.3)

    rt = detail[detail["method"] == "RT_A3C_Realtime_MLN"].sort_values("chunk")
    axes[2].bar(rt["chunk"], rt["active_emerging_rule_count"], color="#52b788", label="active emerging rules")
    axes[2].bar(rt["chunk"], rt["rule_additions"], bottom=rt["active_emerging_rule_count"], color="#2a9d8f", alpha=0.35, label="added rules")
    axes[2].axvline(args.drift_chunk, color="#343a40", ls="--", lw=1.0)
    axes[2].set_title("RT-A3C rule refresh")
    axes[2].set_xlabel("Stream chunk")
    axes[2].set_ylabel("Rule count")
    axes[2].legend(frameon=False, loc="upper left", fontsize=8.0)
    fig.suptitle("Pressure-test evidence: fixed structures fail under emerging rules", y=0.985, fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Fig7_Strong_Drift_Structure_Evidence.{ext}", dpi=260 if ext == "png" else None)
    plt.close(fig)


def write_explanation(table_df, selected, rule_names, args):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    selected_text = "\n".join(f"- {idx}: {rule_names[idx]}" for idx in selected)
    text = f"""# 强规则漂移压力测试说明

## 实验目的

该实验专门验证固定结构方法在“新规则后半段才出现”的数据流中是否会失效。常规数据流实验主要验证实时性和温和分布变化下的稳定性，而本实验制造更强的规则结构漂移：前半段标签由 old rules 决定，漂移点后标签主要由 emerging rules 决定。

## 数据流设计

- 初始训练集和验证集只来自 pre-drift 阶段。
- 数据流共有 {args.chunks} 个 chunk，每个 chunk 含 {args.chunk_size} 个样本。
- 第 {args.drift_chunk} 个 chunk 是规则漂移点。
- old rules 位于特征索引 {OLD_RULES[0]} 到 {OLD_RULES[-1]}。
- emerging rules 位于特征索引 {EMERGING_RULES[0]} 到 {EMERGING_RULES[-1]}，在初始训练阶段支持度很低，漂移后支持度显著升高并决定标签。

## 固定结构基线的定义

OnlineWeight MLN fixed structure 和 Static MaxEnt fixed structure 只使用初始训练阶段选出的规则集合。它们可以更新已有规则权重，但不能添加 emerging rules，因此适合用来验证固定结构方法在强规则漂移下的局限。

初始固定结构规则为：

{selected_text}

## 结果解读

表3、图6和图7应一起使用。重点看四个指标：

- Post-drift F1：漂移后分类性能。
- Post-drift Log-loss：漂移后概率推理质量，越低越好。
- Recovery lag chunks：漂移后恢复到 F1 >= {args.recovery_f1:.2f} 所需 chunk 数。
- Active emerging rules：RT-A3C-MLN 后台结构刷新后激活的 emerging rules 数量。

如果 OnlineWeight 或 Static 的 F1 在常规实验中较高，但在这里 post-drift F1 或 Log-loss 明显变差，就可以说明它们的优势来自固定结构场景，并不能证明规则结构自适应能力。RT-A3C-MLN 若同时出现 active emerging rules 增加和低阻塞周期，则说明本文方法能够在规则结构发生变化时进行实时自适应。
"""
    (ARTIFACT_DIR / "Strong_Rule_Drift_explanation.md").write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--n-features", type=int, default=420)
    parser.add_argument("--train-size", type=int, default=3600)
    parser.add_argument("--val-size", type=int, default=900)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--drift-chunk", type=int, default=5)
    parser.add_argument("--initial-rule-budget", type=int, default=22)
    parser.add_argument("--rolling-window", type=int, default=2200)
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
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    detail, initial, selected, rule_names = run_experiment(args)
    summary = summarize(detail, args)
    table = make_summary_table(summary)

    detail.to_csv(OUT_DIR / "strong_rule_drift_detail.csv", index=False)
    initial.to_csv(OUT_DIR / "strong_rule_drift_initial_rules.csv", index=False)
    summary.to_csv(OUT_DIR / "strong_rule_drift_summary.csv", index=False)
    (OUT_DIR / "strong_rule_drift_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (OUT_DIR / "strong_rule_drift_rule_names.json").write_text(json.dumps(rule_names, indent=2), encoding="utf-8")

    save_table(table)
    plot_trajectory(detail, args)
    plot_structure(detail, summary, args)
    write_explanation(table, selected, rule_names, args)

    print(table.to_string(index=False))
    print(f"[done] results: {OUT_DIR}")
    print(f"[done] artifacts: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
