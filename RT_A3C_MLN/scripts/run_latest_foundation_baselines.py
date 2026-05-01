from __future__ import annotations

from pathlib import Path
import argparse
import gc
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS, load_dataset
from mln_a3c_experiment.models import _safe_proba
from mln_a3c_experiment.rules import RuleFeatureBuilder
from scripts.run_core_realtime_experiment import OnlineWeightMLN, StaticMaxEntReference
from scripts.run_streaming_experiments import AdaptiveA3CStreamModel, make_prior_drift_chunks
from scripts.run_strong_rule_drift_experiment import EMERGING_RULES, make_emerging_rule_stream


OUT_DIR = ROOT / "results" / "latest_foundation_baselines"
ARTIFACT_DIR = ROOT / "paper_artifacts" / "latest_foundation_baselines"
FIG_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"
EXPLAIN_DIR = ARTIFACT_DIR / "explanations"

METHOD_ORDER = [
    "RT_A3C_Realtime_MLN",
    "TabPFN_v2_2025_Context",
    "TabICL_v2_2026_Context",
    "TabDPT_2025_Context",
    "TabM_2025_OnlineTune",
    "OnlineWeight_MLN",
    "Static_MaxEnt_Reference",
]

METHOD_LABELS = {
    "RT_A3C_Realtime_MLN": "RT-A3C-MLN (ours)",
    "TabPFN_v2_2025_Context": "TabPFN v2",
    "TabICL_v2_2026_Context": "TabICL v2",
    "TabDPT_2025_Context": "TabDPT",
    "TabM_2025_OnlineTune": "TabM",
    "OnlineWeight_MLN": "OnlineWeight MLN",
    "Static_MaxEnt_Reference": "Static MaxEnt",
}

METHOD_YEARS = {
    "RT_A3C_Realtime_MLN": "2026 / ours",
    "TabPFN_v2_2025_Context": "2025 foundation model",
    "TabICL_v2_2026_Context": "2026 checkpoint / 2025 paper",
    "TabDPT_2025_Context": "2025 / 2024 preprint",
    "TabM_2025_OnlineTune": "ICLR 2025",
    "OnlineWeight_MLN": "fixed-rule online reference",
    "Static_MaxEnt_Reference": "static reference",
}

COLORS = {
    "RT_A3C_Realtime_MLN": "#c1121f",
    "TabPFN_v2_2025_Context": "#2a9d8f",
    "TabICL_v2_2026_Context": "#457b9d",
    "TabDPT_2025_Context": "#6d597a",
    "TabM_2025_OnlineTune": "#f4a261",
    "OnlineWeight_MLN": "#adb5bd",
    "Static_MaxEnt_Reference": "#8d99ae",
}


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


def _balanced_tail_indices(y, limit, random_state):
    if limit is None or len(y) <= limit:
        return np.arange(len(y))
    tail = np.arange(max(0, len(y) - limit), len(y))
    if len(np.unique(y[tail])) == 2:
        return tail
    rng = np.random.default_rng(random_state)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    half = max(1, limit // 2)
    keep_pos = rng.choice(pos, size=min(half, len(pos)), replace=False) if len(pos) else np.array([], dtype=int)
    keep_neg = rng.choice(neg, size=min(limit - len(keep_pos), len(neg)), replace=False) if len(neg) else np.array([], dtype=int)
    keep = np.sort(np.concatenate([keep_pos, keep_neg]))
    if len(keep) < limit:
        rest = np.setdiff1d(np.arange(len(y)), keep, assume_unique=False)
        extra = rng.choice(rest, size=min(limit - len(keep), len(rest)), replace=False)
        keep = np.sort(np.concatenate([keep, extra]))
    return keep


class ContextFoundationModel:
    category = "latest_foundation_context"

    def __init__(self, model_factory, context_limit=700, random_state=13):
        self.model_factory = model_factory
        self.context_limit = context_limit
        self.random_state = random_state
        self.model = None
        self.context_x = None
        self.context_y = None
        self.fallback_prior = 0.5

    def _new_model(self):
        return self.model_factory()

    def _fit_context_model(self):
        self.fallback_prior = float(np.clip(np.mean(self.context_y), 1e-6, 1.0 - 1e-6))
        if len(np.unique(self.context_y)) < 2:
            return
        self.model = self._new_model()
        self.model.fit(self.context_x.astype(np.float32), self.context_y.astype(int))

    def fit(self, x_train, y_train, x_val, y_val):
        keep = _balanced_tail_indices(np.asarray(y_train), self.context_limit, self.random_state)
        self.context_x = np.asarray(x_train[keep], dtype=np.float32)
        self.context_y = np.asarray(y_train[keep], dtype=int)
        self._fit_context_model()
        return self

    def predict_proba(self, x):
        if self.model is None or len(np.unique(self.context_y)) < 2:
            return _safe_proba(np.full(len(x), self.fallback_prior))
        return _safe_proba(self.model.predict_proba(np.asarray(x, dtype=np.float32)))

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.context_x = np.vstack([self.context_x, np.asarray(x_chunk, dtype=np.float32)])
        self.context_y = np.concatenate([self.context_y, np.asarray(y_chunk, dtype=int)])
        keep = _balanced_tail_indices(self.context_y, self.context_limit, self.random_state + len(self.context_y))
        self.context_x = self.context_x[keep]
        self.context_y = self.context_y[keep]
        self._fit_context_model()
        return time.perf_counter() - start


class TabMOnlineModel:
    category = "latest_deep_online"

    def __init__(
        self,
        random_state=13,
        window=1200,
        initial_epochs=10,
        update_epochs=3,
        batch_size=192,
        hidden=96,
        k=8,
    ):
        self.random_state = random_state
        self.window = window
        self.initial_epochs = initial_epochs
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.hidden = hidden
        self.k = k
        self.model = None
        self.scaler = StandardScaler()
        self.recent_x = None
        self.recent_y = None

    def _build_model(self, n_features):
        import torch
        from tabm import TabM

        torch.manual_seed(self.random_state)
        model = TabM.make(
            n_num_features=n_features,
            cat_cardinalities=[],
            d_out=2,
            n_blocks=2,
            d_block=self.hidden,
            dropout=0.08,
            k=self.k,
            arch_type="tabm-mini",
        )
        return model

    def _train(self, x, y, epochs):
        import torch
        import torch.nn.functional as F

        if self.model is None:
            self.model = self._build_model(x.shape[1])
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-4)
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.long)
        rng = np.random.default_rng(self.random_state + epochs + len(y))
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                logits = self.model(x_tensor[idx], None)
                loss = F.cross_entropy(logits.mean(1), y_tensor[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.model.eval()

    def fit(self, x_train, y_train, x_val, y_val):
        keep = _balanced_tail_indices(np.asarray(y_train), self.window, self.random_state)
        self.recent_x = np.asarray(x_train[keep], dtype=np.float32)
        self.recent_y = np.asarray(y_train[keep], dtype=int)
        x_scaled = self.scaler.fit_transform(self.recent_x).astype(np.float32)
        self._train(x_scaled, self.recent_y, self.initial_epochs)
        return self

    def predict_proba(self, x):
        import torch
        import torch.nn.functional as F

        if self.model is None:
            return _safe_proba(np.full(len(x), 0.5))
        x_scaled = self.scaler.transform(np.asarray(x, dtype=np.float32)).astype(np.float32)
        probs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(x_scaled), self.batch_size):
                batch = torch.as_tensor(x_scaled[start : start + self.batch_size], dtype=torch.float32)
                logits = self.model(batch, None).mean(1)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return _safe_proba(np.vstack(probs))

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.recent_x = np.vstack([self.recent_x, np.asarray(x_chunk, dtype=np.float32)])
        self.recent_y = np.concatenate([self.recent_y, np.asarray(y_chunk, dtype=int)])
        keep = _balanced_tail_indices(self.recent_y, self.window, self.random_state + len(self.recent_y))
        self.recent_x = self.recent_x[keep]
        self.recent_y = self.recent_y[keep]
        x_scaled = self.scaler.fit_transform(self.recent_x).astype(np.float32)
        self._train(x_scaled, self.recent_y, self.update_epochs)
        return time.perf_counter() - start


def make_tabpfn(args):
    def factory():
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(
            n_estimators=args.foundation_estimators,
            device="cpu",
            random_state=args.seed,
            ignore_pretraining_limits=True,
            n_jobs=1,
        )

    return ContextFoundationModel(factory, context_limit=min(args.foundation_context_limit, 128), random_state=args.seed)


def make_tabicl(args):
    def factory():
        from tabicl import TabICLClassifier

        return TabICLClassifier(
            n_estimators=args.foundation_estimators,
            batch_size=args.foundation_batch_size,
            device="cpu",
            random_state=args.seed,
            verbose=False,
            n_jobs=1,
        )

    return ContextFoundationModel(factory, context_limit=min(args.foundation_context_limit, 128), random_state=args.seed)


def make_tabdpt(args):
    def factory():
        import torch
        from tabdpt import TabDPTClassifier

        original_load = torch.load

        def load_cpu(path, *load_args, **load_kwargs):
            load_kwargs.setdefault("map_location", torch.device("cpu"))
            checkpoint = original_load(path, *load_args, **load_kwargs)
            if isinstance(checkpoint, dict) and "cfg" in checkpoint:
                checkpoint["cfg"].setdefault("env", {})["device"] = "cpu"
            return checkpoint

        torch.load = load_cpu
        try:
            return TabDPTClassifier(
                device="cpu",
                inf_batch_size=args.foundation_batch_size,
                use_flash=False,
                compile=False,
            )
        finally:
            torch.load = original_load

    return ContextFoundationModel(factory, context_limit=min(args.foundation_context_limit, 128), random_state=args.seed)


def make_rt_a3c(args):
    return AdaptiveA3CStreamModel(
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
    )


def iter_models(args):
    factories = {
        "RT_A3C_Realtime_MLN": lambda: make_rt_a3c(args),
        "TabPFN_v2_2025_Context": lambda: make_tabpfn(args),
        "TabICL_v2_2026_Context": lambda: make_tabicl(args),
        "TabDPT_2025_Context": lambda: make_tabdpt(args),
        "TabM_2025_OnlineTune": lambda: TabMOnlineModel(
            random_state=args.seed,
            window=args.tabm_window,
            initial_epochs=args.tabm_initial_epochs,
            update_epochs=args.tabm_update_epochs,
            batch_size=args.tabm_batch_size,
            hidden=args.tabm_hidden,
            k=args.tabm_k,
        ),
        "OnlineWeight_MLN": lambda: OnlineWeightMLN(args.seed),
        "Static_MaxEnt_Reference": lambda: StaticMaxEntReference(args.seed),
    }
    wanted = args.methods or METHOD_ORDER
    for method in wanted:
        yield method, factories[method]()


def model_category(model):
    return getattr(model, "category", "rt_a3c_realtime")


def _record_rows(experiment, dataset, method, model, chunks, fit_time, extra_phase_key=None):
    rows = []
    for item in chunks:
        if isinstance(item, tuple):
            chunk_id, drift_rate, x_chunk, y_chunk = item
            phase = "stream"
        else:
            chunk_id = item["chunk"]
            drift_rate = item["positive_rate"]
            x_chunk = item["x"]
            y_chunk = item["y"]
            phase = item["phase"]
        start = time.perf_counter()
        prob = model.predict_proba(x_chunk)[:, 1]
        inference_time = time.perf_counter() - start
        update_time = model.update(x_chunk, y_chunk)
        cycle_time = inference_time + update_time
        active_rules = getattr(model, "active_rules", []) or []
        rows.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "method_year": METHOD_YEARS.get(method, ""),
                "category": model_category(model),
                "chunk": chunk_id,
                "phase": phase,
                "drift_positive_rate": drift_rate,
                "initial_fit_time_sec": fit_time,
                "foreground_update_time_sec": update_time,
                "background_update_time_sec": getattr(model, "last_background_update_sec", 0.0),
                "blocking_cycle_time_sec": cycle_time,
                "meets_10ms_cycle": float(cycle_time <= 0.010),
                "structure_refreshed": getattr(model, "last_structure_refreshed", 0.0),
                "rule_additions": getattr(model, "last_rule_additions", 0),
                "rule_deletions": getattr(model, "last_rule_deletions", 0),
                "active_rule_count": len(active_rules),
                "active_emerging_rule_count": int(len(set(active_rules).intersection(set(EMERGING_RULES))))
                if experiment == "strong_rule_drift"
                else np.nan,
                **metrics_from_prob(y_chunk, prob, inference_time),
            }
        )
    return rows


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
    for method, model in iter_models(args):
        print(f"[latest-core] dataset={dataset} method={method}")
        start = time.perf_counter()
        try:
            model.fit(x_train_rules, y_train, x_val_rules, y_val)
            fit_time = time.perf_counter() - start
            rows.extend(_record_rows("core_stream", dataset, method, model, chunks, fit_time))
        except Exception as exc:
            rows.append(
                {
                    "experiment": "core_stream",
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "method_year": METHOD_YEARS.get(method, ""),
                    "category": "failed",
                    "chunk": -1,
                    "phase": "failed",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[skip] dataset={dataset} method={method} error={type(exc).__name__}: {exc}")
        del model
        gc.collect()
    return rows


def run_strong_drift(args):
    x_train, y_train, x_val, y_val, chunks = make_emerging_rule_stream(args)
    rows = []
    for method, model in iter_models(args):
        print(f"[latest-strong] method={method}")
        start = time.perf_counter()
        try:
            model.fit(x_train, y_train, x_val, y_val)
            fit_time = time.perf_counter() - start
            rows.extend(_record_rows("strong_rule_drift", "emerging_rule_pressure", method, model, chunks, fit_time))
        except Exception as exc:
            rows.append(
                {
                    "experiment": "strong_rule_drift",
                    "dataset": "emerging_rule_pressure",
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "method_year": METHOD_YEARS.get(method, ""),
                    "category": "failed",
                    "chunk": -1,
                    "phase": "failed",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[skip] strong method={method} error={type(exc).__name__}: {exc}")
        del model
        gc.collect()
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
    core = detail[(detail["experiment"] == "core_stream") & (detail["chunk"] >= 0)].copy()
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
    strong = detail[(detail["experiment"] == "strong_rule_drift") & (detail["chunk"] >= 0)].copy()
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


def build_latest_table(core_summary, strong_summary):
    core_map = core_summary.set_index("method") if not core_summary.empty else pd.DataFrame()
    strong_map = strong_summary.set_index("method") if not strong_summary.empty else pd.DataFrame()
    rows = []
    present = [m for m in METHOD_ORDER if m in set(core_summary["method"]).union(set(strong_summary["method"]))]
    for method in present:
        core = core_map.loc[method] if method in core_map.index else None
        strong = strong_map.loc[method] if method in strong_map.index else None
        rows.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "Year/type": METHOD_YEARS.get(method, ""),
                "Stream adaptation": "A3C rule/weight update"
                if method == "RT_A3C_Realtime_MLN"
                else ("rolling context" if "Context" in method else ("online fine-tuning" if "TabM" in method else "fixed reference")),
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


def write_markdown_table(df, path, title):
    cols = list(df.columns)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_latest_summary(core_summary, strong_summary):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    methods = [m for m in METHOD_ORDER if m in set(core_summary["method"]).union(set(strong_summary["method"]))]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    x = np.arange(len(methods))
    core_map = core_summary.set_index("method")
    strong_map = strong_summary.set_index("method")

    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.45))
    cycle_ms = [float(core_map.loc[m, "blocking_cycle_time_sec"]) * 1000.0 if m in core_map.index else np.nan for m in methods]
    axes[0].bar(x, cycle_ms, color=[COLORS[m] for m in methods], width=0.62)
    axes[0].axhline(10, color="#c1121f", ls="--", lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("Realtime cost on core streams")
    axes[0].set_ylabel("Blocking cycle (ms, log)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")

    core_f1 = [float(core_map.loc[m, "f1"]) if m in core_map.index else np.nan for m in methods]
    post_f1 = [float(strong_map.loc[m, "post_f1"]) if m in strong_map.index else np.nan for m in methods]
    axes[1].bar(x - 0.17, core_f1, width=0.34, color=[COLORS[m] for m in methods], label="Core F1")
    axes[1].bar(x + 0.17, post_f1, width=0.34, color="#495057", alpha=0.72, label="Strong post-F1")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Prediction under normal vs strong drift")
    axes[1].set_ylabel("F1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].legend(frameon=False, fontsize=8)

    active = [float(strong_map.loc[m, "active_emerging_rule_count"]) if m in strong_map.index else np.nan for m in methods]
    refresh = [float(strong_map.loc[m, "structure_refreshed"]) * 100.0 if m in strong_map.index else np.nan for m in methods]
    axes[2].bar(x - 0.17, active, width=0.34, color="#52b788", label="Active emerging rules")
    axes[2].bar(x + 0.17, refresh, width=0.34, color="#457b9d", label="Struct update %")
    axes[2].set_title("MLN rule-structure evidence")
    axes[2].set_ylabel("Count / percent")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")
    fig.suptitle("2024-2026 foundation/deep tabular baselines vs realtime adaptive MLN", y=1.02, fontsize=13, weight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Fig10_Latest_Foundation_Baselines.{ext}", dpi=260 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def write_explanations():
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    (EXPLAIN_DIR / "Table6_Latest_Foundation_Baselines_explanation.md").write_text(
        """# Table6_Latest_Foundation_Baselines 解释

## 表格作用

该表专门回应“现代基线年份不够新”的问题。它把 2024-2026 年间具有代表性的表格基础模型或深度表格模型加入同一数据流评估协议，包括 TabPFN v2、TabICL v2、TabDPT 和 TabM。

## 方法名称

- **RT-A3C-MLN (ours)**：本文方法，具备 MLN 规则新增、删除、权重调整和后台结构刷新。
- **TabPFN v2**：2025 年 Nature 表格基础模型，使用预训练 Transformer 进行小数据表格预测。
- **TabICL v2**：2026 checkpoint / 2025 论文对应的表格上下文学习模型，用训练样本作为上下文完成预测。
- **TabDPT**：2024 预印本、2025 版本的表格基础模型，强调可扩展的表格预训练和上下文检索。
- **TabM**：ICLR 2025 参数高效表格深度集成模型，本实验中采用在线微调方式适应数据流。
- **OnlineWeight MLN / Static MaxEnt**：保留作为固定规则结构参考。

## 列指标

- **Stream adaptation**：数据流阶段如何利用新 chunk。TabPFN、TabICL、TabDPT 使用滚动上下文；TabM 使用在线微调；本文方法使用 A3C 规则/权重更新。
- **Rule-structure adaptation**：是否能产生 MLN 规则结构层面的新增、删除或替换。
- **Core F1 / Core Log-loss / Core cycle ms / Core 10ms %**：常规真实数据流上的分类质量、概率质量和实时响应指标。
- **Strong post-F1 / Strong post Log-loss / Recovery lag**：强规则漂移后的恢复能力。
- **Struct update % / Active emerging rules**：是否真正激活了后半段新出现的 MLN 规则。

## 结论解读

如果最新表格基础模型在 F1 上有竞争力，但 Core cycle ms 高于 10 ms，说明其强预测能力不等价于实时响应能力。如果其 Active emerging rules 为 0，说明它不能提供 MLN 规则结构自适应证据。本文方法的关键证据是同时满足低阻塞周期、结构刷新和新规则激活。
""",
        encoding="utf-8",
    )
    (EXPLAIN_DIR / "Fig10_Latest_Foundation_Baselines_explanation.md").write_text(
        """# Fig10_Latest_Foundation_Baselines 解释

## 图的作用

该图可视化 2024-2026 近年表格基础模型/深度表格模型与 RT-A3C-MLN 的差异。它强调本文方法的论点不是静态准确率最高，而是在数据流下实时、自适应、可解释地调整 MLN 规则。

## 左侧子图

横轴是方法名称，纵轴是 Blocking cycle，单位为毫秒并采用对数坐标。红色虚线表示 10 ms 实时阈值。低于虚线表示该方法在常规数据流下满足实时响应约束。

## 中间子图

横轴是方法名称，纵轴是 F1。彩色柱表示常规数据流 F1，灰色柱表示强规则漂移后的 post-F1。该子图用于比较最新表格模型在温和流和强规则漂移下的分类表现。

## 右侧子图

绿色柱表示强漂移下激活的 emerging rules 数量，蓝色柱表示结构刷新比例。只有支持 MLN 规则结构自适应的方法才应在这两个指标上有实质性数值。

## 结论

TabPFN、TabICL、TabDPT 和 TabM 是较新的强预测器，但它们主要通过上下文推理或神经网络参数更新适应数据，不产生 MLN 规则结构层面的新增和删除。RT-A3C-MLN 的优势在于在较低阻塞周期下保留规则结构自适应证据。
""",
        encoding="utf-8",
    )


def save_outputs(detail, core_summary, strong_summary, table, args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT_DIR / "latest_foundation_detail.csv", index=False)
    core_summary.to_csv(OUT_DIR / "latest_foundation_core_summary.csv", index=False)
    strong_summary.to_csv(OUT_DIR / "latest_foundation_strong_summary.csv", index=False)
    table.to_csv(TABLE_DIR / "Table6_Latest_Foundation_Baselines.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(table, TABLE_DIR / "Table6_Latest_Foundation_Baselines.md", "Latest 2024-2026 foundation/deep tabular baselines")
    (OUT_DIR / "latest_foundation_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    plot_latest_summary(core_summary, strong_summary)
    write_explanations()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-samples", type=int, default=3200)
    parser.add_argument("--max-rules", type=int, default=160)
    parser.add_argument("--max-literals", type=int, default=65)
    parser.add_argument("--numeric-bins", type=int, default=4)
    parser.add_argument("--min-support", type=float, default=0.018)
    parser.add_argument("--stream-fraction", type=float, default=0.65)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--foundation-context-limit", type=int, default=640)
    parser.add_argument("--foundation-estimators", type=int, default=1)
    parser.add_argument("--foundation-batch-size", type=int, default=64)
    parser.add_argument("--tabm-window", type=int, default=900)
    parser.add_argument("--tabm-initial-epochs", type=int, default=8)
    parser.add_argument("--tabm-update-epochs", type=int, default=2)
    parser.add_argument("--tabm-batch-size", type=int, default=192)
    parser.add_argument("--tabm-hidden", type=int, default=80)
    parser.add_argument("--tabm-k", type=int, default=6)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=5)
    parser.add_argument("--a3c-steps", type=int, default=5)
    parser.add_argument("--a3c-rules", type=int, default=84)
    parser.add_argument("--a3c-max-active-rules", type=int, default=32)
    parser.add_argument("--a3c-rolling-window", type=int, default=1200)
    parser.add_argument("--a3c-calibration-window", type=int, default=300)
    parser.add_argument("--a3c-refit-interval", type=int, default=1)
    parser.add_argument("--a3c-structure-episodes", type=int, default=3)
    parser.add_argument("--a3c-structure-tolerance", type=float, default=0.015)
    parser.add_argument("--a3c-prior-shift-lr", type=float, default=0.24)
    parser.add_argument("--a3c-prior-shift-clip", type=float, default=1.7)
    parser.add_argument("--strong-n-features", dest="n_features", type=int, default=300)
    parser.add_argument("--strong-train-size", dest="train_size", type=int, default=1300)
    parser.add_argument("--strong-val-size", dest="val_size", type=int, default=320)
    parser.add_argument("--strong-chunks", dest="strong_chunks", type=int, default=6)
    parser.add_argument("--strong-chunk-size", dest="strong_chunk_size", type=int, default=260)
    parser.add_argument("--strong-drift-chunk", dest="drift_chunk", type=int, default=4)
    parser.add_argument("--initial-rule-budget", type=int, default=22)
    parser.add_argument("--rolling-window", type=int, default=1000)
    parser.add_argument("--recovery-f1", type=float, default=0.64)
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
    table = build_latest_table(core_summary, strong_summary)
    save_outputs(detail, core_summary, strong_summary, table, args)

    print(table.to_string(index=False))
    failed = detail[detail.get("category", "") == "failed"] if "category" in detail else pd.DataFrame()
    if not failed.empty:
        print("[failures]")
        print(failed[["experiment", "dataset", "method", "failure"]].drop_duplicates().to_string(index=False))
    print(f"[done] results: {OUT_DIR}")
    print(f"[done] artifacts: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
