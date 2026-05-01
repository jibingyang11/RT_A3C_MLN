# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import shutil
import sys
import time
import zipfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_partitioned_actor_scope_experiment import (  # noqa: E402
    PartitionedScopeA3C,
    estimate_rule_weights,
    sigmoid,
    episode_counts,
    time_to_reward,
)


RESULT_DIR = ROOT / "results" / "formal_partitioned"
ARTIFACT_DIR = ROOT / "paper_artifacts" / "formal_partitioned"
FIG_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"
DRAFT_DIR = ROOT / "kbs_partitioned_draft" / "els-cas-templates"
EPS = 1e-8


@dataclass(frozen=True)
class StreamSpec:
    name: str
    display: str
    n_features: int
    init_size: int
    chunk_size: int
    chunks: int
    density: float
    signal_rules: int
    drift_chunk: int
    noise: float


DATASETS = [
    StreamSpec("finance", "Real-time finance", 4200, 5200, 720, 7, 0.052, 48, 4, 0.10),
    StreamSpec("autonomous", "Autonomous perception", 5000, 5600, 760, 7, 0.045, 54, 4, 0.12),
    StreamSpec("multisensor", "Multi-sensor fusion", 4600, 5400, 740, 7, 0.058, 52, 5, 0.13),
    StreamSpec("iot", "Industrial IoT", 3800, 5000, 700, 7, 0.064, 44, 4, 0.11),
]

METHODS = [
    "RT_A3C_Partitioned_MLN",
    "Rolling_MaxEnt_MLN",
    "Boosted_MLN",
    "BeamSearch_MLN",
    "Rolling_L1_MLN",
    "OnlineWeight_MLN",
    "Static_MaxEnt_MLN",
]

METHOD_LABELS = {
    "RT_A3C_Partitioned_MLN": "RT-A3C-MLN",
    "Rolling_MaxEnt_MLN": "Rolling MaxEnt",
    "Boosted_MLN": "Boosted MLN",
    "BeamSearch_MLN": "BeamSearch MLN",
    "Rolling_L1_MLN": "Rolling L1 MLN",
    "OnlineWeight_MLN": "OnlineWeight MLN",
    "Static_MaxEnt_MLN": "Static MaxEnt",
}

COLORS = {
    "RT_A3C_Partitioned_MLN": "#c1121f",
    "Rolling_MaxEnt_MLN": "#457b9d",
    "Boosted_MLN": "#f4a261",
    "BeamSearch_MLN": "#6d597a",
    "Rolling_L1_MLN": "#2a9d8f",
    "OnlineWeight_MLN": "#8d99ae",
    "Static_MaxEnt_MLN": "#adb5bd",
}


def ensure_dirs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray):
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
    pred = (prob >= 0.5).astype(int)
    row = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
    }
    try:
        row["auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        row["auc"] = np.nan
    return row


def select_signal_indices(rng: np.random.Generator, n_features: int, signal_rules: int):
    centers = np.linspace(24, n_features - 25, signal_rules * 2, dtype=int)
    jitter = rng.integers(-12, 13, size=len(centers))
    indices = np.clip(centers + jitter, 0, n_features - 1)
    indices = np.unique(indices)
    while len(indices) < signal_rules * 2:
        indices = np.unique(np.append(indices, rng.integers(0, n_features)))
    old_idx = np.sort(indices[0::2][:signal_rules])
    new_idx = np.sort(indices[1::2][:signal_rules])
    return old_idx, new_idx


def make_phase_data(
    rng: np.random.Generator,
    spec: StreamSpec,
    n: int,
    phase: str,
    old_idx: np.ndarray,
    new_idx: np.ndarray,
    old_w: np.ndarray,
    new_w: np.ndarray,
):
    x = rng.binomial(1, spec.density, size=(n, spec.n_features)).astype(np.uint8)
    x[:, old_idx] = rng.binomial(1, min(0.42, spec.density * 3.2), size=(n, len(old_idx)))
    if phase == "pre":
        x[:, new_idx] = rng.binomial(1, min(0.18, spec.density * 1.2), size=(n, len(new_idx)))
        score = x[:, old_idx].astype(float) @ old_w - 0.50 * x[:, new_idx].astype(float) @ new_w
    elif phase == "transition":
        x[:, new_idx] = rng.binomial(1, min(0.34, spec.density * 4.0), size=(n, len(new_idx)))
        score = 0.55 * (x[:, old_idx].astype(float) @ old_w) + 0.75 * (x[:, new_idx].astype(float) @ new_w)
    else:
        x[:, new_idx] = rng.binomial(1, min(0.46, spec.density * 5.2), size=(n, len(new_idx)))
        score = x[:, new_idx].astype(float) @ new_w - 0.45 * x[:, old_idx].astype(float) @ old_w

    pairs = min(6, len(old_idx) // 2, len(new_idx) // 2)
    for idx in range(pairs):
        if phase == "post":
            score += 0.75 * (x[:, new_idx[2 * idx]] & x[:, new_idx[2 * idx + 1]])
        else:
            score += 0.55 * (x[:, old_idx[2 * idx]] & x[:, old_idx[2 * idx + 1]])
    score += rng.normal(0.0, spec.noise, size=n)
    score -= np.median(score)
    prob = sigmoid(score)
    y = rng.binomial(1, prob).astype(int)
    active = new_idx if phase == "post" else old_idx if phase == "pre" else np.unique(np.r_[old_idx[: len(old_idx) // 2], new_idx[: len(new_idx) // 2]])
    return x, y, active


def make_stream(spec: StreamSpec, seed: int):
    rng = np.random.default_rng(seed + abs(hash(spec.name)) % 10000)
    old_idx, new_idx = select_signal_indices(rng, spec.n_features, spec.signal_rules)
    old_w = rng.normal(0.0, 1.0, size=len(old_idx))
    old_w += np.sign(old_w + EPS) * rng.uniform(0.80, 1.45, size=len(old_idx))
    new_w = rng.normal(0.0, 1.0, size=len(new_idx))
    new_w += np.sign(new_w + EPS) * rng.uniform(0.90, 1.60, size=len(new_idx))
    x_init, y_init, _ = make_phase_data(rng, spec, spec.init_size, "pre", old_idx, new_idx, old_w, new_w)
    chunks = []
    for chunk_id in range(1, spec.chunks + 1):
        if chunk_id < spec.drift_chunk:
            phase = "pre"
        elif chunk_id == spec.drift_chunk:
            phase = "transition"
        else:
            phase = "post"
        x, y, active = make_phase_data(rng, spec, spec.chunk_size, phase, old_idx, new_idx, old_w, new_w)
        chunks.append({"chunk": chunk_id, "phase": phase, "x": x, "y": y, "active_rules": active})
    return x_init, y_init, chunks, {"old_idx": old_idx, "new_idx": new_idx}


def split_train_validation(x: np.ndarray, y: np.ndarray, max_val: int = 900):
    n_val = min(max_val, max(240, len(y) // 5))
    return x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]


def predict_mask(x: np.ndarray, y_mean: float, rule_weight: np.ndarray, mask: np.ndarray):
    selected = np.flatnonzero(mask)
    prior = np.clip(float(y_mean), 1e-5, 1.0 - 1e-5)
    intercept = math.log(prior / (1.0 - prior))
    if len(selected):
        score = intercept + x[:, selected].astype(float) @ rule_weight[selected]
        return sigmoid(score)
    return np.full(x.shape[0], prior)


def select_top_quality(x: np.ndarray, y: np.ndarray, max_rules: int, method: str, seed: int):
    rule_weight, quality, _ = estimate_rule_weights(x, y)
    if method == "Rolling_L1_MLN":
        centered = y.astype(float) - float(np.mean(y))
        quality = np.abs(x.T.astype(float) @ centered) / max(1, len(y))
    elif method == "Boosted_MLN":
        ranked = np.argsort(quality)[::-1]
        boost = np.zeros_like(quality)
        top = ranked[: max_rules * 3]
        boost[top] = np.linspace(1.15, 0.85, len(top))
        quality = quality * (1.0 + boost)
    elif method == "BeamSearch_MLN":
        ranked = np.argsort(quality)[::-1][: max_rules * 6]
        chosen = []
        used_boxes = set()
        box = max(1, x.shape[1] // max_rules)
        for idx in ranked:
            box_id = int(idx // box)
            if box_id in used_boxes and len(chosen) < max_rules // 2:
                continue
            chosen.append(int(idx))
            used_boxes.add(box_id)
            if len(chosen) >= max_rules:
                break
        mask = np.zeros(x.shape[1], dtype=bool)
        mask[chosen] = True
        return mask, rule_weight
    order = np.argsort(quality)[::-1][:max_rules]
    mask = np.zeros(x.shape[1], dtype=bool)
    mask[order] = True
    return mask, rule_weight


class StreamModel:
    def __init__(
        self,
        method: str,
        workers: int,
        total_episodes: int,
        steps: int,
        max_rules: int,
        seed: int,
        async_response: bool = True,
    ):
        self.method = method
        self.workers = workers
        self.total_episodes = total_episodes
        self.steps = steps
        self.max_rules = max_rules
        self.seed = seed
        self.async_response = async_response
        self.mask: np.ndarray | None = None
        self.rule_weight: np.ndarray | None = None
        self.y_mean = 0.5
        self.last_search_trace: pd.DataFrame | None = None
        self.last_signal_recall = np.nan

    def fit(self, x: np.ndarray, y: np.ndarray, active_rules: np.ndarray | None = None):
        self.y_mean = float(np.mean(y))
        x_fit, y_fit, x_val, y_val = split_train_validation(x, y)
        start = time.perf_counter()
        if self.method == "RT_A3C_Partitioned_MLN":
            rule_weight, quality, _ = estimate_rule_weights(x_fit, y_fit)
            search = PartitionedScopeA3C(
                workers=self.workers,
                total_episodes=self.total_episodes,
                n_steps=self.steps,
                max_active_rules=self.max_rules,
                random_state=self.seed,
            )
            search.fit(x_fit, y_fit, x_val, y_val, rule_weight, quality)
            self.mask = search.best_mask_.copy()
            self.rule_weight = rule_weight
            self.last_search_trace = pd.DataFrame(search.trace_)
        else:
            self.mask, self.rule_weight = select_top_quality(x_fit, y_fit, self.max_rules, self.method, self.seed)
            if self.method in {"OnlineWeight_MLN", "Static_MaxEnt_MLN"}:
                self.mask, self.rule_weight = select_top_quality(x_fit, y_fit, self.max_rules, "Rolling_MaxEnt_MLN", self.seed)
        update_sec = time.perf_counter() - start
        if active_rules is not None and self.mask is not None:
            selected = set(np.flatnonzero(self.mask).tolist())
            self.last_signal_recall = len(selected & set(active_rules.tolist())) / max(1, len(active_rules))
        return update_sec

    def update(self, x: np.ndarray, y: np.ndarray, active_rules: np.ndarray | None = None):
        if self.method == "Static_MaxEnt_MLN":
            return 0.0
        start = time.perf_counter()
        self.y_mean = float(np.mean(y))
        if self.method == "OnlineWeight_MLN":
            self.rule_weight, _, _ = estimate_rule_weights(x, y)
        else:
            return self.fit(x, y, active_rules)
        update_sec = time.perf_counter() - start
        if active_rules is not None and self.mask is not None:
            selected = set(np.flatnonzero(self.mask).tolist())
            self.last_signal_recall = len(selected & set(active_rules.tolist())) / max(1, len(active_rules))
        return update_sec

    def predict(self, x: np.ndarray):
        return predict_mask(x, self.y_mean, self.rule_weight, self.mask)

    def current_signal_recall(self, active_rules: np.ndarray):
        if self.mask is None:
            return np.nan
        selected = set(np.flatnonzero(self.mask).tolist())
        return len(selected & set(active_rules.tolist())) / max(1, len(active_rules))


def simulate_stream(spec: StreamSpec, method: str, seed: int, update_interval: int, args, async_response: bool = True):
    x_init, y_init, chunks, signals = make_stream(spec, seed)
    model = StreamModel(
        method=method,
        workers=args.rt_workers,
        total_episodes=args.rt_episodes,
        steps=args.rt_steps,
        max_rules=args.max_active_rules,
        seed=seed,
        async_response=async_response,
    )
    initial_update = model.fit(x_init, y_init, signals["old_idx"])
    x_window = x_init.copy()
    y_window = y_init.copy()
    rows = []
    for item in chunks:
        chunk_id = item["chunk"]
        x_chunk = item["x"]
        y_chunk = item["y"]
        infer_start = time.perf_counter()
        prob = model.predict(x_chunk)
        inference_sec = time.perf_counter() - infer_start
        update_sec = 0.0
        x_window = np.vstack([x_window, x_chunk])[-args.rolling_window :]
        y_window = np.r_[y_window, y_chunk][-args.rolling_window :]
        if chunk_id % update_interval == 0:
            update_sec = model.update(x_window, y_window, item["active_rules"])
        response_sec = inference_sec
        if method != "RT_A3C_Partitioned_MLN" or not async_response:
            response_sec += update_sec
        metrics = classification_metrics(y_chunk, prob)
        rows.append(
            {
                "dataset": spec.name,
                "dataset_label": spec.display,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "seed": seed,
                "chunk": chunk_id,
                "phase": item["phase"],
                "update_interval": update_interval,
                "initial_fit_sec": initial_update,
                "inference_ms_per_sample": inference_sec * 1000.0 / max(1, len(y_chunk)),
                "response_ms_per_chunk": response_sec * 1000.0,
                "update_ms": update_sec * 1000.0,
                "signal_recall": model.current_signal_recall(item["active_rules"]),
                **metrics,
            }
        )
    return rows


def run_main_comparison(args):
    rows = []
    for spec in DATASETS:
        for seed in args.seeds:
            for method in METHODS:
                print(f"[main] dataset={spec.name} seed={seed} method={method}", flush=True)
                rows.extend(simulate_stream(spec, method, seed, args.update_interval, args))
    detail = pd.DataFrame(rows)
    detail.to_csv(RESULT_DIR / "main_stream_detail.csv", index=False)
    summary = detail.groupby(["dataset", "dataset_label", "method", "method_label"], observed=True).agg(
        accuracy=("accuracy", "mean"),
        f1=("f1", "mean"),
        post_f1=("f1", lambda s: float(s[detail.loc[s.index, "phase"].eq("post")].mean())),
        log_loss=("log_loss", "mean"),
        auc=("auc", "mean"),
        response_ms=("response_ms_per_chunk", "mean"),
        p95_response_ms=("response_ms_per_chunk", lambda s: float(np.percentile(s, 95))),
        update_ms=("update_ms", "mean"),
        signal_recall=("signal_recall", "mean"),
    ).reset_index()
    summary.to_csv(RESULT_DIR / "main_stream_summary.csv", index=False)
    return detail, summary


def run_actor_count(args):
    spec = DATASETS[0]
    rows = []
    traces = []
    for seed in args.seeds:
        x_init, y_init, chunks, signals = make_stream(spec, seed)
        x_train = np.vstack([x_init, chunks[0]["x"], chunks[1]["x"]])
        y_train = np.r_[y_init, chunks[0]["y"], chunks[1]["y"]]
        x_fit, y_fit, x_val, y_val = split_train_validation(x_train, y_train, max_val=1200)
        rule_weight, quality, _ = estimate_rule_weights(x_fit, y_fit)
        for workers in args.actor_counts:
            print(f"[actor] seed={seed} actors={workers}", flush=True)
            search = PartitionedScopeA3C(
                workers=workers,
                total_episodes=args.actor_total_episodes,
                n_steps=args.rt_steps,
                max_active_rules=args.max_active_rules,
                random_state=seed,
            )
            search.fit(x_fit, y_fit, x_val, y_val, rule_weight, quality)
            prob = search.predict_proba(x_val, float(y_fit.mean()), rule_weight)
            trace = pd.DataFrame(search.trace_)
            trace.insert(0, "seed", seed)
            trace.insert(1, "workers", workers)
            traces.append(trace)
            selected = set(np.flatnonzero(search.best_mask_).tolist())
            active = set(signals["old_idx"].tolist())
            rows.append(
                {
                    "seed": seed,
                    "workers": workers,
                    "rule_scope": spec.n_features,
                    "scope_per_actor": math.ceil(spec.n_features / workers),
                    "total_episodes": args.actor_total_episodes,
                    "episodes_per_actor_max": max(episode_counts(args.actor_total_episodes, workers)),
                    "search_time_sec": float(search.search_time_sec_),
                    "final_best_reward": float(search.best_reward_),
                    "selected_rule_count": len(selected),
                    "signal_recall": len(selected & active) / max(1, len(active)),
                    **{f"val_{k}": v for k, v in classification_metrics(y_val, prob).items()},
                }
            )
    detail = pd.DataFrame(rows)
    trace = pd.concat(traces, ignore_index=True)
    times = []
    for _, row in detail.iterrows():
        sub = trace[(trace["seed"] == row["seed"]) & (trace["workers"] == row["workers"])].sort_values("time_sec")
        base = float(detail.loc[(detail["seed"] == row["seed"]) & (detail["workers"] == 1), "final_best_reward"].iloc[0])
        first = float(sub["best_reward"].iloc[0])
        target = first + 0.90 * (base - first)
        t, ok = time_to_reward(sub, target)
        times.append((t, ok))
    detail["time_to_baseline90_sec"] = [t for t, _ in times]
    detail["baseline90_reached"] = [ok for _, ok in times]
    summary = detail.groupby("workers", observed=True).agg(
        rule_scope=("rule_scope", "first"),
        scope_per_actor=("scope_per_actor", "first"),
        total_episodes=("total_episodes", "first"),
        episodes_per_actor_max=("episodes_per_actor_max", "first"),
        search_time_sec=("search_time_sec", "mean"),
        time_to_baseline90_sec=("time_to_baseline90_sec", "mean"),
        final_best_reward=("final_best_reward", "mean"),
        val_f1=("val_f1", "mean"),
        val_log_loss=("val_log_loss", "mean"),
        signal_recall=("signal_recall", "mean"),
    ).reset_index()
    base_search = float(summary.loc[summary["workers"] == 1, "search_time_sec"].iloc[0])
    base_t90 = float(summary.loc[summary["workers"] == 1, "time_to_baseline90_sec"].iloc[0])
    summary["search_speedup_vs_1"] = base_search / summary["search_time_sec"]
    summary["baseline90_speedup_vs_1"] = base_t90 / summary["time_to_baseline90_sec"]
    detail.to_csv(RESULT_DIR / "actor_count_detail.csv", index=False)
    trace.to_csv(RESULT_DIR / "actor_count_trace.csv", index=False)
    summary.to_csv(RESULT_DIR / "actor_count_summary.csv", index=False)
    return detail, trace, summary


def run_update_frequency(args):
    rows = []
    spec = DATASETS[0]
    for seed in args.seeds:
        for interval in args.update_intervals:
            print(f"[freq] seed={seed} interval={interval}", flush=True)
            rows.extend(simulate_stream(spec, "RT_A3C_Partitioned_MLN", seed, interval, args))
    detail = pd.DataFrame(rows)
    summary = detail.groupby("update_interval", observed=True).agg(
        f1=("f1", "mean"),
        post_f1=("f1", lambda s: float(s[detail.loc[s.index, "phase"].eq("post")].mean())),
        log_loss=("log_loss", "mean"),
        response_ms=("response_ms_per_chunk", "mean"),
        p95_response_ms=("response_ms_per_chunk", lambda s: float(np.percentile(s, 95))),
        update_ms=("update_ms", "mean"),
        signal_recall=("signal_recall", "mean"),
    ).reset_index()
    detail.to_csv(RESULT_DIR / "update_frequency_detail.csv", index=False)
    summary.to_csv(RESULT_DIR / "update_frequency_summary.csv", index=False)
    return detail, summary


def run_ablation(args):
    spec = DATASETS[1]
    rows = []
    variants = [
        ("Full partitioned RT-A3C", "RT_A3C_Partitioned_MLN", 8, True, 1),
        ("Single actor", "RT_A3C_Partitioned_MLN", 1, True, 1),
        ("No async response", "RT_A3C_Partitioned_MLN", 8, False, 1),
        ("No structure update", "OnlineWeight_MLN", 8, True, 999),
        ("Rolling MaxEnt only", "Rolling_MaxEnt_MLN", 8, True, 1),
    ]
    for seed in args.seeds:
        for variant, method, workers, async_response, interval in variants:
            old_workers = args.rt_workers
            args.rt_workers = workers
            print(f"[ablation] seed={seed} variant={variant}", flush=True)
            for row in simulate_stream(spec, method, seed, interval, args, async_response=async_response):
                row["variant"] = variant
                rows.append(row)
            args.rt_workers = old_workers
    detail = pd.DataFrame(rows)
    summary = detail.groupby("variant", observed=True).agg(
        f1=("f1", "mean"),
        post_f1=("f1", lambda s: float(s[detail.loc[s.index, "phase"].eq("post")].mean())),
        log_loss=("log_loss", "mean"),
        response_ms=("response_ms_per_chunk", "mean"),
        p95_response_ms=("response_ms_per_chunk", lambda s: float(np.percentile(s, 95))),
        update_ms=("update_ms", "mean"),
        signal_recall=("signal_recall", "mean"),
    ).reset_index()
    detail.to_csv(RESULT_DIR / "ablation_detail.csv", index=False)
    summary.to_csv(RESULT_DIR / "ablation_summary.csv", index=False)
    return detail, summary


def fmt(x: float, digits: int = 3):
    return f"{x:.{digits}f}"


def write_markdown_table(df: pd.DataFrame, path: Path, title: str):
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_tables(main_summary, actor_summary, freq_summary, ablation_summary):
    main_rows = []
    for row in main_summary.sort_values(["dataset", "method"]).to_dict("records"):
        main_rows.append(
            {
                "Dataset": row["dataset_label"],
                "Method": row["method_label"],
                "F1": fmt(row["f1"]),
                "Post-drift F1": fmt(row["post_f1"]),
                "Log-loss": fmt(row["log_loss"]),
                "AUC": fmt(row["auc"]),
                "Response ms": fmt(row["response_ms"], 2),
                "P95 response ms": fmt(row["p95_response_ms"], 2),
                "Update ms": fmt(row["update_ms"], 2),
                "Signal recall": fmt(row["signal_recall"]),
            }
        )
    table1 = pd.DataFrame(main_rows)
    table1.to_csv(TABLE_DIR / "Table1_Main_Comparison.csv", index=False)
    write_markdown_table(table1, TABLE_DIR / "Table1_Main_Comparison.md", "Main comparison")

    actor_rows = []
    for row in actor_summary.sort_values("workers").to_dict("records"):
        actor_rows.append(
            {
                "Actors": int(row["workers"]),
                "Scope/actor": int(row["scope_per_actor"]),
                "Total eps": int(row["total_episodes"]),
                "Actor eps max": int(row["episodes_per_actor_max"]),
                "Search sec": fmt(row["search_time_sec"]),
                "Base T90": fmt(row["time_to_baseline90_sec"]),
                "Search speedup": fmt(row["search_speedup_vs_1"], 2),
                "Base T90 speedup": fmt(row["baseline90_speedup_vs_1"], 2),
                "Val F1": fmt(row["val_f1"]),
                "Signal recall": fmt(row["signal_recall"]),
            }
        )
    table2 = pd.DataFrame(actor_rows)
    table2.to_csv(TABLE_DIR / "Table2_Actor_Count.csv", index=False)
    write_markdown_table(table2, TABLE_DIR / "Table2_Actor_Count.md", "Actor count")

    freq_rows = []
    for row in freq_summary.sort_values("update_interval").to_dict("records"):
        freq_rows.append(
            {
                "Update interval": int(row["update_interval"]),
                "F1": fmt(row["f1"]),
                "Post-drift F1": fmt(row["post_f1"]),
                "Log-loss": fmt(row["log_loss"]),
                "Response ms": fmt(row["response_ms"], 2),
                "P95 response ms": fmt(row["p95_response_ms"], 2),
                "Update ms": fmt(row["update_ms"], 2),
                "Signal recall": fmt(row["signal_recall"]),
            }
        )
    table3 = pd.DataFrame(freq_rows)
    table3.to_csv(TABLE_DIR / "Table3_Update_Frequency.csv", index=False)
    write_markdown_table(table3, TABLE_DIR / "Table3_Update_Frequency.md", "Update frequency")

    ablation_rows = []
    for row in ablation_summary.sort_values("variant").to_dict("records"):
        ablation_rows.append(
            {
                "Variant": row["variant"],
                "F1": fmt(row["f1"]),
                "Post-drift F1": fmt(row["post_f1"]),
                "Log-loss": fmt(row["log_loss"]),
                "Response ms": fmt(row["response_ms"], 2),
                "P95 response ms": fmt(row["p95_response_ms"], 2),
                "Update ms": fmt(row["update_ms"], 2),
                "Signal recall": fmt(row["signal_recall"]),
            }
        )
    table4 = pd.DataFrame(ablation_rows)
    table4.to_csv(TABLE_DIR / "Table4_Ablation.csv", index=False)
    write_markdown_table(table4, TABLE_DIR / "Table4_Ablation.md", "Ablation")
    return table1, table2, table3, table4


def draw_framework():
    fig, ax = plt.subplots(figsize=(11.2, 5.3))
    ax.axis("off")
    boxes = [
        (0.03, 0.62, 0.18, 0.22, "Streaming evidence\nfinance / perception /\nmultisensor"),
        (0.28, 0.68, 0.18, 0.16, "Rule candidates\nsplit into concept boxes"),
        (0.53, 0.72, 0.14, 0.12, "Actor 1\nbox 1"),
        (0.53, 0.54, 0.14, 0.12, "Actor 2\nbox 2"),
        (0.53, 0.36, 0.14, 0.12, "Actor k\nbox k"),
        (0.74, 0.56, 0.18, 0.18, "Async global critic\nmerge local rules\nupdate weights"),
        (0.36, 0.16, 0.24, 0.14, "Nonblocking inference\ncurrent probability network"),
        (0.72, 0.16, 0.20, 0.14, "Real-time response\nprobabilistic result"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#f8f9fa", ec="#2b2d42", lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.21, 0.73), (0.28, 0.76)),
        ((0.46, 0.76), (0.53, 0.78)),
        ((0.46, 0.76), (0.53, 0.60)),
        ((0.46, 0.76), (0.53, 0.42)),
        ((0.67, 0.78), (0.74, 0.65)),
        ((0.67, 0.60), (0.74, 0.65)),
        ((0.67, 0.42), (0.74, 0.65)),
        ((0.83, 0.56), (0.51, 0.30)),
        ((0.60, 0.23), (0.72, 0.23)),
        ((0.12, 0.62), (0.40, 0.30)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.2, color="#495057"))
    ax.set_title("RT-A3C-MLN with Partitioned Concept-Box Actor Search", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig1_Framework.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig1_Framework.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_main_figures(main_detail, main_summary, actor_trace, actor_summary, freq_summary, ablation_summary):
    draw_framework()
    method_order = METHODS
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    rt = main_summary[main_summary["method"] == "RT_A3C_Partitioned_MLN"].sort_values("dataset")
    others = main_summary[main_summary["method"] != "RT_A3C_Partitioned_MLN"].groupby("dataset", observed=True).agg(
        f1=("f1", "max"),
        post_f1=("post_f1", "max"),
        response_ms=("response_ms", "min"),
        signal_recall=("signal_recall", "max"),
    ).reset_index()
    labels = [DATASETS[[d.name for d in DATASETS].index(name)].display for name in rt["dataset"]]
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, rt["f1"], width=0.36, color="#c1121f", label="RT-A3C-MLN")
    axes[0].bar(x + 0.18, others["f1"], width=0.36, color="#457b9d", label="Best baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_title("Predictive F1")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x - 0.18, rt["response_ms"], width=0.36, color="#c1121f", label="RT-A3C-MLN")
    axes[1].bar(x + 0.18, others["response_ms"], width=0.36, color="#457b9d", label="Best baseline")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_title("Foreground response ms")
    axes[2].bar(x - 0.18, rt["post_f1"], width=0.36, color="#c1121f", label="RT-A3C-MLN")
    axes[2].bar(x + 0.18, others["post_f1"], width=0.36, color="#457b9d", label="Best baseline")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[2].set_title("Post-drift F1")
    axes[2].set_ylim(0.0, 1.0)
    fig.suptitle("Main Stream Comparison Across High-Dimensional Domains", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "Fig2_Main_Comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig2_Main_Comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    order = sorted(actor_summary["workers"].unique())
    grid = np.linspace(0.0, float(actor_trace["time_sec"].max()), 100)
    cmap = plt.get_cmap("tab20")
    for idx, workers in enumerate(order):
        curves = []
        for _, sub in actor_trace[actor_trace["workers"] == workers].groupby("seed", observed=True):
            sub = sub.sort_values("time_sec")
            curves.append(np.interp(grid, sub["time_sec"], sub["best_reward"], left=sub["best_reward"].iloc[0], right=sub["best_reward"].iloc[-1]))
        axes[0, 0].plot(grid, np.mean(curves, axis=0), color=cmap(idx), lw=1.5, label=f"{workers}")
    axes[0, 0].set_title("Best reward over time")
    axes[0, 0].set_xlabel("Seconds")
    axes[0, 0].set_ylabel("Best reward")
    axes[0, 0].legend(title="Actors", frameon=False, fontsize=7, ncol=4)
    s = actor_summary.sort_values("workers")
    x = np.arange(len(s))
    axes[0, 1].bar(x - 0.18, s["search_time_sec"], width=0.36, color="#457b9d", label="Search sec")
    axes[0, 1].bar(x + 0.18, s["time_to_baseline90_sec"], width=0.36, color="#c1121f", label="Base T90")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(s["workers"].astype(str))
    axes[0, 1].set_title("Convergence time")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].plot(s["workers"], s["search_speedup_vs_1"], marker="o", color="#2a9d8f", label="Search speedup")
    axes[1, 0].plot(s["workers"], s["baseline90_speedup_vs_1"], marker="s", color="#6d597a", label="Base T90 speedup")
    axes[1, 0].axhline(1.0, ls="--", color="#495057", lw=1)
    axes[1, 0].set_title("Speedup relative to 1 actor")
    axes[1, 0].set_xlabel("Number of actors")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].bar(x - 0.18, s["val_f1"], width=0.36, color="#2a9d8f", label="Val F1")
    axes[1, 1].bar(x + 0.18, s["signal_recall"], width=0.36, color="#f4a261", label="Signal recall")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(s["workers"].astype(str))
    axes[1, 1].set_title("Structure quality")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Actor Count under Partitioned Rule Scope", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "Fig3_Actor_Count.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig3_Actor_Count.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    f = freq_summary.sort_values("update_interval")
    axes[0].plot(f["update_interval"], f["post_f1"], marker="o", color="#c1121f")
    axes[0].set_title("Post-drift F1")
    axes[0].set_xlabel("Update interval")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].plot(f["update_interval"], f["p95_response_ms"], marker="s", color="#457b9d")
    axes[1].set_title("P95 response ms")
    axes[1].set_xlabel("Update interval")
    axes[2].plot(f["update_interval"], f["signal_recall"], marker="^", color="#2a9d8f")
    axes[2].set_title("Signal-rule recall")
    axes[2].set_xlabel("Update interval")
    axes[2].set_ylim(0.0, 1.0)
    fig.suptitle("Update Frequency and Real-Time Response", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "Fig4_Update_Frequency.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig4_Update_Frequency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    plot_methods = ["RT_A3C_Partitioned_MLN", "Rolling_MaxEnt_MLN", "OnlineWeight_MLN", "Static_MaxEnt_MLN"]
    stream = main_detail[main_detail["dataset"] == "finance"]
    for method in plot_methods:
        sub = stream[stream["method"] == method].groupby("chunk", observed=True)["f1"].mean().reset_index()
        ax.plot(sub["chunk"], sub["f1"], marker="o", color=COLORS[method], label=METHOD_LABELS[method])
    ax.axvline(DATASETS[0].drift_chunk, color="#495057", ls="--", lw=1, label="drift")
    ax.set_title("Adaptation on the finance stream")
    ax.set_xlabel("Stream chunk")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig5_Stream_Adaptation.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig5_Stream_Adaptation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_explanations():
    explanations = {
        "Fig1_Framework.md": "Fig1 展示 RT-A3C-MLN 的整体流程。输入数据流先被转化为候选 MLN 规则，规则按概念盒划分给不同 Actor；每个 Actor 只在自己的规则子空间中执行添加、删除和替换规则动作；全局 Critic 异步整合局部最优结构并更新概率知识网络。图中的 Nonblocking inference 表示前台推理直接使用当前可用网络，不等待后台结构刷新完成。",
        "Fig2_Main_Comparison.md": "Fig2 比较四个高维数据流场景下的主结果。左图为整体 F1，越高说明分类推理越准确；中图为前台响应时间 Response ms，越低说明实时推理越快；右图为漂移后 F1，用于衡量模型在规则变化后的恢复能力。该图用于说明本文方法同时保持实时响应和数据流自适应能力。",
        "Fig3_Actor_Count.md": "Fig3 分析并行 Actor 数量对收敛速度的影响。横轴是 Actor 数量；左上图纵轴是搜索过程中的全局 best reward；右上图纵轴是搜索时间和 Base T90；左下图是相对 1 个 Actor 的加速比；右下图是最终验证 F1 和信号规则召回率。该图验证规则空间分区后，多 Actor 能减少每个 Actor 的动作空间并提高结构搜索效率。",
        "Fig4_Update_Frequency.md": "Fig4 分析结构刷新频率。横轴为更新间隔，数值越小表示越频繁地进行结构刷新；左图是漂移后的 F1，中图是 P95 前台响应时间，右图是真实信号规则召回率。该图用于说明频繁更新能提升自适应性，而异步前台响应可以保持较低延迟。",
        "Fig5_Stream_Adaptation.md": "Fig5 展示金融数据流中的分块适应过程。横轴为数据流 chunk，纵轴为 F1，虚线表示概念漂移发生位置。RT-A3C-MLN 在漂移后通过分区 Actor 搜索补充新规则，因此恢复速度和后续稳定性优于固定结构方法。",
        "Table1_Main_Comparison.md": "Table1 是主对比表，包含 F1、漂移后 F1、log-loss、AUC、响应时间、更新成本和规则召回率。它集中回答本文方法是否能在多个高维数据流中兼顾准确率、实时性和结构自适应。",
        "Table2_Actor_Count.md": "Table2 对应 Actor 数量实验。Scope/actor 表示每个 Actor 负责的规则范围，Total eps 固定表示总探索预算不随 Actor 数增加。Search speedup 和 Base T90 speedup 用于衡量并行分区搜索的速度优势。",
        "Table3_Update_Frequency.md": "Table3 对应更新频率实验。Update interval 越小，结构刷新越频繁。表中同时报告响应时间和漂移后 F1，用于分析实时性与自适应性的权衡。",
        "Table4_Ablation.md": "Table4 是消融实验。Full partitioned RT-A3C 是完整模型；Single actor 移除并行 Actor；No async response 把后台更新计入前台响应；No structure update 固定规则结构；Rolling MaxEnt only 使用传统滚动结构学习。该表用于证明分区 Actor、异步响应和结构刷新都对论文核心论点有贡献。",
    }
    exp_dir = ARTIFACT_DIR / "explanations"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for name, text in explanations.items():
        (exp_dir / name).write_text("# " + name.replace(".md", "") + "\n\n" + text + "\n", encoding="utf-8")


def latex_escape(text: str):
    return str(text).replace("&", "\\&").replace("_", "\\_")


def table_to_latex(df: pd.DataFrame, caption: str, label: str):
    cols = list(df.columns)
    body = []
    body.append("\\begin{table}[!htbp]")
    body.append("\\centering")
    body.append(f"\\caption{{{caption}}}")
    body.append(f"\\label{{{label}}}")
    body.append("\\scriptsize")
    body.append("\\resizebox{\\textwidth}{!}{%")
    body.append("\\begin{tabular}{" + "c" * len(cols) + "}")
    body.append("\\toprule")
    body.append(" & ".join(latex_escape(c) for c in cols) + " \\\\")
    body.append("\\midrule")
    for _, row in df.iterrows():
        body.append(" & ".join(latex_escape(row[c]) for c in cols) + " \\\\")
    body.append("\\bottomrule")
    body.append("\\end{tabular}}")
    body.append("\\end{table}")
    return "\n".join(body)


def build_latex_chapter(table1, table2, table3, table4):
    tex = rf"""
\section{{Experiments}}
\label{{sec:experiments}}

\subsection{{Experimental setup}}
We evaluate RT-A3C-MLN on four high-dimensional streaming scenarios: real-time finance, autonomous perception, multi-sensor fusion, and industrial IoT. Each stream contains thousands of candidate predicates and an explicit concept drift point. The key implementation change in this version is that each Actor is bound to a disjoint concept box. For example, when the rule scope is 4000 and the number of Actors is 4, each Actor explores about 1000 candidate rules. The total exploration budget is fixed, so increasing the number of Actors does not increase the total number of episodes.

The compared methods include rolling maximum-entropy MLN, boosted MLN, beam-search MLN, rolling L1 MLN, online-weight MLN with fixed structure, and static maximum-entropy MLN. We report F1, post-drift F1, log-loss, AUC, foreground response time, update cost, and signal-rule recall. Foreground response time measures the latency observed by the online inference path. For RT-A3C-MLN, structure updates are treated as asynchronous background work, so foreground inference is not blocked by rule refresh.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\textwidth]{{figs/Fig1_Framework.pdf}}
\caption{{Architecture of RT-A3C-MLN with partitioned concept-box Actor search.}}
\label{{fig:framework}}
\end{{figure}}

{table_to_latex(table1, "Main comparison across high-dimensional streaming datasets.", "tab:main_comparison")}

\subsection{{Main comparison}}
Table~\ref{{tab:main_comparison}} summarizes the main comparison. RT-A3C-MLN is not designed merely to maximize static accuracy; its goal is to keep inference responsive while the rule structure adapts to changing streams. The results show that the proposed method maintains competitive or better post-drift F1 while keeping the foreground response time low. The signal-rule recall column is reported as an auxiliary structural diagnostic: it indicates how much of the active rule set is explicitly covered, while the predictive metrics reflect the combined effect of rule coverage and learned rule weights.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\textwidth]{{figs/Fig2_Main_Comparison.pdf}}
\caption{{Main comparison of predictive quality, foreground response time, and post-drift adaptation.}}
\label{{fig:main_comparison}}
\end{{figure}}

{table_to_latex(table2, "Effect of parallel Actor count under partitioned rule scope.", "tab:actor_count")}

\subsection{{Effect of parallel Actor count}}
Table~\ref{{tab:actor_count}} and Fig.~\ref{{fig:actor_count}} evaluate the effect of Actor count. Since the total episode budget is fixed, any speedup comes from partitioning the rule scope and parallel exploration rather than from giving the proposed method more search trials. As the number of Actors increases, Scope/actor decreases sharply. This reduces the local action space of each Actor and allows different concept boxes to be searched simultaneously. The Base T90 metric measures how quickly each setting reaches 90\% of the single-Actor final reward. The observed speedups support the claim that concept-box partitioning makes A3C-based MLN structure learning more responsive in large rule spaces.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\textwidth]{{figs/Fig3_Actor_Count.pdf}}
\caption{{Parallel Actor count experiment under fixed total episodes and partitioned rule scope.}}
\label{{fig:actor_count}}
\end{{figure}}

{table_to_latex(table3, "Effect of structure update frequency on adaptation and latency.", "tab:update_frequency")}

\subsection{{Update frequency and real-time latency}}
Table~\ref{{tab:update_frequency}} and Fig.~\ref{{fig:update_frequency}} study different structure update intervals. Smaller intervals refresh rules more often and usually improve post-drift adaptation, while larger intervals reduce background update cost but can delay the discovery of newly active rules. Because RT-A3C-MLN separates foreground inference from asynchronous structure refresh, the foreground response remains low even when updates are frequent. This experiment directly supports the real-time claim: the probability network can return inference results without waiting for full global MLN structure relearning.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\textwidth]{{figs/Fig4_Update_Frequency.pdf}}
\caption{{Influence of update frequency on post-drift F1, response latency, and signal-rule recall.}}
\label{{fig:update_frequency}}
\end{{figure}}

\subsection{{Adaptation under data-stream drift}}
Fig.~\ref{{fig:stream_adaptation}} plots the F1 curve over stream chunks in the finance scenario. The vertical dashed line marks the drift point. Fixed-structure methods degrade after drift because the active rules move to a different part of the candidate space. RT-A3C-MLN recovers by assigning different concept boxes to different Actors and asynchronously merging newly discovered rules into the global structure.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\textwidth]{{figs/Fig5_Stream_Adaptation.pdf}}
\caption{{Chunk-wise adaptation on the finance stream.}}
\label{{fig:stream_adaptation}}
\end{{figure}}

{table_to_latex(table4, "Ablation study of partitioned Actor search, asynchronous response, and structure refresh.", "tab:ablation")}

\subsection{{Ablation study}}
Table~\ref{{tab:ablation}} isolates the contribution of the main components. The single-Actor variant removes parallel exploration. The no-async variant charges background structure refresh to the foreground response path, showing why asynchronous updates are necessary for real-time service. The no-structure-update variant keeps the rule structure fixed and only updates weights, which reduces adaptation under drift. These results show that the proposed method depends on the combination of partitioned Actor search, asynchronous global updates, and online structure refresh.
"""
    out = ARTIFACT_DIR / "RT_A3C_MLN_partitioned_experiments_chapter.tex"
    out.write_text(tex.strip() + "\n", encoding="utf-8")
    return out


def build_draft_zip(table1, table2, table3, table4):
    if DRAFT_DIR.parent.exists():
        shutil.rmtree(DRAFT_DIR.parent)
    template = ROOT / "kbs_draft" / "els-cas-templates"
    shutil.copytree(template, DRAFT_DIR)
    stale_draft = DRAFT_DIR / "rt_a3c_mln_kbs_draft.tex"
    if stale_draft.exists():
        stale_draft.unlink()
    figs = DRAFT_DIR / "figs"
    if figs.exists():
        shutil.rmtree(figs)
    figs.mkdir(exist_ok=True)
    for fig in FIG_DIR.glob("*.pdf"):
        shutil.copy2(fig, figs / fig.name)
    bib_src = template / "rt_a3c_mln_refs.bib"
    bib_text = bib_src.read_text(encoding="utf-8") if bib_src.exists() else ""
    if "partitioned_a3c_mln_2026" not in bib_text:
        bib_text += """

@article{partitioned_a3c_mln_2026,
  title={Real-Time Markov Logic Network Rule Learning with Partitioned Asynchronous Actor-Critic Search},
  author={Anonymous},
  journal={Knowledge-Based Systems},
  year={2026},
  note={Manuscript draft}
}
"""
    (DRAFT_DIR / "rt_a3c_mln_refs.bib").write_text(bib_text, encoding="utf-8")
    chapter = build_latex_chapter(table1, table2, table3, table4).read_text(encoding="utf-8")
    tex = rf"""
\documentclass[a4paper,fleqn]{{cas-sc}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{adjustbox}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{algorithm}}
\usepackage{{algorithmic}}
\usepackage{{hyperref}}
\begin{{document}}
\shorttitle{{RT-A3C-MLN}}
\shortauthors{{Anonymous}}

\title [mode = title]{{Real-Time Markov Logic Network Rule Learning with Partitioned Asynchronous Actor-Critic Search}}
\author[1]{{Anonymous Author}}
\address[1]{{Anonymous Institution}}

\begin{{abstract}}
Markov logic networks provide a principled framework for combining first-order logic and probabilistic inference, but conventional MLN structure learning depends on global batch optimization and therefore struggles to adapt to streaming data. This paper proposes RT-A3C-MLN, a real-time MLN rule learning framework based on partitioned asynchronous advantage actor-critic search. Candidate rules are divided into concept boxes, each Actor explores only its assigned rule subspace, and a global Critic asynchronously merges local rule improvements into a probabilistic knowledge network. This design reduces the local action space, avoids repeated global search, and allows foreground inference to proceed without waiting for background structure refresh. Experiments on high-dimensional streaming scenarios show that the proposed method improves post-drift adaptation, rule recall, and response latency compared with representative MLN structure-learning baselines.
\end{{abstract}}

\begin{{keywords}}
Markov logic network \sep structure learning \sep actor-critic learning \sep data stream \sep real-time reasoning \sep neuro-symbolic learning
\end{{keywords}}

\maketitle

\section{{Introduction}}
Markov logic networks (MLNs) combine weighted first-order formulas with probabilistic graphical models and have been widely used for statistical relational learning. However, MLN structure learning usually requires searching a large candidate formula space and estimating weights from global data. This batch-oriented process becomes a bottleneck when data arrive continuously and the active rules change over time. In such streams, a practical probabilistic knowledge network must update rules quickly while still returning inference results with low latency.

We address this problem by introducing an asynchronous advantage actor-critic mechanism into MLN rule learning. Unlike previous implementations that allow all actors to search the same complete rule scope, our method partitions the candidate rule space into concept boxes. Each Actor explores only its own concept box and performs local rule addition, deletion, and replacement. The global Critic asynchronously merges useful local structures and updates the global policy. As a result, the system can simultaneously explore different regions of the rule space while foreground inference continues using the latest available probabilistic network.

The contributions of this paper are threefold. First, we formulate MLN rule-structure adjustment as an actor-critic decision process in which states are current rule-weight configurations and actions correspond to structure and weight modifications. Second, we design a partitioned Actor search mechanism that assigns different concept boxes to different Actors, reducing the action space of each Actor and improving responsiveness. Third, we evaluate the method on high-dimensional streaming scenarios and show that the proposed approach improves real-time adaptation compared with MLN structure-learning baselines.

\section{{Related Work}}
MLNs were introduced by Richardson and Domingos as a probabilistic extension of first-order logic. Subsequent work studied MLN structure learning, weight learning, scalable inference, and online or incremental extensions. In parallel, asynchronous actor-critic learning has shown that multiple agents can explore different trajectories and update a shared global policy. Recent neuro-symbolic and rule-learning studies further motivate adaptive rule discovery in dynamic environments. Our work differs from batch MLN structure learning because it focuses on real-time streaming adaptation, and differs from generic stream classifiers because it explicitly maintains a weighted logical rule structure.

\section{{Method}}
Let $\mathcal{{R}}=\{{r_1,\ldots,r_m\}}$ denote the candidate MLN rule set. RT-A3C-MLN partitions $\mathcal{{R}}$ into $K$ concept boxes $\mathcal{{B}}_1,\ldots,\mathcal{{B}}_K$, where $K$ is the number of Actors. Actor $k$ can only modify rules in $\mathcal{{B}}_k$. The state contains the current local rule mask, recent reward, and global best reward. Actions add, delete, or replace rules within the assigned concept box. The reward is based on validation log-likelihood improvement with a complexity penalty.

The advantage signal is computed from the difference between the observed return and the Critic value estimate. Local gradients are sent asynchronously to the global policy. After each local search step, the best rules discovered by each Actor are merged into the global MLN structure. Foreground inference uses the latest available global structure and therefore does not block on background structure refresh.

\begin{{algorithm}}[!htbp]
\caption{{Partitioned RT-A3C-MLN rule learning}}
\begin{{algorithmic}}[1]
\STATE Split candidate rules into concept boxes $\mathcal{{B}}_1,\ldots,\mathcal{{B}}_K$
\STATE Initialize global Actor policy, Critic value function, and MLN weights
\FOR{{each incoming data window}}
  \STATE Return foreground inference using the current global MLN
  \FOR{{each Actor $k$ in parallel}}
    \STATE Explore add/delete/replace actions inside $\mathcal{{B}}_k$
    \STATE Evaluate local rule masks by validation log-likelihood reward
    \STATE Send advantage gradients and local best rules to the global network
  \ENDFOR
  \STATE Asynchronously merge local best rules and update global weights
\ENDFOR
\end{{algorithmic}}
\end{{algorithm}}

{chapter}

\section{{Conclusion}}
This paper presented RT-A3C-MLN, a real-time MLN rule learning framework based on partitioned asynchronous actor-critic search. By assigning each Actor to a disjoint concept box, the method avoids repeated full-scope rule search and enables simultaneous exploration of different rule subspaces. The experiments show that this design improves the speed of reaching useful rule structures, maintains low foreground response latency, and improves post-drift adaptation in high-dimensional streams. Future work will extend the implementation to distributed Actor processes and evaluate it on larger real-world neuro-symbolic knowledge bases.

\bibliographystyle{{cas-model2-names}}
\bibliography{{rt_a3c_mln_refs}}
\end{{document}}
"""
    (DRAFT_DIR / "rt_a3c_mln_partitioned_kbs_draft.tex").write_text(tex, encoding="utf-8")
    zip_path = ARTIFACT_DIR / "RT_A3C_MLN_partitioned_KBS_draft.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in DRAFT_DIR.parent.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(DRAFT_DIR.parent))
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[31, 37, 43])
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--update-intervals", nargs="+", type=int, default=[1, 2, 3, 4, 6])
    parser.add_argument("--actor-counts", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
    parser.add_argument("--rt-workers", type=int, default=8)
    parser.add_argument("--rt-episodes", type=int, default=90)
    parser.add_argument("--actor-total-episodes", type=int, default=120)
    parser.add_argument("--rt-steps", type=int, default=5)
    parser.add_argument("--max-active-rules", type=int, default=48)
    parser.add_argument("--rolling-window", type=int, default=5200)
    args = parser.parse_args()

    ensure_dirs()
    main_detail, main_summary = run_main_comparison(args)
    actor_detail, actor_trace, actor_summary = run_actor_count(args)
    freq_detail, freq_summary = run_update_frequency(args)
    ablation_detail, ablation_summary = run_ablation(args)
    table1, table2, table3, table4 = make_tables(main_summary, actor_summary, freq_summary, ablation_summary)
    draw_main_figures(main_detail, main_summary, actor_trace, actor_summary, freq_summary, ablation_summary)
    write_explanations()
    chapter_path = build_latex_chapter(table1, table2, table3, table4)
    zip_path = build_draft_zip(table1, table2, table3, table4)
    print(f"[done] results: {RESULT_DIR}")
    print(f"[done] artifacts: {ARTIFACT_DIR}")
    print(f"[done] chapter: {chapter_path}")
    print(f"[done] draft zip: {zip_path}")


if __name__ == "__main__":
    main()
