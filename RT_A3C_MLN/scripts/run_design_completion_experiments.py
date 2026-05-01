from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.models import _safe_proba
from scripts.run_streaming_experiments import AdaptiveA3CStreamModel
from scripts.run_strong_rule_drift_experiment import EMERGING_RULES, make_emerging_rule_stream


RESULT_DIR = ROOT / "results" / "formal_all"
ARTIFACT_DIR = ROOT / "paper_artifacts"
FIG_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"
KBS_DIR = ROOT / "kbs_draft" / "els-cas-templates"
KBS_FIG_DIR = KBS_DIR / "figs"


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


def fmt(value, digits=3):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_ms(value, digits=3):
    return fmt(float(value) * 1000.0, digits)


def fmt_pct(value, digits=1):
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}"


def load_base_config():
    cfg_path = RESULT_DIR / "formal_config.json"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "seed": 31,
        "a3c_episodes": 5,
        "a3c_steps": 5,
        "a3c_rules": 84,
        "a3c_max_active_rules": 32,
        "a3c_rolling_window": 900,
        "a3c_calibration_window": 260,
        "a3c_structure_episodes": 3,
        "a3c_structure_tolerance": 0.015,
        "a3c_prior_shift_lr": 0.24,
        "a3c_prior_shift_clip": 1.7,
        "strong_n_features": 220,
        "strong_train_size": 1300,
        "strong_val_size": 320,
        "strong_chunks": 6,
        "strong_chunk_size": 260,
        "strong_drift_chunk": 4,
        "recovery_f1": 0.64,
    }


def make_strong_args(cfg, seed):
    return argparse.Namespace(
        seed=seed,
        n_features=cfg["strong_n_features"],
        train_size=cfg["strong_train_size"],
        val_size=cfg["strong_val_size"],
        chunks=cfg["strong_chunks"],
        chunk_size=cfg["strong_chunk_size"],
        drift_chunk=cfg["strong_drift_chunk"],
    )


def make_model(cfg, seed, workers, update_interval, episodes=None, structure_episodes=None):
    episodes = cfg["a3c_episodes"] if episodes is None else max(1, int(episodes))
    structure_episodes = (
        cfg["a3c_structure_episodes"] if structure_episodes is None else max(1, int(structure_episodes))
    )
    return AdaptiveA3CStreamModel(
        random_state=seed,
        workers=workers,
        episodes=episodes,
        steps=cfg["a3c_steps"],
        rules=cfg["a3c_rules"],
        ultra_fast_update=True,
        max_active_rules=cfg["a3c_max_active_rules"],
        rolling_window=cfg["a3c_rolling_window"],
        calibration_window=cfg["a3c_calibration_window"],
        refit_interval=update_interval,
        async_refit=True,
        prior_shift_lr=cfg["a3c_prior_shift_lr"],
        prior_shift_clip=cfg["a3c_prior_shift_clip"],
        structure_refresh=True,
        structure_refresh_interval=update_interval,
        structure_refresh_episodes=structure_episodes,
        structure_refresh_tolerance=cfg["a3c_structure_tolerance"],
    )


def run_one_stream(cfg, seed, workers, update_interval, experiment, episodes=None, structure_episodes=None):
    strong_args = make_strong_args(cfg, seed)
    x_train, y_train, x_val, y_val, chunks = make_emerging_rule_stream(strong_args)
    episodes = cfg["a3c_episodes"] if episodes is None else max(1, int(episodes))
    structure_episodes = (
        cfg["a3c_structure_episodes"] if structure_episodes is None else max(1, int(structure_episodes))
    )
    model = make_model(
        cfg,
        seed,
        workers=workers,
        update_interval=update_interval,
        episodes=episodes,
        structure_episodes=structure_episodes,
    )
    fit_start = time.perf_counter()
    model.fit(x_train, y_train, x_val, y_val)
    fit_time = time.perf_counter() - fit_start

    rows = []
    for item in chunks:
        x_chunk = item["x"]
        y_chunk = item["y"]
        start = time.perf_counter()
        prob = model.predict_proba(x_chunk)[:, 1]
        inference_time = time.perf_counter() - start
        update_start = time.perf_counter()
        foreground_update = model.update(x_chunk, y_chunk)
        update_wall = time.perf_counter() - update_start
        background_update = float(getattr(model, "last_background_update_sec", 0.0))
        active_rules = getattr(model, "active_rules", []) or []
        rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "workers": workers,
                "update_interval": update_interval,
                "fixed_rule_scope": cfg["a3c_rules"],
                "per_actor_episodes": episodes,
                "total_actor_episodes": episodes * workers,
                "per_actor_structure_episodes": structure_episodes,
                "total_structure_episodes": structure_episodes * workers,
                "n_step_return": cfg["a3c_steps"],
                "chunk": item["chunk"],
                "phase": item["phase"],
                "initial_fit_time_sec": fit_time,
                "foreground_update_time_sec": foreground_update,
                "background_update_time_sec": background_update,
                "update_wall_time_sec": update_wall,
                "blocking_cycle_time_sec": inference_time + foreground_update,
                "wall_cycle_time_sec": inference_time + update_wall,
                "meets_10ms_cycle": float(inference_time + foreground_update <= 0.010),
                "structure_refreshed": float(getattr(model, "last_structure_refreshed", 0.0)),
                "rule_additions": int(getattr(model, "last_rule_additions", 0)),
                "rule_deletions": int(getattr(model, "last_rule_deletions", 0)),
                "active_rule_count": int(getattr(model, "last_active_rule_count", len(active_rules))),
                "active_emerging_rule_count": int(len(set(active_rules).intersection(set(EMERGING_RULES)))),
                "positive_rate": float(item["positive_rate"]),
                **metrics_from_prob(y_chunk, prob, inference_time),
            }
        )
    return rows


def recovery_lag(valid, group_cols, drift_chunk, threshold):
    rows = []
    for key, sub in valid.groupby(group_cols, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        after = sub[(sub["chunk"] > drift_chunk) & (sub["f1"] >= threshold)]
        recovery_chunk = int(after["chunk"].min()) if len(after) else -1
        row = {col: value for col, value in zip(group_cols, key)}
        row["recovery_chunk"] = recovery_chunk
        row["recovery_lag_chunks"] = recovery_chunk - drift_chunk if recovery_chunk > 0 else -1
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_actor(detail, cfg):
    metrics = [
        "f1",
        "log_loss",
        "blocking_cycle_time_sec",
        "wall_cycle_time_sec",
        "foreground_update_time_sec",
        "background_update_time_sec",
        "meets_10ms_cycle",
        "structure_refreshed",
        "rule_additions",
        "rule_deletions",
        "active_emerging_rule_count",
        "initial_fit_time_sec",
    ]
    summary = detail.groupby("workers", observed=True)[metrics].mean().reset_index()
    scope = detail.groupby("workers", observed=True)["fixed_rule_scope"].first().reset_index(name="fixed_rule_scope")
    budget = detail.groupby("workers", observed=True).agg(
        per_actor_episodes=("per_actor_episodes", "first"),
        total_actor_episodes=("total_actor_episodes", "first"),
        per_actor_structure_episodes=("per_actor_structure_episodes", "first"),
        total_structure_episodes=("total_structure_episodes", "first"),
    ).reset_index()
    pre = detail[detail["phase"] == "pre"].groupby("workers", observed=True)["f1"].mean().reset_index(name="pre_f1")
    post = detail[detail["phase"] == "post"].groupby("workers", observed=True)["f1"].mean().reset_index(name="post_f1")
    final = detail.groupby(["workers", "seed"], observed=True).tail(1).groupby("workers", observed=True)["f1"].mean().reset_index(name="final_f1")
    rec = recovery_lag(detail, ["workers", "seed"], cfg["strong_drift_chunk"], cfg["recovery_f1"])
    rec_summary = rec.groupby("workers", observed=True).agg(
        recovery_rate=("recovery_chunk", lambda x: float(np.mean(np.asarray(x) > 0))),
        recovery_lag_chunks=("recovery_lag_chunks", lambda x: float(np.mean([v for v in x if v > 0])) if np.any(np.asarray(x) > 0) else np.nan),
    ).reset_index()
    return (
        summary.merge(scope, on="workers")
        .merge(budget, on="workers")
        .merge(pre, on="workers")
        .merge(post, on="workers")
        .merge(final, on="workers")
        .merge(rec_summary, on="workers")
    )


def summarize_update(detail, cfg):
    metrics = [
        "f1",
        "log_loss",
        "blocking_cycle_time_sec",
        "wall_cycle_time_sec",
        "foreground_update_time_sec",
        "background_update_time_sec",
        "meets_10ms_cycle",
        "structure_refreshed",
        "rule_additions",
        "rule_deletions",
        "active_emerging_rule_count",
        "initial_fit_time_sec",
    ]
    summary = detail.groupby("update_interval", observed=True)[metrics].mean().reset_index()
    scope = detail.groupby("update_interval", observed=True)["fixed_rule_scope"].first().reset_index(name="fixed_rule_scope")
    post = detail[detail["phase"] == "post"].groupby("update_interval", observed=True).agg(
        post_f1=("f1", "mean"),
        post_log_loss=("log_loss", "mean"),
    ).reset_index()
    rec = recovery_lag(detail, ["update_interval", "seed"], cfg["strong_drift_chunk"], cfg["recovery_f1"])
    rec_summary = rec.groupby("update_interval", observed=True).agg(
        recovery_rate=("recovery_chunk", lambda x: float(np.mean(np.asarray(x) > 0))),
        recovery_lag_chunks=("recovery_lag_chunks", lambda x: float(np.mean([v for v in x if v > 0])) if np.any(np.asarray(x) > 0) else np.nan),
    ).reset_index()
    return summary.merge(scope, on="update_interval").merge(post, on="update_interval").merge(rec_summary, on="update_interval")


def write_markdown_table(df, path, title):
    columns = list(df.columns)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_actor_table(summary):
    rows = []
    for row in summary.sort_values("workers").to_dict("records"):
        lag = row["recovery_lag_chunks"]
        rows.append(
            {
                "Actors": int(row["workers"]),
                "Rule scope": int(row["fixed_rule_scope"]),
                "Actor eps": int(row["per_actor_episodes"]),
                "Fit sec": fmt(row["initial_fit_time_sec"], 3),
                "Pre F1": fmt(row["pre_f1"], 3),
                "Post F1": fmt(row["post_f1"], 3),
                "Final F1": fmt(row["final_f1"], 3),
                "Recovery rate": fmt_pct(row["recovery_rate"], 1),
                "Recovery lag": "not recovered" if pd.isna(lag) else fmt(lag, 2),
                "Cycle ms": fmt_ms(row["blocking_cycle_time_sec"], 3),
                "Wall ms": fmt_ms(row["wall_cycle_time_sec"], 3),
                "Background ms": fmt_ms(row["background_update_time_sec"], 3),
                "Struct %": fmt_pct(row["structure_refreshed"], 1),
                "Emerging rules": fmt(row["active_emerging_rule_count"], 2),
            }
        )
    return pd.DataFrame(rows)


def make_update_table(summary):
    rows = []
    for row in summary.sort_values("update_interval").to_dict("records"):
        lag = row["recovery_lag_chunks"]
        rows.append(
            {
                "Refresh interval": int(row["update_interval"]),
                "Update frequency": f"1/{int(row['update_interval'])} chunks",
                "Rule scope": int(row["fixed_rule_scope"]),
                "Cycle ms": fmt_ms(row["blocking_cycle_time_sec"], 3),
                "Wall cycle ms": fmt_ms(row["wall_cycle_time_sec"], 3),
                "Background ms": fmt_ms(row["background_update_time_sec"], 3),
                "10ms %": fmt_pct(row["meets_10ms_cycle"], 1),
                "Post F1": fmt(row["post_f1"], 3),
                "Post Log-loss": fmt(row["post_log_loss"], 3),
                "Struct %": fmt_pct(row["structure_refreshed"], 1),
                "Emerging rules": fmt(row["active_emerging_rule_count"], 2),
                "Recovery lag": "not recovered" if pd.isna(lag) else fmt(lag, 2),
            }
        )
    return pd.DataFrame(rows)


def draw_actor_figure(detail, summary):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = sorted(summary["workers"].unique())
    cmap = plt.get_cmap("tab20")
    colors = {workers: cmap(idx % cmap.N) for idx, workers in enumerate(order)}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))

    for workers in order:
        sub = detail[detail["workers"] == workers].groupby("chunk", observed=True)["f1"].mean().reset_index()
        axes[0, 0].plot(sub["chunk"], sub["f1"], marker="o", label=f"{workers} actors", color=colors[workers], linewidth=1.5)
    axes[0, 0].axvline(4, color="#495057", ls="--", lw=1)
    axes[0, 0].set_title("F1 convergence after emerging-rule drift")
    axes[0, 0].set_xlabel("Stream chunk")
    axes[0, 0].set_ylabel("F1")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=3)

    x = np.arange(len(order))
    s = summary.set_index("workers").loc[order].reset_index()
    axes[0, 1].bar(x, s["initial_fit_time_sec"], width=0.55, color="#457b9d", label="Initial fit")
    ax_rate = axes[0, 1].twinx()
    ax_rate.plot(x, s["recovery_rate"] * 100, color="#c1121f", marker="o", lw=2, label="Recovery rate")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(v) for v in order])
    axes[0, 1].set_title("Convergence cost and recovery success")
    axes[0, 1].set_xlabel("Number of actors")
    axes[0, 1].set_ylabel("Initial fit (s)")
    ax_rate.set_ylabel("Recovery rate (%)")
    ax_rate.set_ylim(0, 100)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    handles2, labels2 = ax_rate.get_legend_handles_labels()
    axes[0, 1].legend(handles + handles2, labels + labels2, frameon=False, fontsize=8, loc="upper left")

    axes[1, 0].bar(x, s["blocking_cycle_time_sec"] * 1000, color="#2a9d8f", label="Foreground")
    ax_wall = axes[1, 0].twinx()
    ax_wall.plot(x, s["wall_cycle_time_sec"] * 1000, color="#457b9d", marker="s", lw=2, label="Measured wall")
    axes[1, 0].axhline(10, color="#c1121f", ls="--", lw=1)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([str(v) for v in order])
    axes[1, 0].set_title("Foreground cycle vs. measured wall cycle")
    axes[1, 0].set_ylabel("Foreground ms")
    ax_wall.set_ylabel("Wall ms")
    axes[1, 0].set_xlabel("Number of actors")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    handles2, labels2 = ax_wall.get_legend_handles_labels()
    axes[1, 0].legend(handles + handles2, labels + labels2, frameon=False, fontsize=8, loc="upper left")

    axes[1, 1].bar(x - 0.18, s["structure_refreshed"] * 100, width=0.36, color="#457b9d", label="Struct %")
    axes[1, 1].bar(x + 0.18, s["active_emerging_rule_count"], width=0.36, color="#2a9d8f", label="Emerging rules")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(v) for v in order])
    axes[1, 1].set_title("Rule-structure adaptation evidence")
    axes[1, 1].set_xlabel("Number of actors")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle("Parallel actor count sensitivity", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "Fig5_Actor_Count_Convergence.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig5_Actor_Count_Convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_update_figure(detail, summary):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = sorted(summary["update_interval"].unique())
    x = np.arange(len(order))
    s = summary.set_index("update_interval").loc[order].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))

    axes[0, 0].plot(order, s["blocking_cycle_time_sec"] * 1000, marker="o", color="#c1121f", label="Foreground")
    axes[0, 0].plot(order, s["wall_cycle_time_sec"] * 1000, marker="s", color="#457b9d", label="Measured wall")
    axes[0, 0].axhline(10, color="#495057", ls="--", lw=1)
    axes[0, 0].set_title("Latency vs. structure refresh interval")
    axes[0, 0].set_xlabel("Refresh interval (chunks)")
    axes[0, 0].set_ylabel("ms")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].bar(x, s["background_update_time_sec"] * 1000, color="#6d597a")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(v) for v in order])
    axes[0, 1].set_title("Background structure-search work")
    axes[0, 1].set_ylabel("ms")
    axes[0, 1].set_xlabel("Refresh interval (chunks)")

    axes[1, 0].bar(x - 0.18, s["post_f1"], width=0.36, color="#2a9d8f", label="Post F1")
    axes[1, 0].bar(x + 0.18, s["post_log_loss"], width=0.36, color="#495057", label="Post log-loss")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([str(v) for v in order])
    axes[1, 0].set_title("Post-drift inference quality")
    axes[1, 0].set_xlabel("Refresh interval (chunks)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].bar(x - 0.18, s["structure_refreshed"] * 100, width=0.36, color="#457b9d", label="Struct %")
    axes[1, 1].bar(x + 0.18, s["active_emerging_rule_count"], width=0.36, color="#2a9d8f", label="Emerging rules")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(v) for v in order])
    axes[1, 1].set_title("Adaptation evidence vs. update frequency")
    axes[1, 1].set_xlabel("Refresh interval (chunks)")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle("Structure refresh frequency sensitivity", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "Fig6_Update_Frequency_Latency.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig6_Update_Frequency_Latency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def sync_figures_to_kbs():
    KBS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["Fig5_Actor_Count_Convergence.pdf", "Fig6_Update_Frequency_Latency.pdf"]:
        src = FIG_DIR / name
        if src.exists():
            (KBS_FIG_DIR / name).write_bytes(src.read_bytes())


def run_actor_sweep(cfg, seeds, actor_counts):
    rows = []
    for seed in seeds:
        for workers in actor_counts:
            episodes = cfg.get("actor_sweep_total_episodes")
            structure_episodes = cfg.get("actor_sweep_total_structure_episodes")
            per_actor_episodes = (
                max(1, math.ceil(episodes / workers)) if episodes is not None else cfg["a3c_episodes"]
            )
            per_actor_structure_episodes = (
                max(1, math.ceil(structure_episodes / workers))
                if structure_episodes is not None
                else cfg["a3c_structure_episodes"]
            )
            print(
                f"[actor-sweep] seed={seed} actors={workers} "
                f"rule_scope={cfg['a3c_rules']} eps/actor={per_actor_episodes} "
                f"struct_eps/actor={per_actor_structure_episodes}",
                flush=True,
            )
            rows.extend(
                run_one_stream(
                    cfg,
                    seed,
                    workers=workers,
                    update_interval=1,
                    experiment="actor_count_sweep",
                    episodes=per_actor_episodes,
                    structure_episodes=per_actor_structure_episodes,
                )
            )
    detail = pd.DataFrame(rows)
    summary = summarize_actor(detail, cfg)
    return detail, summary


def run_update_sweep(cfg, seeds, intervals, workers):
    rows = []
    episodes = cfg.get("actor_sweep_total_episodes")
    structure_episodes = cfg.get("actor_sweep_total_structure_episodes")
    per_actor_episodes = max(1, math.ceil(episodes / workers)) if episodes is not None else cfg["a3c_episodes"]
    per_actor_structure_episodes = (
        max(1, math.ceil(structure_episodes / workers))
        if structure_episodes is not None
        else cfg["a3c_structure_episodes"]
    )
    for seed in seeds:
        for interval in intervals:
            print(
                f"[update-sweep] seed={seed} interval={interval} rule_scope={cfg['a3c_rules']} "
                f"eps/actor={per_actor_episodes} struct_eps/actor={per_actor_structure_episodes}",
                flush=True,
            )
            rows.extend(
                run_one_stream(
                    cfg,
                    seed,
                    workers=workers,
                    update_interval=interval,
                    experiment="update_frequency_sweep",
                    episodes=per_actor_episodes,
                    structure_episodes=per_actor_structure_episodes,
                )
            )
    detail = pd.DataFrame(rows)
    summary = summarize_update(detail, cfg)
    return detail, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[31, 37, 43])
    parser.add_argument("--actor-counts", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
    parser.add_argument("--update-intervals", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--workers-for-update", type=int, default=4)
    parser.add_argument(
        "--fixed-rule-scope",
        default="max",
        help="Fixed candidate MLN rule scope for all conditions. Use 'max' to match the full strong-drift feature/rule dimension.",
    )
    parser.add_argument(
        "--large-rule-scope",
        type=int,
        default=None,
        help="Expand the synthetic strong-drift feature/rule dimension before applying --fixed-rule-scope max.",
    )
    parser.add_argument(
        "--actor-total-episodes",
        type=int,
        default=None,
        help="Fixed total A3C exploration episode budget for actor-count scalability sweeps.",
    )
    parser.add_argument(
        "--actor-total-structure-episodes",
        type=int,
        default=None,
        help="Fixed total structure-refresh episode budget for actor-count scalability sweeps.",
    )
    args = parser.parse_args()

    cfg = load_base_config()
    if args.large_rule_scope is not None:
        cfg["strong_n_features"] = max(cfg["strong_n_features"], int(args.large_rule_scope))
    if str(args.fixed_rule_scope).lower() == "max":
        cfg["a3c_rules"] = cfg["strong_n_features"]
    elif args.fixed_rule_scope is not None:
        cfg["a3c_rules"] = int(args.fixed_rule_scope)
    if args.actor_total_episodes is not None:
        cfg["actor_sweep_total_episodes"] = int(args.actor_total_episodes)
    if args.actor_total_structure_episodes is not None:
        cfg["actor_sweep_total_structure_episodes"] = int(args.actor_total_structure_episodes)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    actor_detail, actor_summary = run_actor_sweep(cfg, args.seeds, args.actor_counts)
    update_detail, update_summary = run_update_sweep(cfg, args.seeds, args.update_intervals, args.workers_for_update)

    actor_detail.to_csv(RESULT_DIR / "actor_count_sensitivity_detail.csv", index=False)
    actor_summary.to_csv(RESULT_DIR / "actor_count_sensitivity_summary.csv", index=False)
    update_detail.to_csv(RESULT_DIR / "update_frequency_sensitivity_detail.csv", index=False)
    update_summary.to_csv(RESULT_DIR / "update_frequency_sensitivity_summary.csv", index=False)

    actor_table = make_actor_table(actor_summary)
    update_table = make_update_table(update_summary)
    actor_table.to_csv(TABLE_DIR / "Table3_Actor_Count_Sensitivity.csv", index=False)
    update_table.to_csv(TABLE_DIR / "Table4_Update_Frequency_Latency.csv", index=False)
    write_markdown_table(actor_table, TABLE_DIR / "Table3_Actor_Count_Sensitivity.md", "Actor count sensitivity")
    write_markdown_table(update_table, TABLE_DIR / "Table4_Update_Frequency_Latency.md", "Update frequency and latency sensitivity")

    draw_actor_figure(actor_detail, actor_summary)
    draw_update_figure(update_detail, update_summary)
    sync_figures_to_kbs()

    print("\n[Table3]")
    print(actor_table.to_string(index=False))
    print("\n[Table4]")
    print(update_table.to_string(index=False))
    print(f"\n[done] results: {RESULT_DIR}")
    print(f"[done] figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
