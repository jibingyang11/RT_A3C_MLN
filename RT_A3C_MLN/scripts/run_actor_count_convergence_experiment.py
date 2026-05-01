from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.models import RTA3CMLN, _safe_proba
from scripts.run_design_completion_experiments import load_base_config, make_strong_args
from scripts.run_strong_rule_drift_experiment import make_emerging_rule_stream


RESULT_DIR = ROOT / "results" / "actor_convergence_only"
ARTIFACT_DIR = ROOT / "paper_artifacts" / "actor_convergence_only"
FIG_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"


def prob_metrics(y_true, prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = (prob >= 0.5).astype(int)
    row = {
        "val_accuracy": float(accuracy_score(y_true, pred)),
        "val_f1": float(f1_score(y_true, pred, zero_division=0)),
        "val_log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
    }
    try:
        row["val_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        row["val_auc"] = np.nan
    return row


def threshold_time(trace: pd.DataFrame, frac: float):
    if trace.empty:
        return np.nan, np.nan
    first = float(trace["best_reward"].iloc[0])
    final = float(trace["best_reward"].max())
    if final <= first + 1e-12:
        row = trace.iloc[0]
        return float(row["time_sec"]), float(row["eval_count"])
    target = first + frac * (final - first)
    reached = trace[trace["best_reward"] >= target]
    row = reached.iloc[0] if len(reached) else trace.iloc[-1]
    return float(row["time_sec"]), float(row["eval_count"])


def time_to_reward(trace: pd.DataFrame, target: float):
    if trace.empty:
        return np.nan, 0.0
    reached = trace[trace["best_reward"] >= target]
    if len(reached):
        return float(reached.iloc[0]["time_sec"]), 1.0
    return float(trace["time_sec"].max()), 0.0


def reward_at_time(trace: pd.DataFrame, cutoff_sec: float):
    if trace.empty:
        return np.nan
    eligible = trace[trace["time_sec"] <= cutoff_sec]
    if len(eligible):
        return float(eligible["best_reward"].max())
    return float(trace["best_reward"].iloc[0])


def make_wide_sparse_rule_stream(cfg, seed: int):
    """Generate a large sparse rule-selection stress test.

    The informative predicates are deliberately spread across the full rule
    scope rather than concentrated in the first few features. This makes the
    task a genuine wide-search problem for the actor controller.
    """
    rng = np.random.default_rng(seed)
    n_features = int(cfg["strong_n_features"])
    train_size = int(cfg["strong_train_size"])
    val_size = int(cfg["strong_val_size"])
    density = float(cfg.get("wide_sparse_density", 0.075))
    signal_rules = int(cfg.get("wide_sparse_signal_rules", 28))
    interaction_rules = int(cfg.get("wide_sparse_interaction_rules", 8))
    noise_scale = float(cfg.get("wide_sparse_noise_scale", 0.20))

    x_train = rng.binomial(1, density, size=(train_size, n_features)).astype(np.uint8)
    x_val = rng.binomial(1, density, size=(val_size, n_features)).astype(np.uint8)
    signal_idx = np.sort(rng.choice(n_features, size=min(signal_rules, n_features), replace=False))
    weights = rng.normal(0.0, 1.0, size=len(signal_idx))
    weights += np.sign(weights + 1e-9) * rng.uniform(0.65, 1.35, size=len(signal_idx))
    interaction_len = min(interaction_rules * 2, len(signal_idx))
    interaction_len -= interaction_len % 2
    interaction_idx = signal_idx[:interaction_len]

    def make_y(x):
        score = x[:, signal_idx] @ weights
        for left, right in interaction_idx.reshape(-1, 2):
            score += 0.95 * (x[:, left] & x[:, right])
        distractors = rng.choice(n_features, size=min(40, n_features), replace=False)
        score += noise_scale * x[:, distractors].sum(axis=1)
        score -= np.median(score)
        score = np.clip(score, -50.0, 50.0)
        prob = 1.0 / (1.0 + np.exp(-score))
        return rng.binomial(1, prob).astype(int)

    y_train = make_y(x_train)
    y_val = make_y(x_val)
    return x_train, y_train, x_val, y_val, {"signal_idx": signal_idx.tolist()}


def write_markdown_table(df: pd.DataFrame, path: Path, title: str):
    columns = list(df.columns)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(
    cfg,
    seed: int,
    workers: int,
    total_episodes: int,
    total_structure_episodes: int,
    budget_mode: str,
    per_actor_override: int | None,
):
    if cfg.get("data_mode") == "wide_sparse":
        x_train, y_train, x_val, y_val, _ = make_wide_sparse_rule_stream(cfg, seed)
    else:
        args = make_strong_args(cfg, seed)
        x_train, y_train, x_val, y_val, _ = make_emerging_rule_stream(args)
    if budget_mode == "fixed_per_actor":
        per_actor_episodes = max(1, int(per_actor_override if per_actor_override is not None else total_episodes))
    else:
        per_actor_episodes = max(1, math.ceil(total_episodes / workers))
    model = RTA3CMLN(
        workers=workers,
        episodes=per_actor_episodes,
        n_steps=cfg["a3c_steps"],
        controller_rules=cfg["a3c_rules"],
        random_state=seed,
        ensemble_penalty=0.001,
        fast_mode=False,
        max_active_rules=cfg.get("a3c_max_active_rules"),
        total_episode_budget=total_episodes if budget_mode == "fixed_total" else None,
    )
    start = time.perf_counter()
    model.fit(x_train, y_train, x_val, y_val, rules=None)
    fit_sec = time.perf_counter() - start
    prob = _safe_proba(model.predict_proba(x_val))[:, 1]
    trace = pd.DataFrame(model.training_trace_)
    if trace.empty:
        trace = pd.DataFrame(
            [{"time_sec": fit_sec, "eval_count": 0.0, "best_reward": float(model.best_reward_)}]
        )
    trace.insert(0, "seed", seed)
    trace.insert(1, "workers", workers)
    trace.insert(2, "rule_scope", cfg["a3c_rules"])
    trace.insert(3, "per_actor_episodes", per_actor_episodes)
    trace.insert(4, "total_actor_episodes", total_episodes if budget_mode == "fixed_total" else per_actor_episodes * workers)

    t50, e50 = threshold_time(trace, 0.50)
    t80, e80 = threshold_time(trace, 0.80)
    t90, e90 = threshold_time(trace, 0.90)
    t95, e95 = threshold_time(trace, 0.95)
    selected = model.selected_rule_indices()
    row = {
        "seed": seed,
        "workers": workers,
        "budget_mode": budget_mode,
        "rule_scope": cfg["a3c_rules"],
        "per_actor_episodes": per_actor_episodes,
        "total_actor_episodes": total_episodes if budget_mode == "fixed_total" else per_actor_episodes * workers,
        "fit_time_sec": float(fit_sec),
        "unique_evals": float(getattr(model, "eval_count_", 0)),
        "evals_per_sec": float(getattr(model, "eval_count_", 0) / max(fit_sec, 1e-9)),
        "time_to_50_sec": t50,
        "time_to_80_sec": t80,
        "time_to_90_sec": t90,
        "time_to_95_sec": t95,
        "evals_to_50": e50,
        "evals_to_80": e80,
        "evals_to_90": e90,
        "evals_to_95": e95,
        "final_best_reward": float(model.best_reward_),
        "selected_rule_count": int(len(selected)),
        **prob_metrics(y_val, prob),
    }
    return row, trace


def add_common_target_metrics(detail: pd.DataFrame, trace: pd.DataFrame, target_frac: float = 0.90):
    detail = detail.copy()
    if detail.empty or trace.empty:
        detail["common90_target_reward"] = np.nan
        detail["time_to_common90_sec"] = np.nan
        detail["common90_reached"] = 0.0
        detail["reward_at_5_sec"] = np.nan
        detail["reward_at_10_sec"] = np.nan
        detail["reward_at_20_sec"] = np.nan
        return detail
    global_best_by_seed = detail.groupby("seed", observed=True)["final_best_reward"].max().to_dict()
    common_targets = []
    common_times = []
    reached_flags = []
    reward_at_5 = []
    reward_at_10 = []
    reward_at_20 = []
    for _, row in detail.iterrows():
        sub = trace[(trace["seed"] == row["seed"]) & (trace["workers"] == row["workers"])].sort_values("time_sec")
        if sub.empty:
            common_targets.append(np.nan)
            common_times.append(np.nan)
            reached_flags.append(0.0)
            reward_at_5.append(np.nan)
            reward_at_10.append(np.nan)
            reward_at_20.append(np.nan)
            continue
        first = float(sub["best_reward"].iloc[0])
        global_best = float(global_best_by_seed[row["seed"]])
        target = first + target_frac * (global_best - first)
        t_common, reached = time_to_reward(sub, target)
        common_targets.append(target)
        common_times.append(t_common)
        reached_flags.append(reached)
        reward_at_5.append(reward_at_time(sub, 5.0))
        reward_at_10.append(reward_at_time(sub, 10.0))
        reward_at_20.append(reward_at_time(sub, 20.0))
    detail["common90_target_reward"] = common_targets
    detail["time_to_common90_sec"] = common_times
    detail["common90_reached"] = reached_flags
    detail["reward_at_5_sec"] = reward_at_5
    detail["reward_at_10_sec"] = reward_at_10
    detail["reward_at_20_sec"] = reward_at_20
    return detail


def summarize(detail: pd.DataFrame):
    agg = detail.groupby("workers", observed=True).agg(
        budget_mode=("budget_mode", "first"),
        rule_scope=("rule_scope", "first"),
        per_actor_episodes=("per_actor_episodes", "first"),
        total_actor_episodes=("total_actor_episodes", "first"),
        fit_time_sec=("fit_time_sec", "mean"),
        fit_time_std=("fit_time_sec", "std"),
        unique_evals=("unique_evals", "mean"),
        evals_per_sec=("evals_per_sec", "mean"),
        time_to_80_sec=("time_to_80_sec", "mean"),
        time_to_90_sec=("time_to_90_sec", "mean"),
        time_to_95_sec=("time_to_95_sec", "mean"),
        time_to_common90_sec=("time_to_common90_sec", "mean"),
        common90_reached=("common90_reached", "mean"),
        reward_at_5_sec=("reward_at_5_sec", "mean"),
        reward_at_10_sec=("reward_at_10_sec", "mean"),
        reward_at_20_sec=("reward_at_20_sec", "mean"),
        evals_to_90=("evals_to_90", "mean"),
        final_best_reward=("final_best_reward", "mean"),
        val_f1=("val_f1", "mean"),
        val_log_loss=("val_log_loss", "mean"),
        val_auc=("val_auc", "mean"),
        selected_rule_count=("selected_rule_count", "mean"),
    ).reset_index()
    base_fit = float(agg.loc[agg["workers"] == agg["workers"].min(), "fit_time_sec"].iloc[0])
    base_t90 = float(agg.loc[agg["workers"] == agg["workers"].min(), "time_to_90_sec"].iloc[0])
    base_common90 = float(agg.loc[agg["workers"] == agg["workers"].min(), "time_to_common90_sec"].iloc[0])
    base_evals_per_sec = float(agg.loc[agg["workers"] == agg["workers"].min(), "evals_per_sec"].iloc[0])
    agg["fit_speedup_vs_1"] = base_fit / agg["fit_time_sec"]
    agg["t90_speedup_vs_1"] = base_t90 / agg["time_to_90_sec"].replace(0, np.nan)
    agg["common90_speedup_vs_1"] = base_common90 / agg["time_to_common90_sec"].replace(0, np.nan)
    agg["throughput_speedup_vs_1"] = agg["evals_per_sec"] / max(base_evals_per_sec, 1e-9)
    return agg


def make_table(summary: pd.DataFrame):
    rows = []
    for row in summary.sort_values("workers").to_dict("records"):
        rows.append(
            {
                "Actors": int(row["workers"]),
                "Rule scope": int(row["rule_scope"]),
                "Actor eps": int(row["per_actor_episodes"]),
                "Total eps": int(row["total_actor_episodes"]),
                "Fit sec": f"{row['fit_time_sec']:.3f}",
                "Own T90": f"{row['time_to_90_sec']:.3f}",
                "Common T90": f"{row['time_to_common90_sec']:.3f}",
                "Common speedup": f"{row['common90_speedup_vs_1']:.2f}",
                "Evals/s": f"{row['evals_per_sec']:.1f}",
                "Best reward": f"{row['final_best_reward']:.4f}",
                "Val F1": f"{row['val_f1']:.3f}",
                "Val LL": f"{row['val_log_loss']:.3f}",
                "Selected": f"{row['selected_rule_count']:.1f}",
            }
        )
    return pd.DataFrame(rows)


def normalized_trace(trace: pd.DataFrame):
    rows = []
    for (seed, workers), sub in trace.groupby(["seed", "workers"], observed=True):
        sub = sub.sort_values("time_sec")
        first = float(sub["best_reward"].iloc[0])
        final = float(sub["best_reward"].max())
        denom = final - first
        out = sub.copy()
        out["reward_progress"] = 1.0 if denom <= 1e-12 else (out["best_reward"] - first) / denom
        out["reward_progress"] = out["reward_progress"].clip(0.0, 1.0)
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else trace


def draw_figure(detail: pd.DataFrame, trace: pd.DataFrame, summary: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = sorted(summary["workers"].unique())
    cmap = plt.get_cmap("tab20")
    colors = {workers: cmap(idx % cmap.N) for idx, workers in enumerate(order)}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))

    max_time = float(trace["time_sec"].max()) if len(trace) else 1.0
    grid = np.linspace(0.0, max_time, 80)
    for workers in order:
        curves = []
        for _, sub in trace[trace["workers"] == workers].groupby("seed", observed=True):
            sub = sub.sort_values("time_sec")
            x = sub["time_sec"].to_numpy()
            y = sub["best_reward"].to_numpy()
            if len(x) == 1:
                curves.append(np.full_like(grid, y[0], dtype=float))
            else:
                curves.append(np.interp(grid, x, y, left=y[0], right=y[-1]))
        if curves:
            axes[0, 0].plot(grid, np.mean(curves, axis=0), color=colors[workers], lw=1.6, label=f"{workers} actors")
    target = float(detail["common90_target_reward"].mean()) if "common90_target_reward" in detail else np.nan
    if np.isfinite(target):
        axes[0, 0].axhline(target, color="#495057", ls="--", lw=1.0, label="common 90% target")
    axes[0, 0].set_title("A3C best-reward convergence")
    axes[0, 0].set_xlabel("Wall-clock seconds")
    axes[0, 0].set_ylabel("Best reward")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=3)

    s = summary.set_index("workers").loc[order].reset_index()
    x = np.arange(len(order))
    axes[0, 1].bar(x - 0.18, s["fit_time_sec"], width=0.36, color="#457b9d", label="Fit time")
    axes[0, 1].bar(x + 0.18, s["time_to_common90_sec"], width=0.36, color="#c1121f", label="Common T90")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(v) for v in order])
    axes[0, 1].set_title("Convergence time")
    axes[0, 1].set_xlabel("Number of actors")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(order, s["common90_speedup_vs_1"], marker="o", color="#2a9d8f", label="Common T90 speedup")
    axes[1, 0].plot(order, s["throughput_speedup_vs_1"], marker="s", color="#6d597a", label="Eval throughput speedup")
    axes[1, 0].axhline(1.0, color="#495057", ls="--", lw=1)
    axes[1, 0].set_title("Speedup relative to one actor")
    axes[1, 0].set_xlabel("Number of actors")
    axes[1, 0].set_ylabel("Speedup")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].bar(x - 0.18, s["val_f1"], width=0.36, color="#2a9d8f", label="Validation F1")
    axes[1, 1].bar(x + 0.18, s["val_log_loss"], width=0.36, color="#495057", label="Validation log-loss")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(v) for v in order])
    axes[1, 1].set_title("Converged validation quality")
    axes[1, 1].set_xlabel("Number of actors")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle("Parallel Actor Count vs. A3C Convergence Speed", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "Fig_Actor_Count_Convergence_Only.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig_Actor_Count_Convergence_Only.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[31, 37, 43])
    parser.add_argument("--actor-counts", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
    parser.add_argument("--rule-scope", type=int, default=720)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--total-episodes", type=int, default=120)
    parser.add_argument("--total-structure-episodes", type=int, default=24)
    parser.add_argument("--budget-mode", choices=["fixed_total", "fixed_per_actor"], default="fixed_total")
    parser.add_argument("--per-actor-episodes", type=int, default=None)
    parser.add_argument("--max-active-rules", type=int, default=None)
    parser.add_argument("--data-mode", choices=["strong_drift_pre", "wide_sparse"], default="strong_drift_pre")
    parser.add_argument("--wide-sparse-density", type=float, default=0.075)
    parser.add_argument("--wide-sparse-signal-rules", type=int, default=28)
    args = parser.parse_args()

    cfg = load_base_config()
    cfg["strong_n_features"] = max(int(args.rule_scope), cfg["strong_n_features"])
    cfg["a3c_rules"] = int(args.rule_scope)
    if args.train_size is not None:
        cfg["strong_train_size"] = int(args.train_size)
    if args.val_size is not None:
        cfg["strong_val_size"] = int(args.val_size)
    if args.steps is not None:
        cfg["a3c_steps"] = int(args.steps)
    if args.max_active_rules is not None:
        cfg["a3c_max_active_rules"] = int(args.max_active_rules)
    cfg["data_mode"] = args.data_mode
    cfg["wide_sparse_density"] = float(args.wide_sparse_density)
    cfg["wide_sparse_signal_rules"] = int(args.wide_sparse_signal_rules)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    traces = []
    for seed in args.seeds:
        for workers in args.actor_counts:
            if args.budget_mode == "fixed_per_actor":
                eps = max(1, int(args.per_actor_episodes if args.per_actor_episodes is not None else args.total_episodes))
            else:
                eps = max(1, math.ceil(args.total_episodes / workers))
            print(
                f"[actor-convergence] seed={seed} actors={workers} data={args.data_mode} "
                f"budget={args.budget_mode} rule_scope={args.rule_scope} eps/actor={eps}",
                flush=True,
            )
            row, trace = run_one(
                cfg,
                seed,
                workers,
                args.total_episodes,
                args.total_structure_episodes,
                args.budget_mode,
                args.per_actor_episodes,
            )
            rows.append(row)
            traces.append(trace)

    detail = pd.DataFrame(rows)
    trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    detail = add_common_target_metrics(detail, trace)
    summary = summarize(detail)
    table = make_table(summary)

    detail.to_csv(RESULT_DIR / "actor_count_convergence_detail.csv", index=False)
    trace.to_csv(RESULT_DIR / "actor_count_convergence_trace.csv", index=False)
    summary.to_csv(RESULT_DIR / "actor_count_convergence_summary.csv", index=False)
    table.to_csv(TABLE_DIR / "Table_Actor_Count_Convergence_Only.csv", index=False)
    write_markdown_table(table, TABLE_DIR / "Table_Actor_Count_Convergence_Only.md", "Actor count convergence-only experiment")
    draw_figure(detail, trace, summary)

    print("\n[Actor convergence table]")
    print(table.to_string(index=False))
    print(f"\n[done] results: {RESULT_DIR}")
    print(f"[done] artifacts: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
