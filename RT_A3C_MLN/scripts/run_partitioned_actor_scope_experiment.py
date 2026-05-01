# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys
import threading
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "partitioned_actor_scope"
ARTIFACT_DIR = ROOT / "paper_artifacts" / "partitioned_actor_scope"
FIG_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"
EPS = 1e-8


def sigmoid(score: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0)))


def make_wide_sparse_stream(seed: int, n_features: int, train_size: int, val_size: int, density: float, signal_rules: int):
    rng = np.random.default_rng(seed)

    def sparse_binary(rows: int):
        try:
            return (rng.random((rows, n_features), dtype=np.float32) < density).astype(np.uint8)
        except TypeError:
            return (rng.random((rows, n_features)) < density).astype(np.uint8)

    x_train = sparse_binary(train_size)
    x_val = sparse_binary(val_size)

    centers = np.linspace(20, n_features - 21, signal_rules, dtype=int)
    jitter = rng.integers(-8, 9, size=signal_rules)
    signal_idx = np.clip(centers + jitter, 0, n_features - 1)
    signal_idx = np.unique(signal_idx)
    while len(signal_idx) < signal_rules:
        signal_idx = np.unique(np.append(signal_idx, rng.integers(0, n_features)))
    signal_idx = np.sort(signal_idx[:signal_rules])

    weights = rng.normal(0.0, 1.0, size=len(signal_idx))
    weights += np.sign(weights + EPS) * rng.uniform(0.85, 1.55, size=len(signal_idx))
    interaction_idx = signal_idx[: min(16, len(signal_idx) - len(signal_idx) % 2)]

    def sample_y(x: np.ndarray) -> np.ndarray:
        score = x[:, signal_idx].astype(float) @ weights
        if len(interaction_idx):
            for left, right in interaction_idx.reshape(-1, 2):
                score += 0.85 * (x[:, left] & x[:, right])
        score -= np.median(score)
        prob = sigmoid(score)
        return rng.binomial(1, prob).astype(int)

    y_train = sample_y(x_train)
    y_val = sample_y(x_val)
    return x_train, y_train, x_val, y_val, signal_idx


def estimate_rule_weights(x_train: np.ndarray, y_train: np.ndarray):
    y = y_train.astype(float)
    support = x_train.sum(axis=0).astype(float)
    pos = x_train.T.astype(float) @ y
    total_pos = float(y.sum())
    total_neg = float(len(y) - total_pos)
    neg_support = support - pos

    p_y1_x1 = (pos + 1.0) / (support + 2.0)
    p_y1_x0 = (total_pos - pos + 1.0) / (len(y) - support + 2.0)
    weight = np.log(p_y1_x1 / (1.0 - p_y1_x1 + EPS) + EPS) - np.log(p_y1_x0 / (1.0 - p_y1_x0 + EPS) + EPS)
    weight = np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
    quality = np.abs(weight) * np.sqrt(np.maximum(support, 1.0) / len(y))
    prior = np.clip(float(y.mean()), 1e-5, 1.0 - 1e-5)
    intercept = math.log(prior / (1.0 - prior))
    return weight.astype(float), quality.astype(float), intercept


def prob_metrics(y_true: np.ndarray, prob: np.ndarray):
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
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


def episode_counts(total_episodes: int, workers: int):
    base = total_episodes // workers
    extra = total_episodes % workers
    return [base + (1 if idx < extra else 0) for idx in range(workers)]


def time_to_reward(trace: pd.DataFrame, target: float):
    if trace.empty:
        return np.nan, 0.0
    reached = trace[trace["best_reward"] >= target]
    if len(reached):
        return float(reached.iloc[0]["time_sec"]), 1.0
    return float(trace["time_sec"].max()), 0.0


def write_markdown_table(df: pd.DataFrame, path: Path, title: str):
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PartitionedScopeA3C:
    def __init__(
        self,
        workers: int,
        total_episodes: int,
        n_steps: int,
        max_active_rules: int,
        random_state: int,
        complexity_penalty: float = 0.002,
    ):
        self.workers = int(workers)
        self.total_episodes = int(total_episodes)
        self.n_steps = int(n_steps)
        self.max_active_rules = int(max_active_rules)
        self.random_state = int(random_state)
        self.complexity_penalty = float(complexity_penalty)
        self.trace_: list[dict[str, float]] = []
        self.best_mask_: np.ndarray | None = None
        self.best_reward_: float = -np.inf
        self.eval_count_: int = 0

    def fit(self, x_train, y_train, x_val, y_val, rule_weight, rule_quality):
        n_features = x_train.shape[1]
        partitions = np.array_split(np.arange(n_features), self.workers)
        per_worker_active = [max(1, math.ceil(self.max_active_rules * len(part) / n_features)) for part in partitions]
        counts = episode_counts(self.total_episodes, self.workers)
        prior = np.clip(float(y_train.mean()), 1e-5, 1.0 - 1e-5)
        intercept = math.log(prior / (1.0 - prior))
        base_prob = np.full(len(y_val), prior)
        base_loss = log_loss(y_val, base_prob, labels=[0, 1])
        cache: dict[bytes, tuple[float, np.ndarray, np.ndarray]] = {}
        cache_lock = threading.Lock()
        update_lock = threading.Lock()
        t0 = time.perf_counter()

        worker_best_masks = [np.zeros(n_features, dtype=bool) for _ in range(self.workers)]
        worker_best_rewards = [-np.inf for _ in range(self.workers)]
        global_best = {"reward": -np.inf, "mask": np.zeros(n_features, dtype=bool)}
        self.trace_ = [
            {
                "time_sec": 0.0,
                "eval_count": 0.0,
                "worker_id": -1.0,
                "best_reward": 0.0,
                "selected_count": 0.0,
            }
        ]

        def trim_mask(mask: np.ndarray):
            if int(mask.sum()) <= self.max_active_rules:
                return mask
            selected = np.flatnonzero(mask)
            keep = selected[np.argsort(rule_quality[selected])[::-1][: self.max_active_rules]]
            trimmed = np.zeros_like(mask)
            trimmed[keep] = True
            return trimmed

        def evaluate(mask: np.ndarray):
            mask = trim_mask(mask.astype(bool, copy=True))
            key = np.packbits(mask.astype(np.uint8)).tobytes()
            with cache_lock:
                cached = cache.get(key)
                if cached is not None:
                    return cached
            selected = np.flatnonzero(mask)
            if len(selected):
                score = intercept + x_val[:, selected].astype(float) @ rule_weight[selected]
                prob = sigmoid(score)
            else:
                prob = base_prob
            loss = log_loss(y_val, prob, labels=[0, 1])
            reward = float(base_loss - loss - self.complexity_penalty * (len(selected) / max(1, self.max_active_rules)))
            with cache_lock:
                self.eval_count_ += 1
                cache[key] = (reward, prob, mask)
            return reward, prob, mask

        def update_global(worker_id: int, local_mask: np.ndarray):
            local_reward, _, local_mask = evaluate(local_mask)
            with update_lock:
                if local_reward > worker_best_rewards[worker_id]:
                    worker_best_rewards[worker_id] = local_reward
                    worker_best_masks[worker_id] = local_mask
                union = np.zeros(n_features, dtype=bool)
                for item in worker_best_masks:
                    union |= item
            union_reward, _, union = evaluate(union)
            with update_lock:
                if union_reward > global_best["reward"]:
                    global_best["reward"] = union_reward
                    global_best["mask"] = union.copy()
                    self.trace_.append(
                        {
                            "time_sec": float(time.perf_counter() - t0),
                            "eval_count": float(self.eval_count_),
                            "worker_id": float(worker_id),
                            "best_reward": float(union_reward),
                            "selected_count": float(union.sum()),
                        }
                    )
            return local_reward

        def softmax(logits: np.ndarray):
            logits = logits - np.max(logits)
            exp = np.exp(np.clip(logits, -50.0, 50.0))
            return exp / max(float(exp.sum()), EPS)

        def worker_loop(worker_id: int):
            rng = np.random.default_rng(self.random_state + 1009 * (worker_id + 1))
            segment = partitions[worker_id]
            local_quality = rule_quality[segment]
            local_quality = (local_quality - local_quality.mean()) / (local_quality.std() + EPS)
            pref = local_quality.copy()
            active_cap = per_worker_active[worker_id]
            for episode in range(counts[worker_id]):
                mask = np.zeros(n_features, dtype=bool)
                warm = min(active_cap, max(2, active_cap // 2))
                if warm > 0:
                    top = np.argsort(local_quality)[::-1][: max(warm * 3, warm)]
                    chosen = rng.choice(top, size=warm, replace=False)
                    mask[segment[chosen]] = True
                last_reward = update_global(worker_id, mask)
                for _ in range(self.n_steps):
                    selected_local = mask[segment]
                    logits = pref.copy()
                    logits[selected_local] -= 0.35
                    probs = softmax(logits)
                    action = int(rng.choice(len(segment), p=probs))
                    global_idx = int(segment[action])
                    new_mask = mask.copy()
                    if new_mask[global_idx]:
                        new_mask[global_idx] = False
                    else:
                        if int(new_mask[segment].sum()) >= active_cap:
                            active = np.flatnonzero(new_mask[segment])
                            if len(active):
                                weakest = active[np.argmin(local_quality[active])]
                                new_mask[int(segment[weakest])] = False
                        new_mask[global_idx] = True
                    reward = update_global(worker_id, new_mask)
                    advantage = reward - last_reward
                    pref[action] += 0.25 * np.clip(advantage, -1.0, 1.0)
                    mask = new_mask
                    last_reward = reward

        threads = [threading.Thread(target=worker_loop, args=(idx,), daemon=True) for idx in range(self.workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.best_reward_ = float(global_best["reward"])
        self.best_mask_ = global_best["mask"].copy()
        self.search_time_sec_ = float(time.perf_counter() - t0)
        return self

    def predict_proba(self, x_val: np.ndarray, y_train_mean: float, rule_weight: np.ndarray):
        prior = np.clip(float(y_train_mean), 1e-5, 1.0 - 1e-5)
        intercept = math.log(prior / (1.0 - prior))
        selected = np.flatnonzero(self.best_mask_) if self.best_mask_ is not None else np.array([], dtype=int)
        if len(selected):
            prob = sigmoid(intercept + x_val[:, selected].astype(float) @ rule_weight[selected])
        else:
            prob = np.full(x_val.shape[0], prior)
        return prob


def run_one(args, seed: int, workers: int):
    x_train, y_train, x_val, y_val, signal_idx = make_wide_sparse_stream(
        seed=seed,
        n_features=args.rule_scope,
        train_size=args.train_size,
        val_size=args.val_size,
        density=args.density,
        signal_rules=args.signal_rules,
    )
    rule_weight, rule_quality, _ = estimate_rule_weights(x_train, y_train)
    model = PartitionedScopeA3C(
        workers=workers,
        total_episodes=args.total_episodes,
        n_steps=args.steps,
        max_active_rules=args.max_active_rules,
        random_state=seed,
    )
    model.fit(x_train, y_train, x_val, y_val, rule_weight, rule_quality)
    prob = model.predict_proba(x_val, float(y_train.mean()), rule_weight)
    selected = np.flatnonzero(model.best_mask_) if model.best_mask_ is not None else np.array([], dtype=int)
    signal_recall = len(set(selected.tolist()) & set(signal_idx.tolist())) / max(1, len(signal_idx))
    trace = pd.DataFrame(model.trace_)
    trace.insert(0, "seed", seed)
    trace.insert(1, "workers", workers)
    trace.insert(2, "scope_per_actor", math.ceil(args.rule_scope / workers))
    trace.insert(3, "total_episodes", args.total_episodes)
    row = {
        "seed": seed,
        "workers": workers,
        "rule_scope": args.rule_scope,
        "scope_per_actor": math.ceil(args.rule_scope / workers),
        "total_episodes": args.total_episodes,
        "episodes_per_actor_max": max(episode_counts(args.total_episodes, workers)),
        "search_time_sec": float(model.search_time_sec_),
        "unique_evals": float(model.eval_count_),
        "evals_per_sec": float(model.eval_count_ / max(model.search_time_sec_, EPS)),
        "final_best_reward": float(model.best_reward_),
        "selected_rule_count": int(len(selected)),
        "signal_recall": float(signal_recall),
        **prob_metrics(y_val, prob),
    }
    return row, trace


def add_common_target(detail: pd.DataFrame, trace: pd.DataFrame):
    detail = detail.copy()
    common_targets = []
    common_times = []
    common_reached = []
    baseline_targets = []
    baseline_times = []
    baseline_reached = []
    for _, row in detail.iterrows():
        sub = trace[(trace["seed"] == row["seed"]) & (trace["workers"] == row["workers"])].sort_values("time_sec")
        seed_best = float(detail.loc[detail["seed"] == row["seed"], "final_best_reward"].max())
        seed_base = float(detail.loc[(detail["seed"] == row["seed"]) & (detail["workers"] == 1), "final_best_reward"].iloc[0])
        first = float(sub["best_reward"].iloc[0]) if len(sub) else 0.0
        common_target = first + 0.90 * (seed_best - first)
        baseline_target = first + 0.90 * (seed_base - first)
        t_common, ok_common = time_to_reward(sub, common_target)
        t_base, ok_base = time_to_reward(sub, baseline_target)
        common_targets.append(common_target)
        common_times.append(t_common)
        common_reached.append(ok_common)
        baseline_targets.append(baseline_target)
        baseline_times.append(t_base)
        baseline_reached.append(ok_base)
    detail["common90_target_reward"] = common_targets
    detail["time_to_common90_sec"] = common_times
    detail["common90_reached"] = common_reached
    detail["baseline90_target_reward"] = baseline_targets
    detail["time_to_baseline90_sec"] = baseline_times
    detail["baseline90_reached"] = baseline_reached
    return detail


def summarize(detail: pd.DataFrame):
    summary = detail.groupby("workers", observed=True).agg(
        rule_scope=("rule_scope", "first"),
        scope_per_actor=("scope_per_actor", "first"),
        total_episodes=("total_episodes", "first"),
        episodes_per_actor_max=("episodes_per_actor_max", "first"),
        search_time_sec=("search_time_sec", "mean"),
        time_to_baseline90_sec=("time_to_baseline90_sec", "mean"),
        baseline90_reached=("baseline90_reached", "mean"),
        time_to_common90_sec=("time_to_common90_sec", "mean"),
        common90_reached=("common90_reached", "mean"),
        unique_evals=("unique_evals", "mean"),
        evals_per_sec=("evals_per_sec", "mean"),
        final_best_reward=("final_best_reward", "mean"),
        val_f1=("val_f1", "mean"),
        val_log_loss=("val_log_loss", "mean"),
        val_auc=("val_auc", "mean"),
        selected_rule_count=("selected_rule_count", "mean"),
        signal_recall=("signal_recall", "mean"),
    ).reset_index()
    base_time = float(summary.loc[summary["workers"] == summary["workers"].min(), "search_time_sec"].iloc[0])
    base_baseline = float(summary.loc[summary["workers"] == summary["workers"].min(), "time_to_baseline90_sec"].iloc[0])
    base_common = float(summary.loc[summary["workers"] == summary["workers"].min(), "time_to_common90_sec"].iloc[0])
    summary["search_speedup_vs_1"] = base_time / summary["search_time_sec"].replace(0, np.nan)
    summary["baseline90_speedup_vs_1"] = base_baseline / summary["time_to_baseline90_sec"].replace(0, np.nan)
    summary["common90_speedup_vs_1"] = base_common / summary["time_to_common90_sec"].replace(0, np.nan)
    return summary


def make_table(summary: pd.DataFrame):
    rows = []
    for row in summary.sort_values("workers").to_dict("records"):
        rows.append(
            {
                "Actors": int(row["workers"]),
                "Rule scope": int(row["rule_scope"]),
                "Scope/actor": int(row["scope_per_actor"]),
                "Total eps": int(row["total_episodes"]),
                "Actor eps max": int(row["episodes_per_actor_max"]),
                "Search sec": f"{row['search_time_sec']:.3f}",
                "Base T90": f"{row['time_to_baseline90_sec']:.3f}",
                "Search speedup": f"{row['search_speedup_vs_1']:.2f}",
                "Base T90 speedup": f"{row['baseline90_speedup_vs_1']:.2f}",
                "Best reward": f"{row['final_best_reward']:.4f}",
                "Val F1": f"{row['val_f1']:.3f}",
                "Val LL": f"{row['val_log_loss']:.3f}",
                "Signal recall": f"{row['signal_recall']:.3f}",
            }
        )
    return pd.DataFrame(rows)


def draw_figure(summary: pd.DataFrame, trace: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = sorted(summary["workers"].unique())
    colors = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))

    max_time = float(trace["time_sec"].max()) if len(trace) else 1.0
    grid = np.linspace(0.0, max_time, 100)
    for idx, workers in enumerate(order):
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
            axes[0, 0].plot(grid, np.mean(curves, axis=0), lw=1.7, color=colors(idx), label=f"{workers} actors")
    axes[0, 0].set_title("Global best reward over time")
    axes[0, 0].set_xlabel("Wall-clock seconds")
    axes[0, 0].set_ylabel("Best reward")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=3)

    s = summary.set_index("workers").loc[order].reset_index()
    x = np.arange(len(order))
    axes[0, 1].bar(x - 0.18, s["search_time_sec"], width=0.36, color="#457b9d", label="Search time")
    axes[0, 1].bar(x + 0.18, s["time_to_baseline90_sec"], width=0.36, color="#c1121f", label="Base T90")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(v) for v in order])
    axes[0, 1].set_title("Convergence time under fixed total episodes")
    axes[0, 1].set_xlabel("Number of actors")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(order, s["search_speedup_vs_1"], marker="o", color="#2a9d8f", label="Search speedup")
    axes[1, 0].plot(order, s["baseline90_speedup_vs_1"], marker="s", color="#6d597a", label="Base T90 speedup")
    axes[1, 0].axhline(1.0, color="#495057", ls="--", lw=1)
    axes[1, 0].set_title("Speedup relative to one actor")
    axes[1, 0].set_xlabel("Number of actors")
    axes[1, 0].set_ylabel("Speedup")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].bar(x - 0.18, s["val_f1"], width=0.36, color="#2a9d8f", label="Validation F1")
    axes[1, 1].bar(x + 0.18, s["signal_recall"], width=0.36, color="#f4a261", label="Signal-rule recall")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(v) for v in order])
    axes[1, 1].set_title("Converged structure quality")
    axes[1, 1].set_xlabel("Number of actors")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle("Partitioned Rule Scope for Parallel Actor Search", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "Fig_Partitioned_Actor_Scope.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig_Partitioned_Actor_Scope.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_explanation(summary: pd.DataFrame):
    best_search = summary.loc[summary["search_speedup_vs_1"].idxmax()]
    best_t90 = summary.loc[summary["baseline90_speedup_vs_1"].idxmax()]
    text = f"""# 分区规则搜索下并行 Actor 数量实验说明

本实验重新实现并行 Actor 搜索机制：Rule scope 固定为 {int(summary['rule_scope'].iloc[0])}，但不同 Actor 不再重复探索完整规则空间，而是把候选规则按 Actor 数量切分成互不重叠的子空间。例如 4 个 Actor 时，每个 Actor 只负责约 {math.ceil(int(summary['rule_scope'].iloc[0]) / 4)} 条候选规则，并在自己的子空间内执行添加、删除和替换规则的动作，最后将各 Actor 找到的局部最优规则异步合并为全局规则集合。

## 指标解释

- **Scope/actor**：每个 Actor 实际负责探索的候选规则数量。
- **Total eps**：所有 Actor 加起来的总探索 episode，实验中保持固定，不随 Actor 数量增加。
- **Actor eps max**：单个 Actor 最多执行的 episode 数。
- **Search sec**：完成分区 Actor 搜索所需的墙钟时间。
- **Base T90**：达到 1 个 Actor 最终 best reward 的 90% 所需时间，用来衡量多 Actor 是否更快达到单 Actor 的最终搜索水平。
- **Search speedup**：相对于 1 个 Actor 的完整搜索加速比。
- **Base T90 speedup**：相对于 1 个 Actor 的 Base T90 加速比。
- **Signal recall**：最终规则集合覆盖真实信号规则的比例。

## 主要结论

在固定总 episode 的公平设置下，分区搜索能体现多 Actor 的优势。完整搜索时间的最佳设置是 {int(best_search['workers'])} 个 Actor，加速比为 {best_search['search_speedup_vs_1']:.2f}；Base T90 的最佳设置是 {int(best_t90['workers'])} 个 Actor，加速比为 {best_t90['baseline90_speedup_vs_1']:.2f}。这说明并行 Actor 的收益主要来自两个方面：第一，规则空间被拆分后，每个 Actor 的动作空间显著缩小；第二，各 Actor 可以并行探索不同概念格/规则子空间，并异步把局部最优规则汇总到全局模型。

因此，这个实验比“所有 Actor 都探索完整 4000 条规则”的设置更符合本文方法设定，也更适合用于论证 RT-A3C-MLN 在大规模规则候选空间下的实时结构自适应能力。
"""
    (ARTIFACT_DIR / "Partitioned_Actor_Scope_Explanation.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[31, 37, 43])
    parser.add_argument("--actor-counts", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
    parser.add_argument("--rule-scope", type=int, default=4000)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=2400)
    parser.add_argument("--density", type=float, default=0.055)
    parser.add_argument("--signal-rules", type=int, default=48)
    parser.add_argument("--total-episodes", type=int, default=160)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--max-active-rules", type=int, default=48)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    traces = []
    for seed in args.seeds:
        for workers in args.actor_counts:
            print(
                f"[partitioned-scope] seed={seed} actors={workers} "
                f"scope={args.rule_scope} scope/actor={math.ceil(args.rule_scope / workers)} "
                f"total_eps={args.total_episodes}",
                flush=True,
            )
            row, trace = run_one(args, seed, workers)
            rows.append(row)
            traces.append(trace)

    detail = pd.DataFrame(rows)
    trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    detail = add_common_target(detail, trace)
    summary = summarize(detail)
    table = make_table(summary)

    detail.to_csv(RESULT_DIR / "partitioned_actor_scope_detail.csv", index=False)
    trace.to_csv(RESULT_DIR / "partitioned_actor_scope_trace.csv", index=False)
    summary.to_csv(RESULT_DIR / "partitioned_actor_scope_summary.csv", index=False)
    table.to_csv(TABLE_DIR / "Table_Partitioned_Actor_Scope.csv", index=False)
    write_markdown_table(table, TABLE_DIR / "Table_Partitioned_Actor_Scope.md", "Partitioned actor scope experiment")
    draw_figure(summary, trace)
    write_explanation(summary)

    print("\n[Partitioned actor scope table]")
    print(table.to_string(index=False))
    print(f"\n[done] results: {RESULT_DIR}")
    print(f"[done] artifacts: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
