
"""Run A3C training over formal windows (V6 fixed).

Main differences:
* warm-start explicitly uses external window-level validation arrays;
* candidate scoring uses a balance-aware harmonic term;
* default val_train_blend=0.0 to avoid contaminating model selection with train windows;
* prefers best_ema / best_raw / warmstart / global_ema / global_raw and does not
  flood the selector with every checkpoint by default.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.cross_window_experiment import CrossWindowSplitConfig, split_plan_by_attack_bin
from src.pipeline.formal_experiment import (
    EvalConfig,
    RuleMiningConfig,
    WarmStartConfig,
    build_warmstart_dataset,
    evaluate_baselines,
    greedy_evaluate,
    prepare_window_learning_artifacts,
    save_dataframe,
)
from src.rl.a3c_trainer import A3CConfig, train_a3c
from src.rl.warmstart_helper import WarmStartTrainConfig, train_warmstart_classifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run A3C training over the formal window plan.")
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--plan-csv", type=Path, required=True)

    p.add_argument("--train-per-bin", type=int, default=6)
    p.add_argument("--val-per-bin", type=int, default=2)
    p.add_argument("--test-per-bin", type=int, default=2)

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-episodes-per-worker", type=int, default=120)
    p.add_argument("--n-step", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--entropy-coef", type=float, default=0.002)
    p.add_argument("--value-coef", type=float, default=0.25)
    p.add_argument("--bc-coef", type=float, default=1.2)
    p.add_argument("--bc-focal-gamma", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=3.0)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=12)
    p.add_argument("--sample-attack-oversample", type=float, default=1.0)
    p.add_argument("--val-train-blend", type=float, default=0.0)

    p.add_argument("--ws-epochs", type=int, default=200)
    p.add_argument("--ws-lr", type=float, default=8e-4)
    p.add_argument("--ws-dropout", type=float, default=0.05)
    p.add_argument("--ws-label-smoothing", type=float, default=0.02)
    p.add_argument("--ws-weight-decay", type=float, default=1e-4)
    p.add_argument("--ws-batch-size", type=int, default=64)
    p.add_argument("--ws-val-split", type=float, default=0.0)

    p.add_argument("--min-support", type=float, default=0.1)
    p.add_argument("--min-confidence", type=float, default=0.6)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="a3c_v6")
    p.add_argument("--skip-warmstart", action="store_true")
    return p.parse_args()


def _ensure_attack_bin(plan_df: pd.DataFrame) -> pd.DataFrame:
    if "attack_bin" not in plan_df.columns:
        plan_df = plan_df.copy()
        plan_df["attack_bin"] = plan_df["attack_ratio"].apply(
            lambda r: "low" if r < 0.3 else ("high" if r >= 0.7 else "mid")
        )
    return plan_df


def _infer_window_mode(attack_ratio: float, mining_config: RuleMiningConfig) -> str:
    if attack_ratio < mining_config.low_attack_ratio_threshold:
        return "balanced"
    if attack_ratio >= mining_config.high_attack_ratio_threshold:
        return "relaxed"
    return "standard"


def _collect_window_artifacts(project_root: Path, ids: list[int], mining_config: RuleMiningConfig) -> list[dict[str, Any]]:
    arts = []
    plan_path = project_root / "data" / "stream" / "swat" / "formal_mixed_windows_plan.csv"
    plan_df = pd.read_csv(plan_path) if plan_path.exists() else pd.DataFrame()
    for wid in ids:
        try:
            cache_path = (
                project_root
                / "data"
                / "stream"
                / "swat"
                / "window_rule_states"
                / f"window_{int(wid)}_rule_state.pkl"
            )
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    state_dict = pickle.load(f)
                X_ws, y_ws = build_warmstart_dataset(state_dict)
                plan_row = plan_df[plan_df["window_id"].astype(int) == int(wid)] if not plan_df.empty else pd.DataFrame()
                attack_ratio = float(plan_row.iloc[0]["attack_ratio"]) if not plan_row.empty else float("nan")
                arts.append({
                    "window_id": int(wid),
                    "attack_ratio": attack_ratio,
                    "window_mode": _infer_window_mode(attack_ratio, mining_config) if not np.isnan(attack_ratio) else "cached",
                    "state_dict": state_dict,
                    "X_ws": X_ws,
                    "y_ws": y_ws,
                })
            else:
                arts.append(prepare_window_learning_artifacts(project_root, int(wid), "windows_mixed.csv", mining_config))
        except Exception as exc:
            print(f"[warn] skipping window {wid}: {exc}")
    return arts


def _eval_model_on_windows(model_path: Path, arts: list[dict[str, Any]], warmstart_config: WarmStartConfig, eval_config: EvalConfig) -> pd.DataFrame:
    rows = []
    for art in arts:
        _, metrics_df = greedy_evaluate(model_path, art["state_dict"], warmstart_config, eval_config)
        r = metrics_df.iloc[0].to_dict()
        r.update({
            "window_id": int(art["window_id"]),
            "attack_ratio": float(art["attack_ratio"]),
            "window_mode": art["window_mode"],
            "num_rules": int(art["state_dict"]["num_rules"]),
            "num_attack_rules": int((np.asarray(art["state_dict"]["target_labels"]) == 1).sum()),
            "num_normal_rules": int((np.asarray(art["state_dict"]["target_labels"]) == 0).sum()),
        })
        rows.append(r)
    return pd.DataFrame(rows).sort_values("attack_ratio").reset_index(drop=True) if rows else pd.DataFrame()


def _score(df: pd.DataFrame) -> float:
    if df.empty:
        return -1e9
    acc = float(df["selection_accuracy"].mean())
    keep = float(df["attack_keep_rate"].mean())
    disable = float(df["normal_disable_rate"].mean())
    balance = 2.0 * keep * disable / (keep + disable + 1e-8)
    return 0.4 * acc + 0.6 * balance


def _blended_score(train_df: pd.DataFrame, val_df: pd.DataFrame, alpha: float) -> float:
    train_score = _score(train_df) if not train_df.empty else -1e9
    val_score = _score(val_df) if not val_df.empty else -1e9
    if train_score < -1e8 and val_score < -1e8:
        return -1e9
    if train_score < -1e8:
        return val_score
    if val_score < -1e8:
        return train_score
    return float(alpha * train_score + (1.0 - alpha) * val_score)


def main() -> None:
    args = parse_args()
    plan_df = _ensure_attack_bin(pd.read_csv(args.plan_csv))

    split_config = CrossWindowSplitConfig(
        train_per_bin=args.train_per_bin,
        val_per_bin=args.val_per_bin,
        test_per_bin=args.test_per_bin,
        random_state=args.seed,
    )
    splits = split_plan_by_attack_bin(plan_df, split_config)
    if "_detail_df" in splits:
        print("[split detail]")
        print(splits["_detail_df"])

    mining_config = RuleMiningConfig(min_support=args.min_support, min_confidence=args.min_confidence)
    warmstart_eval_config = WarmStartConfig(seed=args.seed, hidden_dim=args.hidden_dim)
    eval_config = EvalConfig(seed=args.seed)

    train_arts = _collect_window_artifacts(args.project_root, splits["train"], mining_config)
    val_arts = _collect_window_artifacts(args.project_root, splits["val"], mining_config)
    test_arts = _collect_window_artifacts(args.project_root, splits["test"], mining_config)

    if not train_arts:
        raise ValueError("No usable training windows after rule-pool construction.")
    if not test_arts:
        raise ValueError("No usable test windows after rule-pool construction.")

    train_states = [a["state_dict"] for a in train_arts]
    val_states = [a["state_dict"] for a in val_arts]

    a3c_config = A3CConfig(
        num_workers=args.num_workers,
        num_episodes_per_worker=args.num_episodes_per_worker,
        n_step=args.n_step,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        bc_coef=args.bc_coef,
        bc_focal_gamma=args.bc_focal_gamma,
        max_grad_norm=args.max_grad_norm,
        ema_decay=args.ema_decay,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        temperature=args.temperature,
        log_every=args.log_every,
        eval_every=args.eval_every,
        early_stop_patience=args.early_stop_patience,
        sample_attack_oversample=args.sample_attack_oversample,
        val_train_blend=args.val_train_blend,
    )

    models_dir = args.project_root / "outputs" / "models"
    logs_dir = args.project_root / "outputs" / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    warmstart_model = models_dir / f"{args.run_name}_warmstart.pth"
    selected_model_path = models_dir / f"{args.run_name}_selected_policy.pth"

    if not args.skip_warmstart:
        X_train = np.concatenate([a["X_ws"] for a in train_arts], axis=0)
        y_train = np.concatenate([a["y_ws"] for a in train_arts], axis=0)
        X_val = np.concatenate([a["X_ws"] for a in val_arts], axis=0) if val_arts else None
        y_val = np.concatenate([a["y_ws"] for a in val_arts], axis=0) if val_arts else None

        ws_cfg = WarmStartTrainConfig(
            seed=args.seed,
            state_dim=12,
            hidden_dim=args.hidden_dim,
            dropout=args.ws_dropout,
            lr=args.ws_lr,
            weight_decay=args.ws_weight_decay,
            label_smoothing=args.ws_label_smoothing,
            epochs=args.ws_epochs,
            batch_size=args.ws_batch_size,
            val_split=args.ws_val_split,
            deduplicate=True,
        )
        _, ws_info = train_warmstart_classifier(
            X_train,
            y_train,
            warmstart_model,
            ws_cfg,
            X_val=X_val,
            y_val=y_val,
        )
        print(
            f"[warmstart] train_acc={ws_info['train_accuracy']:.4f} "
            f"val_acc={ws_info['best_val_accuracy']:.4f} "
            f"epoch={ws_info['epoch']} "
            f"used_external_val={ws_info['used_external_val']} "
            f"n_train={ws_info['num_train_samples']} "
            f"n_val={ws_info['num_val_samples']}"
        )

    _, history_df = train_a3c(
        train_state_dicts=train_states,
        config=a3c_config,
        save_dir=models_dir,
        run_name=args.run_name,
        initial_model_path=(warmstart_model if warmstart_model.exists() else None),
        val_state_dicts=val_states if val_states else None,
    )
    save_dataframe(history_df, logs_dir / f"{args.run_name}_training_history.csv")

    candidate_paths = []
    best_ema = models_dir / f"{args.run_name}_best_ema.pth"
    best_raw = models_dir / f"{args.run_name}_best_raw.pth"
    global_ema = models_dir / f"{args.run_name}_global_policy_ema.pth"
    global_raw = models_dir / f"{args.run_name}_global_policy.pth"

    if best_ema.exists():
        candidate_paths.append(("best_ema", best_ema))
    if best_raw.exists():
        candidate_paths.append(("best_raw", best_raw))
    if warmstart_model.exists():
        candidate_paths.append(("warmstart", warmstart_model))
    if global_ema.exists():
        candidate_paths.append(("global_ema", global_ema))
    if global_raw.exists():
        candidate_paths.append(("global_raw", global_raw))

    seen = set()
    deduped = []
    for name, p in candidate_paths:
        if str(p) not in seen:
            seen.add(str(p))
            deduped.append((name, p))
    candidate_paths = deduped

    candidates = []
    for name, path in candidate_paths:
        train_df = _eval_model_on_windows(path, train_arts, warmstart_eval_config, eval_config)
        val_df = _eval_model_on_windows(path, val_arts, warmstart_eval_config, eval_config) if val_arts else pd.DataFrame()
        score_train = _score(train_df)
        score_val = _score(val_df) if not val_df.empty else float("nan")
        score_blended = _blended_score(train_df, val_df, args.val_train_blend)
        candidates.append((name, path, train_df, val_df, score_train, score_val, score_blended))

    scoreboard = pd.DataFrame([
        {
            "candidate": name,
            "train_score": score_train,
            "val_score": score_val,
            "blended_score": score_blended,
            "train_acc": float(train_df["selection_accuracy"].mean()) if not train_df.empty else np.nan,
            "train_attack_keep": float(train_df["attack_keep_rate"].mean()) if not train_df.empty else np.nan,
            "train_normal_disable": float(train_df["normal_disable_rate"].mean()) if not train_df.empty else np.nan,
            "val_acc": float(val_df["selection_accuracy"].mean()) if not val_df.empty else np.nan,
            "val_attack_keep": float(val_df["attack_keep_rate"].mean()) if not val_df.empty else np.nan,
            "val_normal_disable": float(val_df["normal_disable_rate"].mean()) if not val_df.empty else np.nan,
        }
        for name, _, train_df, val_df, score_train, score_val, score_blended in candidates
    ]).sort_values(
        ["blended_score", "val_score", "train_score", "train_acc", "train_attack_keep"],
        ascending=False,
    ).reset_index(drop=True)
    save_dataframe(scoreboard, logs_dir / f"{args.run_name}_candidate_scores.csv")
    print("[model selection] candidates:")
    print(scoreboard)

    best_name, best_path, *_ = max(candidates, key=lambda x: (x[6], x[5], x[4]))
    print(f"[model selection] selected={best_name}")
    state = torch.load(best_path, map_location="cpu")
    torch.save(state, selected_model_path)

    rows = []
    for art in test_arts:
        _, metrics_df = greedy_evaluate(selected_model_path, art["state_dict"], warmstart_eval_config, eval_config)
        baselines = evaluate_baselines(art["state_dict"], eval_config)
        r = metrics_df.iloc[0].to_dict()
        r.update({
            "window_id": int(art["window_id"]),
            "attack_ratio": float(art["attack_ratio"]),
            "window_mode": art["window_mode"],
            "selected_model": best_name,
            **baselines,
        })
        rows.append(r)

    summary_df = pd.DataFrame(rows).sort_values("attack_ratio").reset_index(drop=True)
    save_dataframe(summary_df, logs_dir / f"{args.run_name}_test_summary.csv")
    print(summary_df)
    print("mean selection_accuracy =", summary_df["selection_accuracy"].mean())
    print("mean attack_keep_rate  =", summary_df["attack_keep_rate"].mean())
    print("mean normal_disable_rate =", summary_df["normal_disable_rate"].mean())
if __name__ == "__main__":
    main()
