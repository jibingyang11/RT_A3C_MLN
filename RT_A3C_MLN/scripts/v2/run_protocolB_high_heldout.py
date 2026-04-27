"""Protocol B-high: strict split that holds out high-attack windows.

The original Protocol B allocates train windows before validation/test
within each attack-ratio bin.  Because the formal plan contains only
three high-attack windows, that protocol uses all high windows for
training and reports held-out performance only on low/mid windows.

This script keeps the same model and metrics but uses a stress split:

* low/mid bins: 4 train, 1 validation, 2 test windows;
* high bin: 1 train, 0 validation, 2 test windows.

It therefore measures cross-window generalization on all bins without
inventing extra data or changing any metric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.simple_env_v2 import PaperRuleEnv
from src.v2.pipeline import prepare_window_artifacts
from src.v2.policy import (
    A3CConfigV2,
    WarmStartConfigV2,
    a3c_fine_tune,
    greedy_evaluate,
    set_global_seed,
    warmstart_classifier,
)
from src.v2.rule_pool import MiningConfigV2
from src.v2.window_io import load_formal_plan


def high_heldout_split(plan_df: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    detail_rows = []
    for attack_bin in ["low", "mid", "high"]:
        ids = plan_df[plan_df["attack_bin"] == attack_bin]["window_id"].astype(int).tolist()
        rng.shuffle(ids)
        if attack_bin == "high":
            train_n = min(1, len(ids))
            val_n = 0
            test_n = min(2, max(0, len(ids) - train_n))
        else:
            train_n = min(4, len(ids))
            val_n = min(1, max(0, len(ids) - train_n))
            test_n = min(2, max(0, len(ids) - train_n - val_n))
        train_ids = ids[:train_n]
        val_ids = ids[train_n: train_n + val_n]
        test_ids = ids[train_n + val_n: train_n + val_n + test_n]
        out["train"].extend(train_ids)
        out["val"].extend(val_ids)
        out["test"].extend(test_ids)
        detail_rows.append({
            "attack_bin": attack_bin,
            "available": len(ids),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        })
    out["_detail_df"] = pd.DataFrame(detail_rows)
    return out


def evaluate_set(model, art_list: list) -> list[dict]:
    rows = []
    for art in art_list:
        metrics = greedy_evaluate(model, art.state_dict, env_cls=PaperRuleEnv)
        rows.append({
            "window_id": art.window_id,
            "attack_ratio": art.attack_ratio,
            "actual_mode_used": art.actual_mode_used,
            "n_attack_rules": art.n_attack_rules,
            "n_normal_rules": art.n_normal_rules,
            **metrics,
        })
    return rows


def run_one_seed(
    seed: int,
    plan: pd.DataFrame,
    mining_cfg: MiningConfigV2,
    ws_cfg: WarmStartConfigV2,
    a3c_cfg: A3CConfigV2,
    out_dir: Path,
) -> pd.DataFrame | None:
    set_global_seed(seed)
    splits = high_heldout_split(plan, seed)
    splits["_detail_df"].assign(seed=seed).to_csv(
        out_dir / f"protocolB_high_seed{seed}_split_detail.csv", index=False
    )

    def _prepare(ids: list[int]) -> list:
        arts = []
        for wid in ids:
            try:
                arts.append(prepare_window_artifacts(PROJECT_ROOT, int(wid), mining_cfg))
            except Exception as exc:
                print(f"[seed {seed}] skip window {wid}: {exc}")
        return arts

    train_arts = _prepare(splits["train"])
    test_arts = _prepare(splits["test"])
    if not train_arts or not test_arts:
        return None

    X_train = np.concatenate([art.X_ws for art in train_arts], axis=0)
    y_train = np.concatenate([art.y_ws for art in train_arts], axis=0)

    models_dir = PROJECT_ROOT / "outputs" / "v2" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ws_cfg_seed = WarmStartConfigV2(**{**ws_cfg.__dict__, "seed": seed})
    ws_path = models_dir / f"protocolB_high_seed{seed}_warmstart.pth"
    train_acc, ws_model = warmstart_classifier(X_train, y_train, ws_path, ws_cfg_seed)

    ws_df = pd.DataFrame(evaluate_set(ws_model, test_arts))
    ws_df["seed"] = seed
    ws_df["stage"] = "warmstart"
    ws_df["warmstart_train_accuracy"] = train_acc

    curve_rows = []

    def callback(ep: int, payload: dict) -> None:
        for worker_row in payload["workers"]:
            curve_rows.append({"seed": seed, "episode": ep, **worker_row})

    a3c_cfg_seed = A3CConfigV2(**{**a3c_cfg.__dict__, "seed": seed})
    a3c_model = a3c_fine_tune(
        init_model=ws_model,
        train_states=[art.state_dict for art in train_arts],
        env_cls=PaperRuleEnv,
        cfg=a3c_cfg_seed,
        log_callback=callback,
    )
    torch.save(a3c_model.state_dict(), models_dir / f"protocolB_high_seed{seed}_a3c.pth")

    a3c_df = pd.DataFrame(evaluate_set(a3c_model, test_arts))
    a3c_df["seed"] = seed
    a3c_df["stage"] = "warmstart_a3c"
    a3c_df["warmstart_train_accuracy"] = train_acc

    per_window = pd.concat([ws_df, a3c_df], ignore_index=True)
    per_window.to_csv(out_dir / f"protocolB_high_seed{seed}_test_per_window.csv", index=False)
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(
            out_dir / f"protocolB_high_seed{seed}_a3c_training_curve.csv", index=False
        )
    return per_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 42])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    plan = load_formal_plan(PROJECT_ROOT)
    out_dir = PROJECT_ROOT / "outputs" / "logs" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    mining_cfg = MiningConfigV2()
    ws_cfg = WarmStartConfigV2()
    a3c_cfg = A3CConfigV2(num_workers=args.workers, episodes_per_worker=args.episodes)

    all_rows = []
    for seed in args.seeds:
        df = run_one_seed(seed, plan, mining_cfg, ws_cfg, a3c_cfg, out_dir)
        if df is not None:
            all_rows.append(df)
    if not all_rows:
        raise RuntimeError("No high-heldout seed produced results.")

    full = pd.concat(all_rows, ignore_index=True)
    summary_rows = []
    for (seed, stage), group in full.groupby(["seed", "stage"]):
        summary_rows.append({
            "seed": seed,
            "stage": stage,
            "n_test_windows": len(group),
            "mean_attack_keep_rate": float(group["attack_keep_rate"].mean()),
            "mean_normal_disable_rate": float(group["normal_disable_rate"].mean()),
            "mean_selection_accuracy": float(group["selection_accuracy"].mean()),
            "mean_greedy_total_reward": float(group["greedy_total_reward"].mean()),
            "test_window_ids": json.dumps(sorted(group["window_id"].astype(int).tolist())),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "protocolB_high_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
