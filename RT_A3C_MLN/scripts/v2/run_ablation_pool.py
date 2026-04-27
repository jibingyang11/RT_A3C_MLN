"""Adaptive vs forced rule-pool ablation (v2).

Fixes the original ``ablation_modes.csv`` issue (balanced_only and
standard_only producing identical numbers) by using
``build_rule_pool_v2`` with explicit ``force_mode`` so the SAME window
goes through different branches.

For every formal-plan window we run four configurations:
  - balanced_only
  - standard_only
  - relaxed_only
  - adaptive (auto-selected mode)

For each config we record:
  - actual_mode_used  (post-fallback)
  - rule_pool_hash    (different hash => different rule sets)
  - n_attack_rules / n_normal_rules
  - warm-start train accuracy
  - greedy selection accuracy with fresh per-window warm-start

Outputs:
  outputs/logs/v2/ablation_modes_per_window.csv
  outputs/logs/v2/ablation_modes.csv  (paper Table 7 source)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.simple_env_v2 import PaperRuleEnv
from src.v2.pipeline import prepare_window_artifacts
from src.v2.policy import (
    WarmStartConfigV2,
    greedy_evaluate,
    set_global_seed,
    warmstart_classifier,
)
from src.v2.rule_pool import MiningConfigV2
from src.v2.window_io import load_formal_plan


def run_one_window(window_id, attack_bin, mode, mining_cfg, ws_cfg) -> dict | None:
    try:
        art = prepare_window_artifacts(
            PROJECT_ROOT, window_id, mining_cfg,
            force_mode=mode if mode is not None else None,
            cache=False,
        )
    except Exception as e:
        print(f"[ablation skip] window {window_id} mode={mode}: {e}")
        return None
    save_path = (
        PROJECT_ROOT / "outputs" / "v2" / "models" / f"ablation_w{window_id}_{mode}.pth"
    )
    train_acc, model = warmstart_classifier(art.X_ws, art.y_ws, save_path, ws_cfg)
    metrics = greedy_evaluate(model, art.state_dict, env_cls=PaperRuleEnv)
    return {
        "window_id": window_id,
        "attack_ratio": art.attack_ratio,
        "attack_bin": attack_bin,
        "forced_mode": mode if mode is not None else "adaptive",
        "actual_mode_used": art.actual_mode_used,
        "rule_pool_hash": art.rule_pool_hash,
        "n_attack_rules": art.n_attack_rules,
        "n_normal_rules": art.n_normal_rules,
        "warmstart_train_accuracy": train_acc,
        "selection_accuracy": metrics["selection_accuracy"],
        "attack_keep_rate": metrics["attack_keep_rate"],
        "normal_disable_rate": metrics["normal_disable_rate"],
        "greedy_total_reward": metrics["greedy_total_reward"],
    }


def main(seed: int):
    set_global_seed(seed)
    plan = load_formal_plan(PROJECT_ROOT)
    out_dir = PROJECT_ROOT / "outputs" / "logs" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    mining_cfg = MiningConfigV2()
    ws_cfg = WarmStartConfigV2(seed=seed)

    rows = []
    for _, prow in plan.iterrows():
        wid = int(prow["window_id"])
        ab = prow.get("attack_bin", "")
        for mode in ["balanced", "standard", "relaxed", None]:
            r = run_one_window(wid, ab, mode, mining_cfg, ws_cfg)
            if r is not None:
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_modes_per_window.csv", index=False)

    # Aggregate by forced_mode x attack_bin
    pivot = (
        df.groupby(["forced_mode", "attack_bin"])
        ["selection_accuracy"].mean()
        .unstack("attack_bin")
        .reset_index()
    )
    overall = (
        df.groupby("forced_mode")["selection_accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"selection_accuracy": "overall"})
    )
    out = pivot.merge(overall, on="forced_mode", how="left")
    out.to_csv(out_dir / "ablation_modes.csv", index=False)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
