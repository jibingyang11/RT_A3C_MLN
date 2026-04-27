"""Protocol A: in-window rule-selection sanity check (v2).

For each window in the formal plan, we:
  1. Mine the rule pool (paper-aligned thresholds 0.30/0.70).
  2. Build the warm-start dataset from that window's target labels.
  3. Train the warm-start classifier on the SAME window.
  4. Greedy-evaluate on the SAME window.

This intentionally trains and tests on the same window: it is a SANITY
CHECK that the pipeline can recover the intended rule mask in-window.
The paper Section 6.1 must NOT interpret these numbers as generalization
evidence. Cross-window generalization is Protocol B.

Outputs:
  outputs/logs/v2/protocolA_per_window.csv       -- per-window metrics
  outputs/logs/v2/protocolA_rule_confusion.csv   -- per-window TP/FN/TN/FP
  outputs/logs/v2/protocolA_group_stats.csv      -- aggregated by attack_bin
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


def main(out_dir: Path, mining_cfg: MiningConfigV2, ws_cfg: WarmStartConfigV2):
    set_global_seed(ws_cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = load_formal_plan(PROJECT_ROOT)

    rows = []
    confusion_rows = []
    for _, prow in plan.iterrows():
        wid = int(prow["window_id"])
        try:
            art = prepare_window_artifacts(PROJECT_ROOT, wid, mining_cfg)
        except Exception as e:
            print(f"[skip] window {wid}: {e}")
            continue

        model_path = PROJECT_ROOT / "outputs" / "v2" / "models" / f"protocolA_w{wid}.pth"
        train_acc, model = warmstart_classifier(art.X_ws, art.y_ws, model_path, ws_cfg)
        metrics = greedy_evaluate(model, art.state_dict, env_cls=PaperRuleEnv)

        rows.append(
            {
                "window_id": wid,
                "attack_ratio": art.attack_ratio,
                "attack_bin": prow.get("attack_bin", ""),
                "auto_mode": art.auto_mode,
                "requested_mode": art.requested_mode,
                "actual_mode_used": art.actual_mode_used,
                "rule_pool_hash": art.rule_pool_hash,
                "used_support": art.used_support,
                "num_rules": art.state_dict["num_rules"],
                "num_attack_rules": art.n_attack_rules,
                "num_normal_rules": art.n_normal_rules,
                "warmstart_train_accuracy": train_acc,
                "attack_keep_rate": metrics["attack_keep_rate"],
                "normal_disable_rate": metrics["normal_disable_rate"],
                "selection_accuracy": metrics["selection_accuracy"],
                "greedy_total_reward": metrics["greedy_total_reward"],
            }
        )
        confusion_rows.append(
            {
                "window_id": wid,
                "attack_ratio": art.attack_ratio,
                "tp_attack_keep": metrics["tp_attack_keep"],
                "fn_attack_disable": metrics["fn_attack_disable"],
                "tn_normal_disable": metrics["tn_normal_disable"],
                "fp_normal_keep": metrics["fp_normal_keep"],
            }
        )

    per_window = pd.DataFrame(rows).sort_values("attack_ratio").reset_index(drop=True)
    confusion = pd.DataFrame(confusion_rows).sort_values("window_id").reset_index(drop=True)

    per_window.to_csv(out_dir / "protocolA_per_window.csv", index=False)
    confusion.to_csv(out_dir / "protocolA_rule_confusion.csv", index=False)

    if not per_window.empty:
        agg_cols = [
            "attack_keep_rate",
            "normal_disable_rate",
            "selection_accuracy",
            "greedy_total_reward",
        ]
        group_stats = (
            per_window.groupby("attack_bin")[agg_cols]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        group_stats.columns = [
            "_".join(c).strip("_") for c in group_stats.columns.to_flat_index()
        ]
        group_stats.to_csv(out_dir / "protocolA_group_stats.csv", index=False)

    print("Protocol A finished:")
    print(per_window)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mining_cfg = MiningConfigV2()
    ws_cfg = WarmStartConfigV2(seed=args.seed)
    out_dir = PROJECT_ROOT / "outputs" / "logs" / "v2"
    main(out_dir, mining_cfg, ws_cfg)
