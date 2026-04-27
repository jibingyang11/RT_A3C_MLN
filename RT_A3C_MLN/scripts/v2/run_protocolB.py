"""Protocol B: strict cross-window generalization (v2).

For each seed:
  1. Stratified split by attack_bin (low/mid/high).
     We REQUIRE every bin to contribute to train (unlike the legacy
     split where the high bin produced 0 train windows). To do this we
     allocate train_per_bin first, then val, then test.
  2. Concatenate the warm-start training data across train windows.
  3. Train the warm-start classifier on this concatenated set.
  4. Optionally fine-tune with on-policy A3C using PaperRuleEnv.
  5. Greedy-evaluate on the held-out TEST windows.
  6. Save the warm-start checkpoint and the warm-start+A3C checkpoint
     under the SAME seed-aware path so Section 6.3 can reload them.

Outputs:
  outputs/logs/v2/protocolB_seed{seed}_test_per_window.csv
  outputs/logs/v2/protocolB_seed{seed}_split_detail.csv
  outputs/logs/v2/protocolB_seed{seed}_a3c_training_curve.csv
  outputs/logs/v2/protocolB_summary.csv (aggregated across all seeds)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.simple_env_v2 import PaperRuleEnv
from src.v2.pipeline import prepare_window_artifacts, stratified_split
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


def evaluate_set(model, art_list) -> list[dict]:
    rows = []
    for art in art_list:
        m = greedy_evaluate(model, art.state_dict, env_cls=PaperRuleEnv)
        rows.append(
            {
                "window_id": art.window_id,
                "attack_ratio": art.attack_ratio,
                "actual_mode_used": art.actual_mode_used,
                "n_attack_rules": art.n_attack_rules,
                "n_normal_rules": art.n_normal_rules,
                "attack_keep_rate": m["attack_keep_rate"],
                "normal_disable_rate": m["normal_disable_rate"],
                "selection_accuracy": m["selection_accuracy"],
                "greedy_total_reward": m["greedy_total_reward"],
                "tp_attack_keep": m["tp_attack_keep"],
                "fn_attack_disable": m["fn_attack_disable"],
                "tn_normal_disable": m["tn_normal_disable"],
                "fp_normal_keep": m["fp_normal_keep"],
            }
        )
    return rows


def run_one_seed(
    seed: int,
    plan: pd.DataFrame,
    mining_cfg: MiningConfigV2,
    ws_cfg: WarmStartConfigV2,
    a3c_cfg: A3CConfigV2,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)

    splits = stratified_split(plan, seed=seed, train_per_bin=4, val_per_bin=1, test_per_bin=2)
    detail_df: pd.DataFrame = splits["_detail_df"]
    detail_df.assign(seed=seed).to_csv(
        out_dir / f"protocolB_seed{seed}_split_detail.csv", index=False
    )

    def _safe_prepare(ids):
        out = []
        for wid in ids:
            try:
                out.append(prepare_window_artifacts(PROJECT_ROOT, wid, mining_cfg))
            except Exception as e:
                print(f"[seed {seed}] skip window {wid} (rule pool unavailable): {e}")
        return out

    train_arts = _safe_prepare(splits["train"])
    test_arts = _safe_prepare(splits["test"])

    if not train_arts or not test_arts:
        print(f"[seed {seed}] empty train/test set, skipping")
        return None

    X_train = np.concatenate([a.X_ws for a in train_arts], axis=0)
    y_train = np.concatenate([a.y_ws for a in train_arts], axis=0)

    ws_cfg_seed = WarmStartConfigV2(**{**ws_cfg.__dict__, "seed": seed})
    ws_path = out_dir.parent.parent / "v2" / "models" / f"protocolB_seed{seed}_warmstart.pth"
    ws_path.parent.mkdir(parents=True, exist_ok=True)
    train_acc, model = warmstart_classifier(X_train, y_train, ws_path, ws_cfg_seed)

    # Evaluate warm-start
    ws_test_rows = evaluate_set(model, test_arts)
    ws_df = pd.DataFrame(ws_test_rows)
    ws_df["seed"] = seed
    ws_df["stage"] = "warmstart"
    ws_df["warmstart_train_accuracy"] = train_acc

    # A3C fine-tune
    train_states = [a.state_dict for a in train_arts]
    a3c_cfg_seed = A3CConfigV2(**{**a3c_cfg.__dict__, "seed": seed})

    curve_rows = []

    def cb(ep, payload):
        for w in payload["workers"]:
            curve_rows.append({"seed": seed, "episode": ep, **w})

    a3c_model = a3c_fine_tune(
        init_model=model,
        train_states=train_states,
        env_cls=PaperRuleEnv,
        cfg=a3c_cfg_seed,
        log_callback=cb,
    )

    a3c_path = out_dir.parent.parent / "v2" / "models" / f"protocolB_seed{seed}_a3c.pth"
    import torch
    torch.save(a3c_model.state_dict(), a3c_path)

    a3c_test_rows = evaluate_set(a3c_model, test_arts)
    a3c_df = pd.DataFrame(a3c_test_rows)
    a3c_df["seed"] = seed
    a3c_df["stage"] = "warmstart_a3c"
    a3c_df["warmstart_train_accuracy"] = train_acc

    per_window = pd.concat([ws_df, a3c_df], ignore_index=True)
    per_window.to_csv(out_dir / f"protocolB_seed{seed}_test_per_window.csv", index=False)

    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(
            out_dir / f"protocolB_seed{seed}_a3c_training_curve.csv", index=False
        )

    return per_window


def main(seeds: list[int], mining_cfg: MiningConfigV2, ws_cfg: WarmStartConfigV2, a3c_cfg: A3CConfigV2):
    plan = load_formal_plan(PROJECT_ROOT)
    out_dir = PROJECT_ROOT / "outputs" / "logs" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in seeds:
        df = run_one_seed(seed, plan, mining_cfg, ws_cfg, a3c_cfg, out_dir)
        if df is not None:
            all_rows.append(df)

    if not all_rows:
        print("No seeds produced results.")
        return

    full = pd.concat(all_rows, ignore_index=True)

    summary_rows = []
    for (seed, stage), g in full.groupby(["seed", "stage"]):
        summary_rows.append(
            {
                "seed": seed,
                "stage": stage,
                "n_test_windows": len(g),
                "mean_attack_keep_rate": float(g["attack_keep_rate"].mean()),
                "mean_normal_disable_rate": float(g["normal_disable_rate"].mean()),
                "mean_selection_accuracy": float(g["selection_accuracy"].mean()),
                "mean_greedy_total_reward": float(g["greedy_total_reward"].mean()),
                "test_window_ids": json.dumps(sorted(g["window_id"].tolist())),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "protocolB_summary.csv", index=False)
    print(summary_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 42])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    mining_cfg = MiningConfigV2()
    ws_cfg = WarmStartConfigV2()
    a3c_cfg = A3CConfigV2(num_workers=args.workers, episodes_per_worker=args.episodes)
    main(args.seeds, mining_cfg, ws_cfg, a3c_cfg)
