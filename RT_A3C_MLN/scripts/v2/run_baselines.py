"""Internal baselines on Protocol B test windows (v2).

For each seed used in run_protocolB.py we evaluate:
  - all-keep: a_i = KEEP for every i.
  - all-disable: a_i = DISABLE for every i.
  - random (10 trials averaged)
  - confidence-threshold: keep iff confidence >= 0.7
  - warm-start only: load the saved warm-start checkpoint
  - warm-start + A3C: load the saved A3C checkpoint
on the same held-out test windows produced by run_protocolB.py.

Outputs:
  outputs/logs/v2/baselines_protocolB_per_seed.csv  -- one row per (seed, method)
  outputs/logs/v2/baselines_protocolB.csv           -- mean over seeds (paper Table 3)
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

from src.rl.simple_env_v2 import ACTION_DISABLE, ACTION_KEEP, PaperRuleEnv
from src.v2.pipeline import prepare_window_artifacts, stratified_split
from src.v2.policy import PolicyNetV2, WarmStartConfigV2, greedy_evaluate
from src.v2.rule_pool import MiningConfigV2
from src.v2.window_io import load_formal_plan


def metrics_from_actions(actions: list[int], target_labels: np.ndarray, env_state) -> dict:
    actions_arr = np.asarray(actions, dtype=int)
    attack_mask = target_labels == 1
    normal_mask = target_labels == 0
    if attack_mask.sum() > 0:
        akr = float((actions_arr[attack_mask] == 0).mean())
    else:
        akr = float("nan")
    if normal_mask.sum() > 0:
        ndr = float((actions_arr[normal_mask] == 1).mean())
    else:
        ndr = float("nan")
    correct = (
        ((target_labels == 1) & (actions_arr == 0))
        | ((target_labels == 0) & (actions_arr == 1))
    )
    return {
        "attack_keep_rate": akr,
        "normal_disable_rate": ndr,
        "selection_accuracy": float(correct.mean()),
        "greedy_total_reward": float(env_state),
    }


def run_actions_in_env(state_dict: dict, actions: list[int]) -> dict:
    env = PaperRuleEnv(state_dict)
    env.reset()
    total = 0.0
    for t, a in enumerate(actions):
        _, r, _, _ = env.step(t, int(a))
        total += float(r)
    return metrics_from_actions(
        actions,
        np.asarray(state_dict["target_labels"], dtype=int),
        total,
    )


def evaluate_method(name: str, method_fn, test_arts) -> list[dict]:
    rows = []
    for art in test_arts:
        actions = method_fn(art)
        m = run_actions_in_env(art.state_dict, actions)
        rows.append(
            {
                "method": name,
                "window_id": art.window_id,
                "attack_ratio": art.attack_ratio,
                **m,
            }
        )
    return rows


def evaluate_model_method(model: PolicyNetV2, name: str, test_arts) -> list[dict]:
    rows = []
    for art in test_arts:
        m = greedy_evaluate(model, art.state_dict, env_cls=PaperRuleEnv)
        rows.append(
            {
                "method": name,
                "window_id": art.window_id,
                "attack_ratio": art.attack_ratio,
                "attack_keep_rate": m["attack_keep_rate"],
                "normal_disable_rate": m["normal_disable_rate"],
                "selection_accuracy": m["selection_accuracy"],
                "greedy_total_reward": m["greedy_total_reward"],
            }
        )
    return rows


def high_heldout_split(plan_df: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    rows = []
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
        rows.append({"attack_bin": attack_bin, "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids})
    out["_detail_df"] = pd.DataFrame(rows)
    return out


def main(seeds: list[int], mining_cfg: MiningConfigV2, random_trials: int, high_heldout: bool = False):
    plan = load_formal_plan(PROJECT_ROOT)
    out_dir = PROJECT_ROOT / "outputs" / "logs" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = PROJECT_ROOT / "outputs" / "v2" / "models"

    per_seed_rows = []
    for seed in seeds:
        splits = (
            high_heldout_split(plan, seed=seed)
            if high_heldout
            else stratified_split(plan, seed=seed, train_per_bin=4, val_per_bin=1, test_per_bin=2)
        )
        test_arts = []
        for wid in splits["test"]:
            try:
                test_arts.append(prepare_window_artifacts(PROJECT_ROOT, wid, mining_cfg))
            except Exception as e:
                print(f"[baselines seed {seed}] skip window {wid}: {e}")
        if not test_arts:
            continue

        # all-keep
        rows = evaluate_method(
            "all-keep", lambda art: [ACTION_KEEP] * art.state_dict["num_rules"], test_arts
        )
        # all-disable
        rows += evaluate_method(
            "all-disable", lambda art: [ACTION_DISABLE] * art.state_dict["num_rules"], test_arts
        )

        # random — average over `random_trials` per window
        rng = np.random.default_rng(seed)
        random_rows = []
        for art in test_arts:
            n = art.state_dict["num_rules"]
            tr_acc = []
            tr_akr = []
            tr_ndr = []
            tr_rew = []
            for _ in range(random_trials):
                acts = rng.integers(0, 2, size=n).tolist()
                m = run_actions_in_env(art.state_dict, acts)
                tr_acc.append(m["selection_accuracy"])
                tr_akr.append(m["attack_keep_rate"])
                tr_ndr.append(m["normal_disable_rate"])
                tr_rew.append(m["greedy_total_reward"])
            random_rows.append(
                {
                    "method": "random",
                    "window_id": art.window_id,
                    "attack_ratio": art.attack_ratio,
                    "attack_keep_rate": float(np.nanmean(tr_akr)),
                    "normal_disable_rate": float(np.nanmean(tr_ndr)),
                    "selection_accuracy": float(np.mean(tr_acc)),
                    "greedy_total_reward": float(np.mean(tr_rew)),
                }
            )
        rows += random_rows

        # confidence threshold
        def conf_thresh(art):
            acts = []
            for i, r in enumerate(art.rule_pool_df.to_dict(orient="records")):
                acts.append(ACTION_KEEP if float(r["confidence"]) >= 0.7 else ACTION_DISABLE)
            return acts

        rows += evaluate_method("confidence(c>=0.7)", conf_thresh, test_arts)

        # Load warm-start and A3C models
        prefix = "protocolB_high" if high_heldout else "protocolB"
        ws_path = model_dir / f"{prefix}_seed{seed}_warmstart.pth"
        a3c_path = model_dir / f"{prefix}_seed{seed}_a3c.pth"
        if ws_path.exists():
            ws_model = PolicyNetV2()
            ws_model.load_state_dict(torch.load(ws_path, map_location="cpu"))
            ws_model.eval()
            rows += evaluate_model_method(ws_model, "warm-start", test_arts)
        else:
            print(f"[seed {seed}] warm-start checkpoint not found at {ws_path}; run run_protocolB.py first")
        if a3c_path.exists():
            a3c_model = PolicyNetV2()
            a3c_model.load_state_dict(torch.load(a3c_path, map_location="cpu"))
            a3c_model.eval()
            rows += evaluate_model_method(a3c_model, "warm-start+A3C", test_arts)
        else:
            print(f"[seed {seed}] A3C checkpoint not found at {a3c_path}; run run_protocolB.py first")

        df = pd.DataFrame(rows)
        df["seed"] = seed
        per_seed_rows.append(df)

    if not per_seed_rows:
        print("No per-seed rows; aborting.")
        return

    per_seed_df = pd.concat(per_seed_rows, ignore_index=True)
    out_prefix = "baselines_protocolB_high" if high_heldout else "baselines_protocolB"
    per_seed_df.to_csv(out_dir / f"{out_prefix}_per_seed.csv", index=False)

    # Aggregate per method (across seeds AND windows)
    agg = (
        per_seed_df.groupby("method")[
            [
                "attack_keep_rate",
                "normal_disable_rate",
                "selection_accuracy",
                "greedy_total_reward",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    agg.to_csv(out_dir / f"{out_prefix}.csv", index=False)
    print(agg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 42])
    parser.add_argument("--random_trials", type=int, default=10)
    parser.add_argument("--high-heldout", action="store_true")
    args = parser.parse_args()
    main(args.seeds, MiningConfigV2(), args.random_trials, high_heldout=args.high_heldout)
