"""Cross-window ablation of adaptive vs forced rule-pool modes.

The older ablation was an in-window sanity check, so selection accuracy was
almost always 1.0 and the figure/table were not useful as paper evidence.
This script evaluates the rule-pool choice under the same high-heldout
Protocol B split used by the main experiments.

Outputs:
  outputs/logs/v2/ablation_modes_high_per_window.csv
  outputs/logs/v2/ablation_modes_high_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v2.run_protocolB_high_heldout import high_heldout_split
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


MODE_TO_FORCE = {
    "balanced_only": "balanced",
    "standard_only": "standard",
    "relaxed_only": "relaxed",
    "adaptive": None,
}


def attack_bin(ratio: float) -> str:
    if ratio < 0.30:
        return "low"
    if ratio < 0.70:
        return "mid"
    return "high"


def prepare_many(ids: list[int], cfg: MiningConfigV2, force_mode: str | None):
    arts = []
    for wid in ids:
        try:
            arts.append(prepare_window_artifacts(PROJECT_ROOT, int(wid), cfg, force_mode=force_mode))
        except Exception as exc:
            print(f"[ablation-high skip] window={wid} force_mode={force_mode}: {exc}")
    return arts


def eval_rows(seed: int, mode_name: str, stage: str, model, arts) -> list[dict]:
    rows = []
    for art in arts:
        m = greedy_evaluate(model, art.state_dict, env_cls=PaperRuleEnv)
        rows.append(
            {
                "seed": seed,
                "forced_mode": mode_name,
                "stage": stage,
                "window_id": art.window_id,
                "attack_ratio": art.attack_ratio,
                "attack_bin": attack_bin(art.attack_ratio),
                "requested_mode": art.requested_mode,
                "actual_mode_used": art.actual_mode_used,
                "rule_pool_hash": art.rule_pool_hash,
                "n_attack_rules": art.n_attack_rules,
                "n_normal_rules": art.n_normal_rules,
                **m,
            }
        )
    return rows


def run_one(seed: int, mode_name: str, episodes: int, workers: int) -> list[dict]:
    set_global_seed(seed)
    plan = load_formal_plan(PROJECT_ROOT)
    splits = high_heldout_split(plan, seed)
    mining_cfg = MiningConfigV2()
    force_mode = MODE_TO_FORCE[mode_name]

    train_arts = prepare_many(splits["train"], mining_cfg, force_mode)
    test_arts = prepare_many(splits["test"], mining_cfg, force_mode)
    if not train_arts or not test_arts:
        return []

    x_train = np.concatenate([a.X_ws for a in train_arts], axis=0)
    y_train = np.concatenate([a.y_ws for a in train_arts], axis=0)

    model_dir = PROJECT_ROOT / "outputs" / "v2" / "models" / "ablation_high"
    model_dir.mkdir(parents=True, exist_ok=True)
    ws_path = model_dir / f"{mode_name}_seed{seed}_warmstart.pth"
    train_acc, ws_model = warmstart_classifier(
        x_train,
        y_train,
        ws_path,
        WarmStartConfigV2(seed=seed),
    )

    rows = eval_rows(seed, mode_name, "warmstart", ws_model, test_arts)
    for r in rows:
        r["warmstart_train_accuracy"] = train_acc

    a3c_model = a3c_fine_tune(
        init_model=ws_model,
        train_states=[a.state_dict for a in train_arts],
        env_cls=PaperRuleEnv,
        cfg=A3CConfigV2(seed=seed, num_workers=workers, episodes_per_worker=episodes),
    )
    torch.save(a3c_model.state_dict(), model_dir / f"{mode_name}_seed{seed}_a3c.pth")

    a3c_rows = eval_rows(seed, mode_name, "warmstart_a3c", a3c_model, test_arts)
    for r in a3c_rows:
        r["warmstart_train_accuracy"] = train_acc
    rows.extend(a3c_rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 42])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--modes", nargs="+", default=list(MODE_TO_FORCE.keys()))
    parser.add_argument(
        "--warmstart-only",
        action="store_true",
        help="Only evaluate the cross-window warm-start policy. This isolates rule-pool quality from A3C variance.",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "outputs" / "logs" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode_name in args.modes:
        if mode_name not in MODE_TO_FORCE:
            raise ValueError(f"Unknown mode {mode_name}; choose from {list(MODE_TO_FORCE)}")
        for seed in args.seeds:
            print(f"[ablation-high] mode={mode_name} seed={seed}")
            if args.warmstart_only:
                set_global_seed(seed)
                plan = load_formal_plan(PROJECT_ROOT)
                splits = high_heldout_split(plan, seed)
                mining_cfg = MiningConfigV2()
                force_mode = MODE_TO_FORCE[mode_name]
                train_arts = prepare_many(splits["train"], mining_cfg, force_mode)
                test_arts = prepare_many(splits["test"], mining_cfg, force_mode)
                if train_arts and test_arts:
                    x_train = np.concatenate([a.X_ws for a in train_arts], axis=0)
                    y_train = np.concatenate([a.y_ws for a in train_arts], axis=0)
                    model_dir = PROJECT_ROOT / "outputs" / "v2" / "models" / "ablation_high"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    train_acc, ws_model = warmstart_classifier(
                        x_train,
                        y_train,
                        model_dir / f"{mode_name}_seed{seed}_warmstart_only.pth",
                        WarmStartConfigV2(seed=seed),
                    )
                    ws_rows = eval_rows(seed, mode_name, "warmstart", ws_model, test_arts)
                    for r in ws_rows:
                        r["warmstart_train_accuracy"] = train_acc
                    rows.extend(ws_rows)
            else:
                rows.extend(run_one(seed, mode_name, args.episodes, args.workers))

    full = pd.DataFrame(rows)
    full.to_csv(out_dir / "ablation_modes_high_per_window.csv", index=False)

    summary = (
        full.groupby(["forced_mode", "stage"])
        .agg(
            attack_keep_rate_mean=("attack_keep_rate", "mean"),
            attack_keep_rate_std=("attack_keep_rate", "std"),
            normal_disable_rate_mean=("normal_disable_rate", "mean"),
            normal_disable_rate_std=("normal_disable_rate", "std"),
            selection_accuracy_mean=("selection_accuracy", "mean"),
            selection_accuracy_std=("selection_accuracy", "std"),
            greedy_total_reward_mean=("greedy_total_reward", "mean"),
            greedy_total_reward_std=("greedy_total_reward", "std"),
            n_windows=("window_id", "count"),
            mean_rules=("window_id", lambda x: np.nan),
        )
        .reset_index()
    )

    # Add average pool size separately to avoid fighting groupby named agg with
    # two source columns.
    pool_size = full.assign(num_rules=full["n_attack_rules"] + full["n_normal_rules"]).groupby(
        ["forced_mode", "stage"]
    )["num_rules"].mean().reset_index(name="mean_rules")
    summary = summary.drop(columns=["mean_rules"]).merge(pool_size, on=["forced_mode", "stage"], how="left")
    summary.to_csv(out_dir / "ablation_modes_high_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
