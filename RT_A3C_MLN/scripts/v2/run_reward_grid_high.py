"""Small reward-coefficient grid for the high-heldout Protocol B split.

This script is diagnostic: it trains the same warm-start+A3C controller
under several PaperRewardConfig settings and reports held-out rule-selection
metrics.  It does not fabricate or smooth any result; each row is produced by
running the policy on the specified seeds.
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

from scripts.v2.run_protocolB_high_heldout import high_heldout_split
from src.rl.reward_utils_v2 import PaperRewardConfig
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


def make_env_cls(c_attack: float, c_normal: float, c_drift: float):
    cfg = PaperRewardConfig(c_attack=c_attack, c_normal=c_normal, c_drift=c_drift)

    class TunedPaperRuleEnv(PaperRuleEnv):
        def __init__(self, init_state_dict):
            super().__init__(init_state_dict, reward_config=cfg)

    return TunedPaperRuleEnv


def run_one(seed: int, c_attack: float, c_normal: float, c_drift: float, episodes: int, workers: int) -> pd.DataFrame:
    set_global_seed(seed)
    plan = load_formal_plan(PROJECT_ROOT)
    splits = high_heldout_split(plan, seed)
    mining_cfg = MiningConfigV2()

    def prep(ids):
        return [prepare_window_artifacts(PROJECT_ROOT, int(wid), mining_cfg) for wid in ids]

    train_arts = prep(splits["train"])
    test_arts = prep(splits["test"])

    x_train = np.concatenate([a.X_ws for a in train_arts], axis=0)
    y_train = np.concatenate([a.y_ws for a in train_arts], axis=0)
    model_dir = PROJECT_ROOT / "outputs" / "v2" / "models" / "reward_grid"
    model_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ca{c_attack:g}_cn{c_normal:g}_cd{c_drift:g}_seed{seed}".replace(".", "p")

    train_acc, ws_model = warmstart_classifier(
        x_train,
        y_train,
        model_dir / f"{tag}_warmstart.pth",
        WarmStartConfigV2(seed=seed),
    )

    rows = []
    env_cls = make_env_cls(c_attack, c_normal, c_drift)
    for art in test_arts:
        m = greedy_evaluate(ws_model, art.state_dict, env_cls=env_cls)
        rows.append(
            {
                "seed": seed,
                "stage": "warmstart",
                "window_id": art.window_id,
                "attack_ratio": art.attack_ratio,
                "c_attack": c_attack,
                "c_normal": c_normal,
                "c_drift": c_drift,
                "train_accuracy": train_acc,
                "n_attack_rules": art.n_attack_rules,
                "n_normal_rules": art.n_normal_rules,
                **m,
            }
        )

    a3c_model = a3c_fine_tune(
        init_model=ws_model,
        train_states=[a.state_dict for a in train_arts],
        env_cls=env_cls,
        cfg=A3CConfigV2(num_workers=workers, episodes_per_worker=episodes, seed=seed),
    )

    for art in test_arts:
        m = greedy_evaluate(a3c_model, art.state_dict, env_cls=env_cls)
        rows.append(
            {
                "seed": seed,
                "stage": "warmstart_a3c",
                "window_id": art.window_id,
                "attack_ratio": art.attack_ratio,
                "c_attack": c_attack,
                "c_normal": c_normal,
                "c_drift": c_drift,
                "train_accuracy": train_acc,
                "n_attack_rules": art.n_attack_rules,
                "n_normal_rules": art.n_normal_rules,
                **m,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--c-attack", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0])
    parser.add_argument("--c-normal", type=float, nargs="+", default=[2.0, 3.0])
    parser.add_argument("--c-drift", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "outputs" / "logs" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for c_attack in args.c_attack:
        for c_normal in args.c_normal:
            for seed in args.seeds:
                print(f"[grid] seed={seed} c_attack={c_attack} c_normal={c_normal} c_drift={args.c_drift}")
                all_rows.append(run_one(seed, c_attack, c_normal, args.c_drift, args.episodes, args.workers))

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(out_dir / "reward_grid_high_per_window.csv", index=False)
    summary = (
        full.groupby(["c_attack", "c_normal", "c_drift", "stage"])
        .agg(
            attack_keep_rate=("attack_keep_rate", "mean"),
            normal_disable_rate=("normal_disable_rate", "mean"),
            selection_accuracy=("selection_accuracy", "mean"),
            greedy_total_reward=("greedy_total_reward", "mean"),
            n_windows=("window_id", "count"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "reward_grid_high_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
