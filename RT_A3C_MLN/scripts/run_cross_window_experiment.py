from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.cross_window_experiment import (
    CrossWindowSplitConfig,
    run_cross_window_experiment,
)
from src.pipeline.formal_experiment import EvalConfig, RuleMiningConfig, WarmStartConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='运行跨窗口 train/test 的更严格 warm-start 实验。')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--plan-csv', type=Path, required=True)
    parser.add_argument('--train-per-bin', type=int, default=6)
    parser.add_argument('--val-per-bin', type=int, default=2)
    parser.add_argument('--test-per-bin', type=int, default=2)
    parser.add_argument('--min-support', type=float, default=0.1)
    parser.add_argument('--min-confidence', type=float, default=0.6)
    parser.add_argument('--top-attack-rules', type=int, default=10)
    parser.add_argument('--top-normal-rules', type=int, default=10)
    parser.add_argument('--relaxed-normal-rules', type=int, default=3)
    parser.add_argument('--low-attack-threshold', type=float, default=0.30)
    parser.add_argument('--high-attack-threshold', type=float, default=0.70)
    parser.add_argument('--warmstart-epochs', type=int, default=200)
    parser.add_argument('--warmstart-lr', type=float, default=1e-3)
    parser.add_argument('--random-trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_df = pd.read_csv(args.plan_csv)
    split_config = CrossWindowSplitConfig(
        train_per_bin=args.train_per_bin,
        val_per_bin=args.val_per_bin,
        test_per_bin=args.test_per_bin,
        random_state=args.seed,
    )
    mining_config = RuleMiningConfig(
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        top_attack_rules=args.top_attack_rules,
        top_normal_rules=args.top_normal_rules,
        relaxed_normal_rules=args.relaxed_normal_rules,
        low_attack_ratio_threshold=args.low_attack_threshold,
        high_attack_ratio_threshold=args.high_attack_threshold,
    )
    warmstart_config = WarmStartConfig(epochs=args.warmstart_epochs, lr=args.warmstart_lr, seed=args.seed)
    eval_config = EvalConfig(random_trials=args.random_trials, seed=args.seed)

    summary_df, splits, train_acc = run_cross_window_experiment(
        project_root=args.project_root,
        plan_df=plan_df,
        split_config=split_config,
        mining_config=mining_config,
        warmstart_config=warmstart_config,
        eval_config=eval_config,
    )
    print('splits =', splits)
    print('global_train_accuracy =', train_acc)
    print(summary_df)


if __name__ == '__main__':
    main()
