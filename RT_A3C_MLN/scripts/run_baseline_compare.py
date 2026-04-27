"""Run the baseline-comparison experiment and write the summary CSV.

This script produces the table and figure used in Section 6.3 of the
paper (baselines vs. warm-start vs. A3C).  It must be run *after*
``run_a3c_training.py`` because it relies on the saved warm-start and
A3C policies.

Example::

    python scripts/run_baseline_compare.py \\
        --project-root . \\
        --plan-csv data/stream/swat/formal_mixed_windows_plan.csv \\
        --run-name a3c_run
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.pipeline.baseline_compare import compare_baselines
from src.pipeline.cross_window_experiment import CrossWindowSplitConfig, split_plan_by_attack_bin
from src.pipeline.formal_experiment import (
    EvalConfig,
    RuleMiningConfig,
    WarmStartConfig,
    prepare_window_learning_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run baseline comparison on held-out test windows.')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--plan-csv', type=Path, required=True)
    parser.add_argument('--test-per-bin', type=int, default=2)
    parser.add_argument('--train-per-bin', type=int, default=6)
    parser.add_argument('--val-per-bin', type=int, default=2)
    parser.add_argument('--run-name', type=str, default='a3c_run')
    parser.add_argument('--min-support', type=float, default=0.1)
    parser.add_argument('--min-confidence', type=float, default=0.6)
    parser.add_argument('--confidence-threshold', type=float, default=0.7)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_df = pd.read_csv(args.plan_csv)
    if 'attack_bin' not in plan_df.columns:
        plan_df['attack_bin'] = plan_df['attack_ratio'].apply(
            lambda r: 'low' if r < 0.3 else ('high' if r >= 0.7 else 'mid')
        )

    splits = split_plan_by_attack_bin(
        plan_df,
        CrossWindowSplitConfig(
            train_per_bin=args.train_per_bin,
            val_per_bin=args.val_per_bin,
            test_per_bin=args.test_per_bin,
            random_state=args.seed,
        ),
    )

    mining_config = RuleMiningConfig(min_support=args.min_support, min_confidence=args.min_confidence)
    warmstart_config = WarmStartConfig(seed=args.seed, hidden_dim=args.hidden_dim)
    eval_config = EvalConfig(seed=args.seed)

    state_dicts = []
    for wid in splits['test']:
        try:
            cache_path = (
                args.project_root
                / 'data'
                / 'stream'
                / 'swat'
                / 'window_rule_states'
                / f'window_{int(wid)}_rule_state.pkl'
            )
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    state_dicts.append(pickle.load(f))
            else:
                art = prepare_window_learning_artifacts(args.project_root, int(wid), 'windows_mixed.csv', mining_config)
                state_dicts.append(art['state_dict'])
        except Exception as exc:
            print(f'[warn] skipping window {wid}: {exc}')

    models_dir = args.project_root / 'outputs' / 'models'
    ws_path = models_dir / f'{args.run_name}_warmstart.pth'
    a3c_path = models_dir / f'{args.run_name}_global_policy.pth'

    summary = compare_baselines(
        state_dicts=state_dicts,
        warmstart_model_path=ws_path,
        a3c_model_path=a3c_path,
        warmstart_config=warmstart_config,
        eval_config=eval_config,
        confidence_threshold=args.confidence_threshold,
    )

    logs_dir = args.project_root / 'outputs' / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / 'baseline_compare.csv'
    summary.to_csv(out_path, index=False)
    print(summary)
    print(f'[saved] {out_path}')


if __name__ == '__main__':
    main()
