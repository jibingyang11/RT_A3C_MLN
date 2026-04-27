from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.window_experiment import (
    EvalConfig,
    RuleMiningConfig,
    WarmStartConfig,
    run_multi_window_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='运行更正式的多窗口 mixed-window 实验。')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--window-ids', type=int, nargs='*', default=None, help='直接指定窗口 id 列表')
    parser.add_argument('--plan-csv', type=Path, default=None, help='若提供，则从计划表读取 window_id')
    parser.add_argument('--window-source', type=str, default='windows_mixed.csv')
    parser.add_argument('--summary-name', type=str, default='formal_mixed_windows_summary.csv')
    parser.add_argument('--min-support', type=float, default=0.1)
    parser.add_argument('--min-confidence', type=float, default=0.6)
    parser.add_argument('--top-attack-rules', type=int, default=10)
    parser.add_argument('--top-normal-rules', type=int, default=10)
    parser.add_argument('--relaxed-normal-rules', type=int, default=3)
    parser.add_argument('--warmstart-epochs', type=int, default=200)
    parser.add_argument('--warmstart-lr', type=float, default=1e-3)
    parser.add_argument('--random-trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def resolve_window_ids(args: argparse.Namespace) -> list[int]:
    if args.window_ids:
        return [int(x) for x in args.window_ids]
    if args.plan_csv is not None:
        df = pd.read_csv(args.plan_csv)
        if 'window_id' not in df.columns:
            raise ValueError(f'{args.plan_csv} 缺少 window_id 列')
        return [int(x) for x in df['window_id'].tolist()]
    raise ValueError('必须提供 --window-ids 或 --plan-csv')


def main() -> None:
    args = parse_args()
    window_ids = resolve_window_ids(args)

    mining_config = RuleMiningConfig(
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        top_attack_rules=args.top_attack_rules,
        top_normal_rules=args.top_normal_rules,
        relaxed_normal_rules=args.relaxed_normal_rules,
    )
    warmstart_config = WarmStartConfig(
        epochs=args.warmstart_epochs,
        lr=args.warmstart_lr,
        seed=args.seed,
    )
    eval_config = EvalConfig(
        random_trials=args.random_trials,
        seed=args.seed,
    )

    summary_df = run_multi_window_experiment(
        project_root=args.project_root,
        window_ids=window_ids,
        window_source=args.window_source,
        mining_config=mining_config,
        warmstart_config=warmstart_config,
        eval_config=eval_config,
        summary_name=args.summary_name,
    )
    print(summary_df)
    print('mean selection_accuracy =', summary_df['selection_accuracy'].mean())
    print('mean attack_keep_rate =', summary_df['attack_keep_rate'].mean())
    print('mean normal_disable_rate =', summary_df['normal_disable_rate'].mean())


if __name__ == '__main__':
    main()
