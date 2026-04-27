from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.formal_experiment import (
    EvalConfig,
    RuleMiningConfig,
    WarmStartConfig,
    build_clean_summary,
    build_group_stats,
    run_batch_windows,
    save_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='运行严格版多窗口 formal mixed-window 实验。')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--window-ids', type=int, nargs='*', default=None)
    parser.add_argument('--plan-csv', type=Path, default=None)
    parser.add_argument('--window-source', type=str, default='windows_mixed.csv')
    parser.add_argument('--summary-name', type=str, default='formal_mixed_windows_summary.csv')
    parser.add_argument('--fail-name', type=str, default='formal_mixed_windows_failures.csv')
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
    project_root = args.project_root
    window_ids = resolve_window_ids(args)

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

    summary_df, fail_df = run_batch_windows(
        project_root=project_root,
        window_ids=window_ids,
        window_source=args.window_source,
        mining_config=mining_config,
        warmstart_config=warmstart_config,
        eval_config=eval_config,
        continue_on_error=True,
    )

    logs_dir = project_root / 'outputs' / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = logs_dir / args.summary_name
    fail_path = logs_dir / args.fail_name

    if not summary_df.empty:
        summary_df = summary_df.sort_values('attack_ratio').reset_index(drop=True)
        save_dataframe(summary_df, summary_path)
        clean_df = build_clean_summary(
            summary_df,
            low_attack_max=args.low_attack_threshold,
            high_attack_min=args.high_attack_threshold,
        )
        save_dataframe(clean_df, logs_dir / 'formal_mixed_windows_summary_clean.csv')
        build_group_stats(clean_df).to_csv(logs_dir / 'formal_mixed_windows_group_stats.csv')
        print(summary_df)
        print('mean selection_accuracy =', summary_df['selection_accuracy'].mean())
        print('mean attack_keep_rate =', summary_df['attack_keep_rate'].mean())
        print('mean normal_disable_rate =', summary_df['normal_disable_rate'].mean())
    else:
        print('没有成功窗口，summary_df 为空。')

    if not fail_df.empty:
        save_dataframe(fail_df, fail_path)
        print('\nfailures:')
        print(fail_df)
    else:
        print('没有失败窗口。')


if __name__ == '__main__':
    main()
