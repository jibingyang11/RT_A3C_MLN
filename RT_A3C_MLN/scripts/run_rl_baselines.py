"""Run the SOTA deep RL baselines comparison (Section 6.6 of the paper).

The script expects :class:`RLTrainer` objects that wrap DQN, A2C,
PPO, SAC and vanilla A3C.  Writing these wrappers is the main
experimental work required to fill the corresponding table in the
paper; the cleanest path is to adapt ``SimpleRuleEnv`` to the
Gymnasium API and rely on the Stable-Baselines3 reference
implementations.

Example::

    python scripts/run_rl_baselines.py \\
        --project-root . \\
        --plan-csv data/stream/swat/formal_mixed_windows_plan.csv \\
        --num-episodes 100
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.pipeline.baseline_compare import _predict_with_model
from src.pipeline.cross_window_experiment import CrossWindowSplitConfig, split_plan_by_attack_bin
from src.pipeline.formal_experiment import EvalConfig, RuleMiningConfig, WarmStartConfig, prepare_window_learning_artifacts, rollout_actions
from src.pipeline.sota_rl_compare import run_sota_rl_compare, save_sota_rl_compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Deep RL baseline comparison (paper Section 6.6).')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--plan-csv', type=Path, required=True)
    parser.add_argument('--train-per-bin', type=int, default=6)
    parser.add_argument('--val-per-bin', type=int, default=2)
    parser.add_argument('--test-per-bin', type=int, default=2)
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--min-support', type=float, default=0.1)
    parser.add_argument('--min-confidence', type=float, default=0.6)
    parser.add_argument('--ours-run-name', type=str, default='a3c_v7')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--ours-wall-time-sec', type=float, default=33.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def _collect_trainers():
    """Return the list of RLTrainer objects to evaluate.

    By default returns the five named 2023-2026 trainer stubs from
    :mod:`src.adapters.rl_family.adapters`. Replace each pair of
    ``train`` / ``predict`` methods with wrappers around the
    authors' reference code (links inside each trainer class)
    before quoting the paper's table.
    """
    from src.adapters.rl_family.adapters import all_rl_family_trainers
    return list(all_rl_family_trainers())


def _first_episode(curve: pd.DataFrame, column: str, target: float) -> float:
    if curve.empty or column not in curve.columns:
        return float('nan')
    hits = curve[curve[column] >= target]
    if hits.empty:
        return float('nan')
    return float(hits.iloc[0]['episode'])


def _append_ours(
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    test_states: list[dict],
    args: argparse.Namespace,
    eval_config: EvalConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = args.project_root / 'outputs' / 'models' / f'{args.ours_run_name}_selected_policy.pth'
    if not model_path.exists():
        model_path = args.project_root / 'outputs' / 'models' / f'{args.ours_run_name}_global_policy.pth'
    if not model_path.exists() or not test_states:
        return summary, curves

    warmstart_config = WarmStartConfig(seed=args.seed, hidden_dim=args.hidden_dim)
    per_win = []
    for state_dict in test_states:
        actions = _predict_with_model(state_dict, model_path, warmstart_config)
        _, metrics = rollout_actions(state_dict, actions, eval_config)
        per_win.append(metrics)
    metrics_df = pd.DataFrame(per_win)

    curve_path = args.project_root / 'outputs' / 'logs' / f'{args.ours_run_name}_training_curve.csv'
    ours_curve = pd.DataFrame()
    if curve_path.exists():
        raw_curve = pd.read_csv(curve_path)
        if {'episode', 'reward', 'acc'}.issubset(raw_curve.columns):
            ours_curve = raw_curve[['episode', 'reward', 'acc']].rename(
                columns={'reward': 'total_reward', 'acc': 'selection_accuracy'}
            )
            ours_curve['method'] = 'RT-A3C-MLN (ours)'

    ours_summary = pd.DataFrame([{
        'method': 'RT-A3C-MLN (ours)',
        'selection_accuracy': float(metrics_df['selection_accuracy'].mean()),
        'greedy_total_reward': float(metrics_df['greedy_total_reward'].mean()),
        'wall_time_sec': float(args.ours_wall_time_sec),
        'episodes_to_90pct': _first_episode(ours_curve, 'selection_accuracy', 0.90),
        'episodes_to_95pct': _first_episode(ours_curve, 'selection_accuracy', 0.95),
    }])
    summary = pd.concat([summary, ours_summary], ignore_index=True)
    if not ours_curve.empty:
        curves = pd.concat([curves, ours_curve], ignore_index=True)
    return summary, curves


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
    eval_config = EvalConfig(seed=args.seed)

    def _collect(ids):
        arts = []
        for wid in ids:
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
                        arts.append(pickle.load(f))
                else:
                    art = prepare_window_learning_artifacts(args.project_root, int(wid), 'windows_mixed.csv', mining_config)
                    arts.append(art['state_dict'])
            except Exception as exc:
                print(f'[warn] skipping window {wid}: {exc}')
        return arts

    train_states = _collect(splits['train'])
    test_states = _collect(splits['test'])

    trainers = _collect_trainers()
    if not trainers:
        print('[info] no RL trainers registered. Edit _collect_trainers() in this file '
              'to add DQN/A2C/PPO/SAC wrappers before reporting the table.')
        return

    summary, curves = run_sota_rl_compare(
        trainers=trainers,
        train_state_dicts=train_states,
        test_state_dicts=test_states,
        num_episodes=args.num_episodes,
        eval_config=eval_config,
    )
    summary, curves = _append_ours(summary, curves, test_states, args, eval_config)
    summary_path, curves_path = save_sota_rl_compare(summary, curves, args.project_root)
    print(summary)
    print(f'[saved] {summary_path}')
    print(f'[saved] {curves_path}')


if __name__ == '__main__':
    main()
