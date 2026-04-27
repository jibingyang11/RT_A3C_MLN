"""Run the SOTA MLN structure-learner comparison (Section 6.4 of the paper).

This is a stub driver: by default it uses the placeholder adapters in
:mod:`src.pipeline.sota_mln_compare` that return the oracle argmax
policy, so the end-to-end plumbing can be tested without installing
any external tool.  Before quoting the paper table, replace each
placeholder adapter with a wrapper around the real competitor (see
the docstring of :mod:`src.pipeline.sota_mln_compare`).

Example::

    python scripts/run_sota_mln_compare.py \\
        --project-root . \\
        --plan-csv data/stream/swat/formal_mixed_windows_plan.csv \\
        --train-per-bin 6 --val-per-bin 2 --test-per-bin 2
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.adapters.mln_family.adapters import all_mln_family_adapters
from src.pipeline.cross_window_experiment import CrossWindowSplitConfig, split_plan_by_attack_bin
from src.pipeline.formal_experiment import EvalConfig, RuleMiningConfig, WarmStartConfig, prepare_window_learning_artifacts, rollout_actions
from src.pipeline.sota_mln_compare import run_sota_mln_compare, save_sota_mln_compare
from src.rl.ac_model import ActorCriticNet
from src.rl.simple_env import SimpleRuleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='MLN structure-learner comparison (paper Section 6.4).')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--plan-csv', type=Path, required=True)
    parser.add_argument('--train-per-bin', type=int, default=6)
    parser.add_argument('--val-per-bin', type=int, default=2)
    parser.add_argument('--test-per-bin', type=int, default=2)
    parser.add_argument('--min-support', type=float, default=0.1)
    parser.add_argument('--min-confidence', type=float, default=0.6)
    parser.add_argument('--ours-run-name', type=str, default='a3c_v7')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def _append_ours(summary: pd.DataFrame, state_dicts: list[dict], args: argparse.Namespace, eval_config: EvalConfig) -> pd.DataFrame:
    model_path = args.project_root / 'outputs' / 'models' / f'{args.ours_run_name}_selected_policy.pth'
    if not model_path.exists():
        model_path = args.project_root / 'outputs' / 'models' / f'{args.ours_run_name}_global_policy.pth'
    if not model_path.exists() or not state_dicts:
        return summary

    warmstart_config = WarmStartConfig(seed=args.seed, hidden_dim=args.hidden_dim)
    model = ActorCriticNet(
        state_dim=warmstart_config.state_dim,
        action_dim=warmstart_config.action_dim,
        hidden_dim=warmstart_config.hidden_dim,
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    def _predict(state_dict: dict) -> list[int]:
        env = SimpleRuleEnv(state_dict, step_size=eval_config.env_step_size, max_steps=int(state_dict['num_rules']) + 1)
        state = env.reset()
        actions: list[int] = []
        done = False
        while not done and env.current_rule_idx < int(state_dict['num_rules']):
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(model(x)[0], dim=1).item())
            actions.append(action)
            state, _, done, _ = env.step(action)
        return actions

    per_win = []
    inference_ms = []
    for state_dict in state_dicts:
        t0 = time.perf_counter()
        actions = _predict(state_dict)
        inference_ms.append((time.perf_counter() - t0) * 1000.0)
        _, metrics = rollout_actions(state_dict, actions, eval_config)
        per_win.append(metrics)
    metrics_df = pd.DataFrame(per_win)
    ours = pd.DataFrame([{
        'method': 'RT-A3C-MLN (ours)',
        'selection_accuracy': float(metrics_df['selection_accuracy'].mean()),
        'attack_keep_rate': float(metrics_df['attack_keep_rate'].mean()),
        'normal_disable_rate': float(metrics_df['normal_disable_rate'].mean()),
        'greedy_total_reward': float(metrics_df['greedy_total_reward'].mean()),
        'inference_ms_per_window': float(np.mean(inference_ms)),
        'num_windows': int(len(state_dicts)),
    }])
    return pd.concat([summary, ours], axis=0, ignore_index=True)


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

    summary = run_sota_mln_compare(
        state_dicts=state_dicts,
        eval_config=eval_config,
        adapters=list(all_mln_family_adapters()),
    )
    summary = _append_ours(summary, state_dicts, args, eval_config)
    out_path = save_sota_mln_compare(summary, args.project_root)
    print(summary)
    print(f'[saved] {out_path}')


if __name__ == '__main__':
    main()
