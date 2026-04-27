"""Run the adaptive rule-pool ablation and write the summary CSV."""
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
import torch

from src.pipeline.ablation import _MODE_CONFIGS, summarize_mode_ablation
from src.pipeline.formal_experiment import EvalConfig, RuleMiningConfig, build_rule_pool_robust, rollout_actions
from src.rl.ac_model import ActorCriticNet
from src.rl.simple_env import SimpleRuleEnv
from src.rl.state_utils import build_initial_rule_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Ablation study for adaptive rule-pool modes.')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--plan-csv', type=Path, required=True)
    parser.add_argument('--min-support', type=float, default=0.1)
    parser.add_argument('--min-confidence', type=float, default=0.6)
    parser.add_argument('--run-name', type=str, default='a3c_v7')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def _attack_bin(ratio: float) -> str:
    if ratio < 0.30:
        return 'low'
    if ratio >= 0.70:
        return 'high'
    return 'mid'


def _predict_actions(model: ActorCriticNet, state_dict: dict, eval_config: EvalConfig) -> list[int]:
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


def _load_adaptive_state(project_root: Path, window_id: int) -> dict:
    path = project_root / 'data' / 'stream' / 'swat' / 'window_rule_states' / f'window_{int(window_id)}_rule_state.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def _build_forced_state(project_root: Path, window_id: int, attack_ratio: float, mining_config: RuleMiningConfig) -> tuple[dict, str]:
    rules_path = project_root / 'data' / 'stream' / 'swat' / 'window_label_rules' / f'window_{int(window_id)}_label_rules.csv'
    label_rules = pd.read_csv(rules_path)
    pool, mode = build_rule_pool_robust(label_rules, label_rules, attack_ratio, mining_config)
    return build_initial_rule_state(pool), mode


def run_cached_policy_ablation(
    project_root: Path,
    plan_df: pd.DataFrame,
    mining_config: RuleMiningConfig,
    eval_config: EvalConfig,
    run_name: str,
    hidden_dim: int,
) -> pd.DataFrame:
    model_path = project_root / 'outputs' / 'models' / f'{run_name}_selected_policy.pth'
    if not model_path.exists():
        model_path = project_root / 'outputs' / 'models' / f'{run_name}_global_policy.pth'
    model = ActorCriticNet(hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    rows: list[dict] = []
    for cfg_name, overrides in _MODE_CONFIGS.items():
        mining = RuleMiningConfig(
            lag=mining_config.lag,
            min_support=mining_config.min_support,
            min_confidence=mining_config.min_confidence,
            top_attack_rules=mining_config.top_attack_rules,
            top_normal_rules=mining_config.top_normal_rules,
            relaxed_normal_rules=mining_config.relaxed_normal_rules,
            attack_weight_cap=mining_config.attack_weight_cap,
            relaxed_normal_min_weight=mining_config.relaxed_normal_min_weight,
            relaxed_normal_weight_scale=mining_config.relaxed_normal_weight_scale,
            relaxed_normal_weight_cap=mining_config.relaxed_normal_weight_cap,
            low_attack_ratio_threshold=overrides['low_attack_ratio_threshold'],
            high_attack_ratio_threshold=overrides['high_attack_ratio_threshold'],
        )
        for _, plan_row in plan_df.iterrows():
            win_id = int(plan_row['window_id'])
            attack_ratio = float(plan_row['attack_ratio'])
            try:
                if cfg_name == 'adaptive':
                    state_dict = _load_adaptive_state(project_root, win_id)
                    window_mode = 'adaptive'
                else:
                    state_dict, window_mode = _build_forced_state(project_root, win_id, attack_ratio, mining)
                actions = _predict_actions(model, state_dict, eval_config)
                _, metrics = rollout_actions(state_dict, actions, eval_config)
                row = dict(metrics)
            except Exception as exc:
                row = {
                    'num_rules': 0,
                    'num_attack_rules': 0,
                    'num_normal_rules': 0,
                    'attack_keep_rate': 0.0,
                    'normal_disable_rate': 0.0,
                    'selection_accuracy': 0.0,
                    'greedy_total_reward': 0.0,
                    'error_message': str(exc),
                }
                window_mode = 'error'
            row.update({
                'config': cfg_name,
                'window_id': win_id,
                'attack_ratio': attack_ratio,
                'attack_bin': _attack_bin(attack_ratio),
                'window_mode': window_mode,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    plan_df = pd.read_csv(args.plan_csv)
    window_ids = plan_df['window_id'].astype(int).tolist()

    mining_config = RuleMiningConfig(min_support=args.min_support, min_confidence=args.min_confidence)
    eval_config = EvalConfig(seed=args.seed)

    raw = run_cached_policy_ablation(
        project_root=args.project_root,
        plan_df=plan_df,
        mining_config=mining_config,
        eval_config=eval_config,
        run_name=args.run_name,
        hidden_dim=args.hidden_dim,
    )

    summary = summarize_mode_ablation(raw)
    logs_dir = args.project_root / 'outputs' / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(logs_dir / 'ablation_modes_raw.csv', index=False)
    summary.to_csv(logs_dir / 'ablation_modes.csv', index=False)
    print(summary)


if __name__ == '__main__':
    main()
