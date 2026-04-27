"""Baseline comparison for the rule-selection task.

Compares the proposed A3C rule selector against the following
reference baselines on an identical evaluation protocol:

* ``all_keep``      -- keep every candidate rule
* ``all_disable``   -- disable every candidate rule
* ``random``        -- sample Bernoulli(0.5) keep/disable actions
* ``confidence``    -- keep rules whose confidence is above a
                       threshold (logic-only baseline)
* ``warmstart``     -- supervised warm-start classifier
                       (Actor-Critic initialised by cross-entropy)
* ``a3c`` / ``warmstart_a3c`` -- the main method

All baselines are evaluated on the same ``state_dict`` objects
produced by :func:`src.rl.state_utils.build_initial_rule_state`, so
that differences are purely attributable to the action-selection
policy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from src.rl.ac_model import ActorCriticNet
from src.pipeline.formal_experiment import (
    EvalConfig,
    WarmStartConfig,
    get_logits,
    rollout_actions,
)


def _predict_with_model(state_dict: dict, model_path: Path, config: WarmStartConfig) -> list[int]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorCriticNet(state_dim=config.state_dim, action_dim=config.action_dim, hidden_dim=config.hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    from src.rl.simple_env import SimpleRuleEnv
    env = SimpleRuleEnv(state_dict, step_size=0.5, max_steps=int(state_dict['num_rules']))
    state = env.reset()

    actions: list[int] = []
    for rule_idx in range(int(state_dict['num_rules'])):
        x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = get_logits(model(x))
            action = int(torch.argmax(logits, dim=1).item())
        actions.append(action)
        state, _, _, _ = env.step(rule_idx, action)
    return actions


def _confidence_threshold_policy(state_dict: dict, threshold: float = 0.7) -> list[int]:
    """Keep rules whose confidence score is above ``threshold``, else disable.

    This is a purely symbolic baseline that does not use any learning.
    """
    rule_scores = np.asarray(state_dict['rule_scores'], dtype=float)
    actions = [0 if s >= threshold else 1 for s in rule_scores]
    return actions


def compare_baselines(
    state_dicts: Sequence[dict],
    warmstart_model_path: Path | None = None,
    a3c_model_path: Path | None = None,
    warmstart_config: WarmStartConfig | None = None,
    eval_config: EvalConfig | None = None,
    confidence_threshold: float = 0.7,
) -> pd.DataFrame:
    """Run every baseline on the given state dicts and return a summary table."""
    eval_config = eval_config or EvalConfig()
    warmstart_config = warmstart_config or WarmStartConfig()

    def _run_policy(state_dict: dict, method: str, actions: list[int]) -> dict:
        _, metrics = rollout_actions(state_dict, actions, eval_config)
        return {'method': method, **metrics}

    rng = np.random.default_rng(eval_config.seed)
    rows: list[dict] = []
    for state_dict in state_dicts:
        num_rules = int(state_dict['num_rules'])

        rows.append(_run_policy(state_dict, 'all_keep', [0] * num_rules))
        rows.append(_run_policy(state_dict, 'all_disable', [1] * num_rules))
        rows.append(_run_policy(state_dict, 'random', rng.integers(0, 2, size=num_rules).tolist()))
        rows.append(_run_policy(state_dict, 'confidence', _confidence_threshold_policy(state_dict, confidence_threshold)))

        if warmstart_model_path is not None and Path(warmstart_model_path).exists():
            actions = _predict_with_model(state_dict, Path(warmstart_model_path), warmstart_config)
            rows.append(_run_policy(state_dict, 'warmstart', actions))
        if a3c_model_path is not None and Path(a3c_model_path).exists():
            actions = _predict_with_model(state_dict, Path(a3c_model_path), warmstart_config)
            rows.append(_run_policy(state_dict, 'warmstart_a3c', actions))

    if not rows:
        return pd.DataFrame()

    raw_df = pd.DataFrame(rows)
    summary = raw_df.groupby('method').agg(
        attack_keep_rate=('attack_keep_rate', 'mean'),
        normal_disable_rate=('normal_disable_rate', 'mean'),
        selection_accuracy=('selection_accuracy', 'mean'),
        greedy_total_reward=('greedy_total_reward', 'mean'),
        n_windows=('selection_accuracy', 'count'),
    ).reset_index()

    # stable ordering for the paper
    order = ['all_keep', 'all_disable', 'random', 'confidence', 'warmstart', 'warmstart_a3c']
    summary['__ord'] = summary['method'].map({m: i for i, m in enumerate(order)}).fillna(99)
    summary = summary.sort_values('__ord').drop(columns='__ord').reset_index(drop=True)

    return summary
