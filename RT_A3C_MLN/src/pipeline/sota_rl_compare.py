"""Stub driver for the deep RL baselines comparison (Section 6.6).

This module is the RL analogue of :mod:`src.pipeline.sota_mln_compare`.
It wraps four additional RL algorithms around the same rule-selection
MDP defined in :mod:`src.rl.simple_env`:

* DQN   (Mnih et al., 2015)
* A2C   (synchronous actor-critic variant of A3C)
* PPO   (Schulman et al., 2017)
* SAC   (Haarnoja et al., 2018) with discrete-action adaptation

plus ``VanillaA3C`` (the A3C trainer without warm-start and without the
adaptive rule pool) so that the paper can separate the gains of the
A3C algorithm itself from the gains of warm-start + adaptive pool.

The recommended way to implement each trainer is to wrap the
Stable-Baselines3 or CleanRL reference implementation: instead of
re-deriving the algorithm, adapt ``SimpleRuleEnv`` to the Gymnasium
API (a thin wrapper is sufficient), instantiate SB3's
``DQN / A2C / PPO / SAC`` policy, train for a fixed number of
timesteps, and evaluate greedily.

The driver collects both final accuracy/reward and training-curve
data so that Figure 8 of the paper can be produced from the same
CSV.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence
import time

import numpy as np
import pandas as pd


class RLTrainer(Protocol):
    name: str

    def train(
        self,
        train_state_dicts: Sequence[dict],
        num_episodes: int,
    ) -> pd.DataFrame:
        """Train the trainer and return a training-curve DataFrame
        with columns ``episode, total_reward, selection_accuracy``."""
        ...

    def predict(self, state_dict: dict) -> list[int]:
        ...


def run_sota_rl_compare(
    trainers: Sequence[RLTrainer],
    train_state_dicts: Sequence[dict],
    test_state_dicts: Sequence[dict],
    num_episodes: int = 100,
    eval_config=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run each trainer on the common train/test split.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per trainer with columns ``method, selection_accuracy,
        greedy_total_reward, wall_time_sec, episodes_to_90pct,
        episodes_to_95pct``.
    curves_df : pd.DataFrame
        Long-form training curves for every trainer suitable for the
        training-efficiency plot.
    """
    from src.pipeline.formal_experiment import EvalConfig, rollout_actions
    eval_config = eval_config or EvalConfig()

    summary_rows = []
    curves_list = []

    for trainer in trainers:
        t0 = time.perf_counter()
        curve = trainer.train(train_state_dicts, num_episodes=num_episodes)
        wall_time = time.perf_counter() - t0
        curve = curve.copy()
        curve['method'] = trainer.name
        curves_list.append(curve)

        eps_to_09 = _first_episode_reaching(curve, 'selection_accuracy', 0.9)
        eps_to_095 = _first_episode_reaching(curve, 'selection_accuracy', 0.95)

        # greedy test evaluation
        per_win = []
        for state_dict in test_state_dicts:
            actions = trainer.predict(state_dict)
            _, metrics = rollout_actions(state_dict, actions, eval_config)
            per_win.append(metrics)
        per_win_df = pd.DataFrame(per_win)

        summary_rows.append({
            'method': trainer.name,
            'selection_accuracy': float(per_win_df['selection_accuracy'].mean()) if not per_win_df.empty else float('nan'),
            'greedy_total_reward': float(per_win_df['greedy_total_reward'].mean()) if not per_win_df.empty else float('nan'),
            'wall_time_sec': float(wall_time),
            'episodes_to_90pct': eps_to_09,
            'episodes_to_95pct': eps_to_095,
        })

    summary_df = pd.DataFrame(summary_rows)
    curves_df = pd.concat(curves_list, axis=0, ignore_index=True) if curves_list else pd.DataFrame()
    return summary_df, curves_df


def _first_episode_reaching(curve: pd.DataFrame, column: str, target: float) -> float:
    smooth = curve.groupby('episode')[column].mean().rolling(5, min_periods=1).mean()
    hits = smooth[smooth >= target]
    return float(hits.index[0]) if not hits.empty else float('nan')


def save_sota_rl_compare(summary: pd.DataFrame, curves: pd.DataFrame, project_root: Path) -> tuple[Path, Path]:
    logs = project_root / 'outputs' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    summary_path = logs / 'sota_rl_compare.csv'
    curves_path = logs / 'sota_rl_compare_curves.csv'
    summary.to_csv(summary_path, index=False)
    curves.to_csv(curves_path, index=False)
    return summary_path, curves_path
