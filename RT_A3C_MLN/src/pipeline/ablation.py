"""Ablation study on the adaptive rule-pool construction.

This module evaluates four configurations of the rule-pool builder:

* ``balanced_only``  -- force every window into ``balanced`` mode
* ``standard_only``  -- force every window into ``standard`` mode
* ``relaxed_only``   -- force every window into ``relaxed`` mode
* ``adaptive``       -- the full proposal (mode chosen by attack ratio)

For each configuration we re-build the rule pool, retrain the
warm-start policy, and report the four primary metrics aggregated by
attack-ratio bin.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.pipeline.formal_experiment import (
    EvalConfig,
    RuleMiningConfig,
    WarmStartConfig,
    evaluate_baselines,
    greedy_evaluate,
    prepare_window_learning_artifacts,
    train_warmstart_model,
)


_MODE_CONFIGS = {
    'balanced_only': dict(low_attack_ratio_threshold=1.0,  high_attack_ratio_threshold=1.01),
    'standard_only': dict(low_attack_ratio_threshold=-0.01, high_attack_ratio_threshold=1.01),
    'relaxed_only':  dict(low_attack_ratio_threshold=-0.01, high_attack_ratio_threshold=-0.005),
    'adaptive':      dict(low_attack_ratio_threshold=0.30, high_attack_ratio_threshold=0.70),
}


def _attack_bin(ratio: float) -> str:
    if ratio < 0.30:
        return 'low'
    if ratio >= 0.70:
        return 'high'
    return 'mid'


def run_mode_ablation(
    project_root: Path,
    window_ids: Sequence[int],
    window_source: str = 'windows_mixed.csv',
    mining_config: RuleMiningConfig | None = None,
    warmstart_config: WarmStartConfig | None = None,
    eval_config: EvalConfig | None = None,
) -> pd.DataFrame:
    """Run the four mode configurations over ``window_ids`` and return a long-form table."""
    base_mining = mining_config or RuleMiningConfig()
    warmstart_config = warmstart_config or WarmStartConfig()
    eval_config = eval_config or EvalConfig()

    rows: list[dict] = []
    for cfg_name, overrides in _MODE_CONFIGS.items():
        mining = replace(base_mining, **overrides)
        for win_id in window_ids:
            try:
                art = prepare_window_learning_artifacts(project_root, int(win_id), window_source, mining)
            except Exception as exc:
                rows.append({
                    'config': cfg_name,
                    'window_id': int(win_id),
                    'attack_ratio': float('nan'),
                    'attack_bin': 'error',
                    'window_mode': 'error',
                    'error_message': str(exc),
                })
                continue

            model_path = project_root / 'outputs' / 'models' / f'ablation_{cfg_name}_window_{win_id}.pth'
            train_acc, _ = train_warmstart_model(art['X_ws'], art['y_ws'], model_path, warmstart_config)
            _, metrics_df = greedy_evaluate(model_path, art['state_dict'], warmstart_config, eval_config)

            row = metrics_df.iloc[0].to_dict()
            row.update({
                'config': cfg_name,
                'window_id': int(win_id),
                'attack_ratio': float(art['attack_ratio']),
                'attack_bin': _attack_bin(float(art['attack_ratio'])),
                'window_mode': art['window_mode'],
                'warmstart_train_accuracy': float(train_acc),
            })
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_mode_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw ablation rows into one row per (config, bin)."""
    ok = df[df['attack_bin'] != 'error'] if 'attack_bin' in df.columns else df
    if ok.empty:
        return ok
    agg = ok.groupby(['config', 'attack_bin']).agg(
        attack_keep_rate=('attack_keep_rate', 'mean'),
        normal_disable_rate=('normal_disable_rate', 'mean'),
        selection_accuracy=('selection_accuracy', 'mean'),
        greedy_total_reward=('greedy_total_reward', 'mean'),
        n_windows=('window_id', 'count'),
    ).reset_index()
    return agg
