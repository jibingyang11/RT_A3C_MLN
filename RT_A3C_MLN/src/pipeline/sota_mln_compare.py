"""Stub driver for the MLN structure-learner comparison (Section 6.4 of the paper).

This file intentionally contains *only* the interface and the wiring
logic for the comparison experiment. Each competitor needs a small
adapter class that exposes:

    class CompetitorAdapter:
        name: str                                      # row name in the paper table
        def fit(self, state_dict: dict) -> None: ...   # trains on the window's rule pool
        def predict(self, state_dict: dict) -> list[int]:
            '''Return a keep/disable action per candidate rule
            (0 = keep, 1 = disable), identical to SimpleRuleEnv.'''

Reference implementations that can be wrapped:

- MLN (Richardson & Domingos, 2006):
    Alchemy 2.0 -- https://alchemy.cs.washington.edu/
- ILS-MLN (Biba et al., 2008):
    Alchemy 2.0 with ``-ILS`` structure-learning flag.
- Boosted MLN / TreeBoostler (Natarajan et al., 2014; Khot, 2011):
    https://github.com/boost-starai/BoostSRL
- Neural MLN (Marra & Kuzelka, UAI 2021):
    https://github.com/giuseppemarra/NeuralMLN
- Quantified Neural MLN (Jung et al., IJAR 2024):
    supplementary code of the IJAR 2024 paper.
- MLN4KB (Fang et al., WWW 2023):
    https://github.com/fh-wangcheng/MLN4KB

For every adapter:

1. Convert the ``state_dict['rules']`` list into the format the
   external tool expects (this is usually a list of first-order
   clauses plus a label predicate per grounding).
2. Call ``fit`` on the current window's rule pool.
3. In ``predict``, return a 0/1 action per rule following the
   Environment convention in :mod:`src.rl.simple_env`.

The driver below runs every registered adapter on the same set of
state dicts and collects a DataFrame matching the columns of
Table 4 in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence
import time

import numpy as np
import pandas as pd

from src.pipeline.formal_experiment import EvalConfig, rollout_actions


class MLNAdapter(Protocol):
    """Protocol that every MLN competitor adapter must satisfy."""

    name: str

    def fit(self, state_dict: dict) -> None:
        ...

    def predict(self, state_dict: dict) -> list[int]:
        ...


@dataclass
class _NullAdapter:
    """Fallback adapter used until the real wrappers are installed.

    Returns the logit-style argmax baseline (``keep`` iff ``y_i == 1``
    in the warm-start labels) so that ``run_sota_mln_compare`` can be
    end-to-end-tested without any external dependency.  Replace this
    class with a real wrapper for each competitor before quoting the
    table in the paper.
    """

    name: str

    def fit(self, state_dict: dict) -> None:
        return None

    def predict(self, state_dict: dict) -> list[int]:
        labels = np.asarray(state_dict['target_labels'], dtype=int)
        # keep when label is attack, disable otherwise
        return [0 if int(y) == 1 else 1 for y in labels]


def _register_placeholder_adapters() -> list[MLNAdapter]:
    """Return a list of placeholder adapters with the paper's row names."""
    return [
        _NullAdapter(name='MLN'),
        _NullAdapter(name='ILS-MLN'),
        _NullAdapter(name='BoostedMLN'),
        _NullAdapter(name='NeuralMLN'),
        _NullAdapter(name='MLN4KB'),
        _NullAdapter(name='QuantifiedNeuralMLN'),
    ]


def run_sota_mln_compare(
    state_dicts: Sequence[dict],
    eval_config: EvalConfig | None = None,
    adapters: Sequence[MLNAdapter] | None = None,
) -> pd.DataFrame:
    """Run every registered MLN competitor on the given state dicts.

    Parameters
    ----------
    state_dicts:
        Test-window states produced by
        :func:`src.rl.state_utils.build_initial_rule_state`.
    eval_config:
        Shared evaluation configuration.
    adapters:
        Explicit list of adapters.  When ``None`` the placeholder
        adapters in :func:`_register_placeholder_adapters` are used.

    Returns
    -------
    DataFrame
        One row per competitor with the columns used by the paper's
        Table 4: ``method, selection_accuracy, attack_keep_rate,
        normal_disable_rate, inference_ms_per_window, num_rules_avg``.
    """
    eval_config = eval_config or EvalConfig()
    adapters = list(adapters or _register_placeholder_adapters())

    rows: list[dict] = []
    for adapter in adapters:
        per_win_metrics = []
        inference_ms = []
        for state_dict in state_dicts:
            adapter.fit(state_dict)
            t0 = time.perf_counter()
            actions = adapter.predict(state_dict)
            inference_ms.append((time.perf_counter() - t0) * 1000.0)
            _, metrics = rollout_actions(state_dict, actions, eval_config)
            per_win_metrics.append(metrics)
        if not per_win_metrics:
            continue
        metrics_df = pd.DataFrame(per_win_metrics)
        rows.append({
            'method': adapter.name,
            'selection_accuracy': float(metrics_df['selection_accuracy'].mean()),
            'attack_keep_rate': float(metrics_df['attack_keep_rate'].mean()),
            'normal_disable_rate': float(metrics_df['normal_disable_rate'].mean()),
            'greedy_total_reward': float(metrics_df['greedy_total_reward'].mean()),
            'inference_ms_per_window': float(np.mean(inference_ms)),
            'num_windows': int(len(state_dicts)),
        })

    return pd.DataFrame(rows)


def save_sota_mln_compare(df: pd.DataFrame, project_root: Path) -> Path:
    out = project_root / 'outputs' / 'logs' / 'sota_mln_compare.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
