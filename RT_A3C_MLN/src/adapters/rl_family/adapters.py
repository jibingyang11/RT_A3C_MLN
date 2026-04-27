"""Adapter stubs for the five 2023-2026 RL / drift-aware streaming
methods compared in Table 6 of the paper.

Each class implements the :class:`RLTrainer` protocol from
:mod:`src.pipeline.sota_rl_compare`:

    name: str
    def train(self, train_state_dicts, num_episodes) -> pd.DataFrame: ...
    def predict(self, state_dict: dict) -> list[int]: ...

These are deterministic local proxy trainers.  They intentionally avoid
the oracle ``target_labels`` shortcut at prediction time and use only
rule-quality metadata.  Replace each pair with wrappers around the
authors' reference code before presenting the rows as reproduced
third-party results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


def _features(state_dict: dict) -> dict[str, np.ndarray]:
    pool = list(state_dict.get('rule_pool', []))
    n = int(state_dict.get('num_rules', len(pool)))
    score = np.asarray(state_dict.get('rule_scores', np.ones(n)), dtype=float)
    weight = np.asarray(state_dict.get('weights', np.ones(n)), dtype=float)
    support = np.asarray([float(r.get('support', 0.0)) for r in pool], dtype=float)
    confidence = np.asarray([float(r.get('confidence', s)) for r, s in zip(pool, score)], dtype=float)
    lift = np.asarray([float(r.get('lift', 1.0)) for r in pool], dtype=float)
    ant_len = np.asarray([
        len(str(r.get('antecedent_str', '')).split(' & ')) if str(r.get('antecedent_str', '')) else 0
        for r in pool
    ], dtype=float)
    return {
        'score': np.nan_to_num(score, nan=0.0),
        'weight': np.nan_to_num(weight, nan=0.0),
        'support': np.nan_to_num(support, nan=0.0),
        'confidence': np.nan_to_num(confidence, nan=0.0),
        'lift': np.nan_to_num(lift, nan=1.0),
        'ant_len': np.nan_to_num(ant_len, nan=0.0),
    }


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _top_fraction(values: np.ndarray, fraction: float) -> list[int]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return []
    k = max(1, min(n, int(round(float(fraction) * n))))
    keep_idx = np.argsort(-values)[:k]
    actions = np.ones(n, dtype=int)
    actions[keep_idx] = 0
    return actions.tolist()


def _quality_score(state_dict: dict, variant: str) -> np.ndarray:
    f = _features(state_dict)
    if variant == 'ae':
        return f['weight'] * f['score']
    if variant == 'drl':
        return 0.55 * f['confidence'] + 0.30 * _minmax(f['lift']) + 0.15 * _minmax(f['support'])
    if variant == 'dnn':
        return f['confidence'] * np.log1p(np.maximum(f['lift'], 0.0))
    if variant == 'stream':
        return 0.50 * f['confidence'] + 0.30 * _minmax(f['weight']) + 0.20 * _minmax(f['support'])
    compact = 1.0 / np.maximum(f['ant_len'], 1.0)
    return 0.45 * f['confidence'] + 0.35 * _minmax(f['lift']) + 0.20 * compact


def _curve(num_episodes: int, final_acc: float, final_reward: float) -> pd.DataFrame:
    xs = np.arange(num_episodes, dtype=float)
    progress = 1.0 - np.exp(-xs / max(1.0, num_episodes / 5.0))
    acc = 0.50 + (float(final_acc) - 0.50) * progress
    reward = float(final_reward) * progress
    return pd.DataFrame({
        'episode': list(range(num_episodes)),
        'total_reward': reward.tolist(),
        'selection_accuracy': acc.tolist(),
    })


def _estimate_curve(train_state_dicts, num_episodes: int, predictor) -> pd.DataFrame:
    from src.pipeline.formal_experiment import EvalConfig, rollout_actions
    rows = []
    for state_dict in train_state_dicts:
        _, metrics = rollout_actions(state_dict, predictor(state_dict), EvalConfig())
        rows.append(metrics)
    if not rows:
        return _curve(num_episodes, 0.5, 0.0)
    df = pd.DataFrame(rows)
    return _curve(num_episodes, float(df['selection_accuracy'].mean()), float(df['greedy_total_reward'].mean()))


# ---------------------------------------------------------------------------
# 2023
# ---------------------------------------------------------------------------
@dataclass
class StrAEmDDTrainer:
    """Mallick et al., strAEm++DD, IJCNN 2023.

    Reference implementation:
        https://ieeexplore.ieee.org/document/10191328
    Wrapping notes:
        * Re-purpose the autoencoder as a binary classifier on the
          12-d state vector. Replace the streaming drift detector
          with the per-window reset of our pipeline.
    """
    name: str = 'strAEm++DD'
    year: int = 2023

    def train(self, train_state_dicts, num_episodes: int = 100) -> pd.DataFrame:
        return _estimate_curve(train_state_dicts, num_episodes, self.predict)

    def predict(self, state_dict: dict) -> list[int]:
        return _top_fraction(_quality_score(state_dict, 'ae'), 0.30)


# ---------------------------------------------------------------------------
# 2025
# ---------------------------------------------------------------------------
@dataclass
class DRLNIDSTrainer:
    """Dong et al., DRL-NIDS, Springer 2025.

    Reference implementation:
        https://link.springer.com/chapter/10.1007/978-3-031-94445-1_17
    Wrapping notes:
        * The original deep-RL agent observes raw network features.
          Replace the input head with our 12-d rule-level state.
        * Keep the quality-driven drift detector unchanged; trigger
          it at every window boundary.
    """
    name: str = 'DRL-NIDS'
    year: int = 2025

    def train(self, train_state_dicts, num_episodes: int = 100) -> pd.DataFrame:
        return _estimate_curve(train_state_dicts, num_episodes, self.predict)

    def predict(self, state_dict: dict) -> list[int]:
        return _top_fraction(_quality_score(state_dict, 'drl'), 0.45)


@dataclass
class DNNAEDDTrainer:
    """Wang et al., DNN+AE-DD, Applied Sciences 2025.

    Reference implementation:
        https://www.mdpi.com/2076-3417/15/6/3056 (contact authors).
    Wrapping notes:
        * The autoencoder-based concept-drift monitor decides when to
          retrain. Treat our policy network as the classifier head
          and trigger retraining at every window boundary.
    """
    name: str = 'DNN+AE-DD'
    year: int = 2025

    def train(self, train_state_dicts, num_episodes: int = 100) -> pd.DataFrame:
        return _estimate_curve(train_state_dicts, num_episodes, self.predict)

    def predict(self, state_dict: dict) -> list[int]:
        return _top_fraction(_quality_score(state_dict, 'dnn'), 0.40)


@dataclass
class StreamingADBestTrainer:
    """Cao et al., 'Streaming-AD best', Artif. Intell. Rev. 2025.

    Reference implementation:
        https://link.springer.com/article/10.1007/s10462-024-10995-w
    Wrapping notes:
        * The benchmark identifies the strongest concept-drift-aware
          algorithm via a Critical-Difference diagram. Adopt that
          algorithm and reuse the public benchmark wrapper from
          its accompanying code release.
    """
    name: str = 'Streaming-AD best'
    year: int = 2025

    def train(self, train_state_dicts, num_episodes: int = 100) -> pd.DataFrame:
        return _estimate_curve(train_state_dicts, num_episodes, self.predict)

    def predict(self, state_dict: dict) -> list[int]:
        return _top_fraction(_quality_score(state_dict, 'stream'), 0.50)


# ---------------------------------------------------------------------------
# 2026
# ---------------------------------------------------------------------------
@dataclass
class InterpretableACTrainer:
    """Yang et al., Interpretable-AC, ICLR 2026.

    Reference implementation:
        ICLR 2026 paper -- search OpenReview by title.
    Wrapping notes:
        * Interpretable-AC is closest in spirit to RT-A3C-MLN.
          Replace its symbolic-rule head with our candidate rule
          pool to make the comparison strictly architectural.
    """
    name: str = 'Interpretable-AC'
    year: int = 2026

    def train(self, train_state_dicts, num_episodes: int = 100) -> pd.DataFrame:
        return _estimate_curve(train_state_dicts, num_episodes, self.predict)

    def predict(self, state_dict: dict) -> list[int]:
        return _top_fraction(_quality_score(state_dict, 'iac'), 0.45)


def all_rl_family_trainers() -> Sequence:
    """Return every RL/streaming trainer in the paper's row order."""
    return [
        StrAEmDDTrainer(),
        DRLNIDSTrainer(),
        DNNAEDDTrainer(),
        StreamingADBestTrainer(),
        InterpretableACTrainer(),
    ]
