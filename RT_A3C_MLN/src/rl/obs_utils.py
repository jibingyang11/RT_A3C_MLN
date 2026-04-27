"""Observation builder for the rule-selection MDP.

The state remains 12-D.  The last slot exposes the target class of the
current candidate rule (``1`` for ``... => LABEL_ATTACK``, ``0`` for
``... => LABEL_NORMAL``).  This is rule-pool metadata, not a timestamp
ground-truth label: the selector is deciding whether an already mined
attack-oriented or normal-oriented clause should remain active.
"""
from __future__ import annotations

import numpy as np


_STATE_DIM = 12


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if den > 1e-9 else float(default)


def build_state_vector(
    active_mask: np.ndarray,
    weights: np.ndarray,
    rule_scores: np.ndarray,
    rule_idx: int,
    target_labels: np.ndarray,
) -> np.ndarray:
    """Construct a 12-D observation vector for ``rule_idx``.

    The slots are:

      0 active_ratio                          - global mask density
      1 active_weights_mean (z-normalised)    - global signal
      2 active_weights_std  (z-normalised)    - global signal
      3 active_weights_min  (z-normalised)    - global signal
      4 active_weights_max  (z-normalised)    - global signal
      5 active_count / num_rules              - global signal
      6 normalised_idx                        - position in the pool
      7 current_active                        - current-rule mask
      8 current_weight z-score in the window  - current-rule signal
      9 current_score (already in [0,1])      - current-rule signal
     10 score_rank                            - relative position on score
     11 target_class                          - 1 attack-rule, 0 normal-rule
    """
    active_mask = np.asarray(active_mask, dtype=float)
    weights = np.asarray(weights, dtype=float)
    rule_scores = np.asarray(rule_scores, dtype=float)

    if not 0 <= rule_idx < len(active_mask):
        raise IndexError(f'rule_idx out of range: {rule_idx}')

    n = len(active_mask)
    pop_mean = float(weights.mean())
    pop_std = float(weights.std() + 1e-6)

    active_weights = weights[active_mask > 0.5]
    if len(active_weights) == 0:
        mean_w = std_w = min_w = max_w = 0.0
    else:
        mean_w = (float(active_weights.mean()) - pop_mean) / pop_std
        std_w = float(active_weights.std()) / pop_std
        min_w = (float(active_weights.min()) - pop_mean) / pop_std
        max_w = (float(active_weights.max()) - pop_mean) / pop_std

    current_score = float(rule_scores[rule_idx])
    current_weight_z = (float(weights[rule_idx]) - pop_mean) / pop_std
    current_active = float(active_mask[rule_idx])
    normalized_idx = _safe_div(float(rule_idx), float(max(1, n - 1)))

    score_rank = float((rule_scores <= current_score).mean())
    weight_rank = float((weights <= weights[rule_idx]).mean())
    active_ratio = float(active_mask.mean())

    state = np.array([
        active_ratio,
        mean_w,
        std_w,
        min_w,
        max_w,
        _safe_div(float(active_mask.sum()), float(n)),
        normalized_idx,
        current_active,
        current_weight_z,
        current_score,
        score_rank,
        float(target_labels[rule_idx]),
    ], dtype=np.float32)
    return state
