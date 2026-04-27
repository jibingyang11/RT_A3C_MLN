"""Reward utilities aligned with paper Eq. (11).

This module is the v2 reward implementation that exactly matches the paper.
It is intentionally simpler than ``reward_utils.py`` and contains no
``balance_term``, no ``compute_class_scales``, no ``compute_local_action_reward``.

Paper Eq. (11):

    R(m) = (c_a / N) * sum_{i: y_i=1} m_i * c_i * w_i^+
         - (c_n / N) * sum_{i: y_i=0} m_i * c_i * w_i^+
         - c_d * (1 / N) * sum_i |w_i - w_i^0|

with (c_a, c_n, c_d) = (2.0, 3.0, 0.2) and w_i^+ = max(w_i, 0).

Step reward is delta-reward:

    r_t = R(m_{t+1}) - R(m_t)

This is what the paper says it does, full stop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PaperRewardConfig:
    c_attack: float = 2.0
    c_normal: float = 3.0
    c_drift: float = 0.2


PAPER_REWARD_CONFIG = PaperRewardConfig()


def paper_reward(
    active_mask: np.ndarray,
    weights: np.ndarray,
    rule_scores: np.ndarray,
    initial_weights: np.ndarray,
    target_labels: np.ndarray,
    config: PaperRewardConfig = PAPER_REWARD_CONFIG,
) -> float:
    """Compute the per-window reward defined in Eq. (11).

    Parameters
    ----------
    active_mask : (N,) {0,1} mask of currently active rules.
    weights : (N,) current rule weights.
    rule_scores : (N,) per-rule confidence (c_i in the paper).
    initial_weights : (N,) initial rule weights w^0.
    target_labels : (N,) {0,1} target rule label (1 = attack).
    """
    am = np.asarray(active_mask, dtype=float)
    w = np.asarray(weights, dtype=float)
    rs = np.asarray(rule_scores, dtype=float)
    w0 = np.asarray(initial_weights, dtype=float)
    yl = np.asarray(target_labels, dtype=int)

    n = max(1, am.shape[0])
    w_pos = np.clip(w, 0.0, None)

    attack_term = (yl == 1).astype(float) * am * rs * w_pos
    normal_term = (yl == 0).astype(float) * am * rs * w_pos
    drift_term = np.abs(w - w0)

    r = (
        (config.c_attack / n) * float(attack_term.sum())
        - (config.c_normal / n) * float(normal_term.sum())
        - config.c_drift * (1.0 / n) * float(drift_term.sum())
    )
    return float(r)
