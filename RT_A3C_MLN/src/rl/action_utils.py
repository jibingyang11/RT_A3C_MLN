from __future__ import annotations

import numpy as np


ACTION_KEEP = 0
ACTION_DISABLE = 1

ACTION_NAMES = {
    ACTION_KEEP: 'keep',
    ACTION_DISABLE: 'disable',
}


def apply_action(
    active_mask: np.ndarray,
    weights: np.ndarray,
    rule_idx: int,
    action: int,
    step_size: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a binary keep/disable action to one rule."""
    active_mask = np.asarray(active_mask).copy()
    weights = np.asarray(weights).copy()

    if not 0 <= rule_idx < len(active_mask):
        raise IndexError(f'rule_idx out of range: {rule_idx}')

    if action == ACTION_KEEP:
        active_mask[rule_idx] = 1
        return active_mask, weights
    if action == ACTION_DISABLE:
        active_mask[rule_idx] = 0
        weights[rule_idx] = 0.0
        return active_mask, weights

    raise ValueError(f'Unknown action: {action}')
