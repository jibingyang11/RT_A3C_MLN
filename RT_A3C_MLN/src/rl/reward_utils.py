
"""Reward functions for the rule-selection MDP.

V6 changes:
* Rebalances the local reward so the policy no longer collapses to
  all-keep when attack rules are easier to preserve than normal rules
  are to disable.
* Uses milder class scaling; attack rules are no longer over-amplified.
* Keeps the same public API as previous versions.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.rl.action_utils import ACTION_DISABLE, ACTION_KEEP


@dataclass(frozen=True)
class RewardConfig:
    # Local action reward
    attack_keep_reward: float = 1.4
    attack_disable_penalty: float = 3.0
    normal_disable_reward: float = 3.2
    normal_keep_penalty: float = 3.8

    # Global structure reward
    attack_global_coef: float = 0.8
    normal_global_coef: float = 1.8
    balance_coef: float = 0.20
    drift_coef: float = 0.01

    score_floor: float = 0.30
    scale_clip_min: float = 0.85
    scale_clip_max: float = 1.10


DEFAULT_REWARD_CONFIG = RewardConfig()


def _balance_term(active_ratio: float, target_ratio: float = 0.5) -> float:
    diff = abs(float(active_ratio) - float(target_ratio))
    return max(0.0, 1.0 - 2.0 * diff)


def compute_reward(
    active_mask: np.ndarray,
    weights: np.ndarray,
    rule_scores: np.ndarray,
    initial_weights: np.ndarray,
    target_labels: np.ndarray,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    active_mask = np.asarray(active_mask, dtype=float)
    weights = np.asarray(weights, dtype=float)
    rule_scores = np.asarray(rule_scores, dtype=float)
    initial_weights = np.asarray(initial_weights, dtype=float)
    target_labels = np.asarray(target_labels, dtype=int)

    if len(active_mask) == 0:
        return 0.0

    score_w = np.maximum(rule_scores, 0.0) * np.maximum(weights, 0.0)
    attack_active = float(np.sum((target_labels == 1) * active_mask * score_w))
    normal_active = float(np.sum((target_labels == 0) * active_mask * score_w))
    drift_penalty = float(np.mean(np.abs(weights - initial_weights)))

    n_attack = max(1, int((target_labels == 1).sum()))
    n_normal = max(1, int((target_labels == 0).sum()))
    target_ratio = float(n_attack) / float(n_attack + n_normal)
    balance = _balance_term(float(active_mask.mean()), target_ratio=target_ratio)

    reward = (
        config.attack_global_coef * (attack_active / n_attack)
        - config.normal_global_coef * (normal_active / n_normal)
        + config.balance_coef * balance
        - config.drift_coef * drift_penalty
    )
    return float(reward)


def compute_class_scales(
    target_labels: np.ndarray,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> tuple[float, float]:
    target_labels = np.asarray(target_labels, dtype=int)
    n_attack = max(1, int((target_labels == 1).sum()))
    n_normal = max(1, int((target_labels == 0).sum()))

    attack_scale = np.sqrt(n_normal / n_attack)
    normal_scale = np.sqrt(n_attack / n_normal)

    attack_scale = float(np.clip(attack_scale, config.scale_clip_min, config.scale_clip_max))
    normal_scale = float(np.clip(normal_scale, config.scale_clip_min, config.scale_clip_max))
    return attack_scale, normal_scale


def compute_local_action_reward(
    target_label: int,
    action: int,
    rule_score: float,
    attack_scale: float = 1.0,
    normal_scale: float = 1.0,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    s = max(float(rule_score), config.score_floor)
    if int(target_label) == 1:
        base = config.attack_keep_reward if action == ACTION_KEEP else -config.attack_disable_penalty
        return float(attack_scale * s * base)
    base = config.normal_disable_reward if action == ACTION_DISABLE else -config.normal_keep_penalty
    return float(normal_scale * s * base)
