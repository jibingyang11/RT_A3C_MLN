
"""Light-weight sequential environment for window-level rule selection.

V6 changes:
* Slightly increases global reward contribution so the agent does not
  only optimise per-rule local rewards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from src.rl.action_utils import apply_action
from src.rl.obs_utils import build_state_vector
from src.rl.reward_utils import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    compute_class_scales,
    compute_local_action_reward,
    compute_reward,
)


@dataclass
class StepInfo:
    t: int
    rule_idx: int
    action: int
    active: int
    weight: float
    rule_score: float
    target_label: int
    old_reward: float
    new_reward: float
    local_reward: float
    delta_global_reward: float


class SimpleRuleEnv:
    def __init__(
        self,
        init_state_dict: dict[str, Any],
        step_size: float = 0.5,
        max_steps: int = 100,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        global_delta_coef: float = 0.35,
    ) -> None:
        self.init_state_dict = init_state_dict
        self.step_size = step_size
        self.max_steps = max_steps
        self.reward_config = reward_config
        self.global_delta_coef = float(global_delta_coef)
        self.reset()

    def reset(self) -> np.ndarray:
        self.rule_pool = self.init_state_dict['rule_pool']
        self.active_mask = np.asarray(self.init_state_dict['active_mask']).copy()
        self.weights = np.asarray(self.init_state_dict['weights']).copy()
        self.initial_weights = np.asarray(self.init_state_dict['initial_weights']).copy()
        self.rule_scores = np.asarray(self.init_state_dict['rule_scores']).copy()
        self.target_labels = np.asarray(self.init_state_dict['target_labels']).copy()
        self.num_rules = int(self.init_state_dict['num_rules'])
        self.attack_scale, self.normal_scale = compute_class_scales(self.target_labels, self.reward_config)

        self.t = 0
        self.current_rule_idx = 0
        return self._build_obs(self.current_rule_idx)

    def _reward(self) -> float:
        return compute_reward(
            self.active_mask,
            self.weights,
            self.rule_scores,
            self.initial_weights,
            self.target_labels,
            config=self.reward_config,
        )

    def _build_obs(self, rule_idx: int) -> np.ndarray:
        return build_state_vector(
            self.active_mask,
            self.weights,
            self.rule_scores,
            rule_idx,
            self.target_labels,
        )

    def step(self, action_or_rule_idx: int, action: int | None = None):
        if action is None:
            rule_idx = int(self.current_rule_idx)
            action = int(action_or_rule_idx)
        else:
            rule_idx = int(action_or_rule_idx)
            action = int(action)

        if not 0 <= rule_idx < self.num_rules:
            raise IndexError(f'rule_idx out of range: {rule_idx}')

        old_reward = self._reward()

        self.active_mask, self.weights = apply_action(
            self.active_mask,
            self.weights,
            rule_idx,
            action,
            step_size=self.step_size,
        )

        self.t += 1
        local_reward = compute_local_action_reward(
            target_label=int(self.target_labels[rule_idx]),
            action=action,
            rule_score=float(self.rule_scores[rule_idx]),
            attack_scale=self.attack_scale,
            normal_scale=self.normal_scale,
            config=self.reward_config,
        )
        new_reward = self._reward()
        delta_global_reward = float(new_reward - old_reward)
        step_reward = float(local_reward + self.global_delta_coef * delta_global_reward)

        self.current_rule_idx = max(self.current_rule_idx, rule_idx + 1)
        done = bool((self.current_rule_idx >= self.num_rules) or (self.t >= self.max_steps))

        if done:
            next_state = np.zeros_like(self._build_obs(min(self.current_rule_idx, self.num_rules - 1)))
        else:
            next_state = self._build_obs(self.current_rule_idx)

        info = StepInfo(
            t=self.t,
            rule_idx=rule_idx,
            action=action,
            active=int(self.active_mask[rule_idx]),
            weight=float(self.weights[rule_idx]),
            rule_score=float(self.rule_scores[rule_idx]),
            target_label=int(self.target_labels[rule_idx]),
            old_reward=float(old_reward),
            new_reward=float(new_reward),
            local_reward=float(local_reward),
            delta_global_reward=float(delta_global_reward),
        ).__dict__

        return next_state, step_reward, done, info
