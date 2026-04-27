"""Sequential rule-selection env aligned with paper Eq. (11).

Differences from ``simple_env.py``:
* Step reward is the pure delta-reward r_t = R(m_{t+1}) - R(m_t).
* No local_reward, no global_delta_coef, no class_scales.
* Uses ``paper_reward`` from ``reward_utils_v2``.
* The action ``KEEP=0`` keeps the mask unchanged; ``DISABLE=1`` sets m_i := 0.
  The weight w_i is NOT modified (paper's design choice: only the mask changes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.rl.reward_utils_v2 import PAPER_REWARD_CONFIG, PaperRewardConfig, paper_reward


ACTION_KEEP = 0
ACTION_DISABLE = 1


@dataclass
class StepInfoV2:
    t: int
    rule_idx: int
    action: int
    active: int
    weight: float
    rule_score: float
    target_label: int
    old_reward: float
    new_reward: float
    delta_reward: float


class PaperRuleEnv:
    """Episode = one window. Length = num_rules. State = paper Eq. (10) 12-dim vector."""

    def __init__(
        self,
        init_state_dict: dict[str, Any],
        reward_config: PaperRewardConfig = PAPER_REWARD_CONFIG,
    ) -> None:
        self.init_state_dict = init_state_dict
        self.reward_config = reward_config
        self.reset()

    def reset(self) -> np.ndarray:
        self.rule_pool = self.init_state_dict["rule_pool"]
        self.active_mask = np.asarray(self.init_state_dict["active_mask"], dtype=int).copy()
        self.weights = np.asarray(self.init_state_dict["weights"], dtype=float).copy()
        self.initial_weights = np.asarray(
            self.init_state_dict["initial_weights"], dtype=float
        ).copy()
        self.rule_scores = np.asarray(self.init_state_dict["rule_scores"], dtype=float).copy()
        self.target_labels = np.asarray(self.init_state_dict["target_labels"], dtype=int).copy()
        self.num_rules = int(self.init_state_dict["num_rules"])
        self.t = 0
        self.current_rule_idx = 0
        return self._build_state(self.current_rule_idx)

    # ------------------------------------------------------------------
    # State construction (matches paper Eq. (10) exactly)
    # ------------------------------------------------------------------
    def _build_state(self, rule_idx: int) -> np.ndarray:
        am = self.active_mask.astype(float)
        active_w = self.weights[am > 0.5]
        if active_w.size == 0:
            mu_w = sigma_w = w_min = w_max = 0.0
        else:
            mu_w = float(active_w.mean())
            sigma_w = float(active_w.std())
            w_min = float(active_w.min())
            w_max = float(active_w.max())

        denom = max(1, self.num_rules - 1)
        return np.array(
            [
                float(am.mean()),               # bar_m   active ratio
                mu_w,                           # mu_w
                sigma_w,                        # sigma_w
                w_min,                          # w_min
                w_max,                          # w_max
                float(am.sum()),                # ||m||_1
                float(self.num_rules),          # N_k
                float(rule_idx) / float(denom), # pi_i
                float(am[rule_idx]),            # m_i
                float(self.weights[rule_idx]),  # w_i
                float(self.rule_scores[rule_idx]),  # c_i
                float(self.target_labels[rule_idx]),# y_i (used for warm-start; at A3C time we mask this)
            ],
            dtype=np.float32,
        )

    def _reward(self) -> float:
        return paper_reward(
            self.active_mask,
            self.weights,
            self.rule_scores,
            self.initial_weights,
            self.target_labels,
            config=self.reward_config,
        )

    # ------------------------------------------------------------------
    def step(self, rule_idx: int, action: int):
        rule_idx = int(rule_idx)
        action = int(action)
        if not 0 <= rule_idx < self.num_rules:
            raise IndexError(rule_idx)

        old_r = self._reward()
        if action == ACTION_DISABLE:
            self.active_mask[rule_idx] = 0
        # ACTION_KEEP: mask unchanged
        new_r = self._reward()
        delta = float(new_r - old_r)

        self.t += 1
        self.current_rule_idx = max(self.current_rule_idx, rule_idx + 1)
        done = self.current_rule_idx >= self.num_rules

        if done:
            next_state = np.zeros(12, dtype=np.float32)
        else:
            next_state = self._build_state(self.current_rule_idx)

        info = StepInfoV2(
            t=self.t,
            rule_idx=rule_idx,
            action=action,
            active=int(self.active_mask[rule_idx]),
            weight=float(self.weights[rule_idx]),
            rule_score=float(self.rule_scores[rule_idx]),
            target_label=int(self.target_labels[rule_idx]),
            old_reward=old_r,
            new_reward=new_r,
            delta_reward=delta,
        ).__dict__
        return next_state, delta, done, info
