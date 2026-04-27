"""Actor-Critic network for rule selection.

V3 changes vs the previous version:
* Wider hidden layer (default 128) with a residual block so the
  policy can express monotone keep/disable functions of the rank
  features without relying on dropout to fight overfitting.
* Separate policy and value heads with their own intermediate
  layer (better representation capacity for the value function).
* Lower default dropout (0.05) so we do not damage the warm-start
  signal on tiny rule pools.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm1(h)
        h = torch.relu(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return torch.relu(x + h)


class ActorCriticNet(nn.Module):
    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            _ResBlock(hidden_dim, dropout),
            _ResBlock(hidden_dim, dropout),
        )
        self.policy_pre = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.value_pre = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim // 2, action_dim)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # bias init: slightly favour ``keep`` so warm-start does not
        # collapse to all-disable on an imbalanced batch
        with torch.no_grad():
            self.policy_head.bias.copy_(torch.tensor([0.1, -0.1]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.input_proj(x)
        feat = self.trunk(feat)
        logits = self.policy_head(self.policy_pre(feat))
        value = self.value_head(self.value_pre(feat))
        return logits, value
