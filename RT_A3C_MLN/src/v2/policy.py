"""Actor-critic policy network and trainer (v2).

Architecture matches the paper Section 5.6: a 2-hidden-layer MLP with
64 hidden units and ReLU activations, sharing a backbone between the
actor and critic heads.

We deliberately drop the LayerNorm / ResBlock / dropout used in the
legacy ``ac_model.py`` because the paper does not describe them. If
you want to restore the heavier architecture, swap ``PolicyNetV2`` for
the legacy ``ActorCriticNet`` — the rest of the v2 pipeline is agnostic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass(frozen=True)
class WarmStartConfigV2:
    state_dim: int = 12
    action_dim: int = 2
    hidden_dim: int = 64
    epochs: int = 200
    lr: float = 1e-3
    seed: int = 42


@dataclass(frozen=True)
class A3CConfigV2:
    state_dim: int = 12
    action_dim: int = 2
    hidden_dim: int = 64
    num_workers: int = 4
    episodes_per_worker: int = 100
    gamma: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 7e-4
    grad_clip: float = 5.0
    seed: int = 42


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PolicyNetV2(nn.Module):
    """Two-hidden-layer MLP with shared backbone — matches paper Section 5.6."""

    def __init__(self, state_dim: int = 12, action_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        return self.policy_head(feat), self.value_head(feat).squeeze(-1)


# ---------------------------------------------------------------------------
# Warm-start
# ---------------------------------------------------------------------------

def warmstart_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Path,
    cfg: WarmStartConfigV2 | None = None,
) -> tuple[float, PolicyNetV2]:
    cfg = cfg or WarmStartConfigV2()
    set_global_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)

    n_keep = max(int((y_train == 0).sum()), 1)
    n_dis = max(int((y_train == 1).sum()), 1)
    weights = torch.tensor([1.0 / n_keep, 1.0 / n_dis], dtype=torch.float32, device=device)

    model = PolicyNetV2(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss(weight=weights)

    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        logits, _ = model(Xt)
        loss = crit(logits, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xt)[0].argmax(dim=1)
        train_acc = float((pred == yt).float().mean().item())

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return train_acc, model


# ---------------------------------------------------------------------------
# Greedy evaluation under PaperRuleEnv
# ---------------------------------------------------------------------------

def greedy_evaluate(
    model: PolicyNetV2,
    state_dict: dict,
    env_cls: Callable,
) -> dict:
    """Return per-window metrics; uses ``env_cls`` (PaperRuleEnv by default)."""
    device = next(model.parameters()).device
    env = env_cls(state_dict)
    state = env.reset()
    target_labels = np.asarray(state_dict["target_labels"], dtype=int)
    n = int(state_dict["num_rules"])

    actions: list[int] = []
    total_r = 0.0
    for t in range(n):
        x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x)
            a = int(torch.argmax(logits, dim=1).item())
        actions.append(a)
        state, r, _, _ = env.step(t, a)
        total_r += float(r)

    actions_arr = np.asarray(actions, dtype=int)
    attack_mask = target_labels == 1
    normal_mask = target_labels == 0

    if attack_mask.sum() > 0:
        attack_keep_rate = float((actions_arr[attack_mask] == 0).mean())
    else:
        attack_keep_rate = float("nan")
    if normal_mask.sum() > 0:
        normal_disable_rate = float((actions_arr[normal_mask] == 1).mean())
    else:
        normal_disable_rate = float("nan")

    correct = (
        ((target_labels == 1) & (actions_arr == 0))
        | ((target_labels == 0) & (actions_arr == 1))
    )
    sel_acc = float(correct.mean())

    # Confusion matrix on rule actions:
    # TP = correctly kept attack rule; FN = attack rule disabled
    # TN = correctly disabled normal rule; FP = normal rule kept
    tp = int(((target_labels == 1) & (actions_arr == 0)).sum())
    fn = int(((target_labels == 1) & (actions_arr == 1)).sum())
    tn = int(((target_labels == 0) & (actions_arr == 1)).sum())
    fp = int(((target_labels == 0) & (actions_arr == 0)).sum())

    return {
        "attack_keep_rate": attack_keep_rate,
        "normal_disable_rate": normal_disable_rate,
        "selection_accuracy": sel_acc,
        "greedy_total_reward": total_r,
        "actions": actions,
        "tp_attack_keep": tp,
        "fn_attack_disable": fn,
        "tn_normal_disable": tn,
        "fp_normal_keep": fp,
    }


# ---------------------------------------------------------------------------
# A3C fine-tuning (synchronous; matches the paper's small episode lengths)
# ---------------------------------------------------------------------------

def a3c_fine_tune(
    init_model: PolicyNetV2,
    train_states: list[dict],
    env_cls: Callable,
    cfg: A3CConfigV2 | None = None,
    log_callback: Callable[[int, dict], None] | None = None,
) -> PolicyNetV2:
    """Fine-tune ``init_model`` with on-policy A3C-style updates.

    Synchronous because each worker would just block on a shared global
    in our small problem (≤ 30 rules per window). The update is the
    same as the paper's Algorithm 1 with ``num_workers`` independent
    rollouts per episode.
    """
    cfg = cfg or A3CConfigV2()
    set_global_seed(cfg.seed)

    device = next(init_model.parameters()).device
    model = init_model
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    rng = np.random.default_rng(cfg.seed)

    for episode in range(cfg.episodes_per_worker):
        worker_logs = []
        for w in range(cfg.num_workers):
            sd = train_states[int(rng.integers(0, len(train_states)))]
            env = env_cls(sd)
            state = env.reset()
            log_probs, values, rewards, entropies = [], [], [], []

            for t in range(int(sd["num_rules"])):
                x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                logits, value = model(x)
                probs = torch.softmax(logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                action = m.sample()
                log_probs.append(m.log_prob(action))
                values.append(value)
                entropies.append(m.entropy())

                next_state, r, _, _ = env.step(t, int(action.item()))
                rewards.append(float(r))
                state = next_state

            # n-step return = full Monte-Carlo for this short horizon
            R = 0.0
            returns = []
            for r in reversed(rewards):
                R = r + cfg.gamma * R
                returns.insert(0, R)

            log_probs_t = torch.stack(log_probs).squeeze(-1)
            values_t = torch.stack(values).squeeze(-1)
            entropies_t = torch.stack(entropies).squeeze(-1)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

            advantage = returns_t - values_t.detach()
            policy_loss = -(log_probs_t * advantage).mean()
            value_loss = (returns_t - values_t).pow(2).mean()
            entropy = entropies_t.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            worker_logs.append(
                {
                    "worker": w,
                    "total_reward": float(sum(rewards)),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }
            )

        if log_callback is not None:
            log_callback(episode, {"workers": worker_logs})

    return model
