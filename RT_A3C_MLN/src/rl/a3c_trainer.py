
"""A3C trainer (V6 balanced).

Changes:
* Validation / model-selection score uses a harmonic balance between
  attack-keep and normal-disable, so all-keep and all-disable policies
  are penalised strongly.
* Default val_train_blend is 0.0; best model is chosen by validation,
  not train contamination.
* Best EMA checkpoint is still written and preferred downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.rl.ac_model import ActorCriticNet
from src.rl.simple_env import SimpleRuleEnv


@dataclass(frozen=True)
class A3CConfig:
    num_workers: int = 4
    num_episodes_per_worker: int = 120
    n_step: int = 5
    gamma: float = 0.95
    gae_lambda: float = 0.95
    lr: float = 2e-5
    weight_decay: float = 1e-5

    entropy_coef: float = 0.002
    value_coef: float = 0.25
    bc_coef: float = 1.2
    bc_focal_gamma: float = 1.0
    advantage_clip: float = 5.0

    max_grad_norm: float = 3.0
    ema_decay: float = 0.997
    use_ema_for_eval: bool = True

    hidden_dim: int = 128
    state_dim: int = 12
    action_dim: int = 2
    dropout: float = 0.05

    temperature: float = 0.9
    sample_attack_oversample: float = 1.0

    device: str = 'auto'
    seed: int = 42
    log_every: int = 10
    eval_every: int = 10
    early_stop_patience: int = 12
    val_train_blend: float = 0.0


def _select_device(pref: str) -> torch.device:
    if pref == 'cpu':
        return torch.device('cpu')
    if pref == 'cuda':
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _tensor(x, device, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


@torch.no_grad()
def _ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.data.copy_(b.data)


def _bc_target(target_label: int) -> int:
    return 0 if int(target_label) == 1 else 1


def _gae(rewards, values, last_value, gamma, lam, device):
    advantages = [None] * len(rewards)
    gae = torch.tensor(0.0, device=device)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        delta = torch.as_tensor(float(rewards[t]), dtype=torch.float32, device=device) + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages


def _rollout(model, state_dict, device, config: A3CConfig) -> dict:
    env = SimpleRuleEnv(state_dict, max_steps=int(state_dict['num_rules']) + 1)
    obs = env.reset()
    target_labels = np.asarray(state_dict['target_labels'], dtype=np.int64)
    num_rules = int(state_dict['num_rules'])

    logits_list, values_list, log_probs_list = [], [], []
    rewards_list, actions_list, bc_targets_list = [], [], []

    done = False
    step_count = 0
    while not done and step_count < num_rules:
        rule_idx = int(env.current_rule_idx)
        x = _tensor(obs, device).unsqueeze(0)
        logits, value = model(x)
        logits = logits.squeeze(0)
        value = value.squeeze(0).squeeze(-1) if value.dim() > 0 else value
        sample_logits = logits / max(float(config.temperature), 1e-6)
        probs = F.softmax(sample_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action_t = dist.sample()
        action = int(action_t.item())
        log_p_full = F.log_softmax(sample_logits, dim=-1)

        next_obs, reward, done, _ = env.step(action)
        reward = float(np.clip(reward, -5.0, 5.0))

        logits_list.append(logits)
        values_list.append(value)
        log_probs_list.append(log_p_full[action_t])
        rewards_list.append(float(reward))
        actions_list.append(action)
        bc_targets_list.append(_bc_target(int(target_labels[rule_idx])))

        obs = next_obs
        step_count += 1

    if not done:
        with torch.no_grad():
            x = _tensor(obs, device).unsqueeze(0)
            _, last_value = model(x)
            last_value = last_value.squeeze(0).squeeze(-1)
    else:
        last_value = torch.tensor(0.0, device=device)

    advantages = _gae(rewards_list, values_list, last_value, config.gamma, config.gae_lambda, device)
    returns = [adv + v.detach() for adv, v in zip(advantages, values_list)]

    selection_correct = sum(1 for a, y in zip(actions_list, bc_targets_list) if a == y)
    selection_accuracy = selection_correct / max(1, len(actions_list))
    n_attack = sum(1 for y in bc_targets_list if y == 0)
    n_normal = sum(1 for y in bc_targets_list if y == 1)
    attack_keep = sum(1 for a, y in zip(actions_list, bc_targets_list) if y == 0 and a == 0) / max(1, n_attack)
    normal_disable = sum(1 for a, y in zip(actions_list, bc_targets_list) if y == 1 and a == 1) / max(1, n_normal)

    return {
        'logits': torch.stack(logits_list),
        'log_probs': torch.stack(log_probs_list),
        'values': torch.stack(values_list),
        'advantages': torch.stack(advantages),
        'returns': torch.stack(returns),
        'actions': torch.tensor(actions_list, dtype=torch.long, device=device),
        'bc_targets': torch.tensor(bc_targets_list, dtype=torch.long, device=device),
        'rewards_sum': float(sum(rewards_list)),
        'selection_accuracy': float(selection_accuracy),
        'attack_keep_rate': float(attack_keep),
        'normal_disable_rate': float(normal_disable),
        'num_rules': int(num_rules),
    }


def _focal_bc_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    p = torch.exp(log_p)
    counts = torch.bincount(targets, minlength=logits.shape[1]).float()
    counts = torch.clamp(counts, min=1.0)
    class_weights = counts.sum() / (counts.shape[0] * counts)
    class_weights = class_weights.to(logits.device)
    idx = torch.arange(len(targets), device=logits.device)
    pt = p[idx, targets]
    ce = F.nll_loss(log_p, targets, reduction='none', weight=class_weights)
    focal = (1.0 - pt).pow(gamma)
    return (focal * ce).mean()


@torch.no_grad()
def _greedy_metrics(model: ActorCriticNet, state_dict: dict, device: torch.device) -> dict:
    env = SimpleRuleEnv(state_dict, max_steps=int(state_dict['num_rules']) + 1)
    obs = env.reset()
    target_labels = np.asarray(state_dict['target_labels'], dtype=np.int64)

    actions = []
    done = False
    while not done and env.current_rule_idx < int(state_dict['num_rules']):
        x = _tensor(obs, device).unsqueeze(0)
        logits, _ = model(x)
        action = int(torch.argmax(logits.squeeze(0), dim=-1).item())
        obs, _, done, _ = env.step(action)
        actions.append(action)

    bc_targets = np.array([_bc_target(y) for y in target_labels[:len(actions)]], dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    acc = float((actions == bc_targets).mean()) if len(actions) else 0.0
    attack_mask = bc_targets == 0
    normal_mask = bc_targets == 1
    attack_keep = float((actions[attack_mask] == 0).mean()) if attack_mask.any() else 0.0
    normal_disable = float((actions[normal_mask] == 1).mean()) if normal_mask.any() else 0.0
    return {'acc': acc, 'attack_keep': attack_keep, 'normal_disable': normal_disable}


def _selection_score(metrics: dict[str, float]) -> float:
    acc = float(metrics.get('acc', metrics.get('selection_accuracy', 0.0)))
    keep = float(metrics.get('attack_keep', metrics.get('attack_keep_rate', metrics.get('keep', 0.0))))
    disable = float(metrics.get('normal_disable', metrics.get('normal_disable_rate', metrics.get('disable', 0.0))))
    balance = 2.0 * keep * disable / (keep + disable + 1e-8)
    return 0.4 * acc + 0.6 * balance


def train_a3c(
    train_state_dicts: Sequence[dict],
    config: A3CConfig,
    save_dir: Path,
    run_name: str = 'a3c_run',
    initial_model_path: Path | None = None,
    val_state_dicts: Sequence[dict] | None = None,
) -> tuple[ActorCriticNet, pd.DataFrame]:
    device = _select_device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = ActorCriticNet(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    if initial_model_path is not None and Path(initial_model_path).exists():
        state = torch.load(initial_model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f'[A3C] loaded warm-start weights from {initial_model_path}')

    ema_model = copy.deepcopy(model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = save_dir.parent / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    best_epoch = 0
    patience = 0
    history = []

    train_state_dicts = list(train_state_dicts)
    pool_sizes = []
    for sd in train_state_dicts:
        tl = np.asarray(sd['target_labels'])
        attack_ratio = float((tl == 1).mean()) if len(tl) else 0.0
        weight = 1.0 + max(0.0, attack_ratio - 0.5) * 2.0 * max(1.0, config.sample_attack_oversample)
        pool_sizes.append(weight)
    pool_probs = np.asarray(pool_sizes, dtype=float)
    pool_probs = pool_probs / pool_probs.sum()

    for ep in range(1, config.num_episodes_per_worker + 1):
        model.train()
        batch_rollouts = []
        for _ in range(config.num_workers):
            idx = int(np.random.choice(len(train_state_dicts), p=pool_probs))
            batch_rollouts.append(_rollout(model, train_state_dicts[idx], device, config))

        logits = torch.cat([r['logits'] for r in batch_rollouts], dim=0)
        log_probs = torch.cat([r['log_probs'] for r in batch_rollouts], dim=0)
        values = torch.cat([r['values'] for r in batch_rollouts], dim=0)
        advantages = torch.cat([r['advantages'] for r in batch_rollouts], dim=0)
        returns = torch.cat([r['returns'] for r in batch_rollouts], dim=0)
        bc_targets = torch.cat([r['bc_targets'] for r in batch_rollouts], dim=0)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            advantages = advantages.clamp(min=-config.advantage_clip, max=config.advantage_clip)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.smooth_l1_loss(values, returns)
        bc_loss = _focal_bc_loss(logits, bc_targets, gamma=config.bc_focal_gamma)

        loss = policy_loss + config.value_coef * value_loss + config.bc_coef * bc_loss - config.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        _ema_update(ema_model, model, config.ema_decay)

        train_metrics = {
            'reward': float(np.mean([r['rewards_sum'] for r in batch_rollouts])),
            'acc': float(np.mean([r['selection_accuracy'] for r in batch_rollouts])),
            'keep': float(np.mean([r['attack_keep_rate'] for r in batch_rollouts])),
            'disable': float(np.mean([r['normal_disable_rate'] for r in batch_rollouts])),
            'bc': float(bc_loss.item()),
            'v': float(value_loss.item()),
            'H': float(entropy.item()),
        }

        history.append({'episode': ep, **train_metrics})

        if ep % max(1, config.log_every) == 0:
            print(
                f"[A3C][ep {ep}] reward={train_metrics['reward']:.3f} "
                f"acc={train_metrics['acc']:.3f} keep={train_metrics['keep']:.3f} "
                f"disable={train_metrics['disable']:.3f} bc={train_metrics['bc']:.3f} "
                f"v={train_metrics['v']:.3f} H={train_metrics['H']:.3f}"
            )

        if val_state_dicts and (ep % max(1, config.eval_every) == 0):
            ema_eval = [_greedy_metrics(ema_model, sd, device) for sd in val_state_dicts]
            val_metrics = {
                'acc': float(np.mean([m['acc'] for m in ema_eval])),
                'keep': float(np.mean([m['attack_keep'] for m in ema_eval])),
                'disable': float(np.mean([m['normal_disable'] for m in ema_eval])),
            }
            val_score = _selection_score(val_metrics)
            train_score = _selection_score(train_metrics)
            blended = (1.0 - config.val_train_blend) * val_score + config.val_train_blend * train_score

            history[-1].update({
                'val_acc': val_metrics['acc'],
                'val_keep': val_metrics['keep'],
                'val_disable': val_metrics['disable'],
                'val_score': val_score,
                'blended_score': blended,
            })

            print(
                f"[val][ep {ep}] val_acc={val_metrics['acc']:.4f} "
                f"val_keep={val_metrics['keep']:.4f} val_disable={val_metrics['disable']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} blended={blended:.4f}"
            )

            raw_ckpt = save_dir / f'{run_name}_ep{ep:03d}.pth'
            ema_ckpt = save_dir / f'{run_name}_ep{ep:03d}_ema.pth'
            torch.save(model.state_dict(), raw_ckpt)
            torch.save(ema_model.state_dict(), ema_ckpt)

            if blended > best_score:
                best_score = blended
                best_epoch = ep
                patience = 0
                torch.save(model.state_dict(), save_dir / f'{run_name}_best_raw.pth')
                torch.save(ema_model.state_dict(), save_dir / f'{run_name}_best_ema.pth')
            else:
                patience += 1
                if patience >= config.early_stop_patience:
                    print(f'[A3C] early stop at ep {ep}; best epoch={best_epoch} score={best_score:.4f}')
                    break

    history_df = pd.DataFrame(history)
    history_df.to_csv(logs_dir / f'{run_name}_training_curve.csv', index=False)

    torch.save(model.state_dict(), save_dir / f'{run_name}_global_policy.pth')
    torch.save(ema_model.state_dict(), save_dir / f'{run_name}_global_policy_ema.pth')

    best_ema_path = save_dir / f'{run_name}_best_ema.pth'
    if best_ema_path.exists():
        best_state = torch.load(best_ema_path, map_location=device)
        ema_model.load_state_dict(best_state, strict=False)
        print(f'[A3C] returning best EMA model (epoch={best_epoch}, score={best_score:.4f})')
    return ema_model, history_df
