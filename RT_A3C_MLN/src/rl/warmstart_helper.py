
"""Warm-start helper with external window-level validation support (V3).

Changes:
* External validation is the default expected path for the project.
* Deduplicates train and external validation sets separately.
* Reports whether external validation was used and the actual sample counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.rl.ac_model import ActorCriticNet


@dataclass
class WarmStartTrainConfig:
    seed: int = 42
    state_dim: int = 12
    action_dim: int = 2
    hidden_dim: int = 128
    dropout: float = 0.10
    lr: float = 1e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.05
    epochs: int = 200
    batch_size: int = 64
    val_split: float = 0.0
    early_stop_patience: int = 20
    use_class_balance: bool = True
    deduplicate: bool = True
    dedup_round: int = 6


def _deduplicate_xy(X: np.ndarray, y: np.ndarray, round_decimals: int = 6):
    if len(X) == 0:
        return X, y
    Xr = np.round(np.asarray(X, dtype=np.float32), round_decimals)
    y = np.asarray(y, dtype=np.int64)
    keys = np.concatenate([Xr, y.reshape(-1, 1)], axis=1)
    _, uniq_idx = np.unique(keys, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return X[uniq_idx], y[uniq_idx]


def _stratified_split(X: np.ndarray, y: np.ndarray, val_split: float, seed: int):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, val_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        if len(idx) <= 1:
            train_idx.extend(idx.tolist())
            continue
        cut = max(1, int(round(len(idx) * (1.0 - val_split))))
        cut = min(cut, len(idx) - 1)
        train_idx.extend(idx[:cut].tolist())
        val_idx.extend(idx[cut:].tolist())
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    if len(val_idx) == 0:
        val_idx = train_idx.copy()
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def train_warmstart_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_path: Path,
    config: WarmStartTrainConfig,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if config.deduplicate:
        X, y = _deduplicate_xy(X, y, round_decimals=config.dedup_round)

    use_external_val = X_val is not None and y_val is not None and len(X_val) > 0 and len(y_val) > 0
    if use_external_val:
        X_tr, y_tr = X, y
        X_va = np.asarray(X_val, dtype=np.float32)
        y_va = np.asarray(y_val, dtype=np.int64)
        if config.deduplicate:
            X_va, y_va = _deduplicate_xy(X_va, y_va, round_decimals=config.dedup_round)
    else:
        if config.val_split > 0 and len(X) >= 8:
            X_tr, y_tr, X_va, y_va = _stratified_split(X, y, config.val_split, config.seed)
        else:
            X_tr, y_tr = X, y
            X_va, y_va = X, y

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCriticNet(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    if config.use_class_balance:
        counts = np.bincount(y_tr, minlength=config.action_dim).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (counts.shape[0] * counts)
        ce_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        ce_weight = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs), eta_min=config.lr * 0.05
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=min(config.batch_size, max(1, len(X_tr))),
        shuffle=True,
    )
    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    best_val_acc = -1.0
    best_state = None
    best_epoch = 0
    patience = 0

    for epoch in range(config.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = F.cross_entropy(
                logits,
                yb,
                weight=ce_weight,
                label_smoothing=config.label_smoothing,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_va, _ = model(X_va_t)
            preds_va = torch.argmax(logits_va, dim=-1)
            val_acc = float((preds_va == y_va_t).float().mean().item())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_path)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_tr, _ = model(X_tr_t)
        train_acc = float((torch.argmax(logits_tr, dim=-1) == y_tr_t).float().mean().item())
        logits_va, _ = model(X_va_t)
        val_acc_final = float((torch.argmax(logits_va, dim=-1) == y_va_t).float().mean().item())

    info = {
        "train_accuracy": float(train_acc),
        "best_val_accuracy": float(best_val_acc),
        "final_val_accuracy": float(val_acc_final),
        "epoch": int(best_epoch),
        "num_train_samples": int(len(X_tr)),
        "num_val_samples": int(len(X_va)),
        "used_external_val": bool(use_external_val),
    }
    return float(best_val_acc), info
