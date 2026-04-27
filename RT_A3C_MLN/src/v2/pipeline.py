"""End-to-end per-window pipeline (v2).

Calling ``prepare_window_artifacts`` for window i returns the dict that
all v2 protocols feed into. Heavy I/O is cached so repeat-runs are
cheap.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.transaction_utils import build_transactions
from src.v2.rule_pool import (
    LABEL_ATTACK,
    LABEL_NORMAL,
    MiningConfigV2,
    add_transaction_labels,
    build_initial_state_v2,
    build_rule_pool_v2,
    build_warmstart_state_dataset,
    decide_window_mode,
    mine_label_rules_with_fallback,
    one_hot_transactions,
)
from src.v2.window_io import load_window_binary_and_labels


@dataclass
class WindowArtifactsV2:
    window_id: int
    attack_ratio: float
    auto_mode: str
    requested_mode: str
    actual_mode_used: str
    rule_pool_hash: str
    used_support: float
    n_attack_rules: int
    n_normal_rules: int
    rule_pool_df: pd.DataFrame
    state_dict: dict
    X_ws: np.ndarray
    y_ws: np.ndarray


def prepare_window_artifacts(
    project_root: Path,
    window_id: int,
    cfg: MiningConfigV2,
    force_mode: Optional[str] = None,
    plan_name: str = "windows_mixed.csv",
    cache: bool = True,
) -> WindowArtifactsV2:
    # Bump CACHE_VERSION whenever rule_pool.py logic changes; old caches will be
    # automatically invalidated and rebuilt.
    CACHE_VERSION = "v2.1"  # 2026-04 fallback-chain extension to 0.005 + both-labels check
    cache_dir = project_root / "data" / "stream" / "swat" / "v2_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{CACHE_VERSION}_w{window_id}_fm{force_mode or 'auto'}.pkl"
    cache_path = cache_dir / cache_key
    if cache and cache_path.exists():
        import pickle
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df_binary, labels, meta = load_window_binary_and_labels(project_root, window_id, plan_name)
    attack_ratio = meta["attack_ratio"]

    transactions = build_transactions(df_binary, lag=cfg.lag)
    if len(transactions) != len(labels):
        raise ValueError(
            f"window {window_id}: transactions ({len(transactions)}) != labels ({len(labels)})"
        )
    labeled_tx = add_transaction_labels(transactions, labels.tolist())
    df_onehot = one_hot_transactions(labeled_tx)

    label_rules, fallback_rules_all, used_support = mine_label_rules_with_fallback(df_onehot, cfg)
    if label_rules.empty and fallback_rules_all.empty:
        raise ValueError(f"window {window_id}: no rules mineable even after fallback chain")

    auto_mode = decide_window_mode(attack_ratio, cfg)
    pool, requested_mode, actual_mode_used, pool_hash = build_rule_pool_v2(
        label_rules=label_rules,
        fallback_rules_all=fallback_rules_all,
        attack_ratio=attack_ratio,
        cfg=cfg,
        force_mode=force_mode,
    )

    state_dict = build_initial_state_v2(pool)
    X_ws, y_ws = build_warmstart_state_dataset(state_dict)

    art = WindowArtifactsV2(
        window_id=int(window_id),
        attack_ratio=float(attack_ratio),
        auto_mode=auto_mode,
        requested_mode=requested_mode,
        actual_mode_used=actual_mode_used,
        rule_pool_hash=pool_hash,
        used_support=float(used_support),
        n_attack_rules=int((pool["target_label"] == 1).sum()),
        n_normal_rules=int((pool["target_label"] == 0).sum()),
        rule_pool_df=pool,
        state_dict=state_dict,
        X_ws=X_ws,
        y_ws=y_ws,
    )
    if cache:
        import pickle
        with open(cache_path, "wb") as f:
            pickle.dump(art, f)
    return art


def stratified_split(
    plan_df: pd.DataFrame,
    seed: int,
    train_per_bin: int = 4,
    val_per_bin: int = 1,
    test_per_bin: int = 2,
) -> dict:
    """Stratified across attack_bin while ensuring every bin appears in train.

    Falls back to a proportional allocation when a bin has < (train+val+test)
    items, BUT still guarantees train_n >= 1 if available > 0.
    """
    rng = np.random.default_rng(seed)
    detail_rows = []
    out = {"train": [], "val": [], "test": []}
    for ab in ["low", "mid", "high"]:
        sub = plan_df[plan_df["attack_bin"] == ab].copy()
        ids = sub["window_id"].astype(int).tolist()
        rng.shuffle(ids)
        avail = len(ids)
        if avail == 0:
            detail_rows.append(
                {"attack_bin": ab, "available": 0, "train_ids": [], "val_ids": [], "test_ids": []}
            )
            continue

        # Priority order so each bin always contributes to train first.
        train_n = min(train_per_bin, avail)
        rest = avail - train_n
        val_n = min(val_per_bin, rest)
        rest = rest - val_n
        test_n = min(test_per_bin, rest)

        train_ids = ids[:train_n]
        val_ids = ids[train_n : train_n + val_n]
        test_ids = ids[train_n + val_n : train_n + val_n + test_n]

        out["train"].extend(train_ids)
        out["val"].extend(val_ids)
        out["test"].extend(test_ids)
        detail_rows.append(
            {
                "attack_bin": ab,
                "available": avail,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            }
        )
    out["_detail_df"] = pd.DataFrame(detail_rows)
    return out
