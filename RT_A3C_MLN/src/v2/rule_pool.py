"""Rule-mining + adaptive rule-pool construction (v2).

Differences vs ``src/pipeline/window_experiment.py``:
* Default thresholds match the paper: low=0.30, high=0.70.
* ``build_rule_pool_v2`` accepts ``force_mode`` that *actually* forces
  the corresponding branch even when the standard branch could also
  produce a non-empty pool. This is required for the ablation in
  Section 6.7 of the paper.
* Returns a deterministic ``rule_pool_hash`` for audit/debugging.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from src.data.transaction_utils import build_transactions

LABEL_ATTACK = "LABEL_ATTACK"
LABEL_NORMAL = "LABEL_NORMAL"


@dataclass(frozen=True)
class MiningConfigV2:
    """Paper-aligned defaults."""

    lag: int = 2
    min_support: float = 0.1
    min_confidence: float = 0.6
    top_attack_rules: int = 10
    top_normal_rules: int = 10
    relaxed_normal_rules: int = 3
    relaxed_normal_min_weight: float = 0.3
    relaxed_normal_weight_scale: float = 30.0
    relaxed_normal_weight_cap: float = 3.0
    attack_weight_cap: float = 3.0
    low_attack_threshold: float = 0.30  # paper (Eq. 8)
    high_attack_threshold: float = 0.70  # paper (Eq. 8)
    # Adaptive support fallback chain when the strict threshold mines too few items.
    # We step down until BOTH LABEL_ATTACK and LABEL_NORMAL appear as 1-itemsets;
    # a low-attack-ratio window like 4% attack needs support <= 0.04 for the
    # LABEL_ATTACK token to ever appear in freq_items.
    fallback_supports: tuple = (0.08, 0.05, 0.03, 0.02, 0.01, 0.005)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confidence_to_weight(c: float, eps: float = 1e-6) -> float:
    cl = float(np.clip(c, eps, 1 - eps))
    return float(np.log(cl / (1 - cl)))


def _clipped_weight(c: float, wmax: float = 3.0) -> float:
    return float(np.clip(_confidence_to_weight(c), 0.0, wmax))


def _relaxed_weight(score: float, wmin: float, scale: float, wmax: float) -> float:
    return float(np.clip(scale * float(score), wmin, wmax))


def add_transaction_labels(transactions: list[list[str]], labels) -> list[list[str]]:
    out = []
    for tx, y in zip(transactions, labels):
        token = LABEL_ATTACK if int(y) == 1 else LABEL_NORMAL
        out.append(list(tx) + [token])
    return out


def one_hot_transactions(labeled_transactions: list[list[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    onehot = te.fit(labeled_transactions).transform(labeled_transactions)
    return pd.DataFrame(onehot, columns=te.columns_)


def mine_label_rules_with_fallback(
    df_onehot: pd.DataFrame, cfg: MiningConfigV2
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Returns (label_rules, fallback_rules_all, used_support).

    ``fallback_rules_all`` includes ALL label-oriented rules at min_confidence=0
    so downstream relaxed/balanced fallback can score them properly.

    The fallback chain steps down through ``cfg.fallback_supports`` and keeps
    going until BOTH LABEL_ATTACK and LABEL_NORMAL appear as 1-itemsets in
    ``freq_items``. This is required for low- (or high-) attack-ratio windows
    where one of the labels has support below ``cfg.min_support``.
    """
    used_support = cfg.min_support
    chain = [cfg.min_support, *cfg.fallback_supports]
    freq_items = pd.DataFrame()
    have_attack = have_normal = False
    for s in chain:
        freq_items = fpgrowth(df_onehot, min_support=s, use_colnames=True, max_len=2)
        used_support = s
        if freq_items.empty:
            continue
        # Check label tokens appear as singletons (otherwise no label-rule is possible).
        present = set()
        for it in freq_items["itemsets"]:
            if len(it) == 1:
                present.update(it)
        have_attack = LABEL_ATTACK in present
        have_normal = LABEL_NORMAL in present
        if have_attack and have_normal:
            break
    if freq_items.empty or not (have_attack or have_normal):
        empty = pd.DataFrame(columns=["antecedent_str", "consequent_str", "support", "confidence", "lift"])
        return empty, empty, used_support

    rules_strict = association_rules(freq_items, metric="confidence", min_threshold=cfg.min_confidence)
    rules_strict = rules_strict[
        (rules_strict["consequents"].apply(lambda x: len(x) == 1))
        & (rules_strict["consequents"].apply(lambda x: list(x)[0] in [LABEL_ATTACK, LABEL_NORMAL]))
    ].copy()
    if not rules_strict.empty:
        rules_strict["antecedent_str"] = rules_strict["antecedents"].apply(lambda x: " & ".join(sorted(list(x))))
        rules_strict["consequent_str"] = rules_strict["consequents"].apply(lambda x: list(x)[0])
        rules_strict = rules_strict[["antecedent_str", "consequent_str", "support", "confidence", "lift"]].copy()
    else:
        rules_strict = pd.DataFrame(columns=["antecedent_str", "consequent_str", "support", "confidence", "lift"])

    rules_all = association_rules(freq_items, metric="confidence", min_threshold=0.0)
    rules_all = rules_all[
        (rules_all["consequents"].apply(lambda x: len(x) == 1))
        & (rules_all["consequents"].apply(lambda x: list(x)[0] in [LABEL_ATTACK, LABEL_NORMAL]))
    ].copy()
    if not rules_all.empty:
        rules_all["antecedent_str"] = rules_all["antecedents"].apply(lambda x: " & ".join(sorted(list(x))))
        rules_all["consequent_str"] = rules_all["consequents"].apply(lambda x: list(x)[0])
        rules_all = rules_all[["antecedent_str", "consequent_str", "support", "confidence", "lift"]].copy()
    else:
        rules_all = pd.DataFrame(columns=["antecedent_str", "consequent_str", "support", "confidence", "lift"])

    return rules_strict.reset_index(drop=True), rules_all.reset_index(drop=True), used_support


def decide_window_mode(attack_ratio: float, cfg: MiningConfigV2) -> str:
    if attack_ratio < cfg.low_attack_threshold:
        return "balanced"
    if attack_ratio >= cfg.high_attack_threshold:
        return "relaxed"
    return "standard"


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def _rank_relaxed(df: pd.DataFrame, target: str, prior: float, top_k: int) -> pd.DataFrame:
    cand = df[df["consequent_str"] == target].copy()
    if cand.empty:
        return cand
    cand["prior_prob"] = float(prior)
    cand["conf_gain"] = cand["confidence"] - prior
    cand["score_relaxed"] = cand["conf_gain"] + 0.1 * (cand["lift"] - 1.0)
    cand = cand.sort_values(
        ["score_relaxed", "confidence", "lift", "support"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return cand.head(top_k).copy()


def _rule_pool_hash(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"
    payload = df[["antecedent_str", "consequent_str"]].astype(str).to_dict(orient="records")
    s = json.dumps(payload, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def build_rule_pool_v2(
    label_rules: pd.DataFrame,
    fallback_rules_all: pd.DataFrame,
    attack_ratio: float,
    cfg: MiningConfigV2,
    force_mode: Optional[str] = None,
) -> tuple[pd.DataFrame, str, str, str]:
    """Build the mixed rule pool.

    Returns ``(pool_df, requested_mode, actual_mode_used, rule_pool_hash)``.

    * ``requested_mode``: caller's force_mode or the adaptive selector's choice.
    * ``actual_mode_used``: which branch we actually ran. When force_mode is
      provided, this MUST equal the requested mode unless an empty pool would
      be produced — in which case we report the fallback honestly.
    """
    auto_mode = decide_window_mode(attack_ratio, cfg)
    requested = force_mode if force_mode is not None else auto_mode
    if requested not in {"balanced", "standard", "relaxed"}:
        raise ValueError(f"Unknown mode: {requested}")

    attack_rules = label_rules[label_rules["consequent_str"] == LABEL_ATTACK].reset_index(drop=True)
    normal_rules = label_rules[label_rules["consequent_str"] == LABEL_NORMAL].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Build attack_top
    # ------------------------------------------------------------------
    if requested == "balanced":
        # Strict balanced: same number of attack and normal rules.
        attack_target = max(1, min(cfg.top_attack_rules, len(attack_rules)))
        normal_target = max(1, min(cfg.top_normal_rules, len(normal_rules)))
        balanced_target = min(attack_target, normal_target)
        if balanced_target < 1:
            # both empty: try fallback for both
            balanced_target = max(1, min(cfg.top_attack_rules, cfg.top_normal_rules))
        if len(attack_rules) >= balanced_target:
            attack_top = attack_rules.head(balanced_target).copy()
            attack_top["weight"] = attack_top["confidence"].apply(
                lambda x: _clipped_weight(x, cfg.attack_weight_cap)
            )
        else:
            relaxed_attack = _rank_relaxed(
                fallback_rules_all, LABEL_ATTACK, attack_ratio, balanced_target
            )
            attack_top = relaxed_attack.copy()
            if not attack_top.empty:
                attack_top["weight"] = attack_top["score_relaxed"].apply(
                    lambda x: _relaxed_weight(
                        x, cfg.relaxed_normal_min_weight, cfg.relaxed_normal_weight_scale, cfg.attack_weight_cap
                    )
                )
        if not attack_top.empty:
            attack_top["consequent_str"] = LABEL_ATTACK
            attack_top["formula"] = attack_top["antecedent_str"] + " => " + LABEL_ATTACK
            attack_top["target_label"] = 1

        if len(normal_rules) >= balanced_target:
            normal_top = normal_rules.head(balanced_target).copy()
            normal_top["weight"] = normal_top["confidence"].apply(
                lambda x: _clipped_weight(x, cfg.attack_weight_cap)
            )
        else:
            relaxed_normal = _rank_relaxed(
                fallback_rules_all, LABEL_NORMAL, 1.0 - attack_ratio, balanced_target
            )
            normal_top = relaxed_normal.copy()
            if not normal_top.empty:
                normal_top["weight"] = normal_top["score_relaxed"].apply(
                    lambda x: _relaxed_weight(
                        x,
                        cfg.relaxed_normal_min_weight,
                        cfg.relaxed_normal_weight_scale,
                        cfg.relaxed_normal_weight_cap,
                    )
                )
        if not normal_top.empty:
            normal_top["consequent_str"] = LABEL_NORMAL
            normal_top["formula"] = normal_top["antecedent_str"] + " => " + LABEL_NORMAL
            normal_top["target_label"] = 0

    elif requested == "standard":
        # Standard: top-K per side under the strict threshold; NO fallback.
        attack_top_k = min(cfg.top_attack_rules, len(attack_rules))
        normal_top_k = min(cfg.top_normal_rules, len(normal_rules))
        attack_top = attack_rules.head(attack_top_k).copy()
        normal_top = normal_rules.head(normal_top_k).copy()
        if not attack_top.empty:
            attack_top["weight"] = attack_top["confidence"].apply(
                lambda x: _clipped_weight(x, cfg.attack_weight_cap)
            )
            attack_top["formula"] = attack_top["antecedent_str"] + " => " + LABEL_ATTACK
            attack_top["target_label"] = 1
        if not normal_top.empty:
            normal_top["weight"] = normal_top["confidence"].apply(
                lambda x: _clipped_weight(x, cfg.attack_weight_cap)
            )
            normal_top["formula"] = normal_top["antecedent_str"] + " => " + LABEL_NORMAL
            normal_top["target_label"] = 0

    else:
        # relaxed mode: normal pool ALWAYS via score_relaxed
        attack_top_k = min(cfg.top_attack_rules, len(attack_rules))
        attack_top = attack_rules.head(attack_top_k).copy()
        if not attack_top.empty:
            attack_top["weight"] = attack_top["confidence"].apply(
                lambda x: _clipped_weight(x, cfg.attack_weight_cap)
            )
            attack_top["formula"] = attack_top["antecedent_str"] + " => " + LABEL_ATTACK
            attack_top["target_label"] = 1
        relaxed_normal = _rank_relaxed(
            fallback_rules_all, LABEL_NORMAL, 1.0 - attack_ratio, cfg.relaxed_normal_rules
        )
        normal_top = relaxed_normal.copy()
        if not normal_top.empty:
            normal_top["consequent_str"] = LABEL_NORMAL
            normal_top["formula"] = normal_top["antecedent_str"] + " => " + LABEL_NORMAL
            normal_top["weight"] = normal_top["score_relaxed"].apply(
                lambda x: _relaxed_weight(
                    x,
                    cfg.relaxed_normal_min_weight,
                    cfg.relaxed_normal_weight_scale,
                    cfg.relaxed_normal_weight_cap,
                )
            )
            normal_top["target_label"] = 0

    # ------------------------------------------------------------------
    # Honest fallback if forced mode produced an empty side
    # ------------------------------------------------------------------
    actual = requested
    if attack_top is None or attack_top.empty:
        # try fallback ranker
        relaxed_attack = _rank_relaxed(fallback_rules_all, LABEL_ATTACK, attack_ratio, cfg.top_attack_rules)
        attack_top = relaxed_attack.copy()
        if not attack_top.empty:
            attack_top["consequent_str"] = LABEL_ATTACK
            attack_top["formula"] = attack_top["antecedent_str"] + " => " + LABEL_ATTACK
            attack_top["weight"] = attack_top["score_relaxed"].apply(
                lambda x: _relaxed_weight(
                    x, cfg.relaxed_normal_min_weight, cfg.relaxed_normal_weight_scale, cfg.attack_weight_cap
                )
            )
            attack_top["target_label"] = 1
            actual = f"{requested}+attack_fallback"

    if normal_top is None or normal_top.empty:
        relaxed_normal = _rank_relaxed(
            fallback_rules_all, LABEL_NORMAL, 1.0 - attack_ratio, cfg.top_normal_rules
        )
        normal_top = relaxed_normal.copy()
        if not normal_top.empty:
            normal_top["consequent_str"] = LABEL_NORMAL
            normal_top["formula"] = normal_top["antecedent_str"] + " => " + LABEL_NORMAL
            normal_top["weight"] = normal_top["score_relaxed"].apply(
                lambda x: _relaxed_weight(
                    x,
                    cfg.relaxed_normal_min_weight,
                    cfg.relaxed_normal_weight_scale,
                    cfg.relaxed_normal_weight_cap,
                )
            )
            normal_top["target_label"] = 0
            actual = (actual + "+normal_fallback") if "fallback" not in actual else actual

    if attack_top is None or attack_top.empty or normal_top is None or normal_top.empty:
        raise ValueError(
            f"build_rule_pool_v2 produced empty side: requested={requested} "
            f"attack={len(attack_top) if attack_top is not None else 0} "
            f"normal={len(normal_top) if normal_top is not None else 0}"
        )

    cols = [
        "antecedent_str",
        "consequent_str",
        "formula",
        "support",
        "confidence",
        "lift",
        "weight",
        "target_label",
    ]
    pool = pd.concat([attack_top[cols], normal_top[cols]], axis=0, ignore_index=True)
    return pool, requested, actual, _rule_pool_hash(pool)


# ---------------------------------------------------------------------------
# State construction (matches paper Eq. (10))
# ---------------------------------------------------------------------------


def build_initial_state_v2(rule_pool: pd.DataFrame) -> dict:
    weights = rule_pool["weight"].to_numpy(dtype=float)
    confidences = rule_pool["confidence"].to_numpy(dtype=float)
    target_labels = rule_pool["target_label"].to_numpy(dtype=int)
    return {
        "rule_pool": rule_pool.to_dict(orient="records"),
        "active_mask": np.ones(len(rule_pool), dtype=int),
        "weights": weights.copy(),
        "initial_weights": weights.copy(),
        "rule_scores": confidences.copy(),
        "target_labels": target_labels.copy(),
        "num_rules": int(len(rule_pool)),
    }


def build_warmstart_state_dataset(state_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for the supervised warm-start.

    The state vector is the same as PaperRuleEnv produces at t=0.
    The label is ``KEEP`` (=0) iff y_i = 1 (rule is attack) and
    ``DISABLE`` (=1) otherwise — this mirrors the paper.
    """
    am = state_dict["active_mask"].astype(float)
    w = state_dict["weights"].astype(float)
    rs = state_dict["rule_scores"].astype(float)
    yl = state_dict["target_labels"].astype(int)
    n = state_dict["num_rules"]

    active_w = w[am > 0.5]
    if active_w.size == 0:
        mu_w = sigma_w = w_min = w_max = 0.0
    else:
        mu_w = float(active_w.mean())
        sigma_w = float(active_w.std())
        w_min = float(active_w.min())
        w_max = float(active_w.max())

    X = np.zeros((n, 12), dtype=np.float32)
    for i in range(n):
        denom = max(1, n - 1)
        X[i] = np.array(
            [
                float(am.mean()),
                mu_w,
                sigma_w,
                w_min,
                w_max,
                float(am.sum()),
                float(n),
                float(i) / float(denom),
                float(am[i]),
                float(w[i]),
                float(rs[i]),
                float(yl[i]),
            ],
            dtype=np.float32,
        )

    # KEEP = 0 if attack rule, DISABLE = 1 otherwise
    y = np.array([0 if yl[i] == 1 else 1 for i in range(n)], dtype=np.int64)
    return X, y
