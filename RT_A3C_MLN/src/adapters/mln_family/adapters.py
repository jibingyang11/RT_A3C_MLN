"""Adapter stubs for the eight 2023-2026 MLN-family / neuro-symbolic
rule-induction baselines compared in Table 4 of the paper.

Each class implements the :class:`MLNAdapter` protocol from
:mod:`src.pipeline.sota_mln_compare`:

    name: str
    def fit(self, state_dict: dict) -> None: ...
    def predict(self, state_dict: dict) -> list[int]: ...

These local adapters are lightweight, deterministic proxies for the
published systems.  They do not call external authors' code and they
do not read ``state_dict['target_labels']``.  Instead, each proxy
uses only rule-quality metadata available to any structure learner:
support, confidence, lift, weight, score and antecedent length.  Keep
these rows labelled as local proxies unless the ``predict`` methods
are replaced by wrappers around the real reference implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _features(state_dict: dict) -> dict[str, np.ndarray]:
    pool = list(state_dict.get('rule_pool', []))
    n = int(state_dict.get('num_rules', len(pool)))
    score = np.asarray(state_dict.get('rule_scores', np.ones(n)), dtype=float)
    weight = np.asarray(state_dict.get('weights', np.ones(n)), dtype=float)
    support = np.asarray([float(r.get('support', 0.0)) for r in pool], dtype=float)
    confidence = np.asarray([float(r.get('confidence', s)) for r, s in zip(pool, score)], dtype=float)
    lift = np.asarray([float(r.get('lift', 1.0)) for r in pool], dtype=float)
    ant_len = np.asarray([
        len(str(r.get('antecedent_str', '')).split(' & ')) if str(r.get('antecedent_str', '')) else 0
        for r in pool
    ], dtype=float)
    return {
        'score': np.nan_to_num(score, nan=0.0),
        'weight': np.nan_to_num(weight, nan=0.0),
        'support': np.nan_to_num(support, nan=0.0),
        'confidence': np.nan_to_num(confidence, nan=0.0),
        'lift': np.nan_to_num(lift, nan=1.0),
        'ant_len': np.nan_to_num(ant_len, nan=0.0),
    }


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = float(np.min(x)) if len(x) else 0.0
    hi = float(np.max(x)) if len(x) else 1.0
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _top_fraction_actions(values: np.ndarray, fraction: float) -> list[int]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return []
    k = max(1, min(n, int(round(float(fraction) * n))))
    keep_idx = np.argsort(-values)[:k]
    actions = np.ones(n, dtype=int)
    actions[keep_idx] = 0
    return actions.tolist()


def _threshold_actions(values: np.ndarray, threshold: float, min_keep: int = 1) -> list[int]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return []
    actions = np.where(values >= float(threshold), 0, 1).astype(int)
    if int((actions == 0).sum()) < min_keep:
        actions = np.asarray(_top_fraction_actions(values, min_keep / max(1, len(values))), dtype=int)
    return actions.tolist()


# ---------------------------------------------------------------------------
# 2023
# ---------------------------------------------------------------------------
@dataclass
class MLN4KBAdapter:
    """Fang et al., MLN4KB, WWW 2023.

    Reference implementation:
        https://github.com/fh-wangcheng/MLN4KB
    Wrapping notes:
        * MLN4KB ingests a knowledge base of triples plus weighted
          formulas. Convert ``state_dict['rules']`` into the YAML rule
          format used by MLN4KB.
        * Use the ``predict_link_existence`` API to score each rule
          and threshold at the median score to derive keep/disable.
    """
    name: str = 'MLN4KB'
    year: int = 2023

    def fit(self, state_dict: dict) -> None:
        return None

    def predict(self, state_dict: dict) -> list[int]:
        f = _features(state_dict)
        score = 0.55 * f['confidence'] + 0.25 * _minmax(f['lift']) + 0.20 * _minmax(f['support'])
        return _top_fraction_actions(score, 0.45)


@dataclass
class RuleExtractionADAdapter:
    """Li et al., Interpreting Unsupervised Anomaly Detection in Security
    via Rule Extraction, NeurIPS 2023.

    Reference implementation:
        Search openreview.net for the paper title (NeurIPS 2023).
    Wrapping notes:
        * The paper trains an unsupervised anomaly detector on the
          window features and then extracts a rule set from it.
        * Map every extracted rule to one of our candidate rules by
          maximum antecedent overlap, then keep the rules with
          non-zero extracted weight.
    """
    name: str = 'RuleExtractionAD'
    year: int = 2023

    def fit(self, state_dict: dict) -> None:
        return None

    def predict(self, state_dict: dict) -> list[int]:
        f = _features(state_dict)
        short_bonus = 1.0 / np.maximum(f['ant_len'], 1.0)
        score = f['confidence'] * (0.7 + 0.3 * short_bonus)
        return _threshold_actions(score, 0.72)


# ---------------------------------------------------------------------------
# 2024
# ---------------------------------------------------------------------------
@dataclass
class NPLLAdapter:
    """Shi et al., Neural Probabilistic Logic Learning (NPLL), arXiv 2024.

    Reference implementation:
        https://arxiv.org/abs/2407.03704 (supplementary code link
        in the paper).
    Wrapping notes:
        * NPLL combines KG embedding scoring with a continuous
          MLN-style PSL aggregator. Use the same FP-growth rule
          pool as input and the PSL probability of each rule's
          consequent given its antecedent as the keep/disable
          score.
    """
    name: str = 'NPLL'
    year: int = 2024

    def fit(self, state_dict: dict) -> None:
        return None

    def predict(self, state_dict: dict) -> list[int]:
        f = _features(state_dict)
        potential = 1.0 / (1.0 + np.exp(-f['weight']))
        score = potential * f['score'] + 0.15 * _minmax(f['support'])
        return _top_fraction_actions(score, 0.50)


@dataclass
class QuantifiedNeuralMLNAdapter:
    """Jung et al., Quantified Neural Markov Logic Networks, IJAR 2024.

    Reference implementation:
        https://www.sciencedirect.com/science/article/abs/pii/S0888613X24000598
        (link to supplementary code).
    Wrapping notes:
        * Compile our binarised SWaT transactions into the
          relational format used by Neural MLNs.
        * Use the learned potential as the per-rule score; keep
          the top-k by score.
    """
    name: str = 'QuantifiedNeuralMLN'
    year: int = 2024

    def fit(self, state_dict: dict) -> None:
        return None

    def predict(self, state_dict: dict) -> list[int]:
        f = _features(state_dict)
        score = f['confidence'] * np.log1p(np.maximum(f['lift'], 0.0)) * (0.7 + 0.3 * _minmax(f['support']))
        return _top_fraction_actions(score, 0.40)


@dataclass
class IncrementalAffordanceMLNAdapter:
    """Potter et al., Incremental Learning of Affordances using Markov
    Logic Networks, IRC 2024.

    Reference implementation:
        https://ieeexplore.ieee.org/document/...
    Wrapping notes:
        * The original paper learns affordances incrementally; treat
          every window as one increment.
        * Use the maintained rule weights as the keep/disable
          decision threshold.
    """
    name: str = 'IncrementalAffordanceMLN'
    year: int = 2024

    def fit(self, state_dict: dict) -> None:
        return None

    def predict(self, state_dict: dict) -> list[int]:
        f = _features(state_dict)
        score = 0.60 * f['confidence'] + 0.25 * _minmax(f['support']) + 0.15 * _minmax(f['weight'])
        return _threshold_actions(score, float(np.median(score)))


# ---------------------------------------------------------------------------
# 2025
# ---------------------------------------------------------------------------
@dataclass
class NeSyCAdapter:
    """Choi et al., NeSyC: A neuro-symbolic continual learner, 2025.

    Reference implementation:
        https://arxiv.org/abs/2503.00870 (supplementary code).
    Wrapping notes:
        * NeSyC accumulates symbolic rules across tasks. Treat each
          window as one task; expose its rule pool as the symbolic
          memory; query NeSyC for keep/disable per rule.
    """
    name: str = 'NeSyC'
    year: int = 2025

    def fit(self, state_dict: dict) -> None:
        return None

    def predict(self, state_dict: dict) -> list[int]:
        f = _features(state_dict)
        compactness = 1.0 / np.maximum(f['ant_len'], 1.0)
        score = 0.50 * f['confidence'] + 0.30 * _minmax(f['lift']) + 0.20 * compactness
        return _top_fraction_actions(score, 0.45)


@dataclass
class NeuralRuleListsAdapter:
    """Xu et al., Neural Rule Lists, NeurIPS 2025.

    Reference implementation:
        NeurIPS 2025 supplementary code.
    Wrapping notes:
        * Train Neural Rule Lists on the window's transactions, then
          map each output rule back onto our candidate rule by
          maximum antecedent overlap; keep the matched rules.
    """
    name: str = 'NeuralRuleLists'
    year: int = 2025

    def fit(self, state_dict: dict) -> None:
        return None

    def predict(self, state_dict: dict) -> list[int]:
        f = _features(state_dict)
        score = f['confidence'] - 0.04 * np.maximum(f['ant_len'] - 1.0, 0.0) + 0.10 * _minmax(f['support'])
        return _top_fraction_actions(score, 0.35)


@dataclass
class LogicGuardAdapter:
    """Gokhale et al., LogicGuard, 2025.

    Reference implementation:
        https://arxiv.org/abs/2507.03293 (supplementary code).
    Wrapping notes:
        * LogicGuard scores temporal-logic clauses with a critic.
          We use this critic as a per-rule score and keep rules
          whose critic value exceeds 0.
    """
    name: str = 'LogicGuard'
    year: int = 2025

    def fit(self, state_dict: dict) -> None:
        return None

    def predict(self, state_dict: dict) -> list[int]:
        f = _features(state_dict)
        score = 0.45 * _minmax(f['lift']) + 0.35 * f['confidence'] + 0.20 * _minmax(f['weight'])
        return _top_fraction_actions(score, 0.40)


def all_mln_family_adapters() -> Sequence:
    """Return every MLN-family adapter in the paper's row order."""
    return [
        MLN4KBAdapter(),
        RuleExtractionADAdapter(),
        NPLLAdapter(),
        QuantifiedNeuralMLNAdapter(),
        IncrementalAffordanceMLNAdapter(),
        NeSyCAdapter(),
        NeuralRuleListsAdapter(),
        LogicGuardAdapter(),
    ]
