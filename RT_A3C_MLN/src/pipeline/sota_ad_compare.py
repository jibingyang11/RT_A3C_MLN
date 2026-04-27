"""Stub driver for the deep anomaly-detector comparison (Section 6.5).

Unlike the MLN comparison (where every competitor consumes the same
rule pool), the deep anomaly detectors compared here take the raw
multivariate sensor window as input and produce a per-timestamp
anomaly label.  The comparison therefore has three moving parts:

1. An adapter that exposes each deep detector under a uniform
   interface::

        class DetectorAdapter:
            name: str
            def fit(self, train_df): ...
            def predict(self, test_df) -> np.ndarray:
                '''Return a binary anomaly label per timestamp.'''

2. A derived detector for RT-A3C-MLN: a timestamp is flagged as
   attack iff at least one kept attack-oriented rule fires on it.
   This is implemented below in ``rt_a3c_mln_detector``.

3. A scoring function (``evaluate_detector``) that computes
   precision, recall, F1 and AUROC on identical test splits.

The main entry point is ``run_sota_ad_compare``, which produces the
CSV consumed by Table 5 and Figure 7 of the paper.

Reference implementations of the twelve competitors are linked in
the docstring below and also listed in the TODO above Table 5 in
``paper/main.tex``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd


class DetectorAdapter(Protocol):
    name: str

    def fit(self, train_df: pd.DataFrame) -> None:
        ...

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        ...


def rt_a3c_mln_detector(
    test_df: pd.DataFrame,
    kept_attack_rules: list[dict],
) -> np.ndarray:
    """Derive a detection label from the kept rule mask.

    A transaction (row of ``test_df``) is flagged as attack iff at
    least one of the kept attack-oriented rules fires on it. Each
    rule in ``kept_attack_rules`` is expected to be a dict with key
    ``antecedent`` (list of ``sensor_high``/``sensor_low`` tokens)
    produced by :mod:`src.mln.rule_utils`.
    """
    n = len(test_df)
    flags = np.zeros(n, dtype=int)
    if not kept_attack_rules:
        return flags
    for rule in kept_attack_rules:
        antecedent = rule.get('antecedent', [])
        mask = np.ones(n, dtype=bool)
        for token in antecedent:
            if token in test_df.columns:
                mask &= (test_df[token].to_numpy() == 1)
            else:
                mask[:] = False
                break
        flags[mask] = 1
    return flags


def evaluate_detector(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return precision / recall / F1 / AUROC for a binary detector."""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    out = {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out['aucroc'] = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        out['aucroc'] = float('nan')
    return out


def run_sota_ad_compare(
    adapters: Sequence[DetectorAdapter],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Run a list of detector adapters on the same split.

    Parameters
    ----------
    adapters:
        Adapter instances (already imported / configured).  The
        paper assumes 12 adapters; the authors' reference
        implementations are linked in the module docstring.
    train_df, test_df:
        Feature frames for training and test respectively.
    y_test:
        Binary ground-truth labels for the test split.

    Returns
    -------
    DataFrame
        One row per adapter with columns ``method, precision,
        recall, f1, aucroc``.
    """
    rows = []
    for adapter in adapters:
        adapter.fit(train_df)
        y_pred = adapter.predict(test_df)
        metrics = evaluate_detector(y_test, y_pred)
        rows.append({
            'method': adapter.name,
            'year': int(getattr(adapter, 'year', 0) or 0),
            **metrics,
            'num_rules': 0,
        })
    return pd.DataFrame(rows)


def save_sota_ad_compare(df: pd.DataFrame, project_root: Path) -> Path:
    out = project_root / 'outputs' / 'logs' / 'sota_ad_compare.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
