"""Adapter stubs for the twelve 2023-2026 deep anomaly detectors compared
in Table 5 of the paper.

Each class implements the :class:`DetectorAdapter` protocol from
:mod:`src.pipeline.sota_ad_compare`:

    name: str
    def fit(self, train_df: pd.DataFrame) -> None: ...
    def predict(self, test_df: pd.DataFrame) -> np.ndarray: ...

The implementations below are lightweight local proxy detectors.  They
do not call the authors' code; each maps the multivariate binary sensor
window to an anomaly score and thresholds the highest-scoring rows.
Keep the table labelled as local proxies unless these methods are
replaced by wrappers around the real reference implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


def _numeric(test_df: pd.DataFrame) -> np.ndarray:
    X = test_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if X.size == 0:
        return np.zeros((len(test_df), 1), dtype=float)
    return np.nan_to_num(X, nan=0.0)


def _score(test_df: pd.DataFrame, mode: str) -> np.ndarray:
    X = _numeric(test_df)
    centered = X - X.mean(axis=0, keepdims=True)
    if mode == 'l1':
        return np.mean(np.abs(centered), axis=1)
    if mode == 'l2':
        return np.sqrt(np.mean(centered * centered, axis=1))
    if mode == 'max':
        return np.max(np.abs(centered), axis=1)
    if mode == 'rare':
        p = np.clip(X.mean(axis=0, keepdims=True), 1e-3, 1.0 - 1e-3)
        return -np.mean(X * np.log(p) + (1.0 - X) * np.log(1.0 - p), axis=1)
    if mode == 'transition':
        if len(X) <= 1:
            return np.zeros(len(X), dtype=float)
        diff = np.vstack([np.zeros((1, X.shape[1])), np.abs(np.diff(X, axis=0))])
        return np.mean(diff, axis=1)
    return np.mean(centered * centered, axis=1)


def _quantile_predict(test_df: pd.DataFrame, mode: str, attack_fraction: float) -> np.ndarray:
    values = _score(test_df, mode)
    if len(values) == 0:
        return np.zeros(0, dtype=int)
    frac = float(np.clip(attack_fraction, 0.01, 0.99))
    threshold = np.quantile(values, 1.0 - frac)
    return (values >= threshold).astype(int)


# ---------------------------------------------------------------------------
# 2023
# ---------------------------------------------------------------------------
@dataclass
class TimesNetAdapter:
    """Wu et al., TimesNet, ICLR 2023.

    Reference implementation:
        https://github.com/thuml/Time-Series-Library
    """
    name: str = 'TimesNet'
    year: int = 2023

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'l2', 0.30)


@dataclass
class DCdetectorAdapter:
    """Yang et al., DCdetector, KDD 2023.

    Reference implementation:
        https://github.com/DAMO-DI-ML/KDD2023-DCdetector
    """
    name: str = 'DCdetector'
    year: int = 2023

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'max', 0.34)


@dataclass
class MEMTOAdapter:
    """Song et al., MEMTO, NeurIPS 2023.

    Reference implementation:
        https://github.com/gunny97/MEMTO
    """
    name: str = 'MEMTO'
    year: int = 2023

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'rare', 0.36)


# ---------------------------------------------------------------------------
# 2024
# ---------------------------------------------------------------------------
@dataclass
class iTransformerAdapter:
    """Liu et al., iTransformer, ICLR 2024.

    Reference implementation:
        https://github.com/thuml/iTransformer
    """
    name: str = 'iTransformer'
    year: int = 2024

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'transition', 0.28)


# ---------------------------------------------------------------------------
# 2025
# ---------------------------------------------------------------------------
@dataclass
class TiTADAdapter:
    """Wang et al., TiTAD, Electronics 2025.

    Reference implementation:
        Contact authors of https://www.mdpi.com/2079-9292/14/7/1401
    """
    name: str = 'TiTAD'
    year: int = 2025

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'l1', 0.32)


@dataclass
class AdvPertTransformerAdapter:
    """Alasmari & Alhogail, AdvPert-Transformer, Electronics 2025.

    Reference implementation:
        Contact authors of https://www.mdpi.com/2079-9292/14/6/1094
    """
    name: str = 'AdvPertTransformer'
    year: int = 2025

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'max', 0.40)


@dataclass
class TEPAdapter:
    """Xu et al., Pre-training Enhanced Transformer (TEP),
    Information Fusion 2025.

    Reference implementation:
        https://dl.acm.org/doi/10.1016/j.inffus.2025.103171
        (contact authors).
    """
    name: str = 'TEP'
    year: int = 2025

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'rare', 0.42)


@dataclass
class PredTrADAdapter:
    """Schuster et al., PredTrAD, Interspeech 2025.

    Reference implementation:
        https://www.isca-archive.org/interspeech_2025/schuster25_interspeech.html
    """
    name: str = 'PredTrAD'
    year: int = 2025

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'transition', 0.35)


@dataclass
class OracleADAdapter:
    """Kim et al., OracleAD / Structured Temporal Causality,
    NeurIPS 2025.

    Reference implementation:
        https://arxiv.org/abs/2510.16511
    """
    name: str = 'OracleAD'
    year: int = 2025

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'l2', 0.38)


@dataclass
class GDSTAESVDDAdapter:
    """Li et al., GD-ST-AE-SVDD, Expert Systems With Applications 2025.

    Reference implementation:
        https://doi.org/10.1016/j.eswa.2025.126573 (contact authors).
    """
    name: str = 'GD-ST-AE-SVDD'
    year: int = 2025

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'l1', 0.44)


@dataclass
class HybridMFAdapter:
    """Ouafiq et al., Multi-Feature Hybrid AD,
    ACM SecureDL Workshop 2025.

    Reference implementation:
        https://dl.acm.org/doi/10.1145/3709021.3737669
    """
    name: str = 'Hybrid-MF'
    year: int = 2025

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'rare', 0.46)


# ---------------------------------------------------------------------------
# 2026
# ---------------------------------------------------------------------------
@dataclass
class EMATTransEdgeAdapter:
    """Zhang et al., EM-AT / TransEdge,
    Journal of Intelligent Information Systems 2026.

    Reference implementation:
        https://doi.org/10.1007/s10844-026-01043-w
    """
    name: str = 'EM-AT-TransEdge'
    year: int = 2026

    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _quantile_predict(test_df, 'max', 0.48)


def all_detector_family_adapters() -> Sequence:
    """Return every deep-detector adapter in the paper's row order."""
    return [
        TimesNetAdapter(),
        DCdetectorAdapter(),
        MEMTOAdapter(),
        iTransformerAdapter(),
        TiTADAdapter(),
        AdvPertTransformerAdapter(),
        TEPAdapter(),
        PredTrADAdapter(),
        OracleADAdapter(),
        GDSTAESVDDAdapter(),
        HybridMFAdapter(),
        EMATTransEdgeAdapter(),
    ]
