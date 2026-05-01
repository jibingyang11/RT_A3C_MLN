from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import re

import numpy as np
import pandas as pd


def _sanitize(value: object, limit: int = 36) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "missing"
    return text[:limit]


@dataclass(frozen=True)
class Literal:
    column: str
    value: str

    @property
    def text(self) -> str:
        return f"{self.column}={self.value}"


@dataclass(frozen=True)
class Rule:
    literals: tuple[Literal, ...]
    score: float
    support: float

    @property
    def text(self) -> str:
        body = " AND ".join(literal.text for literal in self.literals)
        return f"Target(x)=1 <- {body}"


class RuleFeatureBuilder:
    """Generate MLN-style binary rule features from tabular predicates."""

    def __init__(
        self,
        max_rules: int = 220,
        max_single_literals: int = 80,
        numeric_bins: int = 4,
        min_support: float = 0.02,
        random_state: int = 7,
    ) -> None:
        self.max_rules = max_rules
        self.max_single_literals = max_single_literals
        self.numeric_bins = numeric_bins
        self.min_support = min_support
        self.random_state = random_state
        self.rules_: list[Rule] = []
        self._numeric_edges: dict[str, np.ndarray] = {}
        self._literal_columns: list[Literal] = []
        self._literal_to_index: dict[Literal, int] = {}

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "RuleFeatureBuilder":
        xp = self._fit_preprocess(x)
        literal_matrix, literals = self._build_literal_matrix(xp)
        self._literal_columns = literals
        self._literal_to_index = {literal: idx for idx, literal in enumerate(literals)}
        literal_scores = self._score_columns(literal_matrix, y)

        single_candidates: list[Rule] = []
        for idx, literal in enumerate(literals):
            support = float(literal_matrix[:, idx].mean())
            if support >= self.min_support:
                single_candidates.append(Rule((literal,), literal_scores[idx], support))
        single_candidates.sort(key=lambda rule: rule.score, reverse=True)
        top_singles = single_candidates[: self.max_single_literals]

        pair_candidates: list[Rule] = []
        top_indices = [self._literal_to_index[rule.literals[0]] for rule in top_singles]
        for left, right in combinations(top_indices, 2):
            lit_left = literals[left]
            lit_right = literals[right]
            if lit_left.column == lit_right.column:
                continue
            col = literal_matrix[:, left] & literal_matrix[:, right]
            support = float(col.mean())
            if support < self.min_support:
                continue
            score = self._score_vector(col.astype(np.uint8), y)
            pair_candidates.append(Rule((lit_left, lit_right), score, support))

        candidates = single_candidates + pair_candidates
        candidates.sort(key=lambda rule: (rule.score, rule.support), reverse=True)
        self.rules_ = candidates[: self.max_rules]
        return self

    def fit_transform(self, x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        self.fit(x, y)
        return self.transform(x)

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if not self.rules_:
            raise RuntimeError("RuleFeatureBuilder is not fitted.")
        xp = self._preprocess(x)
        literal_matrix, literals = self._build_literal_matrix(xp, known_literals=self._literal_columns)
        literal_to_index = {literal: idx for idx, literal in enumerate(literals)}
        matrix = np.zeros((len(x), len(self.rules_)), dtype=np.uint8)
        for rule_idx, rule in enumerate(self.rules_):
            active = np.ones(len(x), dtype=bool)
            for literal in rule.literals:
                literal_idx = literal_to_index.get(literal)
                if literal_idx is None:
                    active &= False
                else:
                    active &= literal_matrix[:, literal_idx].astype(bool)
            matrix[:, rule_idx] = active.astype(np.uint8)
        return matrix

    def _fit_preprocess(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = x.copy()
        self._numeric_edges = {}
        for col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce")
            numeric_ratio = series.notna().mean()
            if numeric_ratio > 0.95 and series.nunique(dropna=True) > self.numeric_bins + 1:
                quantiles = np.linspace(0, 1, self.numeric_bins + 1)
                edges = np.unique(np.nanquantile(series.to_numpy(dtype=float), quantiles))
                if len(edges) > 2:
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                    self._numeric_edges[col] = edges
        return self._preprocess(frame)

    def _preprocess(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = x.copy()
        for col in frame.columns:
            if col in self._numeric_edges:
                series = pd.to_numeric(frame[col], errors="coerce")
                bins = pd.cut(series, bins=self._numeric_edges[col], labels=False, include_lowest=True)
                frame[col] = bins.fillna(-1).astype(int).map(lambda val: f"bin_{val}")
            else:
                frame[col] = frame[col].astype(str).fillna("missing").map(_sanitize)
        frame.columns = [_sanitize(col) for col in frame.columns]
        return frame

    def _build_literal_matrix(
        self,
        x: pd.DataFrame,
        known_literals: list[Literal] | None = None,
    ) -> tuple[np.ndarray, list[Literal]]:
        if known_literals is None:
            literals: list[Literal] = []
            columns: list[np.ndarray] = []
            for col in x.columns:
                counts = x[col].value_counts(dropna=False)
                values = counts.index.tolist()
                for value in values:
                    literal = Literal(col, str(value))
                    literals.append(literal)
                    columns.append((x[col].to_numpy() == value).astype(np.uint8))
        else:
            literals = known_literals
            columns = []
            for literal in literals:
                if literal.column in x.columns:
                    columns.append((x[literal.column].astype(str).to_numpy() == literal.value).astype(np.uint8))
                else:
                    columns.append(np.zeros(len(x), dtype=np.uint8))
        if not columns:
            return np.zeros((len(x), 0), dtype=np.uint8), []
        return np.column_stack(columns).astype(np.uint8), literals

    def _score_columns(self, matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
        scores = np.zeros(matrix.shape[1], dtype=float)
        for idx in range(matrix.shape[1]):
            scores[idx] = self._score_vector(matrix[:, idx], y)
        return scores

    def _score_vector(self, feature: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-9
        feature = feature.astype(bool)
        y_bool = y.astype(bool)
        support = feature.mean()
        if support <= 0.0 or support >= 1.0:
            return 0.0
        p_y = y_bool.mean()
        p_f = support
        p_fy = (feature & y_bool).mean()
        p_f_not_y = (feature & ~y_bool).mean()
        p_not_f_y = (~feature & y_bool).mean()
        p_not_f_not_y = (~feature & ~y_bool).mean()
        terms = [
            (p_fy, p_f * p_y),
            (p_f_not_y, p_f * (1 - p_y)),
            (p_not_f_y, (1 - p_f) * p_y),
            (p_not_f_not_y, (1 - p_f) * (1 - p_y)),
        ]
        mi = 0.0
        for observed, expected in terms:
            if observed > eps and expected > eps:
                mi += observed * math.log(observed / expected)
        lift = abs((p_fy / (p_f + eps)) - p_y)
        return float(mi + 0.25 * lift)
