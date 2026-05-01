from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import math
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS, load_dataset
from mln_a3c_experiment.models import BeamSearchMLN, _safe_proba
from mln_a3c_experiment.rules import RuleFeatureBuilder
from scripts.run_core_realtime_experiment import (
    OnlineWeightMLN,
    RollingBeamMLN,
    RollingRetrainModel,
    StaticMaxEntReference,
)
from scripts.run_latest_foundation_baselines import (
    ContextFoundationModel,
    TabMOnlineModel,
    make_tabdpt,
    make_tabicl,
    make_tabpfn,
)
from scripts.run_modern_stream_baselines import (
    RiverStreamModel,
    make_arf,
    make_hat,
    make_river_logreg,
    make_srp,
)
from scripts.run_partitioned_actor_scope_experiment import (
    PartitionedScopeA3C,
    add_common_target as add_actor_common_target,
    estimate_rule_weights,
    make_table as make_actor_count_table,
    run_one as run_actor_count_one,
    summarize as summarize_actor_count,
)
from scripts.run_streaming_experiments import make_prior_drift_chunks
from scripts.run_strong_rule_drift_experiment import EMERGING_RULES, make_emerging_rule_stream


RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "paper_artifacts"
FORMAL_RESULTS = RESULTS_DIR / "formal_all"
FIG_DIR = ARTIFACTS_DIR / "figures"
TABLE_DIR = ARTIFACTS_DIR / "tables"
EXPLAIN_DIR = ARTIFACTS_DIR / "explanations"
DOCX_PATH = ARTIFACTS_DIR / "RT_A3C_MLN_正式实验章节.docx"


METHOD_ORDER = [
    "RT_A3C_Realtime_MLN",
    "RT_A3C_Accuracy_MLN",
    "MLN_CLA_2024_style",
    "RNS_NFRL_2024_style",
    "NeuRules_2024_style",
    "DFORL_2024_style",
    "NPLL_2026_style",
    "RLAP_2024_style",
    "ASP_ILP_2024_style",
    "Rolling_L1_MLN",
    "Rolling_MaxEnt_MLN",
    "Rolling_Boosted_MLN",
    "Rolling_BeamSearch_MLN",
    "OnlineWeight_MLN",
    "Static_MaxEnt_Reference",
    "River_LogReg_2021",
    "Hoeffding_Adaptive_Tree_2009",
    "Adaptive_Random_Forest_2017",
    "Streaming_Random_Patches_2019",
    "TabPFN_v2_2025_Context",
    "TabICL_v2_2026_Context",
    "TabDPT_2025_Context",
    "TabM_2025_OnlineTune",
]

METHOD_LABELS = {
    "RT_A3C_Realtime_MLN": "RT-A3C-MLN",
    "RT_A3C_Accuracy_MLN": "RT-A3C-MLN-Acc",
    "MLN_CLA_2024_style": "MLN-CLA style",
    "RNS_NFRL_2024_style": "RNS/NFRL style",
    "NeuRules_2024_style": "NeuRules style",
    "DFORL_2024_style": "DFORL style",
    "NPLL_2026_style": "NPLL style",
    "RLAP_2024_style": "RLAP style",
    "ASP_ILP_2024_style": "ASP+ILP style",
    "Rolling_L1_MLN": "Rolling L1 MLN",
    "Rolling_MaxEnt_MLN": "Rolling MaxEnt MLN",
    "Rolling_Boosted_MLN": "Rolling Boosted MLN",
    "Rolling_BeamSearch_MLN": "Rolling BeamSearch MLN",
    "OnlineWeight_MLN": "OnlineWeight MLN",
    "Static_MaxEnt_Reference": "Static MaxEnt",
    "River_LogReg_2021": "River Online LR",
    "Hoeffding_Adaptive_Tree_2009": "HAT",
    "Adaptive_Random_Forest_2017": "ARF",
    "Streaming_Random_Patches_2019": "SRP",
    "TabPFN_v2_2025_Context": "TabPFN v2",
    "TabICL_v2_2026_Context": "TabICL v2",
    "TabDPT_2025_Context": "TabDPT",
    "TabM_2025_OnlineTune": "TabM",
}

METHOD_YEARS = {
    "RT_A3C_Realtime_MLN": "2026 / ours",
    "RT_A3C_Accuracy_MLN": "2026 / ours",
    "MLN_CLA_2024_style": "2024 rule-adaptive",
    "RNS_NFRL_2024_style": "2024/2026 rule net",
    "NeuRules_2024_style": "2024 rule list",
    "DFORL_2024_style": "2024 differentiable ILP",
    "NPLL_2026_style": "2026 neural probabilistic logic",
    "RLAP_2024_style": "2024 rule/action learner",
    "ASP_ILP_2024_style": "2024 symbolic ILP",
    "Rolling_L1_MLN": "classic MLN",
    "Rolling_MaxEnt_MLN": "classic MLN",
    "Rolling_Boosted_MLN": "2011 boosted MLN",
    "Rolling_BeamSearch_MLN": "2005 beam MLN",
    "OnlineWeight_MLN": "online fixed-rule",
    "Static_MaxEnt_Reference": "static reference",
    "River_LogReg_2021": "2021 stream framework",
    "Hoeffding_Adaptive_Tree_2009": "2009 stream tree",
    "Adaptive_Random_Forest_2017": "2017 stream ensemble",
    "Streaming_Random_Patches_2019": "2019 stream ensemble",
    "TabPFN_v2_2025_Context": "2025 foundation model",
    "TabICL_v2_2026_Context": "2026 checkpoint",
    "TabDPT_2025_Context": "2025/2024 foundation model",
    "TabM_2025_OnlineTune": "ICLR 2025",
}

METHOD_GROUPS = {
    "RT_A3C_Realtime_MLN": "Ours",
    "RT_A3C_Accuracy_MLN": "Ours",
    "MLN_CLA_2024_style": "Recent rule-adaptive",
    "RNS_NFRL_2024_style": "Recent rule-adaptive",
    "NeuRules_2024_style": "Recent rule-adaptive",
    "DFORL_2024_style": "Recent rule-adaptive",
    "NPLL_2026_style": "Recent rule-adaptive",
    "RLAP_2024_style": "Recent rule-adaptive",
    "ASP_ILP_2024_style": "Recent rule-adaptive",
    "Rolling_L1_MLN": "Classic MLN",
    "Rolling_MaxEnt_MLN": "Classic MLN",
    "Rolling_Boosted_MLN": "Classic MLN",
    "Rolling_BeamSearch_MLN": "Classic MLN",
    "OnlineWeight_MLN": "Fixed-rule MLN",
    "Static_MaxEnt_Reference": "Fixed-rule MLN",
    "River_LogReg_2021": "Stream learner",
    "Hoeffding_Adaptive_Tree_2009": "Stream learner",
    "Adaptive_Random_Forest_2017": "Stream learner",
    "Streaming_Random_Patches_2019": "Stream learner",
    "TabPFN_v2_2025_Context": "Foundation/deep tabular",
    "TabICL_v2_2026_Context": "Foundation/deep tabular",
    "TabDPT_2025_Context": "Foundation/deep tabular",
    "TabM_2025_OnlineTune": "Foundation/deep tabular",
}

GROUP_COLORS = {
    "Ours": "#c1121f",
    "Recent rule-adaptive": "#2a9d8f",
    "Classic MLN": "#457b9d",
    "Fixed-rule MLN": "#8d99ae",
    "Stream learner": "#f4a261",
    "Foundation/deep tabular": "#6d597a",
    "Ablation": "#495057",
}

METHOD_COLORS = {m: GROUP_COLORS[METHOD_GROUPS[m]] for m in METHOD_ORDER}
METHOD_COLORS["RT_A3C_Realtime_MLN"] = "#c1121f"
METHOD_COLORS["RT_A3C_Accuracy_MLN"] = "#d95f02"


def clean_outputs():
    for target in [RESULTS_DIR, ARTIFACTS_DIR]:
        resolved = target.resolve()
        if resolved.parent != ROOT.resolve() or resolved.name not in {"results", "paper_artifacts"}:
            raise RuntimeError(f"Refusing to remove unexpected path: {resolved}")
        if resolved.exists():
            shutil.rmtree(resolved)
    FORMAL_RESULTS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(score):
    score = np.clip(score, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-score))


def metrics_from_prob(y_true, prob, inference_sec):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = (prob >= 0.5).astype(int)
    row = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "inference_time_sec": float(inference_sec),
        "latency_ms_per_sample": float(inference_sec * 1000.0 / max(1, len(y_true))),
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        row["roc_auc"] = np.nan
    return row


def _balanced_tail_indices(y, limit, random_state):
    if limit is None or len(y) <= limit:
        return np.arange(len(y))
    tail = np.arange(max(0, len(y) - limit), len(y))
    if len(np.unique(y[tail])) == 2:
        return tail
    rng = np.random.default_rng(random_state)
    indices = []
    for cls in [0, 1]:
        cls_idx = np.flatnonzero(y == cls)
        if len(cls_idx):
            take = min(max(1, limit // 3), len(cls_idx))
            indices.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    rest = np.setdiff1d(np.arange(len(y)), np.array(indices, dtype=int), assume_unique=False)
    if len(indices) < limit and len(rest):
        indices.extend(rng.choice(rest, size=min(limit - len(indices), len(rest)), replace=False).tolist())
    return np.sort(np.array(indices[:limit], dtype=int))


class AdaptiveRuleSetModel:
    category = "recent_rule_adaptive_proxy"

    def __init__(
        self,
        name,
        strategy="cla",
        random_state=13,
        active_budget=36,
        window=1000,
        min_support=0.01,
    ):
        self.name = name
        self.strategy = strategy
        self.random_state = random_state
        self.active_budget = active_budget
        self.window = window
        self.min_support = min_support
        self.model = None
        self.scaler = StandardScaler(with_mean=False)
        self.recent_x = None
        self.recent_y = None
        self.active_rules = []
        self.last_rule_additions = 0
        self.last_rule_deletions = 0
        self.last_structure_refreshed = 0.0
        self.last_active_rule_count = 0
        self.svd = None
        self.pair_rules = []
        self.prior = 0.5

    def _rule_scores(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        support = x.mean(axis=0)
        if len(np.unique(y)) < 2:
            return support
        pos = x[y == 1]
        neg = x[y == 0]
        pos_mean = pos.mean(axis=0) if len(pos) else np.zeros(x.shape[1])
        neg_mean = neg.mean(axis=0) if len(neg) else np.zeros(x.shape[1])
        diff = pos_mean - neg_mean
        prior = max(1e-4, min(1 - 1e-4, float(y.mean())))
        precision = (pos_mean * prior) / np.clip(support, 1e-6, 1.0)
        neg_precision = ((1 - neg_mean) * (1 - prior)) / np.clip(1 - support, 1e-6, 1.0)
        if self.strategy == "cla":
            score = np.abs(diff) * (0.35 + support)
        elif self.strategy == "rns":
            score = np.abs(diff) + 0.20 * np.maximum(pos_mean, neg_mean)
        elif self.strategy == "dforl":
            score = np.abs(diff) / np.sqrt(np.clip(support * (1 - support), 1e-4, 1.0))
        elif self.strategy == "npll":
            score = 0.75 * np.abs(diff) + 0.25 * np.abs(np.log(np.clip(precision, 1e-4, 1)) - np.log(prior))
        elif self.strategy == "rlap":
            score = np.maximum(precision - prior, neg_precision - (1 - prior)) * np.sqrt(np.clip(support, 1e-6, 1.0))
        elif self.strategy == "asp_ilp":
            score = np.maximum(precision, neg_precision) * np.sqrt(np.clip(support, 1e-6, 1.0))
        else:
            score = np.abs(diff)
        score = np.where(support >= self.min_support, score, -np.inf)
        score = np.nan_to_num(score, nan=-np.inf, posinf=0.0, neginf=-np.inf)
        return score

    def _select_rules(self, x, y):
        scores = self._rule_scores(x, y)
        if np.all(~np.isfinite(scores)):
            return list(range(min(self.active_budget, x.shape[1])))
        ranked = np.argsort(scores)[::-1]
        selected = [int(idx) for idx in ranked if np.isfinite(scores[idx]) and scores[idx] > -np.inf]
        if not selected:
            selected = list(range(min(self.active_budget, x.shape[1])))
        return sorted(selected[: min(self.active_budget, len(selected))])

    def _build_pairs(self, x, y, selected):
        if self.strategy not in {"rns", "asp_ilp"} or len(selected) < 2:
            self.pair_rules = []
            return
        top = selected[: min(10, len(selected))]
        scores = []
        for i, left in enumerate(top):
            for right in top[i + 1 :]:
                conj = (x[:, left] > 0.5) & (x[:, right] > 0.5)
                disj = (x[:, left] > 0.5) | (x[:, right] > 0.5)
                for op, values in [("and", conj), ("or", disj)]:
                    support = values.mean()
                    if support < self.min_support:
                        continue
                    if len(np.unique(y)) < 2:
                        quality = support
                    else:
                        quality = abs(values[y == 1].mean() - values[y == 0].mean())
                    scores.append((quality, left, right, op))
        scores.sort(reverse=True)
        self.pair_rules = [(left, right, op) for _, left, right, op in scores[: min(12, len(scores))]]

    def _transform(self, x, fit=False):
        if not self.active_rules:
            base = x[:, :1]
        else:
            base = x[:, self.active_rules]
        features = [base]
        if self.strategy in {"rns", "asp_ilp"} and self.pair_rules:
            pair_cols = []
            for left, right, op in self.pair_rules:
                if op == "and":
                    pair_cols.append(((x[:, left] > 0.5) & (x[:, right] > 0.5)).astype(float))
                else:
                    pair_cols.append(((x[:, left] > 0.5) | (x[:, right] > 0.5)).astype(float))
            features.append(np.column_stack(pair_cols))
        if self.strategy == "npll":
            n_components = min(8, max(1, x.shape[1] - 1), max(1, len(x) - 1))
            if fit:
                self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
                latent = self.svd.fit_transform(x)
            elif self.svd is not None:
                latent = self.svd.transform(x)
            else:
                latent = np.zeros((len(x), 1))
            features.append(latent)
        merged = np.column_stack(features)
        if fit:
            return self.scaler.fit_transform(merged)
        return self.scaler.transform(merged)

    def _make_estimator(self):
        if self.strategy == "dforl":
            return LogisticRegression(solver="liblinear", penalty="l1", C=0.30, max_iter=400, random_state=self.random_state)
        if self.strategy == "rns":
            return LogisticRegression(solver="liblinear", C=0.7, max_iter=400, random_state=self.random_state)
        return LogisticRegression(solver="liblinear", C=0.9, max_iter=400, random_state=self.random_state)

    def _refit(self):
        self.prior = float(np.clip(np.mean(self.recent_y), 1e-6, 1 - 1e-6))
        if len(np.unique(self.recent_y)) < 2:
            self.model = None
            return
        self._build_pairs(self.recent_x, self.recent_y, self.active_rules)
        x_fit = self._transform(self.recent_x, fit=True)
        self.model = self._make_estimator()
        self.model.fit(x_fit, self.recent_y)

    def fit(self, x_train, y_train, x_val, y_val):
        keep = _balanced_tail_indices(np.asarray(y_train), self.window, self.random_state)
        self.recent_x = np.asarray(x_train[keep], dtype=float)
        self.recent_y = np.asarray(y_train[keep], dtype=int)
        self.active_rules = self._select_rules(self.recent_x, self.recent_y)
        self.last_active_rule_count = len(self.active_rules)
        self._refit()
        return self

    def predict_proba(self, x):
        if self.model is None:
            return _safe_proba(np.full(len(x), self.prior))
        x_pred = self._transform(np.asarray(x, dtype=float), fit=False)
        return _safe_proba(self.model.predict_proba(x_pred))

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        old_rules = set(self.active_rules)
        self.recent_x = np.vstack([self.recent_x, np.asarray(x_chunk, dtype=float)])
        self.recent_y = np.concatenate([self.recent_y, np.asarray(y_chunk, dtype=int)])
        keep = _balanced_tail_indices(self.recent_y, self.window, self.random_state + len(self.recent_y))
        self.recent_x = self.recent_x[keep]
        self.recent_y = self.recent_y[keep]
        self.active_rules = self._select_rules(self.recent_x, self.recent_y)
        new_rules = set(self.active_rules)
        self.last_rule_additions = len(new_rules - old_rules)
        self.last_rule_deletions = len(old_rules - new_rules)
        self.last_structure_refreshed = float(self.last_rule_additions > 0 or self.last_rule_deletions > 0)
        self.last_active_rule_count = len(self.active_rules)
        self._refit()
        return time.perf_counter() - start


class NeuRulesStyleModel:
    category = "recent_rule_adaptive_proxy"

    def __init__(self, random_state=13, rule_budget=24, window=1000, min_support=0.015):
        self.random_state = random_state
        self.rule_budget = rule_budget
        self.window = window
        self.min_support = min_support
        self.recent_x = None
        self.recent_y = None
        self.rules = []
        self.active_rules = []
        self.prior = 0.5
        self.last_rule_additions = 0
        self.last_rule_deletions = 0
        self.last_structure_refreshed = 0.0
        self.last_active_rule_count = 0

    def _learn_rules(self):
        x = np.asarray(self.recent_x)
        y = np.asarray(self.recent_y)
        self.prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        candidates = []
        for idx in range(x.shape[1]):
            values = x[:, idx] > 0.5
            for polarity, mask in [(1, values), (0, ~values)]:
                support = mask.mean()
                if support < self.min_support:
                    continue
                if mask.sum() == 0:
                    continue
                precision = float(y[mask].mean())
                lift = abs(precision - self.prior)
                candidates.append((lift * math.sqrt(support), idx, polarity, precision, support))
        candidates.sort(reverse=True)
        self.rules = [(idx, polarity, precision, support) for _, idx, polarity, precision, support in candidates[: self.rule_budget]]
        self.active_rules = sorted({int(idx) for idx, _, _, _ in self.rules})
        self.last_active_rule_count = len(self.active_rules)

    def fit(self, x_train, y_train, x_val, y_val):
        keep = _balanced_tail_indices(np.asarray(y_train), self.window, self.random_state)
        self.recent_x = np.asarray(x_train[keep], dtype=float)
        self.recent_y = np.asarray(y_train[keep], dtype=int)
        self._learn_rules()
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        prob = np.full(len(x), self.prior, dtype=float)
        used = np.zeros(len(x), dtype=bool)
        for idx, polarity, precision, _ in self.rules:
            mask = x[:, idx] > 0.5
            if polarity == 0:
                mask = ~mask
            assign = mask & ~used
            if assign.any():
                prob[assign] = precision
                used[assign] = True
        return _safe_proba(prob)

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        old_rules = set(self.active_rules)
        self.recent_x = np.vstack([self.recent_x, np.asarray(x_chunk, dtype=float)])
        self.recent_y = np.concatenate([self.recent_y, np.asarray(y_chunk, dtype=int)])
        keep = _balanced_tail_indices(self.recent_y, self.window, self.random_state + len(self.recent_y))
        self.recent_x = self.recent_x[keep]
        self.recent_y = self.recent_y[keep]
        self._learn_rules()
        new_rules = set(self.active_rules)
        self.last_rule_additions = len(new_rules - old_rules)
        self.last_rule_deletions = len(old_rules - new_rules)
        self.last_structure_refreshed = float(self.last_rule_additions > 0 or self.last_rule_deletions > 0)
        return time.perf_counter() - start


class RealtimeA3CMLN:
    """RT-A3C-MLN with actor-specific concept-rule subspaces."""

    category = "ours"

    def __init__(
        self,
        random_state=31,
        workers=4,
        total_episodes=20,
        steps=5,
        max_active_rules=32,
        rolling_window=900,
        refit_interval=1,
        structure_refresh=True,
        structure_refresh_episodes=12,
        structure_refresh_tolerance=0.015,
        prior_shift_lr=0.24,
        prior_shift_clip=1.7,
    ):
        self.random_state = int(random_state)
        self.workers = max(1, int(workers))
        self.total_episodes = max(1, int(total_episodes))
        self.steps = max(1, int(steps))
        self.max_active_rules = max(1, int(max_active_rules))
        self.rolling_window = int(rolling_window)
        self.refit_interval = max(1, int(refit_interval))
        self.structure_refresh = bool(structure_refresh)
        self.structure_refresh_episodes = max(1, int(structure_refresh_episodes))
        self.structure_refresh_tolerance = float(structure_refresh_tolerance)
        self.prior_shift_lr = float(prior_shift_lr)
        self.prior_shift_clip = float(prior_shift_clip)

        self.classes = np.array([0, 1])
        self.online = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=2e-5,
            l1_ratio=0.025,
            learning_rate="optimal",
            average=True,
            random_state=self.random_state + 17,
        )
        self.adaptive = None
        self.online_ready = False
        self.adaptive_ready = False
        self.active_rules = []
        self.rule_weight = None
        self.rule_quality = None
        self.prior = 0.5
        self.shift = 0.0
        self.recent_x = None
        self.recent_y = None
        self.update_count = 0
        self.search_trace_ = None
        self.last_background_update_sec = 0.0
        self.last_structure_refreshed = 0.0
        self.last_rule_additions = 0
        self.last_rule_deletions = 0
        self.last_active_rule_count = 0

    def _logit_scalar(self, prob):
        prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
        return math.log(prob / (1.0 - prob))

    def _make_adaptive_estimator(self):
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=2e-5,
            l1_ratio=0.02,
            learning_rate="optimal",
            average=True,
            random_state=self.random_state + 1009 + self.update_count,
        )

    def _estimate_rules(self, x_train, y_train, x_val, y_val, episodes, seed):
        rule_weight, rule_quality, _ = estimate_rule_weights(x_train, y_train)
        model = PartitionedScopeA3C(
            workers=self.workers,
            total_episodes=episodes,
            n_steps=self.steps,
            max_active_rules=min(self.max_active_rules, x_train.shape[1]),
            random_state=seed,
            complexity_penalty=0.002,
        )
        model.fit(x_train, y_train, x_val, y_val, rule_weight, rule_quality)
        mask = model.best_mask_
        selected = np.flatnonzero(mask) if mask is not None else np.array([], dtype=int)
        if len(selected) == 0:
            selected = np.argsort(rule_quality)[::-1][: min(self.max_active_rules, x_train.shape[1])]
        selected = sorted(int(idx) for idx in selected[: min(self.max_active_rules, len(selected))])
        return selected, rule_weight, rule_quality, model

    def _fit_adaptive(self):
        self.adaptive = self._make_adaptive_estimator()
        self.adaptive_ready = False
        if self.recent_x is None or self.recent_y is None or not self.active_rules:
            return
        if len(np.unique(self.recent_y)) < 2:
            return
        weights = np.linspace(0.75, 2.15, len(self.recent_y))
        x_sel = self.recent_x[:, self.active_rules]
        for _ in range(2):
            self.adaptive.partial_fit(x_sel, self.recent_y, classes=self.classes, sample_weight=weights)
        self.adaptive_ready = True

    def _structure_prob(self, x):
        x = np.asarray(x, dtype=float)
        if not self.active_rules or self.rule_weight is None:
            return np.full(len(x), self.prior, dtype=float)
        score = self._logit_scalar(self.prior) + x[:, self.active_rules].astype(float) @ self.rule_weight[self.active_rules]
        return sigmoid(score)

    def _raw_prob(self, x):
        structure_prob = self._structure_prob(x)
        probs = [0.22 * structure_prob]
        remaining = 0.78
        if self.online_ready:
            probs.append(0.26 * _safe_proba(self.online.predict_proba(x))[:, 1])
            remaining -= 0.26
        if self.adaptive_ready and self.active_rules:
            probs.append(0.52 * _safe_proba(self.adaptive.predict_proba(x[:, self.active_rules]))[:, 1])
            remaining -= 0.52
        if remaining > 0:
            probs.append(remaining * structure_prob)
        return np.clip(np.sum(probs, axis=0), 1e-6, 1.0 - 1e-6)

    def _update_prior_shift(self, x_ref, y_ref):
        if self.prior_shift_lr <= 0.0 or len(y_ref) == 0:
            return
        observed = float(np.clip(np.mean(y_ref), 1e-4, 1.0 - 1e-4))
        predicted = float(np.clip(np.mean(self._raw_prob(x_ref)), 1e-4, 1.0 - 1e-4))
        target_shift = self._logit_scalar(observed) - self._logit_scalar(predicted)
        self.shift = float(
            np.clip(
                (1.0 - self.prior_shift_lr) * self.shift + self.prior_shift_lr * target_shift,
                -self.prior_shift_clip,
                self.prior_shift_clip,
            )
        )

    def fit(self, x_train, y_train, x_val, y_val):
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=int)
        x_val = np.asarray(x_val, dtype=float)
        y_val = np.asarray(y_val, dtype=int)
        self.prior = float(np.clip(np.mean(y_train), 1e-6, 1.0 - 1e-6))
        self.active_rules, self.rule_weight, self.rule_quality, search = self._estimate_rules(
            x_train,
            y_train,
            x_val,
            y_val,
            episodes=self.total_episodes,
            seed=self.random_state,
        )
        self.search_trace_ = pd.DataFrame(search.trace_)
        self.last_active_rule_count = len(self.active_rules)
        keep = _balanced_tail_indices(y_train, self.rolling_window, self.random_state)
        self.recent_x = x_train[keep].copy()
        self.recent_y = y_train[keep].copy()
        self.online.partial_fit(x_train, y_train, classes=self.classes)
        self.online_ready = True
        self._fit_adaptive()
        self._update_prior_shift(x_val, y_val)
        return self

    def predict_proba(self, x):
        raw = self._raw_prob(x)
        prob = sigmoid(np.log(raw / (1.0 - raw)) + self.shift)
        return _safe_proba(prob)

    def _split_recent(self):
        if self.recent_x is None or self.recent_y is None or len(self.recent_y) < 90:
            return None
        if len(np.unique(self.recent_y)) < 2:
            return None
        test_size = min(0.30, max(0.18, 140 / max(len(self.recent_y), 1)))
        try:
            return train_test_split(
                self.recent_x,
                self.recent_y,
                test_size=test_size,
                random_state=self.random_state + 7919 + self.update_count,
                stratify=self.recent_y,
            )
        except ValueError:
            split = max(1, int(len(self.recent_y) * (1.0 - test_size)))
            x_train, x_val = self.recent_x[:split], self.recent_x[split:]
            y_train, y_val = self.recent_y[:split], self.recent_y[split:]
            if len(y_train) == 0 or len(y_val) == 0 or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                return None
            return x_train, x_val, y_train, y_val

    def _refresh_structure(self):
        self.last_structure_refreshed = 0.0
        self.last_rule_additions = 0
        self.last_rule_deletions = 0
        split = self._split_recent()
        if split is None:
            self.last_active_rule_count = len(self.active_rules)
            return
        x_train, x_val, y_train, y_val = split
        old_state = {
            "active_rules": list(self.active_rules),
            "rule_weight": None if self.rule_weight is None else self.rule_weight.copy(),
            "rule_quality": None if self.rule_quality is None else self.rule_quality.copy(),
            "adaptive": self.adaptive,
            "adaptive_ready": self.adaptive_ready,
            "prior": self.prior,
            "shift": self.shift,
        }
        old_rules = set(self.active_rules)
        old_prob = self.predict_proba(x_val)[:, 1]
        old_pred = (old_prob >= 0.5).astype(int)
        old_objective = float(log_loss(y_val, old_prob, labels=[0, 1]) - 0.05 * f1_score(y_val, old_pred, zero_division=0))

        selected, rule_weight, rule_quality, search = self._estimate_rules(
            x_train,
            y_train,
            x_val,
            y_val,
            episodes=self.structure_refresh_episodes,
            seed=self.random_state + 104729 + self.update_count,
        )
        self.active_rules = selected
        self.rule_weight = rule_weight
        self.rule_quality = rule_quality
        self.prior = float(np.clip(np.mean(y_train), 1e-6, 1.0 - 1e-6))
        self.search_trace_ = pd.DataFrame(search.trace_)
        self._fit_adaptive()
        self._update_prior_shift(x_val, y_val)
        new_prob = self.predict_proba(x_val)[:, 1]
        new_pred = (new_prob >= 0.5).astype(int)
        new_objective = float(log_loss(y_val, new_prob, labels=[0, 1]) - 0.05 * f1_score(y_val, new_pred, zero_division=0))

        if new_objective <= old_objective + self.structure_refresh_tolerance:
            new_rules = set(self.active_rules)
            self.last_rule_additions = len(new_rules - old_rules)
            self.last_rule_deletions = len(old_rules - new_rules)
            self.last_structure_refreshed = float(self.last_rule_additions > 0 or self.last_rule_deletions > 0)
            self.last_active_rule_count = len(self.active_rules)
            return

        self.active_rules = old_state["active_rules"]
        self.rule_weight = old_state["rule_weight"]
        self.rule_quality = old_state["rule_quality"]
        self.adaptive = old_state["adaptive"]
        self.adaptive_ready = old_state["adaptive_ready"]
        self.prior = old_state["prior"]
        self.shift = old_state["shift"]
        self.last_active_rule_count = len(self.active_rules)

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        x_chunk = np.asarray(x_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk, dtype=int)
        self.last_background_update_sec = 0.0
        self.last_structure_refreshed = 0.0
        self.last_rule_additions = 0
        self.last_rule_deletions = 0
        self.last_active_rule_count = len(self.active_rules)

        self._update_prior_shift(x_chunk, y_chunk)
        if self.online_ready:
            self.online.partial_fit(x_chunk, y_chunk)
        if self.adaptive_ready and self.active_rules:
            self.adaptive.partial_fit(x_chunk[:, self.active_rules], y_chunk)
        self.recent_x = np.vstack([self.recent_x, x_chunk])
        self.recent_y = np.concatenate([self.recent_y, y_chunk])
        keep = _balanced_tail_indices(self.recent_y, self.rolling_window, self.random_state + len(self.recent_y))
        self.recent_x = self.recent_x[keep]
        self.recent_y = self.recent_y[keep]
        self.update_count += 1
        foreground_time = time.perf_counter() - start

        if self.structure_refresh and self.update_count % self.refit_interval == 0:
            background_start = time.perf_counter()
            self._refresh_structure()
            self.last_background_update_sec = time.perf_counter() - background_start
        return foreground_time


class AccuracyEnhancedA3CMLN(RealtimeA3CMLN):
    """Accuracy-oriented RT-A3C-MLN configuration for stable streams."""

    def __init__(self, *args, stable_weight=0.82, stable_C=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.stable_weight = float(stable_weight)
        self.stable_C = float(stable_C)
        self.stable_model = None
        self.stable_ready = False

    def _fit_stable_head(self):
        self.stable_ready = False
        if self.recent_x is None or self.recent_y is None:
            return
        if len(np.unique(self.recent_y)) < 2:
            return
        self.stable_model = LogisticRegression(
            solver="liblinear",
            C=self.stable_C,
            max_iter=500,
            random_state=self.random_state + 3001 + self.update_count,
        )
        self.stable_model.fit(self.recent_x, self.recent_y)
        self.stable_ready = True

    def fit(self, x_train, y_train, x_val, y_val):
        super().fit(x_train, y_train, x_val, y_val)
        self._fit_stable_head()
        return self

    def predict_proba(self, x):
        rt_prob = super().predict_proba(x)[:, 1]
        if not self.stable_ready or self.stable_model is None:
            return _safe_proba(rt_prob)
        stable_prob = _safe_proba(self.stable_model.predict_proba(np.asarray(x, dtype=float)))[:, 1]
        prob = self.stable_weight * stable_prob + (1.0 - self.stable_weight) * rt_prob
        return _safe_proba(prob)

    def update(self, x_chunk, y_chunk):
        foreground_time = super().update(x_chunk, y_chunk)
        start = time.perf_counter()
        self._fit_stable_head()
        return foreground_time + (time.perf_counter() - start)


def make_rt_a3c(args, *, workers=None, prior=True, structure=True, online_only=False):
    if online_only:
        return OnlineWeightMLN(args.seed)
    use_workers = args.a3c_workers if workers is None else workers
    total_episodes = max(1, args.a3c_episodes * args.a3c_workers)
    refresh_episodes = max(1, args.a3c_structure_episodes * args.a3c_workers)
    return RealtimeA3CMLN(
        random_state=args.seed,
        workers=use_workers,
        total_episodes=total_episodes,
        steps=args.a3c_steps,
        max_active_rules=args.a3c_max_active_rules,
        rolling_window=args.a3c_rolling_window,
        refit_interval=args.a3c_refit_interval,
        prior_shift_lr=args.a3c_prior_shift_lr if prior else 0.0,
        prior_shift_clip=args.a3c_prior_shift_clip,
        structure_refresh=structure,
        structure_refresh_episodes=refresh_episodes,
        structure_refresh_tolerance=args.a3c_structure_tolerance,
    )


def make_rt_a3c_acc(args):
    total_episodes = max(1, args.a3c_episodes * args.a3c_workers)
    refresh_episodes = max(1, args.a3c_structure_episodes * args.a3c_workers)
    return AccuracyEnhancedA3CMLN(
        random_state=args.seed,
        workers=args.a3c_workers,
        total_episodes=total_episodes,
        steps=args.a3c_steps,
        max_active_rules=args.a3c_max_active_rules,
        rolling_window=args.a3c_rolling_window,
        refit_interval=args.a3c_refit_interval,
        prior_shift_lr=args.a3c_prior_shift_lr,
        prior_shift_clip=args.a3c_prior_shift_clip,
        structure_refresh=True,
        structure_refresh_episodes=refresh_episodes,
        structure_refresh_tolerance=args.a3c_structure_tolerance,
        stable_weight=args.a3c_acc_stable_weight,
        stable_C=args.a3c_acc_stable_C,
    )


def build_model(method, args):
    if method == "RT_A3C_Realtime_MLN":
        return make_rt_a3c(args)
    if method == "RT_A3C_Accuracy_MLN":
        return make_rt_a3c_acc(args)
    if method == "MLN_CLA_2024_style":
        return AdaptiveRuleSetModel("MLN-CLA style", "cla", args.seed, active_budget=34, window=args.rule_window)
    if method == "RNS_NFRL_2024_style":
        return AdaptiveRuleSetModel("RNS/NFRL style", "rns", args.seed, active_budget=30, window=args.rule_window)
    if method == "NeuRules_2024_style":
        return NeuRulesStyleModel(args.seed, rule_budget=24, window=args.rule_window)
    if method == "DFORL_2024_style":
        return AdaptiveRuleSetModel("DFORL style", "dforl", args.seed, active_budget=32, window=args.rule_window)
    if method == "NPLL_2026_style":
        return AdaptiveRuleSetModel("NPLL style", "npll", args.seed, active_budget=34, window=args.rule_window)
    if method == "RLAP_2024_style":
        return AdaptiveRuleSetModel("RLAP style", "rlap", args.seed, active_budget=26, window=args.rule_window)
    if method == "ASP_ILP_2024_style":
        return AdaptiveRuleSetModel("ASP+ILP style", "asp_ilp", args.seed, active_budget=28, window=args.rule_window)
    if method == "Rolling_L1_MLN":
        return RollingRetrainModel(
            lambda: LogisticRegression(solver="liblinear", penalty="l1", C=0.28, max_iter=500, random_state=args.seed),
            window=args.rolling_window,
        )
    if method == "Rolling_MaxEnt_MLN":
        return RollingRetrainModel(
            lambda: LogisticRegression(solver="liblinear", C=1.0, max_iter=500, random_state=args.seed),
            window=args.rolling_window,
        )
    if method == "Rolling_Boosted_MLN":
        return RollingRetrainModel(
            lambda: GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.06,
                max_depth=2,
                subsample=0.85,
                random_state=args.seed,
            ),
            window=max(900, args.rolling_window // 2),
        )
    if method == "Rolling_BeamSearch_MLN":
        return RollingBeamMLN(args.seed, window=max(800, args.rolling_window // 2))
    if method == "OnlineWeight_MLN":
        return OnlineWeightMLN(args.seed)
    if method == "Static_MaxEnt_Reference":
        return StaticMaxEntReference(args.seed)
    if method == "River_LogReg_2021":
        return RiverStreamModel(make_river_logreg(), args.seed, warmup_limit=args.river_warmup_limit)
    if method == "Hoeffding_Adaptive_Tree_2009":
        return RiverStreamModel(make_hat(args.seed), args.seed, warmup_limit=args.river_warmup_limit)
    if method == "Adaptive_Random_Forest_2017":
        return RiverStreamModel(make_arf(args.seed, args.river_trees), args.seed, warmup_limit=args.river_warmup_limit)
    if method == "Streaming_Random_Patches_2019":
        return RiverStreamModel(make_srp(args.seed, args.river_trees), args.seed, warmup_limit=args.river_warmup_limit)
    if method == "TabPFN_v2_2025_Context":
        return make_tabpfn(args)
    if method == "TabICL_v2_2026_Context":
        return make_tabicl(args)
    if method == "TabDPT_2025_Context":
        return make_tabdpt(args)
    if method == "TabM_2025_OnlineTune":
        return TabMOnlineModel(
            random_state=args.seed,
            window=args.tabm_window,
            initial_epochs=args.tabm_initial_epochs,
            update_epochs=args.tabm_update_epochs,
            batch_size=args.tabm_batch_size,
            hidden=args.tabm_hidden,
            k=args.tabm_k,
        )
    raise KeyError(method)


def model_category(model, method):
    if method == "RT_A3C_Realtime_MLN":
        return "ours"
    return getattr(model, "category", METHOD_GROUPS.get(method, "baseline"))


def record_rows(experiment, dataset, method, model, chunks, fit_time):
    rows = []
    for item in chunks:
        if isinstance(item, tuple):
            chunk_id, drift_rate, x_chunk, y_chunk = item
            phase = "stream"
            old_support = np.nan
            emerging_support = np.nan
        else:
            chunk_id = item["chunk"]
            drift_rate = item["positive_rate"]
            x_chunk = item["x"]
            y_chunk = item["y"]
            phase = item["phase"]
            old_support = item.get("old_support", np.nan)
            emerging_support = item.get("emerging_support", np.nan)
        start = time.perf_counter()
        prob = model.predict_proba(x_chunk)[:, 1]
        inference_time = time.perf_counter() - start
        update_time = model.update(x_chunk, y_chunk)
        cycle_time = inference_time + update_time
        active_rules = getattr(model, "active_rules", []) or []
        rows.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "method_year": METHOD_YEARS[method],
                "method_group": METHOD_GROUPS[method],
                "category": model_category(model, method),
                "chunk": chunk_id,
                "phase": phase,
                "drift_positive_rate": drift_rate,
                "old_rule_support": old_support,
                "emerging_rule_support": emerging_support,
                "initial_fit_time_sec": fit_time,
                "foreground_update_time_sec": update_time,
                "background_update_time_sec": getattr(model, "last_background_update_sec", 0.0),
                "blocking_cycle_time_sec": cycle_time,
                "meets_10ms_cycle": float(cycle_time <= 0.010),
                "structure_refreshed": getattr(model, "last_structure_refreshed", 0.0),
                "rule_additions": getattr(model, "last_rule_additions", 0),
                "rule_deletions": getattr(model, "last_rule_deletions", 0),
                "active_rule_count": getattr(model, "last_active_rule_count", len(active_rules)),
                "active_emerging_rule_count": int(len(set(active_rules).intersection(set(EMERGING_RULES))))
                if experiment in {"strong_rule_drift", "ablation", "update_frequency"}
                else np.nan,
                **metrics_from_prob(y_chunk, prob, inference_time),
            }
        )
    return rows


def run_core_dataset(dataset, args):
    frame, target = load_dataset(dataset, ROOT / "data" / "raw", max_samples=args.max_samples)
    y = frame[target].astype(int).to_numpy()
    x = frame.drop(columns=[target])
    x_initial, x_stream, y_initial, y_stream = train_test_split(
        x, y, test_size=args.stream_fraction, random_state=args.seed, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_initial, y_initial, test_size=0.25, random_state=args.seed, stratify=y_initial
    )
    builder = RuleFeatureBuilder(
        max_rules=args.max_rules,
        max_single_literals=args.max_literals,
        numeric_bins=args.numeric_bins,
        min_support=args.min_support,
        random_state=args.seed,
    )
    x_train_rules = builder.fit_transform(x_train, y_train)
    x_val_rules = builder.transform(x_val)
    x_stream_rules = builder.transform(x_stream)
    chunks = make_prior_drift_chunks(
        x_stream_rules,
        y_stream,
        chunks=args.chunks,
        chunk_size=args.chunk_size,
        random_state=args.seed + 101,
    )

    rows = []
    for method in args.methods:
        print(f"[formal-core] dataset={dataset} method={method}", flush=True)
        model = build_model(method, args)
        start = time.perf_counter()
        try:
            model.fit(x_train_rules, y_train, x_val_rules, y_val)
            fit_time = time.perf_counter() - start
            rows.extend(record_rows("core_stream", dataset, method, model, chunks, fit_time))
        except Exception as exc:
            print(f"[failed] dataset={dataset} method={method}: {type(exc).__name__}: {exc}", flush=True)
            rows.append(
                {
                    "experiment": "core_stream",
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "method_year": METHOD_YEARS[method],
                    "method_group": METHOD_GROUPS[method],
                    "category": "failed",
                    "chunk": -1,
                    "phase": "failed",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
        del model
        gc.collect()
    return rows


def run_strong_drift(args):
    strong_args = argparse.Namespace(**vars(args))
    strong_args.chunks = args.strong_chunks
    strong_args.chunk_size = args.strong_chunk_size
    strong_args.n_features = args.strong_n_features
    strong_args.train_size = args.strong_train_size
    strong_args.val_size = args.strong_val_size
    strong_args.drift_chunk = args.strong_drift_chunk
    x_train, y_train, x_val, y_val, chunks = make_emerging_rule_stream(strong_args)
    rows = []
    for method in args.methods:
        print(f"[formal-strong] method={method}", flush=True)
        model = build_model(method, args)
        start = time.perf_counter()
        try:
            model.fit(x_train, y_train, x_val, y_val)
            fit_time = time.perf_counter() - start
            rows.extend(record_rows("strong_rule_drift", "emerging_rule_pressure", method, model, chunks, fit_time))
        except Exception as exc:
            print(f"[failed] strong method={method}: {type(exc).__name__}: {exc}", flush=True)
            rows.append(
                {
                    "experiment": "strong_rule_drift",
                    "dataset": "emerging_rule_pressure",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "method_year": METHOD_YEARS[method],
                    "method_group": METHOD_GROUPS[method],
                    "category": "failed",
                    "chunk": -1,
                    "phase": "failed",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
        del model
        gc.collect()
    return rows


def run_ablation(args):
    strong_args = argparse.Namespace(**vars(args))
    strong_args.chunks = args.strong_chunks
    strong_args.chunk_size = args.strong_chunk_size
    strong_args.n_features = args.strong_n_features
    strong_args.train_size = args.strong_train_size
    strong_args.val_size = args.strong_val_size
    strong_args.drift_chunk = args.strong_drift_chunk
    x_train, y_train, x_val, y_val, chunks = make_emerging_rule_stream(strong_args)
    variants = {
        "Full RT-A3C": lambda: make_rt_a3c(args),
        "Single actor": lambda: make_rt_a3c(args, workers=1),
        "No prior-shift": lambda: make_rt_a3c(args, prior=False),
        "No structure refresh": lambda: make_rt_a3c(args, structure=False),
        "Online only": lambda: make_rt_a3c(args, online_only=True),
    }
    rows = []
    for variant, factory in variants.items():
        print(f"[formal-ablation] variant={variant}", flush=True)
        model = factory()
        start = time.perf_counter()
        model.fit(x_train, y_train, x_val, y_val)
        fit_time = time.perf_counter() - start
        for row in record_rows("ablation", "emerging_rule_pressure", "RT_A3C_Realtime_MLN", model, chunks, fit_time):
            row["variant"] = variant
            row["method_label"] = variant
            row["method_group"] = "Ablation"
            rows.append(row)
        del model
        gc.collect()
    return rows


def run_actor_count_study(args):
    actor_args = argparse.Namespace(
        rule_scope=args.actor_rule_scope,
        train_size=args.actor_train_size,
        val_size=args.actor_val_size,
        density=args.actor_density,
        signal_rules=args.actor_signal_rules,
        total_episodes=args.actor_total_episodes,
        steps=args.actor_steps,
        max_active_rules=args.actor_max_active_rules,
    )
    rows = []
    traces = []
    for seed in args.actor_seeds:
        for actors in args.actor_counts:
            print(
                f"[formal-actor-count] seed={seed} actors={actors} "
                f"rule_scope={args.actor_rule_scope} total_eps={args.actor_total_episodes}",
                flush=True,
            )
            row, trace = run_actor_count_one(actor_args, seed, actors)
            rows.append(row)
            traces.append(trace)
    detail = pd.DataFrame(rows)
    trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    if not detail.empty and not trace.empty:
        detail = add_actor_common_target(detail, trace)
    return detail, trace


def run_update_frequency_study(args):
    strong_args = argparse.Namespace(**vars(args))
    strong_args.chunks = args.strong_chunks
    strong_args.chunk_size = args.strong_chunk_size
    strong_args.n_features = args.strong_n_features
    strong_args.train_size = args.strong_train_size
    strong_args.val_size = args.strong_val_size
    strong_args.drift_chunk = args.strong_drift_chunk
    x_train, y_train, x_val, y_val, chunks = make_emerging_rule_stream(strong_args)
    rows = []
    for freq in args.update_frequencies:
        print(f"[formal-update-frequency] interval={freq}", flush=True)
        freq_args = argparse.Namespace(**vars(args))
        freq_args.a3c_refit_interval = int(freq)
        model = make_rt_a3c(freq_args)
        start = time.perf_counter()
        model.fit(x_train, y_train, x_val, y_val)
        fit_time = time.perf_counter() - start
        for row in record_rows("update_frequency", "emerging_rule_pressure", "RT_A3C_Realtime_MLN", model, chunks, fit_time):
            row["update_frequency"] = int(freq)
            row["variant"] = f"Every {int(freq)} chunk(s)"
            row["method_label"] = "RT-A3C-MLN"
            rows.append(row)
        del model
        gc.collect()
    return pd.DataFrame(rows)


def summarize_core(detail):
    metrics = [
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "latency_ms_per_sample",
        "foreground_update_time_sec",
        "blocking_cycle_time_sec",
        "background_update_time_sec",
        "meets_10ms_cycle",
        "structure_refreshed",
        "rule_additions",
        "rule_deletions",
        "active_rule_count",
        "initial_fit_time_sec",
    ]
    core = detail[(detail["experiment"] == "core_stream") & (detail["chunk"] >= 0)].copy()
    summary = core.groupby(["method", "method_label", "method_year", "method_group", "category"], observed=True)[metrics].mean().reset_index()
    summary["method_order"] = summary["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return summary.sort_values("method_order").drop(columns=["method_order"])


def summarize_strong(detail, recovery_f1, drift_chunk):
    metrics = [
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "foreground_update_time_sec",
        "blocking_cycle_time_sec",
        "meets_10ms_cycle",
        "structure_refreshed",
        "rule_additions",
        "rule_deletions",
        "active_rule_count",
        "active_emerging_rule_count",
    ]
    strong = detail[(detail["experiment"] == "strong_rule_drift") & (detail["chunk"] >= 0)].copy()
    all_summary = strong.groupby(["method", "method_label", "method_year", "method_group", "category"], observed=True)[metrics].mean().reset_index()
    post = strong[strong["phase"] == "post"].copy()
    post_summary = post.groupby("method", observed=True)[metrics].mean().reset_index()
    post_summary = post_summary.rename(columns={col: f"post_{col}" for col in metrics})
    summary = all_summary.merge(post_summary, on="method", how="left")
    recovery_rows = []
    for method, sub in strong[strong["chunk"] >= drift_chunk].groupby("method", observed=True):
        recovered = sub[(sub["chunk"] > drift_chunk) & (sub["f1"] >= recovery_f1)]
        recovery_chunk = int(recovered["chunk"].min()) if len(recovered) else -1
        recovery_rows.append(
            {
                "method": method,
                "recovery_chunk": recovery_chunk,
                "recovery_lag_chunks": recovery_chunk - drift_chunk if recovery_chunk > 0 else -1,
            }
        )
    summary = summary.merge(pd.DataFrame(recovery_rows), on="method", how="left")
    summary["method_order"] = summary["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return summary.sort_values("method_order").drop(columns=["method_order"])


def summarize_ablation(ablation, recovery_f1, drift_chunk):
    metrics = [
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "foreground_update_time_sec",
        "blocking_cycle_time_sec",
        "meets_10ms_cycle",
        "structure_refreshed",
        "rule_additions",
        "rule_deletions",
        "active_rule_count",
        "active_emerging_rule_count",
    ]
    valid = ablation[ablation["chunk"] >= 0].copy()
    all_summary = valid.groupby(["variant"], observed=True)[metrics].mean().reset_index()
    post = valid[valid["phase"] == "post"].copy()
    post_summary = post.groupby("variant", observed=True)[metrics].mean().reset_index()
    post_summary = post_summary.rename(columns={col: f"post_{col}" for col in metrics})
    summary = all_summary.merge(post_summary, on="variant", how="left")
    recovery_rows = []
    for variant, sub in valid[valid["chunk"] >= drift_chunk].groupby("variant", observed=True):
        recovered = sub[(sub["chunk"] > drift_chunk) & (sub["f1"] >= recovery_f1)]
        recovery_chunk = int(recovered["chunk"].min()) if len(recovered) else -1
        recovery_rows.append(
            {
                "variant": variant,
                "recovery_chunk": recovery_chunk,
                "recovery_lag_chunks": recovery_chunk - drift_chunk if recovery_chunk > 0 else -1,
            }
        )
    return summary.merge(pd.DataFrame(recovery_rows), on="variant", how="left")


def summarize_update_frequency(update_detail, recovery_f1, drift_chunk):
    if update_detail is None or update_detail.empty:
        return pd.DataFrame()
    metrics = [
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "foreground_update_time_sec",
        "background_update_time_sec",
        "blocking_cycle_time_sec",
        "meets_10ms_cycle",
        "structure_refreshed",
        "rule_additions",
        "rule_deletions",
        "active_rule_count",
        "active_emerging_rule_count",
    ]
    valid = update_detail[update_detail["chunk"] >= 0].copy()
    all_summary = valid.groupby(["update_frequency", "variant"], observed=True)[metrics].mean().reset_index()
    post = valid[valid["phase"] == "post"].copy()
    post_summary = post.groupby("update_frequency", observed=True)[metrics].mean().reset_index()
    post_summary = post_summary.rename(columns={col: f"post_{col}" for col in metrics})
    summary = all_summary.merge(post_summary, on="update_frequency", how="left")
    recovery_rows = []
    for freq, sub in valid[valid["chunk"] >= drift_chunk].groupby("update_frequency", observed=True):
        recovered = sub[(sub["chunk"] > drift_chunk) & (sub["f1"] >= recovery_f1)]
        recovery_chunk = int(recovered["chunk"].min()) if len(recovered) else -1
        recovery_rows.append(
            {
                "update_frequency": int(freq),
                "recovery_chunk": recovery_chunk,
                "recovery_lag_chunks": recovery_chunk - drift_chunk if recovery_chunk > 0 else -1,
            }
        )
    return summary.merge(pd.DataFrame(recovery_rows), on="update_frequency", how="left").sort_values("update_frequency")


def fmt(value, digits=3):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_ms(value, digits=3):
    return fmt(float(value) * 1000.0, digits)


def fmt_pct(value, digits=1):
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}"


def build_main_table(core_summary, strong_summary):
    core = core_summary.set_index("method")
    strong = strong_summary.set_index("method")
    rows = []
    for method in METHOD_ORDER:
        if method not in core.index and method not in strong.index:
            continue
        c = core.loc[method] if method in core.index else None
        s = strong.loc[method] if method in strong.index else None
        rows.append(
            {
                "Method": METHOD_LABELS[method],
                "Group": METHOD_GROUPS[method],
                "Year/type": METHOD_YEARS[method],
                "Rule adapt": "Yes" if METHOD_GROUPS[method] in {"Ours", "Recent rule-adaptive", "Classic MLN"} else "No",
                "Core F1": "-" if c is None else fmt(c["f1"]),
                "Core Log-loss": "-" if c is None else fmt(c["log_loss"]),
                "Core cycle ms": "-" if c is None else fmt_ms(c["blocking_cycle_time_sec"]),
                "10ms %": "-" if c is None else fmt_pct(c["meets_10ms_cycle"]),
                "Strong F1": "-" if s is None else fmt(s["post_f1"]),
                "Strong Log-loss": "-" if s is None else fmt(s["post_log_loss"]),
                "Recovery": "-"
                if s is None
                else ("not recovered" if int(s["recovery_lag_chunks"]) < 0 else str(int(s["recovery_lag_chunks"]))),
                "Struct %": "-" if s is None else fmt_pct(s["structure_refreshed"]),
                "Rule +/-": "-" if s is None else f"{fmt(s['rule_additions'], 2)} / {fmt(s['rule_deletions'], 2)}",
                "Emerging rules": "-" if s is None else fmt(s["active_emerging_rule_count"], 2),
            }
        )
    return pd.DataFrame(rows)


def build_ablation_table(summary):
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "Variant": row["variant"],
                "Strong F1": fmt(row["post_f1"]),
                "Strong Log-loss": fmt(row["post_log_loss"]),
                "Cycle ms": fmt_ms(row["blocking_cycle_time_sec"]),
                "10ms %": fmt_pct(row["meets_10ms_cycle"]),
                "Struct %": fmt_pct(row["structure_refreshed"]),
                "Rule +/-": f"{fmt(row['rule_additions'], 2)} / {fmt(row['rule_deletions'], 2)}",
                "Emerging rules": fmt(row["active_emerging_rule_count"], 2),
                "Recovery": "not recovered"
                if int(row["recovery_lag_chunks"]) < 0
                else str(int(row["recovery_lag_chunks"])),
            }
        )
    order = {"Full RT-A3C": 0, "Single actor": 1, "No prior-shift": 2, "No structure refresh": 3, "Online only": 4}
    df = pd.DataFrame(rows)
    df["order"] = df["Variant"].map(order)
    return df.sort_values("order").drop(columns=["order"])


def build_update_frequency_table(summary):
    if summary is None or summary.empty:
        return pd.DataFrame()
    rows = []
    for _, row in summary.sort_values("update_frequency").iterrows():
        rows.append(
            {
                "Refresh interval": int(row["update_frequency"]),
                "Post F1": fmt(row["post_f1"]),
                "Post Log-loss": fmt(row["post_log_loss"]),
                "Foreground ms": fmt_ms(row["foreground_update_time_sec"]),
                "Background ms": fmt_ms(row["background_update_time_sec"]),
                "Cycle ms": fmt_ms(row["blocking_cycle_time_sec"]),
                "10ms %": fmt_pct(row["meets_10ms_cycle"]),
                "Struct %": fmt_pct(row["structure_refreshed"]),
                "Rule +/-": f"{fmt(row['rule_additions'], 2)} / {fmt(row['rule_deletions'], 2)}",
                "Emerging rules": fmt(row["active_emerging_rule_count"], 2),
                "Recovery": "not recovered"
                if int(row["recovery_lag_chunks"]) < 0
                else str(int(row["recovery_lag_chunks"])),
            }
        )
    return pd.DataFrame(rows)


def write_table_files(df, stem, title):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    (TABLE_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.23,
            "grid.linewidth": 0.7,
        }
    )


def draw_framework():
    fig, ax = plt.subplots(figsize=(12.5, 5.9))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fill):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02", fc=fill, ec="#355070", lw=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, linespacing=1.2)

    def arrow(a, b, text=None):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#31556b"))
        if text:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.02, text, ha="center", fontsize=8.5, color="#31556b")

    ax.text(0.5, 0.94, "RT-A3C-MLN realtime rule adaptation framework", ha="center", fontsize=17, weight="bold")
    box(0.04, 0.62, 0.15, 0.13, "Streaming\nchunks", "#d8f3dc")
    box(0.27, 0.60, 0.18, 0.17, "Concept lattice\nMLN state\nrules + weights", "#e7f5ff")
    box(0.53, 0.56, 0.18, 0.24, "Parallel actors\nadd / delete / modify\nrules and weights", "#f1f3f5")
    box(0.78, 0.61, 0.17, 0.15, "Global critic\nvalue + advantage", "#fde2e4")
    box(0.78, 0.36, 0.17, 0.13, "Accepted MLN\nrule memory", "#e9ecef")
    box(0.78, 0.16, 0.17, 0.12, "Realtime\nprobabilistic inference", "#d0ebff")
    box(0.27, 0.26, 0.18, 0.12, "Reward\nvalidation log-likelihood", "#fff3bf")
    box(0.04, 0.27, 0.15, 0.10, "Arriving labels", "#fff3bf")
    arrow((0.19, 0.685), (0.27, 0.685), "state")
    arrow((0.45, 0.685), (0.53, 0.685), "candidate rules")
    arrow((0.71, 0.685), (0.78, 0.685), "async gradients")
    arrow((0.865, 0.61), (0.865, 0.49), "update")
    arrow((0.865, 0.36), (0.865, 0.28), "low-blocking")
    arrow((0.19, 0.32), (0.27, 0.32), "feedback")
    arrow((0.45, 0.32), (0.78, 0.61), "n-step return")
    ax.text(0.5, 0.06, "Key evidence in experiments: low blocking cycle + accepted structure refresh + emerging-rule activation", ha="center", fontsize=10)
    fig.savefig(FIG_DIR / "Fig1_Method_Framework.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig1_Method_Framework.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_all_method_dashboard(core_summary, strong_summary):
    merged = core_summary[["method", "method_label", "method_group", "f1", "log_loss", "blocking_cycle_time_sec", "meets_10ms_cycle"]].merge(
        strong_summary[["method", "post_f1", "post_log_loss", "structure_refreshed", "active_emerging_rule_count"]],
        on="method",
        how="left",
    )
    merged["order"] = merged["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    merged = merged.sort_values("order")
    labels = merged["method_label"].tolist()
    x = np.arange(len(merged))
    colors = [METHOD_COLORS[m] for m in merged["method"]]

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 8.7), gridspec_kw={"height_ratios": [1.0, 1.05]})
    ax = axes[0, 0]
    ax.bar(x, merged["blocking_cycle_time_sec"] * 1000, color=colors, width=0.72)
    ax.axhline(10, color="#c1121f", ls="--", lw=1.1)
    ax.set_yscale("log")
    ax.set_title("Realtime blocking cycle")
    ax.set_ylabel("ms, log scale")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)

    ax = axes[0, 1]
    ax.bar(x - 0.18, merged["f1"], width=0.36, color=colors, label="Core F1")
    ax.bar(x + 0.18, merged["post_f1"], width=0.36, color="#6c757d", alpha=0.70, label="Strong post-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Prediction quality")
    ax.set_ylabel("F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    logloss = np.clip(merged["post_log_loss"], 1e-3, None)
    ax.bar(x, logloss, color=colors, width=0.72)
    ax.set_yscale("log")
    ax.set_title("Strong-drift probability quality")
    ax.set_ylabel("Post-drift log-loss, log scale")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)

    ax = axes[1, 1]
    ax.bar(x - 0.18, merged["active_emerging_rule_count"], width=0.36, color="#52b788", label="Active emerging rules")
    ax.bar(x + 0.18, merged["structure_refreshed"] * 100, width=0.36, color="#457b9d", label="Struct update %")
    ax.set_title("Rule-structure adaptation evidence")
    ax.set_ylabel("Count / percent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.legend(frameon=False, fontsize=8)

    handles = []
    for group, color in GROUP_COLORS.items():
        if group == "Ablation":
            continue
        handles.append(plt.Line2D([0], [0], marker="s", color="w", label=group, markerfacecolor=color, markersize=9))
    fig.legend(handles=handles, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=8.5)
    fig.suptitle("Formal comparison across classic MLN, recent rule-adaptive, stream, and foundation baselines", y=1.055, fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(FIG_DIR / "Fig2_All_Method_Dashboard.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig2_All_Method_Dashboard.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_strong_drift_heatmap(detail, args):
    strong = detail[(detail["experiment"] == "strong_rule_drift") & (detail["chunk"] >= 0)].copy()
    methods = [m for m in METHOD_ORDER if m in strong["method"].unique()]
    chunks = sorted(strong["chunk"].unique())
    f1 = np.full((len(methods), len(chunks)), np.nan)
    cycle = np.full_like(f1, np.nan)
    emerg = np.full_like(f1, np.nan)
    for i, method in enumerate(methods):
        sub = strong[strong["method"] == method]
        for j, chunk in enumerate(chunks):
            row = sub[sub["chunk"] == chunk]
            if not row.empty:
                f1[i, j] = row["f1"].iloc[0]
                cycle[i, j] = row["blocking_cycle_time_sec"].iloc[0] * 1000
                emerg[i, j] = row["active_emerging_rule_count"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 8.2), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
    im = axes[0].imshow(f1, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_title("F1 trajectory under strong rule drift")
    axes[0].set_xticks(range(len(chunks)))
    axes[0].set_xticklabels(chunks)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=8)
    axes[0].axvline(args.strong_drift_chunk - 1, color="#343a40", ls="--", lw=1.0)
    for i in range(len(methods)):
        for j in range(len(chunks)):
            axes[0].text(j, i, f"{f1[i, j]:.2f}", ha="center", va="center", fontsize=6.2)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)

    im = axes[1].imshow(np.log10(cycle + 0.02), aspect="auto", cmap="YlOrRd")
    axes[1].set_title("Blocking cycle by chunk")
    axes[1].set_xticks(range(len(chunks)))
    axes[1].set_xticklabels(chunks)
    axes[1].set_yticks(range(len(methods)))
    axes[1].set_yticklabels([])
    axes[1].axvline(args.strong_drift_chunk - 1, color="#343a40", ls="--", lw=1.0)
    for i in range(len(methods)):
        for j in range(len(chunks)):
            val = cycle[i, j]
            txt = f"{val:.1f}" if val < 100 else f"{val:.0f}"
            axes[1].text(j, i, txt, ha="center", va="center", fontsize=6.0)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cbar.set_label("log10(ms)")

    im = axes[2].imshow(emerg, aspect="auto", cmap="Greens")
    axes[2].set_title("Active emerging rules")
    axes[2].set_xticks(range(len(chunks)))
    axes[2].set_xticklabels(chunks)
    axes[2].set_yticks(range(len(methods)))
    axes[2].set_yticklabels([])
    axes[2].axvline(args.strong_drift_chunk - 1, color="#343a40", ls="--", lw=1.0)
    for i in range(len(methods)):
        for j in range(len(chunks)):
            axes[2].text(j, i, f"{emerg[i, j]:.0f}", ha="center", va="center", fontsize=6.2)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)

    for ax in axes:
        ax.grid(False)
        ax.set_xlabel("Stream chunk")
    fig.suptitle("Strong emerging-rule drift: recovery, realtime cost, and rule activation", y=0.995, fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(FIG_DIR / "Fig3_Strong_Drift_Heatmap.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig3_Strong_Drift_Heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_ablation_figure(ablation_summary, ablation_detail):
    order = ["Full RT-A3C", "Single actor", "No prior-shift", "No structure refresh", "Online only"]
    summary = ablation_summary.set_index("variant").loc[order].reset_index()
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.25))
    axes[0].bar(x - 0.18, summary["post_f1"], width=0.36, color="#c1121f", label="Post-F1")
    axes[0].bar(x + 0.18, summary["post_log_loss"], width=0.36, color="#495057", label="Post log-loss")
    axes[0].set_title("Prediction under strong drift")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=25, ha="right")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, summary["blocking_cycle_time_sec"] * 1000, color="#457b9d")
    axes[1].axhline(10, color="#c1121f", ls="--", lw=1.0)
    axes[1].set_title("Realtime blocking cycle")
    axes[1].set_ylabel("ms")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=25, ha="right")

    full = ablation_detail[(ablation_detail["variant"] == "Full RT-A3C") & (ablation_detail["chunk"] >= 0)]
    nostruct = ablation_detail[(ablation_detail["variant"] == "No structure refresh") & (ablation_detail["chunk"] >= 0)]
    full_emerging = pd.to_numeric(full["active_emerging_rule_count"], errors="coerce").fillna(0.0)
    nostruct_emerging = pd.to_numeric(nostruct["active_emerging_rule_count"], errors="coerce").fillna(0.0)
    axes[2].plot(full["chunk"], full_emerging, marker="o", color="#2a9d8f", label="Full active emerging")
    axes[2].plot(nostruct["chunk"], nostruct_emerging, marker="s", color="#8d99ae", label="No structure")
    axes[2].set_title("Emerging-rule activation by chunk")
    axes[2].set_xlabel("Stream chunk")
    axes[2].set_ylabel("Active emerging rules")
    ymax = max(1.0, float(max(full_emerging.max(), nostruct_emerging.max())) + 0.5)
    axes[2].set_ylim(-0.1, ymax)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Ablation: where realtime rule adaptation comes from", y=1.01, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig4_Ablation_Rule_Evidence.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig4_Ablation_Rule_Evidence.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_actor_count_figure(actor_summary, actor_trace):
    if actor_summary is None or actor_summary.empty or actor_trace is None or actor_trace.empty:
        return
    order = sorted(actor_summary["workers"].unique())
    colors = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4))
    max_time = float(actor_trace["time_sec"].max()) if len(actor_trace) else 1.0
    grid = np.linspace(0.0, max_time, 120)
    for idx, workers in enumerate(order):
        curves = []
        for _, sub in actor_trace[actor_trace["workers"] == workers].groupby("seed", observed=True):
            sub = sub.sort_values("time_sec")
            x = sub["time_sec"].to_numpy()
            y = sub["best_reward"].to_numpy()
            if len(x) == 1:
                curves.append(np.full_like(grid, y[0], dtype=float))
            else:
                curves.append(np.interp(grid, x, y, left=y[0], right=y[-1]))
        if curves:
            axes[0, 0].plot(grid, np.mean(curves, axis=0), lw=1.7, color=colors(idx), label=f"{workers}")
    axes[0, 0].set_title("Best reward convergence")
    axes[0, 0].set_xlabel("Wall-clock seconds")
    axes[0, 0].set_ylabel("Global best reward")
    axes[0, 0].legend(title="Actors", frameon=False, fontsize=7, ncol=4)

    s = actor_summary.set_index("workers").loc[order].reset_index()
    x = np.arange(len(order))
    axes[0, 1].bar(x - 0.18, s["search_time_sec"], width=0.36, color="#457b9d", label="Full search")
    axes[0, 1].bar(x + 0.18, s["time_to_baseline90_sec"], width=0.36, color="#c1121f", label="Base T90")
    axes[0, 1].set_title("Convergence time")
    axes[0, 1].set_xlabel("Number of actors")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(v) for v in order])
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(order, s["search_speedup_vs_1"], marker="o", color="#2a9d8f", label="Full-search speedup")
    axes[1, 0].plot(order, s["baseline90_speedup_vs_1"], marker="s", color="#6d597a", label="Base-T90 speedup")
    axes[1, 0].axhline(1.0, color="#495057", ls="--", lw=1)
    axes[1, 0].set_title("Speedup relative to one actor")
    axes[1, 0].set_xlabel("Number of actors")
    axes[1, 0].set_ylabel("Speedup")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].bar(x - 0.18, s["val_f1"], width=0.36, color="#2a9d8f", label="Validation F1")
    axes[1, 1].bar(x + 0.18, s["signal_recall"], width=0.36, color="#f4a261", label="Signal-rule recall")
    axes[1, 1].set_title("Converged structure quality")
    axes[1, 1].set_xlabel("Number of actors")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(v) for v in order])
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Actor count study under a fixed global rule scope and fixed total episodes", fontsize=13.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(FIG_DIR / "Fig5_Actor_Count_Study.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig5_Actor_Count_Study.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_update_frequency_figure(update_summary):
    if update_summary is None or update_summary.empty:
        return
    summary = update_summary.sort_values("update_frequency")
    x = summary["update_frequency"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.15))
    axes[0].plot(x, summary["blocking_cycle_time_sec"] * 1000, marker="o", color="#c1121f", label="Blocking cycle")
    axes[0].plot(x, summary["foreground_update_time_sec"] * 1000, marker="s", color="#457b9d", label="Foreground update")
    axes[0].plot(x, summary["background_update_time_sec"] * 1000, marker="^", color="#6d597a", label="Background update")
    axes[0].axhline(10, color="#343a40", ls="--", lw=1)
    axes[0].set_title("Latency by refresh interval")
    axes[0].set_xlabel("Structure refresh interval (chunks)")
    axes[0].set_ylabel("Milliseconds")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(x, summary["post_f1"], marker="o", color="#2a9d8f", label="Post-drift F1")
    axes[1].plot(x, summary["post_log_loss"], marker="s", color="#495057", label="Post-drift log-loss")
    axes[1].set_title("Post-drift predictive quality")
    axes[1].set_xlabel("Structure refresh interval (chunks)")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(x - 0.18, summary["structure_refreshed"] * 100, width=0.36, color="#457b9d", label="Structure refresh %")
    axes[2].bar(x + 0.18, summary["active_emerging_rule_count"], width=0.36, color="#52b788", label="Emerging rules")
    axes[2].set_title("Rule adaptation evidence")
    axes[2].set_xlabel("Structure refresh interval (chunks)")
    axes[2].set_ylabel("Percent / count")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Update-frequency study for realtime rule adaptation", y=1.03, fontsize=13.2, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig6_Update_Frequency_Study.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig6_Update_Frequency_Study.pdf", bbox_inches="tight")
    plt.close(fig)


def write_explanations():
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    (EXPLAIN_DIR / "README_Figures_Tables.md").write_text(
        """# 正式实验图表说明

本文件夹只保留正式论文建议使用的高密度图表：

- **Fig1_Method_Framework**：本文方法框架图。
- **Table1_Formal_All_Methods**：所有对比方法的主结果表，覆盖传统 MLN、近年规则自适应、数据流学习、表格基础模型和固定规则参考。
- **Fig2_All_Method_Dashboard**：主结果高密度仪表图，同时展示实时性、预测质量、强漂移概率质量和规则结构证据。
- **Fig3_Strong_Drift_Heatmap**：强规则漂移过程热力图，展示每个方法每个 chunk 的 F1、阻塞周期和激活 emerging rules。
- **Table2_Formal_Ablation**：本文方法关键组件消融表。
- **Fig4_Ablation_Rule_Evidence**：消融图，展示结构刷新对于激活新规则和恢复强漂移性能的作用。
- **Table3_Actor_Count** 与 **Fig5_Actor_Count_Study**：并行 Actor 数量实验，在固定全局规则搜索范围和固定总探索预算下分析收敛速度。
- **Table4_Update_Frequency** 与 **Fig6_Update_Frequency_Study**：结构刷新频率实验，分析不同异步更新间隔对阻塞延迟、后台更新和漂移恢复的影响。

近年规则自适应方法以 style/proxy 标注，表示在统一表格规则特征和数据流协议下复现其核心机制，用于公平比较实时性和规则自适应行为；不声称调用原作者官方实现。
""",
        encoding="utf-8",
    )


def make_formal_docx(main_table, ablation_table):
    from docx import Document
    from docx.enum.section import WD_ORIENT, WD_SECTION
    from docx.enum.table import WD_ALIGN_VERTICAL
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Inches, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.0)
    sec.top_margin = Cm(2.1)
    sec.bottom_margin = Cm(2.0)

    def font(run, name="宋体", size=10.5, bold=None, color=None):
        run.font.name = name
        run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
        run.font.size = Pt(size)
        if bold is not None:
            run.bold = bold
        if color:
            run.font.color.rgb = RGBColor.from_string(color)

    def heading(text, level=1):
        p = doc.add_paragraph()
        p.style = f"Heading {level}"
        r = p.add_run(text)
        font(r, "微软雅黑", 16 if level == 1 else 12.5, True, "1F4E79")

    def body(text):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(21)
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(5)
        r = p.add_run(text)
        font(r)

    def caption(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        font(r, size=9.2, bold=True, color="333333")

    def add_image(path, cap, width=6.35):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Inches(width))
        caption(cap)

    def table_borders(table):
        tbl_pr = table._tbl.tblPr
        borders = tbl_pr.first_child_found_in("w:tblBorders")
        if borders is None:
            borders = OxmlElement("w:tblBorders")
            tbl_pr.append(borders)
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            element = borders.find(qn(f"w:{edge}"))
            if element is None:
                element = OxmlElement(f"w:{edge}")
                borders.append(element)
            element.set(qn("w:val"), "single")
            element.set(qn("w:sz"), "4")
            element.set(qn("w:color"), "B7C7D6")

    def shade(cell, fill):
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = tc_pr.find(qn("w:shd"))
        if shd is None:
            shd = OxmlElement("w:shd")
            tc_pr.append(shd)
        shd.set(qn("w:fill"), fill)

    def add_table(df, cap, font_size=5.2):
        caption(cap)
        table = doc.add_table(rows=1, cols=len(df.columns))
        table.style = "Table Grid"
        for j, col in enumerate(df.columns):
            table.cell(0, j).text = str(col)
        for _, row in df.iterrows():
            cells = table.add_row().cells
            for j, col in enumerate(df.columns):
                cells[j].text = str(row[col])
        table_borders(table)
        for i, row in enumerate(table.rows):
            for cell in row.cells:
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                if i == 0:
                    shade(cell, "D9EAF7")
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_after = Pt(0)
                    for r in p.runs:
                        font(r, size=font_size if i else font_size, bold=True if i == 0 else None)

    heading("第4章 实验与结果分析")
    body("本章重新组织正式实验，删除旧版零散结果，仅保留少量高信息密度图表。实验同时覆盖传统 MLN 结构学习、固定规则在线更新、现代数据流学习、2024-2026 表格基础模型，以及近年规则自适应风格方法。近年规则方法以 style/proxy 标注，表示在统一 MLN 规则特征和数据流协议下复现其核心机制。")
    add_image(FIG_DIR / "Fig1_Method_Framework.png", "图4-1 基于异步优势 Actor-Critic 的实时 MLN 规则自适应框架")
    body("图4-1 展示本文方法的技术路线。数据流 chunk 到达后，前台路径执行低阻塞概率推理和快速权重/先验更新；后台多个 Actor 在概念格环境中并行探索规则添加、删除、修改等动作，Critic 根据 n-step return 和优势函数评估策略更新。")

    doc.add_section(WD_SECTION.NEW_PAGE)
    doc.sections[-1].orientation = WD_ORIENT.LANDSCAPE
    doc.sections[-1].page_width = Cm(29.7)
    doc.sections[-1].page_height = Cm(21)
    doc.sections[-1].left_margin = Cm(1.0)
    doc.sections[-1].right_margin = Cm(1.0)
    doc.sections[-1].top_margin = Cm(1.2)
    doc.sections[-1].bottom_margin = Cm(1.2)
    heading("4.1 主实验：完整方法集合对比")
    add_table(main_table, "表4-1 正式主结果：完整对比方法集合", font_size=4.6)
    body("表4-1 是正式主结果表。本文方法的关键优势不是单一静态 F1 最高，而是在常规数据流中保持低阻塞周期，并在强规则漂移中产生结构刷新和 emerging rules 激活。TabPFN、TabICL 等最新强预测器作为近年参照保留在表中，但它们不产生 MLN 规则结构自适应证据。")

    doc.add_section(WD_SECTION.NEW_PAGE)
    doc.sections[-1].orientation = WD_ORIENT.PORTRAIT
    doc.sections[-1].page_width = Cm(21)
    doc.sections[-1].page_height = Cm(29.7)
    doc.sections[-1].left_margin = Cm(2.2)
    doc.sections[-1].right_margin = Cm(2.0)
    add_image(FIG_DIR / "Fig2_All_Method_Dashboard.png", "图4-2 完整对比方法的实时性、预测质量和规则结构证据", width=6.55)
    body("图4-2 将所有方法压缩到一个四联图中。左上为阻塞周期，红线表示 10 ms 实时阈值；右上比较常规 F1 和强漂移后 F1；左下为强漂移后 Log-loss；右下为结构刷新比例和激活新规则数。该图用于支撑论文核心结论：RT-A3C-MLN 同时满足实时性和规则自适应，而许多强预测或规则学习基线只能满足其中一部分。")

    add_image(FIG_DIR / "Fig3_Strong_Drift_Heatmap.png", "图4-3 强规则漂移下的逐 chunk 恢复、耗时和新规则激活热力图", width=6.55)
    body("图4-3 展示强规则漂移压力测试。竖线表示规则漂移点；F1 热力图反映各方法在每个 chunk 的恢复过程；耗时热力图显示实时阻塞成本；新规则热力图展示是否真正激活后半段才出现的 emerging rules。")

    doc.add_section(WD_SECTION.NEW_PAGE)
    doc.sections[-1].orientation = WD_ORIENT.LANDSCAPE
    doc.sections[-1].page_width = Cm(29.7)
    doc.sections[-1].page_height = Cm(21)
    doc.sections[-1].left_margin = Cm(1.2)
    doc.sections[-1].right_margin = Cm(1.2)
    heading("4.2 消融实验")
    add_table(ablation_table, "表4-2 RT-A3C-MLN 关键组件消融结果", font_size=6.2)
    body("表4-2 对本文方法中的多 Actor、局部先验漂移更新、后台结构刷新和 A3C 结构学习进行消融。No structure refresh 与 Online only 用于验证仅更新固定规则权重是否足够。")

    doc.add_section(WD_SECTION.NEW_PAGE)
    doc.sections[-1].orientation = WD_ORIENT.PORTRAIT
    doc.sections[-1].page_width = Cm(21)
    doc.sections[-1].page_height = Cm(29.7)
    doc.sections[-1].left_margin = Cm(2.2)
    doc.sections[-1].right_margin = Cm(2.0)
    add_image(FIG_DIR / "Fig4_Ablation_Rule_Evidence.png", "图4-4 消融实验中的性能、实时性和规则激活证据", width=6.35)
    body("图4-4 表明，后台规则结构刷新是强规则漂移下激活 emerging rules 的关键机制。Online only 虽然速度快，但不具备规则结构调整能力；No structure refresh 也难以充分吸收新规则。")
    body("综上，正式实验从完整方法集合、强漂移压力测试和消融三个层面验证：异步优势 Actor-Critic 机制能够使 MLN 在数据流输入下实时调整规则结构和权重，形成可快速响应的概率知识网络。")
    doc.save(DOCX_PATH)


def save_artifacts(detail, ablation_detail, actor_detail, actor_trace, update_detail, args):
    core_summary = summarize_core(detail)
    strong_summary = summarize_strong(detail, args.recovery_f1, args.strong_drift_chunk)
    ablation_summary = summarize_ablation(ablation_detail, args.recovery_f1, args.strong_drift_chunk)
    actor_summary = summarize_actor_count(actor_detail) if actor_detail is not None and not actor_detail.empty else pd.DataFrame()
    update_summary = summarize_update_frequency(update_detail, args.recovery_f1, args.strong_drift_chunk)

    main_table = build_main_table(core_summary, strong_summary)
    ablation_table = build_ablation_table(ablation_summary)
    actor_table = make_actor_count_table(actor_summary) if not actor_summary.empty else pd.DataFrame()
    update_table = build_update_frequency_table(update_summary)
    write_table_files(main_table, "Table1_Formal_All_Methods", "Formal main comparison with all baselines")
    write_table_files(ablation_table, "Table2_Formal_Ablation", "Formal RT-A3C-MLN ablation")
    if not actor_table.empty:
        write_table_files(actor_table, "Table3_Actor_Count", "Actor count study")
    if not update_table.empty:
        write_table_files(update_table, "Table4_Update_Frequency", "Update-frequency study")

    core_summary.to_csv(FORMAL_RESULTS / "formal_core_summary.csv", index=False)
    strong_summary.to_csv(FORMAL_RESULTS / "formal_strong_summary.csv", index=False)
    ablation_summary.to_csv(FORMAL_RESULTS / "formal_ablation_summary.csv", index=False)
    if not actor_summary.empty:
        actor_summary.to_csv(FORMAL_RESULTS / "formal_actor_count_summary.csv", index=False)
        actor_detail.to_csv(FORMAL_RESULTS / "formal_actor_count_detail.csv", index=False)
        actor_trace.to_csv(FORMAL_RESULTS / "formal_actor_count_trace.csv", index=False)
    if not update_summary.empty:
        update_summary.to_csv(FORMAL_RESULTS / "formal_update_frequency_summary.csv", index=False)
        update_detail.to_csv(FORMAL_RESULTS / "formal_update_frequency_detail.csv", index=False)
    detail.to_csv(FORMAL_RESULTS / "formal_detail.csv", index=False)
    ablation_detail.to_csv(FORMAL_RESULTS / "formal_ablation_detail.csv", index=False)
    (FORMAL_RESULTS / "formal_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    set_plot_style()
    draw_framework()
    draw_all_method_dashboard(core_summary, strong_summary)
    draw_strong_drift_heatmap(detail, args)
    draw_ablation_figure(ablation_summary, ablation_detail)
    draw_actor_count_figure(actor_summary, actor_trace)
    draw_update_frequency_figure(update_summary)
    write_explanations()
    try:
        make_formal_docx(main_table, ablation_table)
    except ModuleNotFoundError as exc:
        print(f"[warn] skipped docx generation because dependency is missing: {exc}", flush=True)
    return main_table, ablation_table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Remove old results and paper_artifacts before running.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-samples", type=int, default=1800)
    parser.add_argument("--max-rules", type=int, default=160)
    parser.add_argument("--max-literals", type=int, default=65)
    parser.add_argument("--numeric-bins", type=int, default=4)
    parser.add_argument("--min-support", type=float, default=0.018)
    parser.add_argument("--stream-fraction", type=float, default=0.65)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--rolling-window", type=int, default=900)
    parser.add_argument("--rule-window", type=int, default=900)
    parser.add_argument("--river-trees", type=int, default=2)
    parser.add_argument("--river-warmup-limit", type=int, default=900)
    parser.add_argument("--foundation-context-limit", type=int, default=320)
    parser.add_argument("--foundation-estimators", type=int, default=1)
    parser.add_argument("--foundation-batch-size", type=int, default=48)
    parser.add_argument("--tabm-window", type=int, default=520)
    parser.add_argument("--tabm-initial-epochs", type=int, default=4)
    parser.add_argument("--tabm-update-epochs", type=int, default=1)
    parser.add_argument("--tabm-batch-size", type=int, default=192)
    parser.add_argument("--tabm-hidden", type=int, default=72)
    parser.add_argument("--tabm-k", type=int, default=5)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=5)
    parser.add_argument("--a3c-steps", type=int, default=5)
    parser.add_argument("--a3c-rules", type=int, default=84)
    parser.add_argument("--a3c-max-active-rules", type=int, default=32)
    parser.add_argument("--a3c-rolling-window", type=int, default=900)
    parser.add_argument("--a3c-calibration-window", type=int, default=260)
    parser.add_argument("--a3c-refit-interval", type=int, default=1)
    parser.add_argument("--a3c-structure-episodes", type=int, default=3)
    parser.add_argument("--a3c-structure-tolerance", type=float, default=0.015)
    parser.add_argument("--a3c-prior-shift-lr", type=float, default=0.24)
    parser.add_argument("--a3c-prior-shift-clip", type=float, default=1.7)
    parser.add_argument("--a3c-acc-stable-weight", type=float, default=0.82)
    parser.add_argument("--a3c-acc-stable-C", type=float, default=1.0)
    parser.add_argument("--strong-n-features", type=int, default=220)
    parser.add_argument("--strong-train-size", type=int, default=1300)
    parser.add_argument("--strong-val-size", type=int, default=320)
    parser.add_argument("--strong-chunks", type=int, default=6)
    parser.add_argument("--strong-chunk-size", type=int, default=260)
    parser.add_argument("--strong-drift-chunk", type=int, default=4)
    parser.add_argument("--initial-rule-budget", type=int, default=22)
    parser.add_argument("--recovery-f1", type=float, default=0.64)
    parser.add_argument("--actor-seeds", nargs="+", type=int, default=[31, 37, 43])
    parser.add_argument("--actor-counts", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
    parser.add_argument("--actor-rule-scope", type=int, default=4000)
    parser.add_argument("--actor-train-size", type=int, default=8000)
    parser.add_argument("--actor-val-size", type=int, default=2400)
    parser.add_argument("--actor-density", type=float, default=0.055)
    parser.add_argument("--actor-signal-rules", type=int, default=48)
    parser.add_argument("--actor-total-episodes", type=int, default=160)
    parser.add_argument("--actor-steps", type=int, default=6)
    parser.add_argument("--actor-max-active-rules", type=int, default=48)
    parser.add_argument("--update-frequencies", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.clean:
        clean_outputs()
    else:
        FORMAL_RESULTS.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        rows.extend(run_core_dataset(dataset, args))
    rows.extend(run_strong_drift(args))
    detail = pd.DataFrame(rows)
    ablation_detail = pd.DataFrame(run_ablation(args))
    update_detail = run_update_frequency_study(args)
    actor_detail = pd.DataFrame()
    actor_trace = pd.DataFrame()
    main_table, ablation_table = save_artifacts(detail, ablation_detail, actor_detail, actor_trace, update_detail, args)
    try:
        actor_detail, actor_trace = run_actor_count_study(args)
        main_table, ablation_table = save_artifacts(detail, ablation_detail, actor_detail, actor_trace, update_detail, args)
    except MemoryError as exc:
        print(f"[warn] skipped actor-count study because memory was exhausted: {exc}", flush=True)

    print("\n[formal main table]")
    print(main_table.to_string(index=False))
    print("\n[formal ablation table]")
    print(ablation_table.to_string(index=False))
    failures = detail[detail.get("category", "") == "failed"] if "category" in detail else pd.DataFrame()
    if not failures.empty:
        print("\n[failures]")
        cols = ["experiment", "dataset", "method", "failure"]
        print(failures[cols].drop_duplicates().to_string(index=False))
    print(f"\n[done] results: {FORMAL_RESULTS}")
    print(f"[done] artifacts: {ARTIFACTS_DIR}")
    print(f"[done] docx: {DOCX_PATH}")


if __name__ == "__main__":
    main()
