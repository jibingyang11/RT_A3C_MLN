from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS, load_dataset
from mln_a3c_experiment.models import BeamSearchMLN, RTA3CMLN, _safe_proba
from mln_a3c_experiment.rules import RuleFeatureBuilder


def _logit(prob):
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return np.log(prob / (1.0 - prob))


def _sigmoid(score):
    score = np.clip(score, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-score))


def _prob_metrics(y_true, prob, latency_sec):
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
    pred = (prob >= 0.5).astype(int)
    row = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "latency_ms_per_sample": float(latency_sec * 1000.0 / max(1, len(y_true))),
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        row["roc_auc"] = np.nan
    return row


class StaticProbModel:
    def __init__(self, estimator, selected=None):
        self.estimator = estimator
        self.selected = selected

    def fit(self, x_train, y_train, x_val, y_val):
        x_fit = x_train if self.selected is None else x_train[:, self.selected]
        self.estimator.fit(x_fit, y_train)
        return self

    def predict_proba(self, x):
        x_pred = x if self.selected is None else x[:, self.selected]
        return _safe_proba(self.estimator.predict_proba(x_pred))

    def update(self, x_chunk, y_chunk):
        return 0.0


class BeamStaticModel:
    def __init__(self, random_state=7):
        self.model = BeamSearchMLN(max_selected=24, search_pool=90, random_state=random_state)

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.fit(x_train, y_train, x_val, y_val, rules=None)
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def update(self, x_chunk, y_chunk):
        return 0.0


class OnlineLinearModel:
    def __init__(self, penalty="elasticnet", alpha=5e-5, l1_ratio=0.05, random_state=7):
        self.estimator = SGDClassifier(
            loss="log_loss",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            learning_rate="optimal",
            average=True,
            random_state=random_state,
        )
        self.classes = np.array([0, 1])
        self._is_fit = False

    def fit(self, x_train, y_train, x_val, y_val):
        self.estimator.partial_fit(x_train, y_train, classes=self.classes)
        self._is_fit = True
        return self

    def predict_proba(self, x):
        return _safe_proba(self.estimator.predict_proba(x))

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.estimator.partial_fit(x_chunk, y_chunk)
        return time.perf_counter() - start


class AdaptiveA3CStreamModel:
    """Streaming wrapper for RT-A3C-MLN with delayed-label asynchronous updates."""

    def __init__(
        self,
        random_state=7,
        workers=4,
        episodes=10,
        steps=6,
        rules=160,
        fast_mode=False,
        fast_search_mode=None,
        fast_update=None,
        ultra_fast_update=False,
        max_active_rules=None,
        rolling_window=2400,
        calibration_window=None,
        refit_interval=1,
        async_refit=False,
        prior_shift_lr=0.06,
        prior_shift_clip=0.8,
        structure_refresh=True,
        structure_refresh_interval=None,
        structure_refresh_episodes=None,
        structure_refresh_tolerance=0.0,
    ):
        self.random_state = random_state
        self.workers = workers
        self.episodes = episodes
        self.steps = steps
        self.rules = rules
        self.fast_mode = fast_mode
        self.fast_update = fast_mode if fast_update is None else fast_update
        self.ultra_fast_update = ultra_fast_update
        if self.ultra_fast_update:
            self.fast_update = True
        self.fast_search_mode = fast_mode if fast_search_mode is None else fast_search_mode
        self.max_active_rules = max_active_rules
        self.calibration_window = calibration_window
        self.refit_interval = max(1, refit_interval)
        self.async_refit = async_refit
        self.prior_shift_lr = prior_shift_lr
        self.prior_shift_clip = prior_shift_clip
        self.structure_refresh = structure_refresh
        self.structure_refresh_interval = max(1, structure_refresh_interval or self.refit_interval)
        self.structure_refresh_episodes = (
            max(1, structure_refresh_episodes)
            if structure_refresh_episodes is not None
            else max(1, episodes // 2)
        )
        self.structure_refresh_tolerance = structure_refresh_tolerance
        self.update_count = 0
        self.last_background_update_sec = 0.0
        self.last_structure_refreshed = 0.0
        self.last_rule_additions = 0
        self.last_rule_deletions = 0
        self.last_active_rule_count = 0
        self.a3c = RTA3CMLN(
            workers=workers,
            episodes=episodes,
            n_steps=steps,
            controller_rules=rules,
            random_state=random_state,
            ensemble_penalty=0.001,
            fast_mode=self.fast_search_mode,
        )
        self.online = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=2e-5,
            l1_ratio=0.03,
            learning_rate="optimal",
            average=True,
            random_state=random_state + 17,
        )
        self.classes = np.array([0, 1])
        self.online_weight = 0.15
        self.static_weight = 0.15
        self.rolling_window = rolling_window
        self.active_rules = None
        self.recent_x = None
        self.recent_y = None
        self.adaptive = None
        self.shift = 0.0

    def _make_adaptive_estimator(self):
        if self.fast_update:
            return SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=3e-5,
                l1_ratio=0.02,
                learning_rate="optimal",
                average=True,
                random_state=self.random_state + 29 + self.update_count,
            )
        return LogisticRegression(
            solver="liblinear",
            C=1.0,
            max_iter=600,
            random_state=self.random_state + 29 + self.update_count,
        )

    def fit(self, x_train, y_train, x_val, y_val):
        self.a3c.fit(x_train, y_train, x_val, y_val, rules=None)
        self.online.partial_fit(x_train, y_train, classes=self.classes)
        selected = self.a3c.selected_rule_indices()
        if not selected:
            selected = list(range(min(x_train.shape[1], 32)))
        self.active_rules = self._select_active_rules(selected, x_train.shape[1])
        self.last_active_rule_count = len(self.active_rules)
        self.recent_x = x_train.copy()
        self.recent_y = y_train.copy()
        self.adaptive = self._make_adaptive_estimator()
        self._fit_adaptive()
        self._calibrate_shift(x_val, y_val)
        return self

    def _select_active_rules(self, selected, n_features):
        prefix_size = 24 if self.fast_update else 32
        prefix = list(range(min(n_features, prefix_size)))
        candidates = sorted(set(selected).union(prefix))
        if self.max_active_rules is None or len(candidates) <= self.max_active_rules:
            return candidates
        scored = []
        for idx in candidates:
            weight = self.a3c.rule_weight(idx)
            if weight is None:
                weight = 1.0 / (idx + 1.0)
            scored.append((abs(float(weight)), -idx, idx))
        prefix_keep = prefix[: min(len(prefix), max(8, self.max_active_rules // 4))]
        ordered = list(prefix_keep)
        for _, _, idx in sorted(scored, reverse=True):
            if idx not in ordered:
                ordered.append(idx)
            if len(ordered) >= self.max_active_rules:
                break
        return sorted(ordered[: self.max_active_rules])

    def _raw_prob(self, x, include_static=True):
        online_prob = _safe_proba(self.online.predict_proba(x))[:, 1]
        adaptive_prob = self.adaptive.predict_proba(x[:, self.active_rules])[:, 1]
        if include_static:
            static_prob = self.a3c.predict_proba(x)[:, 1]
            adaptive_weight = max(0.0, 1.0 - self.online_weight - self.static_weight)
            return (
                adaptive_weight * adaptive_prob
                + self.online_weight * online_prob
                + self.static_weight * static_prob
            )
        adaptive_weight = max(0.0, 1.0 - self.online_weight)
        return adaptive_weight * adaptive_prob + self.online_weight * online_prob

    def _calibrate_shift(self, x_ref, y_ref, include_static=True):
        if self.calibration_window is not None and len(y_ref) > self.calibration_window:
            x_ref = x_ref[-self.calibration_window :]
            y_ref = y_ref[-self.calibration_window :]
        raw = self._raw_prob(x_ref, include_static=include_static)
        logits = _logit(raw)
        best_objective = np.inf
        best_shift = 0.0
        for shift in np.linspace(-1.25, 1.25, 51):
            prob = _sigmoid(logits + shift)
            pred = (prob >= 0.5).astype(int)
            objective = 0.08 * log_loss(y_ref, prob, labels=[0, 1]) - f1_score(y_ref, pred, zero_division=0)
            if objective < best_objective:
                best_objective = objective
                best_shift = float(shift)
        self.shift = best_shift

    def _fit_adaptive(self):
        weights = np.linspace(0.65, 2.25, len(self.recent_y))
        if self.fast_update:
            for _ in range(3):
                self.adaptive.partial_fit(
                    self.recent_x[:, self.active_rules],
                    self.recent_y,
                    classes=self.classes,
                    sample_weight=weights,
                )
        else:
            self.adaptive.fit(self.recent_x[:, self.active_rules], self.recent_y, sample_weight=weights)

    def _should_refresh_structure(self):
        return (
            self.structure_refresh
            and self.async_refit
            and self.update_count > 0
            and self.update_count % self.structure_refresh_interval == 0
        )

    def _split_recent_for_structure_refresh(self):
        if self.recent_x is None or self.recent_y is None or len(self.recent_y) < 80:
            return None
        if len(np.unique(self.recent_y)) < 2:
            return None
        test_size = min(0.30, max(0.18, 120 / max(len(self.recent_y), 1)))
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

    def _merge_structure_refresh_rules(self, refreshed: RTA3CMLN, selected: list[int]) -> list[int]:
        if not self.active_rules:
            return self._select_active_rules(selected, self.recent_x.shape[1])
        old_order = list(self.active_rules)
        target = len(old_order)
        protected_count = min(len(old_order), max(8, target // 4))
        protected = set(old_order[:protected_count])
        add_budget = max(1, min(6, target // 12))
        ranked_new = sorted(
            set(selected),
            key=lambda idx: abs(float(refreshed.rule_weight(idx) or 0.0)),
            reverse=True,
        )
        additions = [idx for idx in ranked_new if idx not in old_order][:add_budget]
        if not additions:
            return sorted(old_order)
        removable = [idx for idx in old_order if idx not in protected]
        removable.sort(key=lambda idx: abs(float(self.a3c.rule_weight(idx) or 0.0)))
        removals = set(removable[: min(len(additions), len(removable))])
        merged = [idx for idx in old_order if idx not in removals]
        for idx in additions:
            if idx not in merged:
                merged.append(idx)
        return sorted(merged[:target])

    def _refresh_structure_background(self):
        self.last_structure_refreshed = 0.0
        self.last_rule_additions = 0
        self.last_rule_deletions = 0
        split = self._split_recent_for_structure_refresh()
        if split is None:
            self.last_active_rule_count = len(self.active_rules or [])
            return
        x_train, x_val, y_train, y_val = split
        old_rules = set(self.active_rules or [])
        refreshed = RTA3CMLN(
            workers=self.workers,
            episodes=self.structure_refresh_episodes,
            n_steps=self.steps,
            controller_rules=self.rules,
            random_state=self.random_state + 104729 + self.update_count,
            ensemble_penalty=0.001,
            fast_mode=self.fast_search_mode,
        )
        refreshed.fit(x_train, y_train, x_val, y_val, rules=None)
        selected = refreshed.selected_rule_indices()
        if not selected:
            self.last_active_rule_count = len(self.active_rules or [])
            return
        new_active = self._merge_structure_refresh_rules(refreshed, selected)
        new_rules = set(new_active)
        old_a3c = self.a3c
        old_active_rules = self.active_rules
        old_adaptive = self.adaptive
        old_shift = self.shift
        old_prob = self.predict_proba(x_val)[:, 1]
        old_pred = (old_prob >= 0.5).astype(int)
        old_objective = float(log_loss(y_val, old_prob, labels=[0, 1]) - 0.06 * f1_score(y_val, old_pred, zero_division=0))

        self.a3c = refreshed
        self.active_rules = new_active
        self.adaptive = self._make_adaptive_estimator()
        self._fit_adaptive()
        self._calibrate_shift(self.recent_x, self.recent_y, include_static=True)
        new_prob = self.predict_proba(x_val)[:, 1]
        new_pred = (new_prob >= 0.5).astype(int)
        new_objective = float(log_loss(y_val, new_prob, labels=[0, 1]) - 0.06 * f1_score(y_val, new_pred, zero_division=0))

        if new_objective <= old_objective + self.structure_refresh_tolerance:
            self.last_rule_additions = len(new_rules - old_rules)
            self.last_rule_deletions = len(old_rules - new_rules)
            self.last_active_rule_count = len(new_active)
            self.last_structure_refreshed = 1.0
            return

        self.a3c = old_a3c
        self.active_rules = old_active_rules
        self.adaptive = old_adaptive
        self.shift = old_shift
        self.last_active_rule_count = len(self.active_rules or [])

    def predict_proba(self, x):
        prob = _sigmoid(_logit(self._raw_prob(x)) + self.shift)
        return _safe_proba(prob)

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.last_background_update_sec = 0.0
        self.last_structure_refreshed = 0.0
        self.last_rule_additions = 0
        self.last_rule_deletions = 0
        self.last_active_rule_count = len(self.active_rules or [])
        if self.ultra_fast_update:
            raw = self._raw_prob(x_chunk, include_static=False)
            prior = float(np.clip(np.mean(y_chunk), 1e-4, 1.0 - 1e-4))
            predicted_prior = float(np.clip(np.mean(raw), 1e-4, 1.0 - 1e-4))
            prior_shift = float(_logit(np.array([prior]))[0] - _logit(np.array([predicted_prior]))[0])
            self.shift = float(
                np.clip(
                    (1.0 - self.prior_shift_lr) * self.shift + self.prior_shift_lr * prior_shift,
                    -self.prior_shift_clip,
                    self.prior_shift_clip,
                )
            )
            self.online.partial_fit(x_chunk, y_chunk)
            self.recent_x = np.vstack([self.recent_x, x_chunk])[-self.rolling_window :]
            self.recent_y = np.concatenate([self.recent_y, y_chunk])[-self.rolling_window :]
            self.adaptive.partial_fit(x_chunk[:, self.active_rules], y_chunk)
            self.update_count += 1
            if self._should_refresh_structure():
                blocking_time = time.perf_counter() - start
                background_start = time.perf_counter()
                self._refresh_structure_background()
                self.last_background_update_sec = time.perf_counter() - background_start
                return blocking_time
            return time.perf_counter() - start
        will_refit = (not self.fast_update) and ((self.update_count + 1) % self.refit_interval == 0)
        online_prob = _safe_proba(self.online.predict_proba(x_chunk))[:, 1]
        adaptive_prob = self.adaptive.predict_proba(x_chunk[:, self.active_rules])[:, 1]
        online_loss = log_loss(y_chunk, online_prob, labels=[0, 1])
        adaptive_loss = log_loss(y_chunk, adaptive_prob, labels=[0, 1])
        if self.fast_update:
            losses = np.array([adaptive_loss, online_loss], dtype=float)
            weights = np.exp(-(losses - losses.min()) * 4.0)
            weights = weights / weights.sum()
            self.online_weight = float(min(0.30, max(0.05, weights[1])))
            self.static_weight = min(self.static_weight, 0.18)
        elif will_refit:
            static_prob = self.a3c.predict_proba(x_chunk)[:, 1]
            static_loss = log_loss(y_chunk, static_prob, labels=[0, 1])
            losses = np.array([adaptive_loss, online_loss, static_loss], dtype=float)
            weights = np.exp(-(losses - losses.min()) * 4.0)
            weights = weights / weights.sum()
            self.online_weight = float(min(0.35, max(0.05, weights[1])))
            self.static_weight = float(min(0.35, max(0.05, weights[2])))
        else:
            losses = np.array([adaptive_loss, online_loss], dtype=float)
            weights = np.exp(-(losses - losses.min()) * 4.0)
            weights = weights / weights.sum()
            self.online_weight = float(min(0.30, max(0.05, weights[1])))
        self.online.partial_fit(x_chunk, y_chunk)
        self.recent_x = np.vstack([self.recent_x, x_chunk])[-self.rolling_window :]
        self.recent_y = np.concatenate([self.recent_y, y_chunk])[-self.rolling_window :]
        self.update_count += 1
        if self.fast_update:
            self.adaptive.partial_fit(x_chunk[:, self.active_rules], y_chunk)
        elif will_refit:
            if self.async_refit:
                blocking_time = time.perf_counter() - start
                background_start = time.perf_counter()
                if self._should_refresh_structure():
                    self._refresh_structure_background()
                else:
                    self._fit_adaptive()
                    self._calibrate_shift(self.recent_x, self.recent_y, include_static=True)
                self.last_background_update_sec = time.perf_counter() - background_start
                return blocking_time
            self._fit_adaptive()
        else:
            pass
        include_static = will_refit
        self._calibrate_shift(self.recent_x, self.recent_y, include_static=include_static)
        return time.perf_counter() - start


def build_stream_models(
    random_state,
    a3c_workers,
    a3c_episodes,
    a3c_steps,
    a3c_rules,
    a3c_fast_mode=False,
    a3c_fast_search_mode=None,
    a3c_fast_update=None,
    a3c_ultra_fast_update=False,
    a3c_max_active_rules=None,
    a3c_rolling_window=2400,
    a3c_calibration_window=None,
    a3c_refit_interval=1,
    a3c_async_refit=False,
    a3c_prior_shift_lr=0.06,
    a3c_prior_shift_clip=0.8,
    a3c_structure_refresh=True,
    a3c_structure_refresh_interval=None,
    a3c_structure_refresh_episodes=None,
    a3c_structure_refresh_tolerance=0.0,
):
    return {
        "RT_A3C_MLN": AdaptiveA3CStreamModel(
            random_state=random_state,
            workers=a3c_workers,
            episodes=a3c_episodes,
            steps=a3c_steps,
            rules=a3c_rules,
            fast_mode=a3c_fast_mode,
            fast_search_mode=a3c_fast_search_mode,
            fast_update=a3c_fast_update,
            ultra_fast_update=a3c_ultra_fast_update,
            max_active_rules=a3c_max_active_rules,
            rolling_window=a3c_rolling_window,
            calibration_window=a3c_calibration_window,
            refit_interval=a3c_refit_interval,
            async_refit=a3c_async_refit,
            prior_shift_lr=a3c_prior_shift_lr,
            prior_shift_clip=a3c_prior_shift_clip,
            structure_refresh=a3c_structure_refresh,
            structure_refresh_interval=a3c_structure_refresh_interval,
            structure_refresh_episodes=a3c_structure_refresh_episodes,
            structure_refresh_tolerance=a3c_structure_refresh_tolerance,
        ),
        "MaxEnt_MLN": StaticProbModel(
            LogisticRegression(solver="liblinear", C=1.0, max_iter=700, random_state=random_state)
        ),
        "PLL_MLN": OnlineLinearModel(penalty="l2", alpha=5e-5, l1_ratio=0.0, random_state=random_state),
        "L1_Sparse_MLN": StaticProbModel(
            LogisticRegression(
                solver="liblinear",
                penalty="l1",
                C=0.28,
                max_iter=700,
                random_state=random_state,
            )
        ),
        "BeamSearch_MLN": BeamStaticModel(random_state=random_state),
        "Boosted_MLN": StaticProbModel(
            GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.85,
                random_state=random_state,
            )
        ),
        "Online_SGD_MLN": OnlineLinearModel(
            penalty="elasticnet", alpha=1e-4, l1_ratio=0.05, random_state=random_state
        ),
    }


def make_prior_drift_chunks(x, y, chunks, chunk_size, random_state):
    rng = np.random.default_rng(random_state)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    base = float(np.mean(y))
    amplitude = min(0.35, max(0.12, 0.85 - base, base - 0.05))
    phase = np.linspace(0, 2.5 * np.pi, chunks)
    rates = np.clip(base + amplitude * np.sin(phase), 0.03, 0.97)

    result = []
    for chunk_id, rate in enumerate(rates, start=1):
        n_pos = int(round(chunk_size * rate))
        n_neg = chunk_size - n_pos
        pos_idx = rng.choice(positive, size=n_pos, replace=n_pos > len(positive))
        neg_idx = rng.choice(negative, size=n_neg, replace=n_neg > len(negative))
        indices = np.concatenate([pos_idx, neg_idx])
        rng.shuffle(indices)
        result.append((chunk_id, float(rate), x[indices], y[indices]))
    return result


def run_one_dataset(name, args):
    frame, target = load_dataset(name, ROOT / "data" / "raw", max_samples=args.max_samples)
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

    models = build_stream_models(
        random_state=args.seed,
        a3c_workers=args.a3c_workers,
        a3c_episodes=args.a3c_episodes,
        a3c_steps=args.a3c_steps,
        a3c_rules=min(args.a3c_rules, args.max_rules),
        a3c_fast_mode=getattr(args, "a3c_fast_mode", False),
        a3c_fast_search_mode=getattr(args, "a3c_fast_search_mode", None),
        a3c_fast_update=getattr(args, "a3c_fast_update", None),
        a3c_ultra_fast_update=getattr(args, "a3c_ultra_fast_update", False),
        a3c_max_active_rules=getattr(args, "a3c_max_active_rules", None),
        a3c_rolling_window=getattr(args, "a3c_rolling_window", 2400),
        a3c_calibration_window=getattr(args, "a3c_calibration_window", None),
        a3c_refit_interval=getattr(args, "a3c_refit_interval", 1),
        a3c_async_refit=getattr(args, "a3c_async_refit", False),
        a3c_prior_shift_lr=getattr(args, "a3c_prior_shift_lr", 0.06),
        a3c_prior_shift_clip=getattr(args, "a3c_prior_shift_clip", 0.8),
        a3c_structure_refresh=getattr(args, "a3c_structure_refresh", True),
        a3c_structure_refresh_interval=getattr(args, "a3c_structure_refresh_interval", None),
        a3c_structure_refresh_episodes=getattr(args, "a3c_structure_refresh_episodes", None),
        a3c_structure_refresh_tolerance=getattr(args, "a3c_structure_refresh_tolerance", 0.01),
    )

    fit_times = {}
    for method, model in models.items():
        start = time.perf_counter()
        model.fit(x_train_rules, y_train, x_val_rules, y_val)
        fit_times[method] = time.perf_counter() - start

    chunks = make_prior_drift_chunks(
        x_stream_rules,
        y_stream,
        chunks=args.chunks,
        chunk_size=args.chunk_size,
        random_state=args.seed + 101,
    )

    rows = []
    for chunk_id, drift_rate, x_chunk, y_chunk in chunks:
        for method, model in models.items():
            start = time.perf_counter()
            prob = model.predict_proba(x_chunk)[:, 1]
            latency = time.perf_counter() - start
            metrics = _prob_metrics(y_chunk, prob, latency)
            update_time = model.update(x_chunk, y_chunk)
            background_update_time = getattr(model, "last_background_update_sec", 0.0)
            rows.append(
                {
                    "dataset": name,
                    "method": method,
                    "chunk": chunk_id,
                    "drift_positive_rate": drift_rate,
                    "observed_positive_rate": float(np.mean(y_chunk)),
                    "initial_fit_time_sec": fit_times[method],
                    "update_time_sec": update_time,
                    "background_update_time_sec": background_update_time,
                    **metrics,
                }
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--max-rules", type=int, default=180)
    parser.add_argument("--max-literals", type=int, default=70)
    parser.add_argument("--numeric-bins", type=int, default=4)
    parser.add_argument("--min-support", type=float, default=0.02)
    parser.add_argument("--stream-fraction", type=float, default=0.65)
    parser.add_argument("--chunks", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=420)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=10)
    parser.add_argument("--a3c-steps", type=int, default=6)
    parser.add_argument("--a3c-rules", type=int, default=160)
    parser.add_argument("--a3c-fast-mode", action="store_true")
    parser.add_argument("--a3c-fast-search-mode", action="store_true", default=None)
    parser.add_argument("--a3c-fast-update", action="store_true", default=None)
    parser.add_argument("--a3c-ultra-fast-update", action="store_true")
    parser.add_argument("--a3c-max-active-rules", type=int, default=None)
    parser.add_argument("--a3c-rolling-window", type=int, default=2400)
    parser.add_argument("--a3c-calibration-window", type=int, default=None)
    parser.add_argument("--a3c-refit-interval", type=int, default=1)
    parser.add_argument("--a3c-async-refit", action="store_true")
    parser.add_argument("--a3c-prior-shift-lr", type=float, default=0.06)
    parser.add_argument("--a3c-prior-shift-clip", type=float, default=0.8)
    parser.add_argument("--a3c-no-structure-refresh", dest="a3c_structure_refresh", action="store_false")
    parser.set_defaults(a3c_structure_refresh=True)
    parser.add_argument("--a3c-structure-refresh-interval", type=int, default=None)
    parser.add_argument("--a3c-structure-refresh-episodes", type=int, default=None)
    parser.add_argument("--a3c-structure-refresh-tolerance", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / "results" / "streaming"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        print(f"[stream] dataset={dataset}")
        dataset_rows = run_one_dataset(dataset, args)
        rows.extend(dataset_rows)
        pd.DataFrame(dataset_rows).to_csv(out_dir / f"{dataset}_streaming.csv", index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "streaming_summary.csv", index=False)
    aggregate = (
        frame.groupby("method", observed=True)[
            [
                "accuracy",
                "f1",
                "roc_auc",
                "log_loss",
                "latency_ms_per_sample",
                "update_time_sec",
                "background_update_time_sec",
            ]
        ]
        .mean()
        .sort_values("f1", ascending=False)
    )
    aggregate.to_csv(out_dir / "streaming_aggregate.csv")
    (out_dir / "streaming_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(aggregate.round(6).to_string())
    print(f"[done] wrote {out_dir / 'streaming_summary.csv'}")


if __name__ == "__main__":
    main()
