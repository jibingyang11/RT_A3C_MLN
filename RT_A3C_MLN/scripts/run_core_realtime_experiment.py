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
from mln_a3c_experiment.models import BeamSearchMLN, _safe_proba
from mln_a3c_experiment.rules import RuleFeatureBuilder
from scripts.run_streaming_experiments import AdaptiveA3CStreamModel, make_prior_drift_chunks


def metrics_from_prob(y_true, prob, inference_sec):
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
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


class StaticMaxEntReference:
    category = "static_reference"

    def __init__(self, random_state):
        self.model = LogisticRegression(solver="liblinear", C=1.0, max_iter=700, random_state=random_state)

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.fit(x_train, y_train)
        return self

    def predict_proba(self, x):
        return _safe_proba(self.model.predict_proba(x))

    def update(self, x_chunk, y_chunk):
        return 0.0


class OnlineWeightMLN:
    category = "online_weight_fixed_structure"

    def __init__(self, random_state):
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=7e-5,
            l1_ratio=0.05,
            learning_rate="optimal",
            average=True,
            random_state=random_state,
        )

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.partial_fit(x_train, y_train, classes=np.array([0, 1]))
        return self

    def predict_proba(self, x):
        return _safe_proba(self.model.predict_proba(x))

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.model.partial_fit(x_chunk, y_chunk)
        return time.perf_counter() - start


class RollingRetrainModel:
    category = "adaptive_batch_mln"

    def __init__(self, estimator_factory, window=2400):
        self.estimator_factory = estimator_factory
        self.window = window
        self.model = None
        self.x_recent = None
        self.y_recent = None

    def fit(self, x_train, y_train, x_val, y_val):
        self.x_recent = x_train[-self.window :].copy()
        self.y_recent = y_train[-self.window :].copy()
        self.model = self.estimator_factory()
        self.model.fit(self.x_recent, self.y_recent)
        return self

    def predict_proba(self, x):
        return _safe_proba(self.model.predict_proba(x))

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.x_recent = np.vstack([self.x_recent, x_chunk])[-self.window :]
        self.y_recent = np.concatenate([self.y_recent, y_chunk])[-self.window :]
        self.model = self.estimator_factory()
        self.model.fit(self.x_recent, self.y_recent)
        return time.perf_counter() - start


class RollingBeamMLN:
    category = "adaptive_batch_mln"

    def __init__(self, random_state, window=1800):
        self.random_state = random_state
        self.window = window
        self.model = None
        self.x_recent = None
        self.y_recent = None
        self.x_val = None
        self.y_val = None

    def fit(self, x_train, y_train, x_val, y_val):
        self.x_recent = x_train[-self.window :].copy()
        self.y_recent = y_train[-self.window :].copy()
        self.x_val = x_val
        self.y_val = y_val
        self._fit_model()
        return self

    def _fit_model(self):
        self.model = BeamSearchMLN(max_selected=14, search_pool=55, random_state=self.random_state)
        self.model.fit(self.x_recent, self.y_recent, self.x_val, self.y_val, rules=None)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.x_recent = np.vstack([self.x_recent, x_chunk])[-self.window :]
        self.y_recent = np.concatenate([self.y_recent, y_chunk])[-self.window :]
        self._fit_model()
        return time.perf_counter() - start


def build_models(args):
    return {
        "RT_A3C_Realtime_MLN": AdaptiveA3CStreamModel(
            random_state=args.seed,
            workers=args.a3c_workers,
            episodes=args.a3c_episodes,
            steps=args.a3c_steps,
            rules=args.a3c_rules,
            ultra_fast_update=True,
            max_active_rules=args.a3c_max_active_rules,
            rolling_window=args.a3c_rolling_window,
            calibration_window=args.a3c_calibration_window,
            refit_interval=args.a3c_refit_interval,
            async_refit=True,
            prior_shift_lr=args.a3c_prior_shift_lr,
            prior_shift_clip=args.a3c_prior_shift_clip,
            structure_refresh=True,
            structure_refresh_interval=args.a3c_refit_interval,
            structure_refresh_episodes=args.a3c_structure_episodes,
            structure_refresh_tolerance=args.a3c_structure_tolerance,
        ),
        "OnlineWeight_MLN": OnlineWeightMLN(args.seed),
        "Rolling_MaxEnt_MLN": RollingRetrainModel(
            lambda: LogisticRegression(solver="liblinear", C=1.0, max_iter=700, random_state=args.seed)
        ),
        "Rolling_L1_MLN": RollingRetrainModel(
            lambda: LogisticRegression(
                solver="liblinear",
                penalty="l1",
                C=0.28,
                max_iter=700,
                random_state=args.seed,
            )
        ),
        "Rolling_Boosted_MLN": RollingRetrainModel(
            lambda: GradientBoostingClassifier(
                n_estimators=80,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.85,
                random_state=args.seed,
            ),
            window=1800,
        ),
        "Rolling_BeamSearch_MLN": RollingBeamMLN(args.seed),
        "Static_MaxEnt_Reference": StaticMaxEntReference(args.seed),
    }


def model_category(model):
    return getattr(model, "category", "rt_a3c_realtime")


def run_dataset(dataset, args):
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
    for method, model in build_models(args).items():
        print(f"[core] dataset={dataset} method={method}")
        start = time.perf_counter()
        model.fit(x_train_rules, y_train, x_val_rules, y_val)
        fit_time = time.perf_counter() - start
        for chunk_id, drift_rate, x_chunk, y_chunk in chunks:
            start = time.perf_counter()
            prob = model.predict_proba(x_chunk)[:, 1]
            inference_time = time.perf_counter() - start
            row = metrics_from_prob(y_chunk, prob, inference_time)
            update_time = model.update(x_chunk, y_chunk)
            background_time = getattr(model, "last_background_update_sec", 0.0)
            cycle_time = inference_time + update_time
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "category": model_category(model),
                    "chunk": chunk_id,
                    "drift_positive_rate": drift_rate,
                    "initial_fit_time_sec": fit_time,
                    "foreground_update_time_sec": update_time,
                    "background_update_time_sec": background_time,
                    "structure_refreshed": getattr(model, "last_structure_refreshed", 0.0),
                    "rule_additions": getattr(model, "last_rule_additions", 0),
                    "rule_deletions": getattr(model, "last_rule_deletions", 0),
                    "active_rule_count": getattr(model, "last_active_rule_count", 0),
                    "blocking_cycle_time_sec": cycle_time,
                    "meets_10ms_cycle": float(cycle_time <= 0.010),
                    "meets_5ms_update": float(update_time <= 0.005),
                    **row,
                }
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--max-rules", type=int, default=220)
    parser.add_argument("--max-literals", type=int, default=80)
    parser.add_argument("--numeric-bins", type=int, default=5)
    parser.add_argument("--min-support", type=float, default=0.015)
    parser.add_argument("--stream-fraction", type=float, default=0.70)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=5)
    parser.add_argument("--a3c-steps", type=int, default=5)
    parser.add_argument("--a3c-rules", type=int, default=180)
    parser.add_argument("--a3c-max-active-rules", type=int, default=64)
    parser.add_argument("--a3c-rolling-window", type=int, default=600)
    parser.add_argument("--a3c-calibration-window", type=int, default=300)
    parser.add_argument("--a3c-refit-interval", type=int, default=4)
    parser.add_argument("--a3c-structure-episodes", type=int, default=2)
    parser.add_argument("--a3c-structure-tolerance", type=float, default=0.0)
    parser.add_argument("--a3c-prior-shift-lr", type=float, default=0.20)
    parser.add_argument("--a3c-prior-shift-clip", type=float, default=1.5)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / "results" / "core_claim"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        rows.extend(run_dataset(dataset, args))
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "core_realtime_detail.csv", index=False)
    metrics = [
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "latency_ms_per_sample",
        "foreground_update_time_sec",
        "blocking_cycle_time_sec",
        "background_update_time_sec",
        "structure_refreshed",
        "rule_additions",
        "rule_deletions",
        "active_rule_count",
        "initial_fit_time_sec",
        "meets_10ms_cycle",
        "meets_5ms_update",
    ]
    summary = detail.groupby(["method", "category"], observed=True)[metrics].mean().reset_index()
    summary = summary.sort_values(["category", "foreground_update_time_sec"])
    summary.to_csv(out_dir / "core_realtime_summary.csv", index=False)
    config = vars(args)
    (out_dir / "core_realtime_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(summary.round(6).to_string(index=False))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
