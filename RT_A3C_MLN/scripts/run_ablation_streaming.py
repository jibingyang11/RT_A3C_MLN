from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS, load_dataset
from mln_a3c_experiment.models import RTA3CMLN, _safe_proba
from mln_a3c_experiment.rules import RuleFeatureBuilder
from scripts.run_streaming_experiments import (
    AdaptiveA3CStreamModel,
    _logit,
    _prob_metrics,
    _sigmoid,
    make_prior_drift_chunks,
)


class NoCalibrationA3C(AdaptiveA3CStreamModel):
    def _calibrate_shift(self, x_ref, y_ref):
        self.shift = 0.0


class NoRollingUpdateA3C(AdaptiveA3CStreamModel):
    def update(self, x_chunk, y_chunk):
        return 0.0


class StaticA3COnly:
    def __init__(self, random_state=7, workers=4, episodes=6, steps=5, rules=140):
        self.model = RTA3CMLN(
            workers=workers,
            episodes=episodes,
            n_steps=steps,
            controller_rules=rules,
            random_state=random_state,
            ensemble_penalty=0.001,
        )

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.fit(x_train, y_train, x_val, y_val, rules=None)
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def update(self, x_chunk, y_chunk):
        return 0.0


class PrefixAdaptiveNoA3C:
    def __init__(self, random_state=7, rules=90):
        self.random_state = random_state
        self.rules = rules
        self.active_rules = None
        self.recent_x = None
        self.recent_y = None
        self.adaptive = None
        self.online = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=2e-5,
            l1_ratio=0.03,
            learning_rate="optimal",
            average=True,
            random_state=random_state + 31,
        )
        self.shift = 0.0
        self.rolling_window = 2400

    def fit(self, x_train, y_train, x_val, y_val):
        self.active_rules = list(range(min(self.rules, x_train.shape[1])))
        self.recent_x = x_train.copy()
        self.recent_y = y_train.copy()
        self.adaptive = LogisticRegression(solver="liblinear", C=1.0, max_iter=600, random_state=self.random_state)
        self._fit_adaptive()
        self.online.partial_fit(x_train, y_train, classes=np.array([0, 1]))
        self._calibrate_shift(x_val, y_val)
        return self

    def _fit_adaptive(self):
        weights = np.linspace(0.65, 2.25, len(self.recent_y))
        self.adaptive.fit(self.recent_x[:, self.active_rules], self.recent_y, sample_weight=weights)

    def _raw_prob(self, x):
        adaptive_prob = self.adaptive.predict_proba(x[:, self.active_rules])[:, 1]
        online_prob = _safe_proba(self.online.predict_proba(x))[:, 1]
        return 0.82 * adaptive_prob + 0.18 * online_prob

    def _calibrate_shift(self, x_ref, y_ref):
        logits = _logit(self._raw_prob(x_ref))
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

    def predict_proba(self, x):
        return _safe_proba(_sigmoid(_logit(self._raw_prob(x)) + self.shift))

    def update(self, x_chunk, y_chunk):
        start = time.perf_counter()
        self.online.partial_fit(x_chunk, y_chunk)
        self.recent_x = np.vstack([self.recent_x, x_chunk])[-self.rolling_window :]
        self.recent_y = np.concatenate([self.recent_y, y_chunk])[-self.rolling_window :]
        self._fit_adaptive()
        self._calibrate_shift(self.recent_x, self.recent_y)
        return time.perf_counter() - start


def build_ablation_models(args):
    return {
        "Full_RT_A3C_MLN": AdaptiveA3CStreamModel(
            random_state=args.seed,
            workers=args.a3c_workers,
            episodes=args.a3c_episodes,
            steps=args.a3c_steps,
            rules=args.a3c_rules,
        ),
        "Single_Agent": AdaptiveA3CStreamModel(
            random_state=args.seed,
            workers=1,
            episodes=args.a3c_episodes,
            steps=args.a3c_steps,
            rules=args.a3c_rules,
        ),
        "No_Calibration": NoCalibrationA3C(
            random_state=args.seed,
            workers=args.a3c_workers,
            episodes=args.a3c_episodes,
            steps=args.a3c_steps,
            rules=args.a3c_rules,
        ),
        "No_Rolling_Update": NoRollingUpdateA3C(
            random_state=args.seed,
            workers=args.a3c_workers,
            episodes=args.a3c_episodes,
            steps=args.a3c_steps,
            rules=args.a3c_rules,
        ),
        "Static_A3C_Only": StaticA3COnly(
            random_state=args.seed,
            workers=args.a3c_workers,
            episodes=args.a3c_episodes,
            steps=args.a3c_steps,
            rules=args.a3c_rules,
        ),
        "No_A3C_Structure": PrefixAdaptiveNoA3C(random_state=args.seed, rules=min(90, args.max_rules)),
    }


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

    models = build_ablation_models(args)
    fit_times = {}
    for name_model, model in models.items():
        start = time.perf_counter()
        model.fit(x_train_rules, y_train, x_val_rules, y_val)
        fit_times[name_model] = time.perf_counter() - start

    chunks = make_prior_drift_chunks(
        x_stream_rules,
        y_stream,
        chunks=args.chunks,
        chunk_size=args.chunk_size,
        random_state=args.seed + 101,
    )

    rows = []
    for chunk_id, drift_rate, x_chunk, y_chunk in chunks:
        for name_model, model in models.items():
            start = time.perf_counter()
            prob = model.predict_proba(x_chunk)[:, 1]
            latency = time.perf_counter() - start
            metrics = _prob_metrics(y_chunk, prob, latency)
            update_time = model.update(x_chunk, y_chunk)
            rows.append(
                {
                    "dataset": name,
                    "variant": name_model,
                    "chunk": chunk_id,
                    "drift_positive_rate": drift_rate,
                    "initial_fit_time_sec": fit_times[name_model],
                    "update_time_sec": update_time,
                    **metrics,
                }
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-rules", type=int, default=160)
    parser.add_argument("--max-literals", type=int, default=60)
    parser.add_argument("--numeric-bins", type=int, default=4)
    parser.add_argument("--min-support", type=float, default=0.02)
    parser.add_argument("--stream-fraction", type=float, default=0.65)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=360)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=6)
    parser.add_argument("--a3c-steps", type=int, default=5)
    parser.add_argument("--a3c-rules", type=int, default=140)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / "results" / "additional" / "ablation_streaming"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        print(f"[ablation] dataset={dataset}")
        rows.extend(run_one_dataset(dataset, args))
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "ablation_streaming_detail.csv", index=False)
    metrics = ["accuracy", "f1", "roc_auc", "log_loss", "latency_ms_per_sample", "update_time_sec", "initial_fit_time_sec"]
    summary = detail.groupby("variant", observed=True)[metrics].mean().sort_values("f1", ascending=False)
    summary.to_csv(out_dir / "ablation_streaming_summary.csv")
    print(summary.round(6).to_string())
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
