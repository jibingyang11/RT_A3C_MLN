from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS, load_dataset
from mln_a3c_experiment.models import _safe_proba
from mln_a3c_experiment.rules import RuleFeatureBuilder
from scripts.run_core_realtime_experiment import metrics_from_prob
from scripts.run_streaming_experiments import AdaptiveA3CStreamModel, make_prior_drift_chunks


class OnlineOnlyNoA3C:
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


def build_variants(args):
    common = dict(
        random_state=args.seed,
        episodes=args.a3c_episodes,
        steps=args.a3c_steps,
        rules=args.a3c_rules,
        max_active_rules=args.a3c_max_active_rules,
        rolling_window=args.a3c_rolling_window,
        calibration_window=args.a3c_calibration_window,
        refit_interval=args.a3c_refit_interval,
        async_refit=True,
        structure_refresh=True,
        structure_refresh_interval=args.a3c_refit_interval,
        structure_refresh_episodes=args.a3c_structure_episodes,
        structure_refresh_tolerance=args.a3c_structure_tolerance,
    )
    return {
        "Full_Realtime_A3C": AdaptiveA3CStreamModel(
            **common,
            workers=args.a3c_workers,
            ultra_fast_update=True,
            prior_shift_lr=args.a3c_prior_shift_lr,
            prior_shift_clip=args.a3c_prior_shift_clip,
        ),
        "Single_Actor": AdaptiveA3CStreamModel(
            **common,
            workers=1,
            ultra_fast_update=True,
            prior_shift_lr=args.a3c_prior_shift_lr,
            prior_shift_clip=args.a3c_prior_shift_clip,
        ),
        "No_Prior_Shift": AdaptiveA3CStreamModel(
            **common,
            workers=args.a3c_workers,
            ultra_fast_update=True,
            prior_shift_lr=0.0,
            prior_shift_clip=0.0,
        ),
        "No_Structure_Refresh": AdaptiveA3CStreamModel(
            random_state=args.seed,
            episodes=args.a3c_episodes,
            steps=args.a3c_steps,
            rules=args.a3c_rules,
            max_active_rules=args.a3c_max_active_rules,
            rolling_window=args.a3c_rolling_window,
            calibration_window=args.a3c_calibration_window,
            refit_interval=args.a3c_refit_interval,
            async_refit=True,
            structure_refresh=False,
            workers=args.a3c_workers,
            ultra_fast_update=True,
            prior_shift_lr=args.a3c_prior_shift_lr,
            prior_shift_clip=args.a3c_prior_shift_clip,
        ),
        "Online_Only_No_A3C": OnlineOnlyNoA3C(args.seed),
    }


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
    for variant, model in build_variants(args).items():
        print(f"[core ablation] dataset={dataset} variant={variant}")
        start = time.perf_counter()
        model.fit(x_train_rules, y_train, x_val_rules, y_val)
        fit_time = time.perf_counter() - start
        for chunk_id, drift_rate, x_chunk, y_chunk in chunks:
            start = time.perf_counter()
            prob = model.predict_proba(x_chunk)[:, 1]
            inference_time = time.perf_counter() - start
            metrics = metrics_from_prob(y_chunk, prob, inference_time)
            update_time = model.update(x_chunk, y_chunk)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "chunk": chunk_id,
                    "drift_positive_rate": drift_rate,
                    "initial_fit_time_sec": fit_time,
                    "foreground_update_time_sec": update_time,
                    "background_update_time_sec": getattr(model, "last_background_update_sec", 0.0),
                    "structure_refreshed": getattr(model, "last_structure_refreshed", 0.0),
                    "rule_additions": getattr(model, "last_rule_additions", 0),
                    "rule_deletions": getattr(model, "last_rule_deletions", 0),
                    "active_rule_count": getattr(model, "last_active_rule_count", 0),
                    "blocking_cycle_time_sec": inference_time + update_time,
                    "meets_10ms_cycle": float((inference_time + update_time) <= 0.010),
                    "meets_5ms_update": float(update_time <= 0.005),
                    **metrics,
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
    out_dir = ROOT / "results" / "core_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        rows.extend(run_dataset(dataset, args))
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "core_ablation_detail.csv", index=False)
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
    summary = detail.groupby("variant", observed=True)[metrics].mean().reset_index()
    summary = summary.sort_values("foreground_update_time_sec")
    summary.to_csv(out_dir / "core_ablation_summary.csv", index=False)
    (out_dir / "core_ablation_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(summary.round(6).to_string(index=False))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
