from pathlib import Path
import argparse
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS, load_dataset
from mln_a3c_experiment.rules import RuleFeatureBuilder
from scripts.run_streaming_experiments import AdaptiveA3CStreamModel, _prob_metrics, make_prior_drift_chunks


def build_variants(args):
    common = dict(
        random_state=args.seed,
        workers=args.a3c_workers,
        episodes=args.a3c_episodes,
        steps=args.a3c_steps,
        rules=args.a3c_rules,
    )
    return {
        "Full_Background_Refresh": AdaptiveA3CStreamModel(
            **common,
            fast_search_mode=True,
            fast_update=False,
            rolling_window=2400,
            calibration_window=900,
            refit_interval=1,
            async_refit=True,
        ),
        "Ultra_Fast_Update": AdaptiveA3CStreamModel(
            **common,
            fast_search_mode=True,
            ultra_fast_update=True,
            max_active_rules=96,
            rolling_window=600,
            calibration_window=300,
            refit_interval=10,
            async_refit=True,
            prior_shift_lr=0.20,
            prior_shift_clip=1.5,
        ),
        "No_Fast_Search": AdaptiveA3CStreamModel(
            **common,
            fast_search_mode=False,
            ultra_fast_update=True,
            max_active_rules=96,
            rolling_window=600,
            calibration_window=300,
            refit_interval=10,
            async_refit=True,
            prior_shift_lr=0.20,
            prior_shift_clip=1.5,
        ),
        "No_Prior_Shift": AdaptiveA3CStreamModel(
            **common,
            fast_search_mode=True,
            ultra_fast_update=True,
            max_active_rules=96,
            rolling_window=600,
            calibration_window=300,
            refit_interval=10,
            async_refit=True,
            prior_shift_lr=0.0,
            prior_shift_clip=0.0,
        ),
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
        start = time.perf_counter()
        model.fit(x_train_rules, y_train, x_val_rules, y_val)
        fit_time = time.perf_counter() - start
        for chunk_id, drift_rate, x_chunk, y_chunk in chunks:
            start = time.perf_counter()
            prob = model.predict_proba(x_chunk)[:, 1]
            latency = time.perf_counter() - start
            metrics = _prob_metrics(y_chunk, prob, latency)
            update_time = model.update(x_chunk, y_chunk)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "chunk": chunk_id,
                    "drift_positive_rate": drift_rate,
                    "initial_fit_time_sec": fit_time,
                    "update_time_sec": update_time,
                    "background_update_time_sec": getattr(model, "last_background_update_sec", 0.0),
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
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=5)
    parser.add_argument("--a3c-steps", type=int, default=5)
    parser.add_argument("--a3c-rules", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / "results" / "latest_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        print(f"[latest ablation] dataset={dataset}")
        rows.extend(run_dataset(dataset, args))
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "latest_speed_ablation_detail.csv", index=False)
    metrics = [
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "latency_ms_per_sample",
        "update_time_sec",
        "background_update_time_sec",
        "initial_fit_time_sec",
    ]
    summary = detail.groupby("variant", observed=True)[metrics].mean().sort_values("update_time_sec")
    summary.to_csv(out_dir / "latest_speed_ablation_summary.csv")
    print(summary.round(6).to_string())
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
