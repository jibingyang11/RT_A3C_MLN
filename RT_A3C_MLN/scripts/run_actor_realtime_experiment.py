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


def prepare_dataset(name, args):
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
    return (
        builder.fit_transform(x_train, y_train),
        y_train,
        builder.transform(x_val),
        y_val,
        builder.transform(x_stream),
        y_stream,
    )


def run_config(dataset, mode, workers, chunk_size, args):
    x_train, y_train, x_val, y_val, x_stream, y_stream = prepare_dataset(dataset, args)
    model = AdaptiveA3CStreamModel(
        random_state=args.seed,
        workers=workers,
        episodes=args.a3c_episodes,
        steps=args.a3c_steps,
        rules=min(args.a3c_rules, args.max_rules),
    )
    start = time.perf_counter()
    model.fit(x_train, y_train, x_val, y_val)
    fit_time = time.perf_counter() - start
    rows = []
    chunks = make_prior_drift_chunks(
        x_stream,
        y_stream,
        chunks=args.chunks,
        chunk_size=chunk_size,
        random_state=args.seed + 233,
    )
    for chunk_id, drift_rate, x_chunk, y_chunk in chunks:
        start = time.perf_counter()
        prob = model.predict_proba(x_chunk)[:, 1]
        latency = time.perf_counter() - start
        metrics = _prob_metrics(y_chunk, prob, latency)
        update_time = model.update(x_chunk, y_chunk)
        rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "workers": workers,
                "chunk_size": chunk_size,
                "chunk": chunk_id,
                "drift_positive_rate": drift_rate,
                "initial_fit_time_sec": fit_time,
                "update_time_sec": update_time,
                **metrics,
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=["adult", "bank", "spambase"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-rules", type=int, default=160)
    parser.add_argument("--max-literals", type=int, default=60)
    parser.add_argument("--numeric-bins", type=int, default=4)
    parser.add_argument("--min-support", type=float, default=0.02)
    parser.add_argument("--stream-fraction", type=float, default=0.65)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--worker-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[180, 360, 720])
    parser.add_argument("--a3c-episodes", type=int, default=6)
    parser.add_argument("--a3c-steps", type=int, default=5)
    parser.add_argument("--a3c-rules", type=int, default=140)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / "results" / "additional" / "actor_realtime"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        for workers in args.worker_counts:
            print(f"[actor] dataset={dataset} workers={workers}")
            rows.extend(run_config(dataset, "worker_sweep", workers, 360, args))
        for chunk_size in args.chunk_sizes:
            print(f"[realtime] dataset={dataset} chunk_size={chunk_size}")
            rows.extend(run_config(dataset, "chunk_sweep", 4, chunk_size, args))
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "actor_realtime_detail.csv", index=False)
    metrics = ["accuracy", "f1", "roc_auc", "log_loss", "latency_ms_per_sample", "update_time_sec", "initial_fit_time_sec"]
    summary = detail.groupby(["mode", "workers", "chunk_size"], observed=True)[metrics].mean().reset_index()
    summary.to_csv(out_dir / "actor_realtime_summary.csv", index=False)
    print(summary.round(6).to_string(index=False))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
