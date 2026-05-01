from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS, load_dataset
from mln_a3c_experiment.rules import RuleFeatureBuilder
from scripts.run_streaming_experiments import build_stream_models, _prob_metrics


STRENGTHS = {
    "low": 0.12,
    "medium": 0.25,
    "high": 0.38,
}


def make_strength_chunks(x, y, chunks, chunk_size, amplitude, random_state):
    rng = np.random.default_rng(random_state)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    base = float(np.mean(y))
    rates = np.clip(base + amplitude * np.sin(np.linspace(0, 2.5 * np.pi, chunks)), 0.03, 0.97)
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


def run_one_dataset(name, strength_name, amplitude, args):
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
    )
    for model in models.values():
        model.fit(x_train_rules, y_train, x_val_rules, y_val)

    rows = []
    chunks = make_strength_chunks(
        x_stream_rules,
        y_stream,
        chunks=args.chunks,
        chunk_size=args.chunk_size,
        amplitude=amplitude,
        random_state=args.seed + 191,
    )
    for chunk_id, drift_rate, x_chunk, y_chunk in chunks:
        for method, model in models.items():
            start = time.perf_counter()
            prob = model.predict_proba(x_chunk)[:, 1]
            latency = time.perf_counter() - start
            metrics = _prob_metrics(y_chunk, prob, latency)
            update_time = model.update(x_chunk, y_chunk)
            rows.append(
                {
                    "dataset": name,
                    "strength": strength_name,
                    "amplitude": amplitude,
                    "method": method,
                    "chunk": chunk_id,
                    "positive_rate": drift_rate,
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
    out_dir = ROOT / "results" / "additional" / "drift_strength"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for strength_name, amplitude in STRENGTHS.items():
        for dataset in args.datasets:
            print(f"[drift] strength={strength_name} dataset={dataset}")
            rows.extend(run_one_dataset(dataset, strength_name, amplitude, args))
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "drift_strength_detail.csv", index=False)
    metrics = ["accuracy", "f1", "roc_auc", "log_loss", "latency_ms_per_sample", "update_time_sec"]
    summary = detail.groupby(["strength", "method"], observed=True)[metrics].mean().reset_index()
    summary.to_csv(out_dir / "drift_strength_summary.csv", index=False)
    print(summary.sort_values(["strength", "f1"], ascending=[True, False]).round(6).to_string(index=False))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
