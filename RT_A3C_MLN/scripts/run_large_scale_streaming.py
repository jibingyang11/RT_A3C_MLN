from argparse import Namespace
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS
from scripts.run_streaming_experiments import run_one_dataset


def predictive_score(detail: pd.DataFrame) -> pd.Series:
    metrics = [("accuracy", False), ("f1", False), ("roc_auc", False), ("log_loss", True)]
    rows = []
    for dataset, group in detail.groupby("dataset", observed=True):
        values = group.groupby("method", observed=True)[[metric for metric, _ in metrics]].mean()
        score = pd.Series(0.0, index=values.index)
        for metric, lower_is_better in metrics:
            col = values[metric]
            lo = float(col.min())
            hi = float(col.max())
            if hi <= lo:
                norm = col * 0 + 1
            elif lower_is_better:
                norm = (hi - col) / (hi - lo)
            else:
                norm = (col - lo) / (hi - lo)
            score = score + norm
        rows.append((score / len(metrics)).rename(dataset))
    return pd.concat(rows, axis=1).mean(axis=1).sort_values(ascending=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--max-rules", type=int, default=240)
    parser.add_argument("--max-literals", type=int, default=90)
    parser.add_argument("--numeric-bins", type=int, default=5)
    parser.add_argument("--min-support", type=float, default=0.015)
    parser.add_argument("--stream-fraction", type=float, default=0.70)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=8)
    parser.add_argument("--a3c-steps", type=int, default=6)
    parser.add_argument("--a3c-rules", type=int, default=220)
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "large_scale")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        print(f"[large] dataset={dataset}")
        dataset_rows = run_one_dataset(dataset, Namespace(**vars(args)))
        rows.extend(dataset_rows)
        pd.DataFrame(dataset_rows).to_csv(out_dir / f"{dataset}_large_streaming.csv", index=False)

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "large_streaming_detail.csv", index=False)
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
    aggregate = detail.groupby("method", observed=True)[metrics].mean().sort_values("f1", ascending=False)
    aggregate.to_csv(out_dir / "large_streaming_aggregate.csv")
    score = predictive_score(detail)
    score.to_csv(out_dir / "large_streaming_predictive_scores.csv", header=["score"])
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (out_dir / "large_streaming_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(aggregate.round(6).to_string())
    print("\nPredictive score:")
    print(score.round(6).to_string())
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
