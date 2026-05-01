from argparse import Namespace
from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_streaming_experiments import run_one_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["adult", "bank"])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[5000, 10000, 20000, 50000])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-rules", type=int, default=220)
    parser.add_argument("--max-literals", type=int, default=80)
    parser.add_argument("--numeric-bins", type=int, default=5)
    parser.add_argument("--min-support", type=float, default=0.015)
    parser.add_argument("--stream-fraction", type=float, default=0.70)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=6)
    parser.add_argument("--a3c-steps", type=int, default=5)
    parser.add_argument("--a3c-rules", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / "results" / "large_scale" / "sample_scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_size in args.sample_sizes:
        for dataset in args.datasets:
            print(f"[scale] sample_size={sample_size} dataset={dataset}")
            run_args = Namespace(**vars(args))
            run_args.max_samples = sample_size
            dataset_rows = run_one_dataset(dataset, run_args)
            for row in dataset_rows:
                row["sample_size"] = sample_size
            rows.extend(dataset_rows)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "sample_scale_detail.csv", index=False)
    metrics = ["accuracy", "f1", "roc_auc", "log_loss", "latency_ms_per_sample", "update_time_sec", "initial_fit_time_sec"]
    summary = detail.groupby(["sample_size", "method"], observed=True)[metrics].mean().reset_index()
    summary.to_csv(out_dir / "sample_scale_summary.csv", index=False)
    print(summary.sort_values(["sample_size", "f1"], ascending=[True, False]).round(6).to_string(index=False))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
