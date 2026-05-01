from argparse import Namespace
from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS
from scripts.run_streaming_experiments import run_one_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 3, 5, 7, 11])
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
    out_dir = ROOT / "results" / "additional" / "multiseed_streaming"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in args.seeds:
        seed_args = Namespace(**vars(args))
        seed_args.seed = seed
        for dataset in args.datasets:
            print(f"[multiseed] seed={seed} dataset={dataset}")
            rows = run_one_dataset(dataset, seed_args)
            for row in rows:
                row["seed"] = seed
            all_rows.extend(rows)

    detail = pd.DataFrame(all_rows)
    detail.to_csv(out_dir / "multiseed_streaming_detail.csv", index=False)

    metrics = ["accuracy", "f1", "roc_auc", "log_loss", "latency_ms_per_sample", "update_time_sec"]
    per_seed = detail.groupby(["seed", "method"], observed=True)[metrics].mean().reset_index()
    per_seed.to_csv(out_dir / "multiseed_streaming_per_seed.csv", index=False)

    mean = per_seed.groupby("method", observed=True)[metrics].mean()
    std = per_seed.groupby("method", observed=True)[metrics].std(ddof=1).add_suffix("_std")
    summary = mean.join(std).sort_values("f1", ascending=False)
    summary.to_csv(out_dir / "multiseed_streaming_summary.csv")
    print(summary.round(6).to_string())
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
