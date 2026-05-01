from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS, load_dataset
from mln_a3c_experiment.rules import RuleFeatureBuilder
from scripts.run_streaming_experiments import AdaptiveA3CStreamModel, make_prior_drift_chunks


def extract_rule_weights(model, rules, dataset, chunk, top_k):
    coef = np.ravel(model.adaptive.coef_)
    active_rules = model.active_rules
    order = np.argsort(np.abs(coef))[::-1][:top_k]
    rows = []
    for rank, local_idx in enumerate(order, start=1):
        rule_idx = int(active_rules[int(local_idx)])
        rows.append(
            {
                "dataset": dataset,
                "chunk": chunk,
                "rank": rank,
                "rule_index": rule_idx,
                "weight": float(coef[int(local_idx)]),
                "abs_weight": float(abs(coef[int(local_idx)])),
                "rule": rules[rule_idx].text if rule_idx < len(rules) else f"rule_{rule_idx}",
            }
        )
    return rows


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

    model = AdaptiveA3CStreamModel(
        random_state=args.seed,
        workers=args.a3c_workers,
        episodes=args.a3c_episodes,
        steps=args.a3c_steps,
        rules=min(args.a3c_rules, args.max_rules),
    )
    model.fit(x_train_rules, y_train, x_val_rules, y_val)

    snapshots = extract_rule_weights(model, builder.rules_, name, 0, args.top_k)
    full_initial = {
        int(rule_idx): float(weight)
        for rule_idx, weight in zip(model.active_rules, np.ravel(model.adaptive.coef_))
    }

    chunks = make_prior_drift_chunks(
        x_stream_rules,
        y_stream,
        chunks=args.chunks,
        chunk_size=args.chunk_size,
        random_state=args.seed + 307,
    )
    positive_rates = []
    for chunk_id, drift_rate, x_chunk, y_chunk in chunks:
        model.update(x_chunk, y_chunk)
        positive_rates.append({"dataset": name, "chunk": chunk_id, "positive_rate": drift_rate})
        snapshots.extend(extract_rule_weights(model, builder.rules_, name, chunk_id, args.top_k))

    full_final = {
        int(rule_idx): float(weight)
        for rule_idx, weight in zip(model.active_rules, np.ravel(model.adaptive.coef_))
    }
    delta_rows = []
    for rule_idx in sorted(set(full_initial).union(full_final)):
        initial = full_initial.get(rule_idx, 0.0)
        final = full_final.get(rule_idx, 0.0)
        delta_rows.append(
            {
                "dataset": name,
                "rule_index": rule_idx,
                "initial_weight": initial,
                "final_weight": final,
                "delta": final - initial,
                "abs_delta": abs(final - initial),
                "rule": builder.rules_[rule_idx].text if rule_idx < len(builder.rules_) else f"rule_{rule_idx}",
            }
        )
    delta_rows.sort(key=lambda row: row["abs_delta"], reverse=True)
    return snapshots, delta_rows[: args.delta_top_k], positive_rates


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
    parser.add_argument("--chunk-size", type=int, default=360)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=6)
    parser.add_argument("--a3c-steps", type=int, default=5)
    parser.add_argument("--a3c-rules", type=int, default=140)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--delta-top-k", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / "results" / "additional" / "rule_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = []
    deltas = []
    rates = []
    for dataset in args.datasets:
        print(f"[rules] dataset={dataset}")
        snap_rows, delta_rows, rate_rows = run_one_dataset(dataset, args)
        snapshots.extend(snap_rows)
        deltas.extend(delta_rows)
        rates.extend(rate_rows)
    pd.DataFrame(snapshots).to_csv(out_dir / "rule_weight_snapshots.csv", index=False)
    pd.DataFrame(deltas).to_csv(out_dir / "rule_weight_deltas.csv", index=False)
    pd.DataFrame(rates).to_csv(out_dir / "rule_drift_rates.csv", index=False)
    print(pd.DataFrame(deltas).groupby("dataset", observed=True).head(5).to_string(index=False))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
