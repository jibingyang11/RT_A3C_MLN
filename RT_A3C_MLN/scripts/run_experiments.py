from pathlib import Path
import argparse
import json
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mln_a3c_experiment.data import DATASETS, download_all, load_dataset
from mln_a3c_experiment.metrics import evaluate_classifier
from mln_a3c_experiment.models import build_models
from mln_a3c_experiment.rules import RuleFeatureBuilder


def run_one_dataset(name, args):
    frame, target = load_dataset(name, ROOT / "data" / "raw", max_samples=args.max_samples)
    y = frame[target].astype(int).to_numpy()
    x = frame.drop(columns=[target])

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.25, random_state=args.seed, stratify=y_trainval
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
    x_test_rules = builder.transform(x_test)

    models = build_models(
        random_state=args.seed,
        a3c_workers=args.a3c_workers,
        a3c_episodes=args.a3c_episodes,
        a3c_steps=args.a3c_steps,
        a3c_rules=args.a3c_rules,
    )

    rows = []
    rule_records = []
    for method_name, model in models:
        start = time.perf_counter()
        model.fit(x_train_rules, y_train, x_val_rules, y_val, builder.rules_)
        fit_time = time.perf_counter() - start
        metrics = evaluate_classifier(model, x_test_rules, y_test)
        selected = model.selected_rule_indices()
        rows.append(
            {
                "dataset": name,
                "method": method_name,
                "samples": len(frame),
                "candidate_rules": len(builder.rules_),
                "selected_rules": len(selected),
                "fit_time_sec": round(fit_time, 4),
                **metrics,
            }
        )
        for rank, idx in enumerate(selected[: args.export_top_rules], start=1):
            weight = model.rule_weight(idx)
            rule_records.append(
                {
                    "dataset": name,
                    "method": method_name,
                    "rank": rank,
                    "rule_index": int(idx),
                    "weight": None if weight is None else float(weight),
                    "rule": builder.rules_[idx].text,
                }
            )
    return rows, rule_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--max-rules", type=int, default=220)
    parser.add_argument("--max-literals", type=int, default=80)
    parser.add_argument("--numeric-bins", type=int, default=4)
    parser.add_argument("--min-support", type=float, default=0.02)
    parser.add_argument("--a3c-workers", type=int, default=4)
    parser.add_argument("--a3c-episodes", type=int, default=18)
    parser.add_argument("--a3c-steps", type=int, default=8)
    parser.add_argument("--a3c-rules", type=int, default=220)
    parser.add_argument("--export-top-rules", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_samples = min(args.max_samples, 2500)
        args.max_rules = min(args.max_rules, 120)
        args.max_literals = min(args.max_literals, 50)
        args.a3c_workers = min(args.a3c_workers, 2)
        args.a3c_episodes = min(args.a3c_episodes, 8)
        args.a3c_steps = min(args.a3c_steps, 6)
        args.a3c_rules = min(args.a3c_rules, 96)

    raw_dir = ROOT / "data" / "raw"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_all(raw_dir)

    all_rows = []
    all_rules = []
    for dataset_name in args.datasets:
        print(f"[run] dataset={dataset_name}")
        rows, rules = run_one_dataset(dataset_name, args)
        all_rows.extend(rows)
        all_rules.extend(rules)
        pd.DataFrame(rows).to_csv(results_dir / f"{dataset_name}_summary.csv", index=False)
        pd.DataFrame(rules).to_csv(results_dir / f"{dataset_name}_rules.csv", index=False)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(results_dir / "summary.csv", index=False)
    pd.DataFrame(all_rules).to_csv(results_dir / "selected_rules.csv", index=False)
    (results_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"[done] wrote {results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
