"""Regenerate every figure used in the paper.

Requires matplotlib.  The script expects the usual
``project_root/outputs/logs`` layout already to contain the CSV files
produced by the preceding experiments.  Missing CSVs are skipped with
a warning so that partial re-runs still work.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plot.paper_plots import make_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build every paper figure from the logs/ CSVs.')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    produced = make_all(args.project_root)
    print(f'[done] {len(produced)} figure(s) produced under {args.project_root / "outputs" / "figures"}')
    for p in produced:
        print('  -', p)


if __name__ == '__main__':
    main()
