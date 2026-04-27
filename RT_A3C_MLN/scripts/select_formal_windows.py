from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.window_selection import build_large_scale_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='分层选择 mixed windows，构造更正式的大规模实验计划。')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--per-bin', type=int, default=12, help='每个 attack ratio 分层抽取的窗口数')
    parser.add_argument('--low-attack-max', type=float, default=0.30)
    parser.add_argument('--high-attack-min', type=float, default=0.70)
    parser.add_argument('--min-window-gap', type=int, default=2, help='选中的窗口 id 之间至少间隔多少，避免 50% 重叠')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--out-name', type=str, default='formal_mixed_windows_plan.csv')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = build_large_scale_plan(
        project_root=args.project_root,
        out_name=args.out_name,
        low_attack_max=args.low_attack_max,
        high_attack_min=args.high_attack_min,
        low_per_bin=args.per_bin,
        mid_per_bin=args.per_bin,
        high_per_bin=args.per_bin,
        min_window_gap=args.min_window_gap,
        random_state=args.random_state,
    )
    print(selected)
    print('selected windows =', len(selected))
    if 'attack_bin' in selected.columns:
        print(selected['attack_bin'].value_counts())


if __name__ == '__main__':
    main()
