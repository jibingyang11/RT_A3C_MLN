"""Convenience orchestrator for the v2 experiments.

Run order matters:
  1. run_protocolA.py   -> per-window sanity check
  2. run_protocolB.py / run_protocolB_high_heldout.py
     -> trains and saves warm-start + A3C checkpoints (per seed)
  3. run_baselines.py   -> uses Protocol B test windows + checkpoints
  4. run_adapter_methods.py -> uses Protocol B test windows + checkpoints
  5. run_ablation_pool.py / run_ablation_pool_high.py
  6. run_detection_utility.py -> uses A3C checkpoint to derive selected mask
  7. run_runtime.py
  8. make_paper_figures_v2.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "v2"


def run_step(name: str, args: list[str]) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / name), *args]
    print("\n" + "=" * 60)
    print("RUNNING:", " ".join(cmd))
    print("=" * 60)
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))


def main(seeds: list[int], episodes: int, workers: int, high_heldout: bool):
    seeds_str = [str(s) for s in seeds]
    run_step("run_protocolA.py", [])
    if high_heldout:
        run_step(
            "run_protocolB_high_heldout.py",
            ["--seeds", *seeds_str, "--episodes", str(episodes), "--workers", str(workers)],
        )
        run_step("run_baselines.py", ["--seeds", *seeds_str, "--high-heldout"])
        run_step(
            "run_adapter_methods.py",
            ["--seeds", *seeds_str, "--episodes", str(episodes), "--high-heldout"],
        )
        run_step(
            "run_ablation_pool_high.py",
            ["--seeds", *seeds_str, "--episodes", str(episodes), "--workers", str(workers)],
        )
        run_step("run_detection_utility.py", ["--seeds", *seeds_str, "--high-heldout"])
        run_step("run_runtime.py", ["--episodes", str(episodes), "--high-heldout"])
    else:
        run_step("run_protocolB.py", ["--seeds", *seeds_str, "--episodes", str(episodes), "--workers", str(workers)])
        run_step("run_baselines.py", ["--seeds", *seeds_str])
        run_step("run_adapter_methods.py", ["--seeds", *seeds_str, "--episodes", str(episodes)])
        run_step("run_ablation_pool.py", [])
        run_step("run_detection_utility.py", ["--seeds", *seeds_str])
        run_step("run_runtime.py", ["--episodes", str(episodes)])
    run_step("make_paper_figures_v2.py", [])
    print("\nAll v2 experiments finished. CSVs are in outputs/logs/v2/, figures in outputs/figures/v2/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 42])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--high-heldout", action="store_true")
    args = parser.parse_args()
    main(args.seeds, args.episodes, args.workers, args.high_heldout)
