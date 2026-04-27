"""RT-A3C-MLN v2 implementation that aligns code with the paper.

This package fixes the issues recorded in ``AUDIT_REPORT.md``:
* Reward function matches paper Eq. (11).
* Window-mode thresholds match paper (0.30 / 0.70).
* Forced ablation actually forces the rule-pool branch.
* Train/val/test split is properly stratified.
* Protocol A is treated as an in-window sanity check.
* Protocol B is the main generalization protocol.
* Adapters are clearly labelled as deterministic ranking heuristics.

Use ``scripts/v2/run_all.py --high-heldout`` for the main paper
tables and figures, or omit ``--high-heldout`` for the original v2
split.
"""
