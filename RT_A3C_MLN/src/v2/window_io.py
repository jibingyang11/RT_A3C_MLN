"""Reusable window data loaders for the v2 pipeline.

This module is the single point of entry to the per-window binary
features and labels, so all v2 scripts share the same I/O behaviour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_windows_plan(project_root: Path, plan_name: str = "windows_mixed.csv") -> pd.DataFrame:
    plan_path = project_root / "data" / "stream" / "swat" / plan_name
    return pd.read_csv(plan_path)


def load_window_binary_and_labels(
    project_root: Path, window_id: int, plan_name: str = "windows_mixed.csv"
) -> tuple[pd.DataFrame, pd.Series, dict]:
    binary_path = (
        project_root / "data" / "stream" / "swat" / "window_binary_data" / f"window_{window_id}_binary.csv"
    )
    if not binary_path.exists():
        raise FileNotFoundError(binary_path)
    df_binary = pd.read_csv(binary_path)

    plan_df = load_windows_plan(project_root, plan_name)
    row = plan_df[plan_df["window_id"] == window_id]
    if row.empty:
        raise ValueError(f"window_id={window_id} not found in {plan_name}")
    row = row.iloc[0]
    start_idx = int(row["start_idx"])
    end_idx = int(row["end_idx"])

    y_path = project_root / "data" / "processed" / "swat" / "y_filled.csv"
    y_df = pd.read_csv(y_path)
    if "label" in y_df.columns:
        y_all = y_df["label"].astype(int).reset_index(drop=True)
    else:
        y_all = y_df.iloc[:, 0].astype(int).reset_index(drop=True)
    labels = y_all.iloc[start_idx:end_idx].reset_index(drop=True)
    meta = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "n_samples": int(end_idx - start_idx),
        "attack_ratio": float(labels.mean()),
    }
    return df_binary, labels, meta


def assign_attack_bin(attack_ratio: float, low: float = 0.30, high: float = 0.70) -> str:
    if attack_ratio < low:
        return "low"
    if attack_ratio >= high:
        return "high"
    return "mid"


def load_formal_plan(project_root: Path) -> pd.DataFrame:
    """Loads (or creates) the formal mixed-window plan with attack_bin column."""
    plan_path = project_root / "data" / "stream" / "swat" / "formal_mixed_windows_plan.csv"
    if plan_path.exists():
        plan = pd.read_csv(plan_path)
    else:
        # Fall back to the generic windows file with attack_ratio computed below
        plan = load_windows_plan(project_root, "windows_mixed.csv")
        if "attack_ratio" not in plan.columns:
            raise ValueError("windows_mixed.csv does not contain attack_ratio")
    if "attack_bin" not in plan.columns:
        plan["attack_bin"] = plan["attack_ratio"].apply(assign_attack_bin)
    return plan
