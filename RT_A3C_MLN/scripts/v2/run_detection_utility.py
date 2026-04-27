"""Detection-utility evaluation of the selected rule mask (v2).

Section 6.5 of the paper. Important caveats baked in:
  * We evaluate ONLY on Protocol B held-out test windows. The mining
    rules used in the pool already saw labels during rule mining; that
    is unavoidable. We therefore audit how much of the F1 is "free"
    by also reporting an "all candidate rules kept" detector that
    matches the same rule pool BEFORE A3C's keep/disable decisions.
  * The reported P/R/F1 is computed at transaction (lag-aware) level.
    A transaction is flagged ATTACK iff at least one ACTIVE attack-
    consequent rule fires on it, and otherwise NORMAL.
  * AUROC is computed by ranking transactions by the WEIGHTED count
    of active attack-rules that fire (continuous score) and the
    transaction-level binary attack label.
  * The selected human-readable rule list is exported separately for
    audit.

Outputs:
  outputs/logs/v2/detection_utility_per_seed.csv
  outputs/logs/v2/detection_utility.csv  (paper Table 6)
  outputs/logs/v2/selected_rules_for_paper.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transaction_utils import build_transactions
from src.rl.simple_env_v2 import PaperRuleEnv
from src.v2.pipeline import prepare_window_artifacts, stratified_split
from src.v2.policy import PolicyNetV2, greedy_evaluate
from src.v2.rule_pool import LABEL_ATTACK, MiningConfigV2
from src.v2.window_io import load_formal_plan, load_window_binary_and_labels


def _antecedent_set(rule_row) -> set[str]:
    s = str(rule_row.get("antecedent_str", ""))
    if not s:
        return set()
    return set([t.strip() for t in s.split(" & ") if t.strip()])


def _rule_fires(antecedent: set[str], transaction_items: set[str]) -> bool:
    if not antecedent:
        return False
    return antecedent.issubset(transaction_items)


def _evaluate_transactions(
    transactions: list[list[str]],
    labels: pd.Series,
    rule_pool: pd.DataFrame,
    active_mask: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    rule_records = rule_pool.to_dict(orient="records")
    attack_rule_ix = [
        i
        for i, r in enumerate(rule_records)
        if str(r.get("consequent_str", "")) == LABEL_ATTACK
    ]
    if not attack_rule_ix:
        # no attack rules → trivial detector predicts normal
        n = len(transactions)
        y_true = np.asarray(labels.iloc[: n].astype(int).tolist(), dtype=int)
        y_pred = np.zeros(n, dtype=int)
        score = np.zeros(n, dtype=float)
        return ({"P": 0.0, "R": 0.0, "F1": 0.0, "AUROC": 0.5}, y_true, score)

    active_attack_ix = [i for i in attack_rule_ix if active_mask[i] == 1]
    rule_antecedents = [(_antecedent_set(rule_records[i]), float(rule_records[i].get("weight", 1.0))) for i in active_attack_ix]

    y_pred = []
    score = []
    for tx in transactions:
        items = set(tx)
        any_fire = False
        cont_score = 0.0
        for ant, w in rule_antecedents:
            if _rule_fires(ant, items):
                any_fire = True
                cont_score += max(w, 0.0)
        y_pred.append(1 if any_fire else 0)
        score.append(cont_score)

    y_true = np.asarray(labels.iloc[: len(y_pred)].astype(int).tolist(), dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    score = np.asarray(score, dtype=float)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    if y_true.min() != y_true.max():
        try:
            auroc = float(roc_auc_score(y_true, score))
        except Exception:
            auroc = float("nan")
    else:
        auroc = float("nan")

    return (
        {"P": float(p), "R": float(r), "F1": float(f1), "AUROC": auroc, "n_active_attack_rules": int(len(active_attack_ix))},
        y_true,
        score,
    )


def high_heldout_split(plan_df: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for attack_bin in ["low", "mid", "high"]:
        ids = plan_df[plan_df["attack_bin"] == attack_bin]["window_id"].astype(int).tolist()
        rng.shuffle(ids)
        if attack_bin == "high":
            train_n = min(1, len(ids))
            val_n = 0
            test_n = min(2, max(0, len(ids) - train_n))
        else:
            train_n = min(4, len(ids))
            val_n = min(1, max(0, len(ids) - train_n))
            test_n = min(2, max(0, len(ids) - train_n - val_n))
        out["train"].extend(ids[:train_n])
        out["val"].extend(ids[train_n: train_n + val_n])
        out["test"].extend(ids[train_n + val_n: train_n + val_n + test_n])
    return out


def main(seeds: list[int], high_heldout: bool = False):
    plan = load_formal_plan(PROJECT_ROOT)
    out_dir = PROJECT_ROOT / "outputs" / "logs" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = PROJECT_ROOT / "outputs" / "v2" / "models"

    per_seed_rows = []
    selected_rules_records = []
    for seed in seeds:
        prefix = "protocolB_high" if high_heldout else "protocolB"
        a3c_path = model_dir / f"{prefix}_seed{seed}_a3c.pth"
        if not a3c_path.exists():
            print(f"[seed {seed}] A3C checkpoint missing — run run_protocolB.py first.")
            continue
        model = PolicyNetV2()
        model.load_state_dict(torch.load(a3c_path, map_location="cpu"))
        model.eval()

        splits = (
            high_heldout_split(plan, seed=seed)
            if high_heldout
            else stratified_split(plan, seed=seed, train_per_bin=4, val_per_bin=1, test_per_bin=2)
        )
        for wid in splits["test"]:
            try:
                art = prepare_window_artifacts(PROJECT_ROOT, wid, MiningConfigV2())
            except Exception as e:
                print(f"[detection seed {seed}] skip window {wid}: {e}")
                continue
            df_binary, labels, _ = load_window_binary_and_labels(PROJECT_ROOT, wid)
            transactions = build_transactions(df_binary, lag=2)
            n = min(len(transactions), len(labels))
            transactions = transactions[:n]
            labels = labels.iloc[:n].reset_index(drop=True)

            # Active mask AFTER greedy A3C
            metrics_info = greedy_evaluate(model, art.state_dict, env_cls=PaperRuleEnv)
            actions = np.asarray(metrics_info["actions"], dtype=int)
            active_mask = (actions == 0).astype(int)  # KEEP=0 → active

            our_metrics, y_true, score = _evaluate_transactions(
                transactions, labels, art.rule_pool_df, active_mask
            )

            # baseline: keep all candidate rules ('no selection')
            all_active = np.ones_like(active_mask)
            base_metrics, _, _ = _evaluate_transactions(
                transactions, labels, art.rule_pool_df, all_active
            )

            per_seed_rows.append(
                {
                    "seed": seed,
                    "window_id": wid,
                    "attack_ratio": art.attack_ratio,
                    "ours_P": our_metrics["P"],
                    "ours_R": our_metrics["R"],
                    "ours_F1": our_metrics["F1"],
                    "ours_AUROC": our_metrics["AUROC"],
                    "ours_n_active_attack_rules": our_metrics["n_active_attack_rules"],
                    "all_kept_P": base_metrics["P"],
                    "all_kept_R": base_metrics["R"],
                    "all_kept_F1": base_metrics["F1"],
                    "all_kept_AUROC": base_metrics["AUROC"],
                    "all_kept_n_active_attack_rules": base_metrics["n_active_attack_rules"],
                }
            )

            # Save selected rules per (seed, window) for audit
            for i, row in art.rule_pool_df.iterrows():
                if active_mask[i] == 1 and row["consequent_str"] == LABEL_ATTACK:
                    selected_rules_records.append(
                        {
                            "seed": seed,
                            "window_id": wid,
                            "rule_index": int(i),
                            "antecedent_str": row["antecedent_str"],
                            "consequent_str": row["consequent_str"],
                            "support": row["support"],
                            "confidence": row["confidence"],
                            "lift": row["lift"],
                            "weight": row["weight"],
                        }
                    )

    if per_seed_rows:
        per_seed = pd.DataFrame(per_seed_rows)
        suffix = "_high" if high_heldout else ""
        per_seed.to_csv(out_dir / f"detection_utility{suffix}_per_seed.csv", index=False)

        agg = (
            per_seed[
                [
                    "ours_P",
                    "ours_R",
                    "ours_F1",
                    "ours_AUROC",
                    "ours_n_active_attack_rules",
                    "all_kept_P",
                    "all_kept_R",
                    "all_kept_F1",
                    "all_kept_AUROC",
                    "all_kept_n_active_attack_rules",
                ]
            ]
            .agg(["mean", "std"])
            .T
        )
        agg.to_csv(out_dir / f"detection_utility{suffix}.csv")

    if selected_rules_records:
        suffix = "_high" if high_heldout else ""
        pd.DataFrame(selected_rules_records).to_csv(
            out_dir / f"selected_rules_for_paper{suffix}.csv", index=False
        )
        print(f"Wrote {len(selected_rules_records)} selected attack-rule rows for audit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--high-heldout", action="store_true")
    args = parser.parse_args()
    main(args.seeds, high_heldout=args.high_heldout)
