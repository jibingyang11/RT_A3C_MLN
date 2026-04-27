"""Pure runtime breakdown for our method and adapters (v2).

Splits training time from inference latency so the paper Table 5 (or
the new Table tab:runtime) does NOT mix the two columns.

Outputs:
  outputs/logs/v2/runtime_by_method.csv
  outputs/logs/v2/runtime_per_window.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.mln_family.adapters import (
    LogicGuardAdapter,
    MLN4KBAdapter,
    NeSyCAdapter,
    NPLLAdapter,
    NeuralRuleListsAdapter,
    QuantifiedNeuralMLNAdapter,
    IncrementalAffordanceMLNAdapter,
    RuleExtractionADAdapter,
)
from src.adapters.rl_family.adapters import (
    DNNAEDDTrainer,
    DRLNIDSTrainer,
    InterpretableACTrainer,
    StrAEmDDTrainer,
    StreamingADBestTrainer,
)
from src.rl.simple_env_v2 import PaperRuleEnv
from src.v2.pipeline import prepare_window_artifacts, stratified_split
from src.v2.policy import (
    A3CConfigV2,
    PolicyNetV2,
    WarmStartConfigV2,
    a3c_fine_tune,
    set_global_seed,
    warmstart_classifier,
)
from src.v2.rule_pool import MiningConfigV2
from src.v2.window_io import load_formal_plan


def time_block(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


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
        out["val"].extend(ids[train_n : train_n + val_n])
        out["test"].extend(ids[train_n + val_n : train_n + val_n + test_n])
    return out


def main(seed: int, episodes: int, high_heldout: bool = False):
    plan = load_formal_plan(PROJECT_ROOT)
    out_dir = PROJECT_ROOT / "outputs" / "logs" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = PROJECT_ROOT / "outputs" / "v2" / "models"

    splits = (
        high_heldout_split(plan, seed=seed)
        if high_heldout
        else stratified_split(plan, seed=seed, train_per_bin=4, val_per_bin=1, test_per_bin=2)
    )

    def _safe(ids, label):
        out = []
        for wid in ids:
            try:
                out.append(prepare_window_artifacts(PROJECT_ROOT, wid, MiningConfigV2()))
            except Exception as e:
                print(f"[runtime {label}] skip window {wid}: {e}")
        return out

    train_arts = _safe(splits["train"], "train")
    test_arts = _safe(splits["test"], "test")
    if not train_arts or not test_arts:
        raise RuntimeError("runtime: empty train or test set after skipping")

    rows = []

    # ----- Ours: warm-start + A3C -----
    set_global_seed(seed)
    X_train = np.concatenate([a.X_ws for a in train_arts], axis=0)
    y_train = np.concatenate([a.y_ws for a in train_arts], axis=0)
    ws_path = model_dir / f"runtime_seed{seed}_warmstart.pth"
    (train_acc, ws_model), warmstart_sec = time_block(
        warmstart_classifier, X_train, y_train, ws_path, WarmStartConfigV2(seed=seed)
    )
    train_states = [a.state_dict for a in train_arts]
    a3c_cfg = A3CConfigV2(num_workers=4, episodes_per_worker=episodes, seed=seed)
    (a3c_model,), a3c_sec = time_block(
        lambda: (a3c_fine_tune(ws_model, train_states, env_cls=PaperRuleEnv, cfg=a3c_cfg),)
    )
    total_train_sec = warmstart_sec + a3c_sec

    inference_ms = []
    a3c_device = next(a3c_model.parameters()).device
    for art in test_arts:
        env = PaperRuleEnv(art.state_dict)
        state = env.reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in range(art.state_dict["num_rules"]):
                x = torch.tensor(state, dtype=torch.float32, device=a3c_device).unsqueeze(0)
                logits, _ = a3c_model(x)
                a = int(torch.argmax(logits, dim=1).item())
                state, _, _, _ = env.step(t, a)
        inference_ms.append((time.perf_counter() - t0) * 1000.0)
    rows.append(
        {
            "method": "RT-A3C-MLN (ours)",
            "warmstart_train_sec": warmstart_sec,
            "a3c_train_sec": a3c_sec,
            "total_train_sec": total_train_sec,
            "inference_ms_per_window_mean": float(np.mean(inference_ms)),
            "inference_ms_per_window_std": float(np.std(inference_ms)),
        }
    )

    per_window_rows = [
        {"method": "RT-A3C-MLN (ours)", "window_id": art.window_id, "inference_ms": ms}
        for art, ms in zip(test_arts, inference_ms)
    ]

    # ----- MLN-family adapters (no real training) -----
    mln_pairs = [
        ("MLN4KBAdapter", MLN4KBAdapter()),
        ("RuleExtractionADAdapter", RuleExtractionADAdapter()),
        ("NPLLAdapter", NPLLAdapter()),
        ("QuantifiedNeuralMLNAdapter", QuantifiedNeuralMLNAdapter()),
        ("IncrementalAffordanceMLNAdapter", IncrementalAffordanceMLNAdapter()),
        ("NeSyCAdapter", NeSyCAdapter()),
        ("NeuralRuleListsAdapter", NeuralRuleListsAdapter()),
        ("LogicGuardAdapter", LogicGuardAdapter()),
    ]
    for name, ad in mln_pairs:
        latency = []
        for art in test_arts:
            t0 = time.perf_counter()
            ad.predict(art.state_dict)
            latency.append((time.perf_counter() - t0) * 1000.0)
            per_window_rows.append({"method": name, "window_id": art.window_id, "inference_ms": latency[-1]})
        rows.append(
            {
                "method": name,
                "warmstart_train_sec": 0.0,
                "a3c_train_sec": 0.0,
                "total_train_sec": 0.0,
                "inference_ms_per_window_mean": float(np.mean(latency)),
                "inference_ms_per_window_std": float(np.std(latency)),
            }
        )

    # ----- RL-family adapters (have train()) -----
    rl_pairs = [
        ("StrAEmDDTrainer", StrAEmDDTrainer()),
        ("DRLNIDSTrainer", DRLNIDSTrainer()),
        ("DNNAEDDTrainer", DNNAEDDTrainer()),
        ("StreamingADBestTrainer", StreamingADBestTrainer()),
        ("InterpretableACTrainer", InterpretableACTrainer()),
    ]
    train_states_list = [a.state_dict for a in train_arts]
    for name, ad in rl_pairs:
        t0 = time.perf_counter()
        ad.train(train_states_list, num_episodes=episodes)
        train_sec = time.perf_counter() - t0
        latency = []
        for art in test_arts:
            t0 = time.perf_counter()
            ad.predict(art.state_dict)
            latency.append((time.perf_counter() - t0) * 1000.0)
            per_window_rows.append({"method": name, "window_id": art.window_id, "inference_ms": latency[-1]})
        rows.append(
            {
                "method": name,
                "warmstart_train_sec": 0.0,
                "a3c_train_sec": train_sec,
                "total_train_sec": train_sec,
                "inference_ms_per_window_mean": float(np.mean(latency)),
                "inference_ms_per_window_std": float(np.std(latency)),
            }
        )

    suffix = "_high" if high_heldout else ""
    pd.DataFrame(rows).to_csv(out_dir / f"runtime_by_method{suffix}.csv", index=False)
    pd.DataFrame(per_window_rows).to_csv(out_dir / f"runtime_per_window{suffix}.csv", index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--high-heldout", action="store_true")
    args = parser.parse_args()
    main(args.seed, args.episodes, high_heldout=args.high_heldout)
