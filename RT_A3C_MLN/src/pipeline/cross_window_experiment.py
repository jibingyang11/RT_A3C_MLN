from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.rl.ac_model import ActorCriticNet
from src.rl.simple_env import SimpleRuleEnv
from src.pipeline.formal_experiment import (
    EvalConfig,
    RuleMiningConfig,
    WarmStartConfig,
    build_clean_summary,
    build_group_stats,
    get_logits,
    prepare_window_learning_artifacts,
    save_dataframe,
)


@dataclass(frozen=True)
class CrossWindowSplitConfig:
    # 更正式的默认协议：优先保证 test 覆盖
    train_per_bin: int = 3
    val_per_bin: int = 1
    test_per_bin: int = 2
    random_state: int = 42


def split_plan_by_attack_bin(
    plan_df: pd.DataFrame,
    config: CrossWindowSplitConfig | None = None,
) -> dict[str, list[int]]:
    """
    更正式的 split 策略：
    1) 先保证 test
    2) 再保证 val
    3) 最后剩余给 train
    """
    config = config or CrossWindowSplitConfig()

    if "attack_bin" not in plan_df.columns:
        raise ValueError("plan_df 缺少 attack_bin 列")

    rng = np.random.default_rng(config.random_state)

    splits = {"train": [], "val": [], "test": []}
    split_rows = []

    for attack_bin in ["low", "mid", "high"]:
        sub = plan_df[plan_df["attack_bin"] == attack_bin].copy()
        if sub.empty:
            split_rows.append({
                "attack_bin": attack_bin,
                "available": 0,
                "train_n": 0,
                "val_n": 0,
                "test_n": 0,
                "train_ids": [],
                "val_ids": [],
                "test_ids": [],
            })
            continue

        ids = sub["window_id"].astype(int).tolist()
        rng.shuffle(ids)

        available = len(ids)

        # 先保证 test
        test_n = min(config.test_per_bin, available)
        test_ids = ids[:test_n]

        # 再保证 val
        remaining_after_test = ids[test_n:]
        val_n = min(config.val_per_bin, len(remaining_after_test))
        val_ids = remaining_after_test[:val_n]

        # 最后给 train
        remaining_after_val = remaining_after_test[val_n:]
        train_n = min(config.train_per_bin, len(remaining_after_val))
        train_ids = remaining_after_val[:train_n]

        splits["train"].extend(train_ids)
        splits["val"].extend(val_ids)
        splits["test"].extend(test_ids)

        split_rows.append({
            "attack_bin": attack_bin,
            "available": available,
            "train_n": len(train_ids),
            "val_n": len(val_ids),
            "test_n": len(test_ids),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        })

    splits["_detail_df"] = pd.DataFrame(split_rows)
    return splits


def train_global_warmstart(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: WarmStartConfig,
    save_path: Path,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    num_keep = int((y_train == 0).sum())
    num_disable = int((y_train == 1).sum())

    class_weights = torch.tensor(
        [1.0 / max(num_keep, 1), 1.0 / max(num_disable, 1)],
        dtype=torch.float32,
    ).to(device)

    model = ActorCriticNet(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        logits = get_logits(model(X_tensor))
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        acc = float((get_logits(model(X_tensor)).argmax(dim=1) == y_tensor).float().mean().item())

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return acc


def evaluate_global_model_on_state(
    state_dict: dict,
    model_path: Path,
    config: WarmStartConfig,
    eval_config: EvalConfig,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorCriticNet(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = SimpleRuleEnv(
        state_dict,
        step_size=eval_config.env_step_size,
        max_steps=eval_config.env_max_steps,
    )

    state = env.reset()
    total_reward = 0.0
    rows = []
    target_labels = np.asarray(state_dict["target_labels"])

    for rule_idx in range(int(state_dict["num_rules"])):
        x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = int(torch.argmax(get_logits(model(x)), dim=1).item())

        state, reward, _, info = env.step(rule_idx, action)
        total_reward += float(reward)

        target_label = int(target_labels[rule_idx])
        is_correct = int((target_label == 1 and action == 0) or (target_label == 0 and action == 1))
        rows.append({
            "action": action,
            "target_label": target_label,
            "is_correct": is_correct,
        })

    df = pd.DataFrame(rows)
    attack_mask = df["target_label"] == 1
    normal_mask = df["target_label"] == 0

    return {
        "attack_keep_rate": float((df.loc[attack_mask, "action"] == 0).mean()) if attack_mask.sum() > 0 else np.nan,
        "normal_disable_rate": float((df.loc[normal_mask, "action"] == 1).mean()) if normal_mask.sum() > 0 else np.nan,
        "selection_accuracy": float(df["is_correct"].mean()),
        "greedy_total_reward": float(total_reward),
    }


def run_cross_window_experiment(
    project_root: Path,
    plan_df: pd.DataFrame,
    split_config: CrossWindowSplitConfig | None = None,
    mining_config: RuleMiningConfig | None = None,
    warmstart_config: WarmStartConfig | None = None,
    eval_config: EvalConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, list[int]], float]:
    split_config = split_config or CrossWindowSplitConfig()
    mining_config = mining_config or RuleMiningConfig()
    warmstart_config = warmstart_config or WarmStartConfig()
    eval_config = eval_config or EvalConfig()

    splits = split_plan_by_attack_bin(plan_df, split_config)

    if "_detail_df" in splits:
        print("[split detail]")
        print(splits["_detail_df"])

    train_artifacts = []
    for wid in splits["train"]:
        print(f"[train] preparing window {wid} ...")
        art = prepare_window_learning_artifacts(
            project_root,
            wid,
            "windows_mixed.csv",
            mining_config,
        )
        train_artifacts.append(art)

    test_artifacts = []
    for wid in splits["test"]:
        print(f"[test] preparing window {wid} ...")
        art = prepare_window_learning_artifacts(
            project_root,
            wid,
            "windows_mixed.csv",
            mining_config,
        )
        test_artifacts.append(art)

    if len(train_artifacts) == 0:
        raise ValueError(f"train_artifacts 为空，请检查 splits={splits}")

    if len(test_artifacts) == 0:
        raise ValueError(f"test_artifacts 为空，请检查 splits={splits}")

    X_train = np.concatenate([a["X_ws"] for a in train_artifacts], axis=0)
    y_train = np.concatenate([a["y_ws"] for a in train_artifacts], axis=0)

    model_path = project_root / "outputs" / "models" / "global_cross_window_warmstart.pth"
    train_acc = train_global_warmstart(X_train, y_train, warmstart_config, model_path)

    rows = []
    for art in test_artifacts:
        metrics = evaluate_global_model_on_state(
            art["state_dict"],
            model_path,
            warmstart_config,
            eval_config,
        )
        rows.append({
            "window_id": int(art["window_id"]),
            "attack_ratio": float(art["attack_ratio"]),
            "window_mode": art["window_mode"],
            "num_rules": int(art["state_dict"]["num_rules"]),
            "num_attack_rules": int(np.sum(np.asarray(art["state_dict"]["target_labels"]) == 1)),
            "num_normal_rules": int(np.sum(np.asarray(art["state_dict"]["target_labels"]) == 0)),
            "global_train_accuracy": float(train_acc),
            **metrics,
        })

    summary_df = pd.DataFrame(rows)

    if summary_df.empty:
        raise ValueError(f"cross-window 测试结果为空，请检查 splits 是否合理：{splits}")

    if "attack_ratio" not in summary_df.columns:
        raise ValueError(
            f"summary_df 缺少 attack_ratio 列，当前列为：{summary_df.columns.tolist()}，splits={splits}"
        )

    summary_df = summary_df.sort_values("attack_ratio").reset_index(drop=True)

    save_dataframe(summary_df, project_root / "outputs" / "logs" / "cross_window_test_summary.csv")
    clean_df = build_clean_summary(summary_df)
    save_dataframe(clean_df, project_root / "outputs" / "logs" / "cross_window_test_summary_clean.csv")
    build_group_stats(clean_df).to_csv(project_root / "outputs" / "logs" / "cross_window_group_stats.csv")

    if "_detail_df" in splits:
        save_dataframe(
            splits["_detail_df"],
            project_root / "outputs" / "logs" / "cross_window_split_detail.csv",
        )

    return summary_df, splits, train_acc