from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


BIN_LOW = "low"
BIN_MID = "mid"
BIN_HIGH = "high"


@dataclass(frozen=True)
class StratifiedSelectionConfig:
    # 统一正式实验阈值
    low_attack_max: float = 0.30
    high_attack_min: float = 0.70

    # 每组单独控制数量，更适合正式实验
    low_per_bin: int = 12
    mid_per_bin: int = 12
    high_per_bin: int = 12

    # 非重叠约束：至少相隔 2 个 window_id
    min_window_gap: int = 2
    random_state: int = 42


def assign_attack_bin(attack_ratio: float, config: StratifiedSelectionConfig) -> str:
    if float(attack_ratio) < config.low_attack_max:
        return BIN_LOW
    if float(attack_ratio) >= config.high_attack_min:
        return BIN_HIGH
    return BIN_MID


def load_window_table(project_root: Path, filename: str = "windows_mixed.csv") -> pd.DataFrame:
    path = project_root / "data" / "stream" / "swat" / filename
    df = pd.read_csv(path)
    required = {"window_id", "attack_ratio", "start_idx", "end_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必要列: {sorted(missing)}")
    return df.copy()


def _is_far_enough(window_id: int, selected_ids: list[int], min_gap: int) -> bool:
    return all(abs(int(window_id) - int(x)) >= min_gap for x in selected_ids)


def _select_with_gap(sub: pd.DataFrame, take: int, min_gap: int, random_state: int) -> pd.DataFrame:
    if sub.empty or take <= 0:
        return sub.iloc[0:0].copy()

    rng = np.random.default_rng(random_state)
    sub = sub.sort_values(["attack_ratio", "window_id"]).reset_index(drop=True)

    # 第一轮：随机顺序 + gap 约束
    order = rng.permutation(len(sub))
    chosen_positions: list[int] = []
    chosen_ids: list[int] = []

    for pos in order:
        win_id = int(sub.iloc[pos]["window_id"])
        if _is_far_enough(win_id, chosen_ids, min_gap):
            chosen_positions.append(int(pos))
            chosen_ids.append(win_id)
            if len(chosen_positions) >= take:
                break

    # 第二轮：如果第一轮不够，再按确定性顺序补齐
    if len(chosen_positions) < take:
        for pos in range(len(sub)):
            if pos in chosen_positions:
                continue
            win_id = int(sub.iloc[pos]["window_id"])
            if _is_far_enough(win_id, chosen_ids, min_gap):
                chosen_positions.append(int(pos))
                chosen_ids.append(win_id)
                if len(chosen_positions) >= take:
                    break

    return sub.iloc[sorted(chosen_positions)].copy().reset_index(drop=True)


def select_stratified_mixed_windows(
    windows_df: pd.DataFrame,
    config: StratifiedSelectionConfig | None = None,
) -> pd.DataFrame:
    config = config or StratifiedSelectionConfig()
    df = windows_df.copy()
    df["attack_bin"] = df["attack_ratio"].apply(lambda x: assign_attack_bin(float(x), config))

    target_map = {
        BIN_LOW: config.low_per_bin,
        BIN_MID: config.mid_per_bin,
        BIN_HIGH: config.high_per_bin,
    }

    parts = []
    stats = []

    for i, attack_bin in enumerate([BIN_LOW, BIN_MID, BIN_HIGH]):
        sub = df[df["attack_bin"] == attack_bin].copy()
        available = len(sub)
        target = target_map[attack_bin]
        take = min(target, available)

        if available < target:
            warnings.warn(
                f"{attack_bin} 组可用窗口不足：available={available}, target={target}，将只选择 {take} 个。",
                RuntimeWarning,
            )

        if available == 0:
            stats.append({
                "attack_bin": attack_bin,
                "available": 0,
                "target": target,
                "selected": 0,
            })
            continue

        picked = _select_with_gap(
            sub=sub,
            take=take,
            min_gap=config.min_window_gap,
            random_state=config.random_state + i,
        )
        parts.append(picked)

        stats.append({
            "attack_bin": attack_bin,
            "available": available,
            "target": target,
            "selected": len(picked),
        })

    if not parts:
        return pd.DataFrame(columns=df.columns.tolist())

    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sort_values(["attack_bin", "attack_ratio", "window_id"]).reset_index(drop=True)

    # 把选择统计挂到 attrs，后面 build_large_scale_plan 可直接打印/保存
    out.attrs["selection_stats"] = pd.DataFrame(stats)
    return out


def build_large_scale_plan(
    project_root: Path,
    out_name: str = "formal_mixed_windows_plan.csv",
    low_attack_max: float = 0.30,
    high_attack_min: float = 0.70,
    low_per_bin: int = 12,
    mid_per_bin: int = 12,
    high_per_bin: int = 12,
    min_window_gap: int = 2,
    random_state: int = 42,
    stats_name: str = "formal_mixed_windows_plan_stats.csv",
) -> pd.DataFrame:
    config = StratifiedSelectionConfig(
        low_attack_max=low_attack_max,
        high_attack_min=high_attack_min,
        low_per_bin=low_per_bin,
        mid_per_bin=mid_per_bin,
        high_per_bin=high_per_bin,
        min_window_gap=min_window_gap,
        random_state=random_state,
    )

    windows_df = load_window_table(project_root)
    selected = select_stratified_mixed_windows(windows_df, config=config)

    out_path = project_root / "data" / "stream" / "swat" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)

    stats_df = selected.attrs.get("selection_stats", pd.DataFrame())
    if not stats_df.empty:
        stats_path = project_root / "data" / "stream" / "swat" / stats_name
        stats_df.to_csv(stats_path, index=False)

    return selected