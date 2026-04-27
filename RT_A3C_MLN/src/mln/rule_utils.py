from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-6


def item_to_atom(item: str) -> str:
    """将事务项转换为更适合 MLN 展示的原子名字。"""
    return item.replace('(t-', '_t').replace(')', '')


def confidence_to_weight(confidence: float, eps: float = EPS) -> float:
    """将置信度映射为 logit 风格权重。"""
    clipped = float(np.clip(confidence, eps, 1 - eps))
    return float(np.log(clipped / (1 - clipped)))


def clipped_confidence_weight(confidence: float, wmin: float | None = None, wmax: float | None = None) -> float:
    """将置信度转换为权重，并按需截断。"""
    weight = confidence_to_weight(confidence)
    if wmin is not None:
        weight = max(wmin, weight)
    if wmax is not None:
        weight = min(wmax, weight)
    return float(weight)


def rules_to_mln_df(rules_df: pd.DataFrame) -> pd.DataFrame:
    """将关联规则表转换为 MLN 风格的规则表。

    需要输入列:
    - antecedent_str
    - consequent_str
    - support
    - confidence
    - lift
    """
    required = {'antecedent_str', 'consequent_str', 'support', 'confidence', 'lift'}
    missing = required - set(rules_df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    out = rules_df.copy()
    out['ante_atom'] = out['antecedent_str'].map(item_to_atom)
    out['cons_atom'] = out['consequent_str'].map(item_to_atom)
    out['formula'] = out['ante_atom'] + ' => ' + out['cons_atom']
    out['weight'] = out['confidence'].map(confidence_to_weight)

    return out[[
        'antecedent_str',
        'consequent_str',
        'formula',
        'support',
        'confidence',
        'lift',
        'weight',
    ]]
