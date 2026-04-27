from __future__ import annotations

from typing import List

import pandas as pd


Transaction = List[str]


def build_transactions(df_binary: pd.DataFrame, lag: int = 2) -> List[Transaction]:
    """将二值特征表转换为带时间滞后的事务列表。

    只保留值为 1 的特征项，命名格式为 ``<feature>_high(t-k)``。

    参数
    ----
    df_binary:
        0/1 二值 DataFrame，每一列对应一个特征。
    lag:
        时间滞后阶数。``lag=2`` 表示每条事务由 ``t-2, t-1, t-0`` 三个时刻构成。

    返回
    ----
    List[List[str]]
        事务列表，长度为 ``len(df_binary) - lag``。
    """
    if lag < 0:
        raise ValueError(f'lag must be >= 0, got {lag}')
    if df_binary.empty:
        return []

    cols = list(df_binary.columns)
    transactions: List[Transaction] = []

    for t in range(lag, len(df_binary)):
        items: Transaction = []
        for h in range(lag, -1, -1):
            row = df_binary.iloc[t - h]
            for col in cols:
                if int(row[col]) == 1:
                    items.append(f'{col}_high(t-{h})')
        transactions.append(items)

    return transactions
