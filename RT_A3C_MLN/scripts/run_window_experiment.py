from __future__ import annotations

"""窗口级实验的简化命令行入口。

这个脚本主要作为后续整理统一实验流程的起点，当前不强制替代 notebook。
"""

import argparse
from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.pipeline.window_experiment import (
    add_transaction_labels,
    build_warmstart_dataset,
    greedy_evaluate,
    mine_label_rules,
    one_hot_transactions,
    train_warmstart_model,
)
from src.data.transaction_utils import build_transactions
from src.rl.state_utils import build_initial_rule_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary-csv', type=Path, required=True, help='窗口级二值特征 CSV')
    parser.add_argument('--label-csv', type=Path, required=True, help='与 transactions 对齐的标签 CSV')
    parser.add_argument('--out-dir', type=Path, required=True, help='输出目录')
    parser.add_argument('--lag', type=int, default=2)
    parser.add_argument('--min-support', type=float, default=0.1)
    parser.add_argument('--min-confidence', type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df_binary = pd.read_csv(args.binary_csv)
    y = pd.read_csv(args.label_csv).iloc[:, 0].astype(int).tolist()

    transactions = build_transactions(df_binary, lag=args.lag)
    labeled_transactions = add_transaction_labels(transactions, y)
    df_onehot = one_hot_transactions(labeled_transactions)
    rules = mine_label_rules(df_onehot, min_support=args.min_support, min_confidence=args.min_confidence)

    rules.to_csv(args.out_dir / 'label_rules.csv', index=False)

    # 这里默认使用外部准备好的规则池；如果 rules 为空，直接退出
    if rules.empty:
        raise SystemExit('No label rules mined under current thresholds.')

    # 演示用：直接将 mined rules 当作规则池，需要用户按自己的策略进一步筛选和平衡
    rules['formula'] = rules['antecedent_str'] + ' => ' + rules['consequent_str']
    rules['weight'] = rules['confidence']
    rules['target_label'] = (rules['consequent_str'] == 'LABEL_ATTACK').astype(int)

    state_dict = build_initial_rule_state(rules)
    with open(args.out_dir / 'rule_state.pkl', 'wb') as f:
        pickle.dump(state_dict, f)

    X_ws, y_ws = build_warmstart_dataset(state_dict)
    warmstart = train_warmstart_model(X_ws, y_ws, args.out_dir / 'actor_warmstart.pth')
    trace_df, metrics_df = greedy_evaluate(warmstart.model_path, state_dict)

    trace_df.to_csv(args.out_dir / 'greedy_trace.csv', index=False)
    metrics_df.to_csv(args.out_dir / 'metrics.csv', index=False)
    print('warmstart_acc =', warmstart.train_accuracy)
    print(metrics_df)


if __name__ == '__main__':
    main()
