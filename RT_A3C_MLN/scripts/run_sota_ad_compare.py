"""Run the SOTA deep anomaly-detector comparison (Section 6.5 of the paper).

This driver is intentionally unopinionated about how each competitor
is imported; the caller must register a list of :class:`DetectorAdapter`
objects (see :mod:`src.pipeline.sota_ad_compare`).  The reference
implementations for all twelve detectors are linked in the TODO above
Table 5 of ``paper/main.tex``.

Example::

    python scripts/run_sota_ad_compare.py \\
        --project-root . \\
        --train-csv data/processed/swat/train.csv \\
        --test-csv  data/processed/swat/test.csv \\
        --label-col attack
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.pipeline.sota_ad_compare import (
    evaluate_detector,
    run_sota_ad_compare,
    save_sota_ad_compare,
)
from src.pipeline.cross_window_experiment import CrossWindowSplitConfig, split_plan_by_attack_bin
from src.rl.ac_model import ActorCriticNet
from src.rl.simple_env import SimpleRuleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Deep anomaly-detector comparison (paper Section 6.5).')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--train-csv', type=Path, default=None)
    parser.add_argument('--test-csv', type=Path, default=None)
    parser.add_argument('--plan-csv', type=Path, default=None)
    parser.add_argument('--label-col', type=str, default='attack')
    parser.add_argument('--train-per-bin', type=int, default=6)
    parser.add_argument('--val-per-bin', type=int, default=2)
    parser.add_argument('--test-per-bin', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ours-run-name', type=str, default='a3c_v7')
    parser.add_argument('--hidden-dim', type=int, default=128)
    return parser.parse_args()


def _collect_adapters():
    """Return the list of adapters to evaluate.

    By default returns the twelve named 2023-2026 adapter stubs from
    :mod:`src.adapters.detector_family.adapters`.  Each stub
    currently returns all-zero predictions; replace each
    ``predict`` method with a wrapper around the authors' reference
    code (links inside each adapter class) before quoting the
    paper's table.
    """
    from src.adapters.detector_family.adapters import all_detector_family_adapters
    return list(all_detector_family_adapters())


def _ensure_attack_bin(plan_df: pd.DataFrame) -> pd.DataFrame:
    if 'attack_bin' not in plan_df.columns:
        plan_df = plan_df.copy()
        plan_df['attack_bin'] = plan_df['attack_ratio'].apply(
            lambda r: 'low' if r < 0.3 else ('high' if r >= 0.7 else 'mid')
        )
    return plan_df


def _frames_from_window_plan(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[int]]:
    if args.plan_csv is None:
        raise ValueError('Provide either --train-csv/--test-csv or --plan-csv.')

    plan_df = _ensure_attack_bin(pd.read_csv(args.plan_csv))
    splits = split_plan_by_attack_bin(
        plan_df,
        CrossWindowSplitConfig(
            train_per_bin=args.train_per_bin,
            val_per_bin=args.val_per_bin,
            test_per_bin=args.test_per_bin,
            random_state=args.seed,
        ),
    )
    train_ids = [int(x) for x in splits['train']]
    test_ids = [int(x) for x in splits['test']]

    X_all = pd.read_csv(args.project_root / 'data' / 'processed' / 'swat' / 'X_filled_binary.csv')
    y_df = pd.read_csv(args.project_root / 'data' / 'processed' / 'swat' / 'y_filled.csv')
    y_all = y_df['label'].astype(int).to_numpy() if 'label' in y_df.columns else y_df.iloc[:, 0].astype(int).to_numpy()

    def _concat(ids: list[int]) -> tuple[pd.DataFrame, np.ndarray]:
        frames = []
        labels = []
        for wid in ids:
            row = plan_df[plan_df['window_id'].astype(int) == int(wid)].iloc[0]
            start = int(row['start_idx'])
            end = int(row['end_idx'])
            frames.append(X_all.iloc[start:end].reset_index(drop=True))
            labels.append(y_all[start:end])
        return pd.concat(frames, axis=0, ignore_index=True), np.concatenate(labels, axis=0)

    train_df, _ = _concat(train_ids)
    test_df, y_test = _concat(test_ids)
    return train_df, test_df, y_test, test_ids


def _predict_ours(args: argparse.Namespace, test_ids: list[int]) -> tuple[np.ndarray, float]:
    model_path = args.project_root / 'outputs' / 'models' / f'{args.ours_run_name}_selected_policy.pth'
    if not model_path.exists():
        model_path = args.project_root / 'outputs' / 'models' / f'{args.ours_run_name}_global_policy.pth'
    if not model_path.exists():
        raise FileNotFoundError(f'Cannot find model for {args.ours_run_name}: {model_path}')

    model = ActorCriticNet(hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    preds = []
    kept_counts = []
    for wid in test_ids:
        state_path = (
            args.project_root
            / 'data'
            / 'stream'
            / 'swat'
            / 'window_rule_states'
            / f'window_{int(wid)}_rule_state.pkl'
        )
        with open(state_path, 'rb') as f:
            state_dict = pickle.load(f)

        env = SimpleRuleEnv(state_dict, max_steps=int(state_dict['num_rules']) + 1)
        obs = env.reset()
        actions = []
        done = False
        while not done and env.current_rule_idx < int(state_dict['num_rules']):
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(model(x)[0], dim=1).item())
            actions.append(action)
            obs, _, done, _ = env.step(action)

        kept_attack_rules = []
        kept_count = 0
        for rule, action, target in zip(state_dict['rule_pool'], actions, state_dict['target_labels']):
            if int(action) == 0:
                kept_count += 1
            if int(action) == 0 and int(target) == 1:
                antecedent = [
                    token.strip()
                    for token in str(rule.get('antecedent_str', '')).split(' & ')
                    if token.strip()
                ]
                kept_attack_rules.append(antecedent)
        kept_counts.append(kept_count)

        onehot_path = (
            args.project_root
            / 'data'
            / 'stream'
            / 'swat'
            / 'window_labeled_onehot'
            / f'window_{int(wid)}_labeled_onehot.csv'
        )
        onehot = pd.read_csv(onehot_path)
        flags = np.zeros(len(onehot), dtype=int)
        for antecedent in kept_attack_rules:
            if not antecedent:
                continue
            mask = np.ones(len(onehot), dtype=bool)
            for token in antecedent:
                if token not in onehot.columns:
                    mask[:] = False
                    break
                mask &= onehot[token].to_numpy(dtype=bool)
            flags[mask] = 1
        preds.append(flags)

    return np.concatenate(preds, axis=0), float(np.mean(kept_counts)) if kept_counts else 0.0


def main() -> None:
    args = parse_args()
    test_ids: list[int] = []
    if args.train_csv is not None and args.test_csv is not None:
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
        y_test = test_df[args.label_col].to_numpy(dtype=int)
        test_features = test_df.drop(columns=[args.label_col])
        train_features = train_df.drop(columns=[args.label_col], errors='ignore')
    else:
        train_features, test_features, y_test, test_ids = _frames_from_window_plan(args)

    adapters = _collect_adapters()
    if not adapters:
        print('[info] no adapters registered. Edit _collect_adapters() in this file '
              'to add the 12 deep anomaly detectors before reporting the table.')
        return

    summary = run_sota_ad_compare(
        adapters=adapters,
        train_df=train_features,
        test_df=test_features,
        y_test=y_test,
    )
    if test_ids:
        ours_pred, num_rules = _predict_ours(args, test_ids)
        ours_metrics = evaluate_detector(y_test, ours_pred)
        summary = pd.concat([
            summary,
            pd.DataFrame([{
                'method': 'RT-A3C-MLN (ours)',
                'year': 2026,
                **ours_metrics,
                'num_rules': num_rules,
            }]),
        ], ignore_index=True)
    out_path = save_sota_ad_compare(summary, args.project_root)
    print(summary)
    print(f'[saved] {out_path}')


if __name__ == '__main__':
    main()
