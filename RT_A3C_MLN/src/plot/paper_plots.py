"""Plotting utilities for the RT-A3C-MLN paper.

All figure-generation logic is concentrated here so that every chart
used in the paper can be rebuilt from a single command
(``python scripts/make_paper_figures.py``).  Figures are written to
``outputs/figures/`` and loaded by the LaTeX source in
``paper/main.tex``.

Figure index used by the paper
------------------------------
* ``fig_training_curve.pdf`` -- A3C training reward / selection
  accuracy curve produced from
  ``outputs/logs/a3c_run_training_curve.csv``.
* ``fig_attack_ratio_scatter.pdf`` -- per-window selection accuracy
  against attack ratio, produced from
  ``outputs/logs/formal_mixed_windows_summary_clean.csv``.
* ``fig_group_bar.pdf`` -- bar chart of the three attack-ratio
  groups (low/mid/high) across the four main metrics, produced from
  ``outputs/logs/formal_mixed_windows_group_stats.csv``.
* ``fig_baseline_compare.pdf`` -- method comparison (A3C, warm-start,
  all-keep, all-disable, random) on greedy total reward and
  selection accuracy, produced from
  ``outputs/logs/baseline_compare.csv``.
* ``fig_ablation_modes.pdf`` -- ablation on the adaptive rule-pool
  modes (balanced / standard / relaxed), produced from
  ``outputs/logs/ablation_modes.csv``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _setup_style() -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.dpi': 300,
        'figure.dpi': 120,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
    })


def plot_training_curve(curve_csv: Path, out_path: Path) -> None:
    """Plot episode reward / selection accuracy vs training step."""
    _setup_style()
    df = pd.read_csv(curve_csv)
    if 'total_reward' not in df.columns and 'reward' in df.columns:
        df = df.rename(columns={'reward': 'total_reward'})
    if 'selection_accuracy' not in df.columns and 'acc' in df.columns:
        df = df.rename(columns={'acc': 'selection_accuracy'})
    # aggregate across workers within each episode
    agg = df.groupby('episode').agg(
        total_reward_mean=('total_reward', 'mean'),
        total_reward_std=('total_reward', 'std'),
        selection_accuracy_mean=('selection_accuracy', 'mean'),
        selection_accuracy_std=('selection_accuracy', 'std'),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].plot(agg['episode'], agg['total_reward_mean'], color='#1f77b4', lw=1.8, label='Mean reward')
    axes[0].fill_between(
        agg['episode'],
        agg['total_reward_mean'] - agg['total_reward_std'].fillna(0.0),
        agg['total_reward_mean'] + agg['total_reward_std'].fillna(0.0),
        color='#1f77b4', alpha=0.18,
    )
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Greedy total reward')
    axes[0].set_title('A3C training reward')

    axes[1].plot(agg['episode'], agg['selection_accuracy_mean'], color='#d62728', lw=1.8, label='Selection accuracy')
    axes[1].fill_between(
        agg['episode'],
        agg['selection_accuracy_mean'] - agg['selection_accuracy_std'].fillna(0.0),
        agg['selection_accuracy_mean'] + agg['selection_accuracy_std'].fillna(0.0),
        color='#d62728', alpha=0.18,
    )
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Selection accuracy')
    axes[1].set_ylim(0.4, 1.02)
    axes[1].set_title('A3C selection accuracy')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_attack_ratio_scatter(summary_csv: Path, out_path: Path) -> None:
    """Per-window selection accuracy against attack ratio."""
    _setup_style()
    df = pd.read_csv(summary_csv)
    colors = {'low': '#2ca02c', 'mid': '#ff7f0e', 'high': '#d62728'}
    markers = {'low': 'o', 'mid': 's', 'high': '^'}

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.6))
    for bin_name in ['low', 'mid', 'high']:
        sub = df[df['attack_bin'] == bin_name]
        if sub.empty:
            continue
        ax.scatter(
            sub['attack_ratio'], sub['selection_accuracy'],
            c=colors[bin_name], marker=markers[bin_name], s=55, edgecolor='white', lw=0.6,
            label=f'{bin_name} (n={len(sub)})',
        )
    ax.axhline(0.9, color='gray', lw=0.8, linestyle='--', alpha=0.7)
    ax.set_xlabel('Window attack ratio')
    ax.set_ylabel('Selection accuracy')
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(0.0, 1.0)
    ax.set_title('Per-window selection accuracy on formal mixed windows')
    ax.legend(loc='lower left', frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_group_bar(clean_csv: Path, out_path: Path) -> None:
    """Bar chart of low/mid/high group means for the four main metrics."""
    _setup_style()
    df = pd.read_csv(clean_csv)
    metrics = [
        ('attack_keep_rate', 'Attack keep rate'),
        ('normal_disable_rate', 'Normal disable rate'),
        ('selection_accuracy', 'Selection accuracy'),
        ('greedy_total_reward', 'Greedy total reward (normalised)'),
    ]

    grouped = df.groupby('attack_bin').agg({m[0]: ['mean', 'std', 'count'] for m in metrics})
    bin_order = [b for b in ['low', 'mid', 'high'] if b in grouped.index]

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))
    x = np.arange(len(bin_order))
    palette = ['#2ca02c', '#ff7f0e', '#d62728']

    for ax, (key, name) in zip(axes, metrics):
        means = np.array([grouped.loc[b, (key, 'mean')] for b in bin_order])
        stds = np.array([grouped.loc[b, (key, 'std')] for b in bin_order])
        stds = np.nan_to_num(stds, nan=0.0)
        if key == 'greedy_total_reward' and np.abs(means).max() > 0:
            norm = np.abs(means).max()
            means = means / norm
            stds = stds / norm
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=palette[:len(bin_order)], edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(bin_order)
        ax.set_title(name)
        if key != 'greedy_total_reward':
            ax.set_ylim(0.0, 1.05)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{m:.2f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Formal mixed-window results grouped by attack ratio', y=1.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_baseline_compare(baseline_csv: Path, out_path: Path) -> None:
    """Bar chart comparing A3C/warm-start against trivial baselines."""
    _setup_style()
    df = pd.read_csv(baseline_csv)
    methods = df['method'].tolist()
    selection = df['selection_accuracy'].to_numpy()
    reward = df['greedy_total_reward'].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.3))
    palette = ['#1f77b4', '#2ca02c', '#ff7f0e', '#8c564b', '#9467bd', '#d62728', '#7f7f7f']
    x = np.arange(len(methods))

    axes[0].bar(x, selection, color=palette[:len(methods)], edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha='right')
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel('Selection accuracy')
    axes[0].set_title('Selection accuracy')
    for xi, v in zip(x, selection):
        axes[0].text(xi, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    axes[1].bar(x, reward, color=palette[:len(methods)], edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha='right')
    axes[1].set_ylabel('Greedy total reward')
    axes[1].set_title('Total reward')
    for xi, v in zip(x, reward):
        axes[1].text(xi, v + 0.03 * (1 + abs(v)), f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_ablation_modes(ablation_csv: Path, out_path: Path) -> None:
    """Grouped-bar chart for the rule-pool mode ablation."""
    _setup_style()
    df = pd.read_csv(ablation_csv)

    bins = ['low', 'mid', 'high']
    configs = df['config'].drop_duplicates().tolist()
    metric = 'selection_accuracy'

    x = np.arange(len(bins))
    width = 0.8 / max(1, len(configs))
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.4))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, cfg in enumerate(configs):
        sub = df[df['config'] == cfg]
        vals = [float(sub[sub['attack_bin'] == b][metric].mean()) if not sub[sub['attack_bin'] == b].empty else 0.0 for b in bins]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=cfg, color=palette[i % len(palette)], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel('Selection accuracy')
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Ablation: adaptive rule-pool modes')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_sota_ad_compare(ad_csv: Path, out_path: Path) -> None:
    """Figure 7: F1 and interpretability of deep detectors vs. RT-A3C-MLN.

    Expects a CSV with columns ``method, f1`` (required) and optionally
    ``num_rules`` for the interpretability bar.
    """
    _setup_style()
    df = pd.read_csv(ad_csv)
    if 'num_rules' not in df.columns:
        df['num_rules'] = 0
        df.loc[df['method'].str.contains('RT-A3C-MLN', case=False, na=False), 'num_rules'] = 13

    df = df.sort_values('f1', ascending=True, na_position='first').reset_index(drop=True)
    ours_mask = df['method'].str.contains('RT-A3C-MLN', case=False, na=False)
    bar_colors = ['#d62728' if o else '#1f77b4' for o in ours_mask]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, max(3.2, 0.35 * len(df))))

    axes[0].barh(df['method'], df['f1'].fillna(0.0), color=bar_colors, edgecolor='white')
    axes[0].set_xlabel('F1')
    axes[0].set_xlim(0.0, 1.05)
    axes[0].set_title('Detection F1 on SWaT')
    for i, v in enumerate(df['f1'].fillna(0.0)):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    axes[1].barh(df['method'], df['num_rules'].fillna(0.0), color=bar_colors, edgecolor='white')
    axes[1].set_xlabel('# human-readable rules per decision')
    axes[1].set_title('Interpretability')
    for i, v in enumerate(df['num_rules'].fillna(0.0)):
        axes[1].text(v + max(0.3, 0.02 * df['num_rules'].max()), i, f'{int(v)}', va='center', fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_sota_rl_compare(summary_csv: Path, curves_csv: Path, out_path: Path) -> None:
    """Figure 8: training-efficiency comparison of RL baselines.

    Left panel: per-episode selection accuracy curve for each method.
    Right panel: scatter of wall-time vs. final selection accuracy.
    """
    _setup_style()
    summary = pd.read_csv(summary_csv)
    curves = pd.read_csv(curves_csv) if Path(curves_csv).exists() else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))

    palette = plt.get_cmap('tab10')
    if not curves.empty:
        methods = curves['method'].drop_duplicates().tolist()
        for i, m in enumerate(methods):
            sub = curves[curves['method'] == m]
            agg = sub.groupby('episode')['selection_accuracy'].mean().reset_index()
            ours = 'RT-A3C-MLN' in m
            lw = 2.4 if ours else 1.4
            color = '#d62728' if ours else palette(i)
            axes[0].plot(agg['episode'], agg['selection_accuracy'], label=m, color=color, lw=lw)
        axes[0].legend(fontsize=8, loc='lower right', frameon=False)
        axes[0].set_ylim(0.4, 1.02)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Selection accuracy')
    axes[0].set_title('Training curves')

    if not summary.empty:
        times = summary['wall_time_sec'].fillna(0.0) / 60.0
        accs = summary['selection_accuracy'].fillna(0.0)
        ours_mask = summary['method'].str.contains('RT-A3C-MLN', case=False, na=False)
        colors = ['#d62728' if o else '#1f77b4' for o in ours_mask]
        axes[1].scatter(times, accs, c=colors, s=90, edgecolor='white')
        for _, row in summary.iterrows():
            axes[1].annotate(row['method'],
                             (row['wall_time_sec'] / 60.0, row['selection_accuracy']),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1].set_xlabel('Wall-clock training time (min)')
    axes[1].set_ylabel('Final selection accuracy')
    axes[1].set_ylim(0.4, 1.02)
    axes[1].set_title('Training efficiency')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_all(project_root: Path) -> list[Path]:
    """Build every figure referenced by the paper, skipping missing inputs."""
    figures = project_root / 'outputs' / 'figures'
    logs = project_root / 'outputs' / 'logs'
    produced: list[Path] = []

    training_curve = logs / 'a3c_v7_training_curve.csv'
    if not training_curve.exists():
        training_curve = logs / 'a3c_run_training_curve.csv'

    tasks = [
        (training_curve, figures / 'fig_training_curve.pdf', plot_training_curve),
        (logs / 'formal_mixed_windows_summary_clean.csv', figures / 'fig_attack_ratio_scatter.pdf', plot_attack_ratio_scatter),
        (logs / 'formal_mixed_windows_summary_clean.csv', figures / 'fig_group_bar.pdf', plot_group_bar),
        (logs / 'baseline_compare.csv', figures / 'fig_baseline_compare.pdf', plot_baseline_compare),
        (logs / 'ablation_modes.csv', figures / 'fig_ablation_modes.pdf', plot_ablation_modes),
        (logs / 'sota_ad_compare.csv', figures / 'fig_sota_ad_compare.pdf', plot_sota_ad_compare),
    ]

    for src, dst, fn in tasks:
        if src.exists():
            fn(src, dst)
            produced.append(dst)
        else:
            print(f'[skip] {src} not found; figure {dst.name} not generated.')

    # RL comparison figure needs two inputs
    rl_summary = logs / 'sota_rl_compare.csv'
    rl_curves = logs / 'sota_rl_compare_curves.csv'
    rl_out = figures / 'fig_sota_rl_compare.pdf'
    if rl_summary.exists():
        plot_sota_rl_compare(rl_summary, rl_curves, rl_out)
        produced.append(rl_out)
    else:
        print(f'[skip] {rl_summary} not found; figure {rl_out.name} not generated.')

    return produced
