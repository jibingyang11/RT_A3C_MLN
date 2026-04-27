from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from src.data.transaction_utils import build_transactions
from src.rl.ac_model import ActorCriticNet
from src.rl.simple_env import SimpleRuleEnv
from src.rl.state_utils import build_initial_rule_state


LABEL_ATTACK = 'LABEL_ATTACK'
LABEL_NORMAL = 'LABEL_NORMAL'


@dataclass
class WarmStartResult:
    model_path: Path
    train_accuracy: float
    predictions: np.ndarray


@dataclass(frozen=True)
class RuleMiningConfig:
    lag: int = 2
    min_support: float = 0.1
    min_confidence: float = 0.6
    top_attack_rules: int = 10
    top_normal_rules: int = 10
    relaxed_normal_rules: int = 3
    relaxed_normal_min_weight: float = 0.3
    relaxed_normal_weight_scale: float = 30.0
    relaxed_normal_weight_cap: float = 3.0
    attack_weight_cap: float = 3.0
    high_attack_ratio_threshold: float = 0.75
    low_attack_ratio_threshold: float = 0.25


@dataclass(frozen=True)
class WarmStartConfig:
    epochs: int = 200
    lr: float = 1e-3
    hidden_dim: int = 64
    state_dim: int = 12
    action_dim: int = 2
    seed: int = 42


@dataclass(frozen=True)
class EvalConfig:
    env_step_size: float = 0.5
    env_max_steps: int = 100
    random_trials: int = 20
    seed: int = 42


@dataclass
class WindowExperimentArtifacts:
    window_id: int
    attack_ratio: float
    transactions_path: Path
    labeled_transactions_path: Path
    onehot_path: Path
    freq_items_path: Path
    label_rules_path: Path
    rule_pool_path: Path
    rule_state_path: Path
    model_path: Path
    trace_path: Path
    metrics_path: Path


@dataclass
class WindowExperimentResult:
    window_id: int
    attack_ratio: float
    num_rules: int
    num_attack_rules: int
    num_normal_rules: int
    attack_keep_rate: float
    normal_disable_rate: float
    selection_accuracy: float
    greedy_total_reward: float
    warmstart_train_accuracy: float
    window_mode: str
    all_keep_reward: float
    all_disable_reward: float
    random_mean_reward: float
    random_std_reward: float
    trace_df: pd.DataFrame
    metrics_df: pd.DataFrame
    artifacts: WindowExperimentArtifacts


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_transaction_labels(transactions: list[list[str]], labels: Iterable[int]) -> list[list[str]]:
    labeled = []
    for tx, y in zip(transactions, labels):
        token = LABEL_ATTACK if int(y) == 1 else LABEL_NORMAL
        labeled.append(list(tx) + [token])
    return labeled


def one_hot_transactions(labeled_transactions: list[list[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    onehot = te.fit(labeled_transactions).transform(labeled_transactions)
    return pd.DataFrame(onehot, columns=te.columns_)


def mine_frequent_itemsets(df_onehot: pd.DataFrame, min_support: float = 0.1) -> pd.DataFrame:
    freq_items = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=2)
    if freq_items.empty:
        return pd.DataFrame(columns=['support', 'itemsets', 'itemset_len'])
    freq_items = freq_items.sort_values(['support'], ascending=False).reset_index(drop=True)
    freq_items['itemset_len'] = freq_items['itemsets'].apply(len)
    return freq_items


def mine_label_rules(df_onehot: pd.DataFrame, min_support: float = 0.1, min_confidence: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq_items = mine_frequent_itemsets(df_onehot, min_support=min_support)
    if freq_items.empty:
        empty = pd.DataFrame(columns=['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift'])
        return freq_items, empty

    rules = association_rules(freq_items, metric='confidence', min_threshold=min_confidence).copy()
    if rules.empty:
        empty = pd.DataFrame(columns=['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift'])
        return freq_items, empty

    rules = rules[
        (rules['consequents'].apply(lambda x: len(x) == 1))
        & (rules['consequents'].apply(lambda x: list(x)[0] in [LABEL_ATTACK, LABEL_NORMAL]))
    ].copy()
    if rules.empty:
        empty = pd.DataFrame(columns=['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift'])
        return freq_items, empty

    rules['antecedent_str'] = rules['antecedents'].apply(lambda x: ' & '.join(sorted(list(x))))
    rules['consequent_str'] = rules['consequents'].apply(lambda x: list(x)[0])
    out = rules[['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']].copy()
    out = out.sort_values(
        ['consequent_str', 'confidence', 'lift', 'support'],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return freq_items, out


def mine_relaxed_normal_rules(freq_items: pd.DataFrame, p_normal: float) -> pd.DataFrame:
    if freq_items.empty:
        return pd.DataFrame(columns=['antecedent_str', 'support', 'confidence', 'lift', 'prior_normal', 'conf_gain', 'score_relaxed'])
    rules_all = association_rules(freq_items, metric='confidence', min_threshold=0.0).copy()
    if rules_all.empty:
        return pd.DataFrame(columns=['antecedent_str', 'support', 'confidence', 'lift', 'prior_normal', 'conf_gain', 'score_relaxed'])
    rules_all = rules_all[
        (rules_all['consequents'].apply(lambda x: len(x) == 1))
        & (rules_all['consequents'].apply(lambda x: list(x)[0] == LABEL_NORMAL))
    ].copy()
    if rules_all.empty:
        return pd.DataFrame(columns=['antecedent_str', 'support', 'confidence', 'lift', 'prior_normal', 'conf_gain', 'score_relaxed'])
    rules_all['antecedent_str'] = rules_all['antecedents'].apply(lambda x: ' & '.join(sorted(list(x))))
    rules_all['prior_normal'] = p_normal
    rules_all['conf_gain'] = rules_all['confidence'] - p_normal
    rules_all['score_relaxed'] = rules_all['conf_gain'] + 0.1 * (rules_all['lift'] - 1.0)
    out = rules_all[['antecedent_str', 'support', 'confidence', 'lift', 'prior_normal', 'conf_gain', 'score_relaxed']].copy()
    out = out.sort_values(['score_relaxed', 'confidence', 'lift', 'support'], ascending=[False, False, False, False]).reset_index(drop=True)
    return out


def confidence_to_weight(confidence: float, eps: float = 1e-6) -> float:
    clipped = float(np.clip(confidence, eps, 1 - eps))
    return float(np.log(clipped / (1 - clipped)))


def clipped_weight(confidence: float, wmax: float = 3.0) -> float:
    return float(np.clip(confidence_to_weight(confidence), 0.0, wmax))


def relaxed_normal_weight(score_relaxed: float, wmin: float = 0.3, scale: float = 30.0, wmax: float = 3.0) -> float:
    return float(np.clip(scale * float(score_relaxed), wmin, wmax))


def decide_window_mode(attack_ratio: float, config: RuleMiningConfig) -> str:
    if attack_ratio >= config.high_attack_ratio_threshold:
        return 'relaxed'
    if attack_ratio <= config.low_attack_ratio_threshold:
        return 'balanced'
    return 'standard'


def build_rule_pool(
    label_rules: pd.DataFrame,
    attack_ratio: float,
    config: RuleMiningConfig,
    relaxed_normal_df: pd.DataFrame | None = None,
    fallback_rules_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    label_rules:
        常规阈值下得到的标签规则表，至少包含：
        antecedent_str, consequent_str, support, confidence, lift

    fallback_rules_df:
        用于稀疏窗口 fallback 的候选规则表。
        建议传入更宽松阈值下得到的规则表（例如 confidence>=0 或 rules_all 处理后的结果），
        列格式应与 label_rules 一致。
    """

    def _rank_relaxed_candidates(
        df: pd.DataFrame,
        target_label: str,
        prior_prob: float,
        top_k: int,
    ) -> pd.DataFrame:
        cand = df[df["consequent_str"] == target_label].copy()
        if len(cand) == 0:
            return cand

        cand["prior_prob"] = float(prior_prob)
        cand["conf_gain"] = cand["confidence"] - cand["prior_prob"]
        cand["score_relaxed"] = cand["conf_gain"] + 0.1 * (cand["lift"] - 1.0)

        cand = cand.sort_values(
            ["score_relaxed", "confidence", "lift", "support"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)

        return cand.head(top_k).copy()

    attack_rules = label_rules[label_rules["consequent_str"] == LABEL_ATTACK].reset_index(drop=True)
    normal_rules = label_rules[label_rules["consequent_str"] == LABEL_NORMAL].reset_index(drop=True)

    mode = decide_window_mode(attack_ratio, config)

    # fallback 候选来源：优先用更宽松阈值生成的规则表；没有就退化用当前 label_rules
    candidate_rules = fallback_rules_df if fallback_rules_df is not None else label_rules

    # ===== 1) attack pool =====
    attack_top_k = min(config.top_attack_rules, len(attack_rules))
    if mode == "balanced":
        attack_top_k = min(attack_top_k, max(1, len(attack_rules)))

    attack_top = attack_rules.head(attack_top_k).copy()
    attack_top["formula"] = attack_top["antecedent_str"] + " => " + LABEL_ATTACK
    attack_top["weight"] = attack_top["confidence"].apply(
        lambda x: clipped_weight(x, wmax=config.attack_weight_cap)
    )
    attack_top["target_label"] = 1

    # ===== 2) normal pool =====
    if mode == "relaxed":
        if relaxed_normal_df is None or relaxed_normal_df.empty:
            # relaxed 模式下，normal 候选也允许 fallback
            relaxed_normal_df = _rank_relaxed_candidates(
                candidate_rules,
                target_label=LABEL_NORMAL,
                prior_prob=float(1.0 - attack_ratio),
                top_k=config.relaxed_normal_rules,
            )

        if relaxed_normal_df is None or relaxed_normal_df.empty:
            raise ValueError("relaxed 模式下 relaxed_normal_df 不能为空，且 fallback 后仍为空")

        normal_top = relaxed_normal_df.head(config.relaxed_normal_rules).copy()
        normal_top["consequent_str"] = LABEL_NORMAL
        normal_top["formula"] = normal_top["antecedent_str"] + " => " + LABEL_NORMAL
        normal_top["weight"] = normal_top["score_relaxed"].apply(
            lambda x: relaxed_normal_weight(
                x,
                wmin=config.relaxed_normal_min_weight,
                scale=config.relaxed_normal_weight_scale,
                wmax=config.relaxed_normal_weight_cap,
            )
        )
        normal_top["target_label"] = 0

    else:
        normal_top_k = config.top_normal_rules
        if mode == "balanced":
            normal_top_k = min(config.top_normal_rules, max(1, attack_top_k), len(normal_rules))

        normal_top = normal_rules.head(normal_top_k).copy()
        if len(normal_top) > 0:
            normal_top["formula"] = normal_top["antecedent_str"] + " => " + LABEL_NORMAL
            normal_top["weight"] = normal_top["confidence"].apply(
                lambda x: clipped_weight(x, wmax=config.attack_weight_cap)
            )
            normal_top["target_label"] = 0

    # ===== 3) balanced 稀疏窗口 fallback =====
    # ===== 3) 稀疏窗口 fallback（所有模式都可用）=====
# 先做 balanced 的目标数量约束
    if mode == "balanced":
        need_attack = min(3, config.top_attack_rules)
        need_normal = min(3, config.top_normal_rules)
    else:
        need_attack = max(1, min(config.top_attack_rules, 3 if mode == "standard" else config.top_attack_rules))
        need_normal = max(1, min(config.top_normal_rules, 3 if mode == "standard" else config.top_normal_rules))
    
    # attack 为空或太少：从 candidate_rules 里 relaxed 补
    if len(attack_top) < need_attack:
        relaxed_attack_top = _rank_relaxed_candidates(
            candidate_rules,
            target_label=LABEL_ATTACK,
            prior_prob=float(attack_ratio),
            top_k=need_attack if mode == "balanced" else config.top_attack_rules,
        )
        if len(relaxed_attack_top) > 0:
            relaxed_attack_top["formula"] = relaxed_attack_top["antecedent_str"] + " => " + LABEL_ATTACK
            relaxed_attack_top["weight"] = relaxed_attack_top["score_relaxed"].apply(
                lambda x: relaxed_normal_weight(
                    x,
                    wmin=config.relaxed_normal_min_weight,
                    scale=config.relaxed_normal_weight_scale,
                    wmax=config.attack_weight_cap,
                )
            )
            relaxed_attack_top["target_label"] = 1
            attack_top = relaxed_attack_top.copy()
    
    # normal 为空或太少：从 candidate_rules 里 relaxed 补
    if len(normal_top) < need_normal:
        relaxed_normal_top = _rank_relaxed_candidates(
            candidate_rules,
            target_label=LABEL_NORMAL,
            prior_prob=float(1.0 - attack_ratio),
            top_k=need_normal if mode == "balanced" else (
                config.relaxed_normal_rules if mode == "relaxed" else config.top_normal_rules
            ),
        )
        if len(relaxed_normal_top) > 0:
            relaxed_normal_top["formula"] = relaxed_normal_top["antecedent_str"] + " => " + LABEL_NORMAL
            relaxed_normal_top["weight"] = relaxed_normal_top["score_relaxed"].apply(
                lambda x: relaxed_normal_weight(
                    x,
                    wmin=config.relaxed_normal_min_weight,
                    scale=config.relaxed_normal_weight_scale,
                    wmax=config.relaxed_normal_weight_cap,
                )
            )
            relaxed_normal_top["target_label"] = 0
            normal_top = relaxed_normal_top.copy()
    
    # ===== 4) 最终检查 =====
    if attack_top.empty or normal_top.empty:
        raise ValueError(
            f"规则池不完整: attack={len(attack_top)}, normal={len(normal_top)}, mode={mode}"
        )

    mixed_rule_pool = pd.concat([
        attack_top[[
            "antecedent_str", "consequent_str", "formula",
            "support", "confidence", "lift", "weight", "target_label"
        ]],
        normal_top[[
            "antecedent_str", "consequent_str", "formula",
            "support", "confidence", "lift", "weight", "target_label"
        ]],
    ], axis=0, ignore_index=True)

    return mixed_rule_pool, mode


def build_warmstart_dataset(state_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    active_mask = np.asarray(state_dict['active_mask'], dtype=np.float32)
    weights = np.asarray(state_dict['weights'], dtype=np.float32)
    rule_scores = np.asarray(state_dict['rule_scores'], dtype=np.float32)
    target_labels = np.asarray(state_dict['target_labels'], dtype=np.int64)
    num_rules = int(state_dict['num_rules'])

    def build_ws_state(rule_idx: int) -> np.ndarray:
        active_weights = weights[active_mask > 0.5]
        if len(active_weights) == 0:
            mean_w = std_w = min_w = max_w = 0.0
        else:
            mean_w = float(active_weights.mean())
            std_w = float(active_weights.std())
            min_w = float(active_weights.min())
            max_w = float(active_weights.max())

        return np.array([
            float(active_mask.mean()),
            mean_w,
            std_w,
            min_w,
            max_w,
            float(active_mask.sum()),
            float(num_rules),
            float(rule_idx / max(1, num_rules - 1)),
            float(active_mask[rule_idx]),
            float(weights[rule_idx]),
            float(rule_scores[rule_idx]),
            float(target_labels[rule_idx]),
        ], dtype=np.float32)

    X_ws = np.stack([build_ws_state(i) for i in range(num_rules)], axis=0)
    y_ws = np.array([0 if t == 1 else 1 for t in target_labels], dtype=np.int64)
    return X_ws, y_ws


class _WeightedCrossEntropyWarmStart:
    def __init__(self, config: WarmStartConfig) -> None:
        self.config = config

    def fit(self, X_ws: np.ndarray, y_ws: np.ndarray, save_path: Path) -> WarmStartResult:
        set_global_seed(self.config.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.tensor(X_ws, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_ws, dtype=torch.long).to(device)

        num_keep = int((y_ws == 0).sum())
        num_disable = int((y_ws == 1).sum())
        class_weights = torch.tensor([
            1.0 / max(num_keep, 1),
            1.0 / max(num_disable, 1),
        ], dtype=torch.float32).to(device)

        model = ActorCriticNet(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        for _ in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(X_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_tensor)[0].argmax(dim=1)
            acc = float((pred == y_tensor).float().mean().item())

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        return WarmStartResult(save_path, acc, pred.detach().cpu().numpy())


def train_warmstart_model(
    X_ws: np.ndarray,
    y_ws: np.ndarray,
    save_path: Path,
    config: WarmStartConfig | None = None,
) -> WarmStartResult:
    config = config or WarmStartConfig()
    trainer = _WeightedCrossEntropyWarmStart(config)
    return trainer.fit(X_ws, y_ws, save_path)


def _rollout_actions(state_dict: dict, actions: Sequence[int], eval_config: EvalConfig) -> tuple[pd.DataFrame, dict]:
    env = SimpleRuleEnv(state_dict, step_size=eval_config.env_step_size, max_steps=eval_config.env_max_steps)
    state = env.reset()
    records = []
    total_reward = 0.0
    target_labels = np.asarray(state_dict['target_labels'])

    for rule_idx, action in enumerate(actions):
        next_state, reward, _, info = env.step(rule_idx, int(action))
        total_reward += float(reward)
        target_label = int(target_labels[rule_idx])
        is_correct = int((target_label == 1 and int(action) == 0) or (target_label == 0 and int(action) == 1))
        records.append({
            'rule_idx': rule_idx,
            'action': int(action),
            'reward': float(reward),
            'target_label': target_label,
            'is_correct': is_correct,
            'weight': float(info['weight']),
            'rule_score': float(info['rule_score']),
            'active_after': int(info['active']),
        })
        state = next_state

    trace_df = pd.DataFrame(records)
    attack_mask = trace_df['target_label'] == 1
    normal_mask = trace_df['target_label'] == 0
    metrics = {
        'num_rules': int(state_dict['num_rules']),
        'num_attack_rules': int(attack_mask.sum()),
        'num_normal_rules': int(normal_mask.sum()),
        'attack_keep_rate': float((trace_df.loc[attack_mask, 'action'] == 0).mean()) if attack_mask.sum() > 0 else np.nan,
        'normal_disable_rate': float((trace_df.loc[normal_mask, 'action'] == 1).mean()) if normal_mask.sum() > 0 else np.nan,
        'selection_accuracy': float(trace_df['is_correct'].mean()),
        'greedy_total_reward': float(total_reward),
    }
    return trace_df, metrics


def greedy_evaluate(model_path: Path, state_dict: dict, eval_config: EvalConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_config = eval_config or EvalConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorCriticNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = SimpleRuleEnv(state_dict, step_size=eval_config.env_step_size, max_steps=eval_config.env_max_steps)
    state = env.reset()
    actions: list[int] = []
    for _ in range(int(state_dict['num_rules'])):
        x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x)
            action = int(torch.argmax(logits, dim=1).item())
        actions.append(action)
        state, _, _, _ = env.step(len(actions) - 1, action)

    trace_df, metrics = _rollout_actions(state_dict, actions, eval_config)
    return trace_df, pd.DataFrame([metrics])


def evaluate_baselines(state_dict: dict, eval_config: EvalConfig | None = None) -> dict[str, float]:
    eval_config = eval_config or EvalConfig()
    set_global_seed(eval_config.seed)
    num_rules = int(state_dict['num_rules'])

    _, keep_metrics = _rollout_actions(state_dict, [0] * num_rules, eval_config)
    _, disable_metrics = _rollout_actions(state_dict, [1] * num_rules, eval_config)

    random_rewards = []
    rng = np.random.default_rng(eval_config.seed)
    for _ in range(eval_config.random_trials):
        actions = rng.integers(low=0, high=2, size=num_rules).tolist()
        _, metrics = _rollout_actions(state_dict, actions, eval_config)
        random_rewards.append(metrics['greedy_total_reward'])

    return {
        'all_keep_reward': float(keep_metrics['greedy_total_reward']),
        'all_disable_reward': float(disable_metrics['greedy_total_reward']),
        'random_mean_reward': float(np.mean(random_rewards)) if random_rewards else np.nan,
        'random_std_reward': float(np.std(random_rewards)) if random_rewards else np.nan,
    }


def load_window_binary_and_labels(project_root: Path, window_id: int, window_source: str = 'windows_mixed.csv') -> tuple[pd.DataFrame, pd.Series]:
    binary_path = project_root / 'data' / 'stream' / 'swat' / 'window_binary_data' / f'window_{window_id}_binary.csv'
    if not binary_path.exists():
        raise FileNotFoundError(binary_path)
    df_binary = pd.read_csv(binary_path)

    windows_df = pd.read_csv(project_root / 'data' / 'stream' / 'swat' / window_source)
    row = windows_df[windows_df['window_id'] == window_id]
    if row.empty:
        raise ValueError(f'window_id={window_id} 不在 {window_source} 中')
    row = row.iloc[0]
    start_idx = int(row['start_idx'])
    end_idx = int(row['end_idx'])

    y_df = pd.read_csv(project_root / 'data' / 'processed' / 'swat' / 'y_filled.csv')
    if 'label' in y_df.columns:
        y_all = y_df['label'].astype(int).reset_index(drop=True)
    else:
        y_all = y_df.iloc[:, 0].astype(int).reset_index(drop=True)
    labels = y_all.iloc[start_idx:end_idx].reset_index(drop=True)
    return df_binary, labels


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_pickle(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path


def run_single_window_experiment(
    project_root: Path,
    window_id: int,
    window_source: str = 'windows_mixed.csv',
    mining_config: RuleMiningConfig | None = None,
    warmstart_config: WarmStartConfig | None = None,
    eval_config: EvalConfig | None = None,
) -> WindowExperimentResult:
    mining_config = mining_config or RuleMiningConfig()
    warmstart_config = warmstart_config or WarmStartConfig()
    eval_config = eval_config or EvalConfig()

    df_binary, labels = load_window_binary_and_labels(project_root, window_id, window_source=window_source)
    attack_ratio = float(labels.mean())

    transactions = build_transactions(df_binary, lag=mining_config.lag)
    if len(transactions) != len(labels):
        raise ValueError(f'transactions 与 labels 长度不一致: {len(transactions)} vs {len(labels)}')
    labeled_transactions = add_transaction_labels(transactions, labels.tolist())
    df_onehot = one_hot_transactions(labeled_transactions)
    freq_items, label_rules = mine_label_rules(
        df_onehot,
        min_support=mining_config.min_support,
        min_confidence=mining_config.min_confidence,
    )
    if label_rules.empty:
        raise ValueError(f'window {window_id} 在当前阈值下没有可用标签规则')

    relaxed_normal_df = mine_relaxed_normal_rules(freq_items, p_normal=float(1.0 - attack_ratio))
    # ===== 生成 fallback_rules_df：rules_all_formatted =====
    rules_all = association_rules(
        freq_items,
        metric="confidence",
        min_threshold=0.0
    ).copy()
    
    rules_all = rules_all[
        (rules_all["consequents"].apply(lambda x: len(x) == 1)) &
        (rules_all["consequents"].apply(lambda x: list(x)[0] in [LABEL_ATTACK, LABEL_NORMAL]))
    ].copy()
    
    if len(rules_all) > 0:
        rules_all["antecedent_str"] = rules_all["antecedents"].apply(
            lambda x: " & ".join(sorted(list(x)))
        )
        rules_all["consequent_str"] = rules_all["consequents"].apply(
            lambda x: list(x)[0]
        )
    
        rules_all_formatted = rules_all[
            ["antecedent_str", "consequent_str", "support", "confidence", "lift"]
        ].sort_values(
            ["consequent_str", "confidence", "lift", "support"],
            ascending=[True, False, False, False]
        ).reset_index(drop=True)
    else:
        rules_all_formatted = pd.DataFrame(columns=[
            "antecedent_str", "consequent_str", "support", "confidence", "lift"
        ])
    mixed_rule_pool, window_mode = build_rule_pool(
                                label_rules=label_rules,
                                attack_ratio=attack_ratio,
                                config=mining_config,
                                relaxed_normal_df=relaxed_normal_df,
                                fallback_rules_df=rules_all_formatted,
                            )

    state_dict = build_initial_rule_state(mixed_rule_pool)
    X_ws, y_ws = build_warmstart_dataset(state_dict)

    artifacts_dir = project_root
    transactions_path = save_pickle(transactions, artifacts_dir / 'data' / 'stream' / 'swat' / 'window_transactions' / f'window_{window_id}_transactions.pkl')
    labeled_transactions_path = save_pickle(labeled_transactions, artifacts_dir / 'data' / 'stream' / 'swat' / 'window_labeled_transactions' / f'window_{window_id}_labeled_transactions.pkl')
    onehot_path = save_dataframe(df_onehot, artifacts_dir / 'data' / 'stream' / 'swat' / 'window_labeled_onehot' / f'window_{window_id}_labeled_onehot.csv')
    freq_items_path = save_dataframe(freq_items, artifacts_dir / 'data' / 'stream' / 'swat' / 'window_freq_items' / f'window_{window_id}_freq_items.csv')
    label_rules_path = save_dataframe(label_rules, artifacts_dir / 'data' / 'stream' / 'swat' / 'window_label_rules' / f'window_{window_id}_label_rules.csv')
    if not relaxed_normal_df.empty:
        save_dataframe(relaxed_normal_df, artifacts_dir / 'data' / 'stream' / 'swat' / 'window_label_rules' / f'window_{window_id}_normal_candidates_relaxed.csv')
    rule_pool_path = save_dataframe(mixed_rule_pool, artifacts_dir / 'data' / 'stream' / 'swat' / 'window_rule_pools' / f'window_{window_id}_mixed_rule_pool.csv')
    rule_state_path = save_pickle(state_dict, artifacts_dir / 'data' / 'stream' / 'swat' / 'window_rule_states' / f'window_{window_id}_rule_state.pkl')

    model_path = artifacts_dir / 'outputs' / 'models' / f'window_{window_id}_actor_warmstart.pth'
    warmstart = train_warmstart_model(X_ws, y_ws, model_path, config=warmstart_config)
    trace_df, metrics_df = greedy_evaluate(model_path, state_dict, eval_config=eval_config)
    baselines = evaluate_baselines(state_dict, eval_config=eval_config)

    metrics_df['window_id'] = window_id
    metrics_df['attack_ratio'] = attack_ratio
    metrics_df['window_mode'] = window_mode
    metrics_df['warmstart_train_accuracy'] = warmstart.train_accuracy
    for k, v in baselines.items():
        metrics_df[k] = v

    trace_path = save_dataframe(trace_df, artifacts_dir / 'outputs' / 'logs' / f'window_{window_id}_greedy_trace.csv')
    metrics_path = save_dataframe(metrics_df, artifacts_dir / 'outputs' / 'logs' / f'window_{window_id}_metrics.csv')

    artifacts = WindowExperimentArtifacts(
        window_id=window_id,
        attack_ratio=attack_ratio,
        transactions_path=transactions_path,
        labeled_transactions_path=labeled_transactions_path,
        onehot_path=onehot_path,
        freq_items_path=freq_items_path,
        label_rules_path=label_rules_path,
        rule_pool_path=rule_pool_path,
        rule_state_path=rule_state_path,
        model_path=model_path,
        trace_path=trace_path,
        metrics_path=metrics_path,
    )

    row = metrics_df.iloc[0]
    return WindowExperimentResult(
        window_id=window_id,
        attack_ratio=attack_ratio,
        num_rules=int(row['num_rules']),
        num_attack_rules=int(row['num_attack_rules']),
        num_normal_rules=int(row['num_normal_rules']),
        attack_keep_rate=float(row['attack_keep_rate']),
        normal_disable_rate=float(row['normal_disable_rate']),
        selection_accuracy=float(row['selection_accuracy']),
        greedy_total_reward=float(row['greedy_total_reward']),
        warmstart_train_accuracy=float(row['warmstart_train_accuracy']),
        window_mode=str(row['window_mode']),
        all_keep_reward=float(row['all_keep_reward']),
        all_disable_reward=float(row['all_disable_reward']),
        random_mean_reward=float(row['random_mean_reward']),
        random_std_reward=float(row['random_std_reward']),
        trace_df=trace_df,
        metrics_df=metrics_df,
        artifacts=artifacts,
    )


def run_multi_window_experiment(
    project_root: Path,
    window_ids: Sequence[int],
    window_source: str = 'windows_mixed.csv',
    mining_config: RuleMiningConfig | None = None,
    warmstart_config: WarmStartConfig | None = None,
    eval_config: EvalConfig | None = None,
    summary_name: str = 'formal_mixed_windows_summary.csv',
) -> pd.DataFrame:
    rows = []
    for window_id in window_ids:
        result = run_single_window_experiment(
            project_root=project_root,
            window_id=int(window_id),
            window_source=window_source,
            mining_config=mining_config,
            warmstart_config=warmstart_config,
            eval_config=eval_config,
        )
        rows.append(result.metrics_df.iloc[0].to_dict())

    summary_df = pd.DataFrame(rows)
    summary_path = project_root / 'outputs' / 'logs' / summary_name
    save_dataframe(summary_df.sort_values('window_id').reset_index(drop=True), summary_path)
    return summary_df.sort_values('window_id').reset_index(drop=True)
