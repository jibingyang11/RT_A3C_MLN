from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, fpgrowth

from src.data.transaction_utils import build_transactions
from src.rl.ac_model import ActorCriticNet
from src.rl.simple_env import SimpleRuleEnv
from src.rl.state_utils import build_initial_rule_state

LABEL_ATTACK = 'LABEL_ATTACK'
LABEL_NORMAL = 'LABEL_NORMAL'


@dataclass(frozen=True)
class RuleMiningConfig:
    lag: int = 2
    min_support: float = 0.1
    min_confidence: float = 0.6
    top_attack_rules: int = 10
    top_normal_rules: int = 10
    relaxed_normal_rules: int = 3
    attack_weight_cap: float = 3.0
    relaxed_normal_min_weight: float = 0.3
    relaxed_normal_weight_scale: float = 30.0
    relaxed_normal_weight_cap: float = 3.0
    low_attack_ratio_threshold: float = 0.30
    high_attack_ratio_threshold: float = 0.70


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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_window_binary_data(
    project_root: Path,
    window_id: int,
    window_source: str = 'windows_mixed.csv',
    lag: int = 2,
) -> Path:
    stream_dir = project_root / 'data' / 'stream' / 'swat'
    binary_dir = stream_dir / 'window_binary_data'
    binary_dir.mkdir(parents=True, exist_ok=True)
    out_path = binary_dir / f'window_{window_id}_binary.csv'

    windows_df = pd.read_csv(stream_dir / window_source)
    row = windows_df[windows_df['window_id'] == int(window_id)]
    if row.empty:
        raise ValueError(f'window_id={window_id} 不在 {window_source} 中')
    row = row.iloc[0]
    start_idx = int(row['start_idx'])
    end_idx = int(row['end_idx'])

    X_binary = pd.read_csv(project_root / 'data' / 'processed' / 'swat' / 'X_filled_binary.csv')
    expected_rows = (end_idx - start_idx) + int(lag)

    rebuild = True
    if out_path.exists():
        try:
            cur = pd.read_csv(out_path, nrows=5)
            full_rows = sum(1 for _ in open(out_path, 'r', encoding='utf-8')) - 1
            rebuild = full_rows != expected_rows
        except Exception:
            rebuild = True

    if rebuild:
        end_with_lag = min(len(X_binary), end_idx + int(lag))
        local_df = X_binary.iloc[start_idx:end_with_lag].reset_index(drop=True)
        local_df.to_csv(out_path, index=False)

    return out_path


def load_window_binary_and_labels(
    project_root: Path,
    window_id: int,
    window_source: str = 'windows_mixed.csv',
    lag: int = 2,
) -> tuple[pd.DataFrame, pd.Series, float]:
    binary_path = ensure_window_binary_data(project_root, window_id, window_source=window_source, lag=lag)
    df_binary = pd.read_csv(binary_path)

    windows_df = pd.read_csv(project_root / 'data' / 'stream' / 'swat' / window_source)
    row = windows_df[windows_df['window_id'] == int(window_id)].iloc[0]
    start_idx = int(row['start_idx'])
    end_idx = int(row['end_idx'])

    y_df = pd.read_csv(project_root / 'data' / 'processed' / 'swat' / 'y_filled.csv')
    y_all = y_df['label'].astype(int).reset_index(drop=True) if 'label' in y_df.columns else y_df.iloc[:, 0].astype(int).reset_index(drop=True)
    labels = y_all.iloc[start_idx:end_idx].reset_index(drop=True)
    return df_binary, labels, float(labels.mean())


def add_transaction_labels(transactions: list[list[str]], labels: list[int]) -> list[list[str]]:
    if len(transactions) != len(labels):
        raise ValueError(f'transactions 与 labels 长度不一致: {len(transactions)} vs {len(labels)}')
    return [list(tx) + [LABEL_ATTACK if int(y) == 1 else LABEL_NORMAL] for tx, y in zip(transactions, labels)]


def one_hot_transactions(labeled_transactions: list[list[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    arr = te.fit(labeled_transactions).transform(labeled_transactions)
    return pd.DataFrame(arr, columns=te.columns_)


def mine_frequent_itemsets(df_onehot: pd.DataFrame, min_support: float) -> pd.DataFrame:
    freq_items = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=2)
    if freq_items.empty:
        return pd.DataFrame(columns=['support', 'itemsets', 'itemset_len'])
    freq_items = freq_items.sort_values('support', ascending=False).reset_index(drop=True)
    freq_items['itemset_len'] = freq_items['itemsets'].apply(len)
    return freq_items


def format_label_rules(freq_items: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    if freq_items.empty:
        return pd.DataFrame(columns=['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift'])
    rules = association_rules(freq_items, metric='confidence', min_threshold=min_confidence).copy()
    rules = rules[
        (rules['consequents'].apply(lambda x: len(x) == 1))
        & (rules['consequents'].apply(lambda x: list(x)[0] in [LABEL_ATTACK, LABEL_NORMAL]))
    ].copy()
    if rules.empty:
        return pd.DataFrame(columns=['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift'])
    rules['antecedent_str'] = rules['antecedents'].apply(lambda x: ' & '.join(sorted(list(x))))
    rules['consequent_str'] = rules['consequents'].apply(lambda x: list(x)[0])
    return rules[['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']].sort_values(
        ['consequent_str', 'confidence', 'lift', 'support'],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def format_all_rules_for_fallback(freq_items: pd.DataFrame) -> pd.DataFrame:
    if freq_items.empty:
        return pd.DataFrame(columns=['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift'])
    rules = association_rules(freq_items, metric='confidence', min_threshold=0.0).copy()
    rules = rules[
        (rules['consequents'].apply(lambda x: len(x) == 1))
        & (rules['consequents'].apply(lambda x: list(x)[0] in [LABEL_ATTACK, LABEL_NORMAL]))
    ].copy()
    if rules.empty:
        return pd.DataFrame(columns=['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift'])
    rules['antecedent_str'] = rules['antecedents'].apply(lambda x: ' & '.join(sorted(list(x))))
    rules['consequent_str'] = rules['consequents'].apply(lambda x: list(x)[0])
    return rules[['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']].sort_values(
        ['consequent_str', 'confidence', 'lift', 'support'],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def confidence_to_weight(confidence: float, eps: float = 1e-6) -> float:
    confidence = float(np.clip(confidence, eps, 1 - eps))
    return float(np.log(confidence / (1.0 - confidence)))
    

def clipped_weight(confidence: float, wmax: float = 3.0) -> float:
    return float(np.clip(confidence_to_weight(confidence), 0.0, wmax))


def relaxed_weight(score_relaxed: float, wmin: float = 0.3, scale: float = 30.0, wmax: float = 3.0) -> float:
    return float(np.clip(scale * float(score_relaxed), wmin, wmax))
def relaxed_normal_weight(score_relaxed, wmin=0.3, scale=30.0, wmax=3.0):
    w = scale * float(score_relaxed)
    return float(np.clip(w, wmin, wmax))

def decide_window_mode(attack_ratio: float, config: RuleMiningConfig) -> str:
    if attack_ratio < config.low_attack_ratio_threshold:
        return 'balanced'
    if attack_ratio >= config.high_attack_ratio_threshold:
        return 'relaxed'
    return 'standard'


def rank_relaxed_candidates(df: pd.DataFrame, target_label: str, prior_prob: float, top_k: int) -> pd.DataFrame:
    cand = df[df['consequent_str'] == target_label].copy()
    if cand.empty:
        return cand
    cand['prior_prob'] = float(prior_prob)
    cand['conf_gain'] = cand['confidence'] - cand['prior_prob']
    cand['score_relaxed'] = cand['conf_gain'] + 0.1 * (cand['lift'] - 1.0)
    return cand.sort_values(
        ['score_relaxed', 'confidence', 'lift', 'support'],
        ascending=[False, False, False, False],
    ).head(top_k).reset_index(drop=True)


def build_rule_pool_robust(label_rules, fallback_rules_df, attack_ratio, config):
    mode = decide_window_mode(attack_ratio, config)

    attack_rules = label_rules[label_rules["consequent_str"] == LABEL_ATTACK].reset_index(drop=True)
    normal_rules = label_rules[label_rules["consequent_str"] == LABEL_NORMAL].reset_index(drop=True)

    attack_top = attack_rules.head(config.top_attack_rules).copy()
    if len(attack_top) > 0:
        attack_top["formula"] = attack_top["antecedent_str"] + " => " + LABEL_ATTACK
        attack_top["weight"] = attack_top["confidence"].apply(
            lambda x: clipped_weight(x, wmax=config.attack_weight_cap)
        )
        attack_top["target_label"] = 1

    if mode == "relaxed":
        normal_top = rank_relaxed_candidates(
            fallback_rules_df,
            target_label=LABEL_NORMAL,
            prior_prob=float(1.0 - attack_ratio),
            top_k=config.relaxed_normal_rules,
        ).copy()
        if len(normal_top) > 0:
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
        normal_top = normal_rules.head(config.top_normal_rules).copy()
        if len(normal_top) > 0:
            normal_top["formula"] = normal_top["antecedent_str"] + " => " + LABEL_NORMAL
            normal_top["weight"] = normal_top["confidence"].apply(
                lambda x: clipped_weight(x, wmax=config.attack_weight_cap)
            )
            normal_top["target_label"] = 0

    # 所有模式都允许 fallback，不只是 balanced
    if mode == "balanced":
        need_attack = min(3, config.top_attack_rules)
        need_normal = min(3, config.top_normal_rules)
    elif mode == "relaxed":
        need_attack = max(1, min(config.top_attack_rules, 3))
        need_normal = max(1, config.relaxed_normal_rules)
    else:
        need_attack = max(1, min(config.top_attack_rules, 3))
        need_normal = max(1, min(config.top_normal_rules, 3))

    if len(attack_top) < need_attack:
        relaxed_attack_top = rank_relaxed_candidates(
            fallback_rules_df,
            target_label=LABEL_ATTACK,
            prior_prob=float(attack_ratio),
            top_k=max(need_attack, config.top_attack_rules if mode != "balanced" else need_attack),
        ).copy()
        if len(relaxed_attack_top) > 0:
            relaxed_attack_top["consequent_str"] = LABEL_ATTACK
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

    if len(normal_top) < need_normal:
        relaxed_normal_top = rank_relaxed_candidates(
            fallback_rules_df,
            target_label=LABEL_NORMAL,
            prior_prob=float(1.0 - attack_ratio),
            top_k=max(
                need_normal,
                config.relaxed_normal_rules if mode == "relaxed" else config.top_normal_rules
            ),
        ).copy()
        if len(relaxed_normal_top) > 0:
            relaxed_normal_top["consequent_str"] = LABEL_NORMAL
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

    if len(attack_top) == 0 or len(normal_top) == 0:
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

    return mixed_rule_pool.reset_index(drop=True), mode


def build_warmstart_dataset(state_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    active_mask = np.asarray(state_dict['active_mask'], dtype=np.float32)
    weights = np.asarray(state_dict['weights'], dtype=np.float32)
    rule_scores = np.asarray(state_dict['rule_scores'], dtype=np.float32)
    target_labels = np.asarray(state_dict['target_labels'], dtype=np.int64)
    num_rules = int(state_dict['num_rules'])

    def _build(rule_idx: int) -> np.ndarray:
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

    X_ws = np.stack([_build(i) for i in range(num_rules)], axis=0)
    y_ws = np.array([0 if t == 1 else 1 for t in target_labels], dtype=np.int64)
    return X_ws, y_ws


def get_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        for k in ['policy_logits', 'logits', 'actor_logits']:
            if k in output:
                return output[k]
    return output


def train_warmstart_model(X_ws: np.ndarray, y_ws: np.ndarray, save_path: Path, config: WarmStartConfig) -> tuple[float, np.ndarray]:
    set_global_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X_ws, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_ws, dtype=torch.long).to(device)
    num_keep = int((y_ws == 0).sum())
    num_disable = int((y_ws == 1).sum())
    class_weights = torch.tensor([1.0 / max(num_keep, 1), 1.0 / max(num_disable, 1)], dtype=torch.float32).to(device)
    model = ActorCriticNet(state_dim=config.state_dim, action_dim=config.action_dim, hidden_dim=config.hidden_dim).to(device)
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
        pred = get_logits(model(X_tensor)).argmax(dim=1)
        acc = float((pred == y_tensor).float().mean().item())
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return acc, pred.detach().cpu().numpy()


def rollout_actions(state_dict: dict, actions: list[int], eval_config: EvalConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    env = SimpleRuleEnv(state_dict, step_size=eval_config.env_step_size, max_steps=eval_config.env_max_steps)
    state = env.reset()
    records = []
    total_reward = 0.0
    target_labels = np.asarray(state_dict['target_labels'])
    for rule_idx, action in enumerate(actions):
        state, reward, _, info = env.step(rule_idx, int(action))
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


def greedy_evaluate(model_path: Path, state_dict: dict, config: WarmStartConfig, eval_config: EvalConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorCriticNet(state_dim=config.state_dim, action_dim=config.action_dim, hidden_dim=config.hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    env = SimpleRuleEnv(state_dict, step_size=eval_config.env_step_size, max_steps=eval_config.env_max_steps)
    state = env.reset()
    actions = []
    for _ in range(int(state_dict['num_rules'])):
        x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = int(torch.argmax(get_logits(model(x)), dim=1).item())
        actions.append(action)
        state, _, _, _ = env.step(len(actions) - 1, action)
    trace_df, metrics = rollout_actions(state_dict, actions, eval_config)
    return trace_df, pd.DataFrame([metrics])


def evaluate_baselines(state_dict: dict, eval_config: EvalConfig) -> dict[str, float]:
    num_rules = int(state_dict['num_rules'])
    _, keep_metrics = rollout_actions(state_dict, [0] * num_rules, eval_config)
    _, disable_metrics = rollout_actions(state_dict, [1] * num_rules, eval_config)
    rng = np.random.default_rng(eval_config.seed)
    random_rewards = []
    for _ in range(eval_config.random_trials):
        actions = rng.integers(0, 2, size=num_rules).tolist()
        _, metrics = rollout_actions(state_dict, actions, eval_config)
        random_rewards.append(metrics['greedy_total_reward'])
    return {
        'all_keep_reward': float(keep_metrics['greedy_total_reward']),
        'all_disable_reward': float(disable_metrics['greedy_total_reward']),
        'random_mean_reward': float(np.mean(random_rewards)) if random_rewards else np.nan,
        'random_std_reward': float(np.std(random_rewards)) if random_rewards else np.nan,
    }


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_pickle(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path


def prepare_window_learning_artifacts(
    project_root: Path,
    window_id: int,
    window_source: str,
    mining_config: RuleMiningConfig,
) -> dict:
    df_binary, labels, attack_ratio = load_window_binary_and_labels(
        project_root,
        window_id,
        window_source=window_source,
        lag=mining_config.lag,
    )

    transactions = build_transactions(df_binary, lag=mining_config.lag)
    labeled_transactions = add_transaction_labels(transactions, labels.tolist())
    df_onehot = one_hot_transactions(labeled_transactions)

    # ===== 自适应 support 候选 =====
    support_candidates = [float(mining_config.min_support)]

    # 低攻击比例窗口：逐步放宽 support
    if attack_ratio < 0.30:
        for s in [0.08, 0.05, 0.03]:
            if s not in support_candidates:
                support_candidates.append(s)
    elif attack_ratio < 0.40:
        for s in [0.08, 0.05]:
            if s not in support_candidates:
                support_candidates.append(s)

    last_exc = None
    chosen_min_support = None
    freq_items = None
    label_rules = None
    fallback_rules = None
    mixed_rule_pool = None
    window_mode = None

    # ===== 逐个 support 尝试，直到规则池构造成功 =====
    for min_sup in support_candidates:
        try:
            freq_items_try = mine_frequent_itemsets(df_onehot, min_sup)
            label_rules_try = format_label_rules(freq_items_try, mining_config.min_confidence)
            fallback_rules_try = format_all_rules_for_fallback(freq_items_try)

            mixed_rule_pool_try, window_mode_try = build_rule_pool_robust(
                label_rules_try,
                fallback_rules_try,
                attack_ratio,
                mining_config,
            )

            # 成功则采用这一组结果
            chosen_min_support = float(min_sup)
            freq_items = freq_items_try
            label_rules = label_rules_try
            fallback_rules = fallback_rules_try
            mixed_rule_pool = mixed_rule_pool_try
            window_mode = window_mode_try
            break

        except Exception as e:
            last_exc = e
            continue

    if mixed_rule_pool is None:
        raise ValueError(
            f"window {window_id} 在 support_candidates={support_candidates} 下仍无法构造规则池；"
            f"最后错误: {repr(last_exc)}"
        )

    state_dict = build_initial_rule_state(mixed_rule_pool)
    X_ws, y_ws = build_warmstart_dataset(state_dict)

    return {
        "window_id": int(window_id),
        "attack_ratio": float(attack_ratio),
        "window_mode": window_mode,
        "chosen_min_support": chosen_min_support,
        "transactions": transactions,
        "labeled_transactions": labeled_transactions,
        "df_onehot": df_onehot,
        "freq_items": freq_items,
        "label_rules": label_rules,
        "fallback_rules": fallback_rules,
        "mixed_rule_pool": mixed_rule_pool,
        "state_dict": state_dict,
        "X_ws": X_ws,
        "y_ws": y_ws,
    }

def run_single_window_experiment(
    project_root: Path,
    window_id: int,
    window_source: str = 'windows_mixed.csv',
    mining_config: RuleMiningConfig | None = None,
    warmstart_config: WarmStartConfig | None = None,
    eval_config: EvalConfig | None = None,
) -> pd.DataFrame:
    mining_config = mining_config or RuleMiningConfig()
    warmstart_config = warmstart_config or WarmStartConfig()
    eval_config = eval_config or EvalConfig()

    art = prepare_window_learning_artifacts(project_root, window_id, window_source, mining_config)

    stream_root = project_root / 'data' / 'stream' / 'swat'
    save_pickle(art['transactions'], stream_root / 'window_transactions' / f'window_{window_id}_transactions.pkl')
    save_pickle(art['labeled_transactions'], stream_root / 'window_labeled_transactions' / f'window_{window_id}_labeled_transactions.pkl')
    save_dataframe(art['df_onehot'], stream_root / 'window_labeled_onehot' / f'window_{window_id}_labeled_onehot.csv')
    save_dataframe(art['freq_items'], stream_root / 'window_freq_items' / f'window_{window_id}_freq_items.csv')
    save_dataframe(art['label_rules'], stream_root / 'window_label_rules' / f'window_{window_id}_label_rules.csv')
    save_dataframe(art['mixed_rule_pool'], stream_root / 'window_rule_pools' / f'window_{window_id}_mixed_rule_pool.csv')
    save_pickle(art['state_dict'], stream_root / 'window_rule_states' / f'window_{window_id}_rule_state.pkl')

    model_path = project_root / 'outputs' / 'models' / f'window_{window_id}_actor_warmstart.pth'
    train_acc, _ = train_warmstart_model(art['X_ws'], art['y_ws'], model_path, warmstart_config)
    trace_df, metrics_df = greedy_evaluate(model_path, art['state_dict'], warmstart_config, eval_config)
    baselines = evaluate_baselines(art['state_dict'], eval_config)

    metrics_df['window_id'] = int(window_id)
    metrics_df['attack_ratio'] = float(art['attack_ratio'])
    metrics_df['window_mode'] = art['window_mode']
    metrics_df['warmstart_train_accuracy'] = float(train_acc)
    for k, v in baselines.items():
        metrics_df[k] = v

    save_dataframe(trace_df, project_root / 'outputs' / 'logs' / f'window_{window_id}_greedy_trace.csv')
    save_dataframe(metrics_df, project_root / 'outputs' / 'logs' / f'window_{window_id}_metrics.csv')
    return metrics_df


def run_batch_windows(
    project_root: Path,
    window_ids: list[int],
    window_source: str = 'windows_mixed.csv',
    mining_config: RuleMiningConfig | None = None,
    warmstart_config: WarmStartConfig | None = None,
    eval_config: EvalConfig | None = None,
    continue_on_error: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mining_config = mining_config or RuleMiningConfig()
    warmstart_config = warmstart_config or WarmStartConfig()
    eval_config = eval_config or EvalConfig()
    success_rows = []
    fail_rows = []
    for win_id in window_ids:
        try:
            metrics_df = run_single_window_experiment(project_root, int(win_id), window_source, mining_config, warmstart_config, eval_config)
            success_rows.append(metrics_df.iloc[0].to_dict())
        except Exception as e:
            fail_rows.append({'window_id': int(win_id), 'error_type': type(e).__name__, 'error_message': str(e)})
            if not continue_on_error:
                raise
    return pd.DataFrame(success_rows), pd.DataFrame(fail_rows)


def build_clean_summary(summary_df: pd.DataFrame, low_attack_max: float = 0.30, high_attack_min: float = 0.70) -> pd.DataFrame:
    df = summary_df.copy()
    def _bin(x: float) -> str:
        if x < low_attack_max:
            return 'low'
        if x >= high_attack_min:
            return 'high'
        return 'mid'
    df['attack_bin'] = df['attack_ratio'].apply(lambda x: _bin(float(x)))
    clean_cols = [
        'window_id', 'attack_ratio', 'attack_bin', 'window_mode',
        'num_rules', 'num_attack_rules', 'num_normal_rules',
        'attack_keep_rate', 'normal_disable_rate', 'selection_accuracy',
        'greedy_total_reward', 'warmstart_train_accuracy',
        'all_keep_reward', 'all_disable_reward', 'random_mean_reward', 'random_std_reward',
    ]
    existing = [c for c in clean_cols if c in df.columns]
    return df[existing].sort_values('attack_ratio').reset_index(drop=True)


def build_group_stats(clean_df: pd.DataFrame) -> pd.DataFrame:
    return clean_df.groupby('attack_bin')[[
        'attack_keep_rate', 'normal_disable_rate', 'selection_accuracy', 'greedy_total_reward'
    ]].agg(['mean', 'std', 'count'])
