from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss


EPS = 1e-9


def _safe_proba(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
    if prob.ndim == 1:
        return np.column_stack([1.0 - prob, prob])
    return prob


def _constant_proba(y: np.ndarray, n: int) -> np.ndarray:
    p = float(np.mean(y)) if len(y) else 0.5
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return np.tile([1.0 - p, p], (n, 1))


def _logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return np.log(prob / (1.0 - prob))


def _sigmoid(score: np.ndarray) -> np.ndarray:
    score = np.clip(score, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-score))


@dataclass
class _A3CComponent:
    selected: list[int]
    estimator: LogisticRegression
    val_loss: float
    val_prob: np.ndarray
    coef: np.ndarray
    intercept: float


class RuleModel:
    def fit(self, x_train, y_train, x_val, y_val, rules):
        raise NotImplementedError

    def predict_proba(self, x):
        raise NotImplementedError

    def selected_rule_indices(self) -> list[int]:
        return []

    def rule_weight(self, rule_idx: int) -> float | None:
        return None


class SklearnRuleModel(RuleModel):
    def __init__(self, estimator, threshold: float = 1e-7, all_rules: bool = False) -> None:
        self.estimator = estimator
        self.threshold = threshold
        self.all_rules = all_rules
        self._prior_y: np.ndarray | None = None
        self._n_features = 0

    def fit(self, x_train, y_train, x_val, y_val, rules):
        self._prior_y = np.asarray(y_train)
        self._n_features = x_train.shape[1]
        self.estimator.fit(x_train, y_train)
        return self

    def predict_proba(self, x):
        if hasattr(self.estimator, "predict_proba"):
            return _safe_proba(self.estimator.predict_proba(x))
        if hasattr(self.estimator, "decision_function"):
            score = self.estimator.decision_function(x)
            prob = 1.0 / (1.0 + np.exp(-score))
            return _safe_proba(prob)
        return _constant_proba(self._prior_y, len(x))

    def selected_rule_indices(self) -> list[int]:
        if self.all_rules:
            return list(range(self._n_features))
        if hasattr(self.estimator, "coef_"):
            coef = np.ravel(self.estimator.coef_)
            return [int(idx) for idx in np.flatnonzero(np.abs(coef) > self.threshold)]
        if hasattr(self.estimator, "feature_importances_"):
            imp = np.asarray(self.estimator.feature_importances_)
            return [int(idx) for idx in np.flatnonzero(imp > self.threshold)]
        return list(range(self._n_features))

    def rule_weight(self, rule_idx: int) -> float | None:
        if hasattr(self.estimator, "coef_"):
            coef = np.ravel(self.estimator.coef_)
            if 0 <= rule_idx < len(coef):
                return float(coef[rule_idx])
        if hasattr(self.estimator, "feature_importances_"):
            imp = np.asarray(self.estimator.feature_importances_)
            if 0 <= rule_idx < len(imp):
                return float(imp[rule_idx])
        return None


class OnlineSGDMLN(RuleModel):
    def __init__(self, random_state: int = 7, epochs: int = 5, batch_size: int = 256) -> None:
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.estimator = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-4,
            l1_ratio=0.05,
            learning_rate="optimal",
            random_state=random_state,
        )
        self._n_features = 0
        self._classes = np.array([0, 1])

    def fit(self, x_train, y_train, x_val, y_val, rules):
        self._n_features = x_train.shape[1]
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(len(y_train))
        first = True
        for _ in range(self.epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if first:
                    self.estimator.partial_fit(x_train[batch], y_train[batch], classes=self._classes)
                    first = False
                else:
                    self.estimator.partial_fit(x_train[batch], y_train[batch])
        return self

    def predict_proba(self, x):
        return _safe_proba(self.estimator.predict_proba(x))

    def selected_rule_indices(self) -> list[int]:
        coef = np.ravel(self.estimator.coef_)
        return [int(idx) for idx in np.flatnonzero(np.abs(coef) > 1e-7)]

    def rule_weight(self, rule_idx: int) -> float | None:
        coef = np.ravel(self.estimator.coef_)
        if 0 <= rule_idx < len(coef):
            return float(coef[rule_idx])
        return None


class BeamSearchMLN(RuleModel):
    def __init__(
        self,
        max_selected: int = 28,
        search_pool: int = 100,
        min_improvement: float = 1e-4,
        random_state: int = 7,
    ) -> None:
        self.max_selected = max_selected
        self.search_pool = search_pool
        self.min_improvement = min_improvement
        self.random_state = random_state
        self.selected_: list[int] = []
        self.estimator_: LogisticRegression | None = None
        self.prior_: float = 0.5

    def fit(self, x_train, y_train, x_val, y_val, rules):
        self.prior_ = float(np.mean(y_train))
        candidates = list(range(min(self.search_pool, x_train.shape[1])))
        selected: list[int] = []
        current = log_loss(y_val, np.full(len(y_val), self.prior_), labels=[0, 1])

        for _ in range(self.max_selected):
            best_idx = None
            best_loss = current
            for idx in candidates:
                if idx in selected:
                    continue
                trial = selected + [idx]
                model = LogisticRegression(
                    solver="liblinear",
                    C=0.8,
                    max_iter=300,
                    random_state=self.random_state,
                )
                model.fit(x_train[:, trial], y_train)
                prob = model.predict_proba(x_val[:, trial])[:, 1]
                loss = log_loss(y_val, prob, labels=[0, 1])
                if loss < best_loss:
                    best_loss = loss
                    best_idx = idx
            if best_idx is None or current - best_loss < self.min_improvement:
                break
            selected.append(best_idx)
            current = best_loss

        self.selected_ = selected
        if selected:
            self.estimator_ = LogisticRegression(
                solver="liblinear",
                C=0.8,
                max_iter=500,
                random_state=self.random_state,
            )
            self.estimator_.fit(x_train[:, selected], y_train)
        return self

    def predict_proba(self, x):
        if not self.selected_ or self.estimator_ is None:
            return np.tile([1.0 - self.prior_, self.prior_], (len(x), 1))
        return _safe_proba(self.estimator_.predict_proba(x[:, self.selected_]))

    def selected_rule_indices(self) -> list[int]:
        return list(self.selected_)

    def rule_weight(self, rule_idx: int) -> float | None:
        if self.estimator_ is None or rule_idx not in self.selected_:
            return None
        local_idx = self.selected_.index(rule_idx)
        return float(np.ravel(self.estimator_.coef_)[local_idx])


class RTA3CMLN(RuleModel):
    """Asynchronous advantage actor-critic controller for MLN rule selection."""

    def __init__(
        self,
        workers: int = 4,
        episodes: int = 18,
        n_steps: int = 8,
        controller_rules: int = 220,
        gamma: float = 0.92,
        actor_lr: float = 0.035,
        critic_lr: float = 0.08,
        entropy_beta: float = 0.002,
        complexity_penalty: float = 0.001,
        f1_weight: float = 0.06,
        accuracy_weight: float = 0.02,
        ensemble_penalty: float = 0.0008,
        fast_mode: bool = False,
        fixed_c: float = 1.0,
        fixed_ensemble_size: int = 3,
        fixed_temperature: float = 1.0,
        fixed_shift: float = 0.0,
        max_active_rules: int | None = None,
        total_episode_budget: int | None = None,
        random_state: int = 7,
    ) -> None:
        self.workers = workers
        self.episodes = episodes
        self.n_steps = n_steps
        self.controller_rules = controller_rules
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_beta = entropy_beta
        self.complexity_penalty = complexity_penalty
        self.f1_weight = f1_weight
        self.accuracy_weight = accuracy_weight
        self.ensemble_penalty = ensemble_penalty
        self.fast_mode = fast_mode
        self.fixed_c = fixed_c
        self.fixed_ensemble_size = fixed_ensemble_size
        self.fixed_temperature = fixed_temperature
        self.fixed_shift = fixed_shift
        self.max_active_rules = max_active_rules
        self.total_episode_budget = total_episode_budget
        self.random_state = random_state
        self.selected_: list[int] = []
        self.estimator_: LogisticRegression | None = None
        self.prior_: float = 0.5
        self._weights: dict[int, float] = {}
        self.components_: list[_A3CComponent] = []
        self.component_weights_: np.ndarray = np.array([], dtype=float)
        self.temperature_: float = 1.0
        self.shift_: float = 0.0
        self.training_trace_: list[dict[str, float]] = []
        self.best_reward_: float = -np.inf
        self.eval_count_: int = 0

    def fit(self, x_train, y_train, x_val, y_val, rules):
        self.prior_ = float(np.mean(y_train))
        self.training_trace_ = []
        self.best_reward_ = -np.inf
        self.eval_count_ = 0
        n_controller = min(self.controller_rules, x_train.shape[1])
        if n_controller == 0:
            return self
        max_active = n_controller if self.max_active_rules is None else max(1, min(int(self.max_active_rules), n_controller))

        fit_t0 = time.perf_counter()
        state_dim = n_controller + 4
        action_dim = n_controller
        rng = np.random.default_rng(self.random_state)
        theta = rng.normal(0.0, 0.01, size=(action_dim, state_dim))
        value_w = np.zeros(state_dim, dtype=float)
        update_lock = threading.Lock()
        cache_lock = threading.Lock()
        reward_cache: dict[bytes, float] = {}
        best = {"reward": -np.inf, "mask": np.zeros(n_controller, dtype=bool)}
        eval_counter = {"count": 0}
        completed_episodes = {"count": 0}

        base_loss = log_loss(y_val, np.full(len(y_val), self.prior_), labels=[0, 1])

        def state_from(mask: np.ndarray, last_reward: float, step: int) -> np.ndarray:
            state = np.zeros(state_dim, dtype=float)
            state[:n_controller] = mask.astype(float)
            state[n_controller] = mask.mean()
            state[n_controller + 1] = math.tanh(last_reward)
            state[n_controller + 2] = math.tanh(max(best["reward"], -5.0))
            state[n_controller + 3] = step / max(1, self.n_steps)
            return state

        def softmax(logits: np.ndarray) -> np.ndarray:
            logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            logits = logits - np.max(logits)
            exp = np.exp(logits)
            total = np.sum(exp)
            if not np.isfinite(total) or total <= 0:
                return np.full_like(logits, 1.0 / len(logits), dtype=float)
            probs = exp / total
            if not np.all(np.isfinite(probs)):
                return np.full_like(logits, 1.0 / len(logits), dtype=float)
            return probs

        def evaluate_mask(mask: np.ndarray, worker_id: int | None = None, episode: int | None = None, step: int | None = None) -> float:
            key = np.packbits(mask.astype(np.uint8)).tobytes()
            with cache_lock:
                if key in reward_cache:
                    return reward_cache[key]
            selected = np.flatnonzero(mask).tolist()
            if not selected:
                loss = base_loss
            else:
                model = LogisticRegression(
                    solver="liblinear",
                    C=0.8,
                    max_iter=250,
                    random_state=self.random_state,
                )
                model.fit(x_train[:, selected], y_train)
                prob = model.predict_proba(x_val[:, selected])[:, 1]
                loss = log_loss(y_val, prob, labels=[0, 1])
            reward = float(base_loss - loss - self.complexity_penalty * (len(selected) / n_controller))
            with cache_lock:
                eval_counter["count"] += 1
                eval_id = eval_counter["count"]
                reward_cache[key] = reward
            with update_lock:
                if reward > best["reward"]:
                    best["reward"] = reward
                    best["mask"] = mask.copy()
                    self.best_reward_ = reward
                    self.training_trace_.append(
                        {
                            "time_sec": float(time.perf_counter() - fit_t0),
                            "eval_count": float(eval_id),
                            "completed_episodes": float(completed_episodes["count"]),
                            "worker_id": float(-1 if worker_id is None else worker_id),
                            "episode": float(-1 if episode is None else episode),
                            "step": float(-1 if step is None else step),
                            "best_reward": float(reward),
                            "selected_count": float(len(selected)),
                        }
                    )
            return reward

        def apply_updates(states, actions, rewards, bootstrap):
            nonlocal theta, value_w
            returns = []
            running = bootstrap
            for reward in reversed(rewards):
                running = reward + self.gamma * running
                returns.append(running)
            returns.reverse()
            with update_lock:
                for state, action, ret in zip(states, actions, returns):
                    logits = theta @ state
                    probs = softmax(logits)
                    value = float(value_w @ state)
                    advantage = float(np.clip(ret - value, -5.0, 5.0))
                    grad = -probs[:, None] * state[None, :]
                    grad[action] += state
                    entropy_grad = -probs[:, None] * (np.log(probs + EPS) + 1.0)[:, None] * state[None, :]
                    theta += self.actor_lr * (advantage * grad + self.entropy_beta * entropy_grad)
                    value_w += self.critic_lr * advantage * state
                    theta = np.nan_to_num(theta, nan=0.0, posinf=5.0, neginf=-5.0)
                    value_w = np.nan_to_num(value_w, nan=0.0, posinf=5.0, neginf=-5.0)
                    np.clip(theta, -5.0, 5.0, out=theta)
                    np.clip(value_w, -5.0, 5.0, out=value_w)

        if self.total_episode_budget is None:
            episode_counts = [int(self.episodes)] * int(self.workers)
        else:
            total_budget = max(1, int(self.total_episode_budget))
            base = total_budget // max(1, int(self.workers))
            extra = total_budget % max(1, int(self.workers))
            episode_counts = [base + (1 if idx < extra else 0) for idx in range(int(self.workers))]

        def worker_loop(worker_id: int):
            local_rng = np.random.default_rng(self.random_state + 1009 * (worker_id + 1))
            for episode in range(episode_counts[worker_id]):
                mask = np.zeros(n_controller, dtype=bool)
                if episode % 3 == 0:
                    warm = min(max_active, 6 + (episode % 5))
                    mask[:warm] = True
                elif episode % 3 == 1:
                    warm = min(max_active, max(3, n_controller // 6))
                    mask[local_rng.choice(n_controller, size=warm, replace=False)] = True
                last_reward = evaluate_mask(mask, worker_id=worker_id, episode=episode, step=-1)
                states = []
                actions = []
                rewards = []
                for step in range(self.n_steps):
                    state = state_from(mask, last_reward, step)
                    with update_lock:
                        probs = softmax(theta @ state)
                    action = int(local_rng.choice(action_dim, p=probs))
                    mask = mask.copy()
                    mask[action] = ~mask[action]
                    if mask[action] and int(mask.sum()) > max_active:
                        active = np.flatnonzero(mask)
                        removable = active[active != action]
                        if len(removable):
                            mask[int(local_rng.choice(removable))] = False
                    reward = evaluate_mask(mask, worker_id=worker_id, episode=episode, step=step)
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    last_reward = reward
                    if len(states) == self.n_steps:
                        bootstrap_state = state_from(mask, reward, step + 1)
                        with update_lock:
                            bootstrap = float(value_w @ bootstrap_state)
                        apply_updates(states, actions, rewards, bootstrap)
                        states, actions, rewards = [], [], []
                if states:
                    apply_updates(states, actions, rewards, 0.0)
                with update_lock:
                    completed_episodes["count"] += 1

        threads = [threading.Thread(target=worker_loop, args=(idx,), daemon=True) for idx in range(self.workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.eval_count_ = int(eval_counter["count"])

        masks = self._seed_masks(x_train, y_train, n_controller)
        masks.append(best["mask"].copy())
        masks.extend(self._actor_policy_masks(theta, value_w, n_controller))
        self._fit_policy_ensemble(x_train, y_train, x_val, y_val, masks)
        return self

    def _seed_masks(self, x_train: np.ndarray, y_train: np.ndarray, n_controller: int) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        max_active = n_controller if self.max_active_rules is None else max(1, min(int(self.max_active_rules), n_controller))
        prefix_sizes = [8, 12, 16, 24, 28, 32, 43, 54, 72, 100, 150, n_controller]
        for size in prefix_sizes:
            mask = np.zeros(n_controller, dtype=bool)
            mask[: min(size, max_active, n_controller)] = True
            masks.append(mask)

        try:
            model = LogisticRegression(
                solver="liblinear",
                penalty="l1",
                C=self.fixed_c,
                max_iter=1000,
                random_state=self.random_state,
            )
            model.fit(x_train[:, :n_controller], y_train)
            selected = np.flatnonzero(np.abs(np.ravel(model.coef_)) > 1e-7)
            if len(selected):
                mask = np.zeros(n_controller, dtype=bool)
                coefs = np.abs(np.ravel(model.coef_))
                ranked = selected[np.argsort(coefs[selected])[::-1]]
                mask[ranked[:max_active]] = True
                masks.append(mask)
        except ValueError:
            pass
        return masks

    def _actor_policy_masks(
        self,
        theta: np.ndarray,
        value_w: np.ndarray,
        n_controller: int,
    ) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        max_active = n_controller if self.max_active_rules is None else max(1, min(int(self.max_active_rules), n_controller))
        for density in [0.08, 0.12, 0.18, 0.25, 0.35]:
            mask = np.zeros(n_controller, dtype=bool)
            state = np.zeros(n_controller + 4, dtype=float)
            state[n_controller] = density
            logits = theta @ state
            order = np.argsort(logits)[::-1]
            keep = max(1, min(max_active, n_controller, int(round(n_controller * density))))
            mask[order[:keep]] = True
            masks.append(mask)
        if len(value_w) >= n_controller:
            mask = np.zeros(n_controller, dtype=bool)
            order = np.argsort(value_w[:n_controller])[::-1]
            mask[order[: max(1, min(max_active, n_controller, n_controller // 5))]] = True
            masks.append(mask)
        return masks

    def _fit_policy_ensemble(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        masks: list[np.ndarray],
    ) -> None:
        fixed_c = float(self.fixed_c)
        fixed_ensemble_size = max(1, int(self.fixed_ensemble_size))
        fixed_temperature = float(self.fixed_temperature)
        fixed_shift = float(self.fixed_shift)
        unique_masks: list[np.ndarray] = []
        seen: set[bytes] = set()
        for mask in masks:
            mask = mask.astype(bool)[: x_train.shape[1]]
            if not mask.any():
                continue
            key = np.packbits(mask.astype(np.uint8)).tobytes()
            if key in seen:
                continue
            seen.add(key)
            unique_masks.append(mask)

        components: list[_A3CComponent] = []
        for mask in unique_masks:
            selected = np.flatnonzero(mask).astype(int).tolist()
            try:
                estimator = LogisticRegression(
                    solver="liblinear",
                    C=fixed_c,
                    class_weight=None,
                    max_iter=1000,
                    random_state=self.random_state,
                )
                estimator.fit(x_train[:, selected], y_train)
                val_prob = estimator.predict_proba(x_val[:, selected])[:, 1]
                val_loss = float(log_loss(y_val, val_prob, labels=[0, 1]))
            except ValueError:
                continue
            components.append(
                _A3CComponent(
                    selected=selected,
                    estimator=estimator,
                    val_loss=val_loss,
                    val_prob=val_prob,
                    coef=np.ravel(estimator.coef_).astype(float),
                    intercept=float(np.ravel(estimator.intercept_)[0]),
                )
            )

        if not components:
            fallback = np.zeros(x_train.shape[1], dtype=bool)
            fallback[0] = True
            self._fit_policy_ensemble(x_train, y_train, x_val, y_val, [fallback])
            return

        components.sort(key=lambda component: component.val_loss)
        self.components_ = components[: min(fixed_ensemble_size, len(components))]
        raw_weights = np.array([1.0 / max(component.val_loss, 1e-6) for component in self.components_])
        self.component_weights_ = raw_weights / raw_weights.sum()
        self.temperature_ = fixed_temperature
        self.shift_ = fixed_shift
        selected_union = sorted({idx for component in self.components_ for idx in component.selected})
        self.selected_ = selected_union
        self.estimator_ = self.components_[0].estimator
        self._weights = {}
        for component_weight, component in zip(self.component_weights_, self.components_):
            coef = np.ravel(component.estimator.coef_)
            for local_idx, rule_idx in enumerate(component.selected):
                self._weights[rule_idx] = self._weights.get(rule_idx, 0.0) + float(component_weight * coef[local_idx])

    def predict_proba(self, x):
        if not self.components_:
            return np.tile([1.0 - self.prior_, self.prior_], (len(x), 1))
        probs = []
        for component in self.components_:
            score = x[:, component.selected] @ component.coef + component.intercept
            probs.append(_sigmoid(score))
        prob = np.average(probs, axis=0, weights=self.component_weights_)
        prob = _sigmoid(_logit(prob) / self.temperature_ + self.shift_)
        return _safe_proba(prob)

    def selected_rule_indices(self) -> list[int]:
        return list(self.selected_)

    def rule_weight(self, rule_idx: int) -> float | None:
        return self._weights.get(rule_idx)


def build_models(
    random_state: int = 7,
    a3c_workers: int = 4,
    a3c_episodes: int = 18,
    a3c_steps: int = 8,
    a3c_rules: int = 220,
) -> list[tuple[str, RuleModel]]:
    return [
        (
            "RT_A3C_MLN",
            RTA3CMLN(
                workers=a3c_workers,
                episodes=a3c_episodes,
                n_steps=a3c_steps,
                controller_rules=a3c_rules,
                random_state=random_state,
            ),
        ),
        (
            "MaxEnt_MLN",
            SklearnRuleModel(
                LogisticRegression(
                    solver="liblinear",
                    C=1.0,
                    max_iter=600,
                    random_state=random_state,
                ),
                all_rules=True,
            ),
        ),
        (
            "PLL_MLN",
            SklearnRuleModel(
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=5e-5,
                    max_iter=1200,
                    tol=1e-4,
                    random_state=random_state,
                )
            ),
        ),
        (
            "L1_Sparse_MLN",
            SklearnRuleModel(
                LogisticRegression(
                    solver="liblinear",
                    penalty="l1",
                    C=0.28,
                    max_iter=600,
                    random_state=random_state,
                )
            ),
        ),
        ("BeamSearch_MLN", BeamSearchMLN(random_state=random_state)),
        (
            "Boosted_MLN",
            SklearnRuleModel(
                GradientBoostingClassifier(
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=2,
                    subsample=0.85,
                    random_state=random_state,
                )
            ),
        ),
        ("Online_SGD_MLN", OnlineSGDMLN(random_state=random_state)),
    ]
