from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def evaluate_classifier(model, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    start = time.perf_counter()
    prob = model.predict_proba(x_test)[:, 1]
    latency = time.perf_counter() - start
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 6),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 6),
        "log_loss": round(float(log_loss(y_test, prob, labels=[0, 1])), 6),
        "latency_ms_per_sample": round(float(latency * 1000.0 / max(1, len(y_test))), 6),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, prob)), 6)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics

