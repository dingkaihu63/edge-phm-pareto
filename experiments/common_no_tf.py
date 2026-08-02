"""TensorFlow-free utilities used by the PyTorch experiment pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def add_rolling_std_seeded(
    df_split: pd.DataFrame,
    seed_tail: pd.DataFrame,
    window: int,
    sensor_cols: List[str],
) -> pd.DataFrame:
    combined = (
        df_split.copy() if len(seed_tail) == 0
        else pd.concat([seed_tail, df_split], ignore_index=True)
    )
    for col in sensor_cols:
        combined[col + "_rstd"] = (
            combined[col].rolling(window=window, min_periods=1).std().fillna(0.0)
        )
    return combined.iloc[len(seed_tail):].reset_index(drop=True)


def make_windows(
    x: np.ndarray, y: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n < window:
        raise ValueError(f"sequence length {n} < window {window}")
    idx = np.arange(window)[None, :] + np.arange(n - window + 1)[:, None]
    return x[idx].astype(np.float32), (y[idx].max(axis=1) > 0).astype(np.int32)


def split_chronological(
    df: pd.DataFrame,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
        y[:train_end],
        y[train_end:val_end],
        y[val_end:],
    )


def class_weight_from_y(y: np.ndarray) -> Dict[int, float]:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return {0: 1.0, 1: 1.0}
    return {0: 1.0, 1: float(neg) / float(pos)}


def scale_train_val_test(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(train).astype(np.float32)
    x_va = scaler.transform(val).astype(np.float32)
    x_te = scaler.transform(test).astype(np.float32)
    return x_tr, x_va, x_te, scaler


def calibrate_threshold(
    y_val: np.ndarray, p_val: np.ndarray, beta: float = 2.0
) -> Tuple[float, float]:
    miss_weight = float(beta ** 2)
    best_t, best_cost = 0.5, float("inf")
    for t in np.arange(0.01, 0.99, 0.01):
        y_pred = (p_val >= t).astype(int)
        fn = int(((y_val == 1) & (y_pred == 0)).sum())
        fp = int(((y_val == 0) & (y_pred == 1)).sum())
        cost = miss_weight * fn + fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t, best_cost


def expected_calibration_error(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_pred >= lo) & (p_pred < hi)
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = p_pred[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def evaluate_binary(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    threshold: float,
    beta: float = 2.0,
) -> Dict[str, float]:
    y_pred = (p_pred >= threshold).astype(int)
    n = len(y_true)
    pos = int(y_true.sum())
    neg = int(n - pos)
    auc_roc = roc_auc_score(y_true, p_pred) if pos and neg else float("nan")
    auc_pr = average_precision_score(y_true, p_pred) if pos and neg else float("nan")
    brier = float(brier_score_loss(y_true, p_pred))
    ece = expected_calibration_error(y_true, p_pred, n_bins=10)
    return {
        "n": n,
        "pos": pos,
        "neg": neg,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "brier": brier,
        "ece": ece,
        "threshold": threshold,
    }


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).astype(np.float32)


def run_ml_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    out = {}
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
    )
    rf.fit(x_train, y_train)
    out["random_forest"] = {
        "val": rf.predict_proba(x_val)[:, 1],
        "test": rf.predict_proba(x_test)[:, 1],
    }
    gbm = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=16,
        random_state=seed,
    )
    gbm.fit(x_train, y_train, sample_weight=sample_weight)
    out["gradient_boosting"] = {
        "val": gbm.predict_proba(x_val)[:, 1],
        "test": gbm.predict_proba(x_test)[:, 1],
    }
    return out
