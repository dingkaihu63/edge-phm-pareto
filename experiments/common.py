"""Shared utilities for the journal-version experiments.

All datasets are reduced to the same supervised task: predict whether a
time window contains (or precedes) a failure event within a fixed horizon.
The same preprocessing, model zoo, threshold-calibration and metrics are used
for every dataset so that cross-benchmark comparisons are apples-to-apples.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def add_rolling_std_seeded(
    df_split: pd.DataFrame,
    seed_tail: pd.DataFrame,
    window: int,
    sensor_cols: List[str],
) -> pd.DataFrame:
    """Rolling std with carry-over seed, i.e. the proposed warm-start feature."""
    combined = pd.concat([seed_tail, df_split], ignore_index=True)
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


def build_model(
    time_steps: int,
    n_features: int,
    attention: str = "sigmoid",
    mc_dropout: bool = True,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dropout_rate: float = 0.15,
    attn_temperature: float = 0.5,
    attn_normalize: bool = True,
    seed: int = 42,
) -> tf.keras.Model:
    """Proposed model and its ablation variants.

    attention in {"sigmoid", "softmax", "none"}
    mc_dropout=False removes stochastic inference (dropout layers removed).
    """
    set_seed(seed)
    inp = tf.keras.Input(shape=(time_steps, n_features), name="input")
    x = tf.keras.layers.LSTM(
        lstm_units_1, return_sequences=True, name="lstm_1"
    )(inp)
    h = tf.keras.layers.LSTM(
        lstm_units_2, return_sequences=True, name="lstm_2"
    )(x)

    if attention == "sigmoid":
        score = tf.keras.layers.Dense(1, use_bias=True, name="attn_score")(h)
        if attn_temperature != 1.0:
            score = tf.keras.layers.Lambda(
                lambda s: s / attn_temperature, name="attn_temp"
            )(score)
        alpha = tf.keras.layers.Activation("sigmoid", name="attn_weights")(score)
        if attn_normalize:
            alpha = tf.keras.layers.Lambda(
                lambda a: a / (tf.reduce_sum(a, axis=1, keepdims=True) + 1e-8),
                name="attn_norm",
            )(alpha)
        context = tf.keras.layers.Lambda(
            lambda t: tf.reduce_sum(t[0] * t[1], axis=1), name="context"
        )([h, alpha])
    elif attention == "softmax":
        score = tf.keras.layers.Dense(1, use_bias=True, name="attn_score")(h)
        alpha = tf.keras.layers.Softmax(axis=1, name="attn_weights")(score)
        context = tf.keras.layers.Lambda(
            lambda t: tf.reduce_sum(t[0] * t[1], axis=1), name="context"
        )([h, alpha])
    elif attention == "none":
        context = tf.keras.layers.Lambda(
            lambda t: t[:, -1, :], name="last_hidden"
        )(h)
    else:
        raise ValueError(f"unknown attention type: {attention}")

    if mc_dropout:
        x = tf.keras.layers.Dropout(dropout_rate, name="mc_drop_1")(
            context, training=True
        )
    else:
        x = context
    x = tf.keras.layers.Dense(16, activation="relu", name="dense_hidden")(x)
    if mc_dropout:
        x = tf.keras.layers.Dropout(dropout_rate * 0.5, name="mc_drop_2")(
            x, training=True
        )
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="SigAttn_MC_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc_pr", curve="PR")],
    )
    return model


def build_attention_model(model: tf.keras.Model) -> tf.keras.Model:
    """Dual-output model returning (prediction, attention weights)."""
    inp = model.input
    h = model.get_layer("lstm_2").output
    score = model.get_layer("attn_score")(h)
    if "attn_temp" in [l.name for l in model.layers]:
        score = model.get_layer("attn_temp")(score)
    alpha = model.get_layer("attn_weights")(score)
    if "attn_norm" in [l.name for l in model.layers]:
        alpha = model.get_layer("attn_norm")(alpha)
    context = tf.keras.layers.Lambda(
        lambda t: tf.reduce_sum(t[0] * t[1], axis=1), name="context"
    )([h, alpha])
    x = model.get_layer("dense_hidden")(context)
    out = model.get_layer("output")(x)
    return tf.keras.Model(inputs=inp, outputs=[out, alpha], name="attention_model")


def build_deep_baseline(
    kind: str,
    time_steps: int,
    n_features: int,
    seed: int = 42,
) -> tf.keras.Model:
    """Deterministic deep baselines with the same capacity budget."""
    set_seed(seed)
    inp = tf.keras.Input(shape=(time_steps, n_features))
    if kind == "lstm":
        x = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
        x = tf.keras.layers.LSTM(32, return_sequences=False)(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
    elif kind == "bilstm":
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True)
        )(inp)
        x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
        score = tf.keras.layers.Dense(1)(x)
        alpha = tf.keras.layers.Softmax(axis=1)(score)
        x = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1))(
            [x, alpha]
        )
        x = tf.keras.layers.Dense(16, activation="relu")(x)
    elif kind == "gru":
        x = tf.keras.layers.GRU(64, return_sequences=True)(inp)
        x = tf.keras.layers.GRU(32, return_sequences=True)(x)
        score = tf.keras.layers.Dense(1)(x)
        alpha = tf.keras.layers.Softmax(axis=1)(score)
        x = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1))(
            [x, alpha]
        )
        x = tf.keras.layers.Dense(16, activation="relu")(x)
    elif kind == "transformer":
        x = tf.keras.layers.Dense(64, activation="relu")(inp)
        attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ffn = tf.keras.layers.Dense(64, activation="relu")(x)
        ffn = tf.keras.layers.Dense(64)(ffn)
        x = tf.keras.layers.LayerNormalization()(x + ffn)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
    elif kind == "tcn":
        x = inp
        for i, dilation in enumerate([1, 2, 4, 8, 8]):
            y = tf.keras.layers.Conv1D(
                32, 3, padding="causal", dilation_rate=dilation, activation="relu"
            )(x)
            y = tf.keras.layers.Conv1D(
                32, 3, padding="causal", dilation_rate=dilation
            )(y)
            if i == 0:
                x = y
            else:
                x = tf.keras.layers.Add()([x, y])
            x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
    else:
        raise ValueError(kind)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    use_class_weight: bool = True,
    epochs: int = 80,
    batch_size: int = 128,
    verbose: int = 0,
) -> tf.keras.callbacks.History:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0
        ),
    ]
    class_weight = class_weight_from_y(y_train) if use_class_weight else None
    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose,
    )


def mc_predict(
    model: tf.keras.Model,
    x: np.ndarray,
    samples: int = 50,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.stack(
        [
            model.predict(x, batch_size=batch_size, verbose=0).ravel()
            for _ in range(samples)
        ],
        axis=0,
    )
    return preds.mean(axis=0), preds.std(axis=0)


def calibrate_threshold(
    y_val: np.ndarray, p_val: np.ndarray, beta: float = 2.0
) -> Tuple[float, float]:
    """Minimise the validation cost ``beta^2 * FN + FP`` (F-beta-aligned)."""
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
    auc_pr = (
        average_precision_score(y_true, p_pred) if pos and neg else float("nan")
    )
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


def flatten_windows(
    x: np.ndarray,
) -> np.ndarray:
    return x.reshape(x.shape[0], -1).astype(np.float32)


def run_ml_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

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

    gbm = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        random_state=seed,
    )
    gbm.fit(x_train, y_train, sample_weight=sample_weight)
    out["gradient_boosting"] = {
        "val": gbm.predict_proba(x_val)[:, 1],
        "test": gbm.predict_proba(x_test)[:, 1],
    }
    return out
