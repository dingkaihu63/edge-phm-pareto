"""Test seed-ensembling for the proposed framework on UR3."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_ur3
from torch_common import build_model, mc_predict, set_seed, train_model


def ensemble_config(name, window, units1, units2, dropout, lr, batch, seeds=5):
    data = load_ur3(r"C:\Users\hu\Desktop\比赛", window=window)
    val_preds, test_preds = [], []
    for seed in range(1, seeds + 1):
        set_seed(seed)
        model = build_model(
            data["x_train"].shape[1],
            data["x_train"].shape[2],
            lstm_units_1=units1,
            lstm_units_2=units2,
            dropout_rate=dropout,
            seed=seed,
        )
        train_model(
            model,
            data["x_train"],
            data["y_train"],
            data["x_val"],
            data["y_val"],
            lr=lr,
            batch_size=batch,
            seed=seed,
        )
        pv, _ = mc_predict(model, data["x_val"], samples=50)
        pt, _ = mc_predict(model, data["x_test"], samples=50)
        val_preds.append(pv)
        test_preds.append(pt)
    pv = np.mean(val_preds, axis=0)
    pt = np.mean(test_preds, axis=0)
    threshold, _ = calibrate_threshold(data["y_val"], pv)
    m = evaluate_binary(data["y_test"], pt, threshold)
    print(name, "t", round(threshold, 2), "acc/prec/rec/f2/auc/pr", round(m["accuracy"], 3), round(m["precision"], 3), round(m["recall"], 3), round(m["f2"], 3), round(m["auc_roc"], 3), round(m["auc_pr"], 3))


def main():
    ensemble_config("w15_u96_lr5e-4", 15, 96, 48, 0.10, 5e-4, 128)
    ensemble_config("w10_u96_lr1e-3_b64", 10, 96, 48, 0.10, 1e-3, 64)
    ensemble_config("w20_u96_lr5e-4", 20, 96, 48, 0.10, 5e-4, 128)


if __name__ == "__main__":
    main()
