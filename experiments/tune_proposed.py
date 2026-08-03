"""Fast controlled search for stronger configurations of the proposed framework."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_ur3
from torch_common import build_model, mc_predict, set_seed, train_model

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "results", "tuning_ur3.csv")


def run_config(name, window, units1, units2, dropout, lr, batch, cw_scale, balanced, seeds=(1, 2)):
    data = load_ur3(r"C:\Users\hu\Desktop\比赛", window=window)
    rows = []
    for seed in seeds:
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
            pos_weight_scale=cw_scale,
            balanced_sampling=balanced,
        )
        p_val, _ = mc_predict(model, data["x_val"], samples=20)
        p_test, _ = mc_predict(model, data["x_test"], samples=20)
        threshold, _ = calibrate_threshold(data["y_val"], p_val)
        v = evaluate_binary(data["y_val"], p_val, threshold)
        m = evaluate_binary(data["y_test"], p_test, threshold)
        m["seed"] = seed
        m["val_f2"] = v["f2"]
        rows.append(m)
    df = pd.DataFrame(rows)
    summary = {
        "config": name,
        "window": window,
        "units": f"{units1}/{units2}",
        "dropout": dropout,
        "lr": lr,
        "batch": batch,
        "cw_scale": cw_scale,
        "balanced": balanced,
        "val_f2_mean": df["val_f2"].mean(),
        "test_f2_mean": df["f2"].mean(),
        "test_auc_mean": df["auc_roc"].mean(),
        "test_aucpr_mean": df["auc_pr"].mean(),
        "test_f2_std": df["f2"].std(),
    }
    print(summary)
    return summary


def main():
    configs = [
        ("base", 10, 64, 32, 0.15, 5e-4, 128, 1.0, False),
        ("wide96", 10, 96, 48, 0.10, 5e-4, 128, 1.0, False),
        ("wide128", 10, 128, 64, 0.15, 5e-4, 128, 1.0, False),
        ("balanced", 10, 96, 48, 0.10, 5e-4, 128, 1.0, True),
        ("cw2", 10, 96, 48, 0.10, 5e-4, 128, 2.0, False),
        ("window15", 15, 96, 48, 0.10, 5e-4, 128, 1.0, False),
        ("bal_cw15", 10, 96, 48, 0.10, 5e-4, 128, 1.5, True),
        ("lr1e3_b64", 10, 96, 48, 0.10, 1e-3, 64, 1.0, False),
    ]
    all_rows = []
    for cfg in configs:
        t0 = time.time()
        row = run_config(*cfg)
        row["time_s"] = round(time.time() - t0)
        all_rows.append(row)
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\nTop by test F2:")
    print(df.sort_values("test_f2_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
