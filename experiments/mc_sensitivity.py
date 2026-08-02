"""MC-Dropout sample-count sensitivity for the proposed models."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_cmapss, load_ur3, load_xjtu
from torch_common import build_model, mc_predict

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "results")
MODELS = os.path.join(OUT, "models_torch")

LOADERS = {
    "ur3": lambda: load_ur3(r"C:\Users\hu\Desktop\比赛", seed_rolling=False),
    "cmapss_fd001": lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD001", seed_rolling=False),
    "cmapss_fd003": lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD003", seed_rolling=False),
    "xjtu": lambda: load_xjtu(r"E:\datasets\XJTU-SY\original", r"E:\datasets\XJTU-SY\xjtu_features_full15.csv", seed_rolling=False),
}

CFG = {
    "ur3": dict(lstm_units_1=96, lstm_units_2=48, dropout_rate=0.1),
    "cmapss_fd001": dict(lstm_units_1=64, lstm_units_2=32, dropout_rate=0.15),
    "cmapss_fd003": dict(lstm_units_1=64, lstm_units_2=32, dropout_rate=0.15),
    "xjtu": dict(lstm_units_1=96, lstm_units_2=48, dropout_rate=0.1),
}

LITE = {
    "ur3": dict(lstm_units_1=48, lstm_units_2=24, dropout_rate=0.1),
    "cmapss_fd001": dict(lstm_units_1=32, lstm_units_2=16, dropout_rate=0.15),
    "cmapss_fd003": dict(lstm_units_1=32, lstm_units_2=16, dropout_rate=0.15),
    "xjtu": dict(lstm_units_1=48, lstm_units_2=24, dropout_rate=0.1),
}


def main() -> None:
    rows = []
    for ds, loader in LOADERS.items():
        data = loader()
        for variant, cfg in (("full", CFG), ("proposed_lite", LITE)):
            model = build_model(data["x_train"].shape[1], data["x_train"].shape[2], **cfg[ds])
            model.load_state_dict(torch.load(os.path.join(MODELS, f"{ds}_{variant}.pt"), map_location="cpu"))
            model.eval()
            for k in (10, 20, 50):
                pv, _ = mc_predict(model, data["x_val"], samples=k)
                pt, _ = mc_predict(model, data["x_test"], samples=k)
                th, _ = calibrate_threshold(data["y_val"], pv)
                m = evaluate_binary(data["y_test"], pt, th)
                rows.append({"dataset": ds, "model": variant, "mc_samples": k, **m})
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "mc_sensitivity.csv")
    df.to_csv(path, index=False)
    print(df[["dataset", "model", "mc_samples", "f2", "auc_roc", "brier", "ece"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()