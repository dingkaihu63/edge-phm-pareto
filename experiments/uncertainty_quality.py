"""Selective-prediction quality: MC uncertainty vs pseudo-uncertainty baselines."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_cmapss, load_ur3, load_xjtu

ROOT = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(ROOT, "..", "results", "predictions")
OUT = os.path.join(ROOT, "..", "results")

LOADERS = {
    "ur3": lambda: load_ur3(r"C:\Users\hu\Desktop\比赛", seed_rolling=False),
    "cmapss_fd001": lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD001", seed_rolling=False),
    "cmapss_fd003": lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD003", seed_rolling=False),
    "xjtu": lambda: load_xjtu(r"E:\datasets\XJTU-SY\original", r"E:\datasets\XJTU-SY\xjtu_features_full15.csv", seed_rolling=False),
}

MODELS = {
    "ur3": ["full", "proposed_lite", "lstm", "tcn", "timesnet"],
    "cmapss_fd001": ["full", "proposed_lite", "lstm", "tcn", "timesnet"],
    "cmapss_fd003": ["full", "proposed_lite", "lstm", "tcn", "timesnet"],
    "xjtu": ["full", "proposed_lite", "lstm", "tcn", "timesnet", "patchtst"],
}


def main() -> None:
    rows = []
    for ds, models in MODELS.items():
        data = LOADERS[ds]()
        y_val, y_test = data["y_val"], data["y_test"]
        for model in models:
            z = np.load(os.path.join(PRED, f"{ds}_{model}.npz"))
            pv, p = z["pv"], z["p"]
            if "stdv" in z and np.asarray(z["stdv"]).max() > 0:
                unc_v = np.asarray(z["stdv"])
                unc_t = np.asarray(z["std"])
                unc_type = "MC"
            else:
                unc_v = -np.abs(pv - 0.5)
                unc_t = -np.abs(p - 0.5)
                unc_type = "pseudo"
            for frac in (0.0, 0.10, 0.20):
                if frac == 0.0:
                    keep_v = np.ones(len(pv), dtype=bool)
                    keep_t = np.ones(len(p), dtype=bool)
                else:
                    q = float(np.quantile(unc_v, 1.0 - frac))
                    keep_v = unc_v <= q
                    keep_t = unc_t <= q
                if keep_v.sum() == 0 or keep_t.sum() == 0:
                    continue
                th, _ = calibrate_threshold(y_val[keep_v], pv[keep_v])
                m = evaluate_binary(y_test[keep_t], p[keep_t], th)
                rows.append({
                    "dataset": ds,
                    "model": model,
                    "uncertainty_type": unc_type,
                    "reject_frac": frac,
                    "coverage": float(keep_t.sum()) / len(y_test),
                    "f2": m["f2"],
                    "auc_roc": m["auc_roc"],
                    "auc_pr": m["auc_pr"],
                    "brier": m["brier"],
                    "ece": m["ece"],
                })
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "uncertainty_quality.csv")
    df.to_csv(path, index=False)
    print(df[df["reject_frac"].isin([0.0, 0.10, 0.20])][["dataset", "model", "uncertainty_type", "reject_frac", "coverage", "f2", "auc_roc", "brier", "ece"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()