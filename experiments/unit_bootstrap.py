"""Unit-level block bootstrap for AUROC/AUPRC differences."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

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

BEST = {
    "ur3": "gradient_boosting",
    "cmapss_fd001": "transformer",
    "cmapss_fd003": "patchtst",
    "xjtu": "tcn",
}


def main() -> None:
    rng = np.random.RandomState(123)
    rows = []
    for ds, loader in LOADERS.items():
        data = loader()
        units = data["unit_ids"]["test"]
        y = data["y_test"]
        zp = np.load(os.path.join(PRED, f"{ds}_full.npz"))
        zb = np.load(os.path.join(PRED, f"{ds}_{BEST[ds]}.npz"))
        p_prop, p_best = zp["p"], zb["p"]
        unit_ids = np.unique(units)
        unit_y = {u: y[units == u] for u in unit_ids}
        unit_pp = {u: p_prop[units == u] for u in unit_ids}
        unit_pb = {u: p_best[units == u] for u in unit_ids}
        diffs_auc, diffs_pr = [], []
        for _ in range(2000):
            sample = rng.choice(unit_ids, size=len(unit_ids), replace=True)
            yy, pp, bb = [], [], []
            for u in sample:
                yy.append(unit_y[u])
                pp.append(unit_pp[u])
                bb.append(unit_pb[u])
            yy = np.concatenate(yy)
            pp = np.concatenate(pp)
            bb = np.concatenate(bb)
            if yy.sum() == 0 or (yy == 0).sum() == 0:
                continue
            try:
                diffs_auc.append(roc_auc_score(yy, pp) - roc_auc_score(yy, bb))
                diffs_pr.append(average_precision_score(yy, pp) - average_precision_score(yy, bb))
            except ValueError:
                continue
        diffs_auc = np.array(diffs_auc)
        diffs_pr = np.array(diffs_pr)
        rows.append({
            "dataset": ds,
            "best_baseline": BEST[ds],
            "auc_diff_mean": diffs_auc.mean(),
            "auc_diff_ci_low": np.percentile(diffs_auc, 2.5),
            "auc_diff_ci_high": np.percentile(diffs_auc, 97.5),
            "auc_ci_includes_zero": (np.percentile(diffs_auc, 2.5) <= 0 <= np.percentile(diffs_auc, 97.5)),
            "pr_diff_mean": diffs_pr.mean(),
            "pr_diff_ci_low": np.percentile(diffs_pr, 2.5),
            "pr_diff_ci_high": np.percentile(diffs_pr, 97.5),
            "pr_ci_includes_zero": (np.percentile(diffs_pr, 2.5) <= 0 <= np.percentile(diffs_pr, 97.5)),
        })
        print(rows[-1])
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "unit_bootstrap.csv")
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()