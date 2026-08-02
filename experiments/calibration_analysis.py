"""Post-hoc calibration and MC-uncertainty selective prediction analysis."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_cmapss, load_ur3, load_xjtu

ROOT = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(ROOT, "..", "results", "predictions")
OUT = os.path.join(ROOT, "..", "results")

DATASETS = ["ur3", "cmapss_fd001", "cmapss_fd003", "xjtu"]
LOADERS = {
    "ur3": lambda: load_ur3(r"C:\Users\hu\Desktop\??", seed_rolling=False),
    "cmapss_fd001": lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD001", seed_rolling=False),
    "cmapss_fd003": lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD003", seed_rolling=False),
    "xjtu": lambda: load_xjtu(
        r"E:\datasets\XJTU-SY\original",
        r"E:\datasets\XJTU-SY\xjtu_features_full15.csv",
        seed_rolling=False,
    ),
}

MODELS = {
    "ur3": ["full", "proposed_lite"],
    "cmapss_fd001": ["full", "proposed_lite"],
    "cmapss_fd003": ["full", "proposed_lite"],
    "xjtu": ["full", "proposed_lite", "lstm", "tcn", "timesnet", "patchtst", "gradient_boosting"],
}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _temp_prob(p: np.ndarray, t: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_logit(p) / t))


def _fit_temperature(pv: np.ndarray, yv: np.ndarray) -> float:
    def nll(t: float) -> float:
        q = _temp_prob(pv, t)
        eps = 1e-7
        return float(
            -np.mean(yv * np.log(np.clip(q, eps, 1)) + (1 - yv) * np.log(np.clip(1 - q, eps, 1)))
        )

    res = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded", options={"xatol": 1e-3})
    return float(res.x)


def _eval_row(ds: str, model: str, method: str, pv, yv, p, y, coverage=1.0):
    threshold, _ = calibrate_threshold(yv, pv)
    m = evaluate_binary(y, p, threshold)
    return {
        "dataset": ds,
        "model": model,
        "method": method,
        "threshold": threshold,
        "f2": m["f2"],
        "auc_roc": m["auc_roc"],
        "auc_pr": m["auc_pr"],
        "brier": m["brier"],
        "ece": m["ece"],
        "coverage": coverage,
    }


def main() -> None:
    rows = []
    for ds in DATASETS:
        for model in MODELS[ds]:
            data = LOADERS[ds]()
            z = np.load(os.path.join(PRED, f"{ds}_{model}.npz"))
            pv, yv = z["pv"], data["y_val"]
            p, y = z["p"], data["y_test"]
            stdv = z["stdv"] if "stdv" in z else np.zeros_like(pv)
            std = z["std"] if "std" in z else np.zeros_like(p)

            rows.append(_eval_row(ds, model, "raw", pv, yv, p, y))

            t = _fit_temperature(pv, yv)
            rows.append(
                _eval_row(
                    ds, model, f"temperature (T={t:.2f})",
                    _temp_prob(pv, t), yv, _temp_prob(p, t), y,
                )
            )

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pv, yv)
            rows.append(
                _eval_row(ds, model, "isotonic", iso.predict(pv), yv, iso.predict(p), y)
            )

            if stdv.max() > 0:
                for frac in (0.10, 0.20, 0.30):
                    q = float(np.quantile(stdv, 1.0 - frac))
                    keep_v = stdv <= q
                    keep = std <= q
                    if keep.sum() == 0 or keep_v.sum() == 0:
                        continue
                    rows.append(
                        _eval_row(
                            ds, model, f"MC-reject {frac:.0%}",
                            pv[keep_v], yv[keep_v], p[keep], y[keep],
                            coverage=float(keep.sum()) / len(y),
                        )
                    )
                    t2 = _fit_temperature(pv[keep_v], yv[keep_v])
                    rows.append(
                        _eval_row(
                            ds, model, f"MC-reject {frac:.0%}+temp",
                            _temp_prob(pv[keep_v], t2), yv[keep_v],
                            _temp_prob(p[keep], t2), y[keep],
                            coverage=float(keep.sum()) / len(y),
                        )
                    )

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "calibration_ood.csv")
    df.to_csv(path, index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()