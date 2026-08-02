"""Streaming closed-loop alerting evaluation with unit-level lead-time metrics."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common_no_tf import calibrate_threshold
from prepare_data import load_cmapss, load_ur3, load_xjtu

ROOT = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(ROOT, "..", "results", "predictions")
OUT = os.path.join(ROOT, "..", "results")

LOADERS = {
    "ur3": lambda: load_ur3(r"C:\Users\hu\Desktop\比赛", seed_rolling=False),
    "cmapss_fd001": lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD001", seed_rolling=False),
    "cmapss_fd003": lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD003", seed_rolling=False),
    "xjtu": lambda: load_xjtu(
        r"E:\datasets\XJTU-SY\original",
        r"E:\datasets\XJTU-SY\xjtu_features_full15.csv",
        seed_rolling=False,
    ),
}

MODELS = {
    "ur3": ["full", "proposed_lite", "lstm", "tcn", "timesnet"],
    "cmapss_fd001": ["full", "proposed_lite", "lstm", "tcn", "timesnet"],
    "cmapss_fd003": ["full", "proposed_lite", "lstm", "tcn", "timesnet"],
    "xjtu": ["full", "proposed_lite", "lstm", "tcn", "timesnet", "patchtst", "gradient_boosting"],
}

CONFIRM = [1, 2, 3]


def _calibrate_threshold_far(y_val, p_val, max_far=0.2):
    neg = max(int((y_val == 0).sum()), 1)
    best_t, best_cost = 0.5, float("inf")
    for t in np.arange(0.01, 0.99, 0.01):
        yp = (p_val >= t).astype(int)
        fn = int(((y_val == 1) & (yp == 0)).sum())
        fp = int(((y_val == 0) & (yp == 1)).sum())
        if fp / neg <= max_far:
            cost = 4 * fn + fp
            if cost < best_cost:
                best_cost, best_t = cost, float(t)
    return best_t


def _simulate(p_seq: np.ndarray, threshold: float, confirm: int) -> int:
    streak = 0
    for i in range(len(p_seq)):
        if p_seq[i] >= threshold:
            streak += 1
        else:
            streak = 0
        if streak >= confirm:
            return i
    return -1


def _sequence_metrics(ds: str, model: str, confirm: int, threshold_mode: str):
    data = LOADERS[ds]()
    z = np.load(os.path.join(PRED, f"{ds}_{model}.npz"))
    p = z["p"]
    pv = z["pv"]
    y = data["y_test"]
    units = data["unit_ids"]["test"]
    if threshold_mode == "far20":
        threshold = _calibrate_threshold_far(data["y_val"], pv, max_far=0.2)
    else:
        threshold, _ = calibrate_threshold(data["y_val"], pv)

    failure = 0
    detected = 0
    early = 0
    false_alarm = 0
    non_failure = 0
    leads = []

    for unit in np.unique(units):
        mask = units == unit
        p_seq = p[mask]
        y_seq = y[mask]
        pos = np.where(y_seq == 1)[0]
        is_failure = len(pos) > 0
        alarm = _simulate(p_seq, threshold, confirm)
        if is_failure:
            failure += 1
            onset = int(pos[0])
            if alarm >= 0:
                detected += 1
                leads.append(alarm - onset)
                if alarm <= onset:
                    early += 1
            else:
                pass
        else:
            non_failure += 1
            if alarm >= 0:
                false_alarm += 1

    lead_arr = np.array(leads, dtype=float) if leads else np.array([np.nan])
    return {
        "dataset": ds,
        "model": model,
        "threshold_mode": threshold_mode,
        "threshold": threshold,
        "confirm": confirm,
        "failure_sequences": failure,
        "detected": detected,
        "early": early,
        "missed": failure - detected,
        "false_alarms": false_alarm,
        "non_failure_sequences": non_failure,
        "detection_rate": detected / max(failure, 1),
        "early_rate": early / max(failure, 1),
        "false_alarm_rate": false_alarm / max(non_failure, 1),
        "mean_lead": float(np.nanmean(lead_arr)),
        "median_lead": float(np.nanmedian(lead_arr)),
    }


def main() -> None:
    rows = []
    for ds, models in MODELS.items():
        for model in models:
            for confirm in CONFIRM:
                for threshold_mode in ("cost", "far20"):
                    rows.append(_sequence_metrics(ds, model, confirm, threshold_mode))
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "closed_loop.csv")
    df.to_csv(path, index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()