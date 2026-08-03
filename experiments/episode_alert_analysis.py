"""Episode-level alert analysis with explicit out-of-horizon outcomes.

The first confirmed alert is classified as on-time, premature, missed, or a
false alert. An alert before the first positive window is never counted as a
successful detection. This keeps the streaming analysis aligned with the
window-label horizon used to calibrate the operating threshold.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from common_no_tf import calibrate_threshold
from five_seed_bootstrap import (
    DATASETS,
    MODELS,
    PROPOSED,
    RESULTS,
    SEEDS,
    get_model,
    load_dataset,
)
from torch_common import mc_predict, predict_proba

CONFIRM_WINDOWS = [1, 2, 3]
MC_SAMPLES = 50


def first_confirmed_alarm(probabilities: np.ndarray, threshold: float, confirm: int) -> int:
    streak = 0
    for index, probability in enumerate(probabilities):
        streak = streak + 1 if probability >= threshold else 0
        if streak >= confirm:
            return index
    return -1


def classify_episode(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    confirm: int,
) -> Dict[str, object]:
    alarm = first_confirmed_alarm(probabilities, threshold, confirm)
    positive_indices = np.flatnonzero(labels == 1)
    if len(positive_indices) == 0:
        return {
            "positive_episode": False,
            "on_time": 0,
            "premature": 0,
            "missed": 0,
            "false_alert": int(alarm >= 0),
            "delay": np.nan,
        }

    onset = int(positive_indices[0])
    if alarm < 0:
        outcome = "missed"
    elif alarm < onset:
        outcome = "premature"
    else:
        outcome = "on_time"
    return {
        "positive_episode": True,
        "on_time": int(outcome == "on_time"),
        "premature": int(outcome == "premature"),
        "missed": int(outcome == "missed"),
        "false_alert": 0,
        "delay": float(alarm - onset) if outcome == "on_time" else np.nan,
    }


def model_predictions(dataset: str, seed: int, data, model_name: str):
    model = get_model(dataset, model_name, seed, data)
    if model_name in PROPOSED:
        val_prob, _ = mc_predict(
            model, data["x_val"], samples=MC_SAMPLES, batch_size=512
        )
        test_prob, _ = mc_predict(
            model, data["x_test"], samples=MC_SAMPLES, batch_size=512
        )
    else:
        val_prob = predict_proba(model, data["x_val"], batch_size=512)
        test_prob = predict_proba(model, data["x_test"], batch_size=512)
    threshold, _ = calibrate_threshold(data["y_val"], val_prob)
    return test_prob, threshold


def analyze_dataset(dataset: str) -> pd.DataFrame:
    data = load_dataset(dataset)
    labels = np.asarray(data["y_test"])
    units = np.asarray(data["unit_ids"]["test"])
    rows: List[Dict[str, object]] = []

    for seed in SEEDS:
        for model_name in MODELS:
            probabilities, threshold = model_predictions(dataset, seed, data, model_name)
            for confirm in CONFIRM_WINDOWS:
                outcomes = []
                for unit in np.unique(units):
                    mask = units == unit
                    outcomes.append(
                        classify_episode(
                            probabilities[mask], labels[mask], threshold, confirm
                        )
                    )
                positive = [outcome for outcome in outcomes if outcome["positive_episode"]]
                negative = [outcome for outcome in outcomes if not outcome["positive_episode"]]
                delays = [outcome["delay"] for outcome in positive if outcome["on_time"]]
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model": model_name,
                        "confirm_windows": confirm,
                        "threshold": threshold,
                        "positive_units": len(positive),
                        "negative_units": len(negative),
                        "on_time_rate": sum(x["on_time"] for x in positive) / max(len(positive), 1),
                        "premature_rate": sum(x["premature"] for x in positive) / max(len(positive), 1),
                        "missed_rate": sum(x["missed"] for x in positive) / max(len(positive), 1),
                        "false_alert_rate": (
                            sum(x["false_alert"] for x in negative) / len(negative)
                            if negative
                            else np.nan
                        ),
                        "median_detection_delay": (
                            float(np.median(delays)) if delays else np.nan
                        ),
                    }
                )
        print(f"{dataset}: completed seed {seed}", flush=True)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "on_time_rate",
        "premature_rate",
        "missed_rate",
        "false_alert_rate",
        "median_detection_delay",
    ]
    grouped = frame.groupby(["dataset", "model", "confirm_windows"], as_index=False)
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    counts = grouped[["positive_units", "negative_units"]].first()
    return summary.merge(counts, on=["dataset", "model", "confirm_windows"])


def main() -> None:
    frame = pd.concat([analyze_dataset(dataset) for dataset in DATASETS], ignore_index=True)
    detail_path = RESULTS / "episode_alert_5seeds.csv"
    summary_path = RESULTS / "episode_alert_5seeds_summary.csv"
    frame.to_csv(detail_path, index=False)
    summary = summarize(frame)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)
    print(summary[summary["confirm_windows"] == 2].to_string(index=False))


if __name__ == "__main__":
    main()
