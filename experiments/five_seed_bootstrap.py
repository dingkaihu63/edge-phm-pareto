"""Crossed paired bootstrap over training seeds and physical test units.

The analysis treats engines, bearings, or production cycles as the independent
test units. Each bootstrap replicate resamples the five final training seeds and
one common set of physical units. The common unit sample preserves the crossed
design because every training seed predicts the same held-out physical units.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import fbeta_score

from common_no_tf import calibrate_threshold
from prepare_data import load_cmapss, load_ur3, load_xjtu
from run_experiments_torch import DS_CONFIG, LITE_CONFIG
from torch_common import (
    DEVICE,
    build_deep_baseline,
    build_model,
    mc_predict,
    predict_proba,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS_DIR = RESULTS / "models_torch"
DATASETS = ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]
MODELS = ["full", "proposed_lite", "lstm", "tcn", "timesnet"]
PROPOSED = ["full", "proposed_lite"]
BASELINES = ["lstm", "tcn", "timesnet"]
METRICS = ["f2"]
SEEDS = [1, 2, 3, 4, 5]
N_BOOT = 3000
MC_SAMPLES = 50


def dataset_dirs() -> Dict[str, str]:
    return {
        "ur3": os.environ.get("EDGE_PHM_UR3_DIR", ""),
        "cmapss": os.environ.get("EDGE_PHM_CMAPSS_DIR", ""),
        "xjtu": os.environ.get("EDGE_PHM_XJTU_DIR", ""),
        "xjtu_cache": os.environ.get("EDGE_PHM_XJTU_CACHE", ""),
    }


def load_dataset(ds: str):
    dirs = dataset_dirs()
    if ds == "ur3":
        return load_ur3(dirs["ur3"], seed_rolling=False)
    if ds == "cmapss_fd001":
        return load_cmapss(dirs["cmapss"], "FD001", seed_rolling=False)
    if ds == "cmapss_fd003":
        return load_cmapss(dirs["cmapss"], "FD003", seed_rolling=False)
    if ds == "xjtu":
        return load_xjtu(dirs["xjtu"], dirs["xjtu_cache"], seed_rolling=False)
    raise ValueError(ds)


def model_path(ds: str, model: str, seed: int) -> Path:
    suffix = "" if seed == 1 else f"_seed{seed}"
    return MODELS_DIR / f"{ds}_{model}{suffix}.pt"


def get_model(ds: str, model_name: str, seed: int, data):
    path = model_path(ds, model_name, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Final checkpoint is required for bootstrap analysis: {path}"
        )

    time_steps, n_features = data["x_val"].shape[1:]
    if model_name in PROPOSED:
        cfg = {**DS_CONFIG[ds]}
        if model_name == "proposed_lite":
            cfg.update(LITE_CONFIG[ds])
        model = build_model(
            time_steps,
            n_features,
            attention="sigmoid",
            mc_dropout=True,
            dropout_rate=cfg["dropout"],
            attn_temperature=cfg["tau"],
            lstm_units_1=cfg["units1"],
            lstm_units_2=cfg["units2"],
            seed=seed,
        )
    else:
        model = build_deep_baseline(model_name, time_steps, n_features, seed=seed)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def metric_set(y: np.ndarray, p: np.ndarray, threshold: float):
    if y.sum() == 0 or (y == 0).sum() == 0:
        return None
    predicted = (p >= threshold).astype(int)
    return {
        "f2": fbeta_score(y, predicted, beta=2, zero_division=0),
    }


def predict_seed(ds: str, seed: int, data, units: np.ndarray):
    predictions = {}
    thresholds = {}
    for model_name in MODELS:
        model = get_model(ds, model_name, seed, data)
        if model_name in PROPOSED:
            val_p, _ = mc_predict(
                model, data["x_val"], samples=MC_SAMPLES, batch_size=512
            )
            test_p, _ = mc_predict(
                model, data["x_test"], samples=MC_SAMPLES, batch_size=512
            )
        else:
            val_p = predict_proba(model, data["x_val"], batch_size=512)
            test_p = predict_proba(model, data["x_test"], batch_size=512)
        predictions[model_name] = test_p
        thresholds[model_name] = calibrate_threshold(data["y_val"], val_p)[0]

    y = np.asarray(data["y_test"])
    unit_data = []
    for unit in np.unique(units):
        mask = units == unit
        unit_data.append(
            {"y": y[mask], "p": {name: predictions[name][mask] for name in MODELS}}
        )
    return {"units": unit_data, "thresholds": thresholds}


def resample_seed(seed_record, unit_indices: np.ndarray):
    unit_data = seed_record["units"]
    y = np.concatenate([unit_data[index]["y"] for index in unit_indices])
    probabilities = {
        model: np.concatenate(
            [unit_data[index]["p"][model] for index in unit_indices]
        )
        for model in MODELS
    }
    return {
        model: metric_set(y, probabilities[model], seed_record["thresholds"][model])
        for model in MODELS
    }


def run_dataset(ds: str, rng_seed: int) -> pd.DataFrame:
    data = load_dataset(ds)
    units = np.asarray(data["unit_ids"]["test"])
    seed_records = []
    for seed in SEEDS:
        record = predict_seed(ds, seed, data, units)
        seed_records.append(record)
        print(f"{ds}: loaded seed {seed}", flush=True)

    distributions = {
        (proposed, baseline, metric): []
        for proposed in PROPOSED
        for baseline in BASELINES
        for metric in METRICS
    }
    rng = np.random.RandomState(rng_seed)
    accepted = 0
    while accepted < N_BOOT:
        sampled_seeds = rng.randint(0, len(SEEDS), size=len(SEEDS))
        sampled_units = rng.randint(
            0, len(seed_records[0]["units"]), size=len(seed_records[0]["units"])
        )
        sampled_values = [
            resample_seed(seed_records[index], sampled_units)
            for index in sampled_seeds
        ]
        if any(
            values[model] is None for values in sampled_values for model in MODELS
        ):
            continue
        for proposed in PROPOSED:
            for baseline in BASELINES:
                for metric in METRICS:
                    differences = [
                        values[proposed][metric] - values[baseline][metric]
                        for values in sampled_values
                    ]
                    distributions[(proposed, baseline, metric)].append(
                        float(np.mean(differences))
                    )
        accepted += 1

    rows: List[Dict[str, object]] = []
    for (proposed, baseline, metric), values in distributions.items():
        distribution = np.asarray(values)
        low, high = np.percentile(distribution, [2.5, 97.5])
        rows.append(
            {
                "dataset": ds,
                "proposed": proposed,
                "baseline": baseline,
                "metric": metric,
                "diff_mean": float(distribution.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "ci_includes_zero": bool(low <= 0 <= high),
                "n_bootstrap": accepted,
                "n_seeds": len(SEEDS),
                "n_units": len(np.unique(units)),
                "resampling_design": "crossed_seed_unit",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    frames = [
        run_dataset(dataset, rng_seed=20260803 + index)
        for index, dataset in enumerate(DATASETS)
    ]
    output = pd.concat(frames, ignore_index=True)
    path = RESULTS / "unit_bootstrap_crossed.csv"
    output.to_csv(path, index=False)
    print(path)
    print(output[output["metric"] == "f2"].to_string(index=False))


if __name__ == "__main__":
    main()
