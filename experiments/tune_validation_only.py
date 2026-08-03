"""Validation-only search for non-architectural training improvements."""

from __future__ import annotations

import argparse
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_cmapss, load_ur3, load_xjtu
from run_experiments_torch import DS_CONFIG, find_dataset_dir
from torch_common import build_model, mc_predict, set_seed, train_model


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "results", "tuning_validation_v2.csv")


SEARCH = {
    "ur3": [
        ("base", 10, 0.5, 0.10, False, 0.0),
        ("tau025", 10, 0.25, 0.10, False, 0.0),
        ("tau100", 10, 1.0, 0.10, False, 0.0),
        ("drop005", 10, 0.5, 0.05, False, 0.0),
        ("drop020", 10, 0.5, 0.20, False, 0.0),
        ("clip1", 10, 0.5, 0.10, False, 1.0),
        ("window15", 15, 0.5, 0.10, False, 0.0),
        ("window15_tau025", 15, 0.25, 0.10, False, 0.0),
        ("window15_clip1", 15, 0.5, 0.10, False, 1.0),
    ],
    "cmapss_fd001": [
        ("base", 30, 0.5, 0.15, False, 0.0),
        ("tau025", 30, 0.25, 0.15, False, 0.0),
        ("tau100", 30, 1.0, 0.15, False, 0.0),
        ("drop005", 30, 0.5, 0.05, False, 0.0),
        ("drop010", 30, 0.5, 0.10, False, 0.0),
        ("clip1", 30, 0.5, 0.15, False, 1.0),
    ],
    "cmapss_fd003": [
        ("base", 30, 0.5, 0.15, False, 0.0),
        ("tau025", 30, 0.25, 0.15, False, 0.0),
        ("tau100", 30, 1.0, 0.15, False, 0.0),
        ("drop005", 30, 0.5, 0.05, False, 0.0),
        ("drop010", 30, 0.5, 0.10, False, 0.0),
        ("clip1", 30, 0.5, 0.15, False, 1.0),
    ],
    "xjtu": [
        ("balanced", 20, 0.5, 0.10, True, 0.0),
        ("weighted", 20, 0.5, 0.10, False, 0.0),
        ("balanced_tau025", 20, 0.25, 0.10, True, 0.0),
        ("balanced_tau100", 20, 1.0, 0.10, True, 0.0),
        ("balanced_drop005", 20, 0.5, 0.05, True, 0.0),
        ("balanced_drop020", 20, 0.5, 0.20, True, 0.0),
        ("balanced_clip1", 20, 0.5, 0.10, True, 1.0),
        ("balanced_window30", 30, 0.5, 0.10, True, 0.0),
        ("weighted_window30", 30, 0.5, 0.10, False, 0.0),
    ],
}


@lru_cache(maxsize=None)
def load_data(dataset: str, window: int):
    if dataset == "ur3":
        root = find_dataset_dir("dataset_02052023.xlsx", "EDGE_PHM_UR3_DIR")
        return load_ur3(root, window=window, seed_rolling=False)
    if dataset.startswith("cmapss_"):
        fd = dataset.split("_")[-1].upper()
        root = os.environ.get("EDGE_PHM_CMAPSS_DIR", r"E:\datasets\C-MAPSS")
        return load_cmapss(root, fd=fd, window=window, seed_rolling=False)
    if dataset == "xjtu":
        root = os.environ.get("EDGE_PHM_XJTU_DIR", r"E:\datasets\XJTU-SY\original")
        cache = os.environ.get(
            "EDGE_PHM_XJTU_CACHE", r"E:\datasets\XJTU-SY\xjtu_features_full15.csv"
        )
        return load_xjtu(root, cache, window=window, seed_rolling=False)
    raise ValueError(dataset)


def run_config(dataset: str, config: tuple, seeds: int, mc_samples: int) -> dict:
    name, window, tau, dropout, balanced, grad_clip = config
    data = load_data(dataset, window)
    cfg = DS_CONFIG[dataset]
    metrics = []
    for seed in range(1, seeds + 1):
        set_seed(seed)
        model = build_model(
            data["x_train"].shape[1],
            data["x_train"].shape[2],
            lstm_units_1=cfg["units1"],
            lstm_units_2=cfg["units2"],
            dropout_rate=dropout,
            attn_temperature=tau,
            seed=seed,
        )
        train_model(
            model,
            data["x_train"],
            data["y_train"],
            data["x_val"],
            data["y_val"],
            use_class_weight=True,
            lr=cfg["lr"],
            batch_size=cfg["batch"],
            seed=seed,
            balanced_sampling=balanced,
            grad_clip=grad_clip,
        )
        p_val, _ = mc_predict(model, data["x_val"], samples=mc_samples)
        threshold, _ = calibrate_threshold(data["y_val"], p_val)
        metrics.append(evaluate_binary(data["y_val"], p_val, threshold))

    frame = pd.DataFrame(metrics)
    return {
        "dataset": dataset,
        "config": name,
        "window": window,
        "tau": tau,
        "dropout": dropout,
        "balanced_sampling": balanced,
        "grad_clip": grad_clip,
        "seeds": seeds,
        "val_f2_mean": frame["f2"].mean(),
        "val_f2_std": frame["f2"].std(ddof=1),
        "val_auroc_mean": frame["auc_roc"].mean(),
        "val_auprc_mean": frame["auc_pr"].mean(),
        "val_ece_mean": frame["ece"].mean(),
        "threshold_mean": frame["threshold"].mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(SEARCH))
    parser.add_argument("--configs", default="")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--mc-samples", type=int, default=20)
    args = parser.parse_args()

    rows = []
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    selected = {x.strip() for x in args.configs.split(",") if x.strip()}
    for dataset in datasets:
        for config in SEARCH[dataset]:
            if selected and config[0] not in selected:
                continue
            row = run_config(dataset, config, args.seeds, args.mc_samples)
            rows.append(row)
            print(
                f"{dataset:14s} {row['config']:22s} "
                f"F2={row['val_f2_mean']:.3f} AUPRC={row['val_auprc_mean']:.3f}",
                flush=True,
            )

    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "val_f2_mean", "val_auprc_mean"],
        ascending=[True, False, False],
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"Saved validation-only search to {OUT}")


if __name__ == "__main__":
    main()
