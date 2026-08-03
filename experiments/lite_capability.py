"""Train Lite ablations and collect capability-value metrics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_cmapss, load_ur3, load_xjtu
from run_experiments_torch import DS_CONFIG, LITE_CONFIG
from torch_common import build_model, predict_proba, save_model, set_seed, train_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS_DIR = RESULTS / "models_torch"
PRED_DIR = RESULTS / "predictions"

DATASETS = ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]
VARIANTS = {
    "lite_w_o_attention": dict(attention="none", mc_dropout=True),
    "lite_w_o_mc_dropout": dict(attention="sigmoid", mc_dropout=False),
}


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


def build_lite_variant(ds: str, variant: str, time_steps: int, n_features: int, seed: int):
    cfg = {**DS_CONFIG[ds], **LITE_CONFIG[ds]}
    kwargs = VARIANTS[variant]
    return build_model(
        time_steps,
        n_features,
        attention=kwargs["attention"],
        mc_dropout=kwargs["mc_dropout"],
        dropout_rate=cfg["dropout"],
        attn_temperature=cfg["tau"],
        lstm_units_1=cfg["units1"],
        lstm_units_2=cfg["units2"],
        seed=seed,
    )


def run_ds(ds: str) -> None:
    data = load_dataset(ds)
    cfg = {**DS_CONFIG[ds], **LITE_CONFIG[ds]}
    rows: List[Dict[str, object]] = []
    for variant, kwargs in VARIANTS.items():
        for seed in range(1, 11):
            set_seed(seed)
            model = build_lite_variant(ds, variant, data["x_train"].shape[1], data["x_train"].shape[2], seed)
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
                balanced_sampling=cfg.get("balanced", False),
                grad_clip=cfg.get("grad_clip", 0.0),
            )
            pv = predict_proba(model, data["x_val"], batch_size=512)
            pt = predict_proba(model, data["x_test"], batch_size=512)
            threshold, _ = calibrate_threshold(data["y_val"], pv)
            m = evaluate_binary(data["y_test"], pt, threshold)
            rows.append(
                {
                    "dataset": ds,
                    "model": variant,
                    "seed": seed,
                    "params": sum(p.numel() for p in model.parameters()),
                    "threshold": threshold,
                    "f2": m["f2"],
                    "auc_roc": m["auc_roc"],
                    "auc_pr": m["auc_pr"],
                    "brier": m["brier"],
                    "ece": m["ece"],
                }
            )
            if seed == 1:
                save_model(model, str(MODELS_DIR / f"{ds}_{variant}.pt"))
                np.savez_compressed(
                    PRED_DIR / f"{ds}_{variant}.npz",
                    p=pt,
                    pv=pv,
                    y=data["y_test"],
                    y_val=data["y_val"],
                    threshold=threshold,
                )
        print(ds, variant, "done", flush=True)
    df = pd.DataFrame(rows)
    path = RESULTS / "lite_capability_seeds.csv"
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    args = parser.parse_args()
    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        run_ds(ds)


if __name__ == "__main__":
    main()