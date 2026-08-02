"""PyTorch multi-seed experiment runner."""

from __future__ import annotations

import argparse
import json
import os
import time
import torch
from typing import Dict, List

import numpy as np
import pandas as pd

from common_no_tf import (
    calibrate_threshold,
    class_weight_from_y,
    evaluate_binary,
    flatten_windows,
    run_ml_baselines,
)
from prepare_data import load_cmapss, load_ur3, load_xjtu
from torch_common import (
    build_deep_baseline,
    build_model,
    mc_predict,
    predict_proba,
    save_model,
    set_seed,
    train_model,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "results")
MODELS_DIR = os.path.join(OUT, "models_torch")
PRED_DIR = os.path.join(OUT, "predictions")
SEEDS_CSV = os.path.join(OUT, "results_seeds.csv")
OVERALL_CSV = os.path.join(OUT, "results_overall.csv")


def ensure_dirs() -> None:
    for d in (OUT, MODELS_DIR, PRED_DIR):
        os.makedirs(d, exist_ok=True)


def save_config(seeds: int) -> None:
    cfg = {
        "seeds": seeds,
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ds_config": DS_CONFIG,
    }
    with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def load_dataset(name: str) -> Dict[str, object]:
    seed_rolling = name.endswith("__rolling")
    clean = name.replace("__rolling", "")
    if clean == "ur3":
        return load_ur3(r"C:\Users\hu\Desktop\比赛", seed_rolling=seed_rolling)
    if clean == "cmapss_fd001":
        return load_cmapss(
            r"E:\datasets\C-MAPSS", fd="FD001", seed_rolling=seed_rolling
        )
    if clean == "cmapss_fd003":
        return load_cmapss(
            r"E:\datasets\C-MAPSS", fd="FD003", seed_rolling=seed_rolling
        )
    if clean == "xjtu":
        return load_xjtu(
            r"E:\datasets\XJTU-SY\original",
            r"E:\datasets\XJTU-SY\xjtu_features_full15.csv",
            seed_rolling=seed_rolling,
        )
    raise ValueError(clean)


ABLATIONS = [
    ("full", dict(attention="sigmoid", mc_dropout=True), True, False),
    ("w/o_attention", dict(attention="none", mc_dropout=True), True, False),
    ("softmax_attention", dict(attention="softmax", mc_dropout=True), True, False),
    ("w/o_mc_dropout", dict(attention="sigmoid", mc_dropout=False), True, False),
    ("w/o_class_weight", dict(attention="sigmoid", mc_dropout=True), False, False),
    ("w/_rolling_stats", dict(attention="sigmoid", mc_dropout=True), True, True),
    ("proposed_lite", dict(attention="sigmoid", mc_dropout=True, lite=True), True, False),
]

DEEP_BASELINES = ["lstm", "bilstm", "gru", "transformer", "tcn", "patchtst", "timesnet"]

DS_CONFIG = {
    "ur3": dict(units1=96, units2=48, dropout=0.10, lr=1e-3, batch=64, balanced=False),
    "cmapss_fd001": dict(units1=64, units2=32, dropout=0.15, lr=5e-4, batch=128, balanced=False),
    "cmapss_fd003": dict(units1=64, units2=32, dropout=0.15, lr=5e-4, batch=128, balanced=False),
    "xjtu": dict(units1=96, units2=48, dropout=0.10, lr=1e-3, batch=64, balanced=True),
}

LITE_CONFIG = {
    "ur3": dict(units1=48, units2=24, dropout=0.10, lr=1e-3, batch=64, balanced=False),
    "cmapss_fd001": dict(units1=32, units2=16, dropout=0.15, lr=5e-4, batch=128, balanced=False),
    "cmapss_fd003": dict(units1=32, units2=16, dropout=0.15, lr=5e-4, batch=128, balanced=False),
    "xjtu": dict(units1=48, units2=24, dropout=0.10, lr=1e-3, batch=64, balanced=True),
}


def safe(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")

PRED_STORE = {}


def store_preds(
    ds: str,
    model: str,
    p: np.ndarray,
    std: np.ndarray,
    y: np.ndarray,
    pv: np.ndarray,
    y_val: np.ndarray,
    std_val: np.ndarray,
) -> None:
    key = (ds, model)
    if key not in PRED_STORE:
        PRED_STORE[key] = {"p": [], "std": [], "y": y, "pv": [], "y_val": y_val, "stdv": []}
    PRED_STORE[key]["p"].append(p)
    PRED_STORE[key]["std"].append(std)
    PRED_STORE[key]["pv"].append(pv)
    PRED_STORE[key]["stdv"].append(std_val)


def save_aggregate_preds() -> None:
    for (ds, model), vals in PRED_STORE.items():
        p = np.mean(np.stack(vals["p"]), axis=0)
        std = np.mean(np.stack(vals["std"]), axis=0)
        pv = np.mean(np.stack(vals["pv"]), axis=0)
        stdv = np.mean(np.stack(vals["stdv"]), axis=0)
        np.savez_compressed(
            os.path.join(PRED_DIR, f"{ds}_{safe(model)}.npz"),
            p=p,
            std=std,
            y=vals["y"],
            pv=pv,
            stdv=stdv,
        )


def run_one_deep(
    ds: str,
    model_name: str,
    data: Dict[str, object],
    build_kwargs: Dict[str, object],
    use_cw: bool,
    mc: bool,
    seed: int,
    save_first: bool,
) -> Dict[str, float]:
    set_seed(seed)
    cfg = DS_CONFIG[ds]
    if build_kwargs.get("lite"):
        cfg = LITE_CONFIG[ds]
        build_kwargs = {k: v for k, v in build_kwargs.items() if k != "lite"}
    model = build_model(
        data["x_train"].shape[1],
        data["x_train"].shape[2],
        lstm_units_1=cfg["units1"],
        lstm_units_2=cfg["units2"],
        dropout_rate=cfg["dropout"],
        seed=seed,
        **build_kwargs,
    )
    train_model(
        model,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        use_class_weight=use_cw,
        lr=cfg["lr"],
        batch_size=cfg["batch"],
        seed=seed,
        balanced_sampling=cfg.get("balanced", False),
    )
    if mc:
        p_val, std_val = mc_predict(model, data["x_val"], samples=50)
        p_test, std_test = mc_predict(model, data["x_test"], samples=50)
    else:
        p_val = predict_proba(model, data["x_val"])
        p_test = predict_proba(model, data["x_test"])
        std_test = np.zeros_like(p_test)
        std_val = np.zeros_like(p_val)
    threshold, _ = calibrate_threshold(data["y_val"], p_val)
    metrics = evaluate_binary(data["y_test"], p_test, threshold)
    store_preds(ds, model_name, p_test, std_test, data["y_test"], p_val, data["y_val"], std_val)
    if save_first:
        save_model(model, os.path.join(MODELS_DIR, f"{ds}_{safe(model_name)}.pt"))
    return {
        "model": model_name,
        "family": "proposed/ablation",
        "seed": seed,
        **metrics,
    }


def run_baseline_deep(
    ds: str,
    model_name: str,
    data: Dict[str, object],
    seed: int,
    save_first: bool,
) -> Dict[str, float]:
    set_seed(seed)
    cfg = DS_CONFIG[ds]
    model = build_deep_baseline(
        model_name, data["x_train"].shape[1], data["x_train"].shape[2], seed=seed
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
        balanced_sampling=cfg.get("balanced", False),
    )
    p_val = predict_proba(model, data["x_val"])
    p_test = predict_proba(model, data["x_test"])
    threshold, _ = calibrate_threshold(data["y_val"], p_val)
    metrics = evaluate_binary(data["y_test"], p_test, threshold)
    store_preds(ds, model_name, p_test, np.zeros_like(p_test), data["y_test"], p_val, data["y_val"], np.zeros_like(p_val))
    if save_first:
        save_model(model, os.path.join(MODELS_DIR, f"{ds}_{safe(model_name)}.pt"))
    return {
        "model": model_name,
        "family": "deep baseline",
        "seed": seed,
        **metrics,
    }


def run_ml(
    ds: str,
    data: Dict[str, object],
    seed: int,
) -> List[Dict[str, float]]:
    x_tr = flatten_windows(data["x_train"])
    x_va = flatten_windows(data["x_val"])
    x_te = flatten_windows(data["x_test"])
    cw = class_weight_from_y(data["y_train"])
    sample_weight = np.where(data["y_train"] == 1, cw[1], cw[0]).astype(np.float32)
    preds = run_ml_baselines(x_tr, data["y_train"], x_va, x_te, sample_weight, seed)
    rows = []
    for name, probs in preds.items():
        threshold, _ = calibrate_threshold(data["y_val"], probs["val"])
        metrics = evaluate_binary(data["y_test"], probs["test"], threshold)
        store_preds(
            ds,
            name,
            probs["test"],
            np.zeros_like(probs["test"]),
            data["y_test"],
            probs["val"],
            data["y_val"],
            np.zeros_like(probs["val"]),
        )
        row = {"model": name, "family": "machine learning", "seed": seed, **metrics}
        row["dataset"] = ds
        rows.append(row)
    return rows

METRIC_COLS = [
    "n", "pos", "neg", "accuracy", "precision", "recall", "f1", "f2",
    "auc_roc", "auc_pr", "brier", "ece", "threshold",
]


def build_ensemble_csv() -> None:
    rows = []
    for (ds, model), vals in PRED_STORE.items():
        pv = np.mean(np.stack(vals["pv"]), axis=0)
        p = np.mean(np.stack(vals["p"]), axis=0)
        y_val = vals["y_val"]
        threshold, _ = calibrate_threshold(y_val, pv)
        metrics = evaluate_binary(vals["y"], p, threshold)
        rows.append({"dataset": ds, "model": model, **metrics})
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "results_ensemble.csv")
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old[["dataset", "model"]].apply(tuple, axis=1).isin(
            set(df[["dataset", "model"]].apply(tuple, axis=1))
        )]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby(["dataset", "model", "family"], as_index=False)
    agg = groups.agg({c: ["mean", "std"] for c in METRIC_COLS})
    agg.columns = [
        "_".join(x).strip("_") if isinstance(x, tuple) else x
        for x in agg.columns
    ]
    rename = {
        f"{c}_mean": c for c in METRIC_COLS
    }
    agg = agg.rename(columns=rename)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="ur3,cmapss_fd001,cmapss_fd003,xjtu")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--ml-only", action="store_true")
    parser.add_argument("--models", default="")
    args = parser.parse_args()
    ensure_dirs()
    save_config(args.seeds)

    all_rows = []
    t_start = time.time()
    for ds_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        print(f"\n########## {ds_name} ##########", flush=True)
        data = load_dataset(ds_name)
        print(
            f"windows {len(data['y_train'])}/{len(data['y_val'])}/{len(data['y_test'])}, "
            f"pos {data['y_train'].sum()}/{data['y_val'].sum()}/{data['y_test'].sum()}",
            flush=True,
        )
        seeds = list(range(1, args.seeds + 1))
        run_deep = not args.ml_only
        do_ml = (not args.skip_ml) and (not args.ml_only) or args.ml_only
        only_models = [x.strip() for x in args.models.split(",") if x.strip()]
        if run_deep:
            for model_name, kwargs, use_cw, seed_rolling in ABLATIONS:
                if only_models and model_name not in only_models:
                    continue
                data_use = load_dataset(ds_name + "__rolling") if seed_rolling else data
                for seed in seeds:
                    row = run_one_deep(
                        ds_name,
                        model_name,
                        data_use,
                        kwargs,
                        use_cw,
                        kwargs.get("mc_dropout", True),
                        seed,
                        save_first=(seed == seeds[0]),
                    )
                    row["dataset"] = ds_name
                    all_rows.append(row)
                print(f"  {model_name} done ({time.time()-t_start:.0f}s)", flush=True)

        if run_deep:
            for model_name in DEEP_BASELINES:
                if only_models and model_name not in only_models:
                    continue
                for seed in seeds:
                    row = run_baseline_deep(
                        ds_name, model_name, data, seed, save_first=(seed == seeds[0])
                    )
                    row["dataset"] = ds_name
                    all_rows.append(row)
                print(f"  baseline {model_name} done ({time.time()-t_start:.0f}s)", flush=True)

        if do_ml:
            if only_models:
                for seed in seeds:
                    for row in run_ml(ds_name, data, seed):
                        if row["model"] in only_models:
                            all_rows.append(row)
            else:
                for seed in seeds:
                    all_rows.extend(run_ml(ds_name, data, seed))
            print(f"  ML done ({time.time()-t_start:.0f}s)", flush=True)

    seed_df = pd.DataFrame(all_rows)
    if os.path.exists(SEEDS_CSV):
        existing = pd.read_csv(SEEDS_CSV)
        seed_df = pd.concat([existing, seed_df], ignore_index=True)
        key = ["dataset", "model", "seed"]
        seed_df = seed_df.drop_duplicates(subset=key, keep="last")
    seed_df.to_csv(SEEDS_CSV, index=False)
    overall = aggregate_seeds(seed_df)
    if os.path.exists(OVERALL_CSV):
        old = pd.read_csv(OVERALL_CSV)
        old = old[~old[["dataset", "model"]].apply(tuple, axis=1).isin(
            set(overall[["dataset", "model"]].apply(tuple, axis=1))
        )]
        overall = pd.concat([old, overall], ignore_index=True)
    overall.to_csv(OVERALL_CSV, index=False)
    save_aggregate_preds()
    build_ensemble_csv()
    print(f"\nSaved {len(seed_df)} seed rows and {len(overall)} aggregated rows")
    print(
        overall.groupby(["dataset", "family"])[["f2", "auc_roc", "auc_pr"]]
        .mean()
        .round(3)
    )


if __name__ == "__main__":
    main()
