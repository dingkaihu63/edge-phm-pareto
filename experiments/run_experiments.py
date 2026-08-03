"""Run the complete cross-benchmark experiment matrix."""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from common import (
    build_deep_baseline,
    build_model,
    calibrate_threshold,
    class_weight_from_y,
    evaluate_binary,
    flatten_windows,
    mc_predict,
    run_ml_baselines,
    set_seed,
    train_model,
)
from prepare_data import load_cmapss, load_ur3, load_xjtu


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "results")
MODELS_DIR = os.path.join(OUT, "models")
PRED_DIR = os.path.join(OUT, "predictions")


def ensure_dirs() -> None:
    for d in (OUT, MODELS_DIR, PRED_DIR):
        os.makedirs(d, exist_ok=True)


def save_results(rows: List[Dict[str, object]]) -> None:
    path = os.path.join(OUT, "results_overall.csv")
    new_df = pd.DataFrame(rows)
    if os.path.exists(path):
        old_df = pd.read_csv(path)
        keys = ["dataset", "model"]
        if not new_df.empty and set(keys).issubset(new_df.columns):
            old_df = old_df[
                ~old_df.set_index(keys).index.isin(new_df.set_index(keys).index)
            ]
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


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
]

DEEP_BASELINES = ["lstm", "bilstm", "gru", "transformer", "tcn"]


def save_predictions(dataset: str, model: str, p: np.ndarray, std: np.ndarray, y: np.ndarray) -> None:
    safe = model.replace("/", "_").replace("\\", "_").replace(" ", "_")
    np.savez_compressed(
        os.path.join(PRED_DIR, f"{dataset}_{safe}.npz"), p=p, std=std, y=y
    )


def run_one_deep(
    dataset: str,
    model_name: str,
    data: Dict[str, object],
    build_kwargs: Dict[str, object],
    use_cw: bool,
    mc: bool,
) -> Dict[str, float]:
    set_seed(42)
    model = build_model(
        data["x_train"].shape[1],
        data["x_train"].shape[2],
        **build_kwargs,
    )
    train_model(
        model,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        use_class_weight=use_cw,
    )
    if mc:
        p_val, _ = mc_predict(model, data["x_val"], samples=50)
        p_test, std_test = mc_predict(model, data["x_test"], samples=50)
    else:
        p_val = model.predict(data["x_val"], batch_size=256, verbose=0).ravel()
        p_test = model.predict(data["x_test"], batch_size=256, verbose=0).ravel()
        std_test = np.zeros_like(p_test)
    threshold, _ = calibrate_threshold(data["y_val"], p_val)
    metrics = evaluate_binary(data["y_test"], p_test, threshold)
    save_predictions(dataset, model_name, p_test, std_test, data["y_test"])
    safe = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    model.save(os.path.join(MODELS_DIR, f"{dataset}_{safe}.h5"))
    return {"model": model_name, "family": "proposed/ablation", **metrics}


def run_baseline_deep(
    dataset: str,
    model_name: str,
    data: Dict[str, object],
) -> Dict[str, float]:
    set_seed(42)
    model = build_deep_baseline(
        model_name, data["x_train"].shape[1], data["x_train"].shape[2]
    )
    train_model(
        model,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        use_class_weight=True,
    )
    p_val = model.predict(data["x_val"], batch_size=256, verbose=0).ravel()
    p_test = model.predict(data["x_test"], batch_size=256, verbose=0).ravel()
    threshold, _ = calibrate_threshold(data["y_val"], p_val)
    metrics = evaluate_binary(data["y_test"], p_test, threshold)
    save_predictions(dataset, model_name, p_test, np.zeros_like(p_test), data["y_test"])
    return {"model": model_name, "family": "deep baseline", **metrics}


def run_ml(
    dataset: str,
    data: Dict[str, object],
) -> List[Dict[str, float]]:
    x_tr = flatten_windows(data["x_train"])
    x_va = flatten_windows(data["x_val"])
    x_te = flatten_windows(data["x_test"])
    cw = class_weight_from_y(data["y_train"])
    sample_weight = np.where(data["y_train"] == 1, cw[1], cw[0]).astype(np.float32)
    preds = run_ml_baselines(x_tr, data["y_train"], x_va, x_te, sample_weight)
    rows = []
    for name, probs in preds.items():
        threshold, _ = calibrate_threshold(data["y_val"], probs["val"])
        metrics = evaluate_binary(data["y_test"], probs["test"], threshold)
        save_predictions(
            dataset, name, probs["test"], np.zeros_like(probs["test"]), data["y_test"]
        )
        rows.append({"model": name, "family": "machine learning", **metrics})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="ur3,cmapss_fd001,cmapss_fd003,xjtu",
    )
    parser.add_argument("--skip-ml", action="store_true")
    args = parser.parse_args()
    ensure_dirs()

    all_rows = []
    for ds_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        print(f"\n########## {ds_name} ##########", flush=True)
        t0 = time.time()
        data = load_dataset(ds_name)
        print(
            f"train/val/test windows: {len(data['y_train'])}/"
            f"{len(data['y_val'])}/{len(data['y_test'])}, "
            f"pos: {data['y_train'].sum()}/{data['y_val'].sum()}/{data['y_test'].sum()}",
            flush=True,
        )

        rows = []
        for model_name, kwargs, use_cw, seed_rolling in ABLATIONS:
            if seed_rolling:
                data_variant = load_dataset(ds_name + "__rolling")
            else:
                data_variant = data
            if model_name == "w/_rolling_stats":
                rows.append(
                    run_one_deep(
                        ds_name,
                        model_name,
                        data_variant,
                        kwargs,
                        use_cw,
                        kwargs.get("mc_dropout", True),
                    )
                )
                continue
            rows.append(
                run_one_deep(
                    ds_name,
                    model_name,
                    data_variant,
                    kwargs,
                    use_cw,
                    kwargs.get("mc_dropout", True),
                )
            )
            print(f"  finished {model_name} ({time.time()-t0:.0f}s)", flush=True)

        for name in DEEP_BASELINES:
            rows.append(run_baseline_deep(ds_name, name, data))
            print(f"  finished baseline {name} ({time.time()-t0:.0f}s)", flush=True)

        if not args.skip_ml:
            rows.extend(run_ml(ds_name, data))
            print(f"  finished ML baselines ({time.time()-t0:.0f}s)", flush=True)

        for r in rows:
            r["dataset"] = ds_name
            all_rows.append(r)
        save_results(rows)

    df = pd.read_csv(os.path.join(OUT, "results_overall.csv"))
    print(f"\nTotal rows: {len(df)}")
    print(df.groupby(["dataset", "family"])[["auc_roc", "auc_pr", "f2"]].mean())


if __name__ == "__main__":
    main()
