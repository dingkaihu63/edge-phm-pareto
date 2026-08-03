"""Five-seed unit-level paired bootstrap using final PyTorch checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score

from common_no_tf import calibrate_threshold
from prepare_data import load_cmapss, load_ur3, load_xjtu
from run_experiments_torch import DS_CONFIG, LITE_CONFIG
from torch_common import build_deep_baseline, build_model, predict_proba, save_model, set_seed, train_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS_DIR = RESULTS / "models_torch"
DATASETS = ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]
MODELS = ["full", "proposed_lite", "lstm", "tcn", "timesnet"]
SEEDS = [1, 2, 3, 4, 5]
N_BOOT = 3000


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
    if seed == 1:
        name = "full" if model == "full" else model
        return MODELS_DIR / f"{ds}_{name}.pt"
    return MODELS_DIR / f"{ds}_{model}_seed{seed}.pt"


def get_model(ds: str, model_name: str, seed: int, data):
    t, f = data["x_val"].shape[1], data["x_val"].shape[2]
    path = model_path(ds, model_name, seed)
    if path.exists():
        if model_name in ("full", "proposed_lite"):
            cfg = {**DS_CONFIG[ds]}
            if model_name == "proposed_lite":
                cfg = {**cfg, **LITE_CONFIG[ds]}
            model = build_model(
                t, f, attention="sigmoid", mc_dropout=True,
                dropout_rate=cfg["dropout"], attn_temperature=cfg["tau"],
                lstm_units_1=cfg["units1"], lstm_units_2=cfg["units2"], seed=seed,
            )
        else:
            model = build_deep_baseline(model_name, t, f, seed=seed)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
    set_seed(seed)
    if model_name in ("full", "proposed_lite"):
        cfg = {**DS_CONFIG[ds]}
        if model_name == "proposed_lite":
            cfg = {**cfg, **LITE_CONFIG[ds]}
        model = build_model(
            t, f, attention="sigmoid", mc_dropout=True,
            dropout_rate=cfg["dropout"], attn_temperature=cfg["tau"],
            lstm_units_1=cfg["units1"], lstm_units_2=cfg["units2"], seed=seed,
        )
    else:
        model = build_deep_baseline(model_name, t, f, seed=seed)
    cfg = DS_CONFIG[ds]
    train_model(
        model, data["x_train"], data["y_train"], data["x_val"], data["y_val"],
        use_class_weight=True, lr=cfg["lr"], batch_size=cfg["batch"], seed=seed,
        balanced_sampling=cfg.get("balanced", False), grad_clip=cfg.get("grad_clip", 0.0),
    )
    save_model(model, str(path))
    return model


def metrics(y: np.ndarray, p: np.ndarray, threshold: float):
    if y.sum() == 0 or (y == 0).sum() == 0:
        return None
    return {
        "f2": fbeta_score(y, (p >= threshold).astype(int), beta=2, zero_division=0),
        "auc_roc": roc_auc_score(y, p),
        "auc_pr": average_precision_score(y, p),
    }


def run_ds(ds: str) -> None:
    data = load_dataset(ds)
    y = np.asarray(data["y_test"])
    units = np.asarray(data["unit_ids"]["test"])
    unit_ids = np.unique(units)
    rows: List[Dict[str, object]] = []
    for seed in SEEDS:
        preds = {}
        thresholds = {}
        for model_name in MODELS:
            model = get_model(ds, model_name, seed, data)
            pv = predict_proba(model, data["x_val"], batch_size=512)
            pt = predict_proba(model, data["x_test"], batch_size=512)
            preds[model_name] = pt
            thresholds[model_name] = calibrate_threshold(data["y_val"], pv)[0]
        unit_data = []
        for u in unit_ids:
            mask = units == u
            unit_data.append({"y": y[mask], "p": {m: preds[m][mask] for m in MODELS}})
        rng = np.random.RandomState(1000 + seed)
        diffs = {
            (prop, base, metric): []
            for prop in ("full", "proposed_lite")
            for base in ("lstm", "tcn", "timesnet")
            for metric in ("f2", "auc_roc", "auc_pr")
        }
        for _ in range(N_BOOT):
            idx = rng.randint(0, len(unit_ids), size=len(unit_ids))
            ys = []
            ps = {m: [] for m in MODELS}
            for j in idx:
                ys.append(unit_data[j]["y"])
                for m in MODELS:
                    ps[m].append(unit_data[j]["p"][m])
            ys = np.concatenate(ys)
            ps = {m: np.concatenate(ps[m]) for m in MODELS}
            vals = {m: metrics(ys, ps[m], thresholds[m]) for m in MODELS}
            if any(v is None for v in vals.values()):
                continue
            for prop in ("full", "proposed_lite"):
                for base in ("lstm", "tcn", "timesnet"):
                    for metric in ("f2", "auc_roc", "auc_pr"):
                        diffs[(prop, base, metric)].append(vals[prop][metric] - vals[base][metric])
        for (prop, base, metric), values in diffs.items():
            values = np.asarray(values)
            rows.append(
                {
                    "dataset": ds,
                    "seed": seed,
                    "proposed": prop,
                    "baseline": base,
                    "metric": metric,
                    "diff_mean": float(values.mean()),
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                }
            )
        print(ds, seed, "done", flush=True)
    path = RESULTS / "unit_bootstrap_5seeds.csv"
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def main() -> None:
    for ds in DATASETS:
        run_ds(ds)


if __name__ == "__main__":
    main()