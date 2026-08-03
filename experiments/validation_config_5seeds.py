"""Five-seed validation-selected framework configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score

import evidence_analysis as ea
from common_no_tf import calibrate_threshold
from run_experiments_torch import DS_CONFIG, LITE_CONFIG
from torch_common import build_model, build_deep_baseline, predict_proba, save_model, set_seed, train_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS_DIR = RESULTS / "models_torch"
CONFIGS = {
    "full": dict(attention="sigmoid", mc_dropout=True, lite=False),
    "proposed_lite": dict(attention="sigmoid", mc_dropout=True, lite=True),
    "w/o_attention": dict(attention="none", mc_dropout=True, lite=False),
    "lite_w_o_attention": dict(attention="none", mc_dropout=True, lite=True),
    "lite_w_o_mc_dropout": dict(attention="sigmoid", mc_dropout=False, lite=True),
}
SEEDS = [1, 2, 3, 4, 5]


def model_path(ds: str, name: str, seed: int) -> Path:
    safe = name.replace("/", "_")
    if seed == 1:
        return MODELS_DIR / f"{ds}_{safe}.pt"
    return MODELS_DIR / f"{ds}_{safe}_seed{seed}.pt"


def get_model(ds: str, name: str, seed: int, data):
    cfg = CONFIGS[name]
    if cfg["lite"]:
        model_cfg = {**DS_CONFIG[ds], **LITE_CONFIG[ds]}
    else:
        model_cfg = DS_CONFIG[ds]
    t, f = data["x_val"].shape[1], data["x_val"].shape[2]
    model = build_model(
        t, f, attention=cfg["attention"], mc_dropout=cfg["mc_dropout"],
        dropout_rate=model_cfg["dropout"], attn_temperature=model_cfg["tau"],
        lstm_units_1=model_cfg["units1"], lstm_units_2=model_cfg["units2"], seed=seed,
    )
    path = model_path(ds, name, seed)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
    set_seed(seed)
    cfgd = DS_CONFIG[ds]
    train_model(
        model, data["x_train"], data["y_train"], data["x_val"], data["y_val"],
        use_class_weight=True, lr=cfgd["lr"], batch_size=cfgd["batch"], seed=seed,
        balanced_sampling=cfgd.get("balanced", False), grad_clip=cfgd.get("grad_clip", 0.0),
    )
    save_model(model, str(path))
    return model


def main() -> None:
    rows: List[Dict[str, object]] = []
    for ds in ea.DATASETS:
        data = ea.load_dataset(ds)
        yv = np.asarray(data["y_val"])
        yt = np.asarray(data["y_test"])
        for seed in SEEDS:
            candidates = []
            for name in CONFIGS:
                model = get_model(ds, name, seed, data)
                pv = predict_proba(model, data["x_val"], batch_size=512)
                pt = predict_proba(model, data["x_test"], batch_size=512)
                threshold, _ = calibrate_threshold(yv, pv)
                val_f2 = fbeta_score(yv, (pv >= threshold).astype(int), beta=2, zero_division=0)
                candidates.append(
                    {
                        "name": name,
                        "val_f2": val_f2,
                        "test_f2": fbeta_score(yt, (pt >= threshold).astype(int), beta=2, zero_division=0),
                        "auc_roc": roc_auc_score(yt, pt),
                        "auc_pr": average_precision_score(yt, pt),
                    }
                )
            best = max(candidates, key=lambda r: r["val_f2"])
            rows.append(
                {
                    "dataset": ds,
                    "seed": seed,
                    "selected": best["name"],
                    "val_f2": best["val_f2"],
                    "test_f2": best["test_f2"],
                    "auc_roc": best["auc_roc"],
                    "auc_pr": best["auc_pr"],
                }
            )
            print(ds, seed, best["name"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "validation_config_5seeds.csv", index=False)
    summary = (
        df.groupby("dataset", as_index=False)
        .agg(
            selected_modes=("selected", lambda x: "+".join(x.unique())),
            selected_count=("selected", lambda x: x.value_counts().to_dict()),
            mean_test_f2=("test_f2", "mean"),
            min_test_f2=("test_f2", "min"),
            max_test_f2=("test_f2", "max"),
        )
    )
    summary.to_csv(RESULTS / "validation_config_5seeds_summary.csv", index=False)
    print(df.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()