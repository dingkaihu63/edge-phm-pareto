"""Validation-selected framework configuration on final seed-1 checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import fbeta_score, roc_auc_score, average_precision_score

import evidence_analysis as ea
from common_no_tf import calibrate_threshold
from run_experiments_torch import DS_CONFIG, LITE_CONFIG
from torch_common import build_model, predict_proba

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


def build(ds: str, name: str, data):
    cfg = CONFIGS[name]
    if cfg["lite"]:
        model_cfg = {**DS_CONFIG[ds], **LITE_CONFIG[ds]}
    else:
        model_cfg = DS_CONFIG[ds]
    model = build_model(
        data["x_val"].shape[1], data["x_val"].shape[2],
        attention=cfg["attention"], mc_dropout=cfg["mc_dropout"],
        dropout_rate=model_cfg["dropout"], attn_temperature=model_cfg["tau"],
        lstm_units_1=model_cfg["units1"], lstm_units_2=model_cfg["units2"], seed=1,
    )
    safe_name = name.replace("/", "_")
    path = MODELS_DIR / f"{ds}_{safe_name}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def main() -> None:
    rows: List[Dict[str, object]] = []
    for ds in ea.DATASETS:
        data = ea.load_dataset(ds)
        yv = np.asarray(data["y_val"])
        yt = np.asarray(data["y_test"])
        candidates = []
        for name in CONFIGS:
            model = build(ds, name, data)
            pv = predict_proba(model, data["x_val"], batch_size=512)
            pt = predict_proba(model, data["x_test"], batch_size=512)
            threshold, _ = calibrate_threshold(yv, pv)
            ypv = (pv >= threshold).astype(int)
            val_f2 = fbeta_score(yv, ypv, beta=2, zero_division=0)
            ypt = (pt >= threshold).astype(int)
            candidates.append(
                {
                    "name": name,
                    "val_f2": val_f2,
                    "test_f2": fbeta_score(yt, ypt, beta=2, zero_division=0),
                    "auc_roc": roc_auc_score(yt, pt),
                    "auc_pr": average_precision_score(yt, pt),
                    "threshold": threshold,
                }
            )
        best = max(candidates, key=lambda r: r["val_f2"])
        rows.append({"dataset": ds, "selected": best["name"], **{k: v for k, v in best.items() if k != "name"}})
        print(ds, best, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "validation_config_final.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
