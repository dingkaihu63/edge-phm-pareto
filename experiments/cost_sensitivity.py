"""Decision-cost sensitivity over maintenance cost ratios."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import evidence_analysis as ea
from common_no_tf import calibrate_threshold
from run_experiments_torch import DS_CONFIG, LITE_CONFIG
from torch_common import build_model, mc_predict

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"
MISS_COSTS = [10, 50, 100, 500, 1000]
FP_COSTS = [0.5, 1.0, 2.0, 5.0]
EXPLAIN_COST = 0.5
CONFIRM = 2
ATTN_MODELS = {"full_mc", "lite_mc", "lite_no_mc"}


def simulate(p_seq: np.ndarray, y_seq: np.ndarray, threshold: float) -> tuple[int, int]:
    pos = np.where(y_seq == 1)[0]
    alarm = -1
    streak = 0
    for i in range(len(p_seq)):
        if p_seq[i] >= threshold:
            streak += 1
        else:
            streak = 0
        if streak >= CONFIRM:
            alarm = i
            break
    if len(pos) > 0:
        return (1 if alarm >= 0 else 0), 0
    return 0, (1 if alarm >= 0 else 0)


def main() -> None:
    rows: List[Dict[str, object]] = []
    for ds in ea.DATASETS:
        data = ea.load_dataset(ds)
        y = np.asarray(data["y_test"])
        units = np.asarray(data["unit_ids"]["test"])
        t, f = data["x_val"].shape[1], data["x_val"].shape[2]
        preds = {}
        z = np.load(PRED_DIR / f"{ds}_matched_tcn.npz")
        preds["matched_tcn"] = (z["pv"], z["p"])
        for name, model_name in [("full_mc", "full"), ("lite_mc", "proposed_lite")]:
            model = ea.build_proposed(ds, model_name, t, f)
            pv, _ = mc_predict(model, data["x_val"], samples=50, batch_size=512)
            pt, _ = mc_predict(model, data["x_test"], samples=50, batch_size=512)
            preds[name] = (pv, pt)
        cfg = {**DS_CONFIG[ds], **LITE_CONFIG[ds]}
        model_na = build_model(
            t, f, attention="none", mc_dropout=True,
            dropout_rate=cfg["dropout"], attn_temperature=cfg["tau"],
            lstm_units_1=cfg["units1"], lstm_units_2=cfg["units2"], seed=1,
        )
        model_na.load_state_dict(torch.load(Path(RESULTS) / "models_torch" / f"{ds}_lite_w_o_attention.pt", map_location="cpu"))
        model_na.eval()
        pv_na, _ = mc_predict(model_na, data["x_val"], samples=50, batch_size=512)
        pt_na, _ = mc_predict(model_na, data["x_test"], samples=50, batch_size=512)
        preds["lite_no_attn_mc"] = (pv_na, pt_na)
        z2 = np.load(PRED_DIR / f"{ds}_lite_w_o_mc_dropout.npz")
        preds["lite_no_mc"] = (z2["pv"], z2["p"])

        counts = {}
        for name, (pv, pt) in preds.items():
            threshold, _ = calibrate_threshold(data["y_val"], pv)
            detected = 0
            false_alarms = 0
            missed = 0
            for u in np.unique(units):
                mask = units == u
                det, fa = simulate(pt[mask], y[mask], threshold)
                if len(np.where(y[mask] == 1)[0]) > 0:
                    detected += det
                    missed += 1 - det
                else:
                    false_alarms += fa
            counts[name] = (missed, false_alarms, detected)

        for mc in MISS_COSTS:
            for fc in FP_COSTS:
                for name, (missed, false_alarms, detected) in counts.items():
                    explain = EXPLAIN_COST * (detected + false_alarms) if name in ATTN_MODELS else 0.0
                    cost = mc * missed + fc * false_alarms + explain
                    rows.append(
                        {
                            "dataset": ds,
                            "model": name,
                            "miss_cost": mc,
                            "fp_cost": fc,
                            "missed": missed,
                            "false_alarms": false_alarms,
                            "total_cost": cost,
                        }
                    )
        print(ds, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "cost_sensitivity.csv", index=False)
    # Summary: how often lite_no_attn_mc is lowest.
    best_rows = []
    for (ds, mc, fc), g in df.groupby(["dataset", "miss_cost", "fp_cost"]):
        best = g.loc[g["total_cost"].idxmin()]
        best_rows.append({"dataset": ds, "miss_cost": mc, "fp_cost": fc, "best_model": best["model"], "best_cost": best["total_cost"]})
    summary = pd.DataFrame(best_rows)
    summary.to_csv(RESULTS / "cost_sensitivity_best.csv", index=False)
    print(summary.pivot_table(index=["miss_cost"], columns=["fp_cost"], values="best_model", aggfunc=lambda x: x.iloc[0]).to_string())


if __name__ == "__main__":
    main()