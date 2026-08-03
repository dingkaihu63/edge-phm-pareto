"""Decision-cost comparison for deterministic, MC, and configurable framework variants."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import evidence_analysis as ea
from common_no_tf import calibrate_threshold
import torch
from run_experiments_torch import DS_CONFIG, LITE_CONFIG
from torch_common import build_model, mc_predict

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"
MISS_COST = 100.0
FP_COST = 1.0
EXPLAIN_COST = 0.5
CONFIRM = 2
ATTN_MODELS = {"full_mc", "lite_mc", "lite_no_mc"}


def simulate(p_seq: np.ndarray, y_seq: np.ndarray, threshold: float) -> tuple[int, int, int]:
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
        return (1 if alarm >= 0 else 0), 0, (alarm - int(pos[0]) if alarm >= 0 else -1)
    return 0, (1 if alarm >= 0 else 0), -1


def main() -> None:
    rows: List[Dict[str, object]] = []
    for ds in ea.DATASETS:
        data = ea.load_dataset(ds)
        y = np.asarray(data["y_test"])
        units = np.asarray(data["unit_ids"]["test"])
        t, f = data["x_val"].shape[1], data["x_val"].shape[2]
        preds = {}
        # Matched TCN deterministic.
        z = np.load(PRED_DIR / f"{ds}_matched_tcn.npz")
        preds["matched_tcn"] = (z["pv"], z["p"])
        # MC variants from final seed-1 checkpoints.
        for name, model_name in [("full_mc", "full"), ("lite_mc", "proposed_lite")]:
            model = ea.build_proposed(ds, model_name, t, f)
            pv, _ = mc_predict(model, data["x_val"], samples=50, batch_size=512)
            pt, _ = mc_predict(model, data["x_test"], samples=50, batch_size=512)
            preds[name] = (pv, pt)
        # Lite no-attention MC with Lite widths.
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
        for name, (pv, pt) in preds.items():
            threshold, _ = calibrate_threshold(data["y_val"], pv)
            detected = 0
            false_alarms = 0
            missed = 0
            failure_units = 0
            non_failure = 0
            for u in np.unique(units):
                mask = units == u
                det, fa, _lead = simulate(pt[mask], y[mask], threshold)
                if len(np.where(y[mask] == 1)[0]) > 0:
                    failure_units += 1
                    detected += det
                    missed += 1 - det
                else:
                    non_failure += 1
                    false_alarms += fa
            alarms = detected + false_alarms
            explain_cost = EXPLAIN_COST * alarms if name in ATTN_MODELS else 0.0
            cost = MISS_COST * missed + FP_COST * false_alarms + explain_cost
            cost_no_explain = MISS_COST * missed + FP_COST * false_alarms
            rows.append(
                {
                    "dataset": ds,
                    "model": name,
                    "threshold": threshold,
                    "failure_units": failure_units,
                    "detected_units": detected,
                    "missed_units": missed,
                    "false_alarm_units": false_alarms,
                    "total_cost": cost,
                    "cost_without_explain": cost_no_explain,
                    "explain_cost": explain_cost,
                }
            )
        print(ds, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "decision_cost.csv", index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
