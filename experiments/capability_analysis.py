"""Capability-combination value analysis for the journal revision."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from attention_faithfulness import masked_predict
from common_no_tf import calibrate_threshold
from prepare_data import load_cmapss, load_ur3, load_xjtu
from run_experiments_torch import DS_CONFIG, LITE_CONFIG
from torch_common import build_model, predict_attention

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS_DIR = RESULTS / "models_torch"
PRED_DIR = RESULTS / "predictions"
DATASETS = ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]


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


def lite_attention_drop(ds: str, data) -> Dict[str, float]:
    cfg = {**DS_CONFIG[ds], **LITE_CONFIG[ds]}
    model = build_model(
        data["x_test"].shape[1],
        data["x_test"].shape[2],
        attention="sigmoid",
        mc_dropout=True,
        dropout_rate=cfg["dropout"],
        attn_temperature=cfg["tau"],
        lstm_units_1=cfg["units1"],
        lstm_units_2=cfg["units2"],
        seed=1,
    )
    model.load_state_dict(__import__("torch").load(MODELS_DIR / f"{ds}_proposed_lite.pt", map_location="cpu"))
    model.eval()
    p_base, alpha = predict_attention(model, data["x_test"])
    t = alpha.shape[1]
    k = max(1, int(round(t * 0.2)))
    high = np.argsort(-alpha, axis=1)[:, :k]
    rng = np.random.RandomState(7)
    rand = np.tile(np.arange(t), (len(alpha), 1))
    for i in range(len(rand)):
        rng.shuffle(rand[i])
    rand = rand[:, :k]
    pos = np.asarray(data["y_test"]) == 1
    drop_high = float((p_base - masked_predict(model, data["x_test"], high, k))[pos].mean())
    drop_rand = float((p_base - masked_predict(model, data["x_test"], rand, k))[pos].mean())
    return {"attn_drop20_high": drop_high, "attn_drop20_random": drop_rand}


def pseudo_uncertainty(ds: str) -> Dict[str, float]:
    z = np.load(PRED_DIR / f"{ds}_matched_tcn.npz")
    p, pv, y, yv = z["p"], z["pv"], z["y"], z["y_val"]
    threshold, _ = calibrate_threshold(yv, pv)
    error = ((p >= threshold).astype(int) != y).astype(int)
    unc = -np.abs(p - 0.5)
    err_auc = roc_auc_score(error, unc) if error.sum() and (error == 0).sum() else float("nan")
    unc_v = -np.abs(pv - 0.5)
    risks, covs = [], []
    for c in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        q = float("inf") if c >= 1.0 else float(np.quantile(unc_v, c))
        keep = unc <= q
        covs.append(float(keep.mean()))
        risks.append(float(error[keep].mean()) if keep.any() else float("nan"))
    covs = np.asarray(covs)
    risks = np.asarray(risks)
    valid = ~np.isnan(risks)
    aurc = float(np.trapz(risks[valid], covs[valid]) / max(covs[valid].max() - covs[valid].min(), 1e-12)) if valid.sum() >= 2 else float("nan")
    return {"matched_tcn_error_auc": err_auc, "matched_tcn_aurc50": aurc}


def main() -> None:
    seeds = pd.read_csv(RESULTS / "results_seeds.csv")
    lite_abl = pd.read_csv(RESULTS / "lite_capability_seeds.csv")
    matched = pd.read_csv(RESULTS / "matched_budget_seeds.csv")
    unc = pd.read_csv(RESULTS / "uncertainty_evidence.csv")
    rows: List[Dict[str, object]] = []
    for ds in DATASETS:
        data = load_dataset(ds)
        def f2_mean(df, model):
            return float(df[(df["dataset"] == ds) & (df["model"] == model)]["f2"].mean())
        attn = lite_attention_drop(ds, data)
        pu = pseudo_uncertainty(ds)
        lite_mc = unc[(unc["dataset"] == ds) & (unc["model"] == "proposed_lite") & (unc["uncertainty"] == "MC") & (unc["mc_samples"] == 50)].iloc[0]
        rows.append(
            {
                "dataset": ds,
                "f2_lite": f2_mean(seeds, "proposed_lite"),
                "f2_lite_no_attention": f2_mean(lite_abl, "lite_w_o_attention"),
                "f2_lite_no_mc": f2_mean(lite_abl, "lite_w_o_mc_dropout"),
                "f2_matched_tcn": f2_mean(matched, "matched_tcn"),
                **attn,
                "lite_mc_error_auc": float(lite_mc["error_auc"]),
                "lite_mc_aurc50": float(lite_mc["aurc50"]),
                **pu,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "capability_value.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()