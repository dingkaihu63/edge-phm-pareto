"""Final-checkpoint evidence analyses for the journal revision.

Subcommands:
  unit_bootstrap   -- unit-level paired bootstrap with 5,000 resamples
  streaming        -- streaming alert simulation with k consecutive windows
  uncertainty      -- MC-dropout K sensitivity, selective prediction, error ranking
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    roc_auc_score,
)

from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_cmapss, load_ur3, load_xjtu
from run_experiments_torch import DS_CONFIG, LITE_CONFIG
from torch_common import (
    DEVICE,
    build_deep_baseline,
    build_model,
    mc_predict,
    predict_proba,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS_DIR = RESULTS / "models_torch"

DATASETS = ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]
PROPOSED_MODELS = ["full", "proposed_lite"]
BASELINES = ["lstm", "tcn", "timesnet"]
MODEL_SET = PROPOSED_MODELS + BASELINES


def dataset_dirs() -> Dict[str, str]:
    return {
        "ur3": os.environ.get("EDGE_PHM_UR3_DIR", ""),
        "cmapss": os.environ.get("EDGE_PHM_CMAPSS_DIR", ""),
        "xjtu": os.environ.get("EDGE_PHM_XJTU_DIR", ""),
        "xjtu_cache": os.environ.get("EDGE_PHM_XJTU_CACHE", ""),
    }


def load_dataset(ds: str) -> Dict[str, object]:
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


def build_proposed(ds: str, model_name: str, time_steps: int, n_features: int, seed: int = 1):
    cfg = {**DS_CONFIG[ds]}
    if model_name == "proposed_lite":
        cfg = {**cfg, **LITE_CONFIG[ds]}
    model = build_model(
        time_steps,
        n_features,
        attention="sigmoid",
        mc_dropout=True,
        dropout_rate=cfg["dropout"],
        attn_temperature=cfg["tau"],
        lstm_units_1=cfg["units1"],
        lstm_units_2=cfg["units2"],
        seed=seed,
    )
    path = MODELS_DIR / f"{ds}_{model_name}.pt"
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def build_baseline(ds: str, kind: str, time_steps: int, n_features: int, seed: int = 1):
    model = build_deep_baseline(kind, time_steps, n_features, seed=seed)
    path = MODELS_DIR / f"{ds}_{kind}.pt"
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def get_predictions(ds: str, data: Dict[str, object]) -> Dict[str, Dict[str, np.ndarray]]:
    t, f = data["x_val"].shape[1], data["x_val"].shape[2]
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name in MODEL_SET:
        if name in PROPOSED_MODELS:
            model = build_proposed(ds, name, t, f)
        else:
            model = build_baseline(ds, name, t, f)
        out[name] = {
            "val": predict_proba(model, data["x_val"], batch_size=512),
            "test": predict_proba(model, data["x_test"], batch_size=512),
        }
    return out


def _metric_set(y: np.ndarray, p: np.ndarray, threshold: float):
    if y.sum() == 0 or (y == 0).sum() == 0:
        return None
    return {
        "f2": fbeta_score(y, (p >= threshold).astype(int), beta=2, zero_division=0),
        "auc_roc": roc_auc_score(y, p),
        "auc_pr": average_precision_score(y, p),
    }


def run_unit_bootstrap() -> None:
    from five_seed_bootstrap import main as run_hierarchical_bootstrap

    run_hierarchical_bootstrap()


def run_streaming() -> None:
    from episode_alert_analysis import main as run_episode_alert_analysis

    run_episode_alert_analysis()


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _risk_coverage(
    unc_v: np.ndarray,
    unc_t: np.ndarray,
    y_t: np.ndarray,
    p_t: np.ndarray,
    threshold: float,
    targets: List[float],
) -> Dict[str, float]:
    error = ((p_t >= threshold).astype(int) != y_t).astype(int)
    coverages = []
    risks = []
    for c in targets:
        q = float("inf") if c >= 1.0 else float(np.quantile(unc_v, c))
        keep = unc_t <= q
        cov = float(keep.mean())
        risk = float(error[keep].mean()) if keep.any() else float("nan")
        coverages.append(cov)
        risks.append(risk)
    coverages = np.asarray(coverages)
    risks = np.asarray(risks)
    valid = ~np.isnan(risks)
    aurc = (
        float(
            np.trapz(risks[valid], coverages[valid])
            / max(coverages[valid].max() - coverages[valid].min(), 1e-12)
        )
        if valid.sum() >= 2
        else float("nan")
    )
    return {
        "aurc50": aurc,
        "risk_full": float(risks[-1]),
        "risk_c60": float(risks[1]) if len(risks) > 1 else float("nan"),
        "risk_c80": float(risks[3]) if len(risks) > 3 else float("nan"),
    }


def run_uncertainty() -> None:
    targets = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows: List[Dict[str, object]] = []
    for ds in DATASETS:
        data = load_dataset(ds)
        t, f = data["x_val"].shape[1], data["x_val"].shape[2]
        for model_name in PROPOSED_MODELS:
            model = build_proposed(ds, model_name, t, f)
            for k in (5, 10, 20, 50):
                pv, stdv = mc_predict(model, data["x_val"], samples=k, batch_size=512)
                pt, stdt = mc_predict(model, data["x_test"], samples=k, batch_size=512)
                threshold, _ = calibrate_threshold(data["y_val"], pv)
                m = evaluate_binary(data["y_test"], pt, threshold)
                error = ((pt >= threshold).astype(int) != data["y_test"]).astype(int)
                unc_signals = {
                    "MC": (stdv, stdt),
                    "pseudo_abs": (-np.abs(pv - 0.5), -np.abs(pt - 0.5)),
                    "entropy": (_entropy(pv), _entropy(pt)),
                }
                for unc_name, (uv, ut) in unc_signals.items():
                    if error.sum() == 0 or (error == 0).sum() == 0:
                        err_auc = float("nan")
                    else:
                        err_auc = roc_auc_score(error, ut)
                    corr = spearmanr(ut, error).correlation
                    rc = _risk_coverage(uv, ut, data["y_test"], pt, threshold, targets)
                    rows.append(
                        {
                            "dataset": ds,
                            "model": model_name,
                            "mc_samples": k,
                            "uncertainty": unc_name,
                            "f2": m["f2"],
                            "auc_roc": m["auc_roc"],
                            "auc_pr": m["auc_pr"],
                            "brier": m["brier"],
                            "ece": m["ece"],
                            "error_auc": err_auc,
                            "error_spearman": float(corr) if corr is not None else float("nan"),
                            **rc,
                        }
                    )
                print(ds, model_name, k, flush=True)
    df = pd.DataFrame(rows)
    path = RESULTS / "uncertainty_evidence.csv"
    df.to_csv(path, index=False)
    print(path)
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["unit_bootstrap", "streaming", "uncertainty"])
    args = parser.parse_args()
    if args.command == "unit_bootstrap":
        run_unit_bootstrap()
    elif args.command == "streaming":
        run_streaming()
    else:
        run_uncertainty()


if __name__ == "__main__":
    main()
