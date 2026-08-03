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
    rng = np.random.RandomState(2026)
    rows: List[Dict[str, object]] = []
    for ds in DATASETS:
        data = load_dataset(ds)
        preds = get_predictions(ds, data)
        y = np.asarray(data["y_test"])
        units = np.asarray(data["unit_ids"]["test"])
        thresholds = {
            name: calibrate_threshold(data["y_val"], preds[name]["val"])[0]
            for name in MODEL_SET
        }
        unit_ids = np.unique(units)
        unit_data = []
        for u in unit_ids:
            mask = units == u
            unit_data.append(
                {
                    "y": y[mask],
                    "p": {name: preds[name]["test"][mask] for name in MODEL_SET},
                }
            )
        diffs = {
            (prop, base, metric): []
            for prop in PROPOSED_MODELS
            for base in BASELINES
            for metric in ("f2", "auc_roc", "auc_pr")
        }
        n_iter = 0
        for _ in range(5000):
            idx = rng.randint(0, len(unit_ids), size=len(unit_ids))
            ys = []
            ps = {name: [] for name in MODEL_SET}
            for j in idx:
                ys.append(unit_data[j]["y"])
                for name in MODEL_SET:
                    ps[name].append(unit_data[j]["p"][name])
            ys = np.concatenate(ys)
            ps = {name: np.concatenate(ps[name]) for name in MODEL_SET}
            vals = {
                name: _metric_set(ys, ps[name], thresholds[name]) for name in MODEL_SET
            }
            if any(v is None for v in vals.values()):
                continue
            for prop in PROPOSED_MODELS:
                for base in BASELINES:
                    for metric in ("f2", "auc_roc", "auc_pr"):
                        diffs[(prop, base, metric)].append(
                            vals[prop][metric] - vals[base][metric]
                        )
            n_iter += 1
        observed = {
            name: _metric_set(y, preds[name]["test"], thresholds[name]) for name in MODEL_SET
        }
        for (prop, base, metric), values in diffs.items():
            values = np.asarray(values)
            rows.append(
                {
                    "dataset": ds,
                    "proposed": prop,
                    "baseline": base,
                    "metric": metric,
                    "proposed_value": observed[prop][metric],
                    "baseline_value": observed[base][metric],
                    "diff_mean": float(values.mean()),
                    "diff_sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                    "ci_includes_zero": bool(
                        np.percentile(values, 2.5) <= 0 <= np.percentile(values, 97.5)
                    ),
                    "n_bootstrap": n_iter,
                    "n_units": len(unit_ids),
                }
            )
    df = pd.DataFrame(rows)
    path = RESULTS / "unit_bootstrap_5000.csv"
    df.to_csv(path, index=False)
    print(path)
    print(df[df["metric"] == "f2"].to_string(index=False))


def run_streaming() -> None:
    rows: List[Dict[str, object]] = []
    for ds in DATASETS:
        data = load_dataset(ds)
        preds = get_predictions(ds, data)
        y = np.asarray(data["y_test"])
        units = np.asarray(data["unit_ids"]["test"])
        for model_name in MODEL_SET:
            pv = preds[model_name]["val"]
            p = preds[model_name]["test"]
            threshold, _ = calibrate_threshold(data["y_val"], pv)
            for confirm in (1, 2, 3):
                failures = 0
                detected = 0
                false_alarms = 0
                non_failure = 0
                leads = []
                for u in np.unique(units):
                    mask = units == u
                    p_seq = p[mask]
                    y_seq = y[mask]
                    pos = np.where(y_seq == 1)[0]
                    alarm = -1
                    streak = 0
                    for i in range(len(p_seq)):
                        if p_seq[i] >= threshold:
                            streak += 1
                        else:
                            streak = 0
                        if streak >= confirm:
                            alarm = i
                            break
                    if len(pos) > 0:
                        failures += 1
                        if alarm >= 0:
                            detected += 1
                            leads.append(alarm - int(pos[0]))
                    else:
                        non_failure += 1
                        if alarm >= 0:
                            false_alarms += 1
                lead_arr = np.asarray(leads, dtype=float)
                rows.append(
                    {
                        "dataset": ds,
                        "model": model_name,
                        "threshold": threshold,
                        "confirm_windows": confirm,
                        "failure_units": failures,
                        "detected_units": detected,
                        "missed_units": failures - detected,
                        "detection_rate": detected / max(failures, 1),
                        "non_failure_units": non_failure,
                        "false_alarm_units": false_alarms,
                        "false_alarm_rate": false_alarms / max(non_failure, 1),
                        "mean_lead": float(np.nanmean(lead_arr)) if len(lead_arr) else float("nan"),
                        "median_lead": float(np.nanmedian(lead_arr)) if len(lead_arr) else float("nan"),
                    }
                )
    df = pd.DataFrame(rows)
    path = RESULTS / "streaming_alert.csv"
    df.to_csv(path, index=False)
    print(path)
    print(df.to_string(index=False))


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