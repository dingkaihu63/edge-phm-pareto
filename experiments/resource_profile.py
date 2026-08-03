"""Resource profile: MACs, FP32/INT8 weight sizes, and peak activation proxy."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.environ.get("EDGE_PHM_PYTHON_PACKAGES", ""))
from thop import profile  # noqa: E402

from run_experiments_torch import DS_CONFIG, LITE_CONFIG
from torch_common import build_deep_baseline, build_model
from matched_budget import MatchedBudgetBaseline

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATASETS = ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]
DEEP = ["lstm", "bilstm", "gru", "transformer", "tcn", "patchtst", "timesnet"]


def proposed_model(ds: str, lite: bool, time_steps: int = 30, n_features: int = 14):
    cfg = {**DS_CONFIG[ds], **LITE_CONFIG[ds]} if lite else DS_CONFIG[ds]
    return build_model(time_steps, n_features,


        attention="sigmoid",
        mc_dropout=True,
        dropout_rate=cfg["dropout"],
        attn_temperature=cfg["tau"],
        lstm_units_1=cfg["units1"],
        lstm_units_2=cfg["units2"],
        seed=1,
    )


def matched_widths(ds: str, kind: str):
    df = pd.read_csv(RESULTS / "matched_budget_seeds.csv")
    row = df[(df["dataset"] == ds) & (df["model"] == f"matched_{kind}")].iloc[0]
    return int(row["units_1"]), int(row["units_2"]), int(row["channels"])


def peak_activation(model, x: torch.Tensor) -> float:
    peak = 0

    def hook(_m, _i, out):
        nonlocal peak
        tensors = out if isinstance(out, (tuple, list)) else [out]
        elements = sum(t.numel() for t in tensors if isinstance(t, torch.Tensor))
        peak = max(peak, elements * 4)

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return peak / 1024.0


def profile_model(model, ds: str, model_name: str) -> Dict[str, object]:
    df = pd.read_csv(RESULTS / "deployment_torch.csv")
    dep = df[(df["dataset"] == ds) & (df["model"] == model_name)]
    t = 30 if ds.startswith("cmapss") else (20 if ds == "xjtu" else 10)
    f = 14 if ds.startswith("cmapss") else (24 if ds == "xjtu" else 38)
    x = torch.randn(1, t, f)
    model = model.cpu()
    try:
        macs, _ = profile(model, inputs=(x,), verbose=False)
    except Exception as exc:  # pragma: no cover
        print("profile failed", ds, model_name, exc)
        macs = float("nan")
    params = sum(p.numel() for p in model.parameters())
    if len(dep):
        params = int(dep["params"].iloc[0])
    peak_kb = peak_activation(model, x)
    return {
        "dataset": ds,
        "model": model_name,
        "params": params,
        "macs_per_window": float(macs),
        "macs_million": float(macs) / 1e6,
        "fp32_weight_kb": params * 4 / 1024.0,
        "int8_weight_kb": params * 1 / 1024.0,
        "peak_activation_kb": peak_kb,
    }


def main() -> None:
    rows: List[Dict[str, object]] = []
    for ds in DATASETS:
        t = 30 if ds.startswith("cmapss") else (20 if ds == "xjtu" else 10)
        f = 14 if ds.startswith("cmapss") else (24 if ds == "xjtu" else 38)
        rows.append(profile_model(proposed_model(ds, False, t, f), ds, "proposed"))
        rows.append(profile_model(proposed_model(ds, True, t, f), ds, "proposed_lite"))
        for kind in DEEP:
            model = build_deep_baseline(kind, t, f, seed=1)
            rows.append(profile_model(model, ds, kind))
        for kind in ("lstm", "gru", "tcn"):
            u1, u2, ch = matched_widths(ds, kind)
            model = MatchedBudgetBaseline(kind, t, f, u1, u2, ch)
            rows.append(profile_model(model, ds, f"matched_{kind}"))
        print(ds, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "resource_profile_full.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
