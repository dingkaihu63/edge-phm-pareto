"""Matched-budget baselines: LSTM/GRU/TCN trained below Proposed-Lite parameters."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_cmapss, load_ur3, load_xjtu
from run_experiments_torch import DS_CONFIG
from torch_common import predict_proba, save_model, set_seed, train_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS_DIR = RESULTS / "models_torch"
PRED_DIR = RESULTS / "predictions"

DATASETS = ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]
KINDS = ["lstm", "gru", "tcn"]


class MatchedBudgetBaseline(nn.Module):
    def __init__(
        self,
        kind: str,
        time_steps: int,
        n_features: int,
        units_1: int = 32,
        units_2: int = 16,
        channels: int = 16,
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.lstm1 = nn.LSTM(n_features, units_1, batch_first=True)
            self.lstm2 = nn.LSTM(units_1, units_2, batch_first=True)
            self.dense = nn.Linear(units_2, 16)
        elif kind == "gru":
            self.gru1 = nn.GRU(n_features, units_1, batch_first=True)
            self.gru2 = nn.GRU(units_1, units_2, batch_first=True)
            self.dense = nn.Linear(units_2, 16)
        elif kind == "tcn":
            self.convs = nn.ModuleList()
            self.dilations = [1, 2, 4, 8, 8]
            for i, d in enumerate(self.dilations):
                in_ch = n_features if i == 0 else channels
                self.convs.append(
                    nn.Sequential(
                        nn.Conv1d(in_ch, channels, 3, padding="same", dilation=d),
                        nn.ReLU(),
                        nn.Conv1d(channels, channels, 3, padding="same", dilation=d),
                    )
                )
            self.dense = nn.Linear(channels, 16)
        else:
            raise ValueError(kind)
        self.output = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "lstm":
            h1, _ = self.lstm1(x)
            h2, _ = self.lstm2(h1)
            z = h2[:, -1, :]
            z = F.relu(self.dense(z))
        elif self.kind == "gru":
            h1, _ = self.gru1(x)
            h2, _ = self.gru2(h1)
            z = h2[:, -1, :]
            z = F.relu(self.dense(z))
        else:
            z = x.transpose(1, 2)
            for i, (conv, dilation) in enumerate(zip(self.convs, self.dilations)):
                y = conv(z)
                if i > 0:
                    y = y + z
                z = F.relu(y)
            z = z.transpose(1, 2).mean(dim=1)
            z = F.relu(self.dense(z))
        return self.output(z)


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


def _params_lstm_gru(F: int, h1: int, h2: int, kind: str) -> int:
    factor = 4 if kind == "lstm" else 3
    n = factor * (F * h1 + h1 * h1 + h1)
    n += factor * (h1 * h2 + h2 * h2 + h2)
    n += h2 * 16 + 16 + 16 * 1 + 1
    return n


def _params_tcn(F: int, c: int) -> int:
    n = 0
    for i in range(5):
        in_c = F if i == 0 else c
        n += in_c * c * 3 + c
        n += c * c * 3 + c
    n += c * 16 + 16 + 16 * 1 + 1
    return n


def choose_widths(kind: str, F: int, budget: int) -> Tuple[int, int, int]:
    if kind in ("lstm", "gru"):
        best = (8, 4)
        for h2 in range(4, 40):
            for h1 in range(h2, 96):
                model = MatchedBudgetBaseline(kind, 30, F, h1, h2, 0)
                if sum(p.numel() for p in model.parameters()) <= budget:
                    best = (h1, h2)
        h1, h2 = best
        return h1, h2, 0
    best = 8
    for c in range(8, 64):
        model = MatchedBudgetBaseline("tcn", 30, F, 0, 0, c)
        if sum(p.numel() for p in model.parameters()) <= budget:
            best = c
    return 0, 0, best


def get_budget(ds: str) -> int:
    dep = pd.read_csv(RESULTS / "deployment_torch.csv")
    row = dep[(dep["dataset"] == ds) & (dep["model"] == "proposed_lite")]
    return int(row["params"].iloc[0])


def run_matched(ds: str) -> None:
    data = load_dataset(ds)
    budget = get_budget(ds)
    F = data["x_train"].shape[2]
    rows: List[Dict[str, object]] = []
    for kind in KINDS:
        u1, u2, ch = choose_widths(kind, F, budget)
        cfg = DS_CONFIG[ds]
        for seed in range(1, 11):
            set_seed(seed)
            model = MatchedBudgetBaseline(kind, data["x_train"].shape[1], F, u1, u2, ch).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            train_model(
                model,
                data["x_train"],
                data["y_train"],
                data["x_val"],
                data["y_val"],
                use_class_weight=True,
                lr=cfg["lr"],
                batch_size=cfg["batch"],
                seed=seed,
                balanced_sampling=cfg.get("balanced", False),
                grad_clip=cfg.get("grad_clip", 0.0),
            )
            pv = predict_proba(model, data["x_val"], batch_size=512)
            pt = predict_proba(model, data["x_test"], batch_size=512)
            threshold, _ = calibrate_threshold(data["y_val"], pv)
            m = evaluate_binary(data["y_test"], pt, threshold)
            rows.append(
                {
                    "dataset": ds,
                    "model": f"matched_{kind}",
                    "kind": kind,
                    "seed": seed,
                    "params": sum(p.numel() for p in model.parameters()),
                    "units_1": u1,
                    "units_2": u2,
                    "channels": ch,
                    "threshold": threshold,
                    "f2": m["f2"],
                    "auc_roc": m["auc_roc"],
                    "auc_pr": m["auc_pr"],
                    "brier": m["brier"],
                    "ece": m["ece"],
                }
            )
            if seed == 1:
                save_model(model, str(MODELS_DIR / f"{ds}_matched_{kind}.pt"))
                np.savez_compressed(
                    PRED_DIR / f"{ds}_matched_{kind}.npz",
                    p=pt,
                    pv=pv,
                    y=data["y_test"],
                    y_val=data["y_val"],
                    threshold=threshold,
                )
        print(ds, kind, "done", flush=True)
    return rows



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    args = parser.parse_args()
    all_rows = []
    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        all_rows.extend(run_matched(ds))
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "matched_budget_seeds.csv", index=False)
    print(df.groupby(["dataset", "model"])[["f2", "auc_roc", "auc_pr"]].agg(["mean", "std"]).to_string())

if __name__ == "__main__":
    main()
