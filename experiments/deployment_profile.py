"""Consolidated hardware-free deployment profile: measurements + analytical budget."""

from __future__ import annotations

import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "results")

MODEL_LABEL = {
    "full": "Proposed (full)",
    "proposed_lite": "Proposed-Lite",
}


def main() -> None:
    torch_df = pd.read_csv(os.path.join(OUT, "deployment_torch.csv"))
    onnx_df = pd.read_csv(os.path.join(OUT, "edge_emulation.csv"))
    budget_df = pd.read_csv(os.path.join(OUT, "deployment_budget.csv"))

    torch_df = torch_df[torch_df["model"].isin(["proposed", "proposed_lite"])].copy()
    torch_df["model"] = torch_df["model"].replace({"proposed": "full", "proposed_lite": "proposed_lite"})
    torch_df = torch_df[["dataset", "model", "params", "cpu_ms_single"]]
    torch_df = torch_df.rename(columns={"cpu_ms_single": "torch_cpu_ms_single"})

    onnx_df = onnx_df[["dataset", "model", "fp32_onnx_bytes", "cpu_ms_single", "mc50_ms_single_est"]]
    onnx_df = onnx_df.rename(columns={"cpu_ms_single": "onnx_cpu_ms_single"})
    budget_df = budget_df[["dataset", "model", "macs_million", "fp32_weight_kb", "int8_weight_kb",
                           "cortex_m7_1mac_cycle_ms", "mc50_1mac_cycle_ms"]]

    merged = torch_df.merge(onnx_df, on=["dataset", "model"], how="left").merge(
        budget_df, on=["dataset", "model"], how="left"
    )
    merged["model_label"] = merged["model"].map(MODEL_LABEL)
    merged["fp32_onnx_kb"] = merged["fp32_onnx_bytes"] / 1024
    merged["mc50_onnx_ms_est"] = merged["mc50_ms_single_est"]
    merged = merged.drop(columns=["fp32_onnx_bytes", "mc50_ms_single_est", "model"])
    merged = merged[["dataset", "model_label", "params", "fp32_weight_kb", "int8_weight_kb",
                     "fp32_onnx_kb", "torch_cpu_ms_single", "onnx_cpu_ms_single", "mc50_onnx_ms_est",
                     "macs_million", "cortex_m7_1mac_cycle_ms", "mc50_1mac_cycle_ms"]]
    merged.columns = [
        "dataset", "model", "params", "fp32_weight_kb", "int8_weight_kb",
        "fp32_onnx_kb", "torch_cpu_ms_single", "onnx_cpu_ms_single", "onnx_mc50_ms_est",
        "macs_million", "cortex_m7_ms_single_est", "cortex_m7_mc50_ms_est",
    ]
    merged = merged.round(3)
    path = os.path.join(OUT, "deployment_profile.csv")
    merged.to_csv(path, index=False)
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()