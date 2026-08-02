"""PyTorch deployment metrics: params, CPU/GPU latency, and INT8 dynamic quantization."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_common import (
    DEVICE,
    DeepBaseline,
    ProposedModel,
    build_deep_baseline,
    build_model,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "results")


def measure(model, shape, device, passes=30, batch=100):
    model = model.to(device).eval()
    x = torch.randn(batch, *shape, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(passes):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / passes / batch * 1000
    return elapsed


def main():
    shapes = {
        "ur3": (10, 38),
        "cmapss_fd001": (30, 14),
        "cmapss_fd003": (30, 14),
        "xjtu": (20, 24),
    }
    cfg = {
        "ur3": dict(lstm_units_1=96, lstm_units_2=48, dropout_rate=0.1),
        "cmapss_fd001": dict(lstm_units_1=64, lstm_units_2=32, dropout_rate=0.15),
        "cmapss_fd003": dict(lstm_units_1=64, lstm_units_2=32, dropout_rate=0.15),
        "xjtu": dict(lstm_units_1=96, lstm_units_2=48, dropout_rate=0.1),
    }
    lite = {
        "ur3": dict(lstm_units_1=48, lstm_units_2=24, dropout_rate=0.1),
        "cmapss_fd001": dict(lstm_units_1=32, lstm_units_2=16, dropout_rate=0.15),
        "cmapss_fd003": dict(lstm_units_1=32, lstm_units_2=16, dropout_rate=0.15),
        "xjtu": dict(lstm_units_1=48, lstm_units_2=24, dropout_rate=0.1),
    }
    rows = []
    for ds, shape in shapes.items():
        models = {
            "proposed": build_model(shape[0], shape[1], **cfg[ds]),
            "proposed_lite": build_model(shape[0], shape[1], **lite[ds]),
            "lstm": build_deep_baseline("lstm", shape[0], shape[1]),
            "bilstm": build_deep_baseline("bilstm", shape[0], shape[1]),
            "gru": build_deep_baseline("gru", shape[0], shape[1]),
            "transformer": build_deep_baseline("transformer", shape[0], shape[1]),
            "tcn": build_deep_baseline("tcn", shape[0], shape[1]),
            "patchtst": build_deep_baseline("patchtst", shape[0], shape[1]),
            "timesnet": build_deep_baseline("timesnet", shape[0], shape[1]),
        }
        for name, model in models.items():
            params = sum(p.numel() for p in model.parameters())
            cpu_batch = measure(model, shape, torch.device("cpu"), batch=100)
            cpu_single = measure(model, shape, torch.device("cpu"), batch=1, passes=20)
            gpu_batch = measure(model, shape, DEVICE, batch=100) if DEVICE.type == "cuda" else np.nan
            mc_batch = cpu_batch * 50 if name in ("proposed", "proposed_lite") else np.nan
            mc_single = cpu_single * 50 if name in ("proposed", "proposed_lite") else np.nan
            q_params = q_batch = q_single = np.nan
            if name in ("proposed", "proposed_lite"):
                q_params = float(sum(p.numel() for p in model.parameters() if p.ndim >= 2))
                try:
                    qmodel = torch.quantization.quantize_dynamic(
                        model.cpu(), {nn.LSTM, nn.Linear}, dtype=torch.qint8
                    )
                    q_batch = measure(qmodel, shape, torch.device("cpu"), batch=100)
                    q_single = measure(qmodel, shape, torch.device("cpu"), batch=1, passes=20)
                except Exception as exc:  # pragma: no cover
                    print("quantize failed", ds, name, exc)
            rows.append(
                {
                    "dataset": ds,
                    "model": name,
                    "params": params,
                    "quant_params": q_params,
                    "cpu_ms_batch100": cpu_batch,
                    "cpu_ms_single": cpu_single,
                    "gpu_ms_batch100": gpu_batch,
                    "mc50_cpu_ms_batch100": mc_batch,
                    "mc50_cpu_ms_single": mc_single,
                    "quant_cpu_ms_batch100": q_batch,
                    "quant_cpu_ms_single": q_single,
                }
            )
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "deployment_torch.csv"), index=False)
    print(df.groupby("model")[["params", "cpu_ms_single"]].mean().round(3))


if __name__ == "__main__":
    main()