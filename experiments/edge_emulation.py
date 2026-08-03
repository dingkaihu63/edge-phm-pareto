"""Zero-cost edge-gateway emulation: single-thread ONNX Runtime measurements."""

from __future__ import annotations

import os
import sys
import time

PKG = os.environ.get("EDGE_PHM_PYTHON_PACKAGES", "")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import numpy as np
import psutil
import torch
import onnxruntime as ort

from torch_common import ProposedModel

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "results")
MODELS = os.path.join(OUT, "models_torch")
ONNX_DIR = os.environ.get("EDGE_PHM_ONNX_DIR", "")
os.makedirs(ONNX_DIR, exist_ok=True)

SHAPES = {
    "ur3": (10, 38),
    "cmapss_fd001": (30, 14),
    "cmapss_fd003": (30, 14),
    "xjtu": (20, 24),
}

CFG = {
    "ur3": dict(lstm_units_1=96, lstm_units_2=48, dropout_rate=0.1),
    "cmapss_fd001": dict(lstm_units_1=64, lstm_units_2=32, dropout_rate=0.15),
    "cmapss_fd003": dict(lstm_units_1=64, lstm_units_2=32, dropout_rate=0.15),
    "xjtu": dict(lstm_units_1=96, lstm_units_2=48, dropout_rate=0.1),
}

LITE = {
    "ur3": dict(lstm_units_1=48, lstm_units_2=24, dropout_rate=0.1),
    "cmapss_fd001": dict(lstm_units_1=32, lstm_units_2=16, dropout_rate=0.15),
    "cmapss_fd003": dict(lstm_units_1=32, lstm_units_2=16, dropout_rate=0.15),
    "xjtu": dict(lstm_units_1=48, lstm_units_2=24, dropout_rate=0.1),
}


def _det_model(shape, cfg):
    return ProposedModel(shape[0], shape[1], mc_dropout=False, **cfg)


def _measure(session, x, batch, passes):
    with torch.no_grad():
        for _ in range(3):
            session.run(None, {"input": x})
        start = time.perf_counter()
        for _ in range(passes):
            session.run(None, {"input": x})
        return (time.perf_counter() - start) / passes / batch * 1000


def main() -> None:
    rows = []
    for ds, shape in SHAPES.items():
        for variant, cfg in (("full", CFG), ("proposed_lite", LITE)):
            det = _det_model(shape, cfg[ds])
            det.load_state_dict(torch.load(os.path.join(MODELS, f"{ds}_{variant}.pt"), map_location="cpu"))
            det.eval()
            x = torch.randn(1, *shape, dtype=torch.float32)
            onnx_path = os.path.join(ONNX_DIR, f"{ds}_{variant}.onnx")
            torch.onnx.export(
                det, x, onnx_path,
                input_names=["input"], output_names=["logit"],
                dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
                opset_version=17,
            )
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            xb = np.ascontiguousarray(x.numpy())
            single = _measure(sess, xb, 1, 50)
            onnx_bytes = os.path.getsize(onnx_path)
            data_path = onnx_path + ".data"
            if os.path.exists(data_path):
                onnx_bytes += os.path.getsize(data_path)
            row = {
                "dataset": ds,
                "model": variant,
                "backend": "onnxruntime-cpu",
                "threads": 1,
                "fp32_onnx_bytes": onnx_bytes,
                "cpu_ms_single": single,
                "mc10_ms_single_est": single * 10,
                "mc20_ms_single_est": single * 20,
                "mc50_ms_single_est": single * 50,
            }
            rows.append(row)
            print(row, flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "edge_emulation.csv"), index=False)


if __name__ == "__main__":
    main()