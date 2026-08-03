"""ARM64 QEMU user-mode latency benchmark for final ONNX models."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import onnxruntime as ort

ONNX_DIR = "/mnt/e/edge_phm_onnx"
MODELS = [
    ("ur3", "ur3_full.onnx", "ur3_proposed_lite.onnx", (10, 38)),
    ("cmapss_fd001", "cmapss_fd001_full.onnx", "cmapss_fd001_proposed_lite.onnx", (30, 14)),
    ("cmapss_fd003", "cmapss_fd003_full.onnx", "cmapss_fd003_proposed_lite.onnx", (30, 14)),
    ("xjtu", "xjtu_full.onnx", "xjtu_proposed_lite.onnx", (20, 24)),
]


def measure(path: str, shape: tuple[int, int], passes: int = 50) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    x = np.ascontiguousarray(np.random.randn(1, *shape).astype(np.float32))
    for _ in range(3):
        sess.run(None, {"input": x})
    start = time.perf_counter()
    for _ in range(passes):
        sess.run(None, {"input": x})
    elapsed = (time.perf_counter() - start) / passes * 1000.0
    return {"single_ms": elapsed, "mc50_ms_est": elapsed * 50.0}


def main() -> None:
    rows = []
    for ds, full_name, lite_name, shape in MODELS:
        for variant, name in (("full", full_name), ("proposed_lite", lite_name)):
            path = os.path.join(ONNX_DIR, name)
            if not os.path.exists(path):
                print(json.dumps({"dataset": ds, "model": variant, "error": "missing " + path}))
                continue
            row = measure(path, shape)
            row.update({"dataset": ds, "model": variant, "backend": "qemu-aarch64-user", "cpu_model": "max"})
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open("/tmp/arm_bench.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()