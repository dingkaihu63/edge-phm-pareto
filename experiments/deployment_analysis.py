"""Measure parameter counts and CPU inference latency for edge-deployment comparison."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from common import build_deep_baseline, build_model


RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def measure(name: str, model: tf.keras.Model, shape: tuple, passes: int = 50) -> dict:
    x = np.random.RandomState(0).randn(100, *shape).astype(np.float32)
    # Warm-up
    model.predict(x, batch_size=100, verbose=0)
    start = time.perf_counter()
    for _ in range(passes):
        model.predict(x, batch_size=100, verbose=0)
    single_ms = (time.perf_counter() - start) / passes / 100 * 1000
    return {
        "model": name,
        "params": int(model.count_params()),
        "single_pass_ms_per_window": single_ms,
        "mc50_ms_per_window": single_ms * 50,
    }


def main() -> None:
    shapes = {
        "ur3": (10, 20),
        "cmapss_fd001": (30, 14),
        "cmapss_fd003": (30, 14),
        "xjtu": (20, 12),
    }
    rows = []
    for ds, shape in shapes.items():
        models = {
            "proposed": build_model(shape[0], shape[1]),
            "lstm": build_deep_baseline("lstm", shape[0], shape[1]),
            "bilstm": build_deep_baseline("bilstm", shape[0], shape[1]),
            "gru": build_deep_baseline("gru", shape[0], shape[1]),
            "transformer": build_deep_baseline("transformer", shape[0], shape[1]),
            "tcn": build_deep_baseline("tcn", shape[0], shape[1]),
        }
        for name, model in models.items():
            r = measure(name, model, shape)
            r["dataset"] = ds
            rows.append(r)
            tf.keras.backend.clear_session()
            print(ds, name, r)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "deployment.csv"), index=False)
    print(df.groupby("model")[["params", "single_pass_ms_per_window"]].mean())


if __name__ == "__main__":
    main()
