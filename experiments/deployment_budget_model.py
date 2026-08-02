"""Hardware-free analytical deployment budget: MACs, cycles, memory estimates."""

from __future__ import annotations

import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "results")

SHAPES = {
    "ur3": (10, 38),
    "cmapss_fd001": (30, 14),
    "cmapss_fd003": (30, 14),
    "xjtu": (20, 24),
}

CFG = {
    "ur3": dict(u1=96, u2=48),
    "cmapss_fd001": dict(u1=64, u2=32),
    "cmapss_fd003": dict(u1=64, u2=32),
    "xjtu": dict(u1=96, u2=48),
}

LITE = {
    "ur3": dict(u1=48, u2=24),
    "cmapss_fd001": dict(u1=32, u2=16),
    "cmapss_fd003": dict(u1=32, u2=16),
    "xjtu": dict(u1=48, u2=24),
}

PARAMS = {
    "ur3": {"full": 81106, "proposed_lite": 24442},
    "cmapss_fd001": {"full": 33602, "proposed_lite": 9650},
    "cmapss_fd003": {"full": 33602, "proposed_lite": 9650},
    "xjtu": {"full": 75730, "proposed_lite": 21754},
}


def macs(t: int, f: int, u1: int, u2: int) -> float:
    lstm1 = 4 * (u1 * (f + u1)) * t
    lstm2 = 4 * (u2 * (u1 + u2)) * t
    attn = t * u2
    dense = u2 * 16 + 16
    return float(lstm1 + lstm2 + attn + dense)


def main() -> None:
    rows = []
    for ds, (t, f) in SHAPES.items():
        for variant, cfg in (("full", CFG), ("proposed_lite", LITE)):
            m = macs(t, f, cfg[ds]["u1"], cfg[ds]["u2"])
            params = PARAMS[ds][variant]
            rows.append({
                "dataset": ds,
                "model": variant,
                "macs_per_window": m,
                "macs_million": m / 1e6,
                "params": params,
                "fp32_weight_kb": params * 4 / 1024,
                "int8_weight_kb": params / 1024,
                "cortex_m7_1mac_cycle_ms": m / 480e6 * 1e3,
                "cortex_m7_2cycles_mac_ms": m / 240e6 * 1e3,
                "mc50_1mac_cycle_ms": m / 480e6 * 1e3 * 50,
                "mc50_2cycles_mac_ms": m / 240e6 * 1e3 * 50,
            })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "deployment_budget.csv")
    df.to_csv(path, index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()