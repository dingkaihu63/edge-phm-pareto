"""Save raw risk-coverage curves from final MC checkpoints."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

import evidence_analysis as ea
from common_no_tf import calibrate_threshold
from torch_common import mc_predict

TARGETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]


def main() -> None:
    rows: List[dict] = []
    for ds in ea.DATASETS:
        data = ea.load_dataset(ds)
        t, f = data["x_val"].shape[1], data["x_val"].shape[2]
        for model_name in ea.PROPOSED_MODELS:
            model = ea.build_proposed(ds, model_name, t, f)
            pv, stdv = mc_predict(model, data["x_val"], samples=50, batch_size=512)
            pt, stdt = mc_predict(model, data["x_test"], samples=50, batch_size=512)
            threshold, _ = calibrate_threshold(data["y_val"], pv)
            error = ((pt >= threshold).astype(int) != data["y_test"]).astype(int)
            signals = {
                "MC": (stdv, stdt),
                "pseudo_abs": (-np.abs(pv - 0.5), -np.abs(pt - 0.5)),
                "entropy": (ea._entropy(pv), ea._entropy(pt)),
            }
            for name, (uv, ut) in signals.items():
                for c in TARGETS:
                    q = float("inf") if c >= 1.0 else float(np.quantile(uv, c))
                    keep = ut <= q
                    rows.append(
                        {
                            "dataset": ds,
                            "model": model_name,
                            "uncertainty": name,
                            "target_coverage": c,
                            "actual_coverage": float(keep.mean()),
                            "risk": float(error[keep].mean()) if keep.any() else float("nan"),
                        }
                    )
            print(ds, model_name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ea.RESULTS / "risk_curves.csv", index=False)
    print(df.head())


if __name__ == "__main__":
    main()