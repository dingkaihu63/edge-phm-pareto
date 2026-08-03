"""Run only the TCN published baseline for datasets whose models already exist."""

from __future__ import annotations

import os

import numpy as np

from common import (
    build_deep_baseline,
    calibrate_threshold,
    evaluate_binary,
    mc_predict,
    set_seed,
    train_model,
)
from run_experiments import MODELS_DIR, PRED_DIR, load_dataset, save_results


def main() -> None:
    for ds in ["ur3", "cmapss_fd001", "cmapss_fd003"]:
        print(f"\n### TCN on {ds}", flush=True)
        data = load_dataset(ds)
        set_seed(42)
        model = build_deep_baseline(
            "tcn", data["x_train"].shape[1], data["x_train"].shape[2]
        )
        train_model(
            model,
            data["x_train"],
            data["y_train"],
            data["x_val"],
            data["y_val"],
            use_class_weight=True,
        )
        p_val = model.predict(data["x_val"], batch_size=256, verbose=0).ravel()
        p_test = model.predict(data["x_test"], batch_size=256, verbose=0).ravel()
        threshold, _ = calibrate_threshold(data["y_val"], p_val)
        metrics = evaluate_binary(data["y_test"], p_test, threshold)
        np.savez_compressed(
            os.path.join(PRED_DIR, f"{ds}_tcn.npz"),
            p=p_test,
            std=np.zeros_like(p_test),
            y=data["y_test"],
        )
        model.save(os.path.join(MODELS_DIR, f"{ds}_tcn.h5"))
        row = {
            "dataset": ds,
            "model": "tcn",
            "family": "deep baseline",
            **metrics,
        }
        save_results([row])
        print(f"  tcn done: F2={metrics['f2']:.3f} AUROC={metrics['auc_roc']:.3f}")


if __name__ == "__main__":
    main()
