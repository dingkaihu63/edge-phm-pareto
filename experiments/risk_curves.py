"""Five-seed risk-coverage and error-ranking analysis.

Per-seed MC dispersion is compared with one deterministic confidence ranking.
A five-model seed ensemble supplies a nonredundant epistemic reference based on
variation across independently trained checkpoints.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import evidence_analysis as ea
from common_no_tf import calibrate_threshold
from five_seed_bootstrap import SEEDS, get_model
from torch_common import mc_predict

TARGETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
MC_SAMPLES = 50


def confidence_uncertainty(probability: np.ndarray) -> np.ndarray:
    """Binary uncertainty ranking induced by distance from 0.5."""
    return -np.abs(probability - 0.5)


def error_ranking_auc(error: np.ndarray, uncertainty: np.ndarray) -> float:
    if len(np.unique(error)) < 2:
        return float("nan")
    return float(roc_auc_score(error, uncertainty))


def curve_rows(
    dataset: str,
    model_name: str,
    uncertainty_name: str,
    replicate: str,
    seed: int,
    validation_uncertainty: np.ndarray,
    test_uncertainty: np.ndarray,
    error: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for target in TARGETS:
        cutoff = (
            float("inf")
            if target >= 1.0
            else float(np.quantile(validation_uncertainty, target))
        )
        keep = test_uncertainty <= cutoff
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "uncertainty": uncertainty_name,
                "replicate": replicate,
                "seed": seed,
                "target_coverage": target,
                "actual_coverage": float(keep.mean()),
                "risk": float(error[keep].mean()) if keep.any() else float("nan"),
                "n_kept": int(keep.sum()),
                "n_test_windows": int(len(error)),
            }
        )
    return rows


def seed_predictions(dataset: str, model_name: str, data) -> Dict[int, Tuple[np.ndarray, ...]]:
    predictions: Dict[int, Tuple[np.ndarray, ...]] = {}
    for seed in SEEDS:
        model = get_model(dataset, model_name, seed, data)
        val_mean, val_std = mc_predict(
            model, data["x_val"], samples=MC_SAMPLES, batch_size=512
        )
        test_mean, test_std = mc_predict(
            model, data["x_test"], samples=MC_SAMPLES, batch_size=512
        )
        predictions[seed] = (val_mean, val_std, test_mean, test_std)
    return predictions


def main() -> None:
    curves: List[Dict[str, object]] = []
    rankings: List[Dict[str, object]] = []
    for dataset in ea.DATASETS:
        data = ea.load_dataset(dataset)
        for model_name in ea.PROPOSED_MODELS:
            predictions = seed_predictions(dataset, model_name, data)
            for seed, (val_mean, val_std, test_mean, test_std) in predictions.items():
                threshold, _ = calibrate_threshold(data["y_val"], val_mean)
                error = ((test_mean >= threshold).astype(int) != data["y_test"]).astype(int)
                signals = {
                    "mc_dropout": (val_std, test_std),
                    "confidence": (
                        confidence_uncertainty(val_mean),
                        confidence_uncertainty(test_mean),
                    ),
                }
                for uncertainty_name, (val_uncertainty, test_uncertainty) in signals.items():
                    curves.extend(
                        curve_rows(
                            dataset,
                            model_name,
                            uncertainty_name,
                            "seed",
                            seed,
                            val_uncertainty,
                            test_uncertainty,
                            error,
                        )
                    )
                    rankings.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "uncertainty": uncertainty_name,
                            "replicate": "seed",
                            "seed": seed,
                            "error_ranking_auc": error_ranking_auc(
                                error, test_uncertainty
                            ),
                            "threshold": threshold,
                            "test_error_rate": float(error.mean()),
                        }
                    )

            validation_means = np.stack(
                [predictions[seed][0] for seed in SEEDS], axis=0
            )
            test_means = np.stack(
                [predictions[seed][2] for seed in SEEDS], axis=0
            )
            ensemble_val_mean = validation_means.mean(axis=0)
            ensemble_test_mean = test_means.mean(axis=0)
            ensemble_val_std = validation_means.std(axis=0, ddof=1)
            ensemble_test_std = test_means.std(axis=0, ddof=1)
            ensemble_threshold, _ = calibrate_threshold(
                data["y_val"], ensemble_val_mean
            )
            ensemble_error = (
                (ensemble_test_mean >= ensemble_threshold).astype(int)
                != data["y_test"]
            ).astype(int)
            curves.extend(
                curve_rows(
                    dataset,
                    model_name,
                    "seed_ensemble",
                    "ensemble",
                    0,
                    ensemble_val_std,
                    ensemble_test_std,
                    ensemble_error,
                )
            )
            rankings.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "uncertainty": "seed_ensemble",
                    "replicate": "ensemble",
                    "seed": 0,
                    "error_ranking_auc": error_ranking_auc(
                        ensemble_error, ensemble_test_std
                    ),
                    "threshold": ensemble_threshold,
                    "test_error_rate": float(ensemble_error.mean()),
                }
            )
            print(f"{dataset}: {model_name} uncertainty analysis complete", flush=True)

    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(ea.RESULTS / "risk_curves_5seeds.csv", index=False)
    ranking_frame = pd.DataFrame(rankings)
    ranking_frame.to_csv(ea.RESULTS / "uncertainty_ranking_5seeds.csv", index=False)
    seed_summary = (
        ranking_frame[ranking_frame["replicate"] == "seed"]
        .groupby(["dataset", "model", "uncertainty"])["error_ranking_auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    seed_summary.to_csv(
        ea.RESULTS / "uncertainty_ranking_5seeds_summary.csv", index=False
    )
    print(seed_summary.to_string(index=False))
    print(
        ranking_frame[ranking_frame["replicate"] == "ensemble"].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
