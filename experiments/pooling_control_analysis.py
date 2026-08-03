"""Five-seed control separating learned attention from temporal pooling.

The learned-attention and uniform-mean variants share the same recurrent widths,
dense head, dropout mode, and nominal parameter structure. The terminal-state
variant is retained to show how changing the aggregation rule affects results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from common_no_tf import calibrate_threshold, evaluate_binary
from five_seed_bootstrap import DATASETS, RESULTS, SEEDS, load_dataset
from run_experiments_torch import DS_CONFIG
from torch_common import (
    DEVICE,
    build_model,
    mc_predict,
    save_model,
    set_seed,
    train_model,
)

MODELS_DIR = RESULTS / "models_torch"
MC_SAMPLES = 50
CONFIGS = {
    "learned_attention": {"attention": "sigmoid", "checkpoint": "full"},
    "uniform_mean": {"attention": "mean", "checkpoint": "mean_pooling"},
    "terminal_state": {"attention": "none", "checkpoint": "w_o_attention"},
}


def checkpoint_path(dataset: str, checkpoint: str, seed: int) -> Path:
    suffix = "" if seed == 1 else f"_seed{seed}"
    return MODELS_DIR / f"{dataset}_{checkpoint}{suffix}.pt"


def build_variant(dataset: str, name: str, seed: int, data):
    cfg = DS_CONFIG[dataset]
    return build_model(
        data["x_val"].shape[1],
        data["x_val"].shape[2],
        attention=CONFIGS[name]["attention"],
        mc_dropout=True,
        dropout_rate=cfg["dropout"],
        attn_temperature=cfg["tau"],
        lstm_units_1=cfg["units1"],
        lstm_units_2=cfg["units2"],
        seed=seed,
    )


def get_or_train_variant(dataset: str, name: str, seed: int, data):
    path = checkpoint_path(dataset, CONFIGS[name]["checkpoint"], seed)
    model = build_variant(dataset, name, seed, data)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model
    if name != "uniform_mean":
        raise FileNotFoundError(f"Expected final checkpoint is missing: {path}")

    set_seed(seed)
    cfg = DS_CONFIG[dataset]
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
    save_model(model, str(path))
    return model


def evaluate_variant(dataset: str, name: str, seed: int, data) -> Dict[str, object]:
    model = get_or_train_variant(dataset, name, seed, data)
    val_prob, _ = mc_predict(model, data["x_val"], samples=MC_SAMPLES, batch_size=512)
    test_prob, _ = mc_predict(model, data["x_test"], samples=MC_SAMPLES, batch_size=512)
    threshold, _ = calibrate_threshold(data["y_val"], val_prob)
    return {
        "dataset": dataset,
        "seed": seed,
        "pooling": name,
        "threshold": threshold,
        **evaluate_binary(data["y_test"], test_prob, threshold),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["f2", "auc_roc", "auc_pr", "brier", "ece"]
    summary = frame.groupby(["dataset", "pooling"])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    comparisons = {
        "learned_attention_minus_uniform_mean": (
            "learned_attention",
            "uniform_mean",
        ),
        "terminal_state_minus_uniform_mean": ("terminal_state", "uniform_mean"),
    }
    for dataset in DATASETS:
        pivot = frame[frame["dataset"] == dataset].pivot(
            index="seed", columns="pooling", values="f2"
        )
        for contrast, (left, right) in comparisons.items():
            values = pivot[left] - pivot[right]
            rows.append(
                {
                    "dataset": dataset,
                    "contrast": contrast,
                    "mean_f2_difference": float(values.mean()),
                    "sd_f2_difference": float(values.std(ddof=1)),
                    "min_f2_difference": float(values.min()),
                    "max_f2_difference": float(values.max()),
                    "n_seeds": int(values.count()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        data = load_dataset(dataset)
        for seed in SEEDS:
            for name in CONFIGS:
                rows.append(evaluate_variant(dataset, name, seed, data))
            print(f"{dataset}: completed pooling controls for seed {seed}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "pooling_control_5seeds.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(RESULTS / "pooling_control_5seeds_summary.csv", index=False)
    effects = paired_effects(frame)
    effects.to_csv(RESULTS / "pooling_control_effects.csv", index=False)
    print(summary.to_string(index=False))
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
