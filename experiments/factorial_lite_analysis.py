"""Five-seed 2x2 analysis of temporal attention and MC-Dropout at Lite budget."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from common_no_tf import calibrate_threshold, evaluate_binary
from five_seed_bootstrap import DATASETS, RESULTS, SEEDS, load_dataset
from run_experiments_torch import DS_CONFIG, LITE_CONFIG
from torch_common import (
    DEVICE,
    build_model,
    mc_predict,
    predict_proba,
    save_model,
    set_seed,
    train_model,
)

MODELS_DIR = RESULTS / "models_torch"
MC_SAMPLES = 50
CONFIGS = {
    "attention_mc": {
        "attention": "sigmoid",
        "mc_dropout": True,
        "checkpoint": "proposed_lite",
    },
    "no_attention_mc": {
        "attention": "none",
        "mc_dropout": True,
        "checkpoint": "lite_w_o_attention",
    },
    "attention_no_mc": {
        "attention": "sigmoid",
        "mc_dropout": False,
        "checkpoint": "lite_w_o_mc_dropout",
    },
    "no_attention_no_mc": {
        "attention": "none",
        "mc_dropout": False,
        "checkpoint": "lite_no_attn_no_mc",
    },
}


def checkpoint_path(dataset: str, checkpoint: str, seed: int) -> Path:
    suffix = "" if seed == 1 else f"_seed{seed}"
    return MODELS_DIR / f"{dataset}_{checkpoint}{suffix}.pt"


def build_variant(dataset: str, name: str, seed: int, data):
    spec = CONFIGS[name]
    cfg = {**DS_CONFIG[dataset], **LITE_CONFIG[dataset]}
    time_steps, n_features = data["x_val"].shape[1:]
    return build_model(
        time_steps,
        n_features,
        attention=spec["attention"],
        mc_dropout=spec["mc_dropout"],
        dropout_rate=cfg["dropout"],
        attn_temperature=cfg["tau"],
        lstm_units_1=cfg["units1"],
        lstm_units_2=cfg["units2"],
        seed=seed,
    )


def get_or_train_variant(dataset: str, name: str, seed: int, data):
    spec = CONFIGS[name]
    path = checkpoint_path(dataset, spec["checkpoint"], seed)
    model = build_variant(dataset, name, seed, data)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model

    if name != "no_attention_no_mc":
        raise FileNotFoundError(f"Expected final checkpoint is missing: {path}")

    set_seed(seed)
    cfg = {**DS_CONFIG[dataset], **LITE_CONFIG[dataset]}
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
    if CONFIGS[name]["mc_dropout"]:
        val_prob, _ = mc_predict(
            model, data["x_val"], samples=MC_SAMPLES, batch_size=512
        )
        test_prob, _ = mc_predict(
            model, data["x_test"], samples=MC_SAMPLES, batch_size=512
        )
    else:
        val_prob = predict_proba(model, data["x_val"], batch_size=512)
        test_prob = predict_proba(model, data["x_test"], batch_size=512)
    threshold, _ = calibrate_threshold(data["y_val"], val_prob)
    metrics = evaluate_binary(data["y_test"], test_prob, threshold)
    return {
        "dataset": dataset,
        "seed": seed,
        "configuration": name,
        "attention": CONFIGS[name]["attention"] != "none",
        "mc_inference": CONFIGS[name]["mc_dropout"],
        "threshold": threshold,
        **metrics,
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["f2", "auc_roc", "auc_pr", "brier", "ece"]
    summary = frame.groupby(["dataset", "configuration"])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def component_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    comparisons = {
        "mc_effect_with_attention": ("attention_mc", "attention_no_mc"),
        "mc_effect_without_attention": ("no_attention_mc", "no_attention_no_mc"),
        "attention_effect_with_mc": ("attention_mc", "no_attention_mc"),
        "attention_effect_without_mc": ("attention_no_mc", "no_attention_no_mc"),
    }
    for dataset in DATASETS:
        pivot = frame[frame["dataset"] == dataset].pivot(
            index="seed", columns="configuration", values="f2"
        )
        for effect, (left, right) in comparisons.items():
            values = pivot[left] - pivot[right]
            rows.append(
                {
                    "dataset": dataset,
                    "effect": effect,
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
            print(f"{dataset}: completed seed {seed}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "factorial_lite_5seeds.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(RESULTS / "factorial_lite_5seeds_summary.csv", index=False)
    effects = component_effects(frame)
    effects.to_csv(RESULTS / "factorial_lite_effects.csv", index=False)
    print(summary.to_string(index=False))
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
