"""Attention faithfulness and stability analysis on final checkpoints.

Deletion curves compare high-to-low, low-to-high, random, and randomized-attention
orderings. Five-seed attention stability is measured with Spearman correlation and
Jensen-Shannon divergence.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon

from common_no_tf import calibrate_threshold
from prepare_data import load_cmapss, load_ur3, load_xjtu
from run_experiments_torch import DS_CONFIG
from torch_common import (
    DEVICE,
    build_model,
    predict_attention,
    predict_proba,
    save_model,
    set_seed,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS_DIR = RESULTS / "models_torch"

DATASETS = ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]
SEEDS = [1, 2, 3, 4, 5]
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def dataset_dirs() -> Dict[str, str]:
    return {
        "ur3": os.environ.get("EDGE_PHM_UR3_DIR", ""),
        "cmapss": os.environ.get("EDGE_PHM_CMAPSS_DIR", ""),
        "xjtu": os.environ.get("EDGE_PHM_XJTU_DIR", ""),
        "xjtu_cache": os.environ.get("EDGE_PHM_XJTU_CACHE", ""),
    }


def load_dataset(ds: str):
    dirs = dataset_dirs()
    if ds == "ur3":
        return load_ur3(dirs["ur3"], seed_rolling=False)
    if ds == "cmapss_fd001":
        return load_cmapss(dirs["cmapss"], "FD001", seed_rolling=False)
    if ds == "cmapss_fd003":
        return load_cmapss(dirs["cmapss"], "FD003", seed_rolling=False)
    if ds == "xjtu":
        return load_xjtu(dirs["xjtu"], dirs["xjtu_cache"], seed_rolling=False)
    raise ValueError(ds)


def build_full(ds: str, time_steps: int, n_features: int, seed: int):
    cfg = DS_CONFIG[ds]
    model = build_model(
        time_steps,
        n_features,
        attention="sigmoid",
        mc_dropout=True,
        dropout_rate=cfg["dropout"],
        attn_temperature=cfg["tau"],
        lstm_units_1=cfg["units1"],
        lstm_units_2=cfg["units2"],
        seed=seed,
    )
    return model


def get_full_model(ds: str, data, seed: int):
    t, f = data["x_val"].shape[1], data["x_val"].shape[2]
    model = build_full(ds, t, f, seed)
    path = MODELS_DIR / f"{ds}_full_seed{seed}.pt"
    if seed == 1 and (MODELS_DIR / f"{ds}_full.pt").exists():
        model.load_state_dict(torch.load(MODELS_DIR / f"{ds}_full.pt", map_location=DEVICE))
        return model
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        return model
    cfg = DS_CONFIG[ds]
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


def build_softmax(ds: str, time_steps: int, n_features: int):
    cfg = DS_CONFIG[ds]
    model = build_model(
        time_steps,
        n_features,
        attention="softmax",
        mc_dropout=True,
        dropout_rate=cfg["dropout"],
        attn_temperature=cfg["tau"],
        lstm_units_1=cfg["units1"],
        lstm_units_2=cfg["units2"],
        seed=1,
    )
    model.load_state_dict(
        torch.load(MODELS_DIR / f"{ds}_softmax_attention.pt", map_location=DEVICE)
    )
    model.eval()
    return model


def masked_predict(model, x: np.ndarray, order: np.ndarray, k: int) -> np.ndarray:
    xm = x.copy()
    for i in range(len(x)):
        xm[i, order[i, :k], :] = 0.0
    return predict_proba(model, xm, batch_size=512)


def _random_orders(alpha: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    order = np.tile(np.arange(alpha.shape[1]), (len(alpha), 1))
    for i in range(len(order)):
        rng.shuffle(order[i])
    return order


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).correlation)


def _jsd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.clip(a / a.sum(), 1e-12, 1.0)
    b = np.clip(b / b.sum(), 1e-12, 1.0)
    return float(jensenshannon(a, b, base=2.0))


def run_attention(ds: str) -> None:
    data = load_dataset(ds)
    x_val, y_val, x_test, y_test = (
        data["x_val"],
        data["y_val"],
        data["x_test"],
        data["y_test"],
    )
    t = x_test.shape[1]
    rng = np.random.RandomState(7)
    pos_mask = np.asarray(y_test) == 1
    deletion_rows: List[Dict[str, object]] = []
    stability_rows: List[Dict[str, object]] = []
    alphas_by_seed: Dict[int, np.ndarray] = {}

    for seed in SEEDS:
        model = get_full_model(ds, data, seed)
        p_base, alpha = predict_attention(model, x_test)
        alphas_by_seed[seed] = alpha
        for frac in FRACTIONS:
            k = max(1, int(round(t * frac)))
            high = np.argsort(-alpha, axis=1)[:, :k]
            low = np.argsort(alpha, axis=1)[:, :k]
            rand = _random_orders(alpha, rng)
            p_high = masked_predict(model, x_test, high, k)
            p_low = masked_predict(model, x_test, low, k)
            p_rand = masked_predict(model, x_test, rand, k)
            drop = lambda p: p_base - p  # noqa: E731
            deletion_rows.append(
                {
                    "dataset": ds,
                    "model_type": "full",
                    "seed": seed,
                    "method": "high",
                    "fraction": frac,
                    "k": k,
                    "drop_pos": float(drop(p_high)[pos_mask].mean()),
                    "drop_all_abs": float(np.abs(drop(p_high)).mean()),
                }
            )
            deletion_rows.append(
                {
                    "dataset": ds,
                    "model_type": "full",
                    "seed": seed,
                    "method": "low",
                    "fraction": frac,
                    "k": k,
                    "drop_pos": float(drop(p_low)[pos_mask].mean()),
                    "drop_all_abs": float(np.abs(drop(p_low)).mean()),
                }
            )
            deletion_rows.append(
                {
                    "dataset": ds,
                    "model_type": "full",
                    "seed": seed,
                    "method": "random",
                    "fraction": frac,
                    "k": k,
                    "drop_pos": float(drop(p_rand)[pos_mask].mean()),
                    "drop_all_abs": float(np.abs(drop(p_rand)).mean()),
                }
            )
        # Randomized-attention ordering sanity check.
        model_rand = copy.deepcopy(model)
        set_seed(seed + 1000)
        torch.nn.init.uniform_(model_rand.attn_score.weight, -0.5, 0.5)
        torch.nn.init.uniform_(model_rand.attn_score.bias, -0.5, 0.5)
        _, alpha_rand = predict_attention(model_rand, x_test)
        for frac in FRACTIONS:
            k = max(1, int(round(t * frac)))
            rand_attn_order = np.argsort(-alpha_rand, axis=1)[:, :k]
            p_rand_attn = masked_predict(model, x_test, rand_attn_order, k)
            deletion_rows.append(
                {
                    "dataset": ds,
                    "model_type": "full",
                    "seed": seed,
                    "method": "randomized_attn",
                    "fraction": frac,
                    "k": k,
                    "drop_pos": float((p_base - p_rand_attn)[pos_mask].mean()),
                    "drop_all_abs": float(np.abs(p_base - p_rand_attn).mean()),
                }
            )
        mean_spearman = np.nanmean([_spearman(alpha[i], alpha_rand[i]) for i in range(len(alpha))])
        stability_rows.append(
            {
                "dataset": ds,
                "seed_a": seed,
                "seed_b": seed,
                "kind": "randomized_attn",
                "mean_spearman": mean_spearman,
                "mean_jsd": float("nan"),
            }
        )
        print(ds, "seed", seed, "done", flush=True)

    # Softmax attention deletion comparison (seed-1 checkpoint).
    model_soft = build_softmax(ds, x_test.shape[1], x_test.shape[2])
    p_soft, alpha_soft = predict_attention(model_soft, x_test)
    for frac in FRACTIONS:
        k = max(1, int(round(t * frac)))
        high = np.argsort(-alpha_soft, axis=1)[:, :k]
        low = np.argsort(alpha_soft, axis=1)[:, :k]
        rand = _random_orders(alpha_soft, rng)
        for method, order in (("high", high), ("low", low), ("random", rand)):
            p_m = masked_predict(model_soft, x_test, order, k)
            deletion_rows.append(
                {
                    "dataset": ds,
                    "model_type": "softmax",
                    "seed": 1,
                    "method": method,
                    "fraction": frac,
                    "k": k,
                    "drop_pos": float((p_soft - p_m)[pos_mask].mean()),
                    "drop_all_abs": float(np.abs(p_soft - p_m).mean()),
                }
            )

    # Cross-seed stability.
    for s in SEEDS[1:]:
        mean_sp = np.nanmean(
            [_spearman(alphas_by_seed[1][i], alphas_by_seed[s][i]) for i in range(len(alphas_by_seed[1]))]
        )
        mean_jsd = float(
            np.mean([_jsd(alphas_by_seed[1][i], alphas_by_seed[s][i]) for i in range(len(alphas_by_seed[1]))])
        )
        stability_rows.append(
            {
                "dataset": ds,
                "seed_a": 1,
                "seed_b": s,
                "kind": "cross_seed",
                "mean_spearman": mean_spearman if False else mean_sp,
                "mean_jsd": mean_jsd,
            }
        )

    pd.DataFrame(deletion_rows).to_csv(RESULTS / f"attention_deletion_{ds}.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(RESULTS / f"attention_stability_{ds}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    args = parser.parse_args()
    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        run_attention(ds)


if __name__ == "__main__":
    main()