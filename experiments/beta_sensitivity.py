"""False-negative threshold-weight sensitivity on final seed-1 checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

import evidence_analysis as ea
from common_no_tf import calibrate_threshold
from torch_common import predict_proba

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
BETAS = [1, 2, 3, 5]
DATASET_LABELS = {
    "cmapss_fd001": "FD001",
    "cmapss_fd003": "FD003",
    "xjtu": "XJTU-SY",
    "ur3": "UR3",
}


def run() -> None:
    rows: List[Dict[str, object]] = []
    for ds in ea.DATASETS:
        data = ea.load_dataset(ds)
        t, f = data["x_val"].shape[1], data["x_val"].shape[2]
        for model_name in ["full", "proposed_lite"]:
            model = ea.build_proposed(ds, model_name, t, f)
            pv = predict_proba(model, data["x_val"], batch_size=512)
            pt = predict_proba(model, data["x_test"], batch_size=512)
            yv = np.asarray(data["y_val"])
            yt = np.asarray(data["y_test"])
            pos = max(int(yt.sum()), 1)
            neg = max(int((yt == 0).sum()), 1)
            for beta in BETAS:
                threshold, _ = calibrate_threshold(yv, pv, beta=beta)
                yp = (pt >= threshold).astype(int)
                fp = int(((yt == 0) & (yp == 1)).sum())
                fn = int(((yt == 1) & (yp == 0)).sum())
                rows.append(
                    {
                        "dataset": ds,
                        "model": model_name,
                        "beta": beta,
                        "threshold": threshold,
                        "false_positive": fp,
                        "false_negative": fn,
                        "fp_rate": fp / neg,
                        "fn_rate": fn / pos,
                        "f2": fbeta_score(yt, yp, beta=2, zero_division=0),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "beta_sensitivity.csv", index=False)
    print(df.to_string(index=False))


def make_figure() -> None:
    df = pd.read_csv(RESULTS / "beta_sensitivity.csv")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=False)
    for ax, ds in zip(axes.ravel(), ea.DATASETS):
        part = df[df["dataset"] == ds]
        for model, color in [("full", "#C84B31"), ("proposed_lite", "#1F6F8B")]:
            p = part[part["model"] == model].sort_values("beta")
            ax.plot(p["beta"], p["fp_rate"], marker="o", ms=3.5, color=color, ls="-", label=f"{model} FP")
            ax.plot(p["beta"], p["fn_rate"], marker="s", ms=3.5, color=color, ls="--", label=f"{model} FN")
        ax.set_xticks(BETAS)
        ax.set_title(DATASET_LABELS[ds])
        ax.set_xlabel(r"Threshold weight $\beta$")
        ax.grid(axis="y", color="#E4E6E8", lw=0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Rate")
    axes[0, 0].legend(fontsize=5.5, ncol=2)
    fig.suptitle(r"Sensitivity to false-negative threshold weight $\beta$", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_beta_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_beta_sensitivity.svg", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_beta_sensitivity.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_beta_sensitivity.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
    make_figure()
    print("beta sensitivity done")
