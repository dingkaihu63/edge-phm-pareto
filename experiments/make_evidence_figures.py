"""Generate evidence figures for the revised journal manuscripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATASET_LABELS = {
    "cmapss_fd001": "FD001",
    "cmapss_fd003": "FD003",
    "xjtu": "XJTU-SY",
    "ur3": "UR3",
}
MODEL_LABELS = {
    "full": "Full",
    "proposed_lite": "Lite",
    "lstm": "LSTM",
    "tcn": "TCN",
    "timesnet": "TimesNet",
    "matched_lstm": "mLSTM",
    "matched_gru": "mGRU",
    "matched_tcn": "mTCN",
}


def _style() -> None:
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


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_unit_bootstrap() -> None:
    path5 = RESULTS / "unit_bootstrap_5seeds_summary.csv"
    df = pd.read_csv(path5) if path5.exists() else pd.read_csv(RESULTS / "unit_bootstrap_5000.csv")
    df = df[df["metric"] == "f2"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=False)
    for ax, ds in zip(axes.ravel(), ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]):
        part = df[df["dataset"] == ds]
        y = 0
        for prop in ["full", "proposed_lite"]:
            rows = part[part["proposed"] == prop]
            for _, row in rows.iterrows():
                color = "#C84B31" if prop == "full" else "#1F6F8B"
                ax.errorbar(
                    row["ci_low"], y, xerr=[[row["diff_mean"] - row["ci_low"]], [row["ci_high"] - row["diff_mean"]]],
                    fmt="o", color=color, ms=4, elinewidth=1.0, capsize=2,
                )
                ax.axvline(0, color="#444444", lw=0.7)
                y += 1
        ax.set_yticks(range(6))
        ax.set_yticklabels(
            [f"{MODEL_LABELS[r['proposed']]}-{MODEL_LABELS[r['baseline']]}"
             for _, r in part.iterrows()],
            fontsize=6,
        )
        ax.set_xlabel("F2 difference (proposed - baseline)")
        ax.set_title(DATASET_LABELS[ds])
        ax.grid(axis="x", color="#E4E6E8", lw=0.6)
    fig.suptitle("Unit-level paired bootstrap (5,000 resamples)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig_unit_bootstrap")


def fig_streaming_alert() -> None:
    df = pd.read_csv(RESULTS / "streaming_alert.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=True)
    for ax, ds in zip(axes.ravel(), ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]):
        part = df[(df["dataset"] == ds) & (df["model"].isin(MODEL_LABELS))]
        for model, color in [
            ("full", "#C84B31"),
            ("proposed_lite", "#1F6F8B"),
            ("lstm", "#888888"),
            ("tcn", "#9A8F7F"),
            ("timesnet", "#6F7782"),
        ]:
            rows = part[part["model"] == model].sort_values("confirm_windows")
            ax.plot(rows["confirm_windows"], rows["median_lead"], marker="o", ms=3.5, lw=1.2, color=color, label=MODEL_LABELS[model])
        ax.axhline(0, color="#444444", lw=0.7, ls=":")
        ax.set_xticks([1, 2, 3])
        ax.set_title(DATASET_LABELS[ds])
        ax.grid(axis="y", color="#E4E6E8", lw=0.6)
        ax.set_xlabel("Consecutive windows")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median lead time (windows)")
    axes[0, 0].legend(loc="upper right", fontsize=5.5, ncol=2)
    fig.suptitle("Streaming alert utility under cost-optimal thresholds", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig_streaming_alert")


def fig_risk_coverage() -> None:
    df = pd.read_csv(RESULTS / "risk_curves.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharey=False)
    styles = {
        "MC": ("#C84B31", "-"),
        "pseudo_abs": ("#1F6F8B", "-"),
        "entropy": ("#6F7782", "--"),
    }
    for ax, ds in zip(axes.ravel(), ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]):
        part = df[df["dataset"] == ds]
        for model in ["full", "proposed_lite"]:
            for unc, (color, ls) in styles.items():
                rows = part[(part["model"] == model) & (part["uncertainty"] == unc)].sort_values("actual_coverage")
                ax.plot(rows["actual_coverage"], rows["risk"], color=color, ls=ls, lw=1.0,
                        label=f"{model}-{unc}" if ds == "cmapss_fd001" else None)
        ax.set_yscale("log")
        ax.set_title(DATASET_LABELS[ds])
        ax.set_xlabel("Coverage")
        ax.grid(which="both", color="#E4E6E8", lw=0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Error rate (risk)")
    axes[0, 0].legend(fontsize=5.5, ncol=2)
    fig.suptitle("Risk-coverage curves at K=50 MC samples", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig_risk_coverage")


def fig_attention_faithfulness() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    methods = {
        "high": ("#C84B31", "-"),
        "low": ("#888888", "--"),
        "random": ("#9A8F7F", ":"),
        "randomized_attn": ("#1F6F8B", "-."),
    }
    for ax, ds in zip(axes.ravel(), ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]):
        df = pd.read_csv(RESULTS / f"attention_deletion_{ds}.csv")
        df = df[df["model_type"] == "full"]
        for method, (color, ls) in methods.items():
            rows = df[df["method"] == method].groupby("fraction", as_index=False)["drop_pos"].mean()
            ax.plot(rows["fraction"], rows["drop_pos"], color=color, ls=ls, lw=1.2, label=method)
        soft = pd.read_csv(RESULTS / f"attention_deletion_{ds}.csv")
        soft = soft[(soft["model_type"] == "softmax") & (soft["method"] == "high")]
        if len(soft):
            rows = soft.groupby("fraction", as_index=False)["drop_pos"].mean()
            ax.plot(rows["fraction"], rows["drop_pos"], color="#2D7D46", lw=1.2, ls="--", label="softmax-high")
        ax.set_title(DATASET_LABELS[ds])
        ax.set_xlabel("Fraction of time steps removed")
        ax.grid(color="#E4E6E8", lw=0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean positive-class prob. drop")
    axes[0, 0].legend(fontsize=5.5, ncol=2)
    fig.suptitle("Attention deletion: high-attention masking should remove more evidence", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig_attention_faithfulness")


def main() -> None:
    _style()
    fig_unit_bootstrap()
    fig_streaming_alert()
    fig_risk_coverage()
    fig_attention_faithfulness()
    print("evidence figures written")


if __name__ == "__main__":
    main()