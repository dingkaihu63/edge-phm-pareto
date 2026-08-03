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

MODEL_COLORS = {
    "full": "#B64A3A",
    "proposed_lite": "#277DA1",
    "lstm": "#6C757D",
    "tcn": "#7A6F5D",
    "timesnet": "#4F5965",
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
    df = pd.read_csv(RESULTS / "unit_bootstrap_hierarchical.csv")
    df = df[df["metric"] == "f2"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15), sharex=False)
    for ax, ds in zip(axes.ravel(), ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]):
        part = df[df["dataset"] == ds].copy()
        order = [
            ("full", "lstm"),
            ("full", "tcn"),
            ("full", "timesnet"),
            ("proposed_lite", "lstm"),
            ("proposed_lite", "tcn"),
            ("proposed_lite", "timesnet"),
        ]
        labels = []
        for y, (proposed, baseline) in enumerate(order):
            row = part[
                (part["proposed"] == proposed) & (part["baseline"] == baseline)
            ].iloc[0]
            labels.append(f"{MODEL_LABELS[proposed]} - {MODEL_LABELS[baseline]}")
            color = MODEL_COLORS[proposed]
            ax.errorbar(
                row["diff_mean"],
                y,
                xerr=[
                    [row["diff_mean"] - row["ci_low"]],
                    [row["ci_high"] - row["diff_mean"]],
                ],
                fmt="o",
                color=color,
                ms=4,
                elinewidth=1.1,
                capsize=2,
            )
        ax.axvline(0, color="#343A40", lw=0.8, ls=":")
        ax.set_yticks(range(6))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Paired F2 difference")
        ax.set_title(DATASET_LABELS[ds])
        ax.grid(axis="x", color="#E4E6E8", lw=0.6)
    fig.suptitle(
        "Hierarchical paired bootstrap over training seeds and physical units",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig_unit_bootstrap")


def fig_streaming_alert() -> None:
    df = pd.read_csv(RESULTS / "episode_alert_5seeds_summary.csv")
    df = df[df["confirm_windows"] == 2]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15), sharey=True)
    metrics = [
        ("on_time_rate", "On-time detection", "o", "#2A7F62"),
        ("premature_rate", "Premature alert", "s", "#D9893D"),
        ("missed_rate", "Missed episode", "x", "#B64A3A"),
        ("false_alert_rate", "False alert", "^", "#6C757D"),
    ]
    model_order = ["full", "proposed_lite", "lstm", "tcn", "timesnet"]
    for ax, ds in zip(axes.ravel(), ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]):
        part = df[df["dataset"] == ds].set_index("model")
        x = np.arange(len(model_order))
        for offset, (metric, label, marker, color) in zip(
            [-0.24, -0.08, 0.08, 0.24], metrics
        ):
            means = part.loc[model_order, f"{metric}_mean"].to_numpy(dtype=float)
            errors = part.loc[model_order, f"{metric}_std"].to_numpy(dtype=float)
            valid = np.isfinite(means)
            ax.errorbar(
                x[valid] + offset,
                means[valid],
                yerr=np.nan_to_num(errors[valid], nan=0.0),
                fmt=marker,
                color=color,
                ms=4,
                lw=0,
                elinewidth=0.9,
                capsize=2,
                label=label if ds == "cmapss_fd001" else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[name] for name in model_order], rotation=25, ha="right")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(DATASET_LABELS[ds])
        ax.grid(axis="y", color="#E4E6E8", lw=0.6)
        if ds == "xjtu":
            ax.text(
                0.02,
                0.05,
                "No negative test units",
                transform=ax.transAxes,
                fontsize=5.8,
                color="#6C757D",
            )
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode-level rate")
    axes[0, 0].legend(loc="upper right", fontsize=5.3, ncol=2)
    fig.suptitle("First-alert outcomes with two-window confirmation", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig_streaming_alert")


def fig_risk_coverage() -> None:
    df = pd.read_csv(RESULTS / "risk_curves.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharey=False)
    styles = {"MC": "-", "pseudo_abs": "--", "entropy": ":"}
    for ax, ds in zip(axes.ravel(), ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]):
        part = df[df["dataset"] == ds]
        for model in ["full", "proposed_lite"]:
            for unc, linestyle in styles.items():
                rows = part[(part["model"] == model) & (part["uncertainty"] == unc)].sort_values("actual_coverage")
                ax.plot(
                    rows["actual_coverage"],
                    rows["risk"],
                    color=MODEL_COLORS[model],
                    ls=linestyle,
                    lw=1.15,
                    label=f"{MODEL_LABELS[model]}: {unc}" if ds == "cmapss_fd001" else None,
                )
        ax.set_yscale("log")
        ax.set_title(DATASET_LABELS[ds])
        ax.set_xlabel("Coverage")
        ax.grid(which="both", color="#E4E6E8", lw=0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Error rate (risk)")
    axes[0, 0].legend(fontsize=5.2, ncol=2)
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
            rows = (
                df[df["method"] == method]
                .groupby("fraction", as_index=False)["drop_pos"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.plot(rows["fraction"], rows["mean"], color=color, ls=ls, lw=1.2, label=method)
            if method in {"high", "random"}:
                ax.fill_between(
                    rows["fraction"],
                    rows["mean"] - rows["std"],
                    rows["mean"] + rows["std"],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
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
    fig.suptitle("Temporal-deletion sensitivity across five training seeds", fontsize=9)
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
