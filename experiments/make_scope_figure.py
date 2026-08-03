"""Capability-aware resource/performance figure from final results."""

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
    "w/o_attention": "No-attn",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "tcn": "TCN",
    "patchtst": "PatchTST",
    "timesnet": "TimesNet",
    "matched_lstm": "mLSTM",
    "matched_gru": "mGRU",
    "matched_tcn": "mTCN",
}
MC_CAPABLE = ["full", "proposed_lite", "w/o_attention"]


def _pareto(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["params", "f2_mean"], ascending=[True, False])
    keep = []
    best = -np.inf
    for idx, row in frame.iterrows():
        if row["f2_mean"] > best:
            keep.append(idx)
            best = row["f2_mean"]
    return frame.loc[keep].sort_values("params")


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seeds = pd.read_csv(RESULTS / "results_seeds.csv")
    dep = pd.read_csv(RESULTS / "deployment_torch.csv")
    matched = pd.read_csv(RESULTS / "matched_budget_seeds.csv")

    def mean_params(ds: str, model: str) -> float:
        row = dep[(dep["dataset"] == ds) & (dep["model"] == model)]
        if len(row):
            return float(row["params"].iloc[0])
        return float("nan")

    cmapss_rows = []
    for ds in ["cmapss_fd001", "cmapss_fd003"]:
        for model in list(MODEL_LABELS):
            s = seeds[(seeds["dataset"] == ds) & (seeds["model"] == model)]
            if len(s) == 0:
                continue
            p = mean_params(ds, "proposed" if model == "full" else model)
            if model == "w/o_attention":
                p = mean_params(ds, "proposed")
            cmapss_rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "params": p,
                    "f2_mean": s["f2"].mean(),
                    "f2_sd": s["f2"].std(),
                    "mc_capable": model in MC_CAPABLE,
                }
            )
        for model in ["matched_lstm", "matched_gru", "matched_tcn"]:
            s = matched[(matched["dataset"] == ds) & (matched["model"] == model)]
            if len(s):
                cmapss_rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "params": s["params"].mean(),
                        "f2_mean": s["f2"].mean(),
                        "f2_sd": s["f2"].std(),
                        "mc_capable": False,
                    }
                )
    cmapss = (
        pd.DataFrame(cmapss_rows)
        .groupby("model", as_index=False)
        .agg(params=("params", "mean"), f2_mean=("f2_mean", "mean"), f2_sd=("f2_sd", "mean"), mc_capable=("mc_capable", "first"))
    )

    matched_rows = []
    for ds in DATASET_LABELS:
        lite = seeds[(seeds["dataset"] == ds) & (seeds["model"] == "proposed_lite")]["f2"].mean()
        matched_rows.append(
            {
                "dataset": ds,
                "model": "proposed_lite",
                "f2_mean": lite,
                "f2_sd": seeds[(seeds["dataset"] == ds) & (seeds["model"] == "proposed_lite")]["f2"].std(),
            }
        )
        for model in ["matched_lstm", "matched_gru", "matched_tcn"]:
            s = matched[(matched["dataset"] == ds) & (matched["model"] == model)]
            matched_rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "f2_mean": s["f2"].mean(),
                    "f2_sd": s["f2"].std(),
                }
            )
    matched_plot = pd.DataFrame(matched_rows)

    attention = seeds[seeds["model"].isin(["full", "w/o_attention"])].groupby(
        ["dataset", "model"], as_index=False
    )["f2"].mean().pivot(index="dataset", columns="model", values="f2").reset_index()
    attention["delta_f2"] = attention["full"] - attention["w/o_attention"]
    return cmapss, matched_plot, attention


def make_figure() -> None:
    cmapss, matched_plot, attention = _load()
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
    accent = "#C84B31"
    neutral = "#6F7782"
    signal = "#1F6F8B"
    positive = "#2D7D46"

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    cmapss["params"] = cmapss["params"].clip(lower=1.0)
    front = _pareto(cmapss[cmapss["mc_capable"]])
    ax.plot(front["params"], front["f2_mean"], color=accent, lw=1.3, zorder=1)
    for _, row in cmapss.iterrows():
        marker = "o" if row["mc_capable"] else "s"
        color = accent if row["mc_capable"] else neutral
        ax.scatter(row["params"], row["f2_mean"], marker=marker, s=38 if row["mc_capable"] else 24,
                   color=color, edgecolor="white", linewidth=0.4, zorder=5 if row["mc_capable"] else 2,
                   alpha=0.95 if row["mc_capable"] else 0.75)
    for model, (dx, dy) in {
        "proposed_lite": (3, -11),
        "w/o_attention": (4, 5),
        "full": (4, -15),
        "timesnet": (-28, 5),
        "matched_tcn": (4, -11),
        "matched_gru": (4, 5),
    }.items():
        if model in cmapss["model"].values:
            row = cmapss[cmapss["model"] == model].iloc[0]
            ax.annotate(MODEL_LABELS[model], (row["params"], row["f2_mean"]),
                        xytext=(dx, dy), textcoords="offset points", fontsize=5.8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Mean F2 (FD001/FD003)")
    ax.set_title("C-MAPSS: capability-aware view")
    ax.grid(axis="y", color="#E4E6E8", lw=0.6)

    ax = axes[1]
    datasets = ["cmapss_fd001", "cmapss_fd003", "xjtu", "ur3"]
    order = ["proposed_lite", "matched_lstm", "matched_gru", "matched_tcn"]
    labels = ["Lite", "mLSTM", "mGRU", "mTCN"]
    x = np.arange(len(datasets))
    width = 0.19
    for j, model in enumerate(order):
        vals = [
            matched_plot[(matched_plot["dataset"] == ds) & (matched_plot["model"] == model)]["f2_mean"].iloc[0]
            for ds in datasets
        ]
        bars = ax.bar(x + (j - 1.5) * width, vals, width, color=[accent, neutral, neutral, signal][j], alpha=0.9)
        if model == "proposed_lite":
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=5.2)
    ax.set_xticks(x, [DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel("Mean F2 (10 seeds)")
    ax.set_title("Matched-budget comparison")
    ax.set_ylim(0.45, 0.95)
    ax.grid(axis="y", color="#E4E6E8", lw=0.6)

    ax = axes[2]
    order = ["cmapss_fd001", "cmapss_fd003", "ur3", "xjtu"]
    attention = attention.set_index("dataset").loc[order].reset_index()
    y = np.arange(len(attention))
    colors = [positive if v > 0 else accent for v in attention["delta_f2"]]
    ax.barh(y, attention["delta_f2"], color=colors, height=0.58)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y, [DATASET_LABELS[name] for name in attention["dataset"]])
    ax.invert_yaxis()
    ax.set_xlabel(r"Attention effect, $\Delta$F2 (full - no attention)")
    ax.set_title("Interpretability has a regime cost")
    for yi, v in zip(y, attention["delta_f2"]):
        ax.text(v + (0.002 if v >= 0 else -0.002), yi, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=6)
    ax.set_xlim(-0.065, 0.015)
    ax.grid(axis="x", color="#E4E6E8", lw=0.6)

    for label, axx in zip("abc", axes):
        axx.text(-0.16, 1.08, label, transform=axx.transAxes, fontweight="bold", fontsize=8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.2, top=0.86)
    fig.savefig(FIGURES / "fig_scope_pareto.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_scope_pareto.svg", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_scope_pareto.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_scope_pareto.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    print("capability-aware scope figure written")