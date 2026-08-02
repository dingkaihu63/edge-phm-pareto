"""Publication figures for the PyTorch final experiments."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from common_no_tf import evaluate_binary
from prepare_data import (
    _load_xjtu_features,
    _read_cmapss,
    load_cmapss,
    load_ur3,
    load_xjtu,
)
from torch_common import (
    ProposedModel,
    build_model,
    predict_attention,
    set_seed,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "..", "results")
PRED = os.path.join(RESULTS, "predictions")
MODELS = os.path.join(RESULTS, "models_torch")
FIG = os.path.join(ROOT, "..", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

DS_CFG = {
    "ur3": dict(lstm_units_1=96, lstm_units_2=48, dropout_rate=0.10),
    "cmapss_fd001": dict(lstm_units_1=64, lstm_units_2=32, dropout_rate=0.15),
    "cmapss_fd003": dict(lstm_units_1=64, lstm_units_2=32, dropout_rate=0.15),
    "xjtu": dict(lstm_units_1=96, lstm_units_2=48, dropout_rate=0.10),
}


def safe(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def load_pred(dataset: str, model: str):
    npz = np.load(os.path.join(PRED, f"{dataset}_{safe(model)}.npz"))
    return npz["p"], npz["y"]


def save_fig(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))



def fig_dataset_overview() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.0))
    data = load_ur3(r"C:\Users\hu\Desktop\比赛")
    raw = data["raw_df"]
    ax = axes[0]
    x = np.arange(len(raw))
    ax.plot(x, raw["Current_J3"].values, lw=0.6, color="#1f77b4", label="Current_J3")
    ax.plot(x, raw["Temperature_J3"].values / 50, lw=0.6, color="#d62728", label="Temperature_J3/50")
    fidx = np.where(raw["Label"].values == 1)[0]
    ax.scatter(fidx, np.full_like(fidx, -6.0), s=0.4, color="black", label="fault")
    ax.set_xlabel("Time index (s)")
    ax.set_title("UR3 CobotOps telemetry")
    ax.legend(loc="upper left", fontsize=6)

    train, _ = _read_cmapss(r"E:\datasets\C-MAPSS", "FD001")
    eng = train[train["unit"] == 1].copy()
    ax = axes[1]
    ax.plot(eng["cycle"], eng["s12"], lw=0.8, color="#2ca02c", label="s12")
    ax.plot(eng["cycle"], eng["s4"] / 100, lw=0.8, color="#9467bd", label="s4/100")
    ax.set_xlabel("Cycle")
    ax.set_title("C-MAPSS FD001 run-to-failure")
    ax.legend(loc="upper left", fontsize=6)

    feat = _load_xjtu_features(
        r"E:\datasets\XJTU-SY\original",
        r"E:\datasets\XJTU-SY\xjtu_features_full15.csv",
    )
    ax = axes[2]
    for b in (1, 5, 10, 15):
        part = feat[feat["bearing"] == b]
        ax.plot(part["block"], part["ch0_rms"], lw=0.7, label=f"bearing {b}")
    ax.set_xlabel("Block (1 min)")
    ax.set_title("XJTU-SY vibration RMS")
    ax.legend(loc="upper left", fontsize=6)

    fig.tight_layout()
    save_fig(fig, "fig_datasets")


def fig_roc_pr() -> None:
    datasets = ["ur3", "cmapss_fd001", "cmapss_fd003", "xjtu"]
    labels = ["UR3", "C-MAPSS FD001", "C-MAPSS FD003", "XJTU-SY"]
    models = ["full", "lstm", "gru", "transformer", "tcn", "random_forest", "gradient_boosting"]
    style = {
        "full": ("proposed", "-", "#d62728", 2.2),
        "lstm": ("LSTM", "--", "#1f77b4", 1.5),
        "gru": ("GRU", "-.", "#2ca02c", 1.5),
        "transformer": ("Transformer", ":", "#9467bd", 1.5),
        "tcn": ("TCN", "-.", "#17becf", 1.5),
        "random_forest": ("RF", "--", "#ff7f0e", 1.5),
        "gradient_boosting": ("GBM", "-.", "#8c564b", 1.5),
    }
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for j, (ds, lab) in enumerate(zip(datasets, labels)):
        ax = axes[0, j]
        for m in models:
            try:
                p, y = load_pred(ds, m)
            except FileNotFoundError:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            name, ls, color, lw = style[m]
            ax.plot(fpr, tpr, ls, color=color, lw=lw, label=name)
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{lab} ROC")
        ax.legend(fontsize=6, loc="lower right")

        ax = axes[1, j]
        for m in models:
            try:
                p, y = load_pred(ds, m)
            except FileNotFoundError:
                continue
            prec, rec, _ = precision_recall_curve(y, p)
            name, ls, color, lw = style[m]
            ax.plot(rec, prec, ls, color=color, lw=lw, label=name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{lab} PR")
        ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig_roc_pr")



def fig_ablations() -> None:
    df = pd.read_csv(os.path.join(RESULTS, "results_ensemble.csv"))
    order = [
        "full",
        "w/o_attention",
        "softmax_attention",
        "w/o_mc_dropout",
        "w/o_class_weight",
        "w/_rolling_stats",
    ]
    short = {
        "full": "Full",
        "w/o_attention": "w/o attn",
        "softmax_attention": "softmax",
        "w/o_mc_dropout": "w/o MC",
        "w/o_class_weight": "w/o CW",
        "w/_rolling_stats": "+ rolling",
    }
    datasets = ["ur3", "cmapss_fd001", "cmapss_fd003", "xjtu"]
    labels = ["UR3", "FD001", "FD003", "XJTU"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, ds, lab in zip(axes.ravel(), datasets, labels):
        part = df[(df.dataset == ds) & df.model.isin(order)].set_index("model")
        part = part.reindex(order)
        x = np.arange(len(order))
        ax.plot(x, part["f2"], "-o", color="#d62728", label="F2")
        ax.plot(x, part["auc_roc"], "-s", color="#1f77b4", label="AUROC")
        ax.plot(x, part["auc_pr"], "-^", color="#2ca02c", label="AUPRC")
        ax.set_xticks(x)
        ax.set_xticklabels([short[m] for m in order], rotation=20, ha="right", fontsize=7)
        ax.set_ylim(0.2, 1.05)
        ax.set_title(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig_ablations")



def _reliability(ax, p, y, color, label):
    bins = np.linspace(0, 1, 11)
    conf, acc, count = [], [], []
    for i in range(10):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() == 0:
            continue
        conf.append(p[mask].mean())
        acc.append(y[mask].mean())
        count.append(mask.sum())
    if not conf:
        return
    ax.plot(conf, acc, "-o", color=color, label=label, markersize=3)
    for c, a, n in zip(conf, acc, count):
        ax.text(c, a + 0.02, str(n), fontsize=5, ha="center", color=color)


def fig_calibration() -> None:
    from sklearn.isotonic import IsotonicRegression
    datasets = ["ur3", "cmapss_fd001", "cmapss_fd003", "xjtu"]
    labels = ["UR3", "FD001", "FD003", "XJTU"]
    loaders = {
        "ur3": lambda: load_ur3(r"C:\Users\hu\Desktop\??", seed_rolling=False),
        "cmapss_fd001": lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD001", seed_rolling=False),
        "cmapss_fd003": lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD003", seed_rolling=False),
        "xjtu": lambda: load_xjtu(r"E:\datasets\XJTU-SY\original", r"E:\datasets\XJTU-SY\xjtu_features_full15.csv", seed_rolling=False),
    }
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, ds, lab in zip(axes.ravel(), datasets, labels):
        p, y = load_pred(ds, "full")
        data = loaders[ds]()
        npz = np.load(os.path.join(PRED, f"{ds}_full.npz"))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(npz["pv"], data["y_val"])
        p_iso = iso.predict(p)
        _reliability(ax, p, y, "#d62728", "raw")
        _reliability(ax, p_iso, y, "#1f77b4", "isotonic")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{lab} reliability")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig_calibration")


def fig_attention() -> None:
    specs = [
        ("ur3", load_ur3(r"C:\Users\hu\Desktop\比赛", seed_rolling=False), "UR3"),
        (
            "cmapss_fd001",
            load_cmapss(r"E:\datasets\C-MAPSS", "FD001", seed_rolling=False),
            "FD001",
        ),
        (
            "xjtu",
            load_xjtu(
                r"E:\datasets\XJTU-SY\original",
                r"E:\datasets\XJTU-SY\xjtu_features_full15.csv",
                seed_rolling=False,
            ),
            "XJTU",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    for ax, (ds, data, lab) in zip(axes, specs):
        model = build_model(
            data["x_train"].shape[1],
            data["x_train"].shape[2],
            seed=42,
            **DS_CFG[ds],
        )
        model.load_state_dict(
            __import__("torch").load(
                os.path.join(MODELS, f"{ds}_full.pt"),
                map_location=__import__("torch").device("cpu"),
            )
        )
        model.eval()
        idx = np.where(data["y_test"] == 1)[0][:200]
        _, attn = predict_attention(model, data["x_test"][idx])
        attn = np.asarray(attn).reshape(len(idx), -1)
        ax.plot(attn.mean(axis=0), color="#d62728", lw=1.6)
        ax.fill_between(
            np.arange(attn.shape[1]),
            attn.mean(axis=0) - attn.std(axis=0),
            attn.mean(axis=0) + attn.std(axis=0),
            alpha=0.25,
            color="#d62728",
        )
        ax.set_xlabel("Time step in window")
        ax.set_ylabel("Normalized attention")
        ax.set_title(f"{lab} attention profile")
        n_steps = attn.shape[1]
        ax.axvline(0.25 * n_steps, color="#555555", ls=":", lw=1.0)
        ax.axvline(0.75 * n_steps, color="#555555", ls=":", lw=1.0)
        ax.text(0.25 * n_steps + 0.2, ax.get_ylim()[1] * 0.92, "Early precursors", fontsize=7, color="#555555")
        ax.text(0.75 * n_steps + 0.2, ax.get_ylim()[1] * 0.92, "Near-failure phase", fontsize=7, color="#555555")
    fig.tight_layout()
    save_fig(fig, "fig_attention")



def fig_seed_warmup() -> None:
    data = load_ur3(r"C:\Users\hu\Desktop\比赛")
    raw = data["raw_df"].sort_values("Timestamp")
    col = "Current_J3"
    split = int(len(raw) * 0.7)
    train = raw.iloc[:split]
    val = raw.iloc[split:]
    seed = train.iloc[-9:][col]
    combined = pd.concat([seed, val[col]], ignore_index=True)
    seeded = combined.rolling(10, min_periods=1).std().iloc[9:].values
    naive = val[col].rolling(10, min_periods=1).std().fillna(0).values
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(seeded, color="#1f77b4", lw=1.5, label="seeded rolling std")
    ax.plot(naive, color="#d62728", lw=1.5, ls="--", label="naive rolling std")
    ax.set_xlabel("Validation index from split boundary")
    ax.set_ylabel("Rolling std (Current_J3)")
    ax.legend()
    ax.set_title("Seed warm-up removes boundary discontinuity")
    fig.tight_layout()
    save_fig(fig, "fig_seed_warmup")



def fig_shap_ur3() -> None:
    import shap
    import torch

    data = load_ur3(r"C:\Users\hu\Desktop\比赛", seed_rolling=False)
    full = build_model(
        data["x_train"].shape[1],
        data["x_train"].shape[2],
        seed=42,
        **DS_CFG["ur3"],
    )
    full.load_state_dict(torch.load(os.path.join(MODELS, "ur3_full.pt"), map_location="cpu"))
    det = ProposedModel(
        data["x_train"].shape[1],
        data["x_train"].shape[2],
        mc_dropout=False,
        lstm_units_1=DS_CFG["ur3"]["lstm_units_1"],
        lstm_units_2=DS_CFG["ur3"]["lstm_units_2"],
        dropout_rate=DS_CFG["ur3"]["dropout_rate"],
    )
    det.load_state_dict(full.state_dict())
    det.eval()
    rng = np.random.RandomState(0)
    bg = torch.from_numpy(data["x_train"][rng.choice(len(data["x_train"]), 100, replace=False)]).float()
    fault_idx = np.where(data["y_test"] == 1)[0][:60]
    x = torch.from_numpy(data["x_test"][fault_idx]).float()
    explainer = shap.GradientExplainer(det, bg)
    vals = np.asarray(explainer.shap_values(x)).reshape(x.shape)
    mean_abs = np.abs(vals).mean(axis=(0, 1))
    order = np.argsort(mean_abs)[::-1][:15]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    axes[0].barh(np.arange(len(order)), mean_abs[order][::-1], color="#d62728")
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([data["feature_names"][i] for i in order[::-1]], fontsize=9)
    axes[0].set_xlabel("Mean |SHAP|")
    axes[0].set_title("Feature importance at fault windows")
    top = order[:10]
    im = axes[1].imshow(vals[0, :, top].T, aspect="auto", cmap="RdBu_r")
    axes[1].set_yticks(np.arange(len(top)))
    axes[1].set_yticklabels([data["feature_names"][i] for i in top], fontsize=9)
    axes[1].set_xlabel("Time step")
    axes[1].set_title("SHAP heatmap (one fault window)")
    fig.colorbar(im, ax=axes[1], label="SHAP")
    fig.tight_layout()
    save_fig(fig, "fig_shap_ur3")





def fig_pareto() -> None:
    dep = pd.read_csv(os.path.join(RESULTS, "deployment_torch.csv"))
    ens = pd.read_csv(os.path.join(RESULTS, "results_ensemble.csv"))
    dep = dep.groupby("model", as_index=False).agg(params=("params", "mean"), cpu=("cpu_ms_single", "mean"))
    models = ["proposed", "proposed_lite", "lstm", "bilstm", "gru", "transformer", "tcn", "patchtst", "timesnet"]
    datasets = ["ur3", "cmapss_fd001", "cmapss_fd003", "xjtu"]
    labels = ["UR3", "FD001", "FD003", "XJTU"]
    short = {
        "proposed": "Proposed", "proposed_lite": "Lite", "lstm": "LSTM", "bilstm": "BiLSTM",
        "gru": "GRU", "transformer": "Trans.", "tcn": "TCN", "patchtst": "PatchTST", "timesnet": "TimesNet",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ds, lab in zip(axes.ravel(), datasets, labels):
        part = ens[(ens.dataset == ds) & (ens.model.isin(models))].set_index("model").reindex(models)
        merged = dep.set_index("model").reindex(models)
        mc_models = ["proposed", "proposed_lite"]
        for m in models:
            row = part.loc[m]
            drow = merged.loc[m]
            unc = m in mc_models
            marker = "o" if unc else "s"
            color = "#d62728" if unc else "#1f77b4"
            ax.scatter(drow.params, row.f2, s=40 + 18 * drow.cpu, marker=marker, color=color, alpha=0.85, edgecolors="black", linewidths=0.5, zorder=3)
            ax.annotate(short[m], (drow.params, row.f2), textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("F2")
        ax.set_title(f"{lab} edge-performance Pareto view")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.45, 1.0)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markeredgecolor="black", label="MC-Dropout uncertainty"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#1f77b4", markeredgecolor="black", label="Deterministic baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, "fig_pareto")


if __name__ == "__main__":
    fig_pareto()
    print("pareto figure done")
    fig_dataset_overview()
    print("datasets figure done")
    fig_roc_pr()
    print("roc/pr figure done")
    fig_ablations()
    print("ablation figure done")
    fig_calibration()
    print("calibration figure done")
    fig_attention()
    print("attention figure done")
    fig_seed_warmup()
    print("seed warmup figure done")
    fig_shap_ur3()
    print("shap figure done")
