"""Dataset loaders for UR3 CobotOps, NASA C-MAPSS and XJTU-SY."""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats

from common_no_tf import (
    add_rolling_std_seeded,
    make_windows,
    scale_train_val_test,
    split_chronological,
)


def load_ur3(
    data_dir: str,
    window: int = 10,
    roll_win: int = 10,
    seed_rolling: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, object]:
    xlsx = glob.glob(os.path.join(data_dir, "dataset_02052023.xlsx"))[0]
    df = pd.read_excel(xlsx)
    df.columns = df.columns.str.strip()
    df["Timestamp"] = (
        df["Timestamp"].astype(str).str.strip().str.strip('"').str.strip("'")
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    label = (
        (df["Robot_ProtectiveStop"].fillna(0.0) == 1.0)
        | (df["grip_lost"].fillna(False) == True)  # noqa: E712
    ).astype(int)
    exclude = {
        "Num",
        "Timestamp",
        "Robot_ProtectiveStop",
        "grip_lost",
        "Label",
    }
    base_cols = [c for c in df.columns if c not in exclude]
    sensor_cols = [c for c in base_cols if c != "cycle"]
    # Production-cycle-level chronological split. Cycles are independent
    # episodes; windows never cross cycle boundaries.
    cycle_order = (
        df.groupby("cycle", sort=False)["Timestamp"]
        .min()
        .sort_values()
        .index.tolist()
    )
    cycle_ids = {c: i for i, c in enumerate(cycle_order)}
    df = df[base_cols + ["Timestamp"]].copy()
    df["Label"] = label
    n_cycles = len(cycle_order)
    tr_end = int(n_cycles * train_ratio)
    va_end = int(n_cycles * (train_ratio + val_ratio))
    cycle_splits = {
        "train": cycle_order[:tr_end],
        "val": cycle_order[tr_end:va_end],
        "test": cycle_order[va_end:],
    }
    train_rows = df["cycle"].isin(cycle_splits["train"])
    medians = df.loc[train_rows, sensor_cols].median()
    empty = pd.DataFrame(columns=sensor_cols)

    all_x, all_y = {"train": [], "val": [], "test": []}, {"train": [], "val": [], "test": []}
    unit_ids = {"train": [], "val": [], "test": []}
    for split, cycles in cycle_splits.items():
        for cyc in cycles:
            part = df[df["cycle"] == cyc].sort_values("Timestamp")
            feat = part[sensor_cols].copy()
            if len(feat) < window:
                continue
            feat[:] = feat.ffill().fillna(medians)
            for col in sensor_cols:
                feat[col + "_diff"] = feat[col].diff().fillna(0.0)
            if seed_rolling:
                feat = add_rolling_std_seeded(feat, empty, roll_win, sensor_cols)
            x, yw = make_windows(
                feat.values.astype(np.float32), part["Label"].values, window
            )
            all_x[split].append(x)
            all_y[split].append(yw)
            unit_ids[split].append(np.full(len(x), cycle_ids[cyc], dtype=np.int32))

    x_tr = np.concatenate(all_x["train"])
    x_va = np.concatenate(all_x["val"])
    x_te = np.concatenate(all_x["test"])
    y_tr = np.concatenate(all_y["train"])
    y_va = np.concatenate(all_y["val"])
    y_te = np.concatenate(all_y["test"])
    u_tr = np.concatenate(unit_ids["train"])
    u_va = np.concatenate(unit_ids["val"])
    u_te = np.concatenate(unit_ids["test"])

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_tr.reshape(-1, x_tr.shape[2]))
    x_tr = scaler.transform(x_tr.reshape(-1, x_tr.shape[2])).reshape(x_tr.shape).astype(np.float32)
    x_va = scaler.transform(x_va.reshape(-1, x_va.shape[2])).reshape(x_va.shape).astype(np.float32)
    x_te = scaler.transform(x_te.reshape(-1, x_te.shape[2])).reshape(x_te.shape).astype(np.float32)
    feature_names = sensor_cols + [f"{c}_diff" for c in sensor_cols]
    if seed_rolling:
        feature_names += [f"{c}_rstd" for c in sensor_cols]
    return {
        "name": "UR3 CobotOps",
        "x_train": x_tr,
        "y_train": y_tr,
        "x_val": x_va,
        "y_val": y_va,
        "x_test": x_te,
        "y_test": y_te,
        "unit_ids": {"train": u_tr, "val": u_va, "test": u_te},
        "feature_names": feature_names,
        "scaler": scaler,
        "raw_df": df.assign(Label=label),
    }


CMAPSS_SENSOR_INDEX = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]


def _read_cmapss(base: str, fd: str) -> pd.DataFrame:
    cols = (
        ["unit", "cycle"]
        + [f"op{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)]
    )
    train = pd.read_csv(
        os.path.join(base, f"train_{fd}.txt"), sep=r"\s+", header=None, names=cols
    )
    test = pd.read_csv(
        os.path.join(base, f"test_{fd}.txt"), sep=r"\s+", header=None, names=cols
    )
    rul = pd.read_csv(
        os.path.join(base, f"RUL_{fd}.txt"),
        sep=r"\s+",
        header=None,
        names=["rul"],
    )
    test["rul_end"] = test["unit"].map(
        dict(zip(rul.index + 1, rul["rul"].values))
    )
    test["max_cycle"] = test.groupby("unit")["cycle"].transform("max")
    test["rul"] = test["rul_end"] + (test["max_cycle"] - test["cycle"])
    return train, test


def load_cmapss(
    data_dir: str,
    fd: str = "FD001",
    window: int = 30,
    horizon: int = 30,
    roll_win: int = 5,
    seed_rolling: bool = True,
    val_engines: int = 25,
) -> Dict[str, object]:
    train_raw, test_raw = _read_cmapss(data_dir, fd)
    train_raw["rul"] = (
        train_raw.groupby("unit")["cycle"].transform("max") - train_raw["cycle"]
    )
    sensor_cols = [f"s{i}" for i in CMAPSS_SENSOR_INDEX]

    def prepare(df: pd.DataFrame, is_train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = []
        labels = []
        units = []
        for _, eng in df.groupby("unit"):
            feat = eng[sensor_cols].copy()
            if seed_rolling:
                feat = add_rolling_std_seeded(
                    feat, pd.DataFrame(columns=feat.columns), roll_win, sensor_cols
                )
            y = (eng["rul"] <= horizon).astype(int).values
            x, yw = make_windows(feat.values.astype(np.float32), y, window)
            frames.append(x)
            labels.append(yw)
            units.append(np.full(len(x), int(eng["unit"].iloc[0]), dtype=np.int32))
        return (
            np.concatenate(frames, axis=0) if frames else np.zeros((0, window, 1)),
            np.concatenate(labels) if labels else np.zeros((0,), dtype=np.int32),
            np.concatenate(units) if units else np.zeros((0,), dtype=np.int32),
        )

    x_te_raw, y_te, u_te = prepare(test_raw, False)

    # Engine-wise validation split inside the official training set.
    engines = sorted(train_raw["unit"].unique())
    val_units = set(engines[-val_engines:])
    train_mask = train_raw["unit"].isin(val_units).values
    # Simpler and safer: rebuild windows separately by unit.
    train_frames, train_labels, train_units = [], [], []
    val_frames, val_labels, val_unit_ids = [], [], []
    for _, eng in train_raw.groupby("unit"):
        feat = eng[sensor_cols].copy()
        if seed_rolling:
            feat = add_rolling_std_seeded(
                feat, pd.DataFrame(columns=feat.columns), roll_win, sensor_cols
            )
        y = (eng["rul"] <= horizon).astype(int).values
        x, yw = make_windows(feat.values.astype(np.float32), y, window)
        u = np.full(len(x), int(eng["unit"].iloc[0]), dtype=np.int32)
        if eng["unit"].iloc[0] in val_units:
            val_frames.append(x)
            val_labels.append(yw)
            val_unit_ids.append(u)
        else:
            train_frames.append(x)
            train_labels.append(yw)
            train_units.append(u)
    x_tr = np.concatenate(train_frames)
    y_tr = np.concatenate(train_labels)
    u_tr = np.concatenate(train_units)
    x_va = np.concatenate(val_frames)
    y_va = np.concatenate(val_labels)
    u_va = np.concatenate(val_unit_ids)

    # Standardize on the training split; apply same transform to val/test.
    flat_tr = x_tr.reshape(-1, x_tr.shape[2])
    flat_va = x_va.reshape(-1, x_va.shape[2])
    flat_te = x_te_raw.reshape(-1, x_te_raw.shape[2])
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(flat_tr)
    x_tr = scaler.transform(flat_tr).reshape(x_tr.shape).astype(np.float32)
    x_va = scaler.transform(flat_va).reshape(x_va.shape).astype(np.float32)
    x_te = scaler.transform(flat_te).reshape(x_te_raw.shape).astype(np.float32)

    feature_names = sensor_cols + [f"{c}_rstd" for c in sensor_cols]
    return {
        "name": f"C-MAPSS {fd}",
        "x_train": x_tr,
        "y_train": y_tr,
        "x_val": x_va,
        "y_val": y_va,
        "x_test": x_te,
        "y_test": y_te,
        "unit_ids": {"train": u_tr, "val": u_va, "test": u_te},
        "feature_names": feature_names,
        "scaler": scaler,
        "horizon": horizon,
    }


XJTU_SPLIT = {
    "train": [1, 2, 3, 6, 7, 8, 11],
    "val": [4, 9, 12, 14],
    "test": [5, 10, 13, 15],
}


def _block_features(signal: np.ndarray) -> np.ndarray:
    """Time-domain health features of one 1.28 s acceleration block."""
    abs_sig = np.abs(signal)
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(abs_sig.max())
    kurt = float(stats.kurtosis(signal))
    skew = float(stats.skew(signal))
    crest = float(peak / (rms + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.sign(signal))) / 2))
    return np.array([rms, peak, kurt, skew, crest, zcr])


def _load_xjtu_features(data_dir: str, cache_csv: str) -> pd.DataFrame:
    if os.path.exists(cache_csv):
        return pd.read_csv(cache_csv)
    rows = []
    for mat_path in sorted(glob.glob(os.path.join(data_dir, "bearing*.mat"))):
        num = int(os.path.basename(mat_path).split(".")[0].replace("bearing", ""))
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        rawnet = mat["rawnet"]  # (32768, 2, n_blocks)
        n_blocks = rawnet.shape[2]
        condition = 1 if num <= 5 else (2 if num <= 10 else 3)
        for b in range(n_blocks):
            feats = []
            for ch in range(2):
                feats.extend(_block_features(rawnet[:, ch, b]))
            rows.append([num, condition, b, n_blocks] + feats)
    cols = (
        ["bearing", "condition", "block", "n_blocks"]
        + [f"ch0_{n}" for n in ("rms", "peak", "kurt", "skew", "crest", "zcr")]
        + [f"ch1_{n}" for n in ("rms", "peak", "kurt", "skew", "crest", "zcr")]
    )
    df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def load_xjtu(
    data_dir: str,
    cache_csv: str,
    window: int = 20,
    horizon_frac: float = 0.20,
    roll_win: int = 5,
    seed_rolling: bool = True,
) -> Dict[str, object]:
    df = _load_xjtu_features(data_dir, cache_csv)
    df["horizon"] = np.ceil(df["n_blocks"] * horizon_frac).astype(int)
    df["label"] = (
        (df["n_blocks"] - df["block"]) <= df["horizon"]
    ).astype(int)
    feature_cols = [c for c in df.columns if c.startswith("ch")]

    splits = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}
    unit_ids = {"train": [], "val": [], "test": []}
    for split, bearings in XJTU_SPLIT.items():
        for bearing in bearings:
            part = df[df["bearing"] == bearing].sort_values("block")
            feat = part[feature_cols].copy()
            for col in feature_cols:
                feat[col + "_diff"] = feat[col].diff().fillna(0.0)
            if seed_rolling:
                feat = add_rolling_std_seeded(
                    feat, pd.DataFrame(columns=feat.columns), roll_win, feature_cols
                )
            x, yw = make_windows(feat.values.astype(np.float32), part["label"].values, window)
            splits[split].append(x)
            labels[split].append(yw)
            unit_ids[split].append(np.full(len(x), bearing, dtype=np.int32))

    x_tr = np.concatenate(splits["train"])
    x_va = np.concatenate(splits["val"])
    x_te = np.concatenate(splits["test"])
    y_tr = np.concatenate(labels["train"])
    y_va = np.concatenate(labels["val"])
    y_te = np.concatenate(labels["test"])
    u_tr = np.concatenate(unit_ids["train"])
    u_va = np.concatenate(unit_ids["val"])
    u_te = np.concatenate(unit_ids["test"])

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_tr.reshape(-1, x_tr.shape[2]))
    x_tr = scaler.transform(x_tr.reshape(-1, x_tr.shape[2])).reshape(x_tr.shape).astype(np.float32)
    x_va = scaler.transform(x_va.reshape(-1, x_va.shape[2])).reshape(x_va.shape).astype(np.float32)
    x_te = scaler.transform(x_te.reshape(-1, x_te.shape[2])).reshape(x_te.shape).astype(np.float32)

    feature_names = feature_cols + [f"{c}_diff" for c in feature_cols]
    if seed_rolling:
        feature_names += [f"{c}_rstd" for c in feature_cols]
    return {
        "name": "XJTU-SY bearings",
        "x_train": x_tr,
        "y_train": y_tr,
        "x_val": x_va,
        "y_val": y_va,
        "x_test": x_te,
        "y_test": y_te,
        "unit_ids": {"train": u_tr, "val": u_va, "test": u_te},
        "feature_names": feature_names,
        "scaler": scaler,
        "horizon_frac": horizon_frac,
    }
