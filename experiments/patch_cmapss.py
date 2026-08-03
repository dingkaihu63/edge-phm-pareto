import pathlib
p = pathlib.Path('prepare_data.py')
t = p.read_text(encoding='utf-8')

old = '''    def prepare(df: pd.DataFrame, is_train: bool) -> Tuple[np.ndarray, np.ndarray]:
        frames = []
        labels = []
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
        return (
            np.concatenate(frames, axis=0) if frames else np.zeros((0, window, 1)),
            np.concatenate(labels) if labels else np.zeros((0,), dtype=np.int32),
        )'''
new = '''    def prepare(df: pd.DataFrame, is_train: bool) -> Tuple[np.ndarray, np.ndarray]:
        frames = []
        labels = []
        for _, eng in df.groupby("unit"):
            feat = eng[sensor_cols].copy()
            feat["cycle_progress"] = (
                eng["cycle"] - eng["cycle"].min()
            ) / max(eng["cycle"].max() - eng["cycle"].min(), 1)
            for col in sensor_cols:
                feat[col + "_diff"] = feat[col].diff().fillna(0.0)
            if seed_rolling:
                feat = add_rolling_std_seeded(
                    feat, pd.DataFrame(columns=feat.columns), roll_win, sensor_cols
                )
            if per_engine_std:
                feat = (feat - feat.mean()) / (feat.std() + 1e-6)
            y = (eng["rul"] <= horizon).astype(int).values
            x, yw = make_windows(feat.values.astype(np.float32), y, window)
            frames.append(x)
            labels.append(yw)
        return (
            np.concatenate(frames, axis=0) if frames else np.zeros((0, window, 1)),
            np.concatenate(labels) if labels else np.zeros((0,), dtype=np.int32),
        )'''
assert old in t, 'prepare block'
t = t.replace(old, new)

old = '''    for _, eng in train_raw.groupby("unit"):
        feat = eng[sensor_cols].copy()
        if seed_rolling:
            feat = add_rolling_std_seeded(
                feat, pd.DataFrame(columns=feat.columns), roll_win, sensor_cols
            )
        y = (eng["rul"] <= horizon).astype(int).values
        x, yw = make_windows(feat.values.astype(np.float32), y, window)
        if eng["unit"].iloc[0] in val_units:'''
new = '''    for _, eng in train_raw.groupby("unit"):
        feat = eng[sensor_cols].copy()
        feat["cycle_progress"] = (
            eng["cycle"] - eng["cycle"].min()
        ) / max(eng["cycle"].max() - eng["cycle"].min(), 1)
        for col in sensor_cols:
            feat[col + "_diff"] = feat[col].diff().fillna(0.0)
        if seed_rolling:
            feat = add_rolling_std_seeded(
                feat, pd.DataFrame(columns=feat.columns), roll_win, sensor_cols
            )
        if per_engine_std:
            feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        y = (eng["rul"] <= horizon).astype(int).values
        x, yw = make_windows(feat.values.astype(np.float32), y, window)
        if eng["unit"].iloc[0] in val_units:'''
assert old in t, 'train loop block'
t = t.replace(old, new)

old = '''    roll_win: int = 5,
    seed_rolling: bool = True,
    val_engines: int = 25,
) -> Dict[str, object]:'''
new = '''    roll_win: int = 5,
    seed_rolling: bool = True,
    per_engine_std: bool = True,
    val_engines: int = 25,
) -> Dict[str, object]:'''
assert old in t, 'signature'
t = t.replace(old, new)

p.write_text(t, encoding='utf-8')
print('patched cmapss')
