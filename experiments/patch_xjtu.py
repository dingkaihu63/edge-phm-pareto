import pathlib
p = pathlib.Path('prepare_data.py')
t = p.read_text(encoding='utf-8')

old = '''            part = df[df["bearing"] == bearing].sort_values("block")
            feat = part[feature_cols].copy()
            if seed_rolling:
                feat = add_rolling_std_seeded(
                    feat, pd.DataFrame(columns=feat.columns), roll_win, feature_cols
                )'''
new = '''            part = df[df["bearing"] == bearing].sort_values("block")
            feat = part[feature_cols].copy()
            feat["block_progress"] = np.linspace(0.0, 1.0, len(feat))
            for col in feature_cols:
                feat[col + "_diff"] = feat[col].diff().fillna(0.0)
            if seed_rolling:
                feat = add_rolling_std_seeded(
                    feat, pd.DataFrame(columns=feat.columns), roll_win, feature_cols
                )'''
assert old in t, 'xjtu feature block'
t = t.replace(old, new)

old = '''    feature_names = feature_cols + [f"{c}_rstd" for c in feature_cols]
    if not seed_rolling:
        feature_names = feature_cols'''
new = '''    feature_names = feature_cols + ["block_progress"] + [
        f"{c}_diff" for c in feature_cols
    ]
    if seed_rolling:
        feature_names += [f"{c}_rstd" for c in feature_cols]'''
assert old in t, 'xjtu feature names'
t = t.replace(old, new)
p.write_text(t, encoding='utf-8')
print('patched xjtu')
