import pathlib
p = pathlib.Path('prepare_data.py')
t = p.read_text(encoding='utf-8')

old = '''            feat = part[sensor_cols].copy()
            if len(feat) < window:
                continue
            feat[:] = feat.ffill().fillna(medians)
            if seed_rolling:
                feat = add_rolling_std_seeded(feat, empty, roll_win, sensor_cols)'''
new = '''            feat = part[sensor_cols].copy()
            if len(feat) < window:
                continue
            feat[:] = feat.ffill().fillna(medians)
            feat["cycle_progress"] = np.linspace(0.0, 1.0, len(feat))
            for col in sensor_cols:
                feat[col + "_diff"] = feat[col].diff().fillna(0.0)
            if seed_rolling:
                feat = add_rolling_std_seeded(feat, empty, roll_win, sensor_cols)'''
assert old in t
t = t.replace(old, new)

old = '''    feature_names = sensor_cols + [f"{c}_rstd" for c in sensor_cols]
    if not seed_rolling:
        feature_names = sensor_cols'''
new = '''    feature_names = sensor_cols + ["cycle_progress"] + [
        f"{c}_diff" for c in sensor_cols
    ]
    if seed_rolling:
        feature_names += [f"{c}_rstd" for c in sensor_cols]'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding='utf-8')
print('patched prepare_data')
