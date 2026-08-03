import pathlib
p = pathlib.Path('prepare_data.py')
t = p.read_text(encoding='utf-8')

# Remove UR3 cycle_progress feature creation.
old = '''            feat[:] = feat.ffill().fillna(medians)
            feat["cycle_progress"] = np.linspace(0.0, 1.0, len(feat))
            for col in sensor_cols:'''
new = '''            feat[:] = feat.ffill().fillna(medians)
            for col in sensor_cols:'''
assert old in t, 'ur3 create'
t = t.replace(old, new)

old = '''    feature_names = sensor_cols + ["cycle_progress"] + [
        f"{c}_diff" for c in sensor_cols
    ]
    if seed_rolling:
        feature_names += [f"{c}_rstd" for c in sensor_cols]'''
new = '''    feature_names = sensor_cols + [f"{c}_diff" for c in sensor_cols]
    if seed_rolling:
        feature_names += [f"{c}_rstd" for c in sensor_cols]'''
assert old in t, 'ur3 names'
t = t.replace(old, new)

# Remove XJTU block_progress feature creation.
old = '''            feat = part[feature_cols].copy()
            feat["block_progress"] = np.linspace(0.0, 1.0, len(feat))
            for col in feature_cols:'''
new = '''            feat = part[feature_cols].copy()
            for col in feature_cols:'''
assert old in t, 'xjtu create'
t = t.replace(old, new)

old = '''    feature_names = feature_cols + ["block_progress"] + [
        f"{c}_diff" for c in feature_cols
    ]
    if seed_rolling:
        feature_names += [f"{c}_rstd" for c in feature_cols]'''
new = '''    feature_names = feature_cols + [f"{c}_diff" for c in feature_cols]
    if seed_rolling:
        feature_names += [f"{c}_rstd" for c in feature_cols]'''
assert old in t, 'xjtu names'
t = t.replace(old, new)

p.write_text(t, encoding='utf-8')
print('removed progress features')
