import pathlib, re
p = pathlib.Path('prepare_data.py')
lines = p.read_text(encoding='utf-8').splitlines(keepends=True)
out = []
i = 0
while i < len(lines):
    if 'feature_names = sensor_cols + ["cycle_progress"] + [' in lines[i]:
        block = ''.join(lines[i:i+4])
        if 'C-MAPSS' in ''.join(lines[i:i+12]):
            out.append('    feature_names = sensor_cols + [f"{c}_rstd" for c in sensor_cols]\n')
            if 'if seed_rolling:' in ''.join(lines[i+3:i+5]):
                # skip until 'if not seed_rolling' style block; here new block is:
                # feature_names = ... [0]
                #     f"{c}_diff"... [1]
                #     ] [2]
                # if seed_rolling: [3]
                #     feature_names += ... [4]
                i += 5
                continue
        out.append(block)
        i += 4
        continue
    out.append(lines[i])
    i += 1
t = ''.join(out)

# Revert prepare() block additions.
old = '''            feat = eng[sensor_cols].copy()
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
            y = (eng["rul"] <= horizon).astype(int).values'''
new = '''            feat = eng[sensor_cols].copy()
            if seed_rolling:
                feat = add_rolling_std_seeded(
                    feat, pd.DataFrame(columns=feat.columns), roll_win, sensor_cols
                )
            y = (eng["rul"] <= horizon).astype(int).values'''
assert old in t, 'prepare revert'
t = t.replace(old, new)

old = '''        feat = eng[sensor_cols].copy()
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
        y = (eng["rul"] <= horizon).astype(int).values'''
new = '''        feat = eng[sensor_cols].copy()
        if seed_rolling:
            feat = add_rolling_std_seeded(
                feat, pd.DataFrame(columns=feat.columns), roll_win, sensor_cols
            )
        y = (eng["rul"] <= horizon).astype(int).values'''
assert old in t, 'train loop revert'
t = t.replace(old, new)

p.write_text(t, encoding='utf-8')
print('reverted cmapss features')
