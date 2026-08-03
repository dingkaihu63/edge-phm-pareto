import pathlib
p = pathlib.Path('tune_proposed.py')
t = p.read_text(encoding='utf-8')
old = '        threshold, _ = calibrate_threshold(data["y_val"], p_val)\n        m = evaluate_binary(data["y_test"], p_test, threshold)\n        m["seed"] = seed\n        rows.append(m)'
new = '        threshold, _ = calibrate_threshold(data["y_val"], p_val)\n        v = evaluate_binary(data["y_val"], p_val, threshold)\n        m = evaluate_binary(data["y_test"], p_test, threshold)\n        m["seed"] = seed\n        m["val_f2"] = v["f2"]\n        rows.append(m)'
assert old in t
t = t.replace(old, new)
t = t.replace('"val_f2_mean": df["f2"].mean(),', '"val_f2_mean": df["val_f2"].mean(),')
p.write_text(t, encoding='utf-8')
print('ok')
