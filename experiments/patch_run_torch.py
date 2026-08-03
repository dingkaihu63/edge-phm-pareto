import pathlib
p = pathlib.Path('run_experiments_torch.py')
t = p.read_text(encoding='utf-8')

old = '''DEEP_BASELINES = ["lstm", "bilstm", "gru", "transformer", "tcn"]'''
new = '''DEEP_BASELINES = ["lstm", "bilstm", "gru", "transformer", "tcn"]

DS_CONFIG = {
    "ur3": dict(units1=96, units2=48, dropout=0.10, lr=1e-3, batch=64),
    "cmapss_fd001": dict(units1=64, units2=32, dropout=0.15, lr=5e-4, batch=128),
    "cmapss_fd003": dict(units1=64, units2=32, dropout=0.15, lr=5e-4, batch=128),
    "xjtu": dict(units1=96, units2=48, dropout=0.10, lr=1e-3, batch=64),
}'''
assert old in t
t = t.replace(old, new)

old = '''    set_seed(seed)
    model = build_model(
        data["x_train"].shape[1],
        data["x_train"].shape[2],
        seed=seed,
        **build_kwargs,
    )
    train_model(
        model,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        use_class_weight=use_cw,
        seed=seed,
    )'''
new = '''    set_seed(seed)
    cfg = DS_CONFIG[ds]
    model = build_model(
        data["x_train"].shape[1],
        data["x_train"].shape[2],
        lstm_units_1=cfg["units1"],
        lstm_units_2=cfg["units2"],
        dropout_rate=cfg["dropout"],
        seed=seed,
        **build_kwargs,
    )
    train_model(
        model,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        use_class_weight=use_cw,
        lr=cfg["lr"],
        batch_size=cfg["batch"],
        seed=seed,
    )'''
assert old in t
t = t.replace(old, new)

old = '''    set_seed(seed)
    model = build_deep_baseline(
        model_name, data["x_train"].shape[1], data["x_train"].shape[2], seed=seed
    )
    train_model(
        model,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        use_class_weight=True,
        seed=seed,
    )'''
new = '''    set_seed(seed)
    cfg = DS_CONFIG[ds]
    model = build_deep_baseline(
        model_name, data["x_train"].shape[1], data["x_train"].shape[2], seed=seed
    )
    train_model(
        model,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        use_class_weight=True,
        lr=cfg["lr"],
        batch_size=cfg["batch"],
        seed=seed,
    )'''
assert old in t
t = t.replace(old, new)

p.write_text(t, encoding='utf-8')
print('patched run_experiments_torch')
