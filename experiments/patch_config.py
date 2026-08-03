import pathlib
p = pathlib.Path('run_experiments_torch.py')
t = p.read_text(encoding='utf-8')

old = '''DS_CONFIG = {
    "ur3": dict(units1=96, units2=48, dropout=0.10, lr=1e-3, batch=64),
    "cmapss_fd001": dict(units1=64, units2=32, dropout=0.15, lr=5e-4, batch=128),
    "cmapss_fd003": dict(units1=64, units2=32, dropout=0.15, lr=5e-4, batch=128),
    "xjtu": dict(units1=96, units2=48, dropout=0.10, lr=1e-3, batch=64),
}'''
new = '''DS_CONFIG = {
    "ur3": dict(units1=96, units2=48, dropout=0.10, lr=1e-3, batch=64, balanced=False),
    "cmapss_fd001": dict(units1=64, units2=32, dropout=0.15, lr=5e-4, batch=128, balanced=False),
    "cmapss_fd003": dict(units1=64, units2=32, dropout=0.15, lr=5e-4, batch=128, balanced=False),
    "xjtu": dict(units1=96, units2=48, dropout=0.10, lr=1e-3, batch=64, balanced=True),
}'''
assert old in t
t = t.replace(old, new)

old = '''        use_class_weight=use_cw,
        lr=cfg["lr"],
        batch_size=cfg["batch"],
        seed=seed,
    )'''
new = '''        use_class_weight=use_cw,
        lr=cfg["lr"],
        batch_size=cfg["batch"],
        seed=seed,
        balanced_sampling=cfg.get("balanced", False),
    )'''
assert old in t
t = t.replace(old, new)

old = '''        use_class_weight=True,
        lr=cfg["lr"],
        batch_size=cfg["batch"],
        seed=seed,
    )'''
new = '''        use_class_weight=True,
        lr=cfg["lr"],
        batch_size=cfg["batch"],
        seed=seed,
        balanced_sampling=cfg.get("balanced", False),
    )'''
assert old in t
t = t.replace(old, new)

p.write_text(t, encoding='utf-8')
print('patched config')
