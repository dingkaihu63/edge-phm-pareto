import pathlib
p = pathlib.Path('run_experiments_torch.py')
t = p.read_text(encoding='utf-8')
old = 'import argparse\nimport os\nimport time'
new = 'import argparse\nimport json\nimport os\nimport time\nimport torch'
assert old in t
t = t.replace(old, new)

old = 'def ensure_dirs() -> None:\n    for d in (OUT, MODELS_DIR, PRED_DIR):\n        os.makedirs(d, exist_ok=True)'
new = '''def ensure_dirs() -> None:
    for d in (OUT, MODELS_DIR, PRED_DIR):
        os.makedirs(d, exist_ok=True)


def save_config(seeds: int) -> None:
    cfg = {
        "seeds": seeds,
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ds_config": DS_CONFIG,
    }
    with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)'''
assert old in t
t = t.replace(old, new)

old = '    ensure_dirs()\n\n    all_rows = []'
new = '    ensure_dirs()\n    save_config(args.seeds)\n\n    all_rows = []'
assert old in t
t = t.replace(old, new)

p.write_text(t, encoding='utf-8')
print('config saving added')
