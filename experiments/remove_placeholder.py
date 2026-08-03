import pathlib
p = pathlib.Path('make_figures_torch.py')
t = p.read_text(encoding='utf-8')
old = '''def load_model_torch(ds: str, name: str):
    model = build_model(1, 1, seed=42)  # placeholder, replaced below
    del model
    return None


'''
assert old in t
t = t.replace(old, '')
p.write_text(t, encoding='utf-8')
print('removed placeholder')
