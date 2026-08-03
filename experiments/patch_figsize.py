import pathlib
p = pathlib.Path('make_figures_torch.py')
t = p.read_text(encoding='utf-8')
t = t.replace('fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))', 'fig, axes = plt.subplots(1, 3, figsize=(12, 5.0))')
t = t.replace('fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))', 'fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))')
p.write_text(t, encoding='utf-8')
print('figure sizes updated')
