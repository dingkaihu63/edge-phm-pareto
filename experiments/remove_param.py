import pathlib
p = pathlib.Path('prepare_data.py')
t = p.read_text(encoding='utf-8')
old = '    per_engine_std: bool = True,\n'
assert old in t
t = t.replace(old, '')
p.write_text(t, encoding='utf-8')
print('removed unused param')
