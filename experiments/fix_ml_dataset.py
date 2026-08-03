import pathlib
import pandas as pd
import numpy as np

# Patch run_ml to include dataset for future runs.
p = pathlib.Path('run_experiments_torch.py')
t = p.read_text(encoding='utf-8')
old = '''        rows.append(
            {"model": name, "family": "machine learning", "seed": seed, **metrics}
        )
    return rows'''
new = '''        row = {"model": name, "family": "machine learning", "seed": seed, **metrics}
        row["dataset"] = ds
        rows.append(row)
    return rows'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding='utf-8')

# Fix existing seed CSV.
path = '../results/results_seeds.csv'
df = pd.read_csv(path)
current = None
for i in range(len(df)):
    ds = df.at[i, 'dataset']
    if pd.notna(ds):
        current = ds
    else:
        df.at[i, 'dataset'] = current
df.to_csv(path, index=False)

# Regenerate aggregate CSV.
from run_experiments_torch import aggregate_seeds, METRIC_COLS
agg = aggregate_seeds(df)
agg.to_csv('../results/results_overall.csv', index=False)
print('fixed; rows', len(df), 'aggregated', len(agg))
