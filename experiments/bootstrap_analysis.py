"""Paired bootstrap analysis for AUROC/AUPRC differences between models."""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = os.path.join('..', 'results', 'predictions')
OUT = os.path.join('..', 'results', 'bootstrap_diff.csv')

def safe(name):
    return name.replace('/', '_').replace(' ', '_')

datasets = ['ur3', 'cmapss_fd001', 'cmapss_fd003', 'xjtu']
models = ['lstm', 'bilstm', 'gru', 'transformer', 'tcn', 'patchtst', 'timesnet', 'random_forest', 'gradient_boosting']

rows = []
rng = np.random.RandomState(42)
for ds in datasets:
    zp = np.load(os.path.join(BASE, f'{ds}_full.npz'))
    p_prop, y = zp['p'], zp['y']
    n = len(y)
    best = None
    best_auc = -1
    best_name = None
    for m in models:
        z = np.load(os.path.join(BASE, f'{ds}_{safe(m)}.npz'))
        a = roc_auc_score(y, z['p'])
        if a > best_auc:
            best_auc = a
            best_name = m
            best = z['p']
    diff_auc = []
    diff_pr = []
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        a1 = roc_auc_score(y[idx], p_prop[idx])
        a2 = roc_auc_score(y[idx], best[idx])
        p1 = average_precision_score(y[idx], p_prop[idx])
        p2 = average_precision_score(y[idx], best[idx])
        diff_auc.append(a1 - a2)
        diff_pr.append(p1 - p2)
    diff_auc = np.array(diff_auc)
    diff_pr = np.array(diff_pr)
    rows.append({
        'dataset': ds,
        'best_baseline': best_name,
        'proposed_auc': roc_auc_score(y, p_prop),
        'best_auc': best_auc,
        'auc_diff_mean': diff_auc.mean(),
        'auc_diff_ci_low': np.percentile(diff_auc, 2.5),
        'auc_diff_ci_high': np.percentile(diff_auc, 97.5),
        'auc_ci_includes_zero': (np.percentile(diff_auc, 2.5) <= 0 <= np.percentile(diff_auc, 97.5)),
        'pr_diff_mean': diff_pr.mean(),
        'pr_diff_ci_low': np.percentile(diff_pr, 2.5),
        'pr_diff_ci_high': np.percentile(diff_pr, 97.5),
        'pr_ci_includes_zero': (np.percentile(diff_pr, 2.5) <= 0 <= np.percentile(diff_pr, 97.5)),
    })
    print(rows[-1])
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
