"""Validation-based variant selection using saved 5-seed ensemble predictions."""

import os
import numpy as np
import pandas as pd

from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_ur3, load_cmapss, load_xjtu

BASE = os.path.join('..', 'results', 'predictions')

def safe(m):
    return m.replace('/', '_').replace(' ', '_')

loaders = {
    'ur3': lambda: load_ur3(r"C:\Users\hu\Desktop\比赛", seed_rolling=False),
    'cmapss_fd001': lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD001", seed_rolling=False),
    'cmapss_fd003': lambda: load_cmapss(r"E:\datasets\C-MAPSS", "FD003", seed_rolling=False),
    'xjtu': lambda: load_xjtu(r"E:\datasets\XJTU-SY\original", r"E:\datasets\XJTU-SY\xjtu_features_full15.csv", seed_rolling=False),
}

variants = ['full', 'w/o_attention', 'softmax_attention', 'w/o_mc_dropout', 'w/o_class_weight']
rows = []
for ds, loader in loaders.items():
    data = loader()
    best_v, best_f2, best_th = None, -1, None
    for v in variants:
        z = np.load(os.path.join(BASE, f'{ds}_{safe(v)}.npz'))
        th, _ = calibrate_threshold(data['y_val'], z['pv'])
        vm = evaluate_binary(data['y_val'], z['pv'], th)
        if vm['f2'] > best_f2:
            best_f2, best_v, best_th = vm['f2'], v, th
    z = np.load(os.path.join(BASE, f'{ds}_{safe(best_v)}.npz'))
    tm = evaluate_binary(data['y_test'], z['p'], best_th)
    zl = np.load(os.path.join(BASE, f'{ds}_lstm.npz'))
    thl, _ = calibrate_threshold(data['y_val'], zl['pv'])
    lm = evaluate_binary(data['y_test'], zl['p'], thl)
    rows.append({
        'dataset': ds, 'selected': best_v, 'sel_val_f2': round(best_f2, 3),
        'sel_test_f2': round(tm['f2'], 3), 'sel_auc': round(tm['auc_roc'], 3), 'sel_pr': round(tm['auc_pr'], 3),
        'lstm_test_f2': round(lm['f2'], 3), 'lstm_auc': round(lm['auc_roc'], 3), 'lstm_pr': round(lm['auc_pr'], 3),
    })
    print(rows[-1])
pd.DataFrame(rows).to_csv(os.path.join('..', 'results', 'validation_selection.csv'), index=False)
