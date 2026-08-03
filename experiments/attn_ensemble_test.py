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

def load(ds, m):
    z = np.load(os.path.join(BASE, f'{ds}_{safe(m)}.npz'))
    return z['pv'], z['p'], z['y']

rows = []
for ds, loader in loaders.items():
    data = loader()
    sets = {
        'full': load(ds, 'full'),
        'no_attn': load(ds, 'w/o_attention'),
        'softmax': load(ds, 'softmax_attention'),
        'no_mc': load(ds, 'w/o_mc_dropout'),
        'lstm': load(ds, 'lstm'),
    }
    combos = {
        'attn_mix': ['full', 'no_attn'],
        'attn_mix3': ['full', 'no_attn', 'softmax'],
        'attn_mix_mc': ['full', 'no_attn', 'no_mc'],
    }
    for cname, members in combos.items():
        pv = np.mean([sets[m][0] for m in members], axis=0)
        pt = np.mean([sets[m][1] for m in members], axis=0)
        th, _ = calibrate_threshold(data['y_val'], pv)
        r = evaluate_binary(data['y_test'], pt, th)
        lr = evaluate_binary(data['y_test'], sets['lstm'][1], calibrate_threshold(data['y_val'], sets['lstm'][0])[0])
        print(ds, cname, 'F2', round(r['f2'],3), 'AUC', round(r['auc_roc'],3), 'PR', round(r['auc_pr'],3), '| LSTM', round(lr['f2'],3), round(lr['auc_roc'],3), round(lr['auc_pr'],3))
