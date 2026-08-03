import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss
from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_xjtu

data = load_xjtu(r"E:\datasets\XJTU-SY\original", r"E:\datasets\XJTU-SY\xjtu_features_full15.csv", seed_rolling=False)
base = os.path.join('..', 'results', 'predictions')
models = ['full','w/o_attention','softmax_attention','w/o_mc_dropout','w/o_class_weight','w/_rolling_stats','lstm','bilstm','gru','transformer','tcn','random_forest','gradient_boosting']
for m in models:
    sf = m.replace('/','_').replace(' ','_')
    z = np.load(os.path.join(base, f'xjtu_{sf}.npz'))
    pv = z['pv']; p = z['p']
    logit = lambda q: np.log(np.clip(q,1e-7,1-1e-7)/(1-np.clip(q,1e-7,1-1e-7)))
    lv, lt = logit(pv), logit(p)
    best = minimize_scalar(lambda T: brier_score_loss(data['y_val'], 1/(1+np.exp(-lv/T))), bounds=(0.05,10), method='bounded')
    pvc = 1/(1+np.exp(-lv/best.x)); ptc = 1/(1+np.exp(-lt/best.x))
    th,_ = calibrate_threshold(data['y_val'], pvc)
    r = evaluate_binary(data['y_test'], ptc, th)
    print(m, 'T', round(best.x,2), 't', round(th,2), 'acc/prec/rec/f2/auc/pr', round(r['accuracy'],3), round(r['precision'],3), round(r['recall'],3), round(r['f2'],3), round(r['auc_roc'],3), round(r['auc_pr'],3))
