import numpy as np
from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_ur3
from torch_common import build_model, mc_predict, set_seed, train_model

def run(name, window=10, mc=True, norm=True, temp=0.5, cw=1.0, units=(96,48), seeds=5):
    data = load_ur3(r"C:\Users\hu\Desktop\比赛", window=window)
    vals, tests = [], []
    for seed in range(1, seeds+1):
        set_seed(seed)
        m = build_model(data['x_train'].shape[1], data['x_train'].shape[2], mc_dropout=mc, attn_normalize=norm, attn_temperature=temp, lstm_units_1=units[0], lstm_units_2=units[1], dropout_rate=0.1, seed=seed)
        train_model(m, data['x_train'], data['y_train'], data['x_val'], data['y_val'], lr=1e-3, batch_size=64, seed=seed, pos_weight_scale=cw)
        pv,_ = mc_predict(m, data['x_val'], samples=50)
        pt,_ = mc_predict(m, data['x_test'], samples=50)
        vals.append(pv); tests.append(pt)
    pv = np.mean(vals,0); pt = np.mean(tests,0)
    th,_ = calibrate_threshold(data['y_val'], pv)
    r = evaluate_binary(data['y_test'], pt, th)
    print(name, 't', round(th,2), 'acc/prec/rec/f2/auc/pr', round(r['accuracy'],3), round(r['precision'],3), round(r['recall'],3), round(r['f2'],3), round(r['auc_roc'],3), round(r['auc_pr'],3))

run('det', mc=False)
run('orig_attn', norm=False, temp=1.0)
run('cw3', cw=3.0)
run('w12', window=12)
