import numpy as np
from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_xjtu
from torch_common import build_model, mc_predict, set_seed, train_model

def run(name, window=20, units=(96,48), dropout=0.1, lr=1e-3, batch=64, cw=1.0, bal=False, seeds=3):
    data = load_xjtu(r"E:\datasets\XJTU-SY\original", r"E:\datasets\XJTU-SY\xjtu_features_full15.csv", seed_rolling=False, window=window)
    vals, tests = [], []
    for seed in range(1, seeds+1):
        set_seed(seed)
        m = build_model(data['x_train'].shape[1], data['x_train'].shape[2], lstm_units_1=units[0], lstm_units_2=units[1], dropout_rate=dropout, seed=seed)
        train_model(m, data['x_train'], data['y_train'], data['x_val'], data['y_val'], lr=lr, batch_size=batch, seed=seed, pos_weight_scale=cw, balanced_sampling=bal)
        pv,_ = mc_predict(m, data['x_val'], samples=50)
        pt,_ = mc_predict(m, data['x_test'], samples=50)
        vals.append(pv); tests.append(pt)
    pv = np.mean(vals,0); pt = np.mean(tests,0)
    th,_ = calibrate_threshold(data['y_val'], pv)
    r = evaluate_binary(data['y_test'], pt, th)
    print(name, 't', round(th,2), 'acc/prec/rec/f2/auc/pr', round(r['accuracy'],3), round(r['precision'],3), round(r['recall'],3), round(r['f2'],3), round(r['auc_roc'],3), round(r['auc_pr'],3))

run('base')
run('u128', units=(128,64))
run('w30', window=30)
run('bal', bal=True)
run('cw15', cw=1.5)
run('lr5e4', lr=5e-4, batch=128)
run('u128_w30', units=(128,64), window=30)
