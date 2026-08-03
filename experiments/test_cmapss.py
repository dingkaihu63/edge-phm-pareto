import numpy as np
from common_no_tf import calibrate_threshold, evaluate_binary
from prepare_data import load_cmapss
from torch_common import build_model, build_deep_baseline, mc_predict, set_seed, train_model

def run(kind, fd, units=(96,48), dropout=0.1, lr=1e-3, batch=64, seeds=3):
    data = load_cmapss(r"E:\datasets\C-MAPSS", fd)
    print(fd, 'features', data['x_train'].shape[2], 'windows', len(data['y_train']))
    vals, tests = [], []
    for seed in range(1, seeds+1):
        set_seed(seed)
        if kind == 'proposed':
            m = build_model(data['x_train'].shape[1], data['x_train'].shape[2], lstm_units_1=units[0], lstm_units_2=units[1], dropout_rate=dropout, seed=seed)
        else:
            m = build_deep_baseline(kind, data['x_train'].shape[1], data['x_train'].shape[2], seed=seed)
        train_model(m, data['x_train'], data['y_train'], data['x_val'], data['y_val'], lr=lr, batch_size=batch, seed=seed)
        pv,_ = mc_predict(m, data['x_val'], samples=50)
        pt,_ = mc_predict(m, data['x_test'], samples=50)
        vals.append(pv); tests.append(pt)
    pv = np.mean(vals,0); pt = np.mean(tests,0)
    th,_ = calibrate_threshold(data['y_val'], pv)
    r = evaluate_binary(data['y_test'], pt, th)
    print(kind, fd, 't', round(th,2), 'acc/prec/rec/f2/auc/pr', round(r['accuracy'],3), round(r['precision'],3), round(r['recall'],3), round(r['f2'],3), round(r['auc_roc'],3), round(r['auc_pr'],3))

run('proposed','FD001')
run('lstm','FD001')
run('proposed','FD003')
run('lstm','FD003')
