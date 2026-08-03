import pandas as pd
e = pd.read_csv('../results/results_ensemble.csv')
o = pd.read_csv('../results/results_overall.csv')
order = ['full','lstm','bilstm','gru','transformer','tcn','random_forest','gradient_boosting']
for ds in ['ur3','cmapss_fd001','cmapss_fd003','xjtu']:
    print('\n---', ds, '---')
    for m in order:
        r = e[(e.dataset==ds)&(e.model==m)].iloc[0]
        s = o[(o.dataset==ds)&(o.model==m)].iloc[0]
        print(f'{m:18s} & {r["accuracy"]:.3f} & {r["precision"]:.3f} & {r["recall"]:.3f} & {r["f2"]:.3f} & {r["auc_roc"]:.3f} & {r["auc_pr"]:.3f} & {s["f2_std"]:.3f} & {s["auc_roc_std"]:.3f} \\\\')
print('\n--- ablations ---')
for ds in ['ur3','cmapss_fd001','cmapss_fd003','xjtu']:
    print('\n', ds)
    for m in ['full','w/o_attention','softmax_attention','w/o_mc_dropout','w/o_class_weight','w/_rolling_stats']:
        r = e[(e.dataset==ds)&(e.model==m)].iloc[0]
        s = o[(o.dataset==ds)&(o.model==m)].iloc[0]
        print(f'{m:20s} & {r["f2"]:.3f} & {r["auc_roc"]:.3f} & {r["auc_pr"]:.3f} & {r["brier"]:.3f} & {r["ece"]:.3f} & {s["f2_std"]:.3f} \\\\')
