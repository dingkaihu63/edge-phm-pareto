import pathlib
p = pathlib.Path('deployment_analysis_torch.py')
t = p.read_text(encoding='utf-8')
t = t.replace('dict(units1=96, units2=48, dropout=0.1)', 'dict(lstm_units_1=96, lstm_units_2=48, dropout_rate=0.1)')
t = t.replace('dict(units1=64, units2=32, dropout=0.15)', 'dict(lstm_units_1=64, lstm_units_2=32, dropout_rate=0.15)')
p.write_text(t, encoding='utf-8')
print('fixed')
