# edge-phm-pareto

Lightweight, uncertainty-aware, and explainable fault prediction for edge industrial time series.

This repository contains the PyTorch pipeline and experimental results for a lightweight recurrent attention framework with MC-Dropout, cost-sensitive thresholds, post-hoc calibration, uncertainty-based selective prediction, and streaming closed-loop alerting. Evaluation benchmarks: UR3 CobotOps, NASA C-MAPSS FD001/FD003, and the complete 15-run XJTU-SY bearing dataset.

## Highlights

- 10-seed unified evaluation protocol
- Proposed-Lite variant with 16,374 average parameters
- Baselines: LSTM, BiLSTM, GRU, Transformer, TCN, PatchTST, TimesNet, random forest, HistGBM
- Complete component ablation on every benchmark
- Post-hoc calibration and MC-uncertainty selective prediction
- Streaming closed-loop alerting with lead-time and false-alarm analysis
- Edge-performance Pareto analysis across parameters, latency, F2, and uncertainty capability
- Vector publication figures

## Key Results

| Benchmark | Full F2 | Full AUROC | Proposed-Lite F2 |
|---|---:|---:|---:|
| UR3 CobotOps | 0.738 | 0.854 | 0.742 |
| C-MAPSS FD001 | 0.866 | 0.996 | 0.829 |
| C-MAPSS FD003 | 0.889 | 0.998 | 0.880 |
| XJTU-SY | 0.550 | 0.661 | 0.522 |

Calibration: isotonic regression reduces XJTU-SY ECE from 0.510 to 0.040. Selective prediction: rejecting 10% of the most uncertain windows raises XJTU-SY F2 from 0.550 to 0.601.

## Requirements

- Python 3.10
- PyTorch 2.x with CUDA
- numpy, pandas, scipy, scikit-learn, matplotlib
- shap (for explainability figures)

## Data

- UR3 CobotOps: `dataset_02052023.xlsx` (UCI)
- C-MAPSS: official NASA PCoE FD001/FD003 text files
- XJTU-SY: official `bearing*.mat` files or the precomputed `xjtu_features_full15.csv`

Update the paths in `prepare_data.py` and the experiment scripts to match your local data layout.

## Reproduce

```bash
python prepare_data.py
python run_experiments_torch.py --seeds 10 --datasets ur3,cmapss_fd001,cmapss_fd003,xjtu
python calibration_analysis.py
python closed_loop_eval.py
python deployment_analysis_torch.py
python make_figures_torch.py
```

## Repository Layout

- `experiments/`: dataset loaders, model definitions, experiment runners, analyses, figure scripts
- `results/`: seed-level results, ensemble predictions, calibration, closed-loop, deployment tables
- `figures/`: generated vector figures

## Citation

Please cite the associated manuscript once it is published. A BibTeX entry will be added here at that time.
## License

MIT License. See [LICENSE](LICENSE) for details.

## Hardware-Free Validation

See [docs/HARDWARE_FREE_VALIDATION.md](docs/HARDWARE_FREE_VALIDATION.md) for the evidence ladder and reproduction commands.
