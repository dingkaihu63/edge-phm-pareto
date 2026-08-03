# edge-phm-pareto

Reproducible experiments for a configurable lightweight recurrent framework for industrial fault-state assessment.

The repository contains the PyTorch training, evaluation, uncertainty, temporal-attribution, physical-unit bootstrap, episode-alert, and deployment-reference pipelines used in the associated manuscript. C-MAPSS FD001/FD003 are the primary gradual-degradation tasks, XJTU-SY is a cross-bearing stress test, and UR3 CobotOps is an auxiliary event-window task.

## Scope

- Ten independently trained seeds for the main model and baseline comparisons
- Five-seed hierarchical paired bootstrap over training seeds and physical test units
- Five-seed Lite-budget 2 x 2 analysis of attention and dropout-enabled MC inference
- Episode-level alert outcomes that distinguish on-time, premature, missed, and false alerts
- Attention-deletion and uncertainty risk-coverage analyses
- ONNX, QEMU ARM64, parameter, storage, and MAC-count deployment references

The repository does not claim global accuracy-resource Pareto optimality, causal explanation, physical controller deployment, or a validated maintenance-cost advantage.

## Main Ten-Seed Results

| Benchmark | Full F2 | Full AUROC | Proposed-Lite F2 |
|---|---:|---:|---:|
| C-MAPSS FD001 | 0.853 +/- 0.011 | 0.994 +/- 0.001 | 0.819 +/- 0.026 |
| C-MAPSS FD003 | 0.889 +/- 0.011 | 0.994 +/- 0.004 | 0.870 +/- 0.015 |
| XJTU-SY | 0.544 +/- 0.016 | 0.713 +/- 0.055 | 0.539 +/- 0.012 |
| UR3 CobotOps | 0.719 +/- 0.031 | 0.834 +/- 0.022 | 0.705 +/- 0.023 |

These values are seed means and sample standard deviations. Each seed uses its own validation-selected threshold; test probabilities are not averaged across seeds.

## Environment

```powershell
conda env create -f environment.yml
conda activate edge-phm-pareto
```

The reported run used Python 3.10.20, PyTorch 2.14.0.dev20260711+cu130, scikit-learn 1.7.2, and matplotlib 3.10.9. The environment file specifies compatible package ranges because the exact PyTorch development build is not a stable public release.

## Data

- UR3 CobotOps: `dataset_02052023.xlsx` from UCI
- C-MAPSS: official NASA PCoE FD001 and FD003 text files
- XJTU-SY: official `bearing*.mat` files, optionally with a precomputed feature cache

Set the data locations before running experiments:

```powershell
$env:EDGE_PHM_UR3_DIR='C:\path\to\ur3'
$env:EDGE_PHM_CMAPSS_DIR='C:\path\to\C-MAPSS'
$env:EDGE_PHM_XJTU_DIR='C:\path\to\XJTU-SY\original'
$env:EDGE_PHM_XJTU_CACHE='C:\path\to\xjtu_features_full15.csv'
```

## Reproduce

Run commands from the repository root. The complete 10-seed training run is computationally expensive; final checkpoints and result tables are included for analysis-only reproduction.

```powershell
python experiments/run_experiments_torch.py --seeds 10 --datasets ur3,cmapss_fd001,cmapss_fd003,xjtu
python experiments/factorial_lite_analysis.py
python experiments/five_seed_bootstrap.py
python experiments/episode_alert_analysis.py
python experiments/attention_faithfulness.py
python experiments/risk_curves.py
python experiments/draw_architecture.py
python experiments/make_evidence_figures.py
python -m unittest discover -s tests -v
```

The analysis scripts require the final seed 1-5 checkpoints and fail explicitly when a required checkpoint is missing. They do not silently retrain models, except `factorial_lite_analysis.py`, which trains the previously missing no-attention/no-MC cells.

## Repository Layout

- `experiments/`: data loaders, models, training, analysis, and figure scripts
- `results/`: seed-level summaries, analysis tables, and model checkpoints
- `figures/`: generated PDF/SVG/PNG publication figures
- `tests/`: deterministic and MC-Dropout mode checks
- `docs/`: deployment-reference notes

The manuscript PDF is intentionally not distributed as a repository artifact.

## Citation

Please cite the associated manuscript once it is published. A final BibTeX entry will be added after publication.

## License

MIT License. See [LICENSE](LICENSE) for details.
