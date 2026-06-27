# World Cup Prediction Experiments

Target to beat on the same 62-match 2026 walk-forward backtest:

| Model | Accuracy | Log-loss | Brier |
|---|---:|---:|---:|
| Current XGBoost + Dixon-Coles ensemble | 64.5% | 0.8858 | 0.1791 |

## Implemented Experiment Suite

All experiments use `experiment_common.py`, which rebuilds the same historical feature matrix, uses the same 2026 completed-match walk-forward loop as `backtest_2026_wc.py`, and records accuracy, log-loss, Brier, ECE, and MCE.

Run the full suite:

```bash
python3 run_experiments.py
```

Each experiment writes or updates:

- `experiments_results.csv`: comparison table, one row per experiment.
- `experiments_results.jsonl`: append-only run log.
- `experiments_<experiment>_matches.csv`: per-match probabilities and predictions.

## Experiments

| Approach | Script | What It Tests | Status |
|---|---|---|---|
| LightGBM + Bayesian optimization | `experiment_lgbm.py` | LightGBM multiclass model, Optuna TPE search over tree shape, learning rate, sampling, and regularization; Dixon-Coles blend retained. | Implemented; skips cleanly if `lightgbm` is unavailable, uses fixed LightGBM defaults if only `optuna` is unavailable. |
| Neural network | `experiment_nn.py` | 256/128/64 feedforward network with BatchNorm, ReLU, Dropout, draw-weighted cross-entropy; sklearn MLP fallback. | Implemented. |
| Calibration | `experiment_calibration.py` | Platt scaling, temperature scaling, per-class isotonic on ensemble output, and beta-style calibration; reports ECE/MCE. | Implemented. |
| Better ensemble | `experiment_ensemble.py` | Manual stacking over XGBoost, Random Forest, Logistic Regression, and Dixon-Coles; dynamic draw-context blending. | Implemented. |
| Draw-specific model | `experiment_draw_model.py` | Binary draw classifier plus non-draw home/away classifier; threshold override variant. | Implemented. |
| Feature additions | `experiment_features.py` | Elo/form interactions, Elo polynomial terms, squad-value polynomial, Elo bins, and draw/low-goal interaction. | Implemented. |
| Combined candidate | `experiment_combined.py` | Engineered features + LightGBM/XGBoost fallback + Dixon-Coles + isotonic-on-ensemble calibration. | Implemented; not merged into production until metrics are known. |

## Current Results

I could not execute Python commands in this session because command execution was rejected by the environment, so no new backtest metrics are available yet. The implementation is set up so the first successful run of `python3 run_experiments.py` will populate the table below through `experiments_results.csv`.

| Experiment | Accuracy | Log-loss | Brier | ECE | MCE | Beats All? | Notes |
|---|---:|---:|---:|---:|---:|---|---|
| Baseline target | 64.5% | 0.8858 | 0.1791 | n/a | n/a | n/a | Existing model state from prompt. |
| LightGBM + Optuna + Dixon-Coles | pending | pending | pending | pending | pending | pending | Requires execution. |
| Neural Network + Dixon-Coles | pending | pending | pending | pending | pending | pending | Requires execution. |
| Calibration variants | pending | pending | pending | pending | pending | pending | Requires execution. |
| Stacking / dynamic blend | pending | pending | pending | pending | pending | pending | Requires execution. |
| Draw-specific variants | pending | pending | pending | pending | pending | pending | Requires execution. |
| Engineered features | pending | pending | pending | pending | pending | pending | Requires execution. |
| Combined candidate | pending | pending | pending | pending | pending | pending | Requires execution. |

## Production Merge Decision

No experiment has been merged into `predict_2026.py`, `explain_match.py`, or `monte_carlo_2026.py` yet. That is intentional: the project constraint is to keep what works and discard what does not, and the comparison metrics must be known before replacing the current production pipeline.
