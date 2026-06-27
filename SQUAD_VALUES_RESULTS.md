# Squad Market Values Integration

Integrates Transfermarkt squad market values (`data/squad_values.csv`,
169 rows / 86 teams / 2014–2026) into the World Cup prediction pipeline.

## What was added

### `shared.py`
- `SQUAD_VALUE_FEATURE_COLUMNS` — the four new columns, treated like the odds
  columns (kept as `NaN` when missing so XGBoost learns a "no valuation" split
  instead of imputing 0 EUR).
- `SQUAD_VALUE_YEARS = (2014, 2018, 2022, 2026)` — editions with coverage.
- `load_squad_values(path)` → `dict[(canonical_team, year)] ->
  {total_squad_value_eur, avg_player_value_eur}`. Team names are harmonized via
  `NAME_ALIASES` on load (CSV "Iran"/"South Korea"/"Ivory Coast" → state keys
  "IR Iran"/"Korea Republic"/"Côte d'Ivoire").
- `_squad_value_year(match_year)` / `squad_value_for_team(...)` — year-lag
  lookup: a match uses the most recent WC edition **at or before** its calendar
  year (2023 matches → 2022 squads, 2026 fixtures → 2026 values).
- `compute_match_features(..., squad_values=None)` now emits four log-scaled
  features:
  - `squad_value` = `log1p(home total value)`
  - `opp_squad_value` = `log1p(away total value)`
  - `squad_value_diff` = `log1p(home) − log1p(away)`
  - `squad_value_ratio` = `(home + 1) / (away + 1)` (raw ratio)
  All four are `NaN` when either team lacks a valuation for the lagged year.
- `finalize_feature_frame` / `prepare_prediction_frame` exclude the squad-value
  columns from the `fillna(0)` pass (same NaN-preserving convention as odds).

### Wiring (no behavioural change to existing features)
- `backtest_2026_wc.py` — loads squad values, threads them through
  `train_model(...)` and `predict_match(...)`.
- `predict_2026.py` — `train_model_bundle(...)` loads/uses them, stored on the
  `PredictionBundle` and a module-level `_SQUAD_VALUES` used at prediction time.
- `monte_carlo_2026.py` — module-level `_SQUAD_VALUES`, used in training and in
  every simulated match.
- `explain_match.py` — `ModelBundle.squad_values` populated from the training
  bundle and passed into `compute_match_features` for SHAP explanations.

All call sites pass `squad_values` so the trailing parameter never breaks the
existing positional contract.

## Backtest comparison

Run:

```bash
python3 backtest_2026_wc.py
```

| Metric    | Previous (no squad values) | New (with squad values) |
|-----------|---------------------------:|------------------------:|
| Accuracy  | 64.5%                      | _pending run_           |
| Log-loss  | 0.8858                     | _pending run_           |
| Brier     | 0.1791                     | _pending run_           |

> Note: the backtest/predict commands could not be executed in this session
> (shell execution was unavailable). Run the command above to populate the
> "New" column; `python3 predict_2026.py` verifies the 2026 bracket still
> produces a full prediction.
