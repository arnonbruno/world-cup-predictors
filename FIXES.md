# Fixes Applied

## shared.py
- Added shared country-name harmonization, explicit boolean parsing, Elo helpers, country feature loading, XGBoost validation/early stopping, FIFA-style group standings, 2026 group-state construction from `data/results.csv`, third-place ranking with goals for, and R32 bracket construction.

## collect_data.py
- Canonicalized historical winners, runners-up, semifinalists, hosts, and participant lists before computing labels and history features.
- Fixed same-year World Bank leakage by using only `year - 1` or earlier data.
- Fixed ISO3 mappings for Honduras and the UK home nations; added canonical aliases for USA, Korea, Iran, Ivory Coast, Cape Verde, Curacao, Uzbekistan, and Jordan.
- Updated historical feature computation to operate on canonical country names.

## enrich_dataset.py
- Harmonized country names before applying FIFA rankings and Elo ratings.
- Replaced leaked `is_former_champion` construction with `wc_titles_before > 0`.

## predict_2026.py
- Replaced hard-coded group standings with standings built from completed 2026 rows in `data/results.csv`.
- Simulates every unplayed group match from the CSV, including Groups H and I.
- Allows group-stage draws in deterministic prediction.
- Uses shared FIFA standings sort order: points, goal difference, goals for.
- Uses shared R32 bracket construction and dynamic third-place insertion.
- Fixed feature names to match the dataset (`urbanization_pct`, `health_spending_pct_gdp`, `fifa_rank`).
- Uses the same country-feature set during training and inference via latest prior World Cup features.
- Added XGBoost validation split with early stopping and holdout accuracy/log-loss reporting.
- Excludes metadata pseudo-teams by using shared 2026 group definitions.

## monte_carlo_2026.py
- Removed stale hard-coded group tables, remaining-match lists, and third-place placeholder constants.
- Builds base standings from `data/results.csv` and samples only unplayed fixtures.
- Fixed third-place bracket insertion by using shared R32 slot assignment in one key space.
- Includes goals for in third-place ranking.
- Fixed feature names and train/inference country-feature mismatch.
- Added XGBoost validation split with early stopping and holdout accuracy/log-loss reporting.

## analyze_explain.py
- Fixed feature names and train/inference country-feature mismatch.
- Added XGBoost validation split with early stopping and holdout accuracy/log-loss reporting.
- Builds/simulates 2026 standings from the CSV and explains the computed Brazil R32 pairing instead of assuming Brazil vs Japan.
- Fixed multiclass SHAP handling for both list and array outputs.

## sota_analysis.py
- Moved regularized-logit feature selection inside each LOWCO training fold.
- Recomputes univariate feature ranking and VIF pruning from the training fold only before fitting that fold.

## backtest.py
- Converted the winner backtest to chronological training only (`wc_year < test_year`).
- Skips 1930 because no prior World Cup data exists.
- Replaced hard-coded absolute data path with a repository-relative path.
- Canonicalized Germany winner entries.

## match_predictor.py
- Replaced fragile `bool(row.neutral)` parsing with explicit true-value parsing everywhere neutral flags are consumed.

## Validation
- Static searches confirm the reviewed stale patterns are gone from Python sources.
- The required parse command could not be executed because command execution was rejected in this session.
