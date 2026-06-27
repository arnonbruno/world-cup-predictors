# World Cup Predictors — SOTA Econometric & Causation Analysis

## Goal
Create a comprehensive, leakage-free analysis script `sota_analysis.py` in this directory that identifies the true predictors of World Cup winners using state-of-the-art econometric and machine learning methods.

## Input Data
- `data/world_cup_predictors_dataset.csv` — 467 rows (81 countries × 22 WCs 1930–2022), 83 columns
- Target: `won_wc` (binary, 1 = winner that WC year, 0 = not winner)
- 21 winners out of 467 (4.5%, extreme class imbalance)

## CRITICAL: Data Leakage Audit

The current dataset has several columns that LEAK information about the outcome. The script must handle ALL of these:

### Columns to EXCLUDE entirely (direct outcome leaks):
- `runner_up`, `semifinalist`, `finalist`, `top4` — all derived from the same tournament outcome as `won_wc`
- `is_winner` — literally equal to `won_wc`
- `gdp_per_capita_vs_winner`, `population_vs_winner` — computed relative to the winner, so if team == winner, value is always 1.0, making it a perfect predictor

### Columns to KEEP but audit carefully:
- `is_host` — legitimate predictor (known before tournament starts)
- `elo_rating`, `fifa_rank`, `fifa_rank_inverse` — legitimate (pre-tournament rankings)
- `football_tradition`, `football_power_index`, `is_former_champion`, `is_strong_europe`, `is_strong_sa` — all computed from history BEFORE the current WC year, so no leakage (already verified in `collect_data.py`)
- `wc_titles_before`, `wc_finals_before`, `wc_semifinals_before`, `wc_participations_before`, `years_since_last_wc`, `years_since_last_win`, `years_since_last_final` — all `_before` columns, safe

### Columns requiring validation:
- `gdp_per_capita_vs_avg`, `population_vs_avg` — cross-sectional averages within that WC year. These are safe (average of ALL participants, not winner-specific)
- All World Bank indicators — collected with lag logic (year-1 fallback). Validate this doesn't accidentally use post-tournament data.

## Required Analyses

### Phase 1: Data Audit & Leakage Report
Print a complete leakage audit showing which columns are safe, which leak, and which need validation. Before doing ANY analysis.

### Phase 2: Preprocessing
- Drop leaking columns
- Handle missing values: impute with tournament-year cross-sectional median (not global median)
- Create proper train/test splits: **Leave-One-World-Cout-Out (LOWCO)** — train on N-1 WCs, test on 1. This is the ONLY valid CV strategy for this data since countries appear in multiple WCs and would leak via random splits.
- StandardScaler fit on train only, transform test

### Phase 3: Univariate Analysis (with proper corrections)
- Point-biserial correlation for each predictor with `won_wc`
- Mann-Whitney U test (non-parametric alternative)
- Cohen's d effect size
- Bootstrap 95% CI on effect sizes (1000 iterations)
- Bonferroni and FDR (Benjamini-Hochberg) corrections for multiple comparisons
- Report both corrected and uncorrected p-values

### Phase 4: Multivariate Logistic Regression
- Unpenalized logistic regression with top features
- Variance Inflation Factor (VIF) analysis — remove features with VIF > 5
- Regularized logistic regression (L1, L2, ElasticNet) with LOWCO CV
- Report: coefficients, odds ratios, confidence intervals via profile likelihood
- Hosmer-Lemeshow goodness-of-fit test

### Phase 5: Tree-Based Models (with LOWCO CV)
- Random Forest with LOWCO cross-validation
- XGBoost with LOWCO cross-validation  
- SHAP values for both (if shap available, otherwise permutation importance)
- Partial Dependence Plots for top 5 features

### Phase 6: Causal Analysis
- Difference-in-Differences: Does being host CAUSE winning? Compare winners who hosted vs winners who didn't
- Granger-like analysis: Do lagged economic indicators (t-1, t-4, t-8) predict winning better than contemporaneous?
- Propensity Score Matching: Match countries on GDP/population, compare football outcomes
- Instrumental Variable attempt: Is there a valid instrument for Elo rating?
- Directed Acyclic Graph (DAG) — draw the hypothesized causal structure (mermaid/text diagram in output)

### Phase 7: Synthesis Ranking
- Combine all methods into a unified importance ranking
- Average rank across: univariate (FDR-adjusted), logistic (|coef|), RF importance, XGBoost importance, SHAP
- Rank by number of methods in which each variable appears in top-10
- Category breakdown (economy, football, demographics, health, etc.)

### Phase 8: Predictive Power Assessment
- Using ONLY pre-tournament information, build the best predictive model
- Report LOWCO CV AUC, precision, recall, F1 for each fold
- Confusion matrix aggregated across all folds
- Calibration plot (predicted vs observed win rate by decile)
- Brier score

## Output Requirements
- Save all results to `output/sota/` directory
- Generate `output/sota/summary.md` with key findings in plain English
- Save figures as PNG to `output/sota/figures/`
- Print a clear executive summary to stdout at the end

## Technical Requirements
- Use ONLY `pandas`, `numpy`, `scipy`, `sklearn`, `xgboost`, `statsmodels`. Install `shap` if possible, otherwise use permutation importance.
- No data leakage: fit scalers/imputers on train, transform test
- All random states = 42 for reproducibility
- Handle the extreme class imbalance (4.5% positive rate) with `class_weight='balanced'` or `scale_pos_weight`

## File Structure
Write a single `sota_analysis.py` at `/var/mnt/DATA/Hermes/workspace/world-cup-predictors/sota_analysis.py` that can be run with `python3 sota_analysis.py` and produces all outputs.