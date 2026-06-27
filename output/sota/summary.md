# World Cup Predictors: SOTA Analysis Summary

## What was done
- Ran a full leakage audit and dropped all direct leakage columns before analysis.
- Used strict Leave-One-World-Cup-Out validation for all predictive models.
- Fit train-only imputers/scalers in every fold (no test leakage).
- Estimated effect sizes with 95% bootstrap CIs and adjusted p-values with both Bonferroni and FDR.

## Leakage controls
- Direct leakage columns dropped: runner_up, semifinalist, finalist, top4, is_winner, gdp_per_capita_vs_winner, population_vs_winner.
- Post-tournament columns excluded from prediction: total_goals_in_tournament, avg_goals_per_match.
- Cross-sectional within-year ratio columns were retained after validation.

## Key univariate signals (FDR-aware)
- Top 10 predictors by univariate significance/effect: is_former_champion, elo_rating, football_tradition, is_strong_sa, is_host, football_power_index, years_since_last_final, years_since_last_win, wc_semifinals_before, wc_finals_before.

## Multivariate logistic results
- Top logistic features (FDR-ranked): elo_rating, trade_pct_gdp, football_tradition, wc_participations_before, fifa_rank, fifa_rank_inverse, is_host, gdp_total_rank, wc_semifinals_before, population_log.
- Hosmer-Lemeshow goodness-of-fit: chi2=1.064, p=0.9978.

## Regularized logistic (LOWCO tuning)
- Best L1/L2/ElasticNet settings are in `regularized_logit_lowco_scores.csv` (best mean AUC observed: 0.9656).

## Tree models (LOWCO)
- Random Forest mean LOWCO AUC: 0.9090.
- XGBoost mean LOWCO AUC: 0.9409.

## Causal analysis snapshots
- Difference-in-Differences hosting effect: 0.0537 (95% CI 0.0215 to 0.0847).
- Propensity score matching ATT (hosting -> winning): 0.1364 (95% CI -0.0455 to 0.3182, pairs=22).
- Granger-like lag comparison: t4 AUC=0.6846, t8 AUC=0.6821, t1_like AUC=0.6779
- IV attempt for Elo (instrument `wc_titles_before`): first-stage F=140.9659046244874, assessment=likely weak/invalid.

## Unified ranking (across methods)
- Features strongest across methods:
  - elo_rating (category=football, top10_methods=5, avg_rank=3.20)
  - is_former_champion (category=football, top10_methods=4, avg_rank=2.00)
  - football_power_index (category=football, top10_methods=4, avg_rank=2.75)
  - football_tradition (category=football, top10_methods=3, avg_rank=12.80)
  - wc_semifinals_before (category=football, top10_methods=3, avg_rank=17.60)
  - is_host (category=other, top10_methods=2, avg_rank=12.20)
  - population_rank (category=demographics, top10_methods=2, avg_rank=14.00)
  - wc_participations_before (category=football, top10_methods=2, avg_rank=16.60)
  - years_since_last_win (category=football, top10_methods=2, avg_rank=16.80)
  - years_since_last_final (category=football, top10_methods=2, avg_rank=19.80)

## Best predictive model (LOWCO)
- Selected model: `best_regularized_logit`.
- Mean AUC=0.9656, Precision=0.2690, Recall=0.8182, F1=0.3605.
- Mean fold Brier=0.1513, pooled OOF Brier=0.1636.

## Caveats
- This is observational data; causal effects should be interpreted as suggestive, not definitive.
- IV validity for Elo is weak and exclusion restriction is difficult to justify with available columns.
