# Code and Model Review

## Executive Summary

**Overall verdict: fail for model validity and published headline claims.** Several scripts contain useful ideas, and the newer incremental and SOTA scripts make real attempts to avoid leakage. However, the project as a whole is not reliable enough to support the reported winner rates or 2026 predictions.

Key findings:

- There is confirmed leakage in the country-level dataset pipeline. `enrich_dataset.py` creates `is_former_champion` from all winners in the full dataset, including future winners, and `collect_data.py` uses same-year World Bank indicators despite claiming a lag.
- The 2026 prediction scripts do not correctly simulate all remaining group matches. `predict_2026.py`, `monte_carlo_2026.py`, and `analyze_explain.py` hard-code Groups H and I as complete even though `data/results.csv` still has unplayed H/I matches on 2026-06-26.
- `monte_carlo_2026.py` has a bracket propagation bug: it computes best third-place teams but fails to insert them because it mixes R32 array indices with FIFA match numbers. Placeholder teams like `PAR`, `SWE`, `ECU`, and `KOR` can remain in the simulated bracket.
- The 86.4% "exact winner" metric in `incremental_predictor.py` is not a legitimate pre-tournament champion prediction. It predicts the winner of the actual historical final after the real finalists and all prior tournament results are already known.
- Country name harmonization is inconsistent. The country-level dataset splits Germany and West Germany history, and the generated dataset appears to miss Germany's 1990 winner label.
- XGBoost models are mostly trained with fixed hyperparameters and limited validation/calibration. Some scripts train on all available data without a proper test split.
- SHAP and feature importances are only partly trustworthy. They describe the fitted models, but several fitted models are trained on leaked or full-sample feature-selected data, so the explanations inherit those problems.

## Answers to Key Questions

1. **Is there any data leakage?** Yes. The most serious leakage is `is_former_champion` in `enrich_dataset.py:197-198`, same-year macro data in `collect_data.py:523-532`, direct winner-relative variables in `collect_data.py:629-641`, and full-sample feature selection before LOWCO regularized logistic modeling in `sota_analysis.py:1392-1414`.

2. **Is the 86.4% exact winner rate legitimate?** No, not as a champion-prediction claim. It is a conditional actual-final pick. `incremental_predictor.py:859-865` sets `final_predicted_winner` only when iterating over the actual final match, and `incremental_predictor.py:892-894` updates the state with actual prior tournament results before that final.

3. **Are Elo ratings computed correctly?** The expected-score formula is standard in `incremental_predictor.py:182-183`. The updates use reasonable but custom choices: K=32 plus a margin multiplier and +50 home advantage in `incremental_predictor.py:309-314`. `match_predictor.py:131-141` uses tournament-specific K factors. These are defensible custom Elo variants, not a single canonical FIFA/Elo implementation.

4. **Is XGBoost overfitting?** Likely in several places. `predict_2026.py:139-143`, `monte_carlo_2026.py:432-436`, and `analyze_explain.py:180-184` fit fixed 300-tree depth-6 models without validation, early stopping, calibration, or a train/test report. `incremental_predictor.py:724-741` has regularization and per-WC validation reporting, but no tuning loop or calibration.

5. **Is the 2026 bracket structure correct?** The requested R16 and QF pairings are implemented correctly in `predict_2026.py:390-409` and `monte_carlo_2026.py:278-302`. However, the R32 third-place assignment and group qualification are not robust. `monte_carlo_2026.py:188-190` and `monte_carlo_2026.py:237-265` are internally inconsistent, and `predict_2026.py:359-380` hard-codes one best-third combination.

6. **Are SHAP values and feature importances trustworthy?** Only as descriptions of the fitted models, not as evidence of valid predictors. `analyze_explain.py` explains a model whose country-feature columns are mismatched with the dataset (`fifa_ranking`, `urban_population_pct`, `health_expenditure_pct_gdp` do not exist in the dataset). `sota_analysis.py:680-864` fits interpretation models on the full dataset, so importances are not out-of-sample explanations.

7. **Any bugs that invalidate results?** Yes. The 2026 remaining-match omissions, Monte Carlo third-place placeholder bug, Germany/West Germany label split, and final-winner metric definition are all result-invalidating for their respective claims.

## Critical Issues

### 1. Future leakage in `is_former_champion`

`enrich_dataset.py:197-198` computes former champions from the complete dataset:

```text
former_champions = set(df[df['won_wc'] == 1]['country'].unique())
df['is_former_champion'] = df['country'].apply(lambda x: 1 if x in former_champions else 0)
```

This marks teams as former champions before they actually won. Spain, France, Argentina, and others can receive future knowledge in early tournaments. This leaks directly into `backtest.py`, `sota_analysis.py`, and `match_predictor.py` meta-modeling unless explicitly removed. It should be replaced with `wc_titles_before > 0`.

### 2. Same-year macro data is used before it would be known

`collect_data.py:522-532` says it uses a lag, but the loop tries `lag=0` first:

```text
for lag in [0, 1, 2]:
    data_year = year - lag
```

For a World Cup played mid-year, full-year GDP, population, health, and other indicators are not known pre-tournament. Use `year - 1` or earlier only, and document indicator release timing.

### 3. Germany/West Germany harmonization breaks historical labels

The project uses inconsistent names across files:

- `collect_data.py:36-44` uses `West Germany` for 1954, 1974, and 1990 winners.
- `collect_data.py:113` lists `Germany` among 1990 participants.
- `collect_data.py:571-574` counts prior achievements by exact string equality.

This means Germany's title history is split, and the 1990 country-level row can fail to have any `won_wc=1` team. Later analysis cannot fix a target label that was generated incorrectly.

### 4. 2026 scripts omit unplayed Groups H and I matches

`data/results.csv` contains unplayed 2026 World Cup matches:

- `Cape Verde` vs `Saudi Arabia`
- `Uruguay` vs `Spain`
- `Norway` vs `France`
- `Senegal` vs `Iraq`

But `predict_2026.py:238-243` hard-codes Groups H and I as complete, and `predict_2026.py:289-296` only simulates Groups J, K, and L. The same pattern is copied into `monte_carlo_2026.py:165-185` and `analyze_explain.py:310-318`. This is either future-result leakage or unsupported manual assumption.

### 5. Monte Carlo third-place bracket insertion is broken

`monte_carlo_2026.py:188-190` maps groups to R32 array indices:

```text
THIRD_MAP = {
    'A': 19, 'B': 17, 'D': 3, 'E': 13, 'F': 9, 'G': 25, 'I': 15, 'L': 29,
}
```

But `monte_carlo_2026.py:262-265` compares those values against FIFA match numbers:

```text
placeholder_map = {3: 74, 9: 77, 13: 79, 15: 80, 17: 81, 19: 82, 25: 85, 29: 87}
for idx, match_num in placeholder_map.items():
    if match_num in third_teams:
        r32_base[idx] = third_teams[match_num]
```

Since `third_teams` is keyed by indices such as `3`, `9`, and `17`, checks like `74 in third_teams` fail. The placeholders remain in the bracket. This invalidates the Monte Carlo 2026 distributions.

### 6. The 86.4% final-winner strategy is not a champion predictor

`incremental_predictor.py:821-895` walks through the actual tournament schedule, updates state with actual results, and only then records the prediction for the actual final at `incremental_predictor.py:859-865`. This is a valid match-by-match final prediction, but it is not a pre-tournament or bracket-simulation champion prediction. Reporting it next to exact winner accuracy is misleading.

### 7. Full-sample feature selection inflates SOTA logistic results

`sota_analysis.py:1381-1394` runs univariate feature selection and VIF pruning on the full sample. `sota_analysis.py:1413-1414` then tunes regularized logistic LOWCO using that selected feature set. The held-out World Cup influences which features are selected, so the reported best regularized logit AUC is optimistic.

## File-by-File Review

### `predict_2026.py`

Major issues:

- `predict_2026.py:238-243` treats Groups H and I as complete, but the CSV still has unplayed matches for those groups.
- `predict_2026.py:289-296` omits the remaining H/I matches entirely.
- `predict_2026.py:330-349` uses `_order` for many groups and best-thirds sorting by points/GD only. FIFA third-place ranking should include goals scored after GD; the code discards GF for third-place sorting.
- `predict_2026.py:334-339` bypasses algorithmic tiebreakers whenever `_order` exists. That makes the group table non-reproducible and hides whether pts/GD/GF were actually applied.
- `predict_2026.py:189-196` never allows a group-stage draw because `predict()` always chooses `ta` or `tb` based on home/away win probabilities. The draw branch in `predict_2026.py:316-321` is unreachable.
- `predict_2026.py:147-157` loads 2022 country features, but `predict_2026.py:66-73` requests columns that do not exist in `data/world_cup_predictors_dataset.csv`: `urban_population_pct`, `health_expenditure_pct_gdp`, and `fifa_ranking`.
- `predict_2026.py:103` trains country-feature columns with an empty country feature map, while inference uses 2022 country features in `predict_2026.py:259-263`. This creates train/inference mismatch.
- `predict_2026.py:139-143` trains XGBoost on all available non-2026-WC matches with no test split, early stopping, or calibration.
- `predict_2026.py:274-280` accidentally includes `_order` as a pseudo-team when marking participation.

Positive notes:

- The requested R16 crossover structure is implemented correctly in `predict_2026.py:390-399`.
- The requested QF pairings are implemented correctly in `predict_2026.py:404-409`.
- Dynamic match features are built before updating match state in `predict_2026.py:97-106`, so the rolling features themselves are temporally ordered during model training.

### `incremental_predictor.py`

Major issues:

- The headline 86.4% exact final-winner rate is conditional on knowing the actual historical final matchup and all prior tournament results. See `incremental_predictor.py:859-865` and `incremental_predictor.py:892-894`.
- `select_final_match_index()` uses `WC_WINNERS` to identify the final in `incremental_predictor.py:329-346`. Stage labels are known historically, but recovering them from the actual champion is a fragile and outcome-dependent implementation.
- Country-level features are limited to the closest prior World Cup year in `incremental_predictor.py:431-444`. This avoids future leakage but can make inputs stale and sparse for teams absent in the previous tournament.
- The model is retrained once per World Cup in `incremental_predictor.py:802`, not before each match. That is consistent with the script docstring, but it means no model adaptation within a tournament.

Positive notes:

- The core walk-forward feature construction is mostly leakage-safe. `build_global_feature_table()` builds rows before calling `update_team_states()` and `update_h2h()` in `incremental_predictor.py:687-719`.
- Backtest training uses only matches before the World Cup start in `incremental_predictor.py:784-805`.
- Elo expected score in `incremental_predictor.py:182-183` is standard. The K=32 plus margin multiplier update in `incremental_predictor.py:309-314` is a reasonable custom variant.
- World Cup experience snapshots are explicitly prior-tournament only in `incremental_predictor.py:447-527`.

Assessment: good match-level walk-forward framework, but the final-winner metric is not a valid champion forecast.

### `monte_carlo_2026.py`

Major issues:

- Same omitted Groups H/I problem as `predict_2026.py`: `monte_carlo_2026.py:165-177` hard-codes standings and `monte_carlo_2026.py:179-185` only simulates J/K/L.
- The third-place insertion bug in `monte_carlo_2026.py:188-190` and `monte_carlo_2026.py:237-265` invalidates R32 assignments.
- `monte_carlo_2026.py:223-230` initially honors `_order` but then re-sorts after stochastic results. That is better than `predict_2026.py`, but still only uses points, GD, GF and not the later FIFA tie-breakers.
- `monte_carlo_2026.py:432-436` trains the same unvalidated XGBoost model as `predict_2026.py`.
- `monte_carlo_2026.py:495-496` computes binomial confidence intervals using percentage units. Algebraically this works because `pct` is 0-100, but it is easy to misread and should be computed on proportions for clarity.

Positive notes:

- Probability sampling from model outputs is conceptually correct in `monte_carlo_2026.py:121-140`.
- Knockout draw handling re-normalizes home/away win probability in `monte_carlo_2026.py:126-128`, which is a reasonable penalty/extra-time approximation.
- R16 and QF pairings match the requested structure in `monte_carlo_2026.py:278-302`.

Assessment: the sampling idea is sound, but the current bracket implementation is not.

### `analyze_explain.py`

Major issues:

- It explains the same flawed 2026 state assumptions as the other 2026 scripts. Groups H/I are hard-coded in `analyze_explain.py:310-318`.
- It has the same feature-name mismatch as `predict_2026.py`: `analyze_explain.py:69-76` asks for columns not present in the dataset, and `analyze_explain.py:124-128` loads those missing names.
- The model is trained with no validation in `analyze_explain.py:180-184`.
- SHAP API handling in `analyze_explain.py:225-230` is fragile. Depending on XGBoost/SHAP versions, multiclass SHAP values may be a list or an array; checking `len(shap_values.shape)` will fail if `shap_values` is a list.
- The script labels SHAP for class 0 as "Brazil win" in `analyze_explain.py:221-239`. That is only true because Brazil is passed as the home team. It is not a generic team-specific class.

Assessment: useful for exploratory debugging, not reliable enough for published explanation.

### `sota_analysis.py`

Major issues:

- LOWCO is implemented in `sota_analysis.py:498-505`, but it is not chronological. For an early held-out tournament, training includes future tournaments. That is acceptable for leave-one-group cross-validation, but not for a historical forecasting claim.
- Full-sample feature selection occurs before LOWCO regularized logistic evaluation in `sota_analysis.py:1381-1414`, creating model-selection leakage.
- Full-dataset interpretation models are fit in `sota_analysis.py:680-864`. Their importances are descriptive, not out-of-sample.
- The causal analysis is weak. The DiD model in `sota_analysis.py:901-941` lacks country and year fixed effects and treats hosting as permanently "post" after first hosting. PSM in `sota_analysis.py:993-1044` lacks balance checks. The IV attempt in `sota_analysis.py:1046-1097` uses prior titles as an instrument for Elo, but prior titles plausibly affect World Cup winning directly through tradition and selection, violating exclusion.

Positive notes:

- Direct leakage columns are explicitly identified and dropped in `sota_analysis.py:64-72` and `sota_analysis.py:1372-1374`.
- Train-only imputation/scaling per LOWCO fold is handled in `sota_analysis.py:527-543`.
- The summary correctly cautions that causal effects are suggestive and the IV is likely invalid.

Assessment: strongest analysis script structurally, but the best AUC is still optimistic due to feature-selection leakage and non-chronological CV.

### `collect_data.py`

Major issues:

- Same-year World Bank data creates temporal leakage in `collect_data.py:523-532`.
- Winner-relative variables are direct target leakage in `collect_data.py:629-641`. SOTA and backtest drop them, but `analysis.py` does not.
- Germany/West Germany, USSR/Russia, Yugoslavia/Serbia, and Czechoslovakia/Czech Republic are not harmonized before computing historical achievement features in `collect_data.py:571-599`.
- `COUNTRY_TO_ISO3` has data quality problems, including `Honduras` mapped to `HUN` in `collect_data.py:70`.
- England, Scotland, Wales, and Northern Ireland are all mapped to `GBR` in `collect_data.py:68`, `collect_data.py:76`, and `collect_data.py:87`, which causes shared World Bank features for distinct FIFA teams.
- `collect_fifa_rankings()` returns only winner Elo proxies in `collect_data.py:162-200`; its result is assigned in `collect_data.py:689` but never used.

Positive notes:

- The file makes the intended panel structure explicit and saves raw World Bank data.
- Historical features are at least intended to be pre-tournament (`*_before`), but the name harmonization bug undermines them.

### `enrich_dataset.py`

Major issues:

- `is_former_champion` is future-leaky in `enrich_dataset.py:197-198`.
- `football_power_index` in `enrich_dataset.py:188-194` combines potentially leaked/static football tradition with Elo and history. If `football_tradition` is manually informed by modern outcomes, it can leak era knowledge.
- Country names are not harmonized before applying rankings and Elo. For example, `FIFA_RANKINGS_AT_WC` uses `Bosnia` in `enrich_dataset.py:76`, but the data uses `Bosnia and Herzegovina`.
- Several ranking dictionaries appear manually curated and should be treated as approximate, not official source-of-truth data.

Positive notes:

- It creates useful `fifa_rank`, `fifa_rank_inverse`, and `elo_rating` fields, but other scripts use inconsistent names such as `fifa_ranking`.

### `backtest.py`

Major issues:

- It is not a true historical out-of-sample backtest. `backtest.py:50-52` trains on all other World Cups, including future tournaments relative to the test year.
- It inherits leaked `is_former_champion` from the dataset because `backtest.py:27-30` does not drop it.
- It inherits the Germany/West Germany label and history problems from `collect_data.py`.
- `DATA_DIR` is hard-coded to one absolute path in `backtest.py:13`.

Positive notes:

- Direct outcome leaks and winner-relative columns are dropped in `backtest.py:27-30`.
- Imputation and scaling are fit on train only in `backtest.py:59-66`.
- Ranking metrics and pooled AUC/Brier are computed in a reasonable way in `backtest.py:76-132`.

Assessment: cross-validated ranking exercise, not a chronological forecasting backtest.

### `match_predictor.py`

Major issues:

- `build_match_feature_dataset()` uses `neutral = bool(row.neutral)` in `match_predictor.py:363`. If the CSV column is read as strings, `bool("False")` is `True`. In this dataset Pandas may infer booleans, but this is fragile. Use a parser like `incremental_predictor.py:165-171`.
- The historical tournament simulator reconstructs knockout trees from actual winners in `match_predictor.py:580-617`. This is not a real bracket simulation from group tables; it simulates along the actual historical bracket path.
- `run_tournament_simulation()` precomputes per-match probabilities before simulations in `match_predictor.py:649-674`, but it does not update team state inside each simulated tournament. Later-round probabilities for alternate simulated teams use pre-tournament state plus cached static matchups, not simulated prior-round fatigue/form/Elo.
- The meta-model in `match_predictor.py:830-916` uses `match_win_prob` generated from a simulator that already used actual historical knockout topology. This limits interpretation as a pre-tournament method.

Positive notes:

- Match features are built in chronological order and updated after row creation in `match_predictor.py:357-415`.
- Time splits are chronological in `match_predictor.py:497-532`.
- The match model has some regularization controls in `match_predictor.py:471-484`.

Assessment: useful match-level work, but historical tournament backtests are not clean bracket simulations.

### `analysis.py`

Major issues:

- This older analysis script does not drop `gdp_per_capita_vs_winner` or `population_vs_winner` in its modeling exclusions. See `analysis.py:61-64`, `analysis.py:134-137`, `analysis.py:175-179`, `analysis.py:299-303`, `analysis.py:358-362`, and `analysis.py:404-408`. Those are direct target-leak columns created in `collect_data.py`.
- It reports in-sample multivariate AUC in `analysis.py:263-266`, which is not a valid performance estimate.
- It scales the full dataset before LeaveOneGroupOut in `analysis.py:257-282`, causing preprocessing leakage into CV.
- Random forest and XGBoost feature importances in `analysis.py:317-344` and `analysis.py:372-390` are full-sample importances and inherit leaked features.
- Univariate p-values are not multiplicity corrected.

Assessment: exploratory only. Do not use its outputs as evidence.

## Model Validity Assessment

### 2026 deterministic prediction

Not valid in its current form. The bracket crossover and QF mapping are mostly correct, but the group stage is not. The script omits real unplayed matches, uses hard-coded standings, cannot draw remaining group games, and trains an unvalidated model with feature-name mismatches.

### 2026 Monte Carlo

Not valid in its current form. Probability sampling is conceptually correct, but R32 third-place insertion is broken and some remaining groups are hard-coded. Reported champion distributions from this script should be discarded until bracket construction is fixed and tested.

### Incremental predictor

Mostly valid as a match-by-match walk-forward classifier. Not valid as evidence that the project can predict 19/22 World Cup champions. The final-winner strategy answers: "given the actual finalists and all actual prior matches, who is favored in the final?"

### Country-level econometric/SOTA model

Partially valid after leakage fixes. The SOTA script is better than `analysis.py`, but current results are inflated by the leaked `is_former_champion` feature, non-chronological LOWCO training, and full-sample feature selection. Causal estimates should be treated as exploratory only.

### SHAP and importance

The strongest repeated signal is team strength: Elo, football power, prior titles/finals, and host status. That is plausible. But exact rankings and importance magnitudes are not trustworthy until leaked/static future-informed features are removed and SHAP is computed on models trained under clean out-of-sample protocols.

## Minor Issues and Edge Cases

- `predict_2026.py`, `monte_carlo_2026.py`, and `analyze_explain.py` duplicate large blocks of feature and bracket logic. This increases the risk of inconsistent fixes.
- `predict_2026.py:45` caps rest days at 60; `incremental_predictor.py:212` does not. The modeling assumptions differ across scripts.
- `predict_2026.py:195` and `monte_carlo_2026.py:121-130` ignore draw probability for advancing teams except as a secondary tiebreak. That is reasonable for knockouts but should be explicit.
- `sota_analysis.py` and `incremental_predictor.py` attempt to install missing packages at runtime. This makes reproducibility weaker than a pinned environment.
- `collect_data.py` has several manually curated football scores, Elo values, and rankings without source validation.
- `backtest.py` prints results but does not save a machine-readable artifact.

## Recommendations

1. **Fix the dataset first.** Harmonize country names before target and historical feature generation. Use a single canonical scheme for Germany/West Germany, USSR/Russia, Yugoslavia/Serbia, Czechoslovakia/Czech Republic, Korea names, Iran, USA, and Ivory Coast/Cote d'Ivoire.

2. **Remove future-leaky features.** Replace `is_former_champion` with `wc_titles_before > 0`. Drop or recompute same-year macro indicators using only values available before each tournament.

3. **Regenerate `world_cup_predictors_dataset.csv`.** After fixes, verify every World Cup year has exactly one `won_wc=1` row and all historical title counts match canonical history.

4. **Centralize feature engineering.** Move shared Elo, H2H, rolling form, country feature lookup, and name harmonization into one module used by all scripts.

5. **Centralize 2026 bracket logic and test it.** Write unit tests for:
   - R16 pairings: W73-W75, W74-W77, W76-W78, W79-W80, W83-W84, W81-W82, W86-W88, W85-W87.
   - QF pairings: W89-W90, W93-W94, W91-W92, W95-W96.
   - Best-third assignment for multiple advancing-group combinations.
   - Propagation of winners through every knockout round.

6. **Simulate all remaining group matches.** Do not hard-code Groups H/I while treating J/K/L as simulated. Compute standings from completed CSV rows plus simulated remaining matches using FIFA tie-breakers.

7. **Use chronological backtests for forecasting claims.** For a World Cup in year Y, train only on data available before that tournament. LOWCO can remain as a robustness check, but do not call it historical forecasting.

8. **Separate match prediction from champion prediction.** Report the 86.4% final-winner metric as "actual final match winner accuracy", not "exact World Cup winner accuracy." For champion prediction, simulate the full tournament from pre-tournament information.

9. **Add calibration and baselines.** Report log-loss, Brier score, calibration curves, and comparisons against Elo-only, FIFA-rank-only, and betting-market-style priors where possible.

10. **Recompute SHAP after leakage fixes.** Use models trained under clean chronological splits, and compute SHAP on held-out folds or clearly label full-sample explanations as descriptive.

## Bottom Line

The project has promising components, especially the incremental match-state machinery, but current results are materially overstated. The 2026 predictions and Monte Carlo distributions should not be trusted until the group-stage completion, third-place bracket mapping, feature-name mismatches, and leakage in the country dataset are fixed.
