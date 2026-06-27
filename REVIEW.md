# World Cup Prediction Pipeline Verification

## Executive Summary

**Overall verdict: FAIL for full pipeline soundness, PASS for several previously fixed items.**

The prior fix cycle did correct important issues: `is_former_champion` is now derived from `wc_titles_before`, World Bank indicators use year-1-or-earlier data, 2026 World Cup matches are excluded from 2026 model training, group-stage draws are supported, `_order` is gone, and the deterministic, Monte Carlo, and explanation scripts now share core feature/bracket helpers.

Remaining issues can still invalidate parts of the prediction results:

- The 2026 Round of 32 third-place assignment is not a complete FIFA allocation algorithm. It maps each third-place group to one fixed slot in `shared.py`, but the 48-team format requires allocation based on which eight third-place groups qualify and each slot's allowed source groups.
- Several 2026 teams have no country-feature rows in `data/world_cup_predictors_dataset.csv`; Algeria is especially concerning because it appears in historical participant lists but is missing from `COUNTRY_TO_ISO3`.
- `compute_match_features()` emits `wc_participations` and `wc_titles`, but the 2026 model training loops do not populate those fields historically. Then 2026 inference increments `wc_participations`, creating a train/inference mismatch.
- Leakage-prone winner-relative columns still exist in the generated dataset and are still used by the legacy `analysis.py`. `sota_analysis.py` still has some full-sample feature selection outside LOWCO for inferential/best-logit paths.

## Answers to Critical Questions

1. **Is there any remaining data leakage?** Yes. The core 2026 predictor avoids the largest direct leaks, but `collect_data.py` still generates winner-relative target-derived columns, `analysis.py` still analyzes them, and `sota_analysis.py` still performs some full-sample feature selection outside LOWCO.

2. **Is the XGBoost model properly validated?** Partially. The shared helper now adds a validation split, holdout metrics, and early stopping for the 2026 XGBoost models. The split is random rather than chronological, and some other XGBoost scripts use backtests/time splits rather than early stopping.

3. **Are the 2026 bracket pairings mathematically correct per FIFA?** R16, QF, SF, final, and third-place propagation are structurally correct. The Round of 32 third-place slot assignment is not sufficient for FIFA's allocation rules.

4. **Are country features loaded correctly for training and inference?** The feature column names are consistent in `shared.py`, and training/inference use the same feature vector. Coverage is incomplete for 2026 teams, so some teams are inferred with missing/default country features.

5. **Does draw handling work in group-stage simulations?** Yes. Deterministic group predictions can return draws, and Monte Carlo group sampling includes draws.

6. **Is the backtest truly chronological?** `backtest.py` is chronological (`wc_year < test_year`) with train-only imputation/scaling. `sota_analysis.py` LOWCO is not chronological by design.

7. **Are there bugs that could invalidate prediction results?** Yes: R32 third-place allocation, missing country-feature coverage, and inconsistent WC experience state features can materially affect 2026 predictions.

## Data Pipeline Verification

### `collect_data.py`

Correct:

- `WC_WINNERS`, runners-up, semifinalists, participants, and hosts are canonicalized through `harmonize_country()` before dataset construction (`collect_data.py:161-171`). This fixes the earlier Germany/West Germany winner-label split.
- Historical title counts are computed with `y < year` (`collect_data.py:592-623`).
- World Bank indicators use lagged data only: the loop tries `year - 1`, `year - 2`, and `year - 3`, never same-year values (`collect_data.py:545-554`). The population fallback also uses only years before the tournament (`collect_data.py:556-561`).
- Honduras maps to `HND`, and UK nations are separate football teams (`England=ENG`, `Scotland=SCO`, `Wales=WAL`, `Northern Ireland=NIR`) in `COUNTRY_TO_ISO3` (`collect_data.py:70-89`).
- Winner-relative columns are generated but are target-derived and must be excluded from predictive models (`collect_data.py:530-531`, `collect_data.py:651-663`).

Remaining issues:

- `COUNTRY_TO_ISO3` lacks `Algeria` even though Algeria is listed as a World Cup participant in 1982, 1986, 2010, and 2014 (`collect_data.py:62-100`, `collect_data.py:115-123`). The generated dataset has no Algeria rows, so Algeria has no historical country features for 2026 inference.
- Historical variant coverage is still incomplete. `Dutch East Indies` appears in the 1938 participants but has no shared alias/ISO mapping and is skipped (`collect_data.py:106`, `collect_data.py:431-432`).
- World Bank filling stops after the first available lag year even if that year has only a subset of indicators (`collect_data.py:545-554`). This is not leakage, but it reduces completeness.

### `enrich_dataset.py`

Correct:

- Country names are harmonized before ranking/Elo enrichment (`enrich_dataset.py:138-149`, `enrich_dataset.py:157-161`).
- `is_former_champion` is now leakage-safe because it is derived from `wc_titles_before > 0`, not from future winners (`enrich_dataset.py:201-202`).
- `football_power_index` uses prior titles/finals plus Elo/tradition (`enrich_dataset.py:193-199`).

Remaining issues:

- Manual ranking dictionaries still have data-quality quirks, such as duplicate France entries in 2014 and `Bosnia` shorthand before harmonization (`enrich_dataset.py:73-82`). These do not create future leakage but should be treated as approximate data.

## Shared Utilities Verification

### `shared.py`

Correct:

- `COUNTRY_FEATURE_COLUMNS` matches actual dataset columns (`shared.py:41-51`).
- `GROUP_2026_TEAMS` matches the 2026 World Cup group fixtures present in `data/results.csv` after harmonization (`shared.py:53-66`).
- `build_2026_group_state()` loads completed 2026 FIFA World Cup rows from `results.csv`, applies completed scores, and returns unplayed fixtures (`shared.py:250-269`).
- `sorted_group_standings()` sorts by points, goal difference, then goals for (`shared.py:272-276`).
- `rank_third_place_teams()` includes goals for after goal difference (`shared.py:279-284`).
- `fit_xgb_with_validation()` implements a validation split, holdout accuracy/log-loss, and best-effort XGBoost early stopping (`shared.py:195-217`).
- `compute_match_features()` builds the same feature dictionary used for training and inference (`shared.py:114-156`).

Remaining issues:

- `NAME_ALIASES` is improved but not exhaustive for all historical variants used elsewhere in the project. It lacks entries such as `East Germany`, `German DR`, `Dutch East Indies`, `Republic of Ireland`, `Burma`, `United Arab Republic`, and `Vietnam Republic` (`shared.py:17-39`).
- `build_round_of_32()` does not implement FIFA's full third-place allocation matrix. `THIRD_SLOT_BY_GROUP` assigns fixed slots by source group (`shared.py:68-77`, `shared.py:308-318`), but official third-place slots depend on which eight third-place groups qualify.
- `compute_match_features()` exposes `wc_participations` and `wc_titles` (`shared.py:138-139`), but the main 2026 training loops do not update those fields historically.

## 2026 Model Pipeline Verification

### `predict_2026.py`

Correct:

- Training excludes 2026 FIFA World Cup matches (`predict_2026.py:54-58`).
- Country features are pulled through `load_country_feature_history()`/`country_features_for_year()`, and the same `compute_match_features()` path is used for training and inference (`predict_2026.py:75-79`, `predict_2026.py:186-193`).
- Group-stage draw handling works: `predict()` returns `None` when draw probability is highest for stage 0, and group simulation applies a 1-1 result (`predict_2026.py:154-163`, `predict_2026.py:221-231`).
- Completed 2026 World Cup matches are processed into match state, and `build_2026_group_state()` supplies completed group tables plus remaining fixtures (`predict_2026.py:195-205`).
- `_order` is no longer present or treated as a pseudo-team.
- R16 and QF pairings use the non-sequential FIFA crossovers (`predict_2026.py:269-293`).
- The third-place match uses semifinal losers, not quarterfinal losers (`predict_2026.py:296-310`).

Remaining issues:

- R32 third-place assignment inherits the incomplete fixed-slot logic from `build_round_of_32()` (`predict_2026.py:264-267`, `shared.py:287-328`).
- Historical `wc_participations` and `wc_titles` are not updated during model training; only `wc_matches` and `wc_wins` are updated for historical World Cup matches (`predict_2026.py:107-110`). Then all 2026 teams get `wc_participations += 1` before inference (`predict_2026.py:206-212`).
- Inference uses 2022-or-earlier country features (`predict_2026.py:186-190`), but several 2026 teams have no rows in the country dataset and therefore receive default/missing values in `compute_match_features()` (`shared.py:125-155`).

### `monte_carlo_2026.py`

Correct:

- Training excludes 2026 FIFA World Cup matches (`monte_carlo_2026.py:243-246`).
- Third-place insertion uses the shared `build_round_of_32()` path, so the old index-vs-match-number bug is fixed (`monte_carlo_2026.py:134-135`).
- Group sampling uses model probabilities directly and allows draws (`monte_carlo_2026.py:104-112`, `monte_carlo_2026.py:118-124`).
- Knockout sampling handles model draw outcomes by selecting an advancing team proportional to non-draw win probabilities (`monte_carlo_2026.py:93-102`).
- R16, QF, SF, third-place, and final structures match `predict_2026.py` (`monte_carlo_2026.py:146-203`).

Remaining issues:

- The Monte Carlo bracket inherits the incomplete FIFA third-place allocation from `shared.py` (`monte_carlo_2026.py:134-135`, `shared.py:287-328`).
- It duplicates the training/state-update logic from `predict_2026.py`, including the unpopulated historical `wc_participations`/`wc_titles` issue (`monte_carlo_2026.py:290-293`, `monte_carlo_2026.py:315-318`).

### `analyze_explain.py`

Correct:

- Training excludes 2026 FIFA World Cup matches (`analyze_explain.py:125-128`).
- It uses `fit_xgb_with_validation()` for validation/early stopping (`analyze_explain.py:177-184`).
- Multiclass SHAP handling supports both list and array outputs via `shap_values_for_class()` and `shap_matrix_for_class()` (`analyze_explain.py:95-116`).
- Feature names come from the actual trained `X.columns.tolist()` and are reused for the explanation frame (`analyze_explain.py:177-184`, `analyze_explain.py:224-225`, `analyze_explain.py:251-252`).

Remaining issues:

- It inherits the same R32 third-place allocation and country-feature coverage issues from shared utilities (`analyze_explain.py:211-216`, `shared.py:287-328`).
- SHAP class 0 means "home-team win", not a team-invariant Brazil class (`analyze_explain.py:216-228`, `analyze_explain.py:247-257`).

## Backtest Verification

### `backtest.py`

Correct:

- Training is strictly chronological: each fold uses `df['wc_year'] < test_year` (`backtest.py:48-56`).
- Imputation and scaling are fit on train only and then applied to the test year (`backtest.py:64-71`).
- Direct outcome leaks, winner-relative variables, and post-tournament aggregate columns are dropped (`backtest.py:28-35`).
- Metrics are reasonable: exact winner, top-3, top-5, pooled AUC, and Brier score (`backtest.py:77-137`).

Remaining issue:

- This is a country-level winner backtest, not a match/bracket simulation. It assumes the tournament field is known.

## SOTA Analysis Verification

### `sota_analysis.py`

Correct:

- Direct leakage columns and post-tournament columns are audited and dropped from the predictive dataframe (`sota_analysis.py:64-72`, `sota_analysis.py:98-102`, `sota_analysis.py:1419-1423`).
- The generic LOWCO evaluator fits year-median imputers and scalers inside each fold on train data only (`sota_analysis.py:515-573`).
- Regularized-logit hyperparameter tuning selects features inside each LOWCO fold (`sota_analysis.py:619-637`).
- The causal section is exploratory, with an explicit DAG and caveats; the IV attempt self-flags weak/invalid instruments (`sota_analysis.py:929-1165`, `sota_analysis.py:1396-1399`).

Remaining issues:

- `lowco_splits()` is leave-one-tournament-out, not chronological (`sota_analysis.py:498-505`). That is fine for robustness analysis, but not for historical forecasting.
- Full-sample feature selection still occurs for the unpenalized inferential logit (`sota_analysis.py:1430-1448`).
- The final `best_regularized_logit` candidate is evaluated using `vif_selected`, which was selected on the full sample (`sota_analysis.py:1441-1448`, `sota_analysis.py:1568-1577`).
- Full-sample RF/XGB/SHAP interpretation models are descriptive, not out-of-sample importance estimates (`sota_analysis.py:729-913`).

## Incremental Predictor Verification

### `incremental_predictor.py`

Correct:

- Walk-forward feature rows are built before updating team state/H2H/tournament progress (`incremental_predictor.py:673-720`).
- Country vectors use only prior World Cup years, not the current tournament year (`incremental_predictor.py:431-444`).
- World Cup experience snapshots are constructed strictly from previous tournaments (`incremental_predictor.py:447-527`).
- Elo updates use a standard expected-score form with a reasonable home-advantage and margin multiplier variant (`incremental_predictor.py:182-188`, `incremental_predictor.py:309-314`).
- Rolling form and goal stats are based on previous matches only (`incremental_predictor.py:197-225`, `incremental_predictor.py:716-719`).
- Per-World-Cup training uses matches before the World Cup start date (`incremental_predictor.py:781-805`).

Remaining issues:

- The country feature lookup includes all numeric columns except a short exclusion list (`incremental_predictor.py:417-428`). Because the source dataset still contains target-derived/post-tournament columns, lagged versions of those fields can enter the match model. This is chronological-safe when strictly prior, but the semantics are questionable and should be explicitly filtered.
- The validation model does not use early stopping (`incremental_predictor.py:724-742`). It reports a time-window validation metric, which is useful, but not the same as the early-stopped XGBoost helper.

## Match Predictor Verification

### `match_predictor.py`

Correct:

- Neutral fields are parsed with `shared.parse_bool`, avoiding `bool("False")` (`match_predictor.py:29`, `match_predictor.py:365`, `match_predictor.py:453`, `match_predictor.py:657`).
- Match features are generated chronologically and state is updated only after feature creation (`match_predictor.py:346-418`).
- Time-split evaluation is chronological (`match_predictor.py:499-534`).
- Meta-model backtest trains on `wc_year < year` and uses train-only imputation/scaling (`match_predictor.py:863-888`).
- The meta-model drops direct leak and post-tournament country-level columns (`match_predictor.py:844-861`).

Remaining issues:

- Historical tournament simulation reconstructs knockout trees from actual winners and actual bracket topology (`match_predictor.py:582-619`). This is acceptable for a retrospective simulator, but it should not be described as a clean pre-tournament bracket forecast.
- Simulated tournament state is not updated match-by-match inside each Monte Carlo run; later-round probabilities use start-of-tournament state plus matchup identity (`match_predictor.py:635-733`).

## Legacy Analysis Verification

### `analysis.py`

Remaining issues:

- This legacy script still does not drop `gdp_per_capita_vs_winner` or `population_vs_winner` from modeling/importance analyses (`analysis.py:61-64`, `analysis.py:134-137`, `analysis.py:175-179`, `analysis.py:299-303`, `analysis.py:358-362`, `analysis.py:404-408`).
- It reports in-sample model metrics and full-sample importances (`analysis.py:257-285`, `analysis.py:317-390`).

Assessment: exploratory only. Do not use `analysis.py` outputs as evidence of model validity.

## Confirmed Fixes

- `is_former_champion` is now derived from `wc_titles_before` without future winners (`enrich_dataset.py:201-202`).
- World Bank data uses at least a one-year lag (`collect_data.py:545-554`).
- Honduras and UK-nation ISO mappings are corrected (`collect_data.py:70-89`).
- `WC_WINNERS` is canonicalized through shared harmonization (`collect_data.py:161-162`).
- Completed 2026 World Cup matches are loaded from `results.csv`, not hard-coded into group tables (`shared.py:250-269`).
- Third-place ranking includes goals for (`shared.py:279-284`).
- XGBoost training in the 2026 scripts now uses `fit_xgb_with_validation()` (`predict_2026.py:114-119`, `monte_carlo_2026.py:297-302`, `analyze_explain.py:177-184`).
- Group-stage draws work in deterministic and Monte Carlo simulations (`predict_2026.py:160-163`, `monte_carlo_2026.py:104-112`).
- `_order` is absent from the current Python codebase.
- Third-place match participants in the 2026 scripts come from semifinal losers (`predict_2026.py:304-310`, `monte_carlo_2026.py:180-197`).

## Bottom Line

The codebase is materially better than the prior review described, and many requested fixes are genuinely present. The pipeline is still not fully sound enough to trust final 2026 prediction results because the R32 third-place assignment is incomplete, 2026 country-feature coverage is sparse for several teams, and some legacy/SOTA paths still allow leakage or full-sample model selection. The country-level chronological backtest is now valid for its stated scope, but the full project should be considered **not yet merge-ready for prediction claims** until the remaining issues above are fixed and regression-tested.

<details>
<summary>Superseded prior review retained below</summary>

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

</details>
