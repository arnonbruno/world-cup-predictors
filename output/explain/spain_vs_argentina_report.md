# Match Explanation: Spain vs Argentina

Stage code: `5` | Year: `2026` | Match date: `2026-06-29`

## Prediction

- Final reported probabilities: **Home 33.9%, Draw 0.0%, Away 66.1%**
- Raw blended 3-way probabilities: **Home 27.0%, Draw 30.8%, Away 42.3%**
- Knockout no-draw probabilities before WC calibration: **Home 39.0%, Draw 0.0%, Away 61.0%**
- *Knockout stage: draw excluded for advancement-style reporting, then WC-specific calibration is applied to the favorite confidence.*
- WC calibration: WC bucket 60%-70%: n=30, avg_conf=64.9%, acc=66.7%, smoothed=66.1%
- Warning: bookmaker odds are unavailable for this fixture, so the prediction is less market-informed.
- Most likely outcome: **Away win**
- Confidence: **66.1%**; top-two margin: **32.3%**; entropy: **0.583**
- Elo-only baseline: **Home 49.3%, Draw 0.0%, Away 50.7%** (Elo difference: Spain minus Argentina = -5.)
- Similar-confidence history: Among 4422 training cases with similar confidence, the top prediction was correct 76.0% of the time.
- Brier/calibration note: Multiclass Brier score on available training data: 0.5476. Full reliability/resolution decomposition needs held-out bins.

## Top SHAP Drivers

SHAP values below are raw multiclass margin contrasts: home-win margin contribution minus away-win margin contribution. Signs show local direction; magnitudes are not probability points.

SHAP values were not available for this run.

## Historical Context

### Spain

- Recent form: win rate 60.0%, goals for 2.200/match, goals against 0.400/match, clean sheets 7.
- Last 10 results: 2025-10-14: W 4.000-0.000 vs Bulgaria; 2025-11-15: W 4.000-0.000 vs Georgia; 2025-11-18: D 2.000-2.000 vs Turkey; 2026-03-27: W 3.000-0.000 vs Serbia; 2026-03-31: D 0.000-0.000 vs Egypt; 2026-06-04: D 1.000-1.000 vs Iraq; 2026-06-08: W 3.000-1.000 vs Peru; 2026-06-15: D 0.000-0.000 vs Cape Verde; 2026-06-21: W 4.000-0.000 vs Saudi Arabia; 2026-06-26: W 1.000-0.000 vs Uruguay
- Elo trajectory: unavailable.
- World Cup history: participations 17, titles 1, recent years [1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026].

### Argentina

- Recent form: win rate 100.0%, goals for 2.900/match, goals against 0.200/match, clean sheets 8.
- Last 10 results: 2025-10-10: W 1.000-0.000 vs Venezuela; 2025-10-14: W 6.000-0.000 vs Puerto Rico; 2025-11-14: W 2.000-0.000 vs Angola; 2026-03-27: W 2.000-1.000 vs Mauritania; 2026-03-31: W 5.000-0.000 vs Zambia; 2026-06-06: W 2.000-0.000 vs Honduras; 2026-06-09: W 3.000-0.000 vs Iceland; 2026-06-16: W 3.000-0.000 vs Algeria; 2026-06-22: W 2.000-0.000 vs Austria; 2026-06-27: W 3.000-1.000 vs Jordan
- Elo trajectory: unavailable.
- World Cup history: participations 19, titles 3, recent years [1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026].

### Head-To-Head

- All-time: Spain 6 wins, 2 draws, Argentina 6 wins (14 matches)
- Last 10 years: Spain 2 wins, 0 draws, Argentina 1 wins (3 matches)

## Match Factor Decomposition

### Team Strength

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `elo`=2,185, `elo_opponent`=2,190, `elo_diff`=-4.902, `elo_sum`=4,376, `elo_pre_tournament`=2,148, `fifa_rank`=2.000

### Form

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `form_win_rate`=0.700, `form_draw_rate`=0.300, `form_loss_rate`=0.000, `avg_goals_scored_5`=1.800, `avg_goals_conceded_5`=0.200, `avg_goals_scored_10`=1.700

### Head-to-Head

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `h2h_matches`=0.564, `h2h_win_rate`=0.564, `h2h_draw_rate`=0.000, `h2h_avg_goals_for`=3.385, `h2h_avg_goals_against`=0.564

### Experience

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `wc_participations`=44.0, `wc_titles`=6.000, `wc_win_rate`=0.500, `stage`=5.000

### Economic/Demographic

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `gdp_per_capita`=30,319, `population`=47,786,102, `health_spending_pct_gdp`=9.680

### Contextual

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `rest_days`=-15.0, `stage`=5.000, `neutral`=1.000

## Counterfactuals

- Underdog Elo +100: Home 40.9%, Draw 0.0%, Away 59.1%. Changed: Spain rolling Elo +100; dependent Elo features recomputed.
- Head-to-head record reversed: Home 39.0%, Draw 0.0%, Away 61.0%. Changed: H2H wins/losses and goals reversed in rolling state; H2H features recomputed.
- Spain has venue advantage: Home 43.9%, Draw 0.0%, Away 56.1%. Changed: Spain marked as non-neutral home side; venue features recomputed.
- Single-feature sensitivity search: No numeric high-impact feature could be perturbed for flip analysis.

## Causal And SOTA Context

- `output/sota/causal_dag.md`: # Hypothesized Causal DAG Host[Hosting] --> Win[World Cup Win]
- `output/sota/causal_did_hosting.csv`: difference_in_differences_hosting,0.05374766382650878,0.021499952949274304,0.0847156525366232,0.000956014080999036
- `output/sota/causal_granger_like.csv`: spec,features,mean_auc_lowco,mean_f1_lowco

## Data Availability Notes

- Could not compute SHAP values: Model type not yet supported by TreeExplainer: <class 'shared.IsotonicProbabilityCalibrator'>

## Runtime Trace

- Loaded country feature history through `shared.load_country_feature_history()`.
- Loaded match history from `/var/mnt/DATA/Hermes/workspace/world-cup-predictors/data/results.csv`.
- Trained model bundle through `predict_2026.train_model_bundle()`.
- Loaded country feature map for 2026 through `shared.country_features_for_year()`.
- Built match features through strict `shared.compute_match_features()` call.
