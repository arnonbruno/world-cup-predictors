# Match Explanation: Brazil vs Norway

Stage code: `2` | Year: `2026` | Match date: `2026-06-29`

## Prediction

- Final reported probabilities: **Home 74.0%, Draw 0.0%, Away 26.0%**
- Raw blended 3-way probabilities: **Home 55.8%, Draw 33.8%, Away 10.5%**
- Knockout no-draw probabilities before WC calibration: **Home 84.2%, Draw 0.0%, Away 15.8%**
- *Knockout stage: draw excluded for advancement-style reporting, then WC-specific calibration is applied to the favorite confidence.*
- WC calibration: WC bucket 80%-90%: n=7, avg_conf=83.9%, acc=57.1%, smoothed=74.0%
- Warning: bookmaker odds are unavailable for this fixture, so the prediction is less market-informed.
- Most likely outcome: **Home win**
- Confidence: **74.0%**; top-two margin: **48.1%**; entropy: **0.521**
- Elo-only baseline: **Home 71.8%, Draw 0.0%, Away 28.2%** (Elo difference: Brazil minus Norway = 162.)
- Similar-confidence history: Among 2380 training cases with similar confidence, the top prediction was correct 82.2% of the time.
- Brier/calibration note: Multiclass Brier score on available training data: 0.5476. Full reliability/resolution decomposition needs held-out bins.

## Top SHAP Drivers

SHAP values below are raw multiclass margin contrasts: home-win margin contribution minus away-win margin contribution. Signs show local direction; magnitudes are not probability points.

SHAP values were not available for this run.

## Historical Context

### Brazil

- Recent form: win rate 60.0%, goals for 2.400/match, goals against 1.100/match, clean sheets 3.
- Last 10 results: 2025-10-14: L 2.000-3.000 vs Japan; 2025-11-15: W 2.000-0.000 vs Senegal; 2025-11-18: D 1.000-1.000 vs Tunisia; 2026-03-26: L 1.000-2.000 vs France; 2026-03-31: W 3.000-1.000 vs Croatia; 2026-05-31: W 6.000-2.000 vs Panama; 2026-06-06: W 2.000-1.000 vs Egypt; 2026-06-13: D 1.000-1.000 vs Morocco; 2026-06-19: W 3.000-0.000 vs Haiti; 2026-06-24: W 3.000-0.000 vs Scotland
- Elo trajectory: unavailable.
- World Cup history: participations 23, titles 5, recent years [1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026].

### Norway

- Recent form: win rate 50.0%, goals for 2.200/match, goals against 1.400/match, clean sheets 1.
- Last 10 results: 2025-10-14: D 1.000-1.000 vs New Zealand; 2025-11-13: W 4.000-1.000 vs Estonia; 2025-11-16: W 4.000-1.000 vs Italy; 2026-03-27: L 1.000-2.000 vs Netherlands; 2026-03-31: D 0.000-0.000 vs Switzerland; 2026-06-01: W 3.000-1.000 vs Sweden; 2026-06-07: D 1.000-1.000 vs Morocco; 2026-06-16: W 4.000-1.000 vs Iraq; 2026-06-22: W 3.000-2.000 vs Senegal; 2026-06-26: L 1.000-4.000 vs France
- Elo trajectory: unavailable.
- World Cup history: participations 4, titles 0, recent years [1938, 1994, 1998, 2026].

### Head-To-Head

- All-time: Brazil 0 wins, 2 draws, Norway 2 wins (4 matches)
- Last 10 years: Brazil 0 wins, 1 draws, Norway 2 wins (3 matches)

## Match Factor Decomposition

### Team Strength

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `elo`=2,093, `elo_opponent`=1,931, `elo_diff`=162, `elo_sum`=4,024, `elo_pre_tournament`=2,077, `fifa_rank`=6.000

### Form

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `form_win_rate`=0.700, `form_draw_rate`=0.200, `form_loss_rate`=0.100, `avg_goals_scored_5`=2.200, `avg_goals_conceded_5`=0.600, `avg_goals_scored_10`=2.400

### Head-to-Head

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `h2h_matches`=0.000, `h2h_win_rate`=0.000, `h2h_draw_rate`=0.000, `h2h_avg_goals_for`=0.000, `h2h_avg_goals_against`=0.000

### Experience

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `wc_participations`=80.0, `wc_titles`=23.0, `wc_win_rate`=0.669, `stage`=2.000

### Economic/Demographic

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `gdp_per_capita`=9,281, `population`=210,306,415, `health_spending_pct_gdp`=9.387

### Contextual

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `rest_days`=0.000, `stage`=2.000, `neutral`=1.000

## Counterfactuals

- Underdog Elo +100: Home 82.1%, Draw 0.0%, Away 17.9%. Changed: Norway rolling Elo +100; dependent Elo features recomputed.
- Head-to-head record reversed: Home 84.2%, Draw 0.0%, Away 15.8%. Changed: H2H wins/losses and goals reversed in rolling state; H2H features recomputed.
- Brazil has venue advantage: Home 87.8%, Draw 0.0%, Away 12.2%. Changed: Brazil marked as non-neutral home side; venue features recomputed.
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
