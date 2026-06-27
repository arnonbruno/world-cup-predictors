# Match Explanation: Brazil vs Japan

Stage code: `1` | Year: `2026` | Match date: `2026-06-29`

## Prediction

- Model probabilities: **Home 63.4%, Draw 0.0%, Away 36.6%**
- *Knockout stage: draw excluded, probabilities renormalized to P(home|no draw) and P(away|no draw)*
- Most likely outcome: **Home win**
- Confidence: **63.4%**; top-two margin: **26.9%**; entropy: **0.598**
- Elo-only baseline: **Home 63.6%, Draw 0.0%, Away 36.4%** (Elo difference: Brazil minus Japan = 97.)
- Similar-confidence history: Among 8359 training cases with similar confidence, the top prediction was correct 65.2% of the time.
- Brier/calibration note: Multiclass Brier score on available training data: 0.5004. Full reliability/resolution decomposition needs held-out bins.

## Top SHAP Drivers

SHAP values below are raw multiclass margin contrasts: home-win margin contribution minus away-win margin contribution. Signs show local direction; magnitudes are not probability points.

| Feature | Value | Contribution | Favors | Plain-English meaning |
|---|---:|---:|---|---|
| `elo_diff` | 96.6 | +0.3775 | Brazil | Elo summarizes team strength from historical results; larger values usually indicate a stronger side. |
| `elo_diff_pre` | 220 | +0.2658 | Brazil | Elo summarizes team strength from historical results; larger values usually indicate a stronger side. |
| `neutral` | 1.000 | -0.1812 | Japan | Contextual venue features describe whether either side receives location or hosting advantage. |
| `avg_goals_scored_10` | 2.400 | -0.1402 | Japan | Goal features capture attacking output and finishing strength. |
| `population` | 210,306,415 | +0.0979 | Brazil | Economic and demographic features are broad background proxies for football infrastructure and talent pool. |
| `stage` | 1.000 | -0.0808 | Japan | Tournament stage features let the model adjust expectations for group and knockout match dynamics. |
| `opp_football_power_index` | 21.3 | +0.0788 | Brazil | This engineered input is part of the model's matchup profile for the two teams. |
| `power_diff` | 164 | +0.0725 | Brazil | This engineered input is part of the model's matchup profile for the two teams. |
| `football_power_index` | 185 | +0.0698 | Brazil | This engineered input is part of the model's matchup profile for the two teams. |
| `urbanization_pct` | 87.3 | +0.0617 | Brazil | This engineered input is part of the model's matchup profile for the two teams. |
| `is_home` | 0.000 | -0.0616 | Japan | This engineered input is part of the model's matchup profile for the two teams. |
| `opp_elo_pre_tournament` | 1,860 | +0.0538 | Brazil | Elo summarizes team strength from historical results; larger values usually indicate a stronger side. |
| `gdp_per_capita` | 9,281 | +0.0428 | Brazil | Economic and demographic features are broad background proxies for football infrastructure and talent pool. |
| `health_spending_pct_gdp` | 9.387 | -0.0306 | Japan | Economic and demographic features are broad background proxies for football infrastructure and talent pool. |
| `avg_goals_conceded_10` | 1.100 | +0.0303 | Brazil | Goal features capture attacking output and finishing strength. |

Waterfall plot: `/var/mnt/DATA/Hermes/workspace/world-cup-predictors/output/explain/brazil_vs_japan_shap.png`
Force plot: `/var/mnt/DATA/Hermes/workspace/world-cup-predictors/output/explain/brazil_vs_japan_force.html`

## Historical Context

### Brazil

- Recent form: win rate 60.0%, goals for 2.400/match, goals against 1.100/match, clean sheets 3.
- Last 10 results: 2025-10-14: L 2.000-3.000 vs Japan; 2025-11-15: W 2.000-0.000 vs Senegal; 2025-11-18: D 1.000-1.000 vs Tunisia; 2026-03-26: L 1.000-2.000 vs France; 2026-03-31: W 3.000-1.000 vs Croatia; 2026-05-31: W 6.000-2.000 vs Panama; 2026-06-06: W 2.000-1.000 vs Egypt; 2026-06-13: D 1.000-1.000 vs Morocco; 2026-06-19: W 3.000-0.000 vs Haiti; 2026-06-24: W 3.000-0.000 vs Scotland
- Elo trajectory: unavailable.
- World Cup history: participations 23, titles 5, recent years [1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026].

### Japan

- Recent form: win rate 70.0%, goals for 2.000/match, goals against 0.700/match, clean sheets 6.
- Last 10 results: 2025-10-10: D 2.000-2.000 vs Paraguay; 2025-10-14: W 3.000-2.000 vs Brazil; 2025-11-14: W 2.000-0.000 vs Ghana; 2025-11-18: W 3.000-0.000 vs Bolivia; 2026-03-28: W 1.000-0.000 vs Scotland; 2026-03-31: W 1.000-0.000 vs England; 2026-05-31: W 1.000-0.000 vs Iceland; 2026-06-14: D 2.000-2.000 vs Netherlands; 2026-06-20: W 4.000-0.000 vs Tunisia; 2026-06-25: D 1.000-1.000 vs Sweden
- Elo trajectory: unavailable.
- World Cup history: participations 8, titles 0, recent years [1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026].

### Head-To-Head

- All-time: Brazil 11 wins, 2 draws, Japan 1 wins (14 matches)
- Last 10 years: Brazil 2 wins, 0 draws, Japan 1 wins (3 matches)

## Match Factor Decomposition

### Team Strength

- Net SHAP contribution: **+0.9181**, favors **Brazil**.
- Raw feature values: `elo`=2,085, `elo_opponent`=1,988, `elo_diff`=96.6, `elo_sum`=4,073, `elo_pre_tournament`=2,080, `fifa_rank`=1.000
- Main local drivers: `elo_diff` (+0.378), `elo_diff_pre` (+0.266), `opp_football_power_index` (+0.079), `power_diff` (+0.072), `football_power_index` (+0.070)

### Form

- Net SHAP contribution: **-0.1099**, favors **Japan**.
- Raw feature values: `form_win_rate`=0.600, `form_draw_rate`=0.200, `form_loss_rate`=0.200, `avg_goals_scored_5`=3.000, `avg_goals_conceded_5`=0.800, `avg_goals_scored_10`=2.400
- Main local drivers: `avg_goals_scored_10` (-0.140), `avg_goals_conceded_10` (+0.030)

### Head-to-Head

- Net SHAP contribution: **+0.0000**, favors **Neither**.
- Raw feature values: `h2h_matches`=14.0, `h2h_win_rate`=0.786, `h2h_draw_rate`=0.143, `h2h_avg_goals_for`=2.643, `h2h_avg_goals_against`=0.571

### Experience

- Net SHAP contribution: **-0.0808**, favors **Japan**.
- Raw feature values: `wc_participations`=80.0, `wc_titles`=23.0, `wc_win_rate`=0.667, `stage`=1.000
- Main local drivers: `stage` (-0.081)

### Economic/Demographic

- Net SHAP contribution: **+0.1101**, favors **Brazil**.
- Raw feature values: `gdp_per_capita`=9,281, `population`=210,306,415, `health_spending_pct_gdp`=9.387
- Main local drivers: `population` (+0.098), `gdp_per_capita` (+0.043), `health_spending_pct_gdp` (-0.031)

### Contextual

- Net SHAP contribution: **-0.2620**, favors **Japan**.
- Raw feature values: `rest_days`=5.000, `stage`=1.000, `neutral`=1.000
- Main local drivers: `neutral` (-0.181), `stage` (-0.081)

## Counterfactuals

- Underdog Elo +100: Home 58.2%, Draw 0.0%, Away 41.8%. Changed: Japan rolling Elo +100; dependent Elo features recomputed.
- Head-to-head record reversed: Home 52.3%, Draw 0.0%, Away 47.7%. Changed: H2H wins/losses and goals reversed in rolling state; H2H features recomputed.
- Brazil has venue advantage: Home 69.7%, Draw 0.0%, Away 30.3%. Changed: Brazil marked as non-neutral home side; venue features recomputed.
- Single-feature sensitivity search: Changing `elo_diff` to -120 moves the opposing win probability by -11.9%; new probabilities are Home 24.3%, Draw 51.1%, Away 24.6%.

## Causal And SOTA Context

- `output/sota/causal_dag.md`: # Hypothesized Causal DAG Host[Hosting] --> Win[World Cup Win]
- `output/sota/causal_did_hosting.csv`: difference_in_differences_hosting,0.05374766382650878,0.021499952949274304,0.0847156525366232,0.000956014080999036
- `output/sota/causal_granger_like.csv`: spec,features,mean_auc_lowco,mean_f1_lowco

## Runtime Trace

- Loaded country feature history through `shared.load_country_feature_history()`.
- Loaded match history from `/var/mnt/DATA/Hermes/workspace/world-cup-predictors/data/results.csv`.
- Trained model bundle through `predict_2026.train_model_bundle()`.
- Loaded country feature map for 2022 through `shared.country_features_for_year()`.
- Built match features through strict `shared.compute_match_features()` call.
- Computed SHAP contrast on the model's raw multiclass margin: home-win margin contribution minus away-win margin contribution.
