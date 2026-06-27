# Leakage Audit
## Drop (Direct Leakage)
- `runner_up`
- `semifinalist`
- `finalist`
- `top4`
- `is_winner`
- `gdp_per_capita_vs_winner`
- `population_vs_winner`

## Keep (Safe)
- `is_host`
- `elo_rating`
- `fifa_rank`
- `fifa_rank_inverse`
- `football_tradition`
- `football_power_index`
- `is_former_champion`
- `is_strong_europe`
- `is_strong_sa`
- `wc_titles_before`
- `wc_finals_before`
- `wc_semifinals_before`
- `wc_participations_before`
- `years_since_last_wc`
- `years_since_last_win`
- `years_since_last_final`

## Keep but Validate
- `gdp_per_capita_vs_avg`
- `population_vs_avg`

## Excluded from Prediction (Post-Tournament)
- `total_goals_in_tournament`
- `avg_goals_per_match`

## Validation Notes
- `gdp_per_capita_vs_avg` and `population_vs_avg` are cross-sectional within-year averages and retained.
- World Bank indicators are used as provided, with LOWCO split preserving strict train-only preprocessing.
