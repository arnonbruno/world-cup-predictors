Integrate Transfermarkt squad market values into the World Cup prediction model.

## DATA
File: `data/squad_values.csv` (169 rows, 86 teams, 2014-2026)
Columns: team, year, total_squad_value_eur, avg_player_value_eur, num_players, source
Coverage: ALL 86 World Cup teams across 2014/2018/2022/2026
Team names match results.csv (e.g., "Iran", "South Korea", "Ivory Coast")

## YOUR TASK

### 1. Read existing code first
Read shared.py, predict_2026.py, backtest_2026_wc.py, explain_match.py, monte_carlo_2026.py to understand the current architecture.

### 2. Add squad value loading to shared.py
Create `load_squad_values()` similar to `load_betting_odds()`:
- Load data/squad_values.csv
- Return a dict keyed by (team, year) to total_squad_value_eur, avg_player_value_eur
- Use year lag (look up 2022 values when predicting 2023 matches, etc.)

### 3. Add squad value features to compute_match_features()
Add these features:
- `squad_value`: home team squad value (EUR, log-scaled)
- `opp_squad_value`: away team squad value (EUR, log-scaled)
- `squad_value_diff`: difference (log-scaled)
- `squad_value_ratio`: ratio of home/away values
These should use the most recent WC year values available before the match date.

### 4. Wire into backtest_2026_wc.py
- Load squad values alongside betting odds
- Pass to compute_match_features

### 5. Wire into predict_2026.py
- Load squad values
- Pass to compute_match_features for 2026 predictions

### 6. Wire into explain_match.py and monte_carlo_2026.py
- Same pattern

### 7. Run backtest and report results
Run `python3 backtest_2026_wc.py` and compare:
- Previous: 64.5% accuracy, 0.8858 log-loss, 0.1791 Brier
- Report new numbers

### 8. Run predict_2026.py to verify 2026 predictions still work

## IMPORTANT NOTES
- squad_values.csv team names match results.csv directly (both use raw names like "Iran", "South Korea", "Ivory Coast")
- The state dict uses harmonized names ("IR Iran", "Korea Republic", "Cote d'Ivoire"). You need to handle this mapping using NAME_ALIASES from shared.py
- Log-scale the squad values (use log1p) since they span 1M to 1.8B EUR
- Handle missing values gracefully (NaN for teams/years without data)
- All files must parse clean (python3 -m py_compile)
- Do NOT break existing features

## DELIVERABLE
Write a brief SQUAD_VALUES_RESULTS.md with the backtest comparison.
