# Incremental Predictor Summary

## Setup
- Strategy: strict chronological incremental prediction (no bracket simulation).
- Model: XGBoost multiclass (`home_win`, `draw`, `away_win`).
- Dynamic state: Elo (K=32, +50 home advantage, margin multiplier), form, rolling goals, rest days, H2H.
- Context: stage, neutral/home-host indicators, tournament progress in current WC, WC experience, country-level features from closest preceding WC year.

## Backtest Results (1930-2022)
- World Cups evaluated: 22
- Exact winner accuracy (most predicted wins): 0.545
- Exact winner accuracy (predicted final winner): 0.864
- Exact winner accuracy (aggregate probability): 0.545
- Top-3 accuracy (aggregate): 0.909
- Top-5 accuracy (aggregate): 0.955
- Match-level accuracy (all WC matches): 0.540
- Match-level log-loss (all WC matches): 1.034
- Winner-vs-non-winner AUC (aggregate score): 0.934

## Historical Baseline Comparison
- Country-level model reference: exact 0.409, top-3 0.636
- Monte Carlo simulator reference: exact 0.318, top-3 0.500

## Per-WC Snapshot
- 1930: actual=Uruguay, most_wins=Argentina, final=Uruguay, aggregate=Argentina, acc=0.611
- 1934: actual=Italy, most_wins=Italy, final=Italy, aggregate=Italy, acc=0.647
- 1938: actual=Italy, most_wins=Brazil, final=Hungary, aggregate=Brazil, acc=0.611
- 1950: actual=Uruguay, most_wins=Brazil, final=Brazil, aggregate=Brazil, acc=0.500
- 1954: actual=Germany, most_wins=Germany, final=Germany, aggregate=Germany, acc=0.577
- 1958: actual=Brazil, most_wins=Brazil, final=Brazil, aggregate=Brazil, acc=0.429
- 1962: actual=Brazil, most_wins=Brazil, final=Brazil, aggregate=Brazil, acc=0.531
- 1966: actual=England, most_wins=England, final=England, aggregate=England, acc=0.562
- 1970: actual=Brazil, most_wins=Brazil, final=Brazil, aggregate=Brazil, acc=0.688
- 1974: actual=Germany, most_wins=Germany, final=Germany, aggregate=Germany, acc=0.474
- 1978: actual=Argentina, most_wins=Brazil, final=Argentina, aggregate=Brazil, acc=0.474
- 1982: actual=Italy, most_wins=Germany, final=Italy, aggregate=Germany, acc=0.481
- 1986: actual=Argentina, most_wins=England, final=Argentina, aggregate=France, acc=0.442
- 1990: actual=Germany, most_wins=Germany, final=Germany, aggregate=Germany, acc=0.577
- 1994: actual=Brazil, most_wins=Brazil, final=Brazil, aggregate=Brazil, acc=0.519
- 1998: actual=France, most_wins=Brazil, final=France, aggregate=Brazil, acc=0.547
- 2002: actual=Brazil, most_wins=Brazil, final=Brazil, aggregate=Brazil, acc=0.594
- 2006: actual=Italy, most_wins=Germany, final=Italy, aggregate=Germany, acc=0.641
- 2010: actual=Spain, most_wins=Spain, final=Spain, aggregate=Spain, acc=0.516
- 2014: actual=Germany, most_wins=Brazil, final=Argentina, aggregate=Brazil, acc=0.547
- 2018: actual=France, most_wins=England, final=France, aggregate=England, acc=0.516
- 2022: actual=Argentina, most_wins=Argentina, final=Argentina, aggregate=Argentina, acc=0.531
