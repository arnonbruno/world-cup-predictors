# Match-Level World Cup Predictor Summary

## Data and leakage controls
- Match-level features are generated strictly in chronological order.
- Every feature for a match uses only data available before kickoff.
- Name harmonization includes Germany/West Germany, USSR/Russia, Yugoslavia/Serbia, and additional aliases.

## Match model
- Model: XGBoost multi-class (`home_win`, `draw`, `away_win`).
- Features: Elo, recent form, rolling goals, H2H, rest days, scorer-based squad proxies, neutral/tournament context.

## Time-based split performance
- Mean XGBoost accuracy: 0.5938
- Mean Elo baseline accuracy: 0.5747
- Mean XGBoost log-loss: 0.8807

## Historical tournament simulation (1930-2022, 1000 Monte Carlo runs each)
- Exact winner accuracy: 0.3182
- Top-3 accuracy: 0.5000
- Top-5 accuracy: 0.6364
- Pooled AUC: 0.8457

## Meta-model (country-level + match win probability)
- Exact winner accuracy: 0.1905
- Top-3 accuracy: 0.5714
- Top-5 accuracy: 0.6190
- Pooled AUC: 0.8210

## Best predicted winners by year
- 1934: actual=Italy, predicted=Italy (OK)
- 1938: actual=Italy, predicted=France (MISS)
- 1950: actual=Uruguay, predicted=Italy (MISS)
- 1954: actual=Germany, predicted=Uruguay (MISS)
- 1958: actual=Brazil, predicted=Germany (MISS)
- 1962: actual=Brazil, predicted=Brazil (OK)
- 1966: actual=England, predicted=Brazil (MISS)
- 1970: actual=Brazil, predicted=Brazil (OK)
- 1974: actual=Germany, predicted=Germany (OK)
- 1978: actual=Argentina, predicted=Italy (MISS)
- 1982: actual=Italy, predicted=Brazil (MISS)
- 1986: actual=Argentina, predicted=Germany (MISS)
- 1990: actual=Germany, predicted=Argentina (MISS)
- 1994: actual=Brazil, predicted=Argentina (MISS)
- 1998: actual=France, predicted=Argentina (MISS)
- 2002: actual=Brazil, predicted=South Korea (MISS)
- 2006: actual=Italy, predicted=Argentina (MISS)
- 2010: actual=Spain, predicted=Argentina (MISS)
- 2014: actual=Germany, predicted=Spain (MISS)
- 2018: actual=France, predicted=Germany (MISS)
- 2022: actual=Argentina, predicted=France (MISS)
