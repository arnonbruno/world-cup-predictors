# Data Inventory for Model Improvement

## Available External Data

### 1. Betting Odds (NEW) — `data/betting_odds.csv`
- **2,144 matches** from 2014-06-12 to 2026-06-24
- Columns: date, home_team, away_team, competition, home_score, away_score, avg_home_odds, avg_draw_odds, avg_away_odds, implied_home_prob, implied_draw_prob, implied_away_prob, source
- Competitions: WC qualifiers (1,748), World Cups (240), Euros (51), Copa América (32), etc.
- **Coverage:** 198 out of 264 WC matches 2014+ have odds
- Source: football-data.co.uk

### 2. FIFA Rankings — NOT AVAILABLE (use Elo as proxy)
- Our Elo ratings already capture team strength
- Can compute Elo deltas as ranking momentum

### 3. Squad Market Values — NOT AVAILABLE
- Would need Transfermarkt scraping
- Skip for now, focus on what we have

## Current Model Stats
- 38 → 52 features (after Claude Opus added 14)
- Backtest: 61.3% acc, log-loss 0.9018, Brier 0.1825
- Main weakness: draws (13/24 misses were draws)

## Current Files
- `shared.py` — feature engineering, state management
- `predict_2026.py` — main prediction pipeline
- `backtest_2026_wc.py` — walk-forward backtest
- `explain_match.py` — SHAP explanations
- `monte_carlo_2026.py` — tournament simulation
