# Incremental Match Predictor — Specification

## Philosophy
Instead of Monte Carlo bracket simulation, predict each World Cup match **incrementally**:
1. For each WC year, use ALL data available before that WC starts to train a model
2. Walk through the actual WC match schedule chronologically
3. For each match: compute features from pre-match state → predict → observe actual result → update state
4. After all matches: team with most predicted wins OR team predicted to win the final = predicted champion

This avoids bracket simulation noise and directly tests whether historical patterns + country context can predict outcomes.

## Data Sources (Already Downloaded)
- `data/results.csv`: 49,477 international matches (1872–2026), columns: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
- `data/goalscorers.csv`: 47,783 goal records
- `data/shootouts.csv`: 677 penalty shootouts
- `data/world_cup_predictors_dataset.csv`: 83 country-level features per WC year (Elo, FIFA ranking, GDP, population, healthcare, football tradition, etc.)

## Country Name Harmonization
Apply BEFORE any processing. Map historical names to modern names:
- "West Germany" → "Germany" (for all years)
- "East Germany" → "East Germany" (keep separate, rarely appears in WCs)
- "Soviet Union" / "USSR" → "Russia"
- "Yugoslavia" → "Serbia"
- "Czechoslovakia" → "Czech Republic"
- "Zaire" → "DR Congo"
- "Burma" → "Myanmar"
- "Ivory Coast" → "Côte d'Ivoire"
- "South Korea" → "Korea Republic" (match FIFA naming used in results.csv)
- "North Korea" → "Korea DPR"

## Match Schedule Extraction
From `results.csv`, filter where `tournament == "FIFA World Cup"` for each year. These are the ACTUAL matches in order (group stage through final). No bracket simulation needed — we walk through real matches.

For WCs before 1930, skip. WC years: 1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022

## Feature Engineering (All Strictly Pre-Match)

### Match-Level Dynamic Features (computed from results.csv history)
For each team in a match, compute from ALL matches strictly before this match date:

1. **Elo rating** — start each team at 1500, update after every match using standard Elo formula (K-factor=32, adjust for home advantage +50, goal margin multiplier)
2. **Form** — W/D/L record over last 10 matches (3 features: win_rate, draw_rate, loss_rate)
3. **Rolling goals** — avg goals scored and conceded over last 5 and last 10 matches (4 features)
4. **Rest days** — days since each team's last match
5. **H2H record** — historical wins/draws/losses between these two specific teams, avg goals for/against (5 features)
6. **WC experience** — number of prior WC participations, prior WC titles, years since last WC title, years since last WC final, prior WC win rate
7. **Tournament progress** — matches played so far in THIS WC, goals scored/conceded so far, points so far, current group standing if group stage

### Country-Level Features (from world_cup_predictors_dataset.csv)
For each team, load their row from the dataset for the CLOSEST PRECEDING WC year. This gives:
- Elo rating (pre-WC), FIFA ranking, football_power_index
- GDP per capita, population, urban population %
- Life expectancy, health expenditure, education spending
- Football tradition metrics (league quality, historical performance)
- All 83 variables available

### Context Features
- **Stage**: group / round_of_16 / quarterfinal / semifinal / final (encode as ordinal 0-4)
- **Neutral**: is the match on neutral ground? (1/0)
- **Elo difference**: team_A_elo - team_B_elo
- **Elo sum**: team_A_elo + team_B_elo (match "level")
- **Home advantage**: is team_A the host nation? (1/0)
- **Form difference**: team_A form - team_B form
- **H2H advantage**: team_A win% vs team_B in H2H

## Model Architecture

### Step 1: Train Match Outcome Model
- **Training data**: ALL international matches before the WC year (expanding window)
- **Labels**: 3-class — `home_win` / `draw` / `away_win` (based on 90min + ET result, NOT penalties)
- **Model**: XGBoost multiclass (or LightGBM if available)
- **Validation**: time-based split — train on matches before year-4, validate on matches year-4 to current
- **Features**: all dynamic features above (no country-level features in this model to keep it general)

### Step 2: Train Context-Enhanced Model (meta-model)
- Same as Step 1 but ADDS country-level features
- This model has richer context but only works for WC matches (country data is WC-specific)

### Step 3: Incremental WC Prediction Loop
```
For each WC year (1930-2022):
    1. Load match schedule from results.csv (tournament == "FIFA World Cup", year)
    2. Build training set: all international matches before WC start date
    3. Train model (or use pre-trained expanding window model)
    4. Initialize match state: Elo ratings at WC start, empty form/H2H/tournament state
    5. For each match in chronological order:
        a. Compute features for both teams from current state
        b. Predict: P(home_win), P(draw), P(away_win)
        c. Record prediction vs actual result
        d. Update state:
           - Update Elo ratings
           - Add match to form rolling window
           - Update H2H records
           - Update tournament progress (goals, points, stage)
    6. After all matches:
        a. Count predicted match wins per team → "predicted_winners" ranking
        b. Check who was predicted to win the final → "final_predictor"
        c. Compare both to actual winner
```

### Winner Prediction Strategy
Two strategies to determine predicted champion:
1. **Most predicted wins**: team with highest count of matches where they were predicted to win
2. **Final winner**: team predicted to win the final match
3. **Aggregate probability**: sum of win probabilities across all matches a team played

Report all three strategies.

## Evaluation

### Per-WC Metrics
- Predicted champion (3 strategies) vs actual winner
- Was actual winner in top-3 by aggregate probability?
- Was actual winner in top-5?
- Match-level accuracy for this WC
- Match-level log-loss

### Aggregate Metrics
- Exact winner accuracy: X/22 for each strategy
- Top-3 accuracy: X/22
- Top-5 accuracy: X/22
- Overall match accuracy across all WC matches
- Overall Brier score / log-loss
- ROC AUC (winner vs non-winner, using aggregate probability)

### Comparison
Compare to previous models:
- Country-level model: 40.9% exact, 63.6% top-3
- Monte Carlo simulator: 31.8% exact, 50% top-3

## Output Files
- `output/incremental/backtest_results.csv` — per-WC: predicted champion, actual winner, correct?, top3?, top5?, match_acc
- `output/incremental/match_predictions.csv` — every WC match with features, predicted probs, actual result
- `output/incremental/feature_importance.png` — XGBoost feature importance (top 30)
- `output/incremental/prediction_heatmap.png` — heatmap of predicted vs actual winners across WCs
- `output/incremental/summary.md` — full markdown report

## Implementation Notes
- Single file: `incremental_predictor.py`, self-contained
- Install missing packages inline (xgboost, etc.)
- All data in `data/` — no downloads
- Handle sparse early WCs (1930-1950) gracefully
- Penalties in shootouts: regular time result is draw, winner gets win in shootout (track separately)
- Code should be well-commented and modular
- Print progress for each WC year as it processes
- Save all outputs before exit

## CRITICAL: Strict Temporal Ordering
The #1 rule: NO FUTURE DATA leaks into features. Every feature must be computed from data available BEFORE the match being predicted. This includes:
- Elo ratings updated only from prior matches
- Form from last N matches BEFORE this one
- H2H from all prior meetings
- Country-level features from the PRECEDING WC year only
- Tournament progress from matches already played in this WC

Violating this invalidates the entire analysis.