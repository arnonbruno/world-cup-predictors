# World Cup Match-Level Predictor — Implementation Spec

## Goal
Build a match-level prediction system that:
1. Collects all international football match results ever played (with stats)
2. For each World Cup, uses pre-tournament features to predict each match outcome
3. Simulates the full tournament bracket to predict the winner
4. Combines match-level predictions with the existing country-level dataset

## Why This Is Better
The current model treats each (country, WC_year) independently. But World Cups are won through a PATH of matches — a team must win 7 matches. Predicting match-by-match captures:
- Group stage dynamics (who advances)
- Knockout bracket luck (who you face)
- Player-level quality (individual match contributions)
- Tactical evolution over time
- Head-to-head records

## Data Sources to Collect

### 1. All International Match Results
Source: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
File: `results.csv` — ~47,000 matches from 1872-2024
Columns: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

Download this CSV from Kaggle (it's public). If Kaggle API unavailable, scrape from football-data.co.uk or use the GitHub mirror:
https://raw.githubusercontent.com/martj42/international_results/master/results.csv

### 2. World Cup Match Details (goals, scorers, cards)
Source: FIFA official or Wikipedia scraping
We need: group_stage flag, knockout_round, venue, attendance, referee

### 3. Player Market Values (proxy for squad quality)
Source: Transfermarkt or Kaggle
https://www.kaggle.com/datasets/davidcariboo/player-scores

### 4. Historical Elo Ratings (per match)
Source: eloratings.net — has match-by-match Elo updates
https://eloratings.net/

## Implementation Plan

### Step 1: Download & Build Match Dataset
- Download all international results (results.csv from Kaggle/GitHub)
- For each match, compute features:
  - Home/away Elo at match time
  - Home/away FIFA rank at match time
  - Home/away recent form (W/D/L in last 10 matches)
  - Home/away goals scored/conceded in last 10 matches
  - Head-to-head historical record
  - Tournament type indicator (WC, qualifier, friendly, continental)
  - Days since last match for each team
  - Average squad age (if available)
  - Average squad market value (if available)

### Step 2: Build Match Prediction Model
- Target: home win / draw / away win (3-class) or home_win (binary)
- Features: all the above, plus existing country-level features from our dataset
- Model: XGBoost or LightGBM (best for tabular data with mixed types)
- Validation: Time-based splits (train on matches before year X, test on year X+)
- Also try Elo-based baseline (just Elo difference predicts ~60% of outcomes)

### Step 3: World Cup Tournament Simulator
For each WC year:
1. Get the group stage draw (which teams in which groups)
2. Predict each group match outcome using the match model
3. Determine group standings (points, GD, GF)
4. Predict each knockout match (R16, QF, SF, Final)
5. The team that wins the final = predicted champion
6. Run 1000 Monte Carlo simulations (sample from predicted probabilities) to get win probabilities

### Step 4: Combine with Existing Model
- The match-level model produces a "tournament win probability" for each team
- Combine this with the existing country-level features (GDP, population, etc.)
- Train a meta-model that uses both signals

### Step 5: Backtest
- For each WC year (1930-2022), run the full tournament simulation
- Report: exact winner accuracy, top-3, top-5, AUC
- Compare with the existing country-level model

## Technical Requirements
- Python script: `match_predictor.py`
- Output: `output/match_predictor/` with results, figures, summary
- Use pandas, numpy, sklearn, xgboost
- Handle missing data gracefully (early WCs have less data)
- All random states = 42
- Save the match dataset as CSV for reuse

## Key Data to Download RIGHT NOW

### URL 1: International match results
```
curl -L -o /var/mnt/DATA/Hermes/workspace/world-cup-predictors/data/results.csv \
  "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
```

### URL 2: World Cup match details (goalscorers, venues, etc.)
```
curl -L -o /var/mnt/DATA/Hermes/workspace/world-cup-predictors/data/wc_goals.csv \
  "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
```

### URL 3: Shootouts data
```
curl -L -o /var/mnt/DATA/Hermes/workspace/world-cup-predictors/data/shootouts.csv \
  "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
```

## File Structure
Write `match_predictor.py` at `/var/mnt/DATA/Hermes/workspace/world-cup-predictors/match_predictor.py`

The script should:
1. Download data if not present
2. Build match-level features
3. Train match prediction model
4. Simulate all WCs
5. Print backtest results
6. Save outputs to `output/match_predictor/`

## Important Constraints
- NO data leakage: only use information available BEFORE each match
- Time-based train/test split: train on all matches before the WC, test on WC matches
- The match model must work for matches between ANY two countries, not just WC participants
- Handle country name changes (West Germany → Germany, etc.)
- For Monte Carlo simulation: sample from predicted probabilities, don't just take argmax