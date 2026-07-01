# FIFA World Cup 2026 Predictor

Machine learning system that predicts FIFA World Cup match outcomes and tournament winners. Combines Elo ratings, match form, head-to-head records, squad market values, and bookmaker odds into an ensemble of LightGBM and Dixon-Coles Poisson models.

## 2026 World Cup Prediction

> **Prediction updated: July 01, 2026** (9 R32 matches complete, 7 remaining)

| Place | Team | Probability |
|-------|------|-------------|
| 🥇 Champion | **Argentina** | 66.1% |
| 🥈 Runner-up | **Spain** | — |
| 🥉 Third | **Brazil** | 66.1% (vs France) |
| 4th | **France** | — |

### Group Stage Results (all 72 matches complete)

| Group | 1st | 2nd | 3rd (best 8) | 4th |
|-------|-----|-----|---------------|-----|
| A | Mexico (9pts) | South Africa (4pts) | Korea Republic (3pts) | Czech Republic (1pts) |
| B | Switzerland (7pts) | Canada (4pts) | Bosnia Herzegovina (4pts) | Qatar (1pts) |
| C | Brazil (7pts) | Morocco (7pts) | Scotland (3pts) | Haiti (0pts) |
| D | USA (6pts) | Australia (4pts) | Paraguay (4pts) | Turkey (3pts) |
| E | Germany (6pts) | Côte d'Ivoire (6pts) | Ecuador (4pts) | Curaçao (1pts) |
| F | Netherlands (7pts) | Japan (5pts) | Sweden (4pts) | Tunisia (0pts) |
| G | Belgium (5pts) | Egypt (5pts) | IR Iran (3pts) | New Zealand (1pts) |
| H | Spain (7pts) | Cape Verde (3pts) | Uruguay (2pts) | Saudi Arabia (2pts) |
| I | France (9pts) | Norway (6pts) | Senegal (3pts) | Iraq (0pts) |
| J | Argentina (9pts) | Austria (4pts) | Algeria (4pts) | Jordan (0pts) |
| K | Colombia (7pts) | Portugal (5pts) | DR Congo (4pts) | Uzbekistan (0pts) |
| L | England (7pts) | Croatia (6pts) | Ghana (4pts) | Panama (0pts) |

**Best 8 third-place teams:** DR Congo (K), Sweden (F), Ecuador (E), Ghana (L), Bosnia Herzegovina (B), Algeria (J), Paraguay (D), Senegal (I)
**Eliminated thirds:** IR Iran (G), Korea Republic (A), Scotland (C), Uruguay (H)

### Round of 32

| Match | Home | Away | Prediction | Confidence | Result |
|-------|------|------|------------|------------|--------|
| M73 | South Africa | Canada | **Canada** | 67.6% | ✅ Canada 1-0 |
| M74 | Germany | Paraguay | **Germany** | 66.1% | ❌ Paraguay wins on pens (1-1, 3-4) |
| M75 | Netherlands | Morocco | **Netherlands** | 66.1% | ❌ Morocco wins on pens (1-1, 2-3) |
| M76 | Brazil | Japan | **Brazil** | 74.0% | ✅ Brazil 2-1 |
| M77 | France | Sweden | **France** | 74.0% | ✅ France 3-0 |
| M78 | Côte d'Ivoire | Norway | **Norway** | 61.0% | ✅ Norway 2-1 |
| M79 | Mexico | Ecuador | **Ecuador** | 61.0% | ❌ Mexico 2-0 |
| M80 | England | DR Congo | **England** | 90.8% | ✅ England 2-1 |
| M81 | USA | Bosnia and Herzegovina | **USA** | 74.0% | *July 1* |
| M82 | Belgium | Senegal | **Belgium** | 67.6% | ✅ Belgium 3-2 (a.e.t.) |
| M83 | Portugal | Croatia | **Portugal** | 66.1% | *July 2* |
| M84 | Spain | Austria | **Spain** | 74.0% | *July 2* |
| M85 | Switzerland | Algeria | **Switzerland** | 61.0% | *July 2* |
| M86 | Argentina | Cape Verde | **Argentina** | 96.8% | *July 3* |
| M87 | Colombia | Ghana | **Colombia** | 95.1% | *July 3* |
| M88 | Australia | Egypt | **Australia** | 66.1% | *July 3* |

**R32 accuracy: 6/9 (67%)** — 3 upsets (2 penalty wins + Mexico over Ecuador)

### Round of 16

| Match | Home | Away | Prediction | Confidence |
|-------|------|------|------------|------------|
| M89 | Canada | Morocco | **Morocco** | 66.1% |
| M90 | Paraguay | France | **France** | 67.6% |
| M91 | Brazil | Norway | **Brazil** | 74.0% |
| M92 | Mexico | England | **England** | 66.1% |
| M93 | Portugal | Spain | **Spain** | 66.1% |
| M94 | USA | Belgium | **Belgium** | 66.1% |
| M95 | Argentina | Australia | **Argentina** | 74.0% |
| M96 | Switzerland | Colombia | **Colombia** | 67.6% |

### Quarterfinals

| Match | Home | Away | Prediction | Confidence |
|-------|------|------|------------|------------|
| M97 | Morocco | France | **France** | 66.1% |
| M98 | Spain | Belgium | **Spain** | 67.6% |
| M99 | Brazil | England | **Brazil** | 66.1% |
| M100 | Argentina | Colombia | **Argentina** | 66.1% |

### Semifinals

| Match | Home | Away | Prediction | Confidence |
|-------|------|------|------------|------------|
| M101 | France | Spain | **Spain** | 61.0% |
| M102 | Brazil | Argentina | **Argentina** | 61.0% |

### Third Place Match

- France vs Brazil → **Brazil** (66.1%)

### Final

- Spain vs Argentina → **Argentina** (66.1%)

### Path to the Final

- **Argentina:** Cape Verde (R32, 96.8%) → Australia (R16, 74.0%) → Colombia (QF, 66.1%) → Brazil (SF, 61.0%) → Spain (Final, 66.1%)
- **Spain:** Austria (R32, 74.0%) → Portugal (R16, 66.1%) → Belgium (QF, 67.6%) → France (SF, 61.0%) → Argentina (Final, 33.9%)
- **Brazil:** Japan (R32, 74.0%) ✅ → Norway (R16, 74.0%) → England (QF, 66.1%) → Argentina (SF, 39.0%) → France (3rd, 66.1%)
- **France:** Sweden (R32, 74.0%) ✅ → Paraguay (R16, 67.6%) → Morocco (QF, 66.1%) → Spain (SF, 39.0%) → Brazil (3rd)

## How It Works

### Model Architecture

The system uses a **blended ensemble** of two models:

1. **Dixon-Coles Poisson Goal Model** (75% weight for group stage, 50% for knockouts) — models goals scored as Poisson distributions with team-specific attack/defense strengths and home advantage. Naturally produces realistic draw probabilities. Blend weight tuned on chronological holdout.

2. **LightGBM Classifier** (25% weight for group stage, 50% for knockouts) — gradient-boosted trees trained on 60+ features per match, with isotonic calibration, draw class weighting (1.6×), and time-decay sample weights (half-life: 4 years). Hyperparameters tuned via Bayesian optimization (Hyperopt TPE, 100 trials). LightGBM selected over XGBoost after backtest comparison showed better log-loss calibration.

### Features (60+ total)

**Team Strength (7):**
- Elo rating (current), Elo difference, Elo sum
- Pre-tournament Elo, FIFA rank, football power index, tradition

**Form & Momentum (10):**
- Win/draw/loss rate (last 20 matches)
- Avg goals scored/conceded (last 5 and 10 matches)
- Elo momentum (change over last 5/10 matches, differential)
- Attack/defense trend (recent 3 vs 10 match baseline)

**Form Score & Streaks (7):**
- Form score (last 30 matches: win=+1, draw=0, loss=-1, summed)
- Current win streak, opponent win streak, unbeaten streak, loss streak
- Form score differential vs opponent

**Clean Sheets & Scoring (4):**
- Clean sheets in last 5/10 matches
- Scoring rate (fraction of last 5 with at least 1 goal)
- Conceding rate (fraction of last 5 with at least 1 goal conceded)

**Goal Difference (3):**
- Goal difference (last 5 and 10 matches)
- Goal difference differential vs opponent

**Head-to-Head (5):**
- H2H matches, win rate, draw rate (time-decayed, 15-year limit)
- Avg goals for/against in H2H

**Opponent-Weighted Form (2):**
- Form weighted by opponent Elo (beating Argentina > beating Haiti)
- Differential vs opponent

**Squad Market Value (4):**
- Team and opponent squad value (log-scaled EUR, from Transfermarkt)
- Value difference and ratio

**Bookmaker Odds (4):**
- Implied home/draw/away probabilities
- Overround (bookmaker margin)

**Draw Propensity (4):**
- Elo parity (1 / (1 + |elo_diff| / 100))
- Combined draw rate of both teams
- Expected total goals, low-scoring indicator

**Fatigue (3):**
- Matches in last 30/90 days, differential

**Context (7):**
- Tournament stage, neutral venue, home advantage
- Rest days, WC participations, titles, WC win rate

**Country Demographics (6):**
- GDP per capita, population, life expectancy
- Urbanization %, health spending % GDP, and others

### Training Pipeline

1. **Data:** ~50,000 international matches (1872–2026)
2. **State management:** Rolling Elo, form windows (20 matches), H2H records, all updated chronologically
3. **Feature engineering:** `shared.py` centralizes all feature computation — same code for training, backtest, and prediction
4. **Validation:** Chronological holdout (last 20% of matches), NOT random split
5. **Calibration:** Isotonic regression on holdout probabilities, WC-specific calibration for knockout matches
6. **Missing data:** LightGBM handles NaN natively (odds and squad values absent for older matches)

### Knockout Stage Handling

Knockout matches cannot end in a draw. The model:
1. Uses reduced Dixon-Coles weight (50% vs 75% in group stage) to avoid over-amplification
2. Renormalizes: P(home | no draw) = P(home) / (P(home) + P(away))
3. Applies WC-specific calibration buckets (not qualifier-dominated all-match calibration)
4. Prints raw 3-way probabilities alongside renormalized ones for transparency

## Performance

### Walk-Forward Backtest (11,909 matches, 2014–2026)

The primary validation uses all matches from 2014 onwards with walk-forward prediction:

| Metric | Value |
|--------|-------|
| **Accuracy** | 59.6% (7,095/11,909) |
| **Log-loss** | 0.8795 |
| **Brier score** | 0.1724 |
| **ECE** | 0.0215 |

**By tournament type:**
- Qualifiers: 64.4% (easiest — big vs small teams)
- Continental: 57.2%
- Friendlies: 56.2%
- World Cup: 54.7% (hardest — even matchups)

**Calibration (well-calibrated across all buckets):**
- 90-100% confidence: 94.5% actual accuracy (gap 0.2%)
- 80-90% confidence: 87.4% actual (gap 2.6%)
- 70-80% confidence: 77.3% actual (gap 2.7%)
- 60-70% confidence: 68.1% actual (gap 3.2%)

**Note:** WC knockout calibration is weaker than overall calibration. At 80-90% confidence on WC matches specifically, actual accuracy is ~57%. The model is aware of this and applies WC-specific calibration for knockout predictions.

### 2026 WC Backtest (81 matches: 72 group + 9 R32)

| Metric | Value |
|--------|-------|
| **Accuracy** | 63.0% (51/81) |
| **Log-loss** | 0.8737 |
| **Brier score** | 0.1747 |

### Model Evolution

| Version | Accuracy | Log-loss | Brier | Key Changes |
|---------|----------|----------|-------|-------------|
| V1 (baseline) | 62.9% | 0.9143 | 0.1850 | 38 features, XGBoost only |
| V2 (Opus review) | 61.3% | 0.9018 | 0.1825 | +14 features, bug fixes |
| V3 (ensemble) | **64.5%** | **0.8858** | **0.1791** | Dixon-Coles, odds, calibration |
| V4 (squad values) | 64.5% | 0.8897 | 0.1797 | +4 squad value features |
| V5 (walk-forward) | 59.6% | **0.8795** | **0.1724** | 11,909-match validation |
| V6 (LightGBM) | 63.0% | **0.8737** | **0.1747** | LightGBM, +15 features (form_score, streaks, clean_sheets, goal_diff), tradition dropped |

## Data Sources

| Data | Source | Records | Coverage |
|------|--------|---------|----------|
| Match results | [martj42/international_results](https://github.com/martj42/international_results) | 49,477 | 1872–2026 |
| Bookmaker odds | [football-data.co.uk](https://football-data.co.uk) | 2,144 | 2014–2026 |
| Squad values | [Transfermarkt](https://www.transfermarkt.com) via [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | 169 team-years | 2014–2026 |
| Country indicators | World Bank API | 83 variables | 1960–2024 |
| Elo ratings | Computed from match history | Continuous | 1872–2026 |

## Project Structure

```
├── shared.py                  # Centralized feature engineering, state, Elo, aliases
├── predict_2026.py            # Main 2026 WC prediction pipeline
├── backtest_2026_wc.py        # Walk-forward backtest on 2026 group matches
├── backtest_walkforward.py    # Walk-forward backtest on ALL 2014+ matches
├── backtest_walkforward_1998.py # Walk-forward backtest on ALL 1998+ matches
├── explain_match.py           # SHAP-based match explanation engine
├── monte_carlo_2026.py        # Monte Carlo tournament simulation (10K runs)
├── feature_selection.py       # Permutation importance feature selection
├── collect_data.py            # World Bank data collection
├── enrich_dataset.py          # FIFA rankings/Elo enrichment
├── sota_analysis.py           # SOTA econometric + ML analysis
├── backtest.py                # LOWCO backtest for country-level model
├── match_predictor.py         # Monte Carlo match simulator
├── incremental_predictor.py   # Walk-forward match predictor
├── analysis.py                # Exploratory analysis
│
├── data/
│   ├── results.csv            # 49,477 match results (1872-2026)
│   ├── betting_odds.csv       # 2,144 matches with bookmaker odds
│   ├── squad_values.csv       # 169 team-years of Transfermarkt values
│   ├── goalscorers.csv        # Goal-level data
│   ├── shootouts.csv          # Penalty shootout data
│   └── world_cup_predictors_dataset.csv  # Country-level features
│
└── output/
    ├── explain/               # SHAP waterfall/force plots per matchup
    └── sota/                  # SOTA analysis outputs
```

## Setup

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost shap scipy matplotlib lightgbm hyperopt

# Download raw data (not in repo — too large)
curl -o data/results.csv https://raw.githubusercontent.com/martj42/international_results/master/results.csv
curl -o data/goalscorers.csv https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv
curl -o data/shootouts.csv https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv

# Run predictions
python3 predict_2026.py              # Full bracket prediction (LightGBM, no tradition)
python3 predict_2026.py --model xgb  # Use XGBoost instead
python3 predict_2026.py --exclude-tradition False  # Include tradition features
python3 backtest_2026_wc.py          # Walk-forward backtest (2026 matches)
python3 backtest_2026_wc.py --compare # Compare XGBoost vs LightGBM variants
python3 backtest_walkforward.py      # Walk-forward backtest (all 2014+ matches)
python3 explain_match.py Brazil Japan --stage 1   # Match explanation with SHAP
python3 monte_carlo_2026.py          # 10K tournament simulations
python3 feature_selection.py         # Feature importance analysis
```

## Requirements

Python 3.11+ with: pandas, numpy, scikit-learn, xgboost, shap, scipy, matplotlib, lightgbm, hyperopt
