# FIFA World Cup 2026 Predictor

Machine learning system that predicts FIFA World Cup match outcomes and tournament winners. Combines Elo ratings, match form, head-to-head records, squad market values, and bookmaker odds into an ensemble of XGBoost and Dixon-Coles Poisson models.

## 2026 World Cup Prediction

> **Prediction updated: June 27, 2026**

| Place | Team | Probability |
|-------|------|-------------|
| 🥇 Champion | **Argentina** | 66.1% |
| 🥈 Runner-up | **Spain** | — |
| 🥉 Third | **Brazil** | 66.1% (vs France) |
| 4th | **France** | — |

**Round of 32 bracket (confirmed vs official FIFA bracket):**
- M73: South Africa vs Canada → Canada (67.6%)
- M74: Germany vs Paraguay → Germany (66.1%)
- M75: Netherlands vs Morocco → Netherlands (66.1%)
- M76: Brazil vs Japan → Brazil (74.0%)
- M77: France vs Sweden → France (74.0%)
- M78: Côte d'Ivoire vs Norway → Norway (61.0%)
- M79: Mexico vs Ecuador → Ecuador (61.0%)
- M80: England vs Austria → England (67.6%)
- M81: USA vs Bosnia and Herzegovina → USA (67.6%)
- M82: Belgium vs Korea Republic → Belgium (67.6%)
- M83: Portugal vs Croatia → Portugal (66.1%)
- M84: Spain vs Algeria → Spain (74.0%)
- M85: Switzerland vs IR Iran → Switzerland (61.0%)
- M86: Argentina vs Cape Verde → Argentina (96.4%)
- M87: Colombia vs Ghana → Colombia (91.3%)
- M88: Australia vs Egypt → Australia (66.1%)

**Path to the final:**
- Argentina: Cape Verde ✅ → Australia → Colombia → Brazil (SF, 61.0%) → Spain (Final, 66.1%)
- Spain: Algeria ✅ → Portugal/Croatia → Belgium/USA → France (SF, 55.6%) → Argentina (Final)
- Brazil: Japan (R32, 74.0%) → Norway (R16) → England (QF, 66.1%) → Argentina (SF, 39.0%) → France (3rd, 66.1%)

## How It Works

### Model Architecture

The system uses a **blended ensemble** of two models:

1. **Dixon-Coles Poisson Goal Model** (75% weight for group stage, 50% for knockouts) — models goals scored as Poisson distributions with team-specific attack/defense strengths and home advantage. Naturally produces realistic draw probabilities. Blend weight tuned on chronological holdout.

2. **XGBoost Classifier** (25% weight for group stage, 50% for knockouts) — gradient-boosted trees trained on 52+ features per match, with isotonic calibration, draw class weighting (1.6×), and time-decay sample weights (half-life: 4 years).

### Features (52+ total)

**Team Strength (7):**
- Elo rating (current), Elo difference, Elo sum
- Pre-tournament Elo, FIFA rank, football power index, tradition

**Form & Momentum (10):**
- Win/draw/loss rate (last 20 matches)
- Avg goals scored/conceded (last 5 and 10 matches)
- Elo momentum (change over last 5/10 matches, differential)
- Attack/defense trend (recent 3 vs 10 match baseline)

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
6. **Missing data:** XGBoost handles NaN natively (odds and squad values absent for older matches)

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

### 2026 WC Group Stage Backtest (62 matches)

| Metric | Value |
|--------|-------|
| **Accuracy** | 64.5% (40/62) |
| **Log-loss** | 0.8957 |
| **Brier score** | 0.1808 |

### Model Evolution

| Version | Accuracy | Log-loss | Brier | Key Changes |
|---------|----------|----------|-------|-------------|
| V1 (baseline) | 62.9% | 0.9143 | 0.1850 | 38 features, XGBoost only |
| V2 (Opus review) | 61.3% | 0.9018 | 0.1825 | +14 features, bug fixes |
| V3 (ensemble) | **64.5%** | **0.8858** | **0.1791** | Dixon-Coles, odds, calibration |
| V4 (squad values) | 64.5% | 0.8897 | 0.1797 | +4 squad value features |
| V5 (walk-forward) | 59.6% | **0.8795** | **0.1724** | 11,909-match validation |

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
pip install pandas numpy scikit-learn xgboost shap scipy matplotlib lightgbm optuna

# Download raw data (not in repo — too large)
curl -o data/results.csv https://raw.githubusercontent.com/martj42/international_results/master/results.csv
curl -o data/goalscorers.csv https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv
curl -o data/shootouts.csv https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv

# Run predictions
python3 predict_2026.py              # Full bracket prediction
python3 backtest_2026_wc.py          # Walk-forward backtest (2026 matches)
python3 backtest_walkforward.py      # Walk-forward backtest (all 2014+ matches)
python3 explain_match.py Brazil Japan --stage 1   # Match explanation with SHAP
python3 monte_carlo_2026.py          # 10K tournament simulations
python3 feature_selection.py         # Feature importance analysis
```

## Requirements

Python 3.11+ with: pandas, numpy, scikit-learn, xgboost, shap, scipy, matplotlib, lightgbm, optuna
