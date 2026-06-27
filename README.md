# FIFA World Cup 2026 Predictor

Machine learning system that predicts FIFA World Cup match outcomes and tournament winners. Combines Elo ratings, match form, head-to-head records, squad market values, and bookmaker odds into an ensemble of XGBoost and Dixon-Coles Poisson models.

## 2026 World Cup Prediction

> **Prediction updated: June 27, 2026**

| Place | Team | Probability |
|-------|------|-------------|
| 🥇 Champion | **Argentina** | 70.5% |
| 🥈 Runner-up | **Spain** | — |
| 🥉 Third | **Brazil** | 71.3% (vs France) |
| 4th | **France** | — |

**Corrected Round of 32 bracket:**
- Match 73: South Africa vs Canada
- Match 74: Germany vs Paraguay
- Match 75: Netherlands vs Morocco
- Match 76: Brazil vs Japan
- Match 77: France vs Sweden
- Match 78: Côte d'Ivoire vs Norway
- Match 79: Mexico vs Ecuador
- Match 80: England vs Austria
- Match 81: USA vs Bosnia and Herzegovina
- Match 82: Belgium vs Korea Republic
- Match 83: Portugal vs Croatia
- Match 84: Spain vs Algeria
- Match 85: Switzerland vs IR Iran
- Match 86: Argentina vs Cape Verde
- Match 87: Colombia vs Ghana
- Match 88: Australia vs Egypt

**Path to the final:**
- Argentina: Cape Verde ✅ → Australia/Egypt path → Colombia/Switzerland path → Brazil (SF, 51.7%) → Spain (Final, 70.5%)
- Spain: Algeria ✅ → Portugal/Croatia path → Belgium/USA path → France (SF, 55.6%) → Argentina (Final)
- Brazil: Japan (R32) → Norway (R16) → England (QF, 73.6%) → Argentina (SF, 48.3%) → France (3rd place)

## How It Works

### Model Architecture

The system uses a **blended ensemble** of two models:

1. **XGBoost Classifier** (25% weight) — gradient-boosted trees trained on 52 features per match, with isotonic calibration, draw class weighting (1.6×), and time-decay sample weights (half-life: 4 years).

2. **Dixon-Coles Poisson Goal Model** (75% weight) — models goals scored as Poisson distributions with team-specific attack/defense strengths and home advantage. Naturally produces realistic draw probabilities. Blend weight tuned on chronological holdout.

### Features (52 total)

**Team Strength (7):**
- Elo rating (current), Elo difference, Elo sum
- Pre-tournament Elo, FIFA rank, football power index, tradition

**Form & Momentum (10):**
- Win/draw/loss rate (last 20 matches)
- Avg goals scored/conceded (last 5 and 10 matches)
- Elo momentum (change over last 5/10 matches, differential)
- Attack/defense trend (recent 3 vs 10 match baseline)

**Head-to-Head (5):**
- H2H matches, win rate, draw rate
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
5. **Calibration:** Isotonic regression on holdout probabilities
6. **Missing data:** XGBoost handles NaN natively (odds and squad values absent for older matches)

### Knockout Stage Handling

Knockout matches cannot end in a draw. The model renormalizes probabilities:
- P(home | no draw) = P(home) / (P(home) + P(away))
- Applied to predictions, Elo baseline, and counterfactuals

## Performance

### Backtest on 2026 World Cup Group Stage (62 matches)

Walk-forward backtest: predict each match, compare to actual result, update state with real result, predict next.

| Metric | Value |
|--------|-------|
| **Accuracy** | 64.5% (40/62) |
| **Log-loss** | 0.8858 |
| **Brier score** | 0.1791 |

**Calibration:**
- 30-40% confidence: 83.3% actual accuracy
- 50-60% confidence: 60.0% actual accuracy
- 70-80% confidence: 70.0% actual accuracy

**Weakness:** Draws remain difficult — 9 of 22 draws were missed at ≥60% confidence. The Dixon-Coles Poisson model (75% of blend) helps significantly with draw estimation.

### Model Evolution

| Version | Accuracy | Log-loss | Brier | Key Changes |
|---------|----------|----------|-------|-------------|
| V1 (baseline) | 62.9% | 0.9143 | 0.1850 | 38 features, XGBoost only |
| V2 (Opus review) | 61.3% | 0.9018 | 0.1825 | +14 features, bug fixes |
| V3 (ensemble) | **64.5%** | **0.8858** | **0.1791** | Dixon-Coles, odds, calibration |
| V4 (squad values) | 64.5% | 0.8897 | 0.1797 | +4 squad value features |

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
├── output/
│   ├── explain/               # SHAP waterfall/force plots per matchup
│   ├── sota/                  # SOTA analysis outputs
│   └── ...
│
├── IMPROVEMENTS.md            # Code review + improvement roadmap
├── IMPROVEMENTS_V2.md         # V2 improvement results
├── REVIEW.md                  # Code review findings
├── REVIEW_EXPLAIN.md          # Explain module review
└── SQUAD_VALUES_RESULTS.md    # Squad value integration results
```

## Setup

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost shap scipy matplotlib

# Download raw data (not in repo — too large)
curl -o data/results.csv https://raw.githubusercontent.com/martj42/international_results/master/results.csv
curl -o data/goalscorers.csv https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv
curl -o data/shootouts.csv https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv

# Run predictions
python3 predict_2026.py              # Full bracket prediction
python3 backtest_2026_wc.py          # Walk-forward backtest
python3 explain_match.py Brazil Japan --stage 1   # Match explanation with SHAP
python3 monte_carlo_2026.py          # 10K tournament simulations
python3 feature_selection.py         # Feature importance analysis
```

## Requirements

Python 3.11+ with: pandas, numpy, scikit-learn, xgboost, shap, scipy, matplotlib
