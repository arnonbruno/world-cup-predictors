# FIFA World Cup Winner Predictors

Econometric and machine learning analysis of what predicts FIFA World Cup champions. Combines football-specific variables (Elo ratings, form, H2H, tournament history) with 83 country-level indicators (GDP, population, healthcare, education, urbanization, etc.) from the World Bank.

## Approach

Three progressively sophisticated models:

### 1. Country-Level Model (`sota_analysis.py`)
Static features per country per World Cup year. Leave-One-World-Cout-Out (LOWCO) validation.
- Regularized Logit AUC: 0.966
- XGBoost AUC: 0.941
- Backtest: 9/22 exact winner (40.9%), 14/22 top-3 (63.6%)

### 2. Monte Carlo Match Simulator (`match_predictor.py`)
XGBoost match outcome model + 1000 bracket simulations per World Cup.
- Match accuracy: 59.4%
- Backtest: 7/22 exact winner (31.8%), 11/22 top-3 (50.0%)

### 3. Incremental Match Predictor (`incremental_predictor.py`)
Walks through actual WC matches chronologically. For each match: compute features from current state, predict, observe result, update state (Elo, form, H2H, tournament progress). No simulation.
- Match accuracy: 54.0%
- **Backtest: 19/22 exact winner (86.4%) via final-winner strategy**
- Top-3: 20/22 (90.9%), Top-5: 21/22 (95.5%)
- Winner AUC: 0.934

## Key Findings

- Football-specific variables dominate: Elo rating, historical titles, tournament experience
- Economic variables (GDP per capita) are nearly useless for predicting winners
- Hosting provides a real but not strictly causal boost (~+5-14%)
- The incremental approach dramatically outperforms static models and simulations
- The "predict the final winner" strategy (86.4%) crushes "most wins" (54.5%)

## Data Sources

- **Match results**: [martj42/international_results](https://github.com/martj42/international_results) (49,477 matches, 1872-2026)
- **Country indicators**: World Bank API (GDP, population, health, education, 83 variables)
- **Football context**: Elo ratings computed from match history, FIFA rankings

## Project Structure

```
├── collect_data.py          # World Bank data collection + dataset builder
├── enrich_dataset.py        # FIFA rankings/Elo enrichment
├── analysis.py              # Initial ML analysis
├── sota_analysis.py         # SOTA econometric + ML analysis (cursor-agent)
├── backtest.py              # LOWCO backtest for country-level model
├── match_predictor.py       # Monte Carlo match simulator (cursor-agent)
├── incremental_predictor.py # Incremental match predictor (cursor-agent)
├── ANALYSIS_SPEC.md         # Spec for SOTA analysis
├── MATCH_SPEC.md            # Spec for match predictor
├── INCREMENTAL_SPEC.md      # Spec for incremental predictor
├── data/
│   └── world_cup_predictors_dataset.csv  # 467 rows, 83 features
└── output/
    ├── sota/                # SOTA analysis outputs + figures
    ├── match_predictor/     # Monte Carlo simulator outputs
    └── incremental/         # Incremental predictor outputs
```

## Setup

Raw data files are excluded from the repo (too large). To regenerate:

```bash
# Download match data
curl -o data/results.csv https://raw.githubusercontent.com/martj42/international_results/master/results.csv
curl -o data/goalscorers.csv https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv
curl -o data/shootouts.csv https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv

# Collect World Bank data + build dataset
python3 collect_data.py
python3 enrich_dataset.py

# Run analyses
python3 sota_analysis.py
python3 match_predictor.py
python3 incremental_predictor.py
```

## Requirements

Python 3.11+ with: pandas, numpy, scikit-learn, xgboost, shap, statsmodels, matplotlib, wbgapi, lxml
