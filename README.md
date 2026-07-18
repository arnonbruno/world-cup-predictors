# FIFA World Cup 2026 Predictor

Machine learning system that predicts FIFA World Cup match outcomes and tournament winners. Combines Elo ratings, match form, head-to-head records, squad market values, and bookmaker odds into an ensemble of LightGBM and Dixon-Coles Poisson models.

## 2026 World Cup Prediction

> **Prediction updated: July 18, 2026** (102 matches complete — 3rd place today, Final tomorrow)

### Two Prediction Methods

Every match is predicted using two independent approaches that run in parallel:

**1. Pipeline (LGBM + DC Blend)** — The production model. LightGBM gradient-boosted trees (75 features) are blended with the Dixon-Coles Poisson goal model (75% DC / 25% LGBM for group stage, 50/50 for knockouts). The blend is then isotonic-calibrated on a chronological holdout, renormalized for knockouts (draws removed), and adjusted with WC-specific calibration buckets. This is the primary prediction.

**2. MC Simulation (100K Monte Carlo)** — Samples 100,000 match outcomes from the fitted Dixon-Coles scoreline probability grid. Each simulation samples goals for both teams from the Poisson distribution parameterized by DC attack/defense strengths, applies the Dixon-Coles low-score correction (rho), and for knockouts resolves draws via extra time (30% increased scoring rate) and penalty shootouts (Elo-based win probability, not 50/50). Produces scoreline distributions, goal market probabilities (BTTS, over/under), and win probabilities.

**Why both?** The pipeline incorporates LGBM's pattern recognition (form, streaks, momentum) on top of the DC goal model, while the MC simulation gives a pure goal-level view from the DC model alone. When both agree, confidence is high. When they diverge, the gap reflects information the goal model can't see.

| Place | Team | Probability |
|-------|------|-------------|
| 🥇 Champion | **Argentina** | 66.1% |
| 🥈 Runner-up | **Spain** | — |
| 🥉 Third | **England** | 61.0% (vs France) |
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

| Match | Home | Away | Pipeline Pred | Pipeline % | MC % | Result |
|-------|------|------|---------------|-----------|------|--------|
| M73 | South Africa | Canada | **Canada** | 67.6% | 81.2% | ✅ Canada 1-0 |
| M74 | Brazil | Japan | **Brazil** | 74.0% | 81.7% | ✅ Brazil 2-1 |
| M75 | Germany | Paraguay | **Germany** | 66.1% | 53.7% | ❌ Paraguay wins on pens (1-1, 3-4) |
| M76 | Netherlands | Morocco | **Netherlands** | 66.1% | 51.7% | ❌ Morocco wins on pens (1-1, 2-3) |
| M77 | Côte d'Ivoire | Norway | **Norway** | 61.0% | 61.6% | ✅ Norway 2-1 |
| M78 | France | Sweden | **France** | 74.0% | 80.8% | ✅ France 3-0 |
| M79 | Mexico | Ecuador | **Ecuador** | 61.0% | 64.8% | ❌ Mexico 2-0 |
| M80 | England | DR Congo | **England** | 90.8% | 89.0% | ✅ England 2-1 |
| M81 | Belgium | Senegal | **Belgium** | 67.6% | 63.5% | ✅ Belgium 3-2 (a.e.t.) |
| M82 | USA | Bosnia and Herzegovina | **USA** | 74.0% | 83.2% | ✅ USA 2-0 |
| M83 | Portugal | Croatia | **Portugal** | 66.1% | 66.8% | ✅ Portugal 2-1 |
| M84 | Spain | Austria | **Spain** | 74.0% | 81.4% | ✅ Spain 3-0 |
| M85 | Switzerland | Algeria | **Switzerland** | 61.0% | 53.9% | ✅ Switzerland 2-0 |
| M86 | Argentina | Cape Verde | **Argentina** | 96.8% | 98.3% | ✅ Argentina 3-2 (a.e.t.) |
| M87 | Colombia | Ghana | **Colombia** | 95.1% | 95.8% | ✅ Colombia 1-0 |
| M88 | Australia | Egypt | **Australia** | 66.1% | 66.7% | ❌ Egypt wins on pens (1-1, 2-4) |

**R32 accuracy (16/16 complete): Pipeline 12/16 (75.0%)** — 4 misses: Germany-Paraguay (pens), Netherlands-Morocco (pens), Mexico-Ecuador (upset), Australia-Egypt (pens)

### Round of 16

| Match | Home | Away | Pipeline Pred | Pipeline % | MC % | Result |
|-------|------|------|---------------|-----------|------|--------|
| M89 | Canada | Morocco | **Morocco** | 66.1% | 60.0% | ✅ Morocco 3-0 |
| M90 | Paraguay | France | **France** | 67.6% | 62.8% | ✅ France 1-0 |
| M91 | Brazil | Norway | **Brazil** | 74.0% | 90.3% | ❌ Norway 2-1 |
| M92 | Mexico | England | **England** | 67.6% | 66.0% | ✅ England 3-2 |
| M93 | Portugal | Spain | **Spain** | 66.1% | 60.2% | ✅ Spain 1-0 |
| M94 | USA | Belgium | **Belgium** | 66.1% | 63.4% | ✅ Belgium 4-1 |
| M95 | Argentina | Egypt | **Argentina** | 93.4% | 93.8% | ✅ Argentina 3-2 |
| M96 | Switzerland | Colombia | **Colombia** | 67.6% | 82.5% | ❌ Switzerland wins on pens (0-0, 4-3) |

**R16 accuracy (8/8 complete): Pipeline 6/8 (75.0%)** — misses: Brazil-Norway (model predicted Brazil at 74%), Switzerland-Colombia (model predicted Colombia at 67.6%, Switzerland won on penalties)

### Quarterfinals

| Match | Home | Away | Pipeline Pred | Pipeline % | MC % | Result |
|-------|------|------|---------------|-----------|------|--------|
| M97 | Morocco | France | **France** | 66.1% | 58.8% | ✅ France 2-0 |
| M98 | Spain | Belgium | **Spain** | 67.6% | 66.6% | ✅ Spain 2-1 |
| M99 | Norway | England | **England** | 67.6% | 74.4% | ✅ England 2-1 |
| M100 | Argentina | Switzerland | **Argentina** | 74.0% | 91.0% | ✅ Argentina 3-1 |

**QF accuracy (4/4 complete): Pipeline 4/4 (100.0%)**

### Semifinals

| Match | Home | Away | Pipeline Pred | Pipeline % | MC % |
|-------|------|------|---------------|-----------|------|
| SF M101 | France | Spain | **Spain** | 61.0% | 59.5% | ✅ Spain 2-0 |
| SF M102 | England | Argentina | **Argentina** | 67.6% | 76.1% | ✅ Argentina 2-1 |

**SF accuracy (2/2 complete): Pipeline 2/2 (100.0%)** — Spain 2-0 France ✅, Argentina 2-1 England ✅

### Third Place Match

- France vs England → **England** (Pipeline 61.0% | MC 53.5%) — July 18, Miami Gardens

### Final

- Spain vs Argentina → **Argentina** (Pipeline 66.1% | MC 73.1%) — July 19, East Rutherford

### Path to the Final

- **Argentina:** Cape Verde (R32, 96.8%) ✅ → Egypt (R16, 93.4%) ✅ → Switzerland (QF, 74.0%) ✅ → England (SF, 67.6%) ✅ → Spain (Final, 66.1%)
- **Spain:** Austria (R32, 74.0%) ✅ → Portugal (R16, 66.1%) ✅ → Belgium (QF, 67.6%) ✅ → France (SF, 61.0%) ✅ → Argentina (Final, 33.9%)
- **England:** DR Congo (R32, 90.8%) ✅ → Mexico (R16, 67.6%) ✅ → Norway (QF, 67.6%) ✅ → Argentina (SF, 32.4%) ❌ → France (3rd, 61.0%)
- **France:** Sweden (R32, 74.0%) ✅ → Paraguay (R16, 67.6%) ✅ → Morocco (QF, 66.1%) ✅ → Spain (SF, 39.0%) ❌ → England (3rd, 39.0%)

## How It Works

### Model Architecture

The system uses a **blended ensemble** of two models:

1. **Dixon-Coles Poisson Goal Model** (75% weight for group stage, 50% for knockouts) — models goals scored as Poisson distributions with team-specific attack/defense strengths and home advantage. Naturally produces realistic draw probabilities. Blend weight tuned on chronological holdout.

2. **LightGBM Classifier** (25% weight for group stage, 50% for knockouts) — gradient-boosted trees trained on 75 features per match, with isotonic calibration, draw class weighting (1.6×), and time-decay sample weights (half-life: 4 years). Hyperparameters tuned via Bayesian optimization (Hyperopt TPE, 100 trials). LightGBM selected over XGBoost after backtest comparison showed better log-loss calibration.

### Monte Carlo Simulation

In addition to the pipeline blend, every match is simulated 100,000 times using the fitted Dixon-Coles model:

1. **Scoreline grid** — Build an 11×11 probability grid from Poisson PMFs for each team's expected goals (lambda), with Dixon-Coles low-score correction (rho) applied to 0-0, 0-1, 1-0, and 1-1 cells.
2. **Sampling** — Draw 100K scorelines from the grid.
3. **Knockout resolution** — For draws in knockout matches: sample extra-time goals (30% increased scoring rate for fatigue/open play), then resolve remaining draws via penalty shootout with Elo-based win probability (`1/(1+10^(-elo_diff/400))`).
4. **Output** — Win probabilities (90-min and knockout), scoreline distribution, average goals, BTTS rate, over/under 2.5.

The MC simulation is a pure goal-model view — it does not incorporate LGBM features (form, streaks, momentum). When it diverges from the pipeline, the gap reflects information the goal model can't see.

### Features (75 total)

**Team Strength (7):** Elo rating (current), Elo difference, Elo sum, Pre-tournament Elo, FIFA rank, football power index, tradition

**Form & Momentum (10):** Win/draw/loss rate (last 20), Avg goals scored/conceded (last 5 and 10), Elo momentum (5/10, differential), Attack/defense trend

**Form Score & Streaks (7):** Form score (last 30: win=+1, draw=0, loss=-1), Win streak, opponent win streak, unbeaten streak, loss streak, form score differential

**Clean Sheets & Scoring (4):** Clean sheets (5/10), scoring rate, conceding rate

**Goal Difference (3):** Goal difference (5/10), differential vs opponent

**Head-to-Head (5):** H2H matches, win rate, draw rate (time-decayed, 15yr), avg goals for/against

**Opponent-Weighted Form (2):** Form weighted by opponent Elo, differential

**Squad Market Value (4):** Team and opponent squad value (log EUR), difference and ratio

**Bookmaker Odds (4):** Implied home/draw/away probabilities, overround

**Draw Propensity (4):** Elo parity, combined draw rate, expected total goals, low-scoring indicator

**Fatigue (3):** Matches in last 30/90 days, differential

**Context (7):** Tournament stage, neutral venue, home advantage, rest days, WC participations, titles, WC win rate

**Country Demographics (6):** GDP per capita, population, life expectancy, urbanization, health spending

### Training Pipeline

1. **Data:** ~50,000 international matches (1872–2026)
2. **State management:** Rolling Elo, form windows (20 matches), H2H records, all updated chronologically
3. **Feature engineering:** `shared.py` centralizes all feature computation — same code for training, backtest, and prediction
4. **Validation:** Chronological holdout (last 20% of matches), NOT random split
5. **Calibration:** Isotonic regression on holdout probabilities, WC-specific calibration for knockout matches
6. **Missing data:** LightGBM handles NaN natively (odds and squad values absent for older matches)

### Knockout Stage Handling

Knockout matches cannot end in a draw. The pipeline:
1. Uses reduced Dixon-Coles weight (50% vs 75% in group stage) to avoid over-amplification
2. Renormalizes: P(home | no draw) = P(home) / (P(home) + P(away))
3. Applies WC-specific calibration buckets (not qualifier-dominated all-match calibration)
4. Prints raw 3-way probabilities alongside renormalized ones for transparency

The MC simulation handles knockouts differently: draws are played out via extra time (30% increased scoring) and penalty shootouts (Elo-weighted), producing a more realistic knockout probability that accounts for the draw→penalty pathway.

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

### 2026 WC Backtest (102 matches: 72 group + 30 KO)

| Metric | Value |
|--------|-------|
| **Accuracy** | 67.6% (69/102) |
| **Log-loss** | 0.7580 |
| **Brier score** | 0.1538 |

**Per-stage:** Group 45/72 (62.5%, LL 0.8806) | R32+R16 18/24 (75.0%, LL 0.4945) | QF+SF 6/6 (100.0%)

### Model Evolution

| Version | Accuracy | Log-loss | Brier | Key Changes |
|---------|----------|----------|-------|-------------|
| V1 (baseline) | 62.9% | 0.9143 | 0.1850 | 38 features, XGBoost only |
| V2 (Opus review) | 61.3% | 0.9018 | 0.1825 | +14 features, bug fixes |
| V3 (ensemble) | **64.5%** | **0.8858** | **0.1791** | Dixon-Coles, odds, calibration |
| V4 (squad values) | 64.5% | 0.8897 | 0.1797 | +4 squad value features |
| V5 (walk-forward) | 59.6% | **0.8795** | **0.1724** | 11,909-match validation |
| V6 (LightGBM) | 63.0% | **0.8328** | **0.1682** | LightGBM, +15 features, tradition dropped, R32 stage detection + neutral flag fix |
| V7 (MC Sim) | **67.6%** | **0.7580** | **0.1538** | Added Dixon-Coles Monte Carlo simulation (100K per match), fixed R16/QF/SF completed-match detection, QF 4/4, SF 2/2 |

## Data Sources

| Data | Source | Records | Coverage |
|------|--------|---------|----------|
| Match results | [martj42/international_results](https://github.com/martj42/international_results) | 49,486 | 1872–2026 |
| Bookmaker odds | [football-data.co.uk](https://football-data.co.uk) | 2,144 | 2014–2026 |
| Squad values | [Transfermarkt](https://www.transfermarkt.com) via [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | 169 team-years | 2014–2026 |
| Country indicators | World Bank API | 83 variables | 1960–2024 |
| Elo ratings | Computed from match history | Continuous | 1872–2026 |

## Project Structure

```
├── shared.py                  # Centralized feature engineering, state, Elo, DC model, MC simulation
├── predict_2026.py            # Main 2026 WC prediction pipeline (pipeline + MC output)
├── backtest_2026_wc.py        # Walk-forward backtest on 2026 WC matches
├── backtest_walkforward.py    # Walk-forward backtest on ALL 2014+ matches
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
│   ├── results.csv            # 49,489 match results (1872-2026)
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

# Run predictions (outputs both Pipeline and MC Sim for every match)
python3 predict_2026.py              # Full bracket prediction (LightGBM, no tradition)
python3 predict_2026.py --model xgb  # Use XGBoost instead
python3 predict_2026.py --debug      # Show LGBM/DC component probabilities
python3 backtest_2026_wc.py          # Walk-forward backtest (2026 matches)
python3 backtest_2026_wc.py --compare # Compare XGBoost vs LightGBM variants
python3 backtest_walkforward.py      # Walk-forward backtest (all 2014+ matches)
python3 explain_match.py Brazil Japan --stage 1   # Match explanation with SHAP
python3 monte_carlo_2026.py          # 10K tournament simulations
python3 feature_selection.py         # Feature importance analysis
```

## Requirements

Python 3.11+ with: pandas, numpy, scikit-learn, xgboost, shap, scipy, matplotlib, lightgbm, hyperopt
