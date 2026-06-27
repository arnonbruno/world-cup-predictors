You are tasked with exploring and implementing alternative ML approaches to improve a FIFA World Cup match prediction model. The current best model achieves 64.5% accuracy, 0.8858 log-loss, and 0.1791 Brier score on a 62-match walk-forward backtest of the 2026 World Cup group stage. You need to beat ALL three metrics.

## CURRENT MODEL STATE

The project is at: /var/mnt/DATA/Hermes/workspace/world-cup-predictors/

### Architecture
- **Ensemble**: Dixon-Coles Poisson goal model (75%) + XGBoost multiclass (25%)
- **52 features**: Elo (7), form/momentum (10), H2H (5), opponent-weighted form (2), squad market value (4), bookmaker odds (4), draw propensity (4), fatigue (3), context (7), country demographics (6)
- **Training**: ~50,000 international matches (1872-2026), chronological holdout validation
- **Calibration**: Isotonic regression on holdout, draw class weight 1.6x, time-decay weighting (half-life 4 years)

### Performance (62 group stage matches, walk-forward)
- Accuracy: 64.5% (40/62)
- Log-loss: 0.8858
- Brier: 0.1791

### Known Weaknesses
1. **Draw prediction**: 9/22 draws missed at >=60% confidence. All 11 original upsets were draws predicted as wins.
2. **High-confidence calibration**: 80-90% bucket has 50% actual accuracy (4 samples) - massively overconfident
3. **Small validation set**: 62 matches is tiny, results may not generalize

### Calibration Buckets
- 30-40% conf: 83.3% actual (underconfident)
- 40-50% conf: 60.0% actual (underconfident)
- 50-60% conf: 60.0% actual (slightly underconfident)
- 60-70% conf: 63.6% actual (well calibrated)
- 70-80% conf: 70.0% actual (well calibrated)
- 80-90% conf: 50.0% actual (OVERCONFIDENT)
- 90-100% conf: 100.0% actual (1 sample)

## YOUR MISSION

Read ALL existing code first (shared.py, predict_2026.py, backtest_2026_wc.py, explain_match.py, monte_carlo_2026.py, feature_selection.py). Understand the full pipeline before making changes.

Then explore and implement these alternative approaches. For each one, implement it, run the backtest, and report metrics. Keep what works, discard what does not.

### APPROACH 1: LightGBM + Bayesian Optimization (HIGH PRIORITY)
- Replace XGBoost with LightGBM (faster, often better on tabular data)
- Use Optuna for Bayesian hyperparameter optimization (50-100 trials)
- Tune: num_leaves, max_depth, learning_rate, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda
- Key: optimize for log-loss, not accuracy (better calibration)
- Keep the Dixon-Coles blend structure

### APPROACH 2: Neural Network (Tabular Deep Learning)
- Try a simple feedforward network on the 52 features
- Architecture: 52 -> 256 -> 128 -> 64 -> 3 (softmax)
- BatchNorm, Dropout (0.3), ReLU activations
- Train with cross-entropy loss + class weights for draws
- Use PyTorch or sklearn MLPClassifier
- Consider TabNet (attention-based tabular model) if available
- Compare against XGBoost/LightGBM

### APPROACH 3: Improved Calibration
- Platt scaling (logistic regression on holdout predictions)
- Temperature scaling (single parameter T, scale logits by 1/T)
- Isotonic regression (already used, but try on the ensemble output, not just XGBoost)
- Beta calibration (3-parameter family, good for probabilities)
- Measure Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)

### APPROACH 4: Better Ensemble
- Instead of fixed 75/25 Dixon-Coles/XGBoost blend:
  - Stacking: train a meta-learner (logistic regression) on out-of-fold predictions from each base model
  - Dynamic blending: weight models differently based on context (e.g., more Dixon-Coles for high-draw matches)
  - Add more base models: Poisson regression, Random Forest, Logistic Regression baseline
  - Use sklearn StackingClassifier or manual stacking

### APPROACH 5: Draw-Specific Model
- Train a separate binary classifier: "will this match be a draw?"
- Use draw-specific features: elo_parity, combined_draw_rate, expected_total_goals
- If P(draw) > threshold, predict draw; otherwise use the win/away model
- Or: train 3 separate binary models (home win vs rest, draw vs rest, away win vs rest)

### APPROACH 6: Feature Engineering Additions
- Elo x form interaction features
- Polynomial features on Elo_diff and squad_value_diff
- Target encoding of categorical features (stage, confederation)
- Binned Elo diff categories (huge_favorite, slight_favorite, coin_flip, underdog)
- Days since last WC appearance

## IMPLEMENTATION INSTRUCTIONS

1. Read ALL files first
2. For each approach, create a SEPARATE experiment file (e.g., experiment_lgbm.py, experiment_nn.py)
3. Each experiment file should:
   - Import from shared.py for feature engineering and state management
   - Load the same data as backtest_2026_wc.py
   - Run the same walk-forward backtest on 62 matches
   - Print accuracy, log-loss, Brier score
   - Save results to a comparison table
4. After testing all approaches, create a COMBINED model that uses the best parts of each
5. Run the final combined model through the full backtest
6. Write EXPERIMENTS.md with results table and analysis

## CONSTRAINTS
- Do NOT break existing code (predict_2026.py, explain_match.py, monte_carlo_2026.py must still work)
- All files must parse clean (python3 -m py_compile)
- The walk-forward backtest must be the same 62 matches for fair comparison
- Install any needed packages (lightgbm, optuna, torch, etc.)
- If a package install fails, skip that approach and note why
- Be honest about results - if something does not improve, say so

## DATA FILES AVAILABLE
- data/results.csv - 49,477 match results (1872-2026)
- data/betting_odds.csv - 2,144 matches with bookmaker implied probabilities
- data/squad_values.csv - 169 team-years of Transfermarkt squad values

## KEY SHARED.PY FUNCTIONS TO USE
- harmonize_country(name) - normalize team names
- compute_match_features(home, away, state, stage, neutral, is_home, match_date, odds, squad_values) - compute all 52 features
- apply_match_to_state(state, home, away, home_goals, away_goals, match_date) - update state after a match
- build_state(results_df, cutoff_date) - build initial state from historical data
- load_betting_odds() - load bookmaker odds
- load_squad_values() - load Transfermarkt values

## TARGET
Beat: 64.5% accuracy, 0.8858 log-loss, 0.1791 Brier score
If you can get to 67%+ accuracy or 0.85 log-loss, that would be a major improvement.

## DELIVERABLE
- EXPERIMENTS.md with results table comparing all approaches
- Best approach merged into the main pipeline (or kept as experiment files if too different)
- Final backtest numbers