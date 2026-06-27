Build a comprehensive walk-forward backtest that validates on ALL available matches, not just 62 from the 2026 World Cup group stage.

## CONTEXT
The current backtest (backtest_2026_wc.py) only validates on 62 completed 2026 WC group matches. But we have 26,952 matches since 1998, or 11,919 since 2014 (with betting odds coverage). The model is being validated on 0.2% of available data.

## YOUR TASK

### 1. Read existing code first
Read: shared.py, predict_2026.py, backtest_2026_wc.py, experiment_common.py
Understand the full pipeline: how features are computed, how state is managed, how Dixon-Coles and XGBoost blend works.

### 2. Create backtest_walkforward.py
Build a walk-forward backtest that:

**Data scope:**
- ALL matches from 2014 onwards (when betting odds are available)
- That is ~11,919 matches
- Use betting_odds.csv (2,144 matches) and squad_values.csv (169 team-years) when available, NaN when not

**Walk-forward procedure:**
For each match in chronological order:
1. Build state from all prior matches (Elo, form, H2H, etc.)
2. Compute features for this match
3. Predict probabilities (home win / draw / away win)
4. Compare to actual result
5. Update state with actual result
6. Move to next match

**Training approach:**
- Train on all matches before a cutoff date (expanding window)
- Retrain every N matches (e.g., every 500 or 1000 matches) to save time
- Use the same model architecture: Dixon-Coles (75%) + XGBoost (25%) with isotonic calibration
- Use the same 52 features as the current model

**Metrics to compute:**
- Overall accuracy, log-loss, Brier score
- Per-year metrics (does the model improve over time?)
- Per-tournament-type metrics (WC, qualifier, friendly, continental)
- Per-confidence-bucket calibration (30-40%, 40-50%, etc.)
- Rolling 500-match accuracy (smoothed trend)
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)

**Output:**
- Print a summary table with per-tournament and per-year breakdowns
- Print calibration table
- Print rolling accuracy trend (first 500, second 500, etc.)
- Save detailed results to backtest_walkforward_results.csv (one row per match)
- Save summary to backtest_walkforward_summary.md

### 3. Also create backtest_walkforward_1998.py (lighter version)
Same as above but starting from 1998, WITHOUT betting odds (since those start in 2014). This gives us 26,952 matches. Use only the features that are always available (Elo, form, H2H, country demographics, WC history). Skip betting odds and squad value features (set to NaN).

### 4. Run both backtests
After creating the files, run them:
```bash
python3 backtest_walkforward.py
python3 backtest_walkforward_1998.py
```

Report the full results.

## IMPORTANT NOTES
- Import from shared.py for all feature engineering, state management, Elo, etc.
- The Dixon-Coles model needs to be retrained periodically (expanding window) since team strengths change over time
- For efficiency: retrain every 1000 matches, not every match. XGBoost and Dixon-Coles use the same features.
- Handle missing data: if betting odds or squad values are not available for a match, use NaN (XGBoost handles this natively)
- The walk-forward must be strictly chronological - NO future data leakage
- Use the same harmonize_country() and compute_match_features() from shared.py
- Print progress every 1000 matches so we can see it is working

## DATA FILES
- data/results.csv - 49,477 match results (1872-2026)
- data/betting_odds.csv - 2,144 matches with bookmaker implied probabilities (2014-2026)
- data/squad_values.csv - 169 team-years of Transfermarkt squad values (2014-2026)

## TARGET
Compare against the 62-match backtest: 64.5% accuracy, 0.8858 log-loss, 0.1791 Brier
The larger backtest will likely show lower accuracy (more matches = harder), but the calibration metrics will be much more reliable.

## DELIVERABLE
- backtest_walkforward.py and backtest_walkforward_1998.py
- backtest_walkforward_results.csv (per-match predictions)
- backtest_walkforward_summary.md (summary tables)
- Run both and report full results
