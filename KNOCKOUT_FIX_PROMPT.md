Fix the knockout prediction pipeline and re-evaluate Brazil 2026 World Cup chances with honest, calibrated probabilities.

## ISSUES TO FIX

Read KNOCKOUT_REVIEW.md first for the full analysis. The core problems:

### Issue 1: Dixon-Coles over-amplification in knockouts
The 75% Dixon-Coles / 25% XGBoost blend inflates knockout probabilities. XGBoost alone says 63.4% for Brazil vs Japan, but the blend pushes it to 83.2%. The Dixon-Coles model is given too much weight for knockout matches where there is no draw.

Fix: For knockout matches, reduce Dixon-Coles weight. The blend alpha should be different for knockouts vs group stage. Suggest: alpha=0.50 for knockouts (50/50 blend) instead of 0.75.

### Issue 2: Stale country features for 2026 teams
Norway country features are from 1998. GDP, population, FIFA rank, football_power_index are 28 years old. This deflates Norway unfairly.

Fix: In shared.py, when looking up country features for a team, if the latest available year is more than 8 years old, use the most recent available year but flag it. Better yet: for 2026 WC participants, we should have current country data. Add a fallback that uses 2024 or 2023 data even if it is not a WC year. The World Bank data has recent values - check data/world_cup_predictors_dataset.csv for 2022 or recent rows.

### Issue 3: No separate knockout calibration
The model calibrates on ALL matches (mostly qualifiers where big teams crush small teams). WC knockout calibration is much worse - 57% accuracy at 80-90% confidence.

Fix: Build a separate calibration for WC matches only. Use the backtest_walkforward_results.csv to extract WC match predictions and compute WC-specific calibration buckets. Apply WC calibration when reporting knockout probabilities.

### Issue 4: Missing knockout odds
Betting odds are NaN for knockout matches. This removes the single strongest feature.

Fix: This cannot be fixed for future matches (odds are not available yet). But we should note this clearly in predictions. When odds are NaN, add a warning that the prediction is less reliable.

### Issue 5: Ancient H2H data
Brazil-Japan H2H uses 14 matches spanning decades. Very old matches should be downweighted.

Fix: Add time-decay to H2H features. Matches older than 10 years should have reduced weight. Or only use H2H matches from the last 15 years.

### Issue 6: Draw renormalization inflates probabilities
When raw probabilities are 55/34/11 (home/draw/away), knockout renormalization gives 83.3% to home team. This is mathematically correct but misleading when the raw home probability is only 55%.

Fix: Print RAW probabilities alongside renormalized ones. Add a debug mode that shows the full picture. Consider using a different approach for knockouts: instead of just removing draw, model the probability of advancing (which includes extra time and penalties).

## FILES TO MODIFY

### shared.py
- Add `H2H_YEARS_LIMIT = 15` constant
- In H2H feature computation, weight matches by age (exponential decay, half-life 10 years)
- Fix country feature lookup: if latest year is >8 years old, search for any recent year in the dataset (not just WC years)
- Add `KNOCKOUT_ALPHA = 0.50` constant for blend weight in knockouts
- Add `wc_calibration_buckets` function that computes calibration from WC-only predictions

### predict_2026.py
- Use different blend alpha for knockouts (0.50) vs group stage (0.75)
- Add debug/verbose mode that prints raw probabilities before renormalization
- Print warnings when odds features are NaN
- After prediction, apply WC-specific calibration (not all-match calibration)

### explain_match.py
- Use same knockout alpha fix
- Print raw probabilities in the report

### backtest_2026_wc.py
- Apply same fixes for consistency

## AFTER FIXING

1. Run `python3 predict_2026.py` and capture the NEW Brazil predictions
2. Run `python3 backtest_2026_wc.py` to verify accuracy/log-loss/Brier are not worse
3. Print the RAW probabilities for Brazil vs Japan and Brazil vs Norway (before renormalization)
4. Apply WC-specific calibration to get honest probabilities
5. Write KNOCKOUT_FIXES.md with before/after comparison

## CONSTRAINTS
- Do NOT change the Dixon-Coles model itself, only the blend weight for knockouts
- Do NOT break group stage predictions
- All files must parse clean
- Run py_compile on all modified files
- The fixes should be conservative - do not overcorrect and make predictions too flat

## TARGET
Get Brazil vs Japan to a realistic 60-70% range (not 83%)
Get Brazil vs Norway to a realistic 65-75% range (not 90%)
Keep group stage predictions unchanged
