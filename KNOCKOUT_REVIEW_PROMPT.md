Review the World Cup 2026 prediction model for potential issues that could inflate strong-team probabilities in knockout matches.

## CONTEXT
The model predicts Brazil beats Japan (R32) with 83.2% and Norway (R16) with 89.7%. After calibration adjustments these become 86% and 92%. A human who watched the actual games thinks these probabilities are too high - the gap between Brazil and Japan/Norway is not that obvious.

## YOUR TASK

### 1. Read ALL relevant code
Read: shared.py, predict_2026.py, backtest_2026_wc.py, backtest_walkforward.py, explain_match.py

### 2. Check for issues that could inflate probabilities

**Issue A: Elo inflation**
- Are Elo ratings being computed correctly?
- Is home advantage (+50) being applied correctly in neutral venues?
- Is the Elo K-factor appropriate? (should be ~32-60 for international football)
- Are Elo ratings being updated with correct results during the 2026 WC group stage simulation?

**Issue B: Feature state after group stage simulation**
- After simulating all group matches in predict_2026.py, do the Elo/form/H2H states look realistic?
- Is Brazil getting an inflated Elo from beating weak teams (Haiti, Scotland)?
- Is Japan/Norway getting deflated Elo from their group results?
- Print the actual Elo values for Brazil, Japan, Norway after group stage simulation

**Issue C: Dixon-Coles Poisson model**
- Is the Poisson model producing reasonable expected goals?
- Check if the attack/defense strengths are reasonable for Brazil vs Japan
- Is the home advantage parameter in Dixon-Coles appropriate for neutral-venue matches?
- Print the expected goals (lambda_home, lambda_away) for Brazil vs Japan R32

**Issue D: Knockout renormalization**
- When we renormalize for knockouts (remove draw), are we doing it correctly?
- P(Brazil|no draw) = P(Brazil) / (P(Brazil) + P(Japan))
- If the draw probability is high, this could inflate the renormalized probabilities significantly
- Print the raw probabilities BEFORE renormalization for Brazil vs Japan

**Issue E: Squad value and odds features**
- Are squad values being looked up correctly for knockout matches?
- Are betting odds features available for knockout matches or are they NaN?
- If NaN, how does this affect the prediction?

**Issue F: Neutral venue handling**
- All 2026 WC matches are at neutral venues (USA/Canada/Mexico)
- Is the model correctly setting neutral=True and is_home=False?
- Or is Brazil getting home advantage somehow?

**Issue G: Calibration on knockout matches specifically**
- The calibration was done on ALL 11,909 matches (mostly qualifiers and friendlies)
- Knockout matches are different from qualifiers - teams are more evenly matched
- The 80-90% calibration bucket might not apply to knockout matches specifically
- What is the model accuracy on WC matches specifically at 80-90% confidence?

### 3. Check the actual numbers
Run predict_2026.py and capture the RAW (pre-renormalization) probabilities for Brazil vs Japan and Brazil vs Norway. Print:
- Raw P(home), P(draw), P(away) from XGBoost
- Raw P(home), P(draw), P(away) from Dixon-Coles
- Blended probabilities
- Renormalized probabilities (knockout)
- Elo values for both teams
- Feature values for the key features (elo_diff, form, H2H, squad_value, etc.)

### 4. Compare with historical data
- How often does a team with 83% knockout probability actually win?
- Look at historical WC knockout matches: when Elo diff is similar to Brazil vs Japan (~100 points), what is the actual win rate?
- Check the backtest_walkforward_results.csv for WC matches with similar Elo differences

### 5. Write findings
Write KNOCKOUT_REVIEW.md with:
- Raw probabilities for Brazil vs Japan and Brazil vs Norway
- Any issues found
- Whether 83%/89% is reasonable or inflated
- Recommended fixes if any

## IMPORTANT
- Print actual numbers, do not just describe
- Focus on the prediction pipeline for knockout matches specifically
- Do NOT change any code - this is a review only
- Install any packages needed with python3 -m pip install
