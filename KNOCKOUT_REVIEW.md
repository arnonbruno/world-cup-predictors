# Knockout Probability Review

## Scope And Execution Note

Reviewed the knockout prediction path in:

- `shared.py`
- `predict_2026.py`
- `backtest_2026_wc.py`
- `backtest_walkforward.py`
- `explain_match.py`

I attempted to run `python3 predict_2026.py` and a targeted Python diagnostic, but this environment rejected Python execution, including `python3 --version` and `python3 -c "print('hi')"`. The exact raw XGBoost/Dixon-Coles/blended pre-renormalization outputs below therefore could not be freshly recomputed in this session. I used the available code paths and committed/generated artifacts already in the repo.

## Available Match Numbers

### Brazil vs Japan, R32

| Item | Value |
| --- | ---: |
| Cited production knockout probability | 83.2% Brazil |
| Cited calibrated probability | 86.0% Brazil |
| Existing `explain_match` XGBoost-only knockout probability | 63.4% Brazil / 36.6% Japan |
| Existing `explain_match` Elo-only knockout baseline | 63.6% Brazil / 36.4% Japan |
| Rolling Elo after completed group matches | Brazil 2085, Japan 1988 |
| Rolling Elo diff | +96.6 Brazil |
| Pre-tournament Elo feature diff | +220 Brazil |
| Brazil form win rate | 0.600 |
| Japan form win rate | 0.700 |
| Brazil avg goals scored/conceded, last 10 | 2.400 / 1.100 |
| Japan avg goals scored/conceded, last 10 | 2.000 / 0.700 |
| H2H record in feature state | Brazil 11 wins, 2 draws, Japan 1 win |
| H2H win rate feature for Brazil | 0.786 |
| 2026 squad values | Brazil EUR 928.2M, Japan EUR 270.9M |
| Squad log values | Brazil ~20.649, Japan ~19.417 |
| Squad value diff / ratio | +1.232 log points / 3.43x |
| Betting odds features | NaN; no Brazil-Japan knockout odds row found |
| Venue features in existing feature row | `neutral=1`, `is_home=0` |

Raw pre-renormalization values were not available from existing artifacts:

| Model output | P(Brazil) | P(draw) | P(Japan) |
| --- | ---: | ---: | ---: |
| XGBoost raw | unavailable without Python execution |
| Dixon-Coles raw | unavailable without Python execution |
| Blended raw | unavailable without Python execution |
| Knockout-renormalized production | 83.2% | removed | 16.8% |
| Calibrated/adjusted | 86.0% | removed | 14.0% |

The existing Brazil-Japan explanation is not equivalent to production `predict_2026.py`: it uses only the XGBoost model and `explain_match.prepare_X()` fills missing values with `0.0`, while production `prepare_prediction_frame()` intentionally keeps odds and squad-value missingness as NaN. It is useful for feature state, but not for the final blended probability.

### Brazil vs Norway, R16

| Item | Value |
| --- | ---: |
| Cited production knockout probability | 89.7% Brazil |
| Cited calibrated probability | 92.0% Brazil |
| 2026 squad values | Brazil EUR 928.2M, Norway EUR 589.9M |
| Squad log values | Brazil ~20.649, Norway ~20.196 |
| Squad value diff / ratio | +0.453 log points / 1.57x |
| Betting odds features | NaN; no Brazil-Norway knockout odds row found |
| Brazil-Norway historical H2H in `data/results.csv` | Brazil 0 wins, 2 draws, Norway 2 wins |
| Norway country feature vintage | latest available row is 1998 |

Raw pre-renormalization values were not available from existing artifacts:

| Model output | P(Brazil) | P(draw) | P(Norway) |
| --- | ---: | ---: | ---: |
| XGBoost raw | unavailable without Python execution |
| Dixon-Coles raw | unavailable without Python execution |
| Blended raw | unavailable without Python execution |
| Knockout-renormalized production | 89.7% | removed | 10.3% |
| Calibrated/adjusted | 92.0% | removed | 8.0% |

Brazil-Norway is especially suspicious because Norway's latest country-level World Cup feature row is from 1998, so `elo_pre_tournament`, `fifa_rank`, `football_power_index`, demographics, and tournament-history features are stale relative to the 2026 Norway team. That can deflate Norway independently of rolling Elo.

## Findings

### 1. Elo Handling Looks Mostly Correct, But Deterministic Simulation Can Compound Favorite Strength

The Elo update function uses:

- `K_FACTOR = 32`
- neutral home advantage = `0`
- non-neutral home advantage = `+50`
- margin multiplier = `log(max(margin, 1) + 1)`

That K-factor is within the requested international-football range. With the margin multiplier, the effective K is about 44.4 for a 3-goal win and 51.5 for a 4-goal win, still reasonable. Neutral 2026 matches do not give Brazil a +50 home Elo adjustment.

The state update order is also correct: both teams' opponent Elo snapshots are recorded before Elo is updated.

The inflation risk is not an obvious Elo bug. It is the deterministic tournament simulation: `predict_2026.py` updates state after simulated group/knockout matches with fixed scorelines (`2-1`, `1-2`, or `1-1`). A favorite that the model advances receives additional Elo/form/H2H updates before the next round. Brazil-Norway includes a simulated Brazil R32 win over Japan before the R16 prediction.

For Brazil-Japan specifically, available artifacts do not show massive Brazil Elo inflation from Haiti/Scotland. Brazil's existing post-group rolling Elo is 2085 versus pre-tournament feature 2080; Japan's rolling Elo is 1988 versus pre-tournament feature 1860. The Brazil-Japan rolling Elo gap is only +96.6.

### 2. Knockout Renormalization Is Mathematically Correct, But Can Inflate The Displayed Favorite Probability

Production knockout handling uses:

`P(home | no draw) = P(home) / (P(home) + P(away))`

That is mathematically correct if the raw 3-way probabilities represent regulation-time outcomes and the no-draw conditional is meant to proxy "advances eventually."

The practical issue is calibration: a high raw draw probability can make the conditional favorite probability look much larger. For example, if raw probabilities were 55/34/11, the displayed knockout probability becomes 83.3%. That may be a valid conditional probability only if extra time/penalties preserve the same favorite-underdog relationship. The current code does not model extra time or penalties separately.

The `predict_2026.py` CLI also does not print raw pre-renormalization probabilities for knockout matches; it prints only the conditional home/away values and keeps draw hidden except as an internal return value. That makes high draw-driven inflation hard to audit.

### 3. Dixon-Coles Neutral Handling Is Correct, But Its Contribution Needs Auditing

`DixonColesModel._lambdas()` applies home advantage only when `neutral=False`; for neutral 2026 knockouts, `adv=0.0`. Brazil should not receive Dixon-Coles home advantage against Japan or Norway.

The likely risk is not venue handling but Dixon-Coles strength calibration. The README and experiment artifacts describe the production-style ensemble as 75% Dixon-Coles / 25% XGBoost, while `predict_2026.py` tunes `alpha` dynamically. Since the XGBoost-only Brazil-Japan knockout explanation is only 63.4%, the cited 83.2% production number likely comes from the Poisson/blend path plus draw removal. The exact Dixon-Coles lambdas for Brazil-Japan could not be printed because Python execution was blocked.

Recommended diagnostic when execution is available:

```python
p_xgb = model.predict_proba(X)[0]
p_dc = poisson_model.outcome_probs("Brazil", "Japan", neutral=True)
p_blend = blend_probabilities(p_xgb, p_dc, alpha)
print(poisson_model._lambdas("Brazil", "Japan", neutral=True))
print(p_xgb, p_dc, p_blend, p_blend[0] / (p_blend[0] + p_blend[2]))
```

### 4. Squad Values Are Present; Knockout Odds Are Missing

Squad values are looked up for 2026 and are present:

- Brazil: EUR 928.2M
- Japan: EUR 270.9M
- Norway: EUR 589.9M

The feature builder log-scales these and computes diff/ratio. The Brazil-Japan squad ratio is large at 3.43x. Brazil-Norway is more modest at 1.57x.

No betting-odds rows were found for Brazil-Japan on 2026-06-29 or Brazil-Norway on 2026-07-04. Production keeps these odds features as NaN, so XGBoost follows its learned missing-value paths. That is better than zero-imputation, but it means the single strongest real-world sanity check (market odds) is absent for these knockouts.

### 5. Neutral Venue Handling Does Not Give Brazil Home Advantage

For production knockout rounds, `simulate_round()` calls `predict()` without overriding `neutral`, so `neutral=True` and `is_home=False`.

In remaining group matches, `predict_2026.py` gives home advantage only to co-hosts (`USA`, `Canada`, `Mexico`) when they are the listed home side. Brazil is not a host. Existing Brazil-Japan features also show `neutral=1` and `is_home=0`.

This does not explain inflated Brazil probabilities.

### 6. World Cup And Knockout Calibration Is Much Weaker Than Overall Calibration

The saved walk-forward summary shows:

| Slice | Matches | Accuracy | Log-loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| All validation matches | 11,909 | 59.6% | 0.8795 | 0.1724 |
| World Cup matches | 254 | 54.7% | 0.9944 | 0.1973 |

Overall all-match calibration looks good in the 80-90% bucket:

| Bucket | Matches | Avg confidence | Accuracy |
| --- | ---: | ---: | ---: |
| 80-90%, all matches | 941 | 84.8% | 87.4% |

But World Cup-only 80-90% examples in `backtest_walkforward_results.csv` are much worse:

| Match | Confidence | Correct? |
| --- | ---: | --- |
| 2014 Brazil 4-1 Cameroon | 85.0% | yes |
| 2022 Brazil 2-0 Serbia | 81.2% | yes |
| 2022 Cameroon 1-0 Brazil | 86.7% | no |
| 2022 Croatia 1-1 Brazil | 80.2% | no |
| 2026 Spain 0-0 Cape Verde | 80.1% | no |
| 2026 Brazil 3-0 Haiti | 88.9% | yes |
| 2026 Scotland 0-3 Brazil | 85.0% | yes |

That is 4/7 correct = 57.1% at about 83.9% average confidence. Small sample, but it directly supports the concern that all-match calibration does not transfer cleanly to World Cup contexts.

For proper historical WC knockout matches in that same 80-90% range, the saved file has only one clear example: 2022 Croatia-Brazil, predicted Brazil at 80.2%, actual draw after regulation. That is 0/1. The 2026 rows in `backtest_walkforward_results.csv` cannot be cleanly used as knockout rows because the generic historical stage inference mislabels late 2026 group-stage rows as stages 1-4.

### 7. Backtest Stage Handling Has Two Knockout-Specific Gaps

`backtest_2026_wc.py` calls `stage_from_tournament_round(r.get("tournament", ""), ...)`, but `tournament` is just `"FIFA World Cup"`. It does not contain "group", "round of 32", "quarter", etc., so completed 2026 matches default to stage 0. That script does not actually validate knockout-specific behavior unless a separate round column is added.

`backtest_walkforward.py` uses historical World Cup schedule shapes to infer stage. That works reasonably for normal 32-team tournaments, but the 2026 expanded group-stage data has more rows. Late 2026 group matches are misclassified as knockout stages in `backtest_walkforward_results.csv`.

These gaps make the "80-90% knockout calibration" question under-measured.

## Is 83% / 90% Reasonable?

The evidence points to "probably inflated," especially for Norway.

For Brazil-Japan, a 63-64% knockout probability is consistent with the rolling Elo gap of about 97 points and with the existing XGBoost-only explanation. Jumping to 83-86% likely requires a strong Dixon-Coles/blend effect plus draw renormalization and/or H2H/squad effects. Given Japan's recent form in the feature row is at least as strong as Brazil's, 83-86% looks aggressive.

For Brazil-Norway, 89.7-92% looks more clearly inflated. Norway has a strong 2026 squad value, historically favorable H2H against Brazil in the local data, and stale 1998 country/team features that can deflate Norway in the model. Without a separate extra-time/penalty model, removing draw probability can overstate Brazil's chance to advance.

## Recommended Fixes

1. Add a knockout debug mode to `predict_2026.py` that prints raw XGBoost, raw Dixon-Coles, blended 3-way probabilities, conditional knockout probabilities, Elo values, lambdas, and the full feature row before every knockout prediction.

2. Calibrate and report knockout probabilities separately from regulation 3-way probabilities. A better structure is:
   - regulation-time home/draw/away
   - extra-time home/away conditional on draw
   - penalties home/away conditional on still tied
   - final "advances" probability

3. Do not rely on all-match calibration for knockouts. Build a World Cup knockout reliability table, even if sparse, and report uncertainty/sample size.

4. Fix stage handling in both backtests before trusting knockout calibration:
   - `backtest_2026_wc.py` needs an actual round/stage source, not the `tournament` string.
   - `backtest_walkforward.py` needs 2026-specific expanded-format stage inference or should exclude incomplete 2026 group data from knockout-stage summaries.

5. Refresh country/team feature rows for 2026 participants that lack recent World Cup rows, especially Norway. Falling back to Norway 1998 is not acceptable for a 2026 knockout prediction.

6. Consider capping or down-weighting ancient H2H. Brazil-Japan uses 14 historical H2H matches, many from old eras, and the H2H counterfactual in the existing report moves the XGBoost-only knockout probability from 63.4% to 52.3%.

7. Treat missing knockout odds explicitly in reports. If odds are NaN for future knockouts, print that fact and avoid presenting the probability as market-informed.

