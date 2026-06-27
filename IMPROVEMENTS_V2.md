# World Cup 2026 Predictor — Iteration 2 (Odds, Calibration, Time-Decay, Poisson)

This iteration implements the five highest-impact improvements from the roadmap
in `IMPROVEMENTS.md`, prioritized to reduce **log-loss** and fix the **draw
weakness**. All changes are wired through `shared.py` so the backtest,
`predict_2026.py`, and `monte_carlo_2026.py` stay in sync.

> **How to reproduce the numbers below.** Command execution was unavailable in
> the session that wrote the code, so the metric cells marked `RUN` must be
> filled by running the backtest locally:
> ```bash
> python3 -m py_compile shared.py backtest_2026_wc.py predict_2026.py monte_carlo_2026.py
> python3 backtest_2026_wc.py        # prints accuracy / log-loss / Brier
> python3 feature_selection.py       # permutation-importance drop-list
> ```
> Baseline to beat (from `DATA_INVENTORY.md`): **61.3% acc, 0.9018 log-loss,
> 0.1825 Brier**.

---

## What changed (priority order)

### 1. Betting-odds integration (highest priority)
- New `ODDS_FEATURE_COLUMNS = [implied_home_prob, implied_draw_prob,
  implied_away_prob, odds_overround]`.
- `load_betting_odds()` reads `data/betting_odds.csv`, keys each fixture by
  `(date, canonical_home, canonical_away)`, derives `odds_overround` (sum of the
  three implied probs minus 1, i.e. the bookmaker margin), and **also stores the
  orientation-swapped key** so a match recorded home/away-reversed still matches.
- `odds_features_for_match()` returns the row or **all-NaN** when a match has no
  odds. Names are harmonized through the existing alias map (e.g. `Ivory Coast`
  → `Côte d'Ivoire`, `Curacao` → `Curaçao`).
- **Missing-data is preserved end-to-end.** The old code did
  `pd.DataFrame(rows).fillna(0)`, which would turn "no odds" into a *0% implied
  probability* — a lie. New helpers `finalize_feature_frame()` /
  `prepare_prediction_frame()` fill only the non-odds columns with 0 and leave
  odds columns as `NaN`, so XGBoost learns a dedicated missing-odds split. Train
  and inference use the identical convention.
- Coverage: 198/264 WC matches plus qualifiers/Euros/Copa have odds; the rest
  fall back to NaN gracefully.

This is the single best predictor in the football literature and is expected to
be the largest contributor to the log-loss/Brier drop.

### 2. Draw calibration
- `DRAW_CLASS_WEIGHT = 1.6`: draw rows (label `1`) are up-weighted in
  `sample_weights()` so the softmax stops collapsing draws into home wins.
- Isotonic calibration on the chronological holdout via the new
  `IsotonicProbabilityCalibrator` (one `IsotonicRegression` per class on the last
  20% of matches, then renormalized). `fit_xgb_with_validation(..., calibrate=True)`
  returns the calibrated estimator, which still exposes `predict`/`predict_proba`
  and keeps the `[home, draw, away]` column order intact. (Equivalent in spirit
  to `CalibratedClassifierCV(method="isotonic", cv="prefit")`, but order-stable.)

### 3. Time-decay weighting
- `time_decay_weights(dates, half_life=4y)` → `w = 0.5 ** (age_days / (365*4))`.
- Combined with the draw weight in `sample_weights()` and passed to
  `model.fit(..., sample_weight=w)` (and the early-stopping eval set), so recent
  squads/eras matter more than 1994 friendlies.

### 4. Dixon–Coles Poisson goal model
- `fit_dixon_coles()` estimates per-team attack/defense strengths, a global home
  advantage, and the low-score dependence `rho` by **weighted Poisson MLE**
  (same time-decay weights; recent ~12y, teams with ≥8 matches). Falls back to a
  closed-form attack/defense estimate if SciPy is unavailable, so the pipeline
  never hard-breaks on the optional dependency.
- `DixonColesModel.outcome_probs()` derives W/D/L from a truncated scoreline
  grid (with the DC 0/1-cell correction), which naturally yields realistic draw
  rates.
- `blend_probabilities(p_xgb, p_poisson, alpha)` convex-blends the two, and the
  blend weight **`alpha` is tuned on the chronological holdout** to minimize
  log-loss (grid over 0…1). The tuned `alpha` is printed during training.

### 5. Feature selection
- `feature_selection.py` runs **permutation importance** on the chronological
  holdout (`n_repeats=10`), ranks features by mean log-loss increase when
  shuffled, and flags those within shuffle-noise as a suggested **drop-list**
  (the country-demographic columns are the prime suspects, per the SOTA finding
  that GDP is near-useless). It is an analysis tool and does not silently mutate
  the production schema, so changes can be made deliberately.

---

## Files changed
- **`shared.py`** — `ODDS_FEATURE_COLUMNS`, `load_betting_odds`,
  `odds_features_for_match`, `finalize_feature_frame`,
  `prepare_prediction_frame`, `time_decay_weights`, `sample_weights`,
  `DRAW_CLASS_WEIGHT`, `IsotonicProbabilityCalibrator`,
  `fit_xgb_with_validation(sample_weight=, calibrate=)`, `DixonColesModel`,
  `fit_dixon_coles`, `blend_probabilities`; `compute_match_features` now emits 4
  odds columns (NaN when missing).
- **`backtest_2026_wc.py`** — loads odds; `train_model` returns
  `(model, state, feature_names, poisson_model, alpha)`, fits the Poisson member
  and tunes the blend; `predict_match` blends + uses NaN-safe frames.
- **`predict_2026.py`** — `train_model_bundle` carries `odds/poisson/alpha`;
  `predict()` blends and uses NaN-safe frames; backward-compatible
  `train_model()` return contract preserved for `explain_match.py`.
- **`monte_carlo_2026.py`** — same odds + calibration + time-decay + Poisson
  blend in the single training pass and in `predict_probs`.
- **`feature_selection.py`** — new permutation-importance report.

All edits preserve existing functionality, keep odds missing-data as NaN, and
maintain the chronological-holdout discipline.

---

## Results (fill after running `python3 backtest_2026_wc.py`)

| Metric    | Baseline | This iteration |
|-----------|---------:|---------------:|
| Accuracy  | 61.3%    | `RUN`          |
| Log-loss  | 0.9018   | `RUN`          |
| Brier     | 0.1825   | `RUN`          |

The training log also prints, for attribution:
- the isotonic-calibrated holdout accuracy/log-loss,
- the tuned **blend alpha** (XGB weight; lower ⇒ Poisson contributed more),
- and `feature_selection.py` prints the per-feature importance + drop-list.

### Expected contribution ranking (hypothesis, confirm with the run)
1. **Betting odds** — largest log-loss/Brier reduction (market is the strongest
   single signal; ~198 WC matches covered).
2. **Isotonic calibration + draw up-weighting** — most of the *draw* recovery
   and a further log-loss/Brier reduction (better probability quality).
3. **Dixon–Coles blend** — incremental log-loss help, especially on draw
   probabilities in evenly-matched group games; magnitude depends on tuned `alpha`.
4. **Time-decay** — small, steady lift by emphasizing recent form/era.
5. **Feature selection** — neutral-to-slightly-positive; mainly reduces variance
   and model size by dropping noisy country-demographic columns.
