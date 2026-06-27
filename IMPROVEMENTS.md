# World Cup 2026 Predictor — Code Review, Fixes & Roadmap

This document is the deliverable for the review of the WC-2026 prediction project. It
covers (1) bugs found and fixed, (2) a prioritized backlog of new variables,
(3) concrete feature-engineering code that has already been implemented,
(4) statistical/ML improvements with implementation sketches, and
(5) a 3-iteration roadmap.

> **Note on benchmarking.** The fixes and 14 new features have been implemented in
> the code. The environment used for this review could not execute Python, so the
> "after" backtest numbers must be produced by running:
> ```bash
> python3 backtest_2026_wc.py        # walk-forward on the 62 group matches
> python3 predict_2026.py            # full bracket prediction
> python3 monte_carlo_2026.py        # 1000-sim tournament distribution
> ```
> Baseline (pre-fix): **62.9% acc, log-loss 0.9143, Brier 0.1850** on 62 group matches.
> The H2H fix alone is expected to move the needle because H2H features were
> previously dead in the backtest (see Bug #1).

---

## Part 1 — Bugs Found and Fixed

All state management (Elo, form, goals, H2H, and the new rolling histories) is now
funneled through two new centralized helpers in `shared.py`:
`update_team_state()` and `apply_match_to_state()`. This single change removes the
class of "scripts drift apart" bugs that caused most of the issues below.

### Bug #1 — H2H features were dead in the backtest (feature computation bug) — FIXED
`backtest_2026_wc.py` stored head-to-head records under **directional** keys
(`(home, away)` and `(away, home)`), but `shared.compute_match_features()` reads them
back under a **sorted** key (`tuple(sorted([team, opponent]))`). The lookup almost
never matched the stored key, so every H2H feature (`h2h_matches`, `h2h_win_rate`,
`h2h_draw_rate`, `h2h_avg_goals_for/against`) was ~0 for the entire backtest.

```python
# BEFORE (backtest_2026_wc.py) — stored directionally, read sorted -> never matches
h2h_key = (ht, at)
state[ht]["h2h"][h2h_key]["matches"] += 1
state[at]["h2h"][(at, ht)]["matches"] += 1
# ... while compute_match_features did:  s["h2h"].get(tuple(sorted([team, opponent])))
```
```python
# AFTER (shared.update_team_state) — single sorted key everywhere
key = tuple(sorted([team, opponent]))
rec = s["h2h"][key]
rec["matches"] += 1
```

### Bug #2 — `stage` trained as constant 0 but used as 0–6 at prediction time — FIXED
The backtest trained with `stage = 0` for **every** historical match, so the model
never learned what `stage` meant. At prediction time it then received `stage` values
of 0–6 (`stage_from_tournament_round`), forcing extrapolation onto a feature the
model treated as constant. Worse, `predict_2026.py`/`monte_carlo_2026.py` used
`final = 5`, a value never present in training (`STAGE_TO_INT` tops out at `final = 4`).

```python
# BEFORE (backtest_2026_wc.py train loop)
stage = 0  # historical matches: stage unknown, use 0
# BEFORE (predict path) returned 0..6, with final mapped to 5/6
```
```python
# AFTER — training now uses the real historical stage map (0..4)
wc_stage_by_index = infer_world_cup_stage_map(df)
stage = wc_stage_by_index.get(int(r.name), 0) if is_world_cup else 0
# AFTER — the 2026 R32/R16/QF/SF/Final scheme is collapsed onto the trained 0..4 range
WC2026_STAGE_TO_TRAIN = {"group":0,"round_of_32":1,"round_of_16":1,
                         "quarterfinal":2,"semifinal":3,"third_place":3,"final":4}
```

### Bug #3 — `neutral`/`is_home` hard-coded during training and prediction — FIXED
The backtest called `compute_match_features(...)` with default `neutral=True,
is_home=False` for **all** matches, including 2026 host fixtures (USA/Canada/Mexico
playing at home with `neutral=False` in the CSV). Home advantage was therefore never
modeled. Both the training loop and the walk-forward prediction now read the real
`neutral` flag and set `is_home = not neutral`.

```python
# BEFORE
rows.append(compute_match_features(ht, at, state, cf, stage, r["date"]))     # neutral=True
predicted, probs = predict_match(model, fn, home, away, state, cf, stage, date)  # neutral=True
```
```python
# AFTER
neutral = parse_bool(r.get("neutral", True)); is_home = not neutral
rows.append(compute_match_features(ht, at, state, cf, stage, r["date"],
                                   neutral=neutral, is_home=is_home))
# predict_2026 also flags the 2026 co-hosts explicitly:
WC2026_HOSTS = {"USA", "Canada", "Mexico"}  # get home advantage at home
```

### Bug #4 — Inconsistent rolling windows across scripts (10 vs 20) — FIXED
`predict_2026.py` and `monte_carlo_2026.py` kept 20-match form/goals windows;
`backtest_2026_wc.py` kept 10. The model was trained on one window length and scored
on another depending on which script ran. Now a single `FORM_WINDOW = 20` constant in
`shared.py` governs all rolling state via `update_team_state()`.

### Bug #5 — Random (non-chronological) validation holdout in some paths — FIXED
`monte_carlo_2026.py` and `backtest_2026_wc.py` called
`fit_xgb_with_validation(model, X, y)` **without** `dates`, which triggers a *random*
train/test split on time-series data (optimistic, mildly leaky for early-stopping).
Both now pass `dates=feature_dates` so the holdout is the chronological tail, matching
`predict_2026.py`.

### Bug #6 — `incremental_predictor.py` leakage audit — VERIFIED CLEAN (no change needed)
The brief flagged `gdp_per_capita_vs_winner` etc. as "not dropped." In the current
code they **are** excluded in `build_country_feature_lookup()` (along with `won_wc`,
`runner_up`, `semifinalist`, `finalist`, `top4`, `is_winner`,
`population_vs_winner`, `total_goals_in_tournament`, `avg_goals_per_match`).
Country features are also looked up with a strict `year < match_year` filter, so a
match never sees its own tournament's country row. The enriched columns
(`gdp_per_capita_vs_avg`, `population_vs_avg`, `is_former_champion`) are
participant-average / prior-title derived and are leakage-safe. **No leak remains.**
The one residual recommendation is to make the exclusion list a shared constant so it
cannot silently drift from the dataset schema (see Roadmap, Iteration 1).

### Summary of files changed
- `shared.py` — added `update_team_state`, `apply_match_to_state`, `FORM_WINDOW`,
  `WC2026_STAGE_TO_TRAIN`, extended `make_team_state` and `compute_match_features`
  (14 new features).
- `backtest_2026_wc.py` — uses shared state; real stage map; real neutral/is_home;
  chronological holdout; collapsed stage codes at prediction.
- `predict_2026.py` — uses shared state; collapsed stage codes; host home-advantage.
- `monte_carlo_2026.py` — uses shared state; real stage map; collapsed stage codes;
  chronological holdout.

---

## Part 2 — Prioritized New Variables (impact vs effort)

Legend: Impact/Effort are H/M/L. "Have data" = derivable from existing CSVs.

| # | Variable | Impact | Effort | Have data | Status |
|---|----------|:---:|:---:|:---:|---|
| 1 | Elo momentum (Δ over last 5/10) | H | L | yes | **Implemented** |
| 2 | Opponent-strength-weighted form | H | L | yes | **Implemented** |
| 3 | Attack/defense trend (recent 3 vs 10) | M-H | L | yes | **Implemented** |
| 4 | Fatigue / congestion (games in 30/90d) | M | L | yes | **Implemented** |
| 5 | Draw-propensity signals (parity, low-scoring) | H (draws!) | L | yes | **Implemented** |
| 6 | FIFA/Coca-Cola monthly ranking + delta | H | M | partial (`enrich`) | Backlog |
| 7 | Transfermarkt squad market value | H | M | no (scrape) | Backlog |
| 8 | Bookmaker implied probabilities | **VH** | M | no (API) | Backlog |
| 9 | Travel distance / time-zone change | M | M | partial (venue city) | Backlog |
| 10 | Rest-days differential (already partial) | M | L | yes | Partial |
| 11 | Squad age profile / experience (caps) | M | H | no (player data) | Backlog |
| 12 | Manager tenure & win rate | L-M | H | no | Backlog |
| 13 | Match stats (xG, shots, possession) | H | H | no | Backlog |
| 14 | Venue altitude / weather | L-M | M | partial (city) | Backlog |
| 15 | Referee tendencies (cards/pens) | L | H | no | Backlog |

**Highest ROI not yet done:** #8 bookmaker odds (single best predictor in football
literature), then #6 FIFA ranking deltas and #7 market values.

---

## Part 3 — Concrete Feature Engineering (already implemented in `shared.py`)

`make_team_state()` now also tracks `elo_history`, `opp_elo`, and `match_dates`, all
maintained by the centralized `update_team_state()`. `compute_match_features()` emits
the following 14 new columns:

```python
# 1. Elo momentum — trajectory, not just level
"elo_momentum_5":  _elo_momentum(s, 5),
"elo_momentum_10": _elo_momentum(s, 10),
"elo_momentum_diff": _elo_momentum(s, 5) - _elo_momentum(o, 5),

# 2. Attack / defense trend — recent 3 matches vs 10-match baseline
"attack_trend":  np.mean(gf3) - np.mean(gf10),
"defense_trend": np.mean(ga3) - np.mean(ga10),   # >0 = leaking more lately

# 3. Opponent-strength-weighted form — beating Argentina >> beating Haiti
"weighted_form":      _opp_weighted_form(s),
"weighted_form_diff": _opp_weighted_form(s) - _opp_weighted_form(o),

# 4. Fatigue / congestion
"fatigue_30": _matches_in_window(s, match_date, 30),
"fatigue_90": _matches_in_window(s, match_date, 90),
"fatigue_diff_30": fatigue_30 - opp_fatigue_30,

# 5. Draw-propensity signals (directly targets the documented draw weakness)
"elo_parity":          1.0 / (1.0 + abs(elo_diff) / 100.0),   # ~1 evenly matched
"combined_draw_rate":  (form_draw_rate + opp_form_draw_rate) / 2.0,
"expected_total_goals": (recent gf/ga of both sides) / 2,
"low_scoring_indicator": 1.0 / (1.0 + expected_total_goals),
```

Helper functions added (`_elo_momentum`, `_opp_weighted_form`,
`_matches_in_window`) are pure functions over the rolling state, so they are
leakage-safe by construction (they only read history recorded *before* the current
match).

### Why these five
The model's single documented weakness is **draws** (all 11 high-confidence misses
were draws predicted as home wins). Features #5 give the model an explicit "this looks
like a draw" signal (parity + low expected goals + both teams drawing recently), which
none of the original 38 features captured. Features #1–#4 add *dynamics* (the original
set was almost entirely static levels), which is where most marginal predictive lift
in football models comes from.

---

## Part 4 — Statistical / ML Improvements (sketches)

### 4.1 Calibrated draw handling (do this first — highest expected lift)
The draw class is under-predicted. Two cheap options:
- **Class weights / `sample_weight`:** up-weight draw rows so the softmax stops
  collapsing draws into home wins.
- **Isotonic calibration on a chronological holdout:** wrap the fitted XGB in
  `sklearn.calibration.CalibratedClassifierCV(method="isotonic", cv="prefit")` using
  the last 20% of matches, then recompute log-loss/Brier.
```python
from sklearn.calibration import CalibratedClassifierCV
cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
cal.fit(X_val, y_val)   # X_val/y_val = chronological tail
```

### 4.2 Separate group-stage vs knockout models
Group and knockout matches have different base rates (no draws in KO, different
intensity). Train two heads on the same features; route by `stage == 0`. Even simpler:
keep one model but add an interaction term `stage * elo_parity`.

### 4.3 Time-decay weighting
Weight training rows by recency so 2024 friendlies matter more than 1994 ones:
```python
half_life_days = 365 * 4
w = 0.5 ** ((max_date - dates).dt.days / half_life_days)
model.fit(X, y, sample_weight=w)
```

### 4.4 Poisson / Dixon–Coles goal model as an ensemble member
Fit team attack/defense strengths to model `Home ~ Poisson(λ_h)`, `Away ~ Poisson(λ_a)`,
derive W/D/L from the scoreline grid (this naturally produces realistic draw
probabilities), then **blend** with XGBoost: `p = α·p_xgb + (1-α)·p_poisson`, tuning
α on the chronological holdout. This is the cleanest fix for the draw problem and also
unlocks scoreline simulation.

### 4.5 Feature selection & importance stability
With 52 features now, run permutation importance per LOWCO fold and drop features
whose importance is indistinguishable from noise across folds (many country-demographic
columns are likely droppable, consistent with the SOTA finding that GDP is near-useless
for *winners*).

### 4.6 Ensemble / stacking
Stack: XGBoost + Elo logistic baseline + Poisson, with a logistic meta-learner trained
on out-of-fold predictions. Expect the biggest gains on log-loss/Brier rather than raw
accuracy.

---

## Part 5 — Analysis Improvements
- **SHAP interaction values** in `explain_match.py` (`shap.TreeExplainer(...).shap_interaction_values`)
  to confirm e.g. `elo_diff × stage` and `elo_parity × combined_draw_rate`.
- **Partial dependence** of P(home win) on `elo_diff` and of P(draw) on `elo_parity`.
- **Prediction intervals** from the Monte Carlo champion distribution (already has
  Wald CIs; add bootstrap over model seeds).
- **Scoreline simulation** once the Poisson model lands (sample (h,a) goals, not just
  W/D/L), which makes `monte_carlo_2026.py` produce realistic goal-difference tables.

---

## Roadmap — Next 3 Iterations

### Iteration 1 — Lock in correctness & draws (this PR + small follow-ups)
- [x] Centralize state (fixes H2H, stage, neutral, window drift).
- [x] Add 14 dynamic + draw-propensity features.
- [ ] Run all three scripts; record new acc / log-loss / Brier vs the 62.9% baseline.
- [ ] Add draw `sample_weight` + isotonic calibration; re-measure Brier/log-loss.
- [ ] Promote the leakage-exclusion list to a shared constant.

### Iteration 2 — External signals (biggest accuracy lever)
- [ ] Ingest bookmaker closing odds (implied probabilities) as features + as a baseline.
- [ ] Add monthly FIFA ranking + 6/12-month ranking delta.
- [ ] Add Transfermarkt squad value (total + top-3 player) with year lag.
- [ ] Travel distance / time-zone change from venue city → host coordinates.

### Iteration 3 — Model architecture
- [ ] Dixon–Coles Poisson goal model + scoreline simulation.
- [ ] Stacked ensemble (XGB + Elo + Poisson) with chronological OOF meta-learner.
- [ ] Separate group/KO heads, time-decay weighting, per-fold feature selection.
- [ ] Full calibration report (reliability diagram, Brier decomposition) in the backtest.
