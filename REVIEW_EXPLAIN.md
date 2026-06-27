# Review: `explain_match.py` and World Cup Prediction Project

This review covers `explain_match.py`, the simple 2026 prediction pipeline in
`shared.py` and `predict_2026.py`, the data collection pipeline in
`collect_data.py`, and the broader project scripts. The core finding is severe:
the current explanation script is explaining an all-zero model input, not the
Brazil-Japan matchup. The SHAP values are computed, but they are SHAP values for
a synthetic zero vector that the model never intended to receive.

## 1. Critical Issues

### 1.1 `explain_match.py` silently builds an all-zero model vector

**Severity:** Critical  
**Files:** `explain_match.py`, `shared.py`, `predict_2026.py`

The failure has three steps:

1. `shared.compute_match_features()` requires accumulated team state:

```python
# shared.py:205-217
def compute_match_features(team, opponent, state, country_features, stage_num, match_date):
    s, o = state[team], state[opponent]
    form = s["form"][-10:] if s["form"] else [0.5]
    ...
    cf, oc = country_features.get(team, {}), country_features.get(opponent, {})
    return {
        "elo": s["elo"], "elo_opponent": o["elo"],
        ...
    }
```

2. `explain_match.py` never passes that state. It tries several incompatible
   signatures:

```python
# explain_match.py:333-354
func = ctx.shared.compute_match_features
row = pd.Series(base)
attempts = [
    ((ctx.home, ctx.away, ctx.year, ctx.stage), {}),
    ((ctx.home, ctx.away), {"year": ctx.year, "stage": ctx.stage}),
    ((row,), {}),
    ((row, home_features, away_features), {}),
    ((dict(base), home_features, away_features), {}),
    ((), {"home": ctx.home, "away": ctx.away, "year": ctx.year, "stage": ctx.stage}),
    ...
]
```

None matches `(team, opponent, state, country_features, stage_num, match_date)`.
The generated report already proves this path failed:

```text
`shared.compute_match_features()` could not build this matchup:
compute_match_features() missing 3 required positional arguments:
'country_features', 'stage_num', and 'match_date'
```

3. After feature construction fails, `prepare_X()` pads every trained model
   feature missing from the fallback row with `0.0`:

```python
# explain_match.py:542-553
def prepare_X(frame: pd.DataFrame, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    ...
    if feature_names:
        for name in feature_names:
            if name not in frame:
                frame[name] = 0.0
        frame = frame[list(feature_names)]
    return frame.fillna(0.0)
```

This is the root cause of `python explain_match.py Brazil Japan --no-interactive`
showing `0.000` for almost every feature. It is not primarily a SHAP issue. It
is a feature construction and state plumbing bug.

**Why this makes the output meaningless:** the model was trained on rows built
from rolling Elo, form, H2H, World Cup history, and country features. The
explanation script predicts on a vector where those values are absent and then
forced to zero. The probabilities and SHAP values describe the model's response
to that invalid vector.

**Required fix:** stop constructing match features before a trained state exists.
Load or train the model together with the accumulated state, then call:

```python
match_date = pd.Timestamp(args.match_date or "2026-06-29")
country_features = shared.country_features_for_year(ctx.country_history, min(ctx.year, 2022))
features = shared.compute_match_features(
    ctx.home,
    ctx.away,
    bundle.state,
    country_features,
    ctx.stage,
    match_date,
)
match_features = pd.DataFrame([features])
X = match_features.reindex(columns=bundle.feature_names)
missing = [c for c in bundle.feature_names if c not in match_features.columns]
if missing:
    raise RuntimeError(f"Feature builder did not produce trained columns: {missing[:20]}")
X = X.fillna(0.0)
```

The key point is that missing trained columns should be an error for core model
features, not silently converted to zeros.

### 1.2 `explain_match.py` discards the state returned by `predict_2026.train_model()`

**Severity:** Critical  
**Files:** `explain_match.py`, `predict_2026.py`

`predict_2026.train_model()` already returns accumulated state:

```python
# predict_2026.py:126-134
X = pd.DataFrame(rows).fillna(0)
y = np.array(labels)
...
model, _metrics = fit_xgb_with_validation(model, X, y, label="XGBoost")
print(f"  Trained on {len(X)} matches")
return model, state, X.columns.tolist()
```

But `explain_match.py` scans the tuple and only keeps the first object with
`predict_proba`:

```python
# explain_match.py:466-470
if isinstance(result, tuple):
    for item in result:
        if has_predict_proba(item):
            notes.detail("Trained model from tuple returned by `predict_2026.train_model()`.")
            return item
```

The returned `state` and `feature_names` are discarded. This is the direct
reason the explanation script cannot reproduce prediction-time features.

**Required fix:** represent the trained model as a bundle, not as a bare model.
At minimum, `load_model_from_predict_module()` must return the model, state,
feature names, and training data if available.

Concrete change:

```python
# explain_match.py
@dataclass
class ModelBundle:
    model: Any
    feature_names: list[str]
    class_labels: list[Any]
    state: Any | None = None
    country_features: dict[str, dict[str, float]] | None = None
    train_X: pd.DataFrame | None = None
    train_y: pd.Series | None = None
```

Then parse the `train_model()` tuple explicitly:

```python
result = module.train_model(results_df, country_history)
if isinstance(result, tuple) and len(result) >= 3 and has_predict_proba(result[0]):
    model, state, feature_names = result[:3]
    return ModelBundle(
        model=model,
        state=state,
        feature_names=list(feature_names),
        class_labels=list(getattr(model, "classes_", OUTCOME_LABELS)),
    )
```

The better fix is in section 4: expose a proper training bundle from
`predict_2026.py` so all callers stop reverse-engineering tuples.

### 1.3 Country features are loaded with the wrong API

**Severity:** Critical  
**Files:** `explain_match.py`, `shared.py`

`shared.country_features_for_year()` takes only `(history, year)` and returns a
mapping for all teams:

```python
# shared.py:273-283
def country_features_for_year(
    history: Dict[str, Dict[int, Dict[str, float]]], year: int
) -> Dict[str, Dict[str, float]]:
    ...
    return features
```

`explain_match.py` calls it as if it can return one team's features:

```python
# explain_match.py:286-318
features = flexible_call(func, attempts)
...
attempts = [
    ((history, team, year), {}),
    ((history, year, team), {}),
    ((team, year, history), {}),
    ...
]
```

Every attempted call is wrong for the current `shared.py`; the generated report
shows both Brazil and Japan failed:

```text
Could not load country features for Brazil: country_features_for_year() got an unexpected keyword argument 'country'
Could not load country features for Japan: country_features_for_year() got an unexpected keyword argument 'country'
```

**Required fix:** replace `country_features_for()` with a project-specific
loader that calls `country_features_for_year(history, year)` once and passes the
entire returned mapping into `compute_match_features()`.

Concrete change:

```python
def country_feature_map_for_year(shared, history, year: int, notes: RuntimeNotes) -> dict[str, dict[str, float]]:
    if shared is None or not hasattr(shared, "country_features_for_year"):
        return {}
    try:
        return shared.country_features_for_year(history, year)
    except Exception as exc:
        notes.warn(f"Could not load country feature map for {year}: {exc}")
        return {}
```

Then:

```python
country_features = country_feature_map_for_year(shared, ctx.country_history, args.year, notes)
features = shared.compute_match_features(home, away, bundle.state, country_features, args.stage, match_date)
```

### 1.4 Calibration cannot work because training features and labels are not returned

**Severity:** Critical  
**Files:** `explain_match.py`, `predict_2026.py`

The report says:

```text
Historical calibration bucket unavailable because training features/labels were not returned.
Brier decomposition unavailable because training features/labels were not returned.
```

This is caused by `predict_2026.train_model()` returning only:

```python
# predict_2026.py:134
return model, state, X.columns.tolist()
```

and by `explain_match.py` discarding everything except the model. The
calibration code itself expects `bundle.train_X` and `bundle.train_y`:

```python
# explain_match.py:989-995
if bundle.train_X is None or bundle.train_y is None or not has_predict_proba(bundle.model):
    return "Historical calibration bucket unavailable because training features/labels were not returned."
```

**Required fix:** return training data from the training pipeline. Prefer a new
bundle-returning function:

```python
# predict_2026.py
@dataclass
class TrainingBundle:
    model: object
    state: object
    feature_names: list[str]
    train_X: pd.DataFrame
    train_y: np.ndarray
    country_history: dict
    country_features_2026: dict

def train_model_bundle(results_df, country_history) -> TrainingBundle:
    ...
    return TrainingBundle(
        model=model,
        state=state,
        feature_names=X.columns.tolist(),
        train_X=X,
        train_y=y,
        country_history=country_history,
        country_features_2026=country_features_for_year(country_history, 2022),
    )

def train_model(results_df, country_history):
    bundle = train_model_bundle(results_df, country_history)
    return bundle.model, bundle.state, bundle.feature_names
```

Then `explain_match.py` can populate `ModelBundle.train_X` and
`ModelBundle.train_y`.

### 1.5 `train_model_with_shared()` cannot work with the current `shared.fit_xgb_with_validation()`

**Severity:** Critical  
**File:** `explain_match.py`, `shared.py`

`shared.fit_xgb_with_validation()` requires `(model, X, y, label)`:

```python
# shared.py:286
def fit_xgb_with_validation(model, X: pd.DataFrame, y: np.ndarray, label: str = "model"):
```

But `explain_match.py` tries to call it with no arguments, a match dataframe, or
keyword aliases:

```python
# explain_match.py:556-568
func = shared.fit_xgb_with_validation
attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [((), {})]
if matches is not None:
    attempts.extend([((matches,), {}), ((), {"matches": matches}), ((), {"df": matches})])
result = flexible_call(func, attempts)
```

This fallback path is unusable for this project. It gives a false sense that the
script can train independently through `shared.py`.

**Required fix:** delete this fallback or replace it with a real call into a
project-level training function that builds `X` and `y` in chronological order.
Do not call `fit_xgb_with_validation()` directly from `explain_match.py`.

### 1.6 Counterfactuals mutate the already-invalid all-zero vector

**Severity:** Critical  
**File:** `explain_match.py`

Counterfactuals operate directly on `X`, not on state or country features:

```python
# explain_match.py:1044-1081
def apply_counterfactual(X: pd.DataFrame, scenario: str, home_is_underdog: bool):
    X_cf = X.copy()
    ...
    if scenario == "underdog_elo_plus_100":
        ...
        X_cf[col] = X_cf[col] + 100
```

Since `X` is currently all zeros, the counterfactuals are also operating on an
invalid baseline. Even after the zero bug is fixed, this approach creates
inconsistent features. For example, changing `elo_diff` without recomputing
`elo`, `elo_opponent`, and `elo_sum` creates a feature vector that could never
come from `compute_match_features()`.

The H2H counterfactual is especially broken:

```python
# explain_match.py:1065-1071
if "h2h" in norm or "head_to_head" in norm or "head2head" in norm:
    value = X_cf.iloc[0][col]
    if is_number(value):
        X_cf[col] = -float(value) if abs(float(value)) > 1 else 1 - float(value)
```

This can turn `h2h_matches=14` into `-14`, which is physically impossible.

**Required fix:** counterfactuals should clone the prediction state, modify
state or inputs, then call `compute_match_features()` again. Example:

```python
def build_features_from_bundle(ctx, bundle, country_features, match_date):
    feat = ctx.shared.compute_match_features(
        ctx.home, ctx.away, bundle.state, country_features, ctx.stage, match_date
    )
    return pd.DataFrame([feat]).reindex(columns=bundle.feature_names).fillna(0.0)

def counterfactual_underdog_elo_plus_100(ctx, bundle, country_features, match_date):
    cf_bundle = copy.deepcopy(bundle)
    underdog = ctx.home if predict_favorite_is_away(...) else ctx.away
    cf_bundle.state[underdog]["elo"] += 100
    return build_features_from_bundle(ctx, cf_bundle, country_features, match_date)
```

If direct feature perturbation is kept for sensitivity analysis, label it as
"feature perturbation", validate ranges, and update dependent columns together.

## 2. Important Issues

### 2.1 `explain_match.py` builds features before building/loading the model bundle

**Severity:** Important  
**File:** `explain_match.py`

The current execution order is:

```python
# explain_match.py:1385-1391
ctx.country_history = load_country_history(shared, root, notes)
home_country = country_features_for(shared, ctx.country_history, home, args.year, notes)
away_country = country_features_for(shared, ctx.country_history, away, args.year, notes)
ctx.matches = discover_match_data(root, args.matches_data, notes)
match_features = build_match_row(ctx, home_country, away_country)
bundle = build_model_bundle(ctx, match_features, args.model_path)
X = prepare_X(match_features, bundle.feature_names)
```

This is backwards for a stateful model. The feature builder needs state and
feature names from training. Build/load the model bundle first, then construct
the match row.

Recommended order:

```python
ctx.country_history = load_country_history(shared, root, notes)
ctx.matches = discover_match_data(root, args.matches_data, notes)
bundle = build_model_bundle(ctx, args.model_path)
country_features = shared.country_features_for_year(ctx.country_history, args.year)
match_features = build_match_features(ctx, bundle, country_features, match_date)
X = align_features_or_fail(match_features, bundle.feature_names)
```

### 2.2 Missing feature columns should be treated as schema errors

**Severity:** Important  
**File:** `explain_match.py`

Padding unknown columns with zero is sometimes reasonable for one-hot encoded
categoricals. It is not reasonable for this model's core numeric features:
`elo`, `elo_diff`, `form_win_rate`, `h2h_matches`, `wc_participations`,
`football_power_index`, and so on.

Replace:

```python
for name in feature_names:
    if name not in frame:
        frame[name] = 0.0
```

with:

```python
def align_features_or_fail(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in frame.columns]
    extra = [name for name in frame.columns if name not in feature_names]
    if missing:
        raise RuntimeError(
            "Feature schema mismatch. Missing trained features: "
            + ", ".join(missing[:25])
            + (" ..." if len(missing) > 25 else "")
        )
    if extra:
        # Extra columns are harmless once logged.
        pass
    return frame.loc[:, list(feature_names)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

If the project later adds one-hot columns, distinguish optional dummy columns
from mandatory continuous features.

### 2.3 SHAP values are raw-margin contributions, not probability-point contributions

**Severity:** Important  
**File:** `explain_match.py`

The SHAP path is:

```python
# explain_match.py:668-702
explainer = shap.TreeExplainer(bundle.model)
raw = explainer.shap_values(X)
...
values = np.asarray(raw[h][0]) - np.asarray(raw[a][0])
...
return values.astype(float), float(base_value), explainer
```

For XGBoost multiclass classifiers, `TreeExplainer` normally explains raw
margin/logit outputs unless configured otherwise. The home-minus-away contrast
can be useful, but the report should not imply these are probability-point
effects. The waterfall/force plot base value is a margin contrast, not
`P(home) - P(away)`.

Required wording change:

```text
Computed SHAP contrast on the model's raw multiclass margin:
home-win margin contribution minus away-win margin contribution.
The signs show local direction; magnitudes are not probability points.
```

After the zero-vector bug is fixed, verify additivity:

```python
margin = bundle.model.predict(X, output_margin=True)
expected_margin_diff = margin[0, home_idx] - margin[0, away_idx]
actual_margin_diff = base_value + shap_values.sum()
assert np.isclose(expected_margin_diff, actual_margin_diff, atol=1e-4)
```

### 2.4 Historical H2H display is correct for Brazil-Japan, but not integrated with the model input

**Severity:** Important  
**File:** `explain_match.py`

The report shows:

```text
Brazil 11 wins, 2 draws, Japan 1 wins (14 matches)
```

That comes from `h2h_record()`, which scans `data/results.csv` directly:

```python
# explain_match.py:858-894
mask = (
    (df[cols.home_team].astype(str).eq(home) & df[cols.away_team].astype(str).eq(away))
    | (df[cols.home_team].astype(str).eq(away) & df[cols.away_team].astype(str).eq(home))
)
...
return {"all_time": summarize(matches), ...}
```

For Brazil-Japan this historical section appears correct. But it is not the
same H2H value used by the model, because the model input row has
`h2h_matches=0.000` and `h2h_win_rate=0.000`. The report therefore contradicts
itself: the narrative says 14 matches, while the model explanation says the
model saw zero H2H matches.

Also, `h2h_record()` does not harmonize the dataframe columns before matching,
so aliases such as `Iran`/`IR Iran`, `South Korea`/`Korea Republic`, and
`Ivory Coast`/`Côte d'Ivoire` may miss matches.

Required fix:

```python
def discover_match_data(...):
    ...
    if shared is not None and hasattr(shared, "harmonize_columns"):
        df = shared.harmonize_columns(df, ["home_team", "away_team", "country"])
```

For explaining historical matches, also filter H2H to matches before the match
date:

```python
if cols.date and match_date is not None:
    matches = matches[matches[cols.date] < match_date]
```

### 2.5 World Cup history counts qualification years as final-tournament participations

**Severity:** Important  
**File:** `explain_match.py`

The generated Brazil report says:

```text
World Cup history: participations 46, titles 0, recent years [2018, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
```

This is wrong. Brazil has final-tournament participations around 22 plus 2026,
not 46. The bug is here:

```python
# explain_match.py:897-923
wc = df[df[cols.tournament].astype(str).str.contains("world cup", case=False, na=False)].copy()
...
years = sorted(dates.dt.year.dropna().astype(int).unique().tolist())
...
"participations": len(years) if years else None,
```

`str.contains("world cup")` includes `FIFA World Cup qualification`. Counting
unique calendar years then counts every qualifying campaign year as a
"participation". The titles calculation is also broken because `results.csv`
does not have a `stage` column, so Brazil's titles become `0`.

Required fix: use curated final-tournament participation data from
`collect_data.WC_PARTICIPANTS` and winners from `shared.WC_WINNERS` or
`collect_data.WC_WINNERS`. Do not infer participations from qualification
matches.

Concrete replacement:

```python
def world_cup_history_from_curated(team: str, through_year: int = 2026) -> dict[str, Any]:
    from collect_data import WC_PARTICIPANTS
    from shared import WC_WINNERS, harmonize_country

    team = harmonize_country(team)
    years = [
        int(year)
        for year, participants in WC_PARTICIPANTS.items()
        if int(year) <= through_year and team in {harmonize_country(t) for t in participants}
    ]
    titles = sum(
        1
        for year, winner in WC_WINNERS.items()
        if int(year) <= through_year and harmonize_country(winner) == team
    )
    return {
        "available": True,
        "participations": len(years),
        "years": years[-8:],
        "titles": titles,
    }
```

If 2026 should count before the tournament starts, add it explicitly from
`GROUP_2026_TEAMS`.

### 2.6 `predict_2026.train_model()` trains stage as a constant zero

**Severity:** Important  
**Files:** `predict_2026.py`, `shared.py`

During training:

```python
# predict_2026.py:86
rows.append(compute_features(ht, at, state, country_feature_cache[feature_year], 0, r['date']))
```

Every historical match is trained with `stage=0`, including historical World
Cup knockout matches. But predictions use stage values from 0 to 5:

```python
# predict_2026.py:281, 298, 308, 316, 330
r32_w = simulate_round(r32, "ROUND OF 32", 1, ...)
r16_w = simulate_round(r16, "ROUND OF 16", 2, ...)
qf_w = simulate_round(qf, "QUARTERFINALS", 3, ...)
sf_w = simulate_round(sf, "SEMIFINALS", 4, ...)
champion, ... = predict(..., stage=5, ...)
```

The model has no learned basis for interpreting stage values 1-5. Any split on
`stage` is accidental or out of distribution.

Required fix: either remove `stage` from the simple model, or compute historical
stage labels during training. The later `incremental_predictor.py` already has
a stage assignment system:

```python
# incremental_predictor.py:349-414
def stage_counts_for_year(...)
def assign_wc_stage_map(...)
```

Move this logic into `shared.py` and reuse it in `predict_2026.py`.

### 2.7 `neutral` and home advantage are ignored by the simple model

**Severity:** Important  
**Files:** `shared.py`, `predict_2026.py`

`shared.compute_match_features()` hard-codes:

```python
# shared.py:231
"stage": stage_num, "neutral": 1, "is_home": 0,
```

The Elo update function accepts `neutral` but ignores it:

```python
# shared.py:183
def update_elo(elo_a: float, elo_b: float, score_a: int, score_b: int, neutral: bool = True):
    ea = expected_score(elo_a, elo_b)
```

This means the simple model cannot learn actual home advantage, host advantage,
or neutral-site context from match history. That is a large omission for
international football.

Required fix:

```python
def compute_match_features(team, opponent, state, country_features, stage_num, match_date, neutral=True, is_home=True):
    ...
    return {
        ...
        "neutral": int(neutral),
        "is_home": int(is_home),
    }

def update_elo(elo_a, elo_b, score_a, score_b, neutral=True):
    home_adv = 0 if neutral else 50
    ea = expected_score(elo_a + home_adv, elo_b)
    ...
```

Training must pass `parse_bool(r["neutral"])` into both functions.

### 2.8 Random validation split is not appropriate for a chronological predictor

**Severity:** Important  
**Files:** `shared.py`, `predict_2026.py`

The feature generation itself is chronological and mostly leakage-safe: a row is
computed before that match updates state.

```python
# predict_2026.py:86-89
rows.append(compute_features(...))
labels.append(...)
state[ht]['elo'], state[at]['elo'] = update_elo(...)
```

However, validation is a random split:

```python
# shared.py:286-307
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)
```

For a model used to predict future matches, random validation mixes eras and is
optimistic. Later scripts correctly use time-based or LOWCO evaluation. The
simple `predict_2026.py` should report a chronological validation window or no
validation claim.

Required fix:

```python
def fit_xgb_time_split(model, X, y, dates, cutoff="2022-01-01"):
    train_mask = dates < pd.Timestamp(cutoff)
    val_mask = dates >= pd.Timestamp(cutoff)
    model.fit(X.loc[train_mask], y[train_mask], eval_set=[(X.loc[val_mask], y[val_mask])], verbose=False)
    ...
```

### 2.9 Country aliases are inconsistent across scripts

**Severity:** Important  
**Files:** `shared.py`, `match_predictor.py`, `incremental_predictor.py`,
`predict_2026.py`, `monte_carlo_2026.py`, `collect_data.py`

The project has multiple alias maps that disagree:

- `shared.py` maps `Iran` to `IR Iran`.
- `match_predictor.py` maps `IR Iran` to `Iran`.
- `incremental_predictor.py` maps `IR Iran` to `Iran`.
- `shared.py` maps `Cote d'Ivoire` to `Côte d'Ivoire`; `incremental_predictor.py`
  maps both variants to `Cote d'Ivoire`.
- `shared.py` collapses `East Germany` and `German DR` into `Germany`;
  `match_predictor.py` preserves `East Germany`.

This causes silent feature fragmentation, missing H2H records, and inconsistent
country joins.

Required fix: all scripts should import only:

```python
from shared import harmonize_country, harmonize_columns, NAME_ALIASES
```

Delete local `NAME_MAP`/`NAME_ALIASES` copies unless a script has a documented
reason to preserve historical entities separately. If preserving East Germany is
desired, make that a single project-wide decision.

### 2.10 `incremental_predictor.py` has a leakage bug in country features

**Severity:** Important  
**File:** `incremental_predictor.py`

The script excludes direct target columns from country features:

```python
# incremental_predictor.py:417-420
excluded = {"wc_year", "won_wc", "runner_up", "semifinalist", "finalist", "top4", "is_winner"}
feature_cols = [c for c in numeric_cols if c not in excluded]
```

But it does **not** exclude known leakage/post-tournament columns such as:

- `gdp_per_capita_vs_winner`
- `population_vs_winner`
- `total_goals_in_tournament`
- `avg_goals_per_match`

Those are explicitly dropped in `sota_analysis.py`, `backtest.py`, and
`match_predictor.py`, but not here. This matters because the README claims the
incremental predictor reaches 86.4% exact winner accuracy. That result may be
inflated by leaked country-level features.

Required fix:

```python
excluded = {
    "wc_year", "won_wc", "runner_up", "semifinalist", "finalist", "top4", "is_winner",
    "gdp_per_capita_vs_winner", "population_vs_winner",
    "total_goals_in_tournament", "avg_goals_per_match",
}
```

Then rerun the incremental backtest and update `README.md`.

## 3. Minor Issues

### 3.1 `explain_match.py` is too permissive and hides integration failures

The script uses broad `flexible_call()` guessing throughout:

```python
# explain_match.py:247-255
def flexible_call(func, attempts):
    ...
    except TypeError as exc:
        errors.append(str(exc))
        continue
```

For a one-repository tool, this is counterproductive. It turns contract
violations into warnings and fallbacks. The explanation script should use the
actual project APIs directly and fail loudly when the schema does not match.

### 3.2 `discover_match_data()` can choose the wrong CSV as the project grows

The current pattern list is broad:

```python
# explain_match.py:126-135
MATCH_DATA_PATTERNS = (
    "data/*match*.csv",
    "data/*result*.csv",
    "data/*game*.csv",
    "data/*.csv",
    ...
)
```

Today it finds `data/results.csv`, but a future `data/match_features.csv` could
be selected first and break historical context. Default directly to
`data/results.csv` for this project.

### 3.3 Titles in `world_cup_history()` cannot work with `results.csv`

`infer_match_columns()` looks for `stage`, but `data/results.csv` has only:

```text
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
```

Because there is no stage column, final wins cannot be inferred. Use
`shared.WC_WINNERS` or curated tournament data.

### 3.4 `causal_insights()` leaks absolute local paths into generated reports

The existing report includes absolute filesystem paths:

```text
`/var/mnt/DATA/Hermes/workspace/world-cup-predictors/output/sota/causal_dag.md`
```

Prefer project-relative paths for portability:

```python
display_path = path.relative_to(root) if path.is_relative_to(root) else path
```

### 3.5 `predict_2026.update_state()` always updates World Cup match counts

The helper is generic by name but does:

```python
# predict_2026.py:163-166
state[ha]['wc_matches'] += 1
state[hb]['wc_matches'] += 1
if sa > sb: state[ha]['wc_wins'] += 1
elif sa < sb: state[hb]['wc_wins'] += 1
```

That is safe only because current calls after training are for 2026 World Cup
matches. Rename it to `update_wc_state()` or pass `is_world_cup`.

### 3.6 `monte_carlo_2026.py` top-four output is a slot share, not a probability

The script prints:

```python
# monte_carlo_2026.py:433-438
top4 = Counter()
for d in [champions, runner_ups, thirds, fourths]:
    for team, count in d.items():
        top4[team] += count
for team, count in top4.most_common(15):
    pct = count / (N_SIMS * 4) * 100
```

If a team appears in the top four in 300 of 1000 simulations, its top-four
probability is 30%, not `300 / 4000 = 7.5%`. Use `count / N_SIMS`.

### 3.7 No tests exist

`tests/**` is empty. The zero-feature bug would have been caught by one
integration test asserting that Brazil-Japan features include non-zero Elo,
form, H2H, and country values.

## 4. Recommended Fix Architecture

### Goal

`explain_match.py` should use exactly the same model, feature schema, state, and
country feature snapshot as the prediction pipeline. The explanation script
should not independently guess how to build rows.

### Preferred architecture

Add a bundle-returning API to `predict_2026.py` and consume it from
`explain_match.py`.

#### Step 1: Add a bundle dataclass to `predict_2026.py`

```python
# predict_2026.py
from dataclasses import dataclass
from typing import Any

@dataclass
class PredictionBundle:
    model: Any
    state: Any
    feature_names: list[str]
    train_X: pd.DataFrame
    train_y: np.ndarray
    country_history: dict
    country_features: dict
```

#### Step 2: Factor training into `train_model_bundle()`

```python
def train_model_bundle(results_df, country_history, exclude_2026_wc=True) -> PredictionBundle:
    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if exclude_2026_wc:
        df = df[~((df["date"].dt.year == 2026) & (df["tournament"] == "FIFA World Cup"))]
    df = df.sort_values("date").reset_index(drop=True)

    state = make_initial_state()
    rows, labels, feature_dates = [], [], []
    country_feature_cache = {}
    active_wc_year = None
    active_wc_teams = set()

    for _, r in df.iterrows():
        ht, at = harmonize(r["home_team"]), harmonize(r["away_team"])
        hs, aw = r["home_score"], r["away_score"]
        if pd.isna(hs) or pd.isna(aw):
            continue

        feature_year = int(r["date"].year)
        if feature_year not in country_feature_cache:
            country_feature_cache[feature_year] = country_features_for_year(country_history, feature_year)

        neutral = parse_bool(r.get("neutral", True))
        stage = infer_stage_for_training_row(r)  # or 0 until proper stage labels exist
        rows.append(compute_features(ht, at, state, country_feature_cache[feature_year], stage, r["date"]))
        labels.append(0 if hs > aw else (1 if hs == aw else 2))
        feature_dates.append(r["date"])

        update_training_state(...)

    X = pd.DataFrame(rows).fillna(0.0)
    y = np.asarray(labels, dtype=int)
    model = xgb.XGBClassifier(...)
    model, metrics = fit_xgb_with_validation(model, X, y, label="XGBoost")

    return PredictionBundle(
        model=model,
        state=state,
        feature_names=X.columns.tolist(),
        train_X=X,
        train_y=y,
        country_history=country_history,
        country_features=country_features_for_year(country_history, 2022),
    )
```

Keep the existing `train_model()` wrapper so current scripts do not break:

```python
def train_model(results_df, country_history):
    bundle = train_model_bundle(results_df, country_history)
    return bundle.model, bundle.state, bundle.feature_names
```

#### Step 3: Add a preparation function for 2026 state

Right now `predict_2026.main()` mutates state after training:

```python
# predict_2026.py:209-226
completed = wc26[wc26['home_score'].notna()].sort_values('date')
for _, r in completed.iterrows():
    update_state(state, ...)
...
state[ht]['wc_participations'] += 1
```

This should be a shared function:

```python
def prepare_2026_state(results, bundle):
    state = bundle.state
    wc26 = results[(results["tournament"] == "FIFA World Cup") & (pd.to_datetime(results["date"]).dt.year == 2026)].copy()
    wc26["date"] = pd.to_datetime(wc26["date"])
    completed = wc26[wc26["home_score"].notna() & wc26["away_score"].notna()].sort_values("date")
    for _, r in completed.iterrows():
        update_state(state, r["home_team"], r["away_team"], int(r["home_score"]), int(r["away_score"]), r["date"])
    for teams in GROUP_2026_TEAMS.values():
        for team in teams:
            state[harmonize(team)]["wc_participations"] += 1
    return state
```

For explaining a future knockout matchup, decide explicitly whether the state
should include simulated group matches. `analyze_explain.py` already does that:

```python
# analyze_explain.py:214-224
for date, home, away, group in remaining_matches:
    probs = predict_probs(model, fl, home, away, state, cf, 0, date)
    ...
    update_state(state, home, away, sa, sb, date)
    apply_group_result(groups, group, home, away, sa, sb)
```

That logic should also be factored if `explain_match.py Brazil Japan --stage 1`
is intended to explain a predicted Round-of-32 matchup after simulated group
matches.

#### Step 4: `explain_match.py` should use one feature-building path

Replace `build_match_row()` with a strict project-specific function:

```python
def build_match_features_from_state(
    ctx: ExplanationContext,
    bundle: ModelBundle,
    country_features: dict[str, dict[str, float]],
    match_date: pd.Timestamp,
) -> pd.DataFrame:
    if bundle.state is None:
        raise RuntimeError("Prediction state is required for match explanations.")
    features = ctx.shared.compute_match_features(
        ctx.home,
        ctx.away,
        bundle.state,
        country_features,
        ctx.stage,
        match_date,
    )
    frame = pd.DataFrame([features])
    missing = [c for c in bundle.feature_names if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Feature schema mismatch: missing {missing[:20]}")
    return frame.loc[:, bundle.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

Add a CLI argument for date:

```python
parser.add_argument("--match-date", default="2026-06-29", help="Date used for rest-days and state cutoff.")
```

#### Step 5: Integration test

Add a test or smoke script that fails on all-zero model features:

```python
def test_explain_brazil_japan_features_not_zero():
    bundle = build_prediction_bundle_for_explain()
    X = build_explain_features("Brazil", "Japan", stage=1, match_date="2026-06-29", bundle=bundle)
    assert X.loc[0, "elo"] > 0
    assert X.loc[0, "elo_opponent"] > 0
    assert X.loc[0, "elo_sum"] > 2500
    assert X.loc[0, "h2h_matches"] >= 1
    assert X.loc[0, "wc_participations"] >= 20
```

### Minimal hotfix

If you want the smallest possible patch:

1. Change `load_model_from_predict_module()` so the `train_model()` tuple is not
   discarded.
2. Add `state` to `ModelBundle`.
3. Delete the current `country_features_for()` calls.
4. Build the match row by calling `shared.compute_match_features(home, away,
   bundle.state, country_features, stage, match_date)`.
5. Make missing trained features an exception.

Pseudo-patch:

```python
# explain_match.py
def load_bundle_from_predict_module(module, notes):
    import shared as _shared
    results_df = pd.read_csv(_shared.DATA_DIR / "results.csv")
    country_history = _shared.load_country_feature_history()
    result = module.train_model(results_df, country_history)
    if not (isinstance(result, tuple) and len(result) >= 3):
        raise RuntimeError("predict_2026.train_model must return (model, state, feature_names)")
    model, state, feature_names = result[:3]
    return ModelBundle(
        model=model,
        state=state,
        feature_names=list(feature_names),
        class_labels=list(getattr(model, "classes_", OUTCOME_LABELS)),
    )

def main(...):
    ...
    country_features = shared.country_features_for_year(ctx.country_history, args.year)
    bundle = load_bundle_from_predict_module(ctx.predict_2026, notes)
    match_features = pd.DataFrame([
        shared.compute_match_features(home, away, bundle.state, country_features, args.stage, pd.Timestamp(args.match_date))
    ])
    X = align_features_or_fail(match_features, bundle.feature_names)
```

This hotfix does not solve calibration unless `predict_2026.py` also returns
`X` and `y`, but it fixes the critical zero-vector bug.

## 5. Code Quality Assessment: `explain_match.py`

`explain_match.py` is ambitious and has useful report sections, but its current
engineering quality is poor for a model explanation tool.

**Strengths:**

- The report structure is useful: predictions, SHAP drivers, historical context,
  factor decomposition, counterfactuals, and runtime notes.
- It records warnings and runtime details, which helped identify the failure.
- It tries to normalize SHAP output shapes across SHAP/XGBoost versions.
- The H2H display logic is reasonable for simple canonical team names.

**Major quality problems:**

- It is "generic" in the wrong place. The repository has a known feature API,
  but the script guesses signatures instead of using it.
- It silently converts schema failures into zero-valued model inputs.
- It trains/loads a model separately from feature state, even though the model
  is stateful by design.
- It uses broad exception handling to continue after critical failures.
- It presents invalid outputs with polished formatting, making the problem easy
  to miss.
- It duplicates historical logic already present elsewhere and gets World Cup
  participation counts wrong.
- It does not have tests or even a basic invariant check that core model
  features are non-zero.

**Overall assessment:** not production-usable until the state and schema issues
are fixed. The file should be shortened and made stricter. A model explanation
tool should fail loudly if it cannot reconstruct the exact trained feature row.

## 6. Model Assessment

### 6.1 Simple `predict_2026.py` model

The simple model is directionally plausible but not fully sound.

**What is good:**

- Feature rows are generated chronologically before each match updates state,
  which avoids the most obvious form/Elo/H2H leakage.
- The state includes meaningful football signals: Elo, recent form, goals,
  rest days, H2H, World Cup experience, country features, and football
  tradition.
- The XGBoost objective is appropriate for three-class match outcome prediction:

```python
# predict_2026.py:128-131
objective='multi:softprob', num_class=3,
eval_metric='mlogloss', random_state=42, verbosity=0
```

**What is weak or wrong:**

- Validation is random, not chronological.
- `stage` is constant during training but varied during prediction.
- `neutral` and `is_home` are hard-coded and do not represent the match.
- Elo ignores home advantage.
- The training function returns no training labels/features for calibration.
- There is no model artifact, schema artifact, or state artifact saved to disk.
- The code is duplicated across `predict_2026.py`, `monte_carlo_2026.py`, and
  `analyze_explain.py`.

**Verdict:** usable as an exploratory script, not as a reliable explanatory or
forecasting pipeline.

### 6.2 `shared.compute_match_features()`

The feature engineering is conceptually reasonable but too implicit. It depends
on a very specific mutable state shape:

```python
state[team] = {
    "elo": ...,
    "form": ...,
    "goals_for": ...,
    "goals_against": ...,
    "last_match": ...,
    "h2h": ...,
    "wc_participations": ...,
    ...
}
```

This state shape is not represented by a dataclass or builder function in
`shared.py`, so other scripts re-create it manually. That is why
`explain_match.py` failed.

Required fix: add a shared state factory:

```python
def make_team_state():
    return {
        "elo": INITIAL_ELO,
        "form": [],
        "goals_for": [],
        "goals_against": [],
        "last_match": None,
        "h2h": defaultdict(lambda: {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0}),
        "wc_participations": 0,
        "wc_titles": 0,
        "wc_wins": 0,
        "wc_matches": 0,
    }
```

Then every script imports the same factory.

### 6.3 Data leakage

The project contains both safe and unsafe pieces.

Mostly safe:

- `predict_2026.train_model()` computes dynamic match features before updating
  state with the current match result.
- `collect_data.py` uses prior tournament history for fields like
  `wc_titles_before` and `wc_participations_before`.
- `sota_analysis.py`, `backtest.py`, and `match_predictor.py` explicitly drop
  direct leakage columns.

Unsafe or suspicious:

- `incremental_predictor.py` does not drop `gdp_per_capita_vs_winner`,
  `population_vs_winner`, `total_goals_in_tournament`, or
  `avg_goals_per_match` from its country feature vector.
- Random validation in the simple XGBoost model is not time-safe.
- Country feature enrichment includes manually curated Elo/FIFA values whose
  exact timestamp should be documented.
- The README's 86.4% incremental result should be treated as suspect until the
  country-feature leakage fix is applied and the backtest rerun.

### 6.4 Country aliases

Alias coverage is not complete enough because it is not centralized. The most
important fix is consistency, not adding an endless list of names. A single
project-wide `harmonize_country()` should be used everywhere, and tests should
cover:

- `Iran` and `IR Iran`
- `South Korea` and `Korea Republic`
- `North Korea` and `Korea DPR`
- `USA`, `United States`, `United States of America`
- `Ivory Coast`, `Cote d'Ivoire`, `Côte d'Ivoire`
- `Curacao`, `Curaçao`
- `UAE`, `United Arab Emirates`
- historical entities such as `West Germany`, `East Germany`, `USSR`, and
  `Yugoslavia`

### 6.5 Bracket simulation

The 48-team shape is broadly represented: 12 groups, top two plus eight
third-place teams, and a Round of 32. The helper in `shared.py` is a good start:

```python
# shared.py:378-417
def build_round_of_32(gw, gr, best_thirds):
    ...
    combo = tuple(sorted(third_by_group))
    slot_by_group = THIRD_PLACE_ALLOCATION_MATRIX[combo]
    ...
```

Remaining issues:

- Group standings tie-breaks only use points, goal difference, goals for, then
  team name. FIFA tie-breaks are more detailed.
- Third-place ranking also ignores full FIFA tie-breaks.
- Predicted scores are synthetic fixed scores (`2-1`, `1-1`) rather than sampled
  goal distributions, so group tables are overconfident and low-variance.
- Knockout probabilities are not renormalized after removing draw outcomes.
- The bracket is hard-coded and should be verified against the final FIFA match
  schedule.

**Verdict:** acceptable for exploratory simulation, but not robust enough to
support high-confidence claims without schedule/tie-break tests.

## 7. Concrete Tests to Add

Add at least these tests after refactoring:

```python
def test_explain_uses_nonzero_project_features():
    X = build_explain_features("Brazil", "Japan", stage=1, match_date="2026-06-29")
    assert X.loc[0, "elo"] > 1000
    assert X.loc[0, "elo_opponent"] > 1000
    assert X.loc[0, "elo_sum"] > 2500
    assert X.loc[0, "h2h_matches"] >= 1
    assert X.loc[0, "stage"] == 1

def test_explain_feature_schema_matches_training_schema():
    bundle = train_or_load_prediction_bundle()
    X = build_explain_features("Brazil", "Japan", bundle=bundle)
    assert list(X.columns) == bundle.feature_names

def test_world_cup_history_counts_final_tournaments_only():
    brazil = world_cup_history_from_curated("Brazil", through_year=2022)
    assert brazil["participations"] == 22
    assert brazil["titles"] == 5

def test_country_feature_api_returns_team_map():
    history = load_country_feature_history()
    features = country_features_for_year(history, 2022)
    assert "Brazil" in features
    assert "Japan" in features
    assert features["Brazil"]["football_tradition"] > features["Japan"]["football_tradition"]

def test_incremental_country_features_drop_leakage_columns():
    _, feature_cols = build_country_feature_lookup(pd.read_csv("data/world_cup_predictors_dataset.csv"))
    forbidden = {
        "gdp_per_capita_vs_winner",
        "population_vs_winner",
        "total_goals_in_tournament",
        "avg_goals_per_match",
    }
    assert forbidden.isdisjoint(feature_cols)
```

## 8. Bottom Line

`explain_match.py` currently does not explain the prediction model. It explains
an all-zero fallback vector caused by a failed call to `compute_match_features()`
and by discarded training state. The fix is architectural, not cosmetic:
prediction and explanation must share a trained bundle containing model, state,
feature names, training data, and country features. Once that is in place, SHAP
and calibration can become meaningful. Until then, the generated probabilities,
SHAP drivers, counterfactuals, and factor decomposition should not be trusted.
