#!/usr/bin/env python3
"""Permutation-importance feature selection for the World Cup predictor.

Trains the model the same way ``backtest_2026_wc.py`` does, then runs
permutation importance on the chronological holdout. Features whose mean
importance is statistically indistinguishable from zero (within one std-dev of
the per-feature shuffle noise across repeats) are flagged as droppable.

This is an *analysis* tool: it does not mutate the production feature set, it
prints a ranked report and a suggested drop-list so the change can be made
deliberately. Many country-demographic columns are expected to be noise.
"""
import warnings

import numpy as np
import pandas as pd

from shared import (
    DATA_DIR,
    ODDS_FEATURE_COLUMNS,
    load_country_feature_history,
    load_betting_odds,
)
from backtest_2026_wc import train_model


def main(n_repeats: int = 10, seed: int = 42):
    warnings.simplefilter("ignore")
    results_df = pd.read_csv(DATA_DIR / "results.csv")
    country_history = load_country_feature_history()
    odds = load_betting_odds()

    # train_model returns (model, state, feature_names, poisson, alpha); we
    # re-build the feature matrix here to evaluate importance on the holdout.
    from backtest_2026_wc import train_model as _tm  # noqa: F401

    # Rebuild features via the same path used in training.
    import backtest_2026_wc as bt

    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[~((df["date"].dt.year == 2026) & (df["tournament"] == "FIFA World Cup"))]
    df = df.sort_values("date").reset_index(drop=True)

    model, state, feature_names, _poisson, _alpha = train_model(
        results_df, country_history, odds=odds
    )

    # Reconstruct X / y / dates by replaying the training loop.
    from shared import (
        country_features_for_year,
        compute_match_features,
        harmonize_country,
        apply_match_to_state,
        infer_world_cup_stage_map,
        make_team_state,
        parse_bool,
        finalize_world_cup_history,
        finalize_feature_frame,
        odds_features_for_match,
    )
    from collections import defaultdict

    state = defaultdict(make_team_state)
    rows, labels, dates = [], [], []
    cache = {}
    active_year, active_teams = None, set()
    stage_map = infer_world_cup_stage_map(df)
    for _, r in df.iterrows():
        ht = harmonize_country(r["home_team"]); at = harmonize_country(r["away_team"])
        hs, aw = r["home_score"], r["away_score"]
        if pd.isna(hs) or pd.isna(aw):
            continue
        hs, aw = int(hs), int(aw)
        fy = int(r["date"].year)
        if fy not in cache:
            cache[fy] = country_features_for_year(country_history, fy)
        is_wc = r["tournament"] == "FIFA World Cup"
        if active_year is not None and (not is_wc or fy != active_year):
            finalize_world_cup_history(state, active_year, active_teams)
            active_year, active_teams = None, set()
        neutral = parse_bool(r.get("neutral", True))
        stage = stage_map.get(int(r.name), 0) if is_wc else 0
        orow = odds_features_for_match(odds, r["date"], ht, at)
        rows.append(compute_match_features(ht, at, state, cache[fy], stage, r["date"],
                                           neutral=neutral, is_home=not neutral, odds_row=orow))
        labels.append(0 if hs > aw else (1 if hs == aw else 2))
        dates.append(r["date"])
        if is_wc:
            if active_year is None:
                active_year = fy
            active_teams.update([ht, at])
        apply_match_to_state(state, ht, at, hs, aw, r["date"], neutral=neutral, is_world_cup=is_wc)
    if active_year is not None:
        finalize_world_cup_history(state, active_year, active_teams)

    X = finalize_feature_frame(rows)[feature_names]
    y = np.array(labels)
    order = pd.Series(pd.to_datetime(dates, errors="coerce")).sort_values().index
    split = max(1, int(len(order) * 0.8))
    val_idx = order[split:]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    from sklearn.metrics import log_loss

    def score(frame):
        p = model.predict_proba(frame)
        return log_loss(y_val, p, labels=[0, 1, 2])

    base = score(X_val)
    rng = np.random.default_rng(seed)
    importances = {}
    for col in feature_names:
        deltas = []
        for _ in range(n_repeats):
            shuffled = X_val.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            deltas.append(score(shuffled) - base)
        importances[col] = (float(np.mean(deltas)), float(np.std(deltas)))

    ranked = sorted(importances.items(), key=lambda kv: -kv[1][0])
    print("\n" + "=" * 70)
    print("  PERMUTATION IMPORTANCE (holdout log-loss increase when shuffled)")
    print("=" * 70)
    print(f"  Baseline holdout log-loss: {base:.4f}\n")
    print(f"  {'Feature':<28s} {'Mean Δ':>10s} {'Std':>8s}")
    print("  " + "-" * 50)
    droppable = []
    for col, (mean, std) in ranked:
        flag = ""
        if mean <= std and mean < 0.0005:
            flag = "  <- noise (droppable)"
            droppable.append(col)
        print(f"  {col:<28s} {mean:>10.5f} {std:>8.5f}{flag}")

    print("\n  Suggested drop-list (importance within noise):")
    print("   ", ", ".join(droppable) if droppable else "(none)")
    print(f"\n  Odds features: {', '.join(ODDS_FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
