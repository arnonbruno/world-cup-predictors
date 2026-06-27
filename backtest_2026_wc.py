"""
Walk-forward backtest of the World Cup 2026 prediction model.

For each completed 2026 WC match in chronological order:
1. Use current state (Elo, form, H2H, etc.) to predict
2. Compare prediction against actual result
3. Update state with the real result
4. Move to next match

Reports: accuracy, log-loss, Brier score, per-stage breakdown, and per-match details.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from shared import (
    INITIAL_ELO,
    DATA_DIR,
    load_country_feature_history,
    load_betting_odds,
    odds_features_for_match,
    country_features_for_year,
    compute_match_features,
    harmonize_country,
    fit_xgb_with_validation,
    apply_match_to_state,
    infer_world_cup_stage_map,
    make_team_state,
    parse_bool,
    finalize_world_cup_history,
    finalize_feature_frame,
    prepare_prediction_frame,
    sample_weights,
    fit_dixon_coles,
    blend_probabilities,
    GROUP_2026_TEAMS,
    WC2026_STAGE_TO_TRAIN,
)
from collections import defaultdict

# ─── Constants ───────────────────────────────────────────────────────────────
RESULT_LABELS = ["Home win", "Draw", "Away win"]
WC_TEAM_SET = {harmonize_country(t) for teams in GROUP_2026_TEAMS.values() for t in teams}


def make_initial_state():
    return defaultdict(make_team_state)


def _tune_blend_alpha(xgb_model, poisson_model, feature_names, X_val, y_val,
                      val_meta):
    """Pick the XGB/Poisson blend weight that minimizes holdout log-loss."""
    from sklearn.metrics import log_loss as _ll

    p_xgb = xgb_model.predict_proba(X_val)
    p_pois = np.array([
        poisson_model.outcome_probs(h, a, neutral=neu)
        for (h, a, neu) in val_meta
    ])
    best_alpha, best_loss = 1.0, np.inf
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = np.array([
            blend_probabilities(p_xgb[i], p_pois[i], alpha) for i in range(len(p_xgb))
        ])
        try:
            loss = _ll(y_val, blended, labels=[0, 1, 2])
        except Exception:
            continue
        if loss < best_loss:
            best_loss, best_alpha = loss, alpha
    return float(best_alpha), float(best_loss)


def train_model(results_df, country_history, odds=None):
    """Train XGBoost + Dixon-Coles on all pre-2026-WC matches.

    Adds: bookmaker implied-probability features, time-decay + draw sample
    weighting, isotonic calibration on the chronological holdout, and a
    Dixon-Coles Poisson member whose blend weight is tuned on that same holdout.
    """
    import xgboost as xgb

    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[~((df["date"].dt.year == 2026) & (df["tournament"] == "FIFA World Cup"))]
    df = df.sort_values("date").reset_index(drop=True)

    state = make_initial_state()
    rows, labels, feature_dates = [], [], []
    match_meta = []  # (home, away, neutral) aligned with rows, for Poisson blend
    country_feature_cache = {}
    active_wc_year = None
    active_wc_teams = set()
    # Stage labels (0=group .. 4=final) for historical WC matches, so the model
    # actually learns what ``stage`` means instead of always seeing a constant 0.
    wc_stage_by_index = infer_world_cup_stage_map(df)

    for _, r in df.iterrows():
        ht = harmonize_country(r["home_team"])
        at = harmonize_country(r["away_team"])
        hs, aw = r["home_score"], r["away_score"]
        if pd.isna(hs) or pd.isna(aw):
            continue
        hs, aw = int(hs), int(aw)

        feature_year = int(r["date"].year)
        if feature_year not in country_feature_cache:
            country_feature_cache[feature_year] = country_features_for_year(country_history, feature_year)

        is_world_cup = r["tournament"] == "FIFA World Cup"
        if active_wc_year is not None and (not is_world_cup or feature_year != active_wc_year):
            finalize_world_cup_history(state, active_wc_year, active_wc_teams)
            active_wc_year, active_wc_teams = None, set()

        neutral = parse_bool(r.get("neutral", True))
        is_home = not neutral  # home side of a non-neutral fixture has home advantage
        stage = wc_stage_by_index.get(int(r.name), 0) if is_world_cup else 0
        odds_row = odds_features_for_match(odds, r["date"], ht, at)
        rows.append(compute_match_features(
            ht, at, state, country_feature_cache[feature_year], stage, r["date"],
            neutral=neutral, is_home=is_home, odds_row=odds_row,
        ))
        labels.append(0 if hs > aw else (1 if hs == aw else 2))
        feature_dates.append(r["date"])
        match_meta.append((ht, at, neutral))

        if is_world_cup:
            if active_wc_year is None:
                active_wc_year = feature_year
            active_wc_teams.update([ht, at])

        # Single shared state update (Elo, form, goals, H2H, momentum, fatigue).
        apply_match_to_state(state, ht, at, hs, aw, r["date"],
                             neutral=neutral, is_world_cup=is_world_cup)

    if active_wc_year is not None:
        finalize_world_cup_history(state, active_wc_year, active_wc_teams)

    X = finalize_feature_frame(rows)
    y = np.array(labels)
    weights = sample_weights(y, feature_dates)
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, verbosity=0,
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
    )
    model, _ = fit_xgb_with_validation(
        model, X, y, label="XGBoost", dates=feature_dates,
        sample_weight=weights, calibrate=True,
    )

    # Dixon-Coles Poisson member + blend-weight tuning on the chronological tail.
    print("  Fitting Dixon-Coles Poisson goal model...")
    poisson_model = fit_dixon_coles(results_df)
    order = pd.Series(pd.to_datetime(feature_dates, errors="coerce")).sort_values().index
    split = max(1, int(len(order) * 0.8))
    val_idx = order[split:]
    X_val = X.iloc[val_idx]
    y_val = y[val_idx]
    val_meta = [match_meta[i] for i in val_idx]
    alpha, blend_loss = _tune_blend_alpha(model, poisson_model, X.columns.tolist(),
                                          X_val, y_val, val_meta)
    print(f"  Tuned blend alpha (XGB weight) = {alpha:.2f}  (holdout log-loss {blend_loss:.3f})")
    return model, state, X.columns.tolist(), poisson_model, alpha


def prepare_2026_state(state, results_df):
    """Update state with completed 2026 WC group matches before backtest starts."""
    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    wc26 = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)].copy()

    # Add WC participations for all qualified teams
    for teams in GROUP_2026_TEAMS.values():
        for team in teams:
            ht = harmonize_country(team)
            state[ht]["wc_participations"] += 1

    return state


def predict_match(model, feature_names, home, away, state, cf, stage, date,
                  neutral=True, is_home=False, odds=None,
                  poisson_model=None, alpha=1.0):
    """Predict a single match. Returns (predicted_label_idx, probs[3]).

    Blends the (calibrated) XGBoost probabilities with the Dixon-Coles Poisson
    member when one is supplied, then applies the knockout draw-renormalization.
    """
    odds_row = odds_features_for_match(odds, date, home, away)
    feat = compute_match_features(home, away, state, cf, stage, date,
                                  neutral=neutral, is_home=is_home, odds_row=odds_row)
    X = prepare_prediction_frame(feat, feature_names)
    probs = np.asarray(model.predict_proba(X)[0], dtype=float)

    if poisson_model is not None and alpha < 1.0:
        p_pois = poisson_model.outcome_probs(home, away, neutral=neutral)
        probs = blend_probabilities(probs, p_pois, alpha)

    # Knockout: renormalize away the draw probability.
    if stage > 0:
        total = probs[0] + probs[2]
        if total > 0:
            probs = np.array([probs[0] / total, 0.0, probs[2] / total])

    predicted = int(np.argmax(probs))
    return predicted, probs


def actual_result(home_score, away_score):
    """Return label index: 0=home win, 1=draw, 2=away win."""
    if home_score > away_score:
        return 0
    elif home_score == away_score:
        return 1
    else:
        return 2


def stage_from_tournament_round(tournament_str, home_team, away_team):
    """Infer the *trained* stage code (0..4) from a tournament/round string.

    The historical training data only contains stages 0..4 (group .. final), so
    the 2026 bracket's richer R32/R16/QF/SF/Final scheme is collapsed onto that
    range via ``WC2026_STAGE_TO_TRAIN``. Returning 5 or 6 (as before) asked the
    model to predict on a ``stage`` value it had never seen during training.
    """
    t = str(tournament_str).lower()
    if "group" in t:
        return WC2026_STAGE_TO_TRAIN["group"]
    if "round of 32" in t or "r32" in t:
        return WC2026_STAGE_TO_TRAIN["round_of_32"]
    if "round of 16" in t or "r16" in t:
        return WC2026_STAGE_TO_TRAIN["round_of_16"]
    if "quarter" in t or "qf" in t:
        return WC2026_STAGE_TO_TRAIN["quarterfinal"]
    if "semi" in t or "sf" in t:
        return WC2026_STAGE_TO_TRAIN["semifinal"]
    if "third" in t or "3rd" in t:
        return WC2026_STAGE_TO_TRAIN["third_place"]
    if "final" in t and "semi" not in t:
        return WC2026_STAGE_TO_TRAIN["final"]
    return 0


def run_backtest():
    print("=" * 70)
    print("  WORLD CUP 2026 WALK-FORWARD BACKTEST")
    print("=" * 70)
    print()

    # Load data
    results_df = pd.read_csv(DATA_DIR / "results.csv")
    country_history = load_country_feature_history()
    odds = load_betting_odds()
    print(f"Loaded bookmaker odds for {len(odds) // 2} fixtures.")

    # Train model on all pre-2026-WC data
    print("Training model on historical data (excluding 2026 WC)...")
    model, state, feature_names, poisson_model, alpha = train_model(
        results_df, country_history, odds=odds,
    )
    print(f"  Trained on {sum(1 for _, r in results_df.iterrows() if r['home_score'] == r['home_score'])} matches")
    print()

    # Prepare 2026 state
    state = prepare_2026_state(state, results_df)

    # Get country features for 2022 (latest available)
    cf = country_features_for_year(country_history, 2022)

    # Get all 2026 WC matches with scores, sorted chronologically
    results_df["date"] = pd.to_datetime(results_df["date"])
    wc26 = results_df[
        (results_df["tournament"] == "FIFA World Cup") & (results_df["date"].dt.year == 2026)
    ].copy()
    wc26 = wc26.sort_values("date").reset_index(drop=True)
    completed = wc26[wc26["home_score"].notna() & wc26["away_score"].notna()].copy()

    print(f"Backtesting {len(completed)} completed matches...")
    print()

    # Walk forward
    results = []
    correct = 0
    total = 0
    probs_list = []
    actuals = []

    for idx, r in completed.iterrows():
        home = harmonize_country(r["home_team"])
        away = harmonize_country(r["away_team"])
        hs = int(r["home_score"])
        aw = int(r["away_score"])
        date = r["date"]
        stage = stage_from_tournament_round(r.get("tournament", ""), home, away)
        neutral = parse_bool(r.get("neutral", True))
        is_home = not neutral

        # Predict BEFORE updating state with this match's result
        predicted_idx, probs = predict_match(
            model, feature_names, home, away, state, cf, stage, date,
            neutral=neutral, is_home=is_home,
            odds=odds, poisson_model=poisson_model, alpha=alpha,
        )
        actual_idx = actual_result(hs, aw)

        is_correct = predicted_idx == actual_idx
        if is_correct:
            correct += 1
        total += 1

        probs_list.append(probs)
        actuals.append(actual_idx)

        # Record
        predicted_label = RESULT_LABELS[predicted_idx]
        actual_label = RESULT_LABELS[actual_idx]
        confidence = float(probs[predicted_idx])
        actual_prob = float(probs[actual_idx])

        results.append({
            "date": date.strftime("%Y-%m-%d"),
            "home": home,
            "away": away,
            "score": f"{hs}-{aw}",
            "stage": stage,
            "predicted": predicted_label,
            "actual": actual_label,
            "correct": is_correct,
            "confidence": confidence,
            "actual_prob": actual_prob,
            "p_home": float(probs[0]),
            "p_draw": float(probs[1]),
            "p_away": float(probs[2]),
        })

        # Update state with the actual result (Elo, form, goals, H2H, momentum,
        # fatigue) through the single shared updater. 2026 WC matches count as WC.
        apply_match_to_state(state, home, away, hs, aw, date,
                             neutral=neutral, is_world_cup=True)

    # ─── Metrics ────────────────────────────────────────────────────────────
    probs_arr = np.array(probs_list)
    actuals_arr = np.array(actuals)

    # Accuracy
    accuracy = correct / total

    # Log-loss
    eps = 1e-15
    log_losses = []
    for i in range(total):
        p = max(min(probs_arr[i][actuals_arr[i]], 1 - eps), eps)
        log_losses.append(-np.log(p))
    avg_log_loss = np.mean(log_losses)

    # Brier score (multiclass)
    brier = 0
    for i in range(total):
        one_hot = np.zeros(3)
        one_hot[actuals_arr[i]] = 1.0
        brier += np.mean((probs_arr[i] - one_hot) ** 2)
    brier /= total

    # ─── Report ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"  Total matches:  {total}")
    print(f"  Correct:        {correct}")
    print(f"  Accuracy:       {accuracy:.1%}")
    print(f"  Log-loss:       {avg_log_loss:.4f}")
    print(f"  Brier score:    {brier:.4f}")
    print()

    # Per-stage breakdown
    stage_names = {0: "Group Stage", 1: "Round of 32", 2: "Round of 16",
                   3: "Quarterfinals", 4: "Semifinals", 5: "Third Place", 6: "Final"}
    stage_results = {}
    for r in results:
        s = r["stage"]
        if s not in stage_results:
            stage_results[s] = {"correct": 0, "total": 0, "log_losses": []}
        stage_results[s]["total"] += 1
        if r["correct"]:
            stage_results[s]["correct"] += 1
        p = max(min(r["actual_prob"], 1 - eps), eps)
        stage_results[s]["log_losses"].append(-np.log(p))

    print("  PER-STAGE BREAKDOWN:")
    print(f"  {'Stage':<20s} {'Correct':>8s} {'Total':>6s} {'Acc':>8s} {'LogLoss':>9s}")
    print("  " + "-" * 55)
    for s in sorted(stage_results.keys()):
        sr = stage_results[s]
        acc = sr["correct"] / sr["total"] if sr["total"] > 0 else 0
        ll = np.mean(sr["log_losses"]) if sr["log_losses"] else 0
        name = stage_names.get(s, f"Stage {s}")
        print(f"  {name:<20s} {sr['correct']:>8d} {sr['total']:>6d} {acc:>8.1%} {ll:>9.4f}")
    print()

    # Confidence buckets
    print("  CALIBRATION (predicted confidence vs actual accuracy):")
    buckets = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for lo, hi in buckets:
        bucket = [r for r in results if lo <= r["confidence"] < hi]
        if bucket:
            bucket_acc = sum(1 for r in bucket if r["correct"]) / len(bucket)
            avg_conf = np.mean([r["confidence"] for r in bucket])
            print(f"    {lo:.0%}-{hi:.0%}: n={len(bucket):3d}, avg_conf={avg_conf:.1%}, actual_acc={bucket_acc:.1%}")
    print()

    # Upsets (model was wrong with high confidence)
    upsets = [r for r in results if not r["correct"] and r["confidence"] >= 0.6]
    if upsets:
        print(f"  UPSETS (wrong with ≥60% confidence): {len(upsets)}")
        for r in sorted(upsets, key=lambda x: -x["confidence"]):
            print(f"    {r['date']}: {r['home']} {r['score']} {r['away']} — "
                  f"Predicted {r['predicted']} ({r['confidence']:.1%}), Actual {r['actual']}")
        print()

    # Correct high-confidence predictions
    nailed = [r for r in results if r["correct"] and r["confidence"] >= 0.7]
    if nailed:
        print(f"  HIGH-CONFIDENCE CORRECT (≥70%): {len(nailed)}")
        for r in sorted(nailed, key=lambda x: -x["confidence"])[:10]:
            print(f"    {r['date']}: {r['home']} {r['score']} {r['away']} — "
                  f"{r['predicted']} ({r['confidence']:.1%})")
        print()

    # Full match log
    print("=" * 70)
    print("  FULL MATCH LOG")
    print("=" * 70)
    print()
    print(f"  {'Date':<12s} {'Match':<35s} {'Score':>5s} {'Predicted':<12s} {'Actual':<12s} {'OK?':>3s} {'Conf':>6s}")
    print("  " + "-" * 85)
    for r in results:
        match_str = f"{r['home']} vs {r['away']}"
        ok = "✅" if r["correct"] else "❌"
        print(f"  {r['date']:<12s} {match_str:<35s} {r['score']:>5s} {r['predicted']:<12s} {r['actual']:<12s} {ok:>3s} {r['confidence']:>6.1%}")

    print()
    print(f"Final accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"Log-loss: {avg_log_loss:.4f} | Brier: {brier:.4f}")

    return results


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        run_backtest()
