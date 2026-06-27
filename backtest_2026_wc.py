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
    country_features_for_year,
    compute_match_features,
    harmonize_country,
    fit_xgb_with_validation,
    update_elo,
    expected_score,
    parse_bool,
    finalize_world_cup_history,
    GROUP_2026_TEAMS,
)
from collections import defaultdict

# ─── Constants ───────────────────────────────────────────────────────────────
RESULT_LABELS = ["Home win", "Draw", "Away win"]
WC_TEAM_SET = {harmonize_country(t) for teams in GROUP_2026_TEAMS.values() for t in teams}


def make_initial_state():
    return defaultdict(lambda: {
        "elo": INITIAL_ELO, "form": [], "goals_for": [], "goals_against": [],
        "last_match": None,
        "h2h": defaultdict(lambda: {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0}),
        "wc_participations": 0, "wc_titles": 0, "wc_wins": 0, "wc_matches": 0,
    })


def train_model(results_df, country_history):
    """Train XGBoost on all pre-2026-WC matches, same as predict_2026.py."""
    import xgboost as xgb

    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[~((df["date"].dt.year == 2026) & (df["tournament"] == "FIFA World Cup"))]
    df = df.sort_values("date").reset_index(drop=True)

    state = make_initial_state()
    rows, labels, feature_dates = [], [], []
    country_feature_cache = {}
    active_wc_year = None
    active_wc_teams = set()

    for _, r in df.iterrows():
        ht = harmonize_country(r["home_team"])
        at = harmonize_country(r["away_team"])
        hs, aw = r["home_score"], r["away_score"]
        if pd.isna(hs) or pd.isna(aw):
            continue

        feature_year = int(r["date"].year)
        if feature_year not in country_feature_cache:
            country_feature_cache[feature_year] = country_features_for_year(country_history, feature_year)

        neutral = parse_bool(r.get("neutral", True))
        stage = 0  # historical matches: stage unknown, use 0
        rows.append(compute_match_features(ht, at, state, country_feature_cache[feature_year], stage, r["date"]))

        if hs > aw:
            labels.append(0)
        elif hs == aw:
            labels.append(1)
        else:
            labels.append(2)

        feature_dates.append(r["date"])

        # Update state
        ea = expected_score(state[ht]["elo"], state[at]["elo"])
        state[ht]["elo"], state[at]["elo"] = update_elo(state[ht]["elo"], state[at]["elo"], int(hs), int(aw), neutral)

        is_world_cup = "world cup" in str(r.get("tournament", "")).lower() and "qualif" not in str(r.get("tournament", "")).lower()
        if is_world_cup:
            if active_wc_year is None:
                active_wc_year = feature_year
            active_wc_teams.update([ht, at])

        # Update form, H2H, goals
        for team, opp, gf, ga, is_home in [(ht, at, hs, aw, True), (at, ht, aw, hs, False)]:
            result = 0.5 if hs == aw else (1.0 if (gf > ga) == is_home else 0.0)
            state[team]["form"].append(result)
            state[team]["form"] = state[team]["form"][-10:]
            state[team]["goals_for"].append(gf)
            state[team]["goals_for"] = state[team]["goals_for"][-10:]
            state[team]["goals_against"].append(ga)
            state[team]["goals_against"] = state[team]["goals_against"][-10:]
            state[team]["last_match"] = r["date"]

        # H2H
        h2h_key = (ht, at)
        state[ht]["h2h"][h2h_key]["matches"] += 1
        state[at]["h2h"][(at, ht)]["matches"] += 1
        if hs > aw:
            state[ht]["h2h"][h2h_key]["wins"] += 1
            state[at]["h2h"][(at, ht)]["losses"] += 1
        elif hs == aw:
            state[ht]["h2h"][h2h_key]["draws"] += 1
            state[at]["h2h"][(at, ht)]["draws"] += 1
        else:
            state[ht]["h2h"][h2h_key]["losses"] += 1
            state[at]["h2h"][(at, ht)]["wins"] += 1
        state[ht]["h2h"][h2h_key]["gf"] += hs
        state[ht]["h2h"][h2h_key]["ga"] += aw
        state[at]["h2h"][(at, ht)]["gf"] += aw
        state[at]["h2h"][(at, ht)]["ga"] += hs

        if is_world_cup:
            for t in [ht, at]:
                state[t]["wc_matches"] += 1
            if hs > aw:
                state[ht]["wc_wins"] += 1
            elif hs < aw:
                state[at]["wc_wins"] += 1

    if active_wc_year is not None:
        finalize_world_cup_history(state, active_wc_year, active_wc_teams)

    X = pd.DataFrame(rows).fillna(0)
    y = np.array(labels)
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, verbosity=0,
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
    )
    model, _ = fit_xgb_with_validation(model, X, y, label="XGBoost")
    return model, state, X.columns.tolist()


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


def predict_match(model, feature_names, home, away, state, cf, stage, date):
    """Predict a single match. Returns (predicted_label_idx, probs[3])."""
    feat = compute_match_features(home, away, state, cf, stage, date)
    X = pd.DataFrame([feat])[feature_names].fillna(0)
    probs = model.predict_proba(X)[0]

    # Knockout: renormalize
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
    """Infer stage code from tournament field or context."""
    t = str(tournament_str).lower()
    if "group" in t:
        return 0
    if "round of 32" in t or "r32" in t:
        return 1
    if "round of 16" in t or "r16" in t:
        return 2
    if "quarter" in t or "qf" in t:
        return 3
    if "semi" in t or "sf" in t:
        return 4
    if "third" in t or "3rd" in t:
        return 5
    if "final" in t and "semi" not in t:
        return 6
    # Default: if both teams in WC set, assume group stage
    return 0


def run_backtest():
    print("=" * 70)
    print("  WORLD CUP 2026 WALK-FORWARD BACKTEST")
    print("=" * 70)
    print()

    # Load data
    results_df = pd.read_csv(DATA_DIR / "results.csv")
    country_history = load_country_feature_history()

    # Train model on all pre-2026-WC data
    print("Training model on historical data (excluding 2026 WC)...")
    model, state, feature_names = train_model(results_df, country_history)
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

        # Predict BEFORE updating state with this match's result
        predicted_idx, probs = predict_match(model, feature_names, home, away, state, cf, stage, date)
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

        # Update state with actual result
        ea = expected_score(state[home]["elo"], state[away]["elo"])
        state[home]["elo"], state[away]["elo"] = update_elo(
            state[home]["elo"], state[away]["elo"], hs, aw, neutral
        )

        # Update form, H2H, goals
        for team, opp, gf, ga, is_home in [(home, away, hs, aw, True), (away, home, aw, hs, False)]:
            result = 0.5 if hs == aw else (1.0 if (gf > ga) == is_home else 0.0)
            state[team]["form"].append(result)
            state[team]["form"] = state[team]["form"][-10:]
            state[team]["goals_for"].append(gf)
            state[team]["goals_for"] = state[team]["goals_for"][-10:]
            state[team]["goals_against"].append(ga)
            state[team]["goals_against"] = state[team]["goals_against"][-10:]
            state[team]["last_match"] = date

        # H2H
        h2h_key = (home, away)
        state[home]["h2h"][h2h_key]["matches"] += 1
        state[away]["h2h"][(away, home)]["matches"] += 1
        if hs > aw:
            state[home]["h2h"][h2h_key]["wins"] += 1
            state[away]["h2h"][(away, home)]["losses"] += 1
        elif hs == aw:
            state[home]["h2h"][h2h_key]["draws"] += 1
            state[away]["h2h"][(away, home)]["draws"] += 1
        else:
            state[home]["h2h"][h2h_key]["losses"] += 1
            state[away]["h2h"][(away, home)]["wins"] += 1
        state[home]["h2h"][h2h_key]["gf"] += hs
        state[home]["h2h"][h2h_key]["ga"] += aw
        state[away]["h2h"][(away, home)]["gf"] += aw
        state[away]["h2h"][(away, home)]["ga"] += hs

        # WC stats
        state[home]["wc_matches"] += 1
        state[away]["wc_matches"] += 1
        if hs > aw:
            state[home]["wc_wins"] += 1
        elif hs < aw:
            state[away]["wc_wins"] += 1

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
