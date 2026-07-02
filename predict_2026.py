#!/usr/bin/env python3
"""
2026 FIFA World Cup Predictor — CORRECTED
Uses correct Wikipedia standings + proper FIFA bracket.
"""
import pandas as pd
import numpy as np
import argparse
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Optional
import warnings
from shared import (
    GROUP_2026_TEAMS,
    KNOCKOUT_ALPHA,
    ODDS_FEATURE_COLUMNS,
    WC2026_STAGE_TO_TRAIN,
    DEFAULT_GBT_MODEL,
    TRADITION_FEATURE_COLUMNS,
    apply_wc_knockout_calibration,
    apply_group_result,
    apply_match_to_state,
    build_2026_group_state,
    build_round_of_32,
    compute_match_features,
    country_feature_staleness_warnings,
    country_features_for_year,
    drop_feature_columns,
    fit_gbt_with_validation,
    finalize_world_cup_history,
    finalize_feature_frame,
    prepare_prediction_frame,
    sample_weights,
    load_betting_odds,
    odds_features_for_match,
    load_squad_values,
    fit_dixon_coles,
    blend_probabilities,
    harmonize_country,
    infer_world_cup_stage_map,
    load_country_feature_history,
    make_team_state,
    parse_neutral_flag,
    rank_third_place_teams,
    sorted_group_standings,
    wc_calibration_buckets,
    wc2026_penalty_winner,
    update_elo,
)
warnings.filterwarnings('ignore')

# 2026 co-hosts get home advantage when they play in their own country.
WC2026_HOSTS = {harmonize_country(t) for t in ("USA", "Canada", "Mexico")}

def harmonize(name):
    return harmonize_country(name)


@dataclass
class PredictionBundle:
    model: Any
    state: Any
    feature_names: list[str]
    train_X: pd.DataFrame
    train_y: np.ndarray
    country_history: dict
    country_features: dict
    odds: dict = None
    poisson_model: Any = None
    alpha: float = 1.0
    squad_values: dict = None


@dataclass
class PredictionDetails:
    winner: Optional[str]
    confidence: float
    p_home: float
    p_draw: float
    p_away: float
    raw_home: float
    raw_draw: float
    raw_away: float
    conditional_home: float
    conditional_away: float
    xgb_home: float
    xgb_draw: float
    xgb_away: float
    dc_home: Optional[float]
    dc_draw: Optional[float]
    dc_away: Optional[float]
    blend_alpha: float
    odds_missing: bool
    calibration_note: str = ""
    mc_home: Optional[float] = None
    mc_draw: Optional[float] = None
    mc_away: Optional[float] = None
    mc_home_ko: Optional[float] = None
    mc_away_ko: Optional[float] = None
    mc_avg_goals_home: Optional[float] = None
    mc_avg_goals_away: Optional[float] = None
    mc_top_scoreline: Optional[str] = None


def compute_features(team, opponent, state, country_features, stage_num, match_date, neutral=True, is_home=False, odds_row=None, squad_values=None):
    return compute_match_features(team, opponent, state, country_features, stage_num, match_date, neutral, is_home, odds_row, squad_values)

def _tune_blend_alpha(model, poisson_model, X_val, y_val, val_meta):
    from sklearn.metrics import log_loss as _ll
    p_xgb = model.predict_proba(X_val)
    p_pois = np.array([poisson_model.outcome_probs(h, a, neutral=neu) for (h, a, neu) in val_meta])
    best_alpha, best_loss = 1.0, np.inf
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = np.array([blend_probabilities(p_xgb[i], p_pois[i], alpha) for i in range(len(p_xgb))])
        try:
            loss = _ll(y_val, blended, labels=[0, 1, 2])
        except Exception:
            continue
        if loss < best_loss:
            best_loss, best_alpha = loss, alpha
    return float(best_alpha)


def train_model_bundle(
    results_df,
    country_history,
    exclude_2026_wc=True,
    odds=None,
    squad_values=None,
    *,
    model_type: str | None = None,
    exclude_tradition: bool = True,
    hyperopt_trials: int = 0,
    lgbm_params: dict | None = None,
):
    if odds is None:
        odds = load_betting_odds()
    if squad_values is None:
        squad_values = load_squad_values()
    df = results_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    if exclude_2026_wc:
        df = df[~((df['date'].dt.year == 2026) & (df['tournament'] == 'FIFA World Cup'))]
    df = df.sort_values('date').reset_index(drop=True)
    
    state = defaultdict(make_team_state)
    
    rows, labels, feature_dates, match_meta = [], [], [], []
    country_feature_cache = {}
    active_wc_year = None
    active_wc_teams = set()
    wc_stage_by_index = infer_world_cup_stage_map(df)
    for _, r in df.iterrows():
        ht, at = harmonize(r['home_team']), harmonize(r['away_team'])
        hs, aw = r['home_score'], r['away_score']
        if pd.isna(hs) or pd.isna(aw): continue
        hs, aw = int(hs), int(aw)
        feature_year = int(r['date'].year)
        is_world_cup = r['tournament'] == 'FIFA World Cup'
        if active_wc_year is not None and (not is_world_cup or feature_year != active_wc_year):
            finalize_world_cup_history(state, active_wc_year, active_wc_teams)
            active_wc_year = None
            active_wc_teams = set()

        if feature_year not in country_feature_cache:
            country_feature_cache[feature_year] = country_features_for_year(country_history, feature_year)
        stage = wc_stage_by_index.get(int(r.name), 0) if is_world_cup else 0
        neutral = parse_neutral_flag(r.get('neutral', True))
        odds_row = odds_features_for_match(odds, r['date'], ht, at)
        rows.append(compute_features(ht, at, state, country_feature_cache[feature_year], stage, r['date'], neutral, not neutral, odds_row, squad_values))
        labels.append(0 if hs > aw else (1 if hs == aw else 2))
        feature_dates.append(r['date'])
        match_meta.append((ht, at, neutral))

        if is_world_cup:
            if active_wc_year is None:
                active_wc_year = feature_year
            active_wc_teams.update([ht, at])

        apply_match_to_state(state, ht, at, hs, aw, r['date'],
                             neutral=neutral, is_world_cup=is_world_cup)

    if active_wc_year is not None:
        finalize_world_cup_history(state, active_wc_year, active_wc_teams)
    
    X = finalize_feature_frame(rows)
    if exclude_tradition:
        X = drop_feature_columns(X, TRADITION_FEATURE_COLUMNS)
    y = np.array(labels)
    weights = sample_weights(y, feature_dates)
    model_type = (model_type or DEFAULT_GBT_MODEL).lower()
    gbt_label = "LightGBM" if model_type == "lgbm" else "XGBoost"
    trials = hyperopt_trials if model_type == "lgbm" else 0
    model, _metrics = fit_gbt_with_validation(
        model_type, X, y,
        dates=feature_dates,
        sample_weight=weights,
        calibrate=True,
        lgbm_params=lgbm_params,
        hyperopt_trials=trials,
        label=gbt_label,
    )
    print(f"  Trained {gbt_label} on {len(X)} matches")

    poisson_model = fit_dixon_coles(results_df)
    order = pd.Series(pd.to_datetime(feature_dates, errors="coerce")).sort_values().index
    split = max(1, int(len(order) * 0.8))
    val_idx = order[split:]
    alpha = _tune_blend_alpha(model, poisson_model, X.iloc[val_idx], y[val_idx],
                              [match_meta[i] for i in val_idx])
    print(f"  Tuned blend alpha ({gbt_label} weight) = {alpha:.2f}")
    return PredictionBundle(
        model=model,
        state=state,
        feature_names=X.columns.tolist(),
        train_X=X,
        train_y=y,
        country_history=country_history,
        country_features=country_features_for_year(country_history, 2026),
        odds=odds,
        poisson_model=poisson_model,
        alpha=alpha,
        squad_values=squad_values,
    )

def train_model(
    results_df,
    country_history,
    *,
    model_type: str | None = None,
    exclude_tradition: bool = True,
    hyperopt_trials: int = 0,
    lgbm_params: dict | None = None,
):
    bundle = train_model_bundle(
        results_df,
        country_history,
        model_type=model_type,
        exclude_tradition=exclude_tradition,
        hyperopt_trials=hyperopt_trials,
    )
    # Stash the ensemble pieces on module-level globals so main() can use them
    # without changing the historical (model, state, fl, X, y) return contract
    # that explain_match.py relies on.
    global _ODDS, _POISSON, _ALPHA, _SQUAD_VALUES
    _ODDS, _POISSON, _ALPHA = bundle.odds, bundle.poisson_model, bundle.alpha
    _SQUAD_VALUES = bundle.squad_values
    return bundle.model, bundle.state, bundle.feature_names, bundle.train_X, bundle.train_y


_ODDS = None
_POISSON = None
_ALPHA = 1.0
_SQUAD_VALUES = None
_WC_CALIBRATION_BUCKETS = []

def prepare_2026_state(results, state):
    wc26 = results[(results['tournament'] == 'FIFA World Cup') &
                   (pd.to_datetime(results['date']).dt.year == 2026)].copy()
    wc26['date'] = pd.to_datetime(wc26['date'])
    completed = wc26[wc26['home_score'].notna() & wc26['away_score'].notna()].sort_values('date')
    for _, r in completed.iterrows():
        neutral = parse_neutral_flag(r.get('neutral', True))
        update_state(state, r['home_team'], r['away_team'], int(r['home_score']), int(r['away_score']), r['date'], neutral=neutral)
    for teams in GROUP_2026_TEAMS.values():
        for team in teams:
            state[harmonize(team)]['wc_participations'] += 1
    return state

def load_country_features():
    return country_features_for_year(load_country_feature_history(), 2026)

def update_state(state, ta, tb, sa, sb, date, neutral=True):
    # All 2026 fixtures handled here are World Cup matches.
    apply_match_to_state(state, ta, tb, sa, sb, date, neutral=neutral, is_world_cup=True)

def _alpha_for_stage(stage: int) -> float:
    return KNOCKOUT_ALPHA if stage > 0 else _ALPHA


def _odds_missing(odds_row: dict) -> bool:
    return any(not np.isfinite(float(odds_row.get(col, np.nan))) for col in ODDS_FEATURE_COLUMNS)


def _compute_mc_details(ha, hb, state, stage):
    """Compute MC simulation results for a match, returning a dict to merge into PredictionDetails."""
    if _POISSON is None:
        return {}
    elo_diff = float(state.get(ha, {}).get("elo", 1500)) - float(state.get(hb, {}).get("elo", 1500))
    is_ko = stage > 0
    mc = _POISSON.mc_simulate(ha, hb, neutral=True, n_sims=100_000,
                               knockout=is_ko, elo_diff=elo_diff if is_ko else None)
    result = {
        "mc_home": mc["p_home"],
        "mc_draw": mc["p_draw"],
        "mc_away": mc["p_away"],
        "mc_avg_goals_home": mc["avg_goals_home"],
        "mc_avg_goals_away": mc["avg_goals_away"],
    }
    if is_ko:
        result["mc_home_ko"] = mc["p_home_ko"]
        result["mc_away_ko"] = mc["p_away_ko"]
    if mc.get("top_scorelines"):
        result["mc_top_scoreline"] = mc["top_scorelines"][0][0]
    return result


def predict_with_details(model, fl, ta, tb, state, cf, stage, date, neutral=True, is_home=False):
    ha, hb = harmonize(ta), harmonize(tb)
    odds_row = odds_features_for_match(_ODDS, date, ha, hb)
    feat = compute_features(ha, hb, state, cf, stage, date, neutral, is_home, odds_row, _SQUAD_VALUES)
    X = prepare_prediction_frame(feat, fl)
    p_xgb = np.asarray(model.predict_proba(X)[0], dtype=float)
    probs = p_xgb.copy()
    p_dc = None
    alpha = _alpha_for_stage(stage)
    if _POISSON is not None and alpha < 1.0:
        p_dc = _POISSON.outcome_probs(ha, hb, neutral=neutral)
        probs = blend_probabilities(probs, p_dc, alpha)
    pa, pd_, pb = probs[0], probs[1], probs[2]
    conditional_home, conditional_away = pa, pb
    calibration_note = ""
    # Group stage: allow draw if it's the most likely outcome
    if stage == 0 and pd_ >= pa and pd_ >= pb:
        return PredictionDetails(
            winner=None, confidence=float(pd_), p_home=float(pa), p_draw=float(pd_), p_away=float(pb),
            raw_home=float(pa), raw_draw=float(pd_), raw_away=float(pb),
            conditional_home=float(pa), conditional_away=float(pb),
            xgb_home=float(p_xgb[0]), xgb_draw=float(p_xgb[1]), xgb_away=float(p_xgb[2]),
            dc_home=float(p_dc[0]) if p_dc is not None else None,
            dc_draw=float(p_dc[1]) if p_dc is not None else None,
            dc_away=float(p_dc[2]) if p_dc is not None else None,
            blend_alpha=float(alpha), odds_missing=_odds_missing(odds_row),
            calibration_note=calibration_note,
            **_compute_mc_details(ha, hb, state, stage),
        )
    # Knockout stages: no draw possible, renormalize to P(home|no draw), P(away|no draw)
    if stage > 0:
        total = pa + pb
        if total > 0:
            pa_cond = pa / total
            pb_cond = pb / total
        else:
            pa_cond, pb_cond = 0.5, 0.5
        conditional_home, conditional_away = pa_cond, pb_cond
        pa_final, pb_final, calibration_note = apply_wc_knockout_calibration(
            pa_cond, pb_cond, _WC_CALIBRATION_BUCKETS,
        )
        winner = ta if pa_cond >= pb_cond else tb
        return PredictionDetails(
            winner=winner, confidence=float(max(pa_final, pb_final)),
            p_home=float(pa_final), p_draw=float(pd_), p_away=float(pb_final),
            raw_home=float(pa), raw_draw=float(pd_), raw_away=float(pb),
            conditional_home=float(pa_cond), conditional_away=float(pb_cond),
            xgb_home=float(p_xgb[0]), xgb_draw=float(p_xgb[1]), xgb_away=float(p_xgb[2]),
            dc_home=float(p_dc[0]) if p_dc is not None else None,
            dc_draw=float(p_dc[1]) if p_dc is not None else None,
            dc_away=float(p_dc[2]) if p_dc is not None else None,
            blend_alpha=float(alpha), odds_missing=_odds_missing(odds_row),
            calibration_note=calibration_note,
            **_compute_mc_details(ha, hb, state, stage),
        )
    winner = ta if pa >= pb else tb
    return PredictionDetails(
        winner=winner, confidence=float(max(pa, pb)), p_home=float(pa), p_draw=float(pd_), p_away=float(pb),
        raw_home=float(pa), raw_draw=float(pd_), raw_away=float(pb),
        conditional_home=float(conditional_home), conditional_away=float(conditional_away),
        xgb_home=float(p_xgb[0]), xgb_draw=float(p_xgb[1]), xgb_away=float(p_xgb[2]),
        dc_home=float(p_dc[0]) if p_dc is not None else None,
        dc_draw=float(p_dc[1]) if p_dc is not None else None,
        dc_away=float(p_dc[2]) if p_dc is not None else None,
        blend_alpha=float(alpha), odds_missing=_odds_missing(odds_row),
        calibration_note=calibration_note,
        **_compute_mc_details(ha, hb, state, stage),
    )


def predict(model, fl, ta, tb, state, cf, stage, date, neutral=True, is_home=False):
    details = predict_with_details(model, fl, ta, tb, state, cf, stage, date, neutral, is_home)
    return details.winner, details.confidence, details.p_home, details.p_draw, details.p_away


def get_completed_knockout_results(results_df):
    """Return completed WC 2026 knockout results keyed by sorted team pair."""
    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    wc26 = df[
        (df["tournament"] == "FIFA World Cup")
        & (df["date"].dt.year == 2026)
        & (df["date"] >= pd.Timestamp("2026-06-28"))
    ]
    completed = wc26[wc26["home_score"].notna() & wc26["away_score"].notna()]
    results = {}
    for _, row in completed.iterrows():
        ha = harmonize(row["home_team"])
        at = harmonize(row["away_team"])
        hs = int(row["home_score"])
        aw = int(row["away_score"])
        key = tuple(sorted([ha, at]))
        results[key] = {
            "home_team": ha,
            "away_team": at,
            "home_score": hs,
            "away_score": aw,
            "is_draw": hs == aw,
        }
    return results


def simulate_round(matches, name, stage, state, model, fl, cf, date, debug=False, completed=None):
    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"{'=' * 70}")
    winners = []
    for label, ta, tb in matches:
        ha, at = harmonize(ta), harmonize(tb)
        key = tuple(sorted([ha, at]))
        penalty_note = None

        if completed and key in completed:
            result = completed[key]
            if not result["is_draw"]:
                winner = (
                    result["home_team"]
                    if result["home_score"] > result["away_score"]
                    else result["away_team"]
                )
                winners.append(winner)
                print(f"  {label}: {ta} vs {tb}")
                print(
                    f"    → ✅ {winner} [ACTUAL: {result['home_score']}-{result['away_score']}]"
                )
                continue
            # Draw in CSV — check if we know the penalty winner
            winner = wc2026_penalty_winner(ha, at)
            if winner:
                winners.append(winner)
                print(f"  {label}: {ta} vs {tb}")
                print(
                    f"    → ✅ {winner} [ACTUAL: {result['home_score']}-{result['away_score']} (penalties)]"
                )
                continue
            # Unknown penalty winner — fall through to prediction
            penalty_note = (
                f"ACTUAL RESULT: {result['home_score']}-{result['away_score']} (penalties)"
            )

        details = predict_with_details(model, fl, ta, tb, state, cf, stage, date)
        winner, prob, pa, pd_, pb = (
            details.winner, details.confidence, details.p_home, details.p_draw, details.p_away
        )
        winners.append(winner)
        if penalty_note is None:
            if winner == ta:
                update_state(state, ta, tb, 2, 1, date)
            else:
                update_state(state, ta, tb, 1, 2, date)
        print(f"  {label}: {ta} vs {tb}")
        if penalty_note:
            print(f"    {penalty_note}")
        if stage > 0:
            print(f"    → ✅ {winner} ({prob:.1%})  [Pipeline P({ta})={pa:.1%} P({tb})={pb:.1%}]")
            if details.mc_home_ko is not None:
                mc_winner_prob = details.mc_home_ko if winner == ta else details.mc_away_ko
                print(f"    🎲 MC sim: {ta} {details.mc_home_ko:.1%} | {tb} {details.mc_away_ko:.1%} "
                      f"(avg {details.mc_avg_goals_home:.1f}-{details.mc_avg_goals_away:.1f}, "
                      f"likely {details.mc_top_scoreline})")
            print(
                f"      raw 3-way: P({ta})={details.raw_home:.1%} "
                f"P(draw)={details.raw_draw:.1%} P({tb})={details.raw_away:.1%}"
            )
            print(
                f"      knockout no-draw: P({ta})={details.conditional_home:.1%} "
                f"P({tb})={details.conditional_away:.1%}; blend alpha={details.blend_alpha:.2f}"
            )
            if details.calibration_note:
                print(f"      calibration: {details.calibration_note}")
            if details.odds_missing:
                print("      WARNING: betting odds are unavailable; this prediction is less market-informed.")
            if debug:
                dc_text = (
                    f"P({ta})={details.dc_home:.1%} P(draw)={details.dc_draw:.1%} P({tb})={details.dc_away:.1%}"
                    if details.dc_home is not None else "unavailable"
                )
                print(
                    f"      XGBoost raw: P({ta})={details.xgb_home:.1%} "
                    f"P(draw)={details.xgb_draw:.1%} P({tb})={details.xgb_away:.1%}"
                )
                print(f"      Dixon-Coles raw: {dc_text}")
        else:
            mc_note = ""
            if details.mc_home is not None:
                mc_note = f" | MC: {details.mc_home:.0%}/{details.mc_draw:.0%}/{details.mc_away:.0%}"
            print(f"    → ✅ {winner} ({prob:.1%})  [P({ta})={pa:.1%} P(draw)={pd_:.1%} P({tb})={pb:.1%}]{mc_note}")
    return winners


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the 2026 FIFA World Cup bracket.")
    parser.add_argument(
        "--debug",
        "--verbose",
        action="store_true",
        help="Print model-component probabilities for knockout matches.",
    )
    parser.add_argument(
        "--model",
        choices=("xgb", "lgbm"),
        default=None,
        help=f"Gradient-boosted tree backend (default: {DEFAULT_GBT_MODEL}).",
    )
    parser.add_argument(
        "--exclude-tradition",
        action="store_true",
        help="Drop football_tradition / tradition_diff features.",
    )
    parser.add_argument(
        "--hyperopt-trials",
        type=int,
        default=0,
        help="Hyperopt trials for LightGBM tuning (default: 0, use baked-in params).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 70)
    print("🏆 2026 FIFA WORLD CUP PREDICTOR")
    print("=" * 70)
    
    print("\n[1/4] Loading data...")
    results = pd.read_csv('data/results.csv')
    country_history = load_country_feature_history()
    cf = country_features_for_year(country_history, 2026)
    print(f"  {len(results)} matches, {len(cf)} countries with features")
    stale_notes = country_feature_staleness_warnings(
        cf,
        [team for teams in GROUP_2026_TEAMS.values() for team in teams],
        2026,
    )
    for note in stale_notes:
        print(f"  WARNING: {note}")
    
    print("[2/4] Training model...")
    model, state, fl, _train_X, _train_y = train_model(
        results,
        country_history,
        model_type=args.model,
        exclude_tradition=args.exclude_tradition,
        hyperopt_trials=args.hyperopt_trials,
    )
    global _WC_CALIBRATION_BUCKETS
    _WC_CALIBRATION_BUCKETS = wc_calibration_buckets()
    if _WC_CALIBRATION_BUCKETS:
        print(f"  Loaded {len(_WC_CALIBRATION_BUCKETS)} World Cup calibration buckets")
    else:
        print("  WARNING: World Cup calibration buckets unavailable; knockout probabilities are uncalibrated")
    
    print("[3/4] Processing completed 2026 WC matches...")
    before_matches = sum(state[t]['wc_matches'] for t in state)
    prepare_2026_state(results, state)
    after_matches = sum(state[t]['wc_matches'] for t in state)
    print(f"  Processed {(after_matches - before_matches) // 2} completed matches")
    
    groups, remaining = build_2026_group_state(results)

    # Mark WC participation for all 48 teams
    all_teams = set()
    for teams in GROUP_2026_TEAMS.values():
        for t in teams:
            all_teams.add(harmonize(t))
    
    print("[4/4] Simulating remaining matches...\n")
    
    # === REMAINING GROUP MATCHES (from the unplayed rows in results.csv) ===
    print("=" * 70)
    print("REMAINING GROUP MATCHES")
    print("=" * 70)
    
    for date, home, away, group in remaining:
        ha, at = harmonize(home), harmonize(away)
        # The 2026 hosts (USA, Canada, Mexico) get genuine home advantage at home.
        home_is_host = ha in WC2026_HOSTS
        details = predict_with_details(
            model, fl, home, away, state, cf, 0, date,
            neutral=not home_is_host, is_home=home_is_host,
        )
        winner, prob, pa, pd_, pb = (
            details.winner, details.confidence, details.p_home, details.p_draw, details.p_away
        )
        if winner == home:
            sa, sb = 2, 1
        elif winner == away:
            sa, sb = 1, 2
        else:
            sa, sb = 1, 1
        update_state(state, ha, at, sa, sb, date, neutral=not home_is_host)
        apply_group_result(groups, group, ha, at, sa, sb)
        arrow = "←" if prob > 0.55 else "~"
        result_label = "draw" if winner is None else winner
        mc_note = ""
        if details.mc_home is not None:
            mc_note = f" | MC: {details.mc_home:.0%}/{details.mc_draw:.0%}/{details.mc_away:.0%}"
        print(f"  {date.strftime('%Y-%m-%d')} {home} vs {away} → {result_label} ({prob:.1%}) {arrow}{mc_note}")
    
    # === FINAL STANDINGS ===
    print(f"\n{'=' * 70}")
    print("FINAL GROUP STANDINGS")
    print(f"{'=' * 70}")
    
    thirds_all = []
    gw, gr = {}, {}
    standings_by_group = {}
    for g in 'ABCDEFGHIJKL':
        st = sorted_group_standings(groups[g])
        standings_by_group[g] = st
        gw[g], gr[g] = st[0][0], st[1][0]
        thirds_all.append((g, st[2][0], st[2][1], st[2][2], st[2][3]))
        print(f"\n  Group {g}:")
        marks = ["✓1st", "✓2nd", "?3rd", " ✗ "]
        for i, (t, pts, gd, gf) in enumerate(st):
            print(f"    {marks[i]} {t}: {pts} pts (GD {'+' if gd >= 0 else ''}{gd})")
    
    # Best 8 thirds (FIFA order: points, goal difference, goals for)
    thirds_all = rank_third_place_teams(standings_by_group)
    best8 = thirds_all[:8]
    elim8 = thirds_all[8:]
    best8_groups = sorted([t[0] for t in best8])
    print(f"\n  Best 8 thirds (groups {','.join(best8_groups)}):")
    for g, t, pts, gd, gf in best8:
        print(f"    {t} ({pts}pts, GD {'+' if gd >= 0 else ''}{gd}, GF {gf}, G{g})")
    print(f"  Eliminated: {', '.join(f'{t[1]} ({t[2]}pts, G{t[0]})' for t in elim8)}")
    
    # === KNOCKOUT BRACKET ===
    r32 = build_round_of_32(gw, gr, best8)
    completed_r32 = get_completed_knockout_results(results)

    r32_w = simulate_round(
        r32, "ROUND OF 32", WC2026_STAGE_TO_TRAIN["round_of_32"],
        state, model, fl, cf, pd.Timestamp('2026-06-29'),
        debug=args.debug, completed=completed_r32,
    )
    
    # FIFA bracket R16 pairings (from Wikipedia — NON-SEQUENTIAL crossover!)
    # M89: W73 vs W75 | M90: W74 vs W77 | M91: W76 vs W78 | M92: W79 vs W80
    # M93: W83 vs W84 | M94: W81 vs W82 | M95: W86 vs W88 | M96: W85 vs W87
    # r32 indices: 0=73, 1=74, 2=75, 3=76, 4=77, 5=78, 6=79, 7=80,
    #              8=81, 9=82, 10=83, 11=84, 12=85, 13=86, 14=87, 15=88
    r16 = [
        ('R16 M89', r32_w[0], r32_w[2]),   # W73 vs W75
        ('R16 M90', r32_w[1], r32_w[4]),   # W74 vs W77
        ('R16 M91', r32_w[3], r32_w[5]),   # W76 vs W78
        ('R16 M92', r32_w[6], r32_w[7]),   # W79 vs W80
        ('R16 M93', r32_w[10], r32_w[11]), # W83 vs W84
        ('R16 M94', r32_w[8], r32_w[9]),   # W81 vs W82
        ('R16 M95', r32_w[13], r32_w[15]), # W86 vs W88
        ('R16 M96', r32_w[12], r32_w[14]), # W85 vs W87
    ]
    r16_w = simulate_round(r16, "ROUND OF 16", WC2026_STAGE_TO_TRAIN["round_of_16"], state, model, fl, cf, pd.Timestamp('2026-07-04'), debug=args.debug)
    
    # QF: M97: W89 vs W90 | M98: W93 vs W94 | M99: W91 vs W92 | M100: W95 vs W96
    # Bracket: (89/90 side) vs (93/94 side) → SF101, (91/92 side) vs (95/96 side) → SF102
    qf = [
        ('QF M97', r16_w[0], r16_w[1]),  # W89 vs W90 (left-top: Germany path vs Canada/Ned path)
        ('QF M98', r16_w[4], r16_w[5]),  # W93 vs W94 (right-top: Spain/Colombia vs USA/Iran)
        ('QF M99', r16_w[2], r16_w[3]),  # W91 vs W92 (left-bot: Brazil path vs Mexico/England)
        ('QF M100', r16_w[6], r16_w[7]), # W95 vs W96 (right-bot: Argentina/Bel vs Swi/Port)
    ]
    qf_w = simulate_round(qf, "QUARTERFINALS", WC2026_STAGE_TO_TRAIN["quarterfinal"], state, model, fl, cf, pd.Timestamp('2026-07-09'), debug=args.debug)
    
    # SF: M101: W97 vs W98 | M102: W99 vs W100
    # Germany side (QF97/98) vs Brazil side (QF99/100) → they can only meet in FINAL
    sf = [
        ('SF M101', qf_w[0], qf_w[1]),  # left side (Germany's half)
        ('SF M102', qf_w[2], qf_w[3]),  # right side (Brazil's half)
    ]
    sf_w = simulate_round(sf, "SEMIFINALS", WC2026_STAGE_TO_TRAIN["semifinal"], state, model, fl, cf, pd.Timestamp('2026-07-14'), debug=args.debug)
    
    # 3rd place: L101 vs L102
    sf_losers = [qf_w[i] for i in range(4) if qf_w[i] not in sf_w]
    print(f"\n{'=' * 70}")
    print("THIRD PLACE MATCH")
    print(f"{'=' * 70}")
    third_details = predict_with_details(model, fl, sf_losers[0], sf_losers[1], state, cf, WC2026_STAGE_TO_TRAIN["third_place"], pd.Timestamp('2026-07-18'))
    third, tp = third_details.winner, third_details.confidence
    print(f"  {sf_losers[0]} vs {sf_losers[1]} → {third} ({tp:.1%})")
    
    # FINAL
    print(f"\n{'=' * 70}")
    print("🏆 FINAL 🏆")
    print(f"{'=' * 70}")
    final_details = predict_with_details(model, fl, sf_w[0], sf_w[1], state, cf, WC2026_STAGE_TO_TRAIN["final"], pd.Timestamp('2026-07-19'))
    champion, cp, pa, pd_, pb = (
        final_details.winner, final_details.confidence,
        final_details.p_home, final_details.p_draw, final_details.p_away,
    )
    runner_up = sf_w[1] if champion == sf_w[0] else sf_w[0]
    fourth = sf_losers[1] if third == sf_losers[0] else sf_losers[0]
    
    print(f"\n  {sf_w[0]} vs {sf_w[1]}")
    print(f"  P({sf_w[0]}): {pa:.1%}  P(draw): {pd_:.1%}  P({sf_w[1]}): {pb:.1%}")
    
    print(f"\n{'=' * 70}")
    print("🏆 2026 FIFA WORLD CUP — FINAL PREDICTION")
    print(f"{'=' * 70}")
    print(f"\n  🥇 CHAMPION:  {champion} ({cp:.1%})")
    print(f"  🥈 Runner-up: {runner_up}")
    print(f"  🥉 Third:     {third}")
    print(f"  4th place:    {fourth}")
    
    print(f"\n  TOP 10 ELO (post-group):")
    elo_list = sorted([(t, state[t]['elo']) for t in all_teams], key=lambda x: -x[1])
    for i, (t, e) in enumerate(elo_list[:10]):
        print(f"    {i+1}. {t}: {e:.0f}")

if __name__ == '__main__':
    main()
