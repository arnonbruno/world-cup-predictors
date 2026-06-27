#!/usr/bin/env python3
"""
Match-level feature analysis: what drives Brazil vs Japan prediction.
Uses SHAP + feature importance + raw probability inspection.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
from shared import (
    GROUP_2026_TEAMS,
    apply_group_result,
    build_2026_group_state,
    build_round_of_32,
    compute_match_features,
    country_features_for_year,
    fit_xgb_with_validation,
    finalize_world_cup_history,
    harmonize_country,
    load_country_feature_history,
    rank_third_place_teams,
    sorted_group_standings,
)
warnings.filterwarnings('ignore')

INITIAL_ELO = 1500
K_FACTOR = 32

def harmonize(name):
    return harmonize_country(name)

def expected_score(ea, eb):
    return 1 / (1 + 10 ** ((eb - ea) / 400))

def update_elo(elo_a, elo_b, sa, sb):
    ea = expected_score(elo_a, elo_b)
    s = 1 if sa > sb else (0.5 if sa == sb else 0)
    margin = abs(sa - sb)
    multiplier = np.log(max(margin, 1) + 1)
    return (elo_a + K_FACTOR * multiplier * (s - ea),
            elo_b + K_FACTOR * multiplier * ((1 - s) - (1 - ea)))

def compute_features(team, opponent, state, cf, stage_num, match_date):
    return compute_match_features(team, opponent, state, cf, stage_num, match_date)

def update_state(state, ta, tb, sa, sb, date):
    ha, hb = harmonize(ta), harmonize(tb)
    state[ha]['elo'], state[hb]['elo'] = update_elo(state[ha]['elo'], state[hb]['elo'], sa, sb)
    for t, gf, ga in [(ha, sa, sb), (hb, sb, sa)]:
        state[t]['form'].append(1 if gf > ga else (0.5 if gf == ga else 0))
        state[t]['form'] = state[t]['form'][-20:]
        state[t]['goals_for'] = (state[t]['goals_for'] + [gf])[-20:]
        state[t]['goals_against'] = (state[t]['goals_against'] + [ga])[-20:]
        state[t]['last_match'] = date
    h2h_key = tuple(sorted([ha, hb]))
    for t in [ha, hb]: state[t]['h2h'][h2h_key]['matches'] += 1
    if sa > sb:
        state[ha]['h2h'][h2h_key]['wins'] += 1
        state[hb]['h2h'][h2h_key]['losses'] += 1
    elif sa == sb:
        state[ha]['h2h'][h2h_key]['draws'] += 1
        state[hb]['h2h'][h2h_key]['draws'] += 1
    else:
        state[ha]['h2h'][h2h_key]['losses'] += 1
        state[hb]['h2h'][h2h_key]['wins'] += 1
    state[ha]['h2h'][h2h_key]['gf'] += sa
    state[ha]['h2h'][h2h_key]['ga'] += sb
    state[hb]['h2h'][h2h_key]['gf'] += sb
    state[hb]['h2h'][h2h_key]['ga'] += sa
    state[ha]['wc_matches'] += 1
    state[hb]['wc_matches'] += 1
    if sa > sb: state[ha]['wc_wins'] += 1
    elif sa < sb: state[hb]['wc_wins'] += 1


def predict_probs(model, fl, ta, tb, state, cf, stage, date):
    ha, hb = harmonize(ta), harmonize(tb)
    feat = compute_features(ha, hb, state, cf, stage, date)
    X = pd.DataFrame([feat])[fl].fillna(0)
    return model.predict_proba(X)[0]


def shap_values_for_class(shap_values, class_idx=0, sample_idx=0):
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_idx])[sample_idx]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        if arr.shape[0] <= 3:
            return arr[class_idx, sample_idx, :]
        return arr[sample_idx, :, class_idx]
    if arr.ndim == 2:
        return arr[sample_idx]
    return arr


def shap_matrix_for_class(shap_values, class_idx=0):
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_idx])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        if arr.shape[0] <= 3:
            return arr[class_idx]
        return arr[:, :, class_idx]
    return arr


def main():
    print("Loading data and training model...")
    results = pd.read_csv('data/results.csv')
    country_history = load_country_feature_history()
    cf = country_features_for_year(country_history, 2022)

    df = results.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[~((df['date'].dt.year == 2026) & (df['tournament'] == 'FIFA World Cup'))]
    df = df.sort_values('date').reset_index(drop=True)

    state = defaultdict(lambda: {
        'elo': INITIAL_ELO, 'form': [], 'goals_for': [], 'goals_against': [],
        'last_match': None,
        'h2h': defaultdict(lambda: {'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0}),
        'wc_participations': 0, 'wc_titles': 0, 'wc_wins': 0, 'wc_matches': 0
    })

    rows, labels = [], []
    country_feature_cache = {}
    active_wc_year = None
    active_wc_teams = set()
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
        rows.append(compute_features(ht, at, state, country_feature_cache[feature_year], 0, r['date']))
        labels.append(0 if hs > aw else (1 if hs == aw else 2))
        state[ht]['elo'], state[at]['elo'] = update_elo(state[ht]['elo'], state[at]['elo'], hs, aw)
        for t, gf, ga, result in [(ht, hs, aw, 'W' if hs>aw else ('D' if hs==aw else 'L')),
                                   (at, aw, hs, 'W' if aw>hs else ('D' if hs==aw else 'L'))]:
            state[t]['form'].append(1 if result=='W' else (0.5 if result=='D' else 0))
            state[t]['form'] = state[t]['form'][-20:]
            state[t]['goals_for'] = (state[t]['goals_for'] + [gf])[-20:]
            state[t]['goals_against'] = (state[t]['goals_against'] + [ga])[-20:]
            state[t]['last_match'] = r['date']
        h2h_key = tuple(sorted([ht, at]))
        for t in [ht, at]: state[t]['h2h'][h2h_key]['matches'] += 1
        if hs > aw:
            state[ht]['h2h'][h2h_key]['wins'] += 1
            state[at]['h2h'][h2h_key]['losses'] += 1
        elif hs == aw:
            state[ht]['h2h'][h2h_key]['draws'] += 1
            state[at]['h2h'][h2h_key]['draws'] += 1
        else:
            state[ht]['h2h'][h2h_key]['losses'] += 1
            state[at]['h2h'][h2h_key]['wins'] += 1
        state[ht]['h2h'][h2h_key]['gf'] += hs
        state[ht]['h2h'][h2h_key]['ga'] += aw
        state[at]['h2h'][h2h_key]['gf'] += aw
        state[at]['h2h'][h2h_key]['ga'] += hs
        if is_world_cup:
            if active_wc_year is None:
                active_wc_year = feature_year
            active_wc_teams.update([ht, at])
            for t in [ht, at]: state[t]['wc_matches'] += 1
            if hs > aw: state[ht]['wc_wins'] += 1
            elif hs < aw: state[at]['wc_wins'] += 1

    if active_wc_year is not None:
        finalize_world_cup_history(state, active_wc_year, active_wc_teams)

    X = pd.DataFrame(rows).fillna(0)
    y = np.array(labels)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               objective='multi:softprob', num_class=3,
                               eval_metric='mlogloss', random_state=42, verbosity=0)
    model, _metrics = fit_xgb_with_validation(model, X, y, label="XGBoost")
    fl = X.columns.tolist()

    # Process completed 2026 WC matches
    wc26 = results[(results['tournament'] == 'FIFA World Cup') &
                    (pd.to_datetime(results['date']).dt.year == 2026)].copy()
    wc26['date'] = pd.to_datetime(wc26['date'])
    completed = wc26[wc26['home_score'].notna()].sort_values('date')
    for _, r in completed.iterrows():
        update_state(state, r['home_team'], r['away_team'],
                     int(r['home_score']), int(r['away_score']), r['date'])
    groups, remaining_matches = build_2026_group_state(results)
    for teams in GROUP_2026_TEAMS.values():
        for t in teams:
            state[harmonize(t)]['wc_participations'] += 1

    for date, home, away, group in remaining_matches:
        probs = predict_probs(model, fl, home, away, state, cf, 0, date)
        pa, pd_, pb = probs
        if pd_ >= pa and pd_ >= pb:
            sa, sb = 1, 1
        elif pa >= pb:
            sa, sb = 2, 1
        else:
            sa, sb = 1, 2
        update_state(state, home, away, sa, sb, date)
        apply_group_result(groups, group, home, away, sa, sb)

    standings_by_group = {g: sorted_group_standings(groups[g]) for g in "ABCDEFGHIJKL"}
    gw = {g: table[0][0] for g, table in standings_by_group.items()}
    gr = {g: table[1][0] for g, table in standings_by_group.items()}
    best8 = rank_third_place_teams(standings_by_group)[:8]
    r32 = build_round_of_32(gw, gr, best8)
    target = next(((home, away) for _label, home, away in r32 if "Brazil" in (home, away)), (r32[0][1], r32[0][2]))
    target_home, target_away = target

    # ── Brazil R32 match: Feature Vector ──
    print("\n" + "=" * 70)
    print(f"{target_home} vs {target_away} — MATCH FEATURE ANALYSIS")
    print("=" * 70)

    feat = compute_features(target_home, target_away, state, cf, 1, pd.Timestamp('2026-06-29'))
    feat_df = pd.DataFrame([feat])[fl].fillna(0)

    probs = model.predict_proba(feat_df)[0]
    print(f"\n  Model probability: {target_home} win {probs[0]:.1%} | Draw {probs[1]:.1%} | {target_away} win {probs[2]:.1%}")

    # ── XGBoost Feature Importance (gain) ──
    print(f"\n{'=' * 70}")
    print("TOP 20 FEATURES (XGBoost gain)")
    print(f"{'=' * 70}")
    importance = model.get_booster().get_score(importance_type='gain')
    imp_sorted = sorted(importance.items(), key=lambda x: -x[1])[:20]
    for fname, gain in imp_sorted:
        bar = '█' * int(gain / imp_sorted[0][1] * 30)
        print(f"  {fname:30s} {gain:8.1f} {bar}")

    # ── SHAP Analysis (global) ──
    print(f"\n{'=' * 70}")
    print(f"SHAP VALUES — {target_home} vs {target_away} (class 0 = {target_home} win)")
    print(f"{'=' * 70}")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feat_df)

    # Class 0 is home-team win for this constructed matchup.
    sv_brazil_win = shap_values_for_class(shap_values, class_idx=0, sample_idx=0)

    # Sort by absolute SHAP value
    shap_impact = list(zip(fl, sv_brazil_win, feat_df.values[0]))
    shap_impact.sort(key=lambda x: -abs(x[1]))

    print(f"\n  Top 15 features pushing {target_home} win probability:")
    for fname, sv, val in shap_impact[:15]:
        direction = f"↑ {target_home}" if sv > 0 else f"↓ {target_away}"
        print(f"    {fname:30s} SHAP={sv:+.4f}  val={val:.1f}  {direction}")

    print(f"\n  Top 10 features hurting Brazil win probability:")
    neg_impact = [x for x in shap_impact if x[1] < 0][:10]
    for fname, sv, val in neg_impact:
        print(f"    {fname:30s} SHAP={sv:+.4f}  val={val:.1f}  ↓ {target_away}")

    # ── Key feature values ──
    print(f"\n{'=' * 70}")
    print(f"KEY FEATURE VALUES — {target_home} vs {target_away}")
    print(f"{'=' * 70}")
    key_features = ['elo', 'elo_opponent', 'elo_diff', 'form_win_rate',
                    'h2h_win_rate', 'wc_titles', 'wc_win_rate',
                    'football_power_index', 'football_tradition',
                    'opp_football_power_index', 'opp_football_tradition',
                    'power_diff', 'tradition_diff', 'stage']
    for fname in key_features:
        if fname in feat:
            print(f"  {fname:30s} = {feat[fname]:.4f}")

    # ── SHAP Dependence Plot (top feature) ──
    print(f"\n{'=' * 70}")
    print("GLOBAL SHAP FEATURE IMPORTANCE (mean |SHAP|)")
    print(f"{'=' * 70}")
    # Use a sample of data for SHAP summary
    sample_idx = np.random.RandomState(42).choice(len(X), min(500, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]
    sv_all = explainer.shap_values(X_sample)
    # Class 0 = home-team win.
    sv0 = shap_matrix_for_class(sv_all, class_idx=0)
    mean_shap = np.abs(sv0).mean(axis=0)
    shap_ranking = sorted(zip(fl, mean_shap), key=lambda x: -x[1])
    for fname, mshap in shap_ranking[:20]:
        bar = '█' * int(mshap / shap_ranking[0][1] * 30)
        print(f"  {fname:30s} {mshap:.4f} {bar}")

    # ── Save SHAP summary plot ──
    try:
        plt.figure(figsize=(12, 10))
        shap.summary_plot(sv0, X_sample, feature_names=fl, show=False, max_display=20)
        plt.title(f'SHAP Summary — {target_home} win probability (class 0)')
        plt.tight_layout()
        plt.savefig('output/mc_2026/shap_brazil_win.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  SHAP summary plot saved to output/mc_2026/shap_brazil_win.png")
    except Exception as e:
        print(f"\n  Could not save SHAP plot: {e}")


if __name__ == '__main__':
    import os
    os.makedirs('output/mc_2026', exist_ok=True)
    main()