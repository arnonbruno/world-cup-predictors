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
warnings.filterwarnings('ignore')

# ── Same setup as monte_carlo_2026.py ──
NAME_MAP = {
    'West Germany': 'Germany', 'Soviet Union': 'Russia', 'USSR': 'Russia',
    'Yugoslavia': 'Serbia', 'Czechoslovakia': 'Czech Republic',
    'Zaire': 'DR Congo', 'Ivory Coast': "Côte d'Ivoire",
    'South Korea': 'Korea Republic', 'North Korea': 'Korea DPR',
    'Iran': 'IR Iran', 'United States': 'USA',
}
INITIAL_ELO = 1500
K_FACTOR = 32

def harmonize(name):
    return NAME_MAP.get(name, name)

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
    s, o = state[team], state[opponent]
    form = s['form'][-10:] if s['form'] else [0.5]
    gf5 = s['goals_for'][-5:] or [0]
    ga5 = s['goals_against'][-5:] or [0]
    gf10 = s['goals_for'][-10:] or [0]
    ga10 = s['goals_against'][-10:] or [0]
    rest = min((match_date - s['last_match']).days if s['last_match'] else 30, 60)
    h2h_key = tuple(sorted([team, opponent]))
    h = s['h2h'].get(h2h_key, {'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0})
    hm = max(h['matches'], 1)
    ctf, ocf = cf.get(team, {}), cf.get(opponent, {})
    return {
        'elo': s['elo'], 'elo_opponent': o['elo'],
        'elo_diff': s['elo'] - o['elo'], 'elo_sum': s['elo'] + o['elo'],
        'form_win_rate': sum(1 for f in form if f == 1) / len(form),
        'form_draw_rate': sum(1 for f in form if f == 0.5) / len(form),
        'form_loss_rate': sum(1 for f in form if f == 0) / len(form),
        'avg_goals_scored_5': np.mean(gf5), 'avg_goals_conceded_5': np.mean(ga5),
        'avg_goals_scored_10': np.mean(gf10), 'avg_goals_conceded_10': np.mean(ga10),
        'rest_days': rest,
        'h2h_matches': h['matches'], 'h2h_win_rate': h['wins'] / hm,
        'h2h_draw_rate': h['draws'] / hm, 'h2h_avg_goals_for': h['gf'] / hm,
        'h2h_avg_goals_against': h['ga'] / hm,
        'wc_participations': s['wc_participations'], 'wc_titles': s['wc_titles'],
        'wc_win_rate': s['wc_wins'] / max(s['wc_matches'], 1),
        'stage': stage_num, 'neutral': 1, 'is_home': 0,
        'gdp_per_capita': ctf.get('gdp_per_capita', np.nan),
        'population': ctf.get('population', np.nan),
        'life_expectancy': ctf.get('life_expectancy', np.nan),
        'urban_population_pct': ctf.get('urban_population_pct', np.nan),
        'health_expenditure_pct_gdp': ctf.get('health_expenditure_pct_gdp', np.nan),
        'elo_pre_tournament': ctf.get('elo_rating', s['elo']),
        'fifa_ranking': ctf.get('fifa_ranking', 100),
        'football_power_index': ctf.get('football_power_index', 0),
        'football_tradition': ctf.get('football_tradition', 0),
        'opp_elo_pre_tournament': ocf.get('elo_rating', o['elo']),
        'opp_football_power_index': ocf.get('football_power_index', 0),
        'opp_football_tradition': ocf.get('football_tradition', 0),
        'elo_diff_pre': ctf.get('elo_rating', s['elo']) - ocf.get('elo_rating', o['elo']),
        'power_diff': ctf.get('football_power_index', 0) - ocf.get('football_power_index', 0),
        'tradition_diff': ctf.get('football_tradition', 0) - ocf.get('football_tradition', 0),
    }

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


def main():
    print("Loading data and training model...")
    results = pd.read_csv('data/results.csv')
    cf_df = pd.read_csv('data/world_cup_predictors_dataset.csv')
    cf_df = cf_df[cf_df['wc_year'] == 2022]
    cf = {}
    for _, r in cf_df.iterrows():
        t = harmonize(r['country'])
        cf[t] = {k: r.get(k, np.nan) for k in [
            'gdp_per_capita', 'population', 'life_expectancy',
            'urban_population_pct', 'health_expenditure_pct_gdp',
            'elo_rating', 'fifa_ranking', 'football_power_index', 'football_tradition'
        ]}

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
    for _, r in df.iterrows():
        ht, at = harmonize(r['home_team']), harmonize(r['away_team'])
        hs, aw = r['home_score'], r['away_score']
        if pd.isna(hs) or pd.isna(aw): continue
        hs, aw = int(hs), int(aw)
        rows.append(compute_features(ht, at, state, {}, 0, r['date']))
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
        if r['tournament'] == 'FIFA World Cup':
            for t in [ht, at]: state[t]['wc_matches'] += 1
            if hs > aw: state[ht]['wc_wins'] += 1
            elif hs < aw: state[at]['wc_wins'] += 1

    X = pd.DataFrame(rows).fillna(0)
    y = np.array(labels)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               objective='multi:softprob', num_class=3,
                               eval_metric='mlogloss', random_state=42, verbosity=0)
    model.fit(X, y)
    fl = X.columns.tolist()

    # Process completed 2026 WC matches
    wc26 = results[(results['tournament'] == 'FIFA World Cup') &
                    (pd.to_datetime(results['date']).dt.year == 2026)].copy()
    wc26['date'] = pd.to_datetime(wc26['date'])
    completed = wc26[wc26['home_score'].notna()].sort_values('date')
    for _, r in completed.iterrows():
        update_state(state, r['home_team'], r['away_team'],
                     int(r['home_score']), int(r['away_score']), r['date'])
    for g_data in GROUPS.values():
        for t in g_data:
            if t.startswith('_'): continue
            state[harmonize(t)]['wc_participations'] += 1

    # ── Brazil vs Japan: Feature Vector ──
    print("\n" + "=" * 70)
    print("BRAZIL vs JAPAN — MATCH FEATURE ANALYSIS")
    print("=" * 70)

    feat = compute_features('Brazil', 'Japan', state, cf, 1, pd.Timestamp('2026-06-29'))
    feat_df = pd.DataFrame([feat])[fl].fillna(0)

    probs = model.predict_proba(feat_df)[0]
    print(f"\n  Model probability: Brazil win {probs[0]:.1%} | Draw {probs[1]:.1%} | Japan win {probs[2]:.1%}")

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
    print("SHAP VALUES — BRAZIL vs JAPAN (class 0 = Brazil win)")
    print(f"{'=' * 70}")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feat_df)

    # shap_values is [3, 1, 38] for 3 classes, 1 sample, 38 features
    # Class 0 = home win (Brazil)
    sv_brazil_win = shap_values[0][0] if len(shap_values.shape) == 3 else shap_values[0]

    # Sort by absolute SHAP value
    shap_impact = list(zip(fl, sv_brazil_win, feat_df.values[0]))
    shap_impact.sort(key=lambda x: -abs(x[1]))

    print(f"\n  Top 15 features pushing Brazil win probability:")
    for fname, sv, val in shap_impact[:15]:
        direction = "↑ Brazil" if sv > 0 else "↓ Japan"
        print(f"    {fname:30s} SHAP={sv:+.4f}  val={val:.1f}  {direction}")

    print(f"\n  Top 10 features hurting Brazil win probability:")
    neg_impact = [x for x in shap_impact if x[1] < 0][:10]
    for fname, sv, val in neg_impact:
        print(f"    {fname:30s} SHAP={sv:+.4f}  val={val:.1f}  ↓ Japan")

    # ── Key feature values ──
    print(f"\n{'=' * 70}")
    print("KEY FEATURE VALUES — Brazil vs Japan")
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
    # Class 0 = Brazil win
    sv0 = sv_all[0] if len(sv_all.shape) == 3 else sv_all
    mean_shap = np.abs(sv0).mean(axis=0)
    shap_ranking = sorted(zip(fl, mean_shap), key=lambda x: -x[1])
    for fname, mshap in shap_ranking[:20]:
        bar = '█' * int(mshap / shap_ranking[0][1] * 30)
        print(f"  {fname:30s} {mshap:.4f} {bar}")

    # ── Save SHAP summary plot ──
    try:
        plt.figure(figsize=(12, 10))
        shap.summary_plot(sv0, X_sample, feature_names=fl, show=False, max_display=20)
        plt.title('SHAP Summary — Brazil win probability (class 0)')
        plt.tight_layout()
        plt.savefig('output/mc_2026/shap_brazil_win.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  SHAP summary plot saved to output/mc_2026/shap_brazil_win.png")
    except Exception as e:
        print(f"\n  Could not save SHAP plot: {e}")


GROUPS = {
    'A': {'Mexico': {'pts': 9, 'gd': 6}, 'South Africa': {'pts': 4, 'gd': -1},
           'Korea Republic': {'pts': 3, 'gd': -1}, 'Czech Republic': {'pts': 1, 'gd': -4},
           '_order': ['Mexico', 'South Africa', 'Korea Republic', 'Czech Republic']},
    'B': {'Switzerland': {'pts': 7, 'gd': 4}, 'Canada': {'pts': 4, 'gd': 5},
           'Bosnia and Herzegovina': {'pts': 4, 'gd': -1}, 'Qatar': {'pts': 1, 'gd': -8},
           '_order': ['Switzerland', 'Canada', 'Bosnia and Herzegovina', 'Qatar']},
    'C': {'Brazil': {'pts': 7, 'gd': 6}, 'Morocco': {'pts': 7, 'gd': 3},
           'Scotland': {'pts': 3, 'gd': -3}, 'Haiti': {'pts': 0, 'gd': -6},
           '_order': ['Brazil', 'Morocco', 'Scotland', 'Haiti']},
    'D': {'USA': {'pts': 6, 'gd': 4}, 'Australia': {'pts': 4, 'gd': 0},
           'Paraguay': {'pts': 4, 'gd': -2}, 'Turkey': {'pts': 3, 'gd': -2},
           '_order': ['USA', 'Australia', 'Paraguay', 'Turkey']},
    'E': {'Germany': {'pts': 6, 'gd': 6}, "Côte d'Ivoire": {'pts': 6, 'gd': 2},
           'Ecuador': {'pts': 4, 'gd': 0}, 'Curaçao': {'pts': 1, 'gd': -8},
           '_order': ['Germany', "Côte d'Ivoire", 'Ecuador', 'Curaçao']},
    'F': {'Netherlands': {'pts': 7, 'gd': 6}, 'Japan': {'pts': 5, 'gd': 4},
           'Sweden': {'pts': 4, 'gd': 0}, 'Tunisia': {'pts': 0, 'gd': -10},
           '_order': ['Netherlands', 'Japan', 'Sweden', 'Tunisia']},
    'G': {'Belgium': {'pts': 5, 'gd': 3, 'gf': 4}, 'Egypt': {'pts': 5, 'gd': 2, 'gf': 5},
           'IR Iran': {'pts': 3, 'gd': 0, 'gf': 3}, 'New Zealand': {'pts': 1, 'gd': -5, 'gf': 3},
           '_order': ['Belgium', 'Egypt', 'IR Iran', 'New Zealand']},
    'H': {'Spain': {'pts': 7, 'gd': 5}, 'Cape Verde': {'pts': 3, 'gd': 0},
           'Uruguay': {'pts': 2, 'gd': -1}, 'Saudi Arabia': {'pts': 2, 'gd': -4},
           '_order': ['Spain', 'Cape Verde', 'Uruguay', 'Saudi Arabia']},
    'I': {'France': {'pts': 9, 'gd': 8}, 'Norway': {'pts': 6, 'gd': 1},
           'Senegal': {'pts': 3, 'gd': 2}, 'Iraq': {'pts': 0, 'gd': -11},
           '_order': ['France', 'Norway', 'Senegal', 'Iraq']},
    'J': {'Argentina': {'pts': 6, 'gd': 5, 'gf': 5}, 'Austria': {'pts': 3, 'gd': 0, 'gf': 3},
           'Algeria': {'pts': 3, 'gd': -2, 'gf': 2}, 'Jordan': {'pts': 0, 'gd': -3, 'gf': 2}},
    'K': {'Colombia': {'pts': 6, 'gd': 3, 'gf': 4}, 'Portugal': {'pts': 4, 'gd': 5, 'gf': 6},
           'DR Congo': {'pts': 1, 'gd': -1, 'gf': 1}, 'Uzbekistan': {'pts': 0, 'gd': -7, 'gf': 1}},
    'L': {'England': {'pts': 4, 'gd': 2, 'gf': 4}, 'Ghana': {'pts': 4, 'gd': 1, 'gf': 1},
           'Croatia': {'pts': 3, 'gd': -1, 'gf': 3}, 'Panama': {'pts': 0, 'gd': -2, 'gf': 0}},
}


if __name__ == '__main__':
    import os
    os.makedirs('output/mc_2026', exist_ok=True)
    main()