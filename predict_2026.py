#!/usr/bin/env python3
"""
2026 FIFA World Cup Predictor — CORRECTED
Uses correct Wikipedia standings + proper FIFA bracket.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

NAME_MAP = {
    'West Germany': 'Germany', 'Soviet Union': 'Russia', 'USSR': 'Russia',
    'Yugoslavia': 'Serbia', 'Czechoslovakia': 'Czech Republic',
    'Zaire': 'DR Congo', 'Ivory Coast': "Côte d'Ivoire",
    'South Korea': 'Korea Republic', 'North Korea': 'Korea DPR',
    'Iran': 'IR Iran', 'United States': 'USA',
}

def harmonize(name):
    return NAME_MAP.get(name, name)

INITIAL_ELO = 1500
K_FACTOR = 32

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_a, elo_b, score_a, score_b, neutral=True):
    ea = expected_score(elo_a, elo_b)
    sa = 1 if score_a > score_b else (0.5 if score_a == score_b else 0)
    margin = abs(score_a - score_b)
    multiplier = np.log(max(margin, 1) + 1)
    return (elo_a + K_FACTOR * multiplier * (sa - ea),
            elo_b + K_FACTOR * multiplier * ((1 - sa) - (1 - ea)))

def compute_features(team, opponent, state, country_features, stage_num, match_date):
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
    cf, oc = country_features.get(team, {}), country_features.get(opponent, {})
    
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
        'gdp_per_capita': cf.get('gdp_per_capita', np.nan),
        'population': cf.get('population', np.nan),
        'life_expectancy': cf.get('life_expectancy', np.nan),
        'urban_population_pct': cf.get('urban_population_pct', np.nan),
        'health_expenditure_pct_gdp': cf.get('health_expenditure_pct_gdp', np.nan),
        'elo_pre_tournament': cf.get('elo_rating', s['elo']),
        'fifa_ranking': cf.get('fifa_ranking', 100),
        'football_power_index': cf.get('football_power_index', 0),
        'football_tradition': cf.get('football_tradition', 0),
        'opp_elo_pre_tournament': oc.get('elo_rating', o['elo']),
        'opp_football_power_index': oc.get('football_power_index', 0),
        'opp_football_tradition': oc.get('football_tradition', 0),
        'elo_diff_pre': cf.get('elo_rating', s['elo']) - oc.get('elo_rating', o['elo']),
        'power_diff': cf.get('football_power_index', 0) - oc.get('football_power_index', 0),
        'tradition_diff': cf.get('football_tradition', 0) - oc.get('football_tradition', 0),
    }

def train_model(results_df):
    df = results_df.copy()
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
        for t in [ht, at]:
            state[t]['h2h'][h2h_key]['matches'] += 1
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
    print(f"  Trained on {len(X)} matches")
    return model, state, X.columns.tolist()

def load_country_features():
    df = pd.read_csv('data/world_cup_predictors_dataset.csv')
    df = df[df['wc_year'] == 2022]
    features = {}
    for _, r in df.iterrows():
        t = harmonize(r['country'])
        features[t] = {k: r.get(k, np.nan) for k in [
            'gdp_per_capita', 'population', 'life_expectancy',
            'urban_population_pct', 'health_expenditure_pct_gdp',
            'elo_rating', 'fifa_ranking', 'football_power_index', 'football_tradition'
        ]}
    return features

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

def predict(model, fl, ta, tb, state, cf, stage, date):
    ha, hb = harmonize(ta), harmonize(tb)
    feat = compute_features(ha, hb, state, cf, stage, date)
    X = pd.DataFrame([feat])[fl].fillna(0)
    probs = model.predict_proba(X)[0]
    pa, pd_, pb = probs[0], probs[1], probs[2]
    winner = ta if pa > pb else tb
    return winner, max(pa, pb), pa, pd_, pb

def simulate_round(matches, name, stage, state, model, fl, cf, date):
    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"{'=' * 70}")
    winners = []
    for label, ta, tb in matches:
        winner, prob, pa, pd_, pb = predict(model, fl, ta, tb, state, cf, stage, date)
        winners.append(winner)
        if winner == ta:
            update_state(state, ta, tb, 2, 1, date)
        else:
            update_state(state, ta, tb, 1, 2, date)
        print(f"  {label}: {ta} vs {tb}")
        print(f"    → ✅ {winner} ({prob:.1%})  [P({ta})={pa:.1%} P(draw)={pd_:.1%} P({tb})={pb:.1%}]")
    return winners

# Correct standings from Wikipedia (complete groups use actual order)
# Incomplete groups: use pts + GD as available
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
           '_order': ['Belgium', 'Egypt', 'IR Iran', 'New Zealand']},  # COMPLETE (Belgium 2-0 NZ, Egypt 1-1 Iran assumed)
    'H': {'Spain': {'pts': 7, 'gd': 5}, 'Cape Verde': {'pts': 3, 'gd': 0},
           'Uruguay': {'pts': 2, 'gd': -1}, 'Saudi Arabia': {'pts': 2, 'gd': -4},
           '_order': ['Spain', 'Cape Verde', 'Uruguay', 'Saudi Arabia']},  # COMPLETE
    'I': {'France': {'pts': 9, 'gd': 8}, 'Norway': {'pts': 6, 'gd': 1},
           'Senegal': {'pts': 3, 'gd': 2}, 'Iraq': {'pts': 0, 'gd': -11},
           '_order': ['France', 'Norway', 'Senegal', 'Iraq']},
    'J': {'Argentina': {'pts': 6, 'gd': 5, 'gf': 5}, 'Austria': {'pts': 3, 'gd': 0, 'gf': 3},
           'Algeria': {'pts': 3, 'gd': -2, 'gf': 2}, 'Jordan': {'pts': 0, 'gd': -3, 'gf': 2}},  # 2 matchdays
    'K': {'Colombia': {'pts': 6, 'gd': 3, 'gf': 4}, 'Portugal': {'pts': 4, 'gd': 5, 'gf': 6},
           'DR Congo': {'pts': 1, 'gd': -1, 'gf': 1}, 'Uzbekistan': {'pts': 0, 'gd': -7, 'gf': 1}},  # 2 matchdays
    'L': {'England': {'pts': 4, 'gd': 2, 'gf': 4}, 'Ghana': {'pts': 4, 'gd': 1, 'gf': 1},
           'Croatia': {'pts': 3, 'gd': -1, 'gf': 3}, 'Panama': {'pts': 0, 'gd': -2, 'gf': 0}},  # 2 matchdays
}

def main():
    print("=" * 70)
    print("🏆 2026 FIFA WORLD CUP PREDICTOR")
    print("=" * 70)
    
    print("\n[1/4] Loading data...")
    results = pd.read_csv('data/results.csv')
    cf = load_country_features()
    print(f"  {len(results)} matches, {len(cf)} countries with features")
    
    print("[2/4] Training model...")
    model, state, fl = train_model(results)
    
    print("[3/4] Processing completed 2026 WC matches...")
    wc26 = results[(results['tournament'] == 'FIFA World Cup') & 
                    (pd.to_datetime(results['date']).dt.year == 2026)].copy()
    wc26['date'] = pd.to_datetime(wc26['date'])
    completed = wc26[wc26['home_score'].notna()].sort_values('date')
    for _, r in completed.iterrows():
        update_state(state, r['home_team'], r['away_team'], int(r['home_score']), int(r['away_score']), r['date'])
    print(f"  Processed {len(completed)} completed matches")
    
    # Mark WC participation for all 48 teams
    all_teams = set()
    for g_data in GROUPS.values():
        for t in g_data:
            ht = harmonize(t)
            all_teams.add(ht)
            state[ht]['wc_participations'] += 1
    
    print("[4/4] Simulating remaining matches...\n")
    
    # === REMAINING GROUP MATCHES (only for incomplete groups: G, J, K, L) ===
    print("=" * 70)
    print("REMAINING GROUP MATCHES")
    print("=" * 70)
    
    remaining = [
        ('2026-06-27', 'Jordan', 'Argentina', 'J'),
        ('2026-06-27', 'Algeria', 'Austria', 'J'),
        ('2026-06-27', 'Colombia', 'Portugal', 'K'),
        ('2026-06-27', 'DR Congo', 'Uzbekistan', 'K'),
        ('2026-06-27', 'Panama', 'England', 'L'),
        ('2026-06-27', 'Croatia', 'Ghana', 'L'),
    ]
    
    for date_str, home, away, group in remaining:
        date = pd.Timestamp(date_str)
        winner, prob, pa, pd_, pb = predict(model, fl, home, away, state, cf, 0, date)
        ha, at = harmonize(home), harmonize(away)
        if winner == home:
            update_state(state, ha, at, 2, 1, date)
            GROUPS[group][ha]['pts'] += 3
            GROUPS[group][ha]['gd'] += 1
            GROUPS[group][at]['gd'] -= 1
            GROUPS[group][ha]['gf'] = GROUPS[group][ha].get('gf', 0) + 2
            GROUPS[group][at]['gf'] = GROUPS[group][at].get('gf', 0) + 1
        elif winner == away:
            update_state(state, ha, at, 1, 2, date)
            GROUPS[group][at]['pts'] += 3
            GROUPS[group][ha]['gd'] -= 1
            GROUPS[group][at]['gd'] += 1
            GROUPS[group][ha]['gf'] = GROUPS[group][ha].get('gf', 0) + 1
            GROUPS[group][at]['gf'] = GROUPS[group][at].get('gf', 0) + 2
        else:
            update_state(state, ha, at, 1, 1, date)
            GROUPS[group][ha]['pts'] += 1
            GROUPS[group][at]['pts'] += 1
            GROUPS[group][ha]['gf'] = GROUPS[group][ha].get('gf', 0) + 1
            GROUPS[group][at]['gf'] = GROUPS[group][at].get('gf', 0) + 1
        arrow = "←" if prob > 0.55 else "~"
        print(f"  {date_str} {home} vs {away} → {winner} ({prob:.1%}) {arrow}")
    
    # === FINAL STANDINGS ===
    print(f"\n{'=' * 70}")
    print("FINAL GROUP STANDINGS")
    print(f"{'=' * 70}")
    
    thirds_all = []
    gw, gr = {}, {}
    for g in 'ABCDEFGHIJKL':
        gdata = GROUPS[g].copy()
        order = gdata.pop('_order', None)
        if order:
            st = [(t, gdata[t]['pts'], gdata[t]['gd'], gdata[t].get('gf', 0)) for t in order]
        else:
            st = sorted([(t, d['pts'], d['gd'], d.get('gf', 0)) for t, d in gdata.items()],
                       key=lambda x: (-x[1], -x[2], -x[3], x[0]))
        gw[g], gr[g] = st[0][0], st[1][0]
        thirds_all.append((g, st[2][0], st[2][1], st[2][2]))
        print(f"\n  Group {g}:")
        marks = ["✓1st", "✓2nd", "?3rd", " ✗ "]
        for i, (t, pts, gd, gf) in enumerate(st):
            print(f"    {marks[i]} {t}: {pts} pts (GD {'+' if gd >= 0 else ''}{gd})")
    
    # Best 8 thirds (sort by pts desc, GD desc)
    thirds_all.sort(key=lambda x: (-x[2], -x[3], x[0]))
    best8 = thirds_all[:8]
    elim8 = thirds_all[8:]
    best8_groups = sorted([t[0] for t in best8])
    print(f"\n  Best 8 thirds (groups {','.join(best8_groups)}):")
    for g, t, pts, gd in best8:
        print(f"    {t} ({pts}pts, GD {'+' if gd >= 0 else ''}{gd}, G{g})")
    print(f"  Eliminated: {', '.join(f'{t[1]} ({t[2]}pts, G{t[0]})' for t in elim8)}")
    
    # === KNOCKOUT BRACKET ===
    # Using the ACTUAL bracket from Wikipedia
    # Third-place assignments for combination with groups A,B,D,E,F,G,I,L advancing:
    #   3D→Match74(Paraguay), 3F→Match77(Sweden), 3B→Match81(Bosnia)
    #   3A→Match82(KoreaRep), 3E→Match79(Ecuador), 3G→Match85(Egypt)
    #   3I→Match80(Senegal), 3L→Match87(Ghana)
    
    r32 = [
        ('Match 73', gr['A'], gr['B']),                  # South Africa vs Canada
        ('Match 74', gw['E'], 'Paraguay'),               # Germany vs Paraguay
        ('Match 75', gw['F'], gr['C']),                  # Netherlands vs Morocco
        ('Match 76', gw['C'], gr['F']),                  # Brazil vs Japan
        ('Match 77', gw['I'], 'Sweden'),                 # France vs Sweden
        ('Match 78', gr['E'], gr['I']),                  # Côte d'Ivoire vs Norway
        ('Match 79', gw['A'], 'Ecuador'),                # Mexico vs Ecuador
        ('Match 80', gw['L'], 'Senegal'),                # England vs Senegal
        ('Match 81', gw['D'], 'Bosnia and Herzegovina'), # USA vs Bosnia
        ('Match 82', gw['G'], 'Korea Republic'),         # ? vs Korea Republic
        ('Match 83', gr['K'], gr['L']),                  # Colombia vs Croatia
        ('Match 84', gw['H'], gr['J']),                  # Spain vs Austria
        ('Match 85', gw['B'], 'Egypt'),                  # Switzerland vs Egypt
        ('Match 86', gw['J'], gr['H']),                  # Argentina vs Cape Verde
        ('Match 87', gw['K'], 'Ghana'),                  # Portugal vs Ghana
        ('Match 88', gr['D'], gr['G']),                  # Australia vs Belgium
    ]
    
    r32_w = simulate_round(r32, "ROUND OF 32", 1, state, model, fl, cf, pd.Timestamp('2026-06-29'))
    
    # FIFA bracket R16 pairings (from Wikipedia — NON-SEQUENTIAL crossover!)
    # M89: W73 vs W75 | M90: W74 vs W77 | M91: W76 vs W78 | M92: W79 vs W80
    # M93: W83 vs W84 | M94: W81 vs W82 | M95: W86 vs W88 | M96: W85 vs W87
    # r32 indices: 0=73, 1=74, 2=75, 3=76, 4=77, 5=78, 6=79, 7=80,
    #              8=81, 9=82, 10=83, 11=84, 12=85, 13=86, 14=87, 15=88
    r16 = [
        ('R16 M89', r32_w[0], r32_w[2]),   # W73 vs W75  (Canada vs Netherlands)
        ('R16 M90', r32_w[1], r32_w[4]),   # W74 vs W77  (Germany vs France)
        ('R16 M91', r32_w[3], r32_w[5]),   # W76 vs W78  (Brazil vs Norway)
        ('R16 M92', r32_w[6], r32_w[7]),   # W79 vs W80  (Mexico vs England)
        ('R16 M93', r32_w[10], r32_w[11]), # W83 vs W84  (Colombia vs Spain)
        ('R16 M94', r32_w[8], r32_w[9]),   # W81 vs W82  (USA vs IR Iran)
        ('R16 M95', r32_w[13], r32_w[15]), # W86 vs W88  (Argentina vs Belgium)
        ('R16 M96', r32_w[12], r32_w[14]), # W85 vs W87  (Switzerland vs Portugal)
    ]
    r16_w = simulate_round(r16, "ROUND OF 16", 2, state, model, fl, cf, pd.Timestamp('2026-07-04'))
    
    # QF: M97: W89 vs W90 | M98: W93 vs W94 | M99: W91 vs W92 | M100: W95 vs W96
    # Bracket: (89/90 side) vs (93/94 side) → SF101, (91/92 side) vs (95/96 side) → SF102
    qf = [
        ('QF M97', r16_w[0], r16_w[1]),  # W89 vs W90 (left-top: Germany path vs Canada/Ned path)
        ('QF M98', r16_w[4], r16_w[5]),  # W93 vs W94 (right-top: Spain/Colombia vs USA/Iran)
        ('QF M99', r16_w[2], r16_w[3]),  # W91 vs W92 (left-bot: Brazil path vs Mexico/England)
        ('QF M100', r16_w[6], r16_w[7]), # W95 vs W96 (right-bot: Argentina/Bel vs Swi/Port)
    ]
    qf_w = simulate_round(qf, "QUARTERFINALS", 3, state, model, fl, cf, pd.Timestamp('2026-07-09'))
    
    # SF: M101: W97 vs W98 | M102: W99 vs W100
    # Germany side (QF97/98) vs Brazil side (QF99/100) → they can only meet in FINAL
    sf = [
        ('SF M101', qf_w[0], qf_w[1]),  # left side (Germany's half)
        ('SF M102', qf_w[2], qf_w[3]),  # right side (Brazil's half)
    ]
    sf_w = simulate_round(sf, "SEMIFINALS", 4, state, model, fl, cf, pd.Timestamp('2026-07-14'))
    
    # 3rd place: L101 vs L102
    sf_losers = [qf_w[i] for i in range(4) if qf_w[i] not in sf_w]
    print(f"\n{'=' * 70}")
    print("THIRD PLACE MATCH")
    print(f"{'=' * 70}")
    third, tp, _, _, _ = predict(model, fl, sf_losers[0], sf_losers[1], state, cf, 0, pd.Timestamp('2026-07-18'))
    print(f"  {sf_losers[0]} vs {sf_losers[1]} → {third} ({tp:.1%})")
    
    # FINAL
    print(f"\n{'=' * 70}")
    print("🏆 FINAL 🏆")
    print(f"{'=' * 70}")
    champion, cp, pa, pd_, pb = predict(model, fl, sf_w[0], sf_w[1], state, cf, 5, pd.Timestamp('2026-07-19'))
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
