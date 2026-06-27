#!/usr/bin/env python3
"""
Monte Carlo simulation of the 2026 World Cup.
Trains model ONCE, then runs N simulations by sampling from
probability distributions instead of argmax.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import defaultdict, Counter
import copy, warnings, time
warnings.filterwarnings('ignore')

N_SIMS = 1000
np.random.seed(42)

# ── Constants ──
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
    return model.predict_proba(X)[0]  # [p_home_win, p_draw, p_away_win]

def sample_ko(probs, ta, tb):
    """Sample knockout result (no draws)."""
    outcome = np.random.choice(3, p=probs)
    if outcome == 0:
        return ta, 2, 1
    elif outcome == 1:
        p_ta = probs[0] / (probs[0] + probs[2])
        return (ta if np.random.random() < p_ta else tb), 1, 1
    else:
        return tb, 1, 2

def sample_group(probs, ta, tb):
    """Sample group result (draws allowed)."""
    outcome = np.random.choice(3, p=probs)
    if outcome == 0:
        return ta, 2, 1
    elif outcome == 1:
        return None, 1, 1
    else:
        return tb, 1, 2

# ── Group standings (Belgium 1st in G) ──
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

REMAINING_GROUPS = [
    ('2026-06-27', 'Jordan', 'Argentina', 'J'),
    ('2026-06-27', 'Algeria', 'Austria', 'J'),
    ('2026-06-27', 'Colombia', 'Portugal', 'K'),
    ('2026-06-27', 'DR Congo', 'Uzbekistan', 'K'),
    ('2026-06-27', 'Panama', 'England', 'L'),
    ('2026-06-27', 'Croatia', 'Ghana', 'L'),
]

THIRD_MAP = {
    'A': 19, 'B': 17, 'D': 3, 'E': 13, 'F': 9, 'G': 25, 'I': 15, 'L': 29,
}


def run_one_sim(model, fl, base_state, cf):
    state = copy.deepcopy(base_state)
    groups = copy.deepcopy(GROUPS)

    # Remaining group matches
    for date_str, home, away, group in REMAINING_GROUPS:
        date = pd.Timestamp(date_str)
        ha, at = harmonize(home), harmonize(away)
        probs = predict_probs(model, fl, home, away, state, cf, 0, date)
        winner, sa, sb = sample_group(probs, home, away)
        update_state(state, ha, at, sa, sb, date)
        if winner == home:
            groups[group][ha]['pts'] += 3; groups[group][ha]['gd'] += 1; groups[group][at]['gd'] -= 1
            groups[group][ha]['gf'] = groups[group][ha].get('gf', 0) + sa
            groups[group][at]['gf'] = groups[group][at].get('gf', 0) + sb
        elif winner == away:
            groups[group][at]['pts'] += 3; groups[group][ha]['gd'] -= 1; groups[group][at]['gd'] += 1
            groups[group][ha]['gf'] = groups[group][ha].get('gf', 0) + sa
            groups[group][at]['gf'] = groups[group][at].get('gf', 0) + sb
        else:
            groups[group][ha]['pts'] += 1; groups[group][at]['pts'] += 1
            groups[group][ha]['gf'] = groups[group][ha].get('gf', 0) + sa
            groups[group][at]['gf'] = groups[group][at].get('gf', 0) + sb

    # Final standings
    thirds_all = []
    gw, gr = {}, {}
    for g in 'ABCDEFGHIJKL':
        gdata = groups[g].copy()
        order = gdata.pop('_order', None)
        if order:
            st = [(t, gdata[t]['pts'], gdata[t]['gd'], gdata[t].get('gf', 0)) for t in order]
        else:
            st = sorted([(t, d['pts'], d['gd'], d.get('gf', 0)) for t, d in gdata.items()],
                       key=lambda x: (-x[1], -x[2], -x[3], x[0]))
        # Re-sort if stochastic results changed standings
        st.sort(key=lambda x: (-x[1], -x[2], -x[3]))
        gw[g], gr[g] = st[0][0], st[1][0]
        thirds_all.append((g, st[2][0], st[2][1], st[2][2]))

    thirds_all.sort(key=lambda x: (-x[2], -x[3], x[0]))
    best8_groups = set(t[0] for t in thirds_all[:8])

    # Build R32
    third_teams = {}
    for g, team, pts, gd in thirds_all[:8]:
        match_num = THIRD_MAP.get(g)
        if match_num:
            third_teams[match_num] = team

    r32_base = [
        gr['A'], gr['B'],  # 0,1
        gw['E'], 'PAR',    # 2,3 (placeholder)
        gw['F'], gr['C'],  # 4,5
        gw['C'], gr['F'],  # 6,7
        gw['I'], 'SWE',    # 8,9
        gr['E'], gr['I'],  # 10,11
        gw['A'], 'ECU',    # 12,13
        gw['L'], 'SEN',    # 14,15
        gw['D'], 'BIH',    # 16,17
        gw['G'], 'KOR',    # 18,19
        gr['K'], gr['L'],  # 20,21
        gw['H'], gr['J'],  # 22,23
        gw['B'], 'EGY',    # 24,25
        gw['J'], gr['H'],  # 26,27
        gw['K'], 'GHA',    # 28,29
        gr['D'], gr['G'],  # 30,31
    ]
    # Fill placeholders with third-place teams
    placeholder_map = {3: 74, 9: 77, 13: 79, 15: 80, 17: 81, 19: 82, 25: 85, 29: 87}
    for idx, match_num in placeholder_map.items():
        if match_num in third_teams:
            r32_base[idx] = third_teams[match_num]

    # Simulate R32
    r32_date = pd.Timestamp('2026-06-29')
    r32_winners = []
    for i in range(0, 32, 2):
        ta, tb = r32_base[i], r32_base[i+1]
        probs = predict_probs(model, fl, ta, tb, state, cf, 1, r32_date)
        winner, sa, sb = sample_ko(probs, ta, tb)
        r32_winners.append(winner)
        update_state(state, harmonize(ta), harmonize(tb), sa, sb, r32_date)

    # R16
    r16_pairs = [
        (r32_winners[0], r32_winners[2]),   # W73 vs W75
        (r32_winners[1], r32_winners[4]),   # W74 vs W77
        (r32_winners[3], r32_winners[5]),   # W76 vs W78
        (r32_winners[6], r32_winners[7]),   # W79 vs W80
        (r32_winners[10], r32_winners[11]), # W83 vs W84
        (r32_winners[8], r32_winners[9]),   # W81 vs W82
        (r32_winners[13], r32_winners[15]), # W86 vs W88
        (r32_winners[12], r32_winners[14]), # W85 vs W87
    ]
    r16_date = pd.Timestamp('2026-07-04')
    r16_winners = []
    for ta, tb in r16_pairs:
        probs = predict_probs(model, fl, ta, tb, state, cf, 2, r16_date)
        winner, sa, sb = sample_ko(probs, ta, tb)
        r16_winners.append(winner)
        update_state(state, harmonize(ta), harmonize(tb), sa, sb, r16_date)

    # QF
    qf_pairs = [
        (r16_winners[0], r16_winners[1]),
        (r16_winners[4], r16_winners[5]),
        (r16_winners[2], r16_winners[3]),
        (r16_winners[6], r16_winners[7]),
    ]
    qf_date = pd.Timestamp('2026-07-09')
    qf_winners = []
    for ta, tb in qf_pairs:
        probs = predict_probs(model, fl, ta, tb, state, cf, 3, qf_date)
        winner, sa, sb = sample_ko(probs, ta, tb)
        qf_winners.append(winner)
        update_state(state, harmonize(ta), harmonize(tb), sa, sb, qf_date)

    # SF
    sf_pairs = [
        (qf_winners[0], qf_winners[1]),
        (qf_winners[2], qf_winners[3]),
    ]
    sf_date = pd.Timestamp('2026-07-14')
    sf_winners, sf_losers = [], []
    for ta, tb in sf_pairs:
        probs = predict_probs(model, fl, ta, tb, state, cf, 4, sf_date)
        winner, sa, sb = sample_ko(probs, ta, tb)
        sf_winners.append(winner)
        sf_losers.append(tb if winner == ta else ta)
        update_state(state, harmonize(ta), harmonize(tb), sa, sb, sf_date)

    # 3rd place
    probs = predict_probs(model, fl, sf_losers[0], sf_losers[1], state, cf, 0, pd.Timestamp('2026-07-18'))
    third_winner, _, _ = sample_ko(probs, sf_losers[0], sf_losers[1])
    third_loser = sf_losers[1] if third_winner == sf_losers[0] else sf_losers[0]

    # Final
    probs = predict_probs(model, fl, sf_winners[0], sf_winners[1], state, cf, 5, pd.Timestamp('2026-07-19'))
    champion, _, _ = sample_ko(probs, sf_winners[0], sf_winners[1])
    runner_up = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]

    # Collect Brazil matches
    brazil_matches = []
    for round_name, pairs, winners in [
        ('R32', [(r32_base[i], r32_base[i+1]) for i in range(0, 32, 2)], r32_winners),
        ('R16', r16_pairs, r16_winners),
        ('QF', qf_pairs, qf_winners),
        ('SF', sf_pairs, sf_winners),
    ]:
        for (ta, tb), w in zip(pairs, winners):
            if 'Brazil' in (ta, tb):
                opp = tb if ta == 'Brazil' else ta
                brazil_matches.append({'round': round_name, 'opponent': opp, 'result': 'W' if w == 'Brazil' else 'L'})

    if champion == 'Brazil':
        brazil_matches.append({'round': 'Final', 'opponent': runner_up, 'result': 'W'})
    elif runner_up == 'Brazil':
        brazil_matches.append({'round': 'Final', 'opponent': champion, 'result': 'L'})
    elif third_winner == 'Brazil':
        brazil_matches.append({'round': '3rd Place', 'opponent': third_loser, 'result': 'W'})
    elif third_loser == 'Brazil':
        brazil_matches.append({'round': '3rd Place', 'opponent': third_winner, 'result': 'L'})

    return {
        'champion': champion, 'runner_up': runner_up,
        'third': third_winner, 'fourth': third_loser,
        'brazil_matches': brazil_matches,
        'brazil_reached': brazil_matches[-1]['round'] if brazil_matches else 'Group',
    }


def main():
    print(f"Running {N_SIMS} Monte Carlo simulations...\n")

    results = pd.read_csv('data/results.csv')
    # Load country features
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

    # Train model once
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
    print(f"  Model trained on {len(X)} matches, {len(fl)} features")

    # Process completed 2026 matches into state
    wc26 = results[(results['tournament'] == 'FIFA World Cup') &
                    (pd.to_datetime(results['date']).dt.year == 2026)].copy()
    wc26['date'] = pd.to_datetime(wc26['date'])
    completed = wc26[wc26['home_score'].notna()].sort_values('date')
    for _, r in completed.iterrows():
        update_state(state, r['home_team'], r['away_team'],
                     int(r['home_score']), int(r['away_score']), r['date'])
    print(f"  Processed {len(completed)} completed 2026 WC matches")

    # Mark WC participation
    for g_data in GROUPS.values():
        for t in g_data:
            if t.startswith('_'): continue
            state[harmonize(t)]['wc_participations'] += 1

    base_state = copy.deepcopy(state)

    # ── Run simulations ──
    champions = Counter()
    runner_ups = Counter()
    thirds = Counter()
    fourths = Counter()
    brazil_reached = Counter()
    brazil_opponents = defaultdict(Counter)
    brazil_results = defaultdict(Counter)
    brazil_probs_by_round = defaultdict(list)  # round -> list of probs for Brazil matches

    start = time.time()
    for i in range(N_SIMS):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (N_SIMS - i - 1) / rate
            print(f"  [{i+1}/{N_SIMS}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

        r = run_one_sim(model, fl, base_state, cf)
        champions[r['champion']] += 1
        runner_ups[r['runner_up']] += 1
        thirds[r['third']] += 1
        fourths[r['fourth']] += 1
        brazil_reached[r['brazil_reached']] += 1
        for m in r['brazil_matches']:
            brazil_opponents[m['round']][m['opponent']] += 1
            brazil_results[m['round']][m['result']] += 1

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/N_SIMS:.2f}s per sim)\n")

    # ── Output ──
    print("=" * 70)
    print(f"CHAMPION DISTRIBUTION ({N_SIMS} simulations)")
    print("=" * 70)
    for team, count in champions.most_common(15):
        pct = count / N_SIMS * 100
        ci_low = max(0, pct - 1.96 * np.sqrt(pct*(100-pct)/N_SIMS))
        ci_high = min(100, pct + 1.96 * np.sqrt(pct*(100-pct)/N_SIMS))
        bar = '█' * int(pct / 2)
        print(f"  {team:25s} {count:4d} ({pct:5.1f}% [{ci_low:.1f}-{ci_high:.1f}]) {bar}")

    print(f"\n{'=' * 70}")
    print(f"BRAZIL — TOURNAMENT PATH ({N_SIMS} simulations)")
    print(f"{'=' * 70}")

    print(f"\n  Furthest round reached:")
    for stage in ['R32', 'R16', 'QF', 'SF', '3rd Place', 'Final']:
        count = brazil_reached.get(stage, 0)
        pct = count / N_SIMS * 100
        bar = '█' * int(pct / 2)
        print(f"    {stage:12s} {count:4d} ({pct:5.1f}%) {bar}")

    br_champ = champions.get('Brazil', 0)
    br_final = br_champ + runner_ups.get('Brazil', 0)
    br_semis = br_final + thirds.get('Brazil', 0) + fourths.get('Brazil', 0)
    print(f"\n  Brazil champion: {br_champ} ({br_champ/N_SIMS*100:.1f}%)")
    print(f"  Brazil in final: {br_final} ({br_final/N_SIMS*100:.1f}%)")
    print(f"  Brazil in semis+: {br_semis} ({br_semis/N_SIMS*100:.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"BRAZIL vs JAPAN (R32 — NEXT MATCH)")
    print(f"{'=' * 70}")
    w = brazil_results['R32'].get('W', 0)
    l = brazil_results['R32'].get('L', 0)
    total = w + l
    if total > 0:
        w_pct = w / total * 100
        ci_low = max(0, w_pct - 1.96 * np.sqrt(w_pct*(100-w_pct)/total))
        ci_high = min(100, w_pct + 1.96 * np.sqrt(w_pct*(100-w_pct)/total))
        print(f"  Brazil advances: {w} ({w_pct:.1f}% [{ci_low:.1f}-{ci_high:.1f}])")
        print(f"  Japan advances:  {l} ({100-w_pct:.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"BRAZIL WIN/LOSS BY ROUND")
    print(f"{'=' * 70}")
    for round_name in ['R32', 'R16', 'QF', 'SF', '3rd Place', 'Final']:
        if brazil_results[round_name]:
            w = brazil_results[round_name].get('W', 0)
            l = brazil_results[round_name].get('L', 0)
            t = w + l
            print(f"  {round_name:12s}  W:{w:4d} ({w/t*100:.1f}%)  L:{l:4d} ({l/t*100:.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"BRAZIL OPPONENTS BY ROUND")
    print(f"{'=' * 70}")
    for round_name in ['R32', 'R16', 'QF', 'SF', '3rd Place', 'Final']:
        if brazil_opponents[round_name]:
            print(f"\n  {round_name}:")
            total_opp = sum(brazil_opponents[round_name].values())
            for opp, count in brazil_opponents[round_name].most_common(10):
                print(f"    vs {opp:25s} {count:4d} ({count/total_opp*100:.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"TOP-4 PROBABILITY")
    print(f"{'=' * 70}")
    top4 = Counter()
    for d in [champions, runner_ups, thirds, fourths]:
        for team, count in d.items():
            top4[team] += count
    for team, count in top4.most_common(15):
        pct = count / (N_SIMS * 4) * 100
        print(f"  {team:25s} {count:5d} ({pct:5.1f}%)")

    # Runner-up distribution
    print(f"\n{'=' * 70}")
    print(f"RUNNER-UP DISTRIBUTION")
    print(f"{'=' * 70}")
    for team, count in runner_ups.most_common(10):
        pct = count / N_SIMS * 100
        print(f"  {team:25s} {count:4d} ({pct:5.1f}%)")


if __name__ == '__main__':
    main()