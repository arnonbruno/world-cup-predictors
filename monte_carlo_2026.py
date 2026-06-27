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
from shared import (
    GROUP_2026_TEAMS,
    WC2026_STAGE_TO_TRAIN,
    apply_group_result,
    apply_match_to_state,
    build_2026_group_state,
    build_round_of_32,
    compute_match_features,
    country_features_for_year,
    fit_xgb_with_validation,
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
    make_team_state,
    parse_bool,
    load_country_feature_history,
    rank_third_place_teams,
    sorted_group_standings,
)
warnings.filterwarnings('ignore')

N_SIMS = 1000
np.random.seed(42)

INITIAL_ELO = 1500
K_FACTOR = 32

def harmonize(name):
    return harmonize_country(name)

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_a, elo_b, score_a, score_b, neutral=True):
    home_advantage = 0 if neutral else 50
    ea = expected_score(elo_a + home_advantage, elo_b)
    sa = 1 if score_a > score_b else (0.5 if score_a == score_b else 0)
    margin = abs(score_a - score_b)
    multiplier = np.log(max(margin, 1) + 1)
    return (elo_a + K_FACTOR * multiplier * (sa - ea),
            elo_b + K_FACTOR * multiplier * ((1 - sa) - (1 - ea)))

_ODDS = None
_POISSON = None
_ALPHA = 1.0
_SQUAD_VALUES = None

def compute_features(team, opponent, state, country_features, stage_num, match_date, neutral=True, is_home=False, odds_row=None, squad_values=None):
    return compute_match_features(team, opponent, state, country_features, stage_num, match_date, neutral, is_home, odds_row, squad_values)

def update_state(state, ta, tb, sa, sb, date, neutral=True):
    apply_match_to_state(state, ta, tb, sa, sb, date, neutral=neutral, is_world_cup=True)

def predict_probs(model, fl, ta, tb, state, cf, stage, date):
    ha, hb = harmonize(ta), harmonize(tb)
    odds_row = odds_features_for_match(_ODDS, date, ha, hb)
    feat = compute_features(ha, hb, state, cf, stage, date, odds_row=odds_row, squad_values=_SQUAD_VALUES)
    X = prepare_prediction_frame(feat, fl)
    probs = np.asarray(model.predict_proba(X)[0], dtype=float)  # [p_home_win, p_draw, p_away_win]
    if _POISSON is not None and _ALPHA < 1.0:
        probs = blend_probabilities(probs, _POISSON.outcome_probs(ha, hb, neutral=True), _ALPHA)
    return probs

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

def run_one_sim(model, fl, base_state, cf, base_groups, remaining_matches):
    state = copy.deepcopy(base_state)
    groups = copy.deepcopy(base_groups)

    # Remaining group matches
    for date, home, away, group in remaining_matches:
        ha, at = harmonize(home), harmonize(away)
        probs = predict_probs(model, fl, home, away, state, cf, 0, date)
        winner, sa, sb = sample_group(probs, home, away)
        update_state(state, ha, at, sa, sb, date)
        apply_group_result(groups, group, ha, at, sa, sb)

    # Final standings
    standings_by_group = {}
    gw, gr = {}, {}
    for g in 'ABCDEFGHIJKL':
        st = sorted_group_standings(groups[g])
        standings_by_group[g] = st
        gw[g], gr[g] = st[0][0], st[1][0]

    thirds_all = rank_third_place_teams(standings_by_group)
    r32_matches = build_round_of_32(gw, gr, thirds_all[:8])

    # Simulate R32
    r32_date = pd.Timestamp('2026-06-29')
    r32_winners = []
    for _, ta, tb in r32_matches:
        probs = predict_probs(model, fl, ta, tb, state, cf, WC2026_STAGE_TO_TRAIN["round_of_32"], r32_date)
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
        probs = predict_probs(model, fl, ta, tb, state, cf, WC2026_STAGE_TO_TRAIN["round_of_16"], r16_date)
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
        probs = predict_probs(model, fl, ta, tb, state, cf, WC2026_STAGE_TO_TRAIN["quarterfinal"], qf_date)
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
        probs = predict_probs(model, fl, ta, tb, state, cf, WC2026_STAGE_TO_TRAIN["semifinal"], sf_date)
        winner, sa, sb = sample_ko(probs, ta, tb)
        sf_winners.append(winner)
        sf_losers.append(tb if winner == ta else ta)
        update_state(state, harmonize(ta), harmonize(tb), sa, sb, sf_date)

    # 3rd place
    probs = predict_probs(model, fl, sf_losers[0], sf_losers[1], state, cf, WC2026_STAGE_TO_TRAIN["third_place"], pd.Timestamp('2026-07-18'))
    third_winner, _, _ = sample_ko(probs, sf_losers[0], sf_losers[1])
    third_loser = sf_losers[1] if third_winner == sf_losers[0] else sf_losers[0]

    # Final
    probs = predict_probs(model, fl, sf_winners[0], sf_winners[1], state, cf, WC2026_STAGE_TO_TRAIN["final"], pd.Timestamp('2026-07-19'))
    champion, _, _ = sample_ko(probs, sf_winners[0], sf_winners[1])
    runner_up = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]

    # Collect Brazil matches
    brazil_matches = []
    for round_name, pairs, winners in [
        ('R32', [(ta, tb) for _, ta, tb in r32_matches], r32_winners),
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

    global _ODDS, _POISSON, _ALPHA, _SQUAD_VALUES
    results = pd.read_csv('data/results.csv')
    country_history = load_country_feature_history()
    cf = country_features_for_year(country_history, 2022)
    base_groups, remaining_matches = build_2026_group_state(results)
    _ODDS = load_betting_odds()
    _SQUAD_VALUES = load_squad_values()

    # Train model once
    df = results.copy()
    df['date'] = pd.to_datetime(df['date'])
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
        neutral = parse_bool(r.get('neutral', True))
        stage = wc_stage_by_index.get(int(r.name), 0) if is_world_cup else 0
        odds_row = odds_features_for_match(_ODDS, r['date'], ht, at)
        rows.append(compute_features(ht, at, state, country_feature_cache[feature_year], stage, r['date'], neutral, not neutral, odds_row, _SQUAD_VALUES))
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
    y = np.array(labels)
    weights = sample_weights(y, feature_dates)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               objective='multi:softprob', num_class=3,
                               eval_metric='mlogloss', random_state=42, verbosity=0)
    model, _metrics = fit_xgb_with_validation(model, X, y, label="XGBoost",
                                              dates=feature_dates,
                                              sample_weight=weights, calibrate=True)
    fl = X.columns.tolist()
    print(f"  Model trained on {len(X)} matches, {len(fl)} features")

    # Dixon-Coles Poisson member + blend-weight tuning on the chronological tail.
    _POISSON = fit_dixon_coles(results)
    from sklearn.metrics import log_loss as _ll
    order = pd.Series(pd.to_datetime(feature_dates, errors="coerce")).sort_values().index
    split = max(1, int(len(order) * 0.8))
    val_idx = order[split:]
    p_xgb_val = model.predict_proba(X.iloc[val_idx])
    p_pois_val = np.array([_POISSON.outcome_probs(match_meta[i][0], match_meta[i][1], neutral=match_meta[i][2]) for i in val_idx])
    best_alpha, best_loss = 1.0, np.inf
    for a_ in np.linspace(0.0, 1.0, 21):
        bl = np.array([blend_probabilities(p_xgb_val[k], p_pois_val[k], a_) for k in range(len(val_idx))])
        try:
            lo = _ll(y[val_idx], bl, labels=[0, 1, 2])
        except Exception:
            continue
        if lo < best_loss:
            best_loss, best_alpha = lo, a_
    _ALPHA = float(best_alpha)
    print(f"  Tuned blend alpha (XGB weight) = {_ALPHA:.2f}")

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
    for teams in GROUP_2026_TEAMS.values():
        for t in teams:
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

        r = run_one_sim(model, fl, base_state, cf, base_groups, remaining_matches)
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
        pct = count / N_SIMS * 100
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