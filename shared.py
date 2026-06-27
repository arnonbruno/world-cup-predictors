"""Shared utilities for World Cup predictor scripts."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

NAME_ALIASES = {
    "West Germany": "Germany",
    "East Germany": "Germany",
    "German DR": "Germany",
    "Soviet Union": "Russia",
    "USSR": "Russia",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Czechoslovakia": "Czech Republic",
    "Dutch East Indies": "Indonesia",
    "Dutch Guyana": "Suriname",
    "Republic of Ireland": "Ireland",
    "Burma": "Myanmar",
    "United Arab Republic": "Egypt",
    "Vietnam Republic": "Vietnam",
    "South Vietnam": "Vietnam",
    "Zaire": "DR Congo",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Côte dIvoire": "Côte d'Ivoire",
    "Cote dIvoire": "Côte d'Ivoire",
    "South Korea": "Korea Republic",
    "Korea, South": "Korea Republic",
    "North Korea": "Korea DPR",
    "Korea, North": "Korea DPR",
    "Iran": "IR Iran",
    "United States": "USA",
    "United States of America": "USA",
    "UAE": "United Arab Emirates",
    "Cape Verde Islands": "Cape Verde",
    "Bosnia": "Bosnia and Herzegovina",
    "Korea South": "Korea Republic",
    "Korea North": "Korea DPR",
    "Curacao": "Curaçao",
}

COUNTRY_FEATURE_COLUMNS = [
    "gdp_per_capita",
    "population",
    "life_expectancy",
    "urbanization_pct",
    "health_spending_pct_gdp",
    "elo_rating",
    "fifa_rank",
    "football_power_index",
    "football_tradition",
]

# Betting-odds derived features merged onto matches that have bookmaker odds.
# Matches without odds get NaN for every column (XGBoost handles missing values
# natively, so no imputation is required). ``odds_overround`` is the bookmaker
# margin (sum of implied probabilities minus 1); a useful signal of how confident
# / liquid the market was on a given fixture.
ODDS_FEATURE_COLUMNS = [
    "implied_home_prob",
    "implied_draw_prob",
    "implied_away_prob",
    "odds_overround",
]

# Transfermarkt squad market-value features. Values span ~1M to ~1.8B EUR, so
# every level/diff is log-scaled (log1p) before it reaches the model. A team/year
# with no Transfermarkt coverage yields NaN for every column; XGBoost handles the
# missing split natively, so no imputation is performed for these columns.
SQUAD_VALUE_FEATURE_COLUMNS = [
    "squad_value",
    "opp_squad_value",
    "squad_value_diff",
    "squad_value_ratio",
]

# World Cup editions for which Transfermarkt squad values are available. Lookups
# use the most recent edition at or before the match year (a "year lag"), so a
# 2023 friendly is valued with the 2022 squad and a 2026 fixture with 2026 values.
SQUAD_VALUE_YEARS = (2014, 2018, 2022, 2026)

GROUP_2026_TEAMS = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Côte d'Ivoire", "Ecuador", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

WC_WINNERS = {
    1930: "Uruguay",
    1934: "Italy",
    1938: "Italy",
    1950: "Uruguay",
    1954: "Germany",
    1958: "Brazil",
    1962: "Brazil",
    1966: "England",
    1970: "Brazil",
    1974: "Germany",
    1978: "Argentina",
    1982: "Italy",
    1986: "Argentina",
    1990: "Germany",
    1994: "Brazil",
    1998: "France",
    2002: "Brazil",
    2006: "Italy",
    2010: "Spain",
    2014: "Germany",
    2018: "France",
    2022: "Argentina",
}

THIRD_SLOT_LABELS = {
    3: "M74",
    9: "M77",
    13: "M79",
    15: "M80",
    17: "M81",
    19: "M82",
    25: "M85",
    29: "M87",
}
THIRD_SLOT_ALLOWED_GROUPS = {
    3: set("ABCDF"),   # 1E vs third from A/B/C/D/F
    9: set("CDFGH"),   # 1I vs third from C/D/F/G/H
    13: set("CEFHI"),  # 1A vs third from C/E/F/H/I
    15: set("EHIJK"),  # 1L vs third from E/H/I/J/K
    17: set("BEFIJ"),  # 1D vs third from B/E/F/I/J
    19: set("AEHIJ"),  # 1G vs third from A/E/H/I/J
    25: set("EFGIJ"),  # 1B vs third from E/F/G/I/J
    29: set("DEIJL"),  # 1K vs third from D/E/I/J/L
}
THIRD_SLOTS = [3, 9, 13, 15, 17, 19, 25, 29]
INITIAL_ELO = 1500
K_FACTOR = 32
STAGE_TO_INT = {"group": 0, "round_of_16": 1, "quarterfinal": 2, "semifinal": 3, "final": 4}
# Blend weight used by ``blend_probabilities`` for knockout matches. In this
# codebase alpha is the XGBoost weight, so 0.50 means a conservative 50/50
# XGBoost / Dixon-Coles blend after group-stage probabilities are unchanged.
KNOCKOUT_ALPHA = 0.50
H2H_YEARS_LIMIT = 15
H2H_HALF_LIFE_YEARS = 10
COUNTRY_FEATURE_STALE_YEARS = 8

# Rolling window length for form / goals (kept identical across every script so the
# features the model is trained on match the features used at prediction time).
FORM_WINDOW = 20

# Knockout stage codes used by the 2026 prediction/backtest pipeline. Historical
# training only ever sees stages 0..4 (see ``STAGE_TO_INT``), so the 2026 pipeline
# must collapse its richer R32/R16/QF/SF/Final scheme onto the same 0..4 range to
# avoid asking the model to extrapolate to a ``stage`` value it never observed.
WC2026_STAGE_TO_TRAIN = {
    "group": 0,
    "round_of_32": 1,
    "round_of_16": 1,
    "quarterfinal": 2,
    "semifinal": 3,
    "third_place": 3,
    "final": 4,
}


def _third_place_assignment_for_combo(groups: Sequence[str]) -> Dict[str, int]:
    """Assign third-place groups to FIFA R32 slots for one qualifying combination."""
    remaining_groups = set(groups)
    assignment: Dict[str, int] = {}

    def search(open_slots: List[int], available: set) -> bool:
        if not open_slots:
            return not available

        slot = min(
            open_slots,
            key=lambda s: (
                len(THIRD_SLOT_ALLOWED_GROUPS[s] & available),
                THIRD_SLOTS.index(s),
            ),
        )
        candidates = sorted(THIRD_SLOT_ALLOWED_GROUPS[slot] & available)
        for group in candidates:
            assignment[group] = slot
            if search([s for s in open_slots if s != slot], available - {group}):
                return True
            assignment.pop(group, None)
        return False

    if not search(THIRD_SLOTS.copy(), remaining_groups):
        combo = ",".join(groups)
        raise ValueError(f"No FIFA third-place allocation found for groups: {combo}")
    return dict(assignment)


THIRD_PLACE_ALLOCATION_MATRIX = {
    combo: _third_place_assignment_for_combo(combo)
    for combo in combinations("ABCDEFGHIJKL", 8)
}


def harmonize_country(name: object) -> object:
    """Return the canonical project country/team name."""
    if pd.isna(name):
        return name
    text = str(name).strip()
    return NAME_ALIASES.get(text, text)


def harmonize_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(harmonize_country)
    return out


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def update_elo(elo_a: float, elo_b: float, score_a: int, score_b: int, neutral: bool = True):
    home_advantage = 0 if neutral else 50
    ea = expected_score(elo_a + home_advantage, elo_b)
    sa = 1 if score_a > score_b else (0.5 if score_a == score_b else 0)
    margin = abs(score_a - score_b)
    multiplier = np.log(max(margin, 1) + 1)
    return (
        elo_a + K_FACTOR * multiplier * (sa - ea),
        elo_b + K_FACTOR * multiplier * ((1 - sa) - (1 - ea)),
    )


def make_team_state() -> Dict[str, object]:
    """Create the mutable rolling state expected by match feature builders."""
    return {
        "elo": INITIAL_ELO,
        "form": [],
        "goals_for": [],
        "goals_against": [],
        "last_match": None,
        # New rolling history needed for momentum / opponent-weighted / fatigue features.
        "elo_history": [],          # Elo value AFTER each match (chronological)
        "opp_elo": [],              # opponent Elo BEFORE each match (chronological)
        "match_dates": [],          # date of each rolling match (chronological)
        "h2h": defaultdict(lambda: {
            "matches": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0,
            "entries": [],
        }),
        "wc_participations": 0,
        "wc_titles": 0,
        "wc_wins": 0,
        "wc_matches": 0,
    }


def _trim(seq: list, n: int = FORM_WINDOW) -> list:
    """Return the last ``n`` items of ``seq`` (helper to keep windows consistent)."""
    return seq[-n:]


def update_team_state(
    state,
    team: str,
    opponent: str,
    gf: int,
    ga: int,
    match_date,
    *,
    neutral: bool = True,
    is_world_cup: bool = False,
) -> None:
    """Apply one match result to ONE team's rolling state.

    Centralizing this here guarantees that every script (predict_2026, backtest,
    monte_carlo) updates Elo, form, goals, H2H and the new rolling histories in
    exactly the same way, which previously diverged (different form windows and a
    broken H2H key in the backtest). Call once per team, per match.
    """
    s, o = state[team], state[opponent]
    opp_elo_before = o["elo"]

    # Form / goals rolling windows.
    result = 1 if gf > ga else (0.5 if gf == ga else 0)
    s["form"] = _trim(s["form"] + [result])
    s["goals_for"] = _trim(s["goals_for"] + [gf])
    s["goals_against"] = _trim(s["goals_against"] + [ga])
    s["opp_elo"] = _trim(s["opp_elo"] + [opp_elo_before])
    s["match_dates"] = _trim(s["match_dates"] + [match_date])
    s["last_match"] = match_date

    # H2H (always keyed by the sorted pair so lookups are symmetric).
    key = tuple(sorted([team, opponent]))
    rec = s["h2h"][key]
    rec["matches"] += 1
    if gf > ga:
        rec["wins"] += 1
    elif gf == ga:
        rec["draws"] += 1
    else:
        rec["losses"] += 1
    rec["gf"] += gf
    rec["ga"] += ga
    rec.setdefault("entries", []).append({
        "date": match_date,
        "result": result,
        "gf": gf,
        "ga": ga,
    })

    if is_world_cup:
        s["wc_matches"] += 1
        if gf > ga:
            s["wc_wins"] += 1


def apply_match_to_state(
    state,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    match_date,
    *,
    neutral: bool = True,
    is_world_cup: bool = False,
) -> None:
    """Apply a full match (both sides + Elo) to the shared rolling state.

    Elo must be updated AFTER both teams' opponent-Elo snapshots are recorded, so
    that ``opp_elo`` reflects the pre-match strength of the opponent.
    """
    home = harmonize_country(home)
    away = harmonize_country(away)
    pre_home, pre_away = state[home]["elo"], state[away]["elo"]

    update_team_state(state, home, away, home_score, away_score, match_date,
                      neutral=neutral, is_world_cup=is_world_cup)
    update_team_state(state, away, home, away_score, home_score, match_date,
                      neutral=neutral, is_world_cup=is_world_cup)

    new_home, new_away = update_elo(pre_home, pre_away, home_score, away_score, neutral)
    state[home]["elo"] = new_home
    state[away]["elo"] = new_away
    state[home]["elo_history"] = _trim(state[home]["elo_history"] + [new_home])
    state[away]["elo_history"] = _trim(state[away]["elo_history"] + [new_away])


def finalize_world_cup_history(state, wc_year: int, participants: Iterable[str]) -> None:
    """Update state once a World Cup has completed."""
    teams = {harmonize_country(team) for team in participants}
    for team in teams:
        state[team]["wc_participations"] += 1

    winner = WC_WINNERS.get(int(wc_year))
    if winner in teams:
        state[winner]["wc_titles"] += 1


def _elo_momentum(state_entry, lookback: int) -> float:
    """Elo gained/lost over the last ``lookback`` matches (0 if not enough history)."""
    hist = state_entry.get("elo_history", [])
    if len(hist) < 2:
        return 0.0
    window = hist[-(lookback + 1):]
    return float(window[-1] - window[0])


def _opp_weighted_form(state_entry) -> float:
    """Recent win/draw points weighted by the strength of the opponent faced.

    A win against a 2000-Elo side counts far more than a win against a 1300-Elo
    side. Returns a strength-weighted average of recent results in [0, 1]-ish range.
    """
    form = state_entry.get("form", [])
    opp = state_entry.get("opp_elo", [])
    n = min(len(form), len(opp))
    if n == 0:
        return 0.5
    form, opp = form[-n:], opp[-n:]
    # Weight = opponent strength relative to 1500 baseline (clipped to stay positive).
    weights = [max(0.25, e / 1500.0) for e in opp]
    total_w = sum(weights)
    if total_w == 0:
        return float(np.mean(form))
    return float(sum(f * w for f, w in zip(form, weights)) / total_w)


def _matches_in_window(state_entry, match_date, days: int) -> int:
    """Number of matches played in the ``days`` leading up to ``match_date`` (fatigue)."""
    dates = state_entry.get("match_dates", [])
    if not dates or match_date is None:
        return 0
    count = 0
    for d in dates:
        if d is None:
            continue
        try:
            delta = (match_date - d).days
        except TypeError:
            continue
        if 0 <= delta <= days:
            count += 1
    return count


def _weighted_h2h_record(record: dict, match_date) -> dict:
    """Return recency-weighted H2H stats from one team's perspective.

    The historical aggregate is retained for backward compatibility, but when
    per-match entries are available we only use the last ``H2H_YEARS_LIMIT``
    years and exponentially decay them with a 10-year half-life.
    """
    entries = record.get("entries") or []
    if not entries:
        return dict(record)
    try:
        current_date = pd.to_datetime(match_date)
    except Exception:
        current_date = pd.Timestamp.max
    cutoff = current_date - pd.DateOffset(years=H2H_YEARS_LIMIT)
    weighted = {"matches": 0.0, "wins": 0.0, "draws": 0.0, "losses": 0.0, "gf": 0.0, "ga": 0.0}
    for item in entries:
        try:
            item_date = pd.to_datetime(item.get("date"))
        except Exception:
            item_date = pd.NaT
        if pd.isna(item_date) or item_date < cutoff or item_date >= current_date:
            continue
        age_years = max((current_date - item_date).days / 365.25, 0.0)
        weight = 0.5 ** (age_years / H2H_HALF_LIFE_YEARS)
        result = float(item.get("result", 0.5))
        weighted["matches"] += weight
        weighted["wins"] += weight if result == 1 else 0.0
        weighted["draws"] += weight if result == 0.5 else 0.0
        weighted["losses"] += weight if result == 0 else 0.0
        weighted["gf"] += weight * float(item.get("gf", 0.0))
        weighted["ga"] += weight * float(item.get("ga", 0.0))
    return weighted


def compute_match_features(team, opponent, state, country_features, stage_num, match_date, neutral=True, is_home=False, odds_row=None, squad_values=None):
    s, o = state[team], state[opponent]
    form = s["form"][-10:] if s["form"] else [0.5]
    opp_form = o["form"][-10:] if o["form"] else [0.5]
    gf5 = s["goals_for"][-5:] or [0]
    ga5 = s["goals_against"][-5:] or [0]
    gf10 = s["goals_for"][-10:] or [0]
    ga10 = s["goals_against"][-10:] or [0]
    gf3 = s["goals_for"][-3:] or [0]
    ga3 = s["goals_against"][-3:] or [0]
    rest = min((match_date - s["last_match"]).days if s["last_match"] else 30, 60)
    h2h_key = tuple(sorted([team, opponent]))
    h = s["h2h"].get(h2h_key, {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0})
    h = _weighted_h2h_record(h, match_date)
    hm = max(h["matches"], 1)
    cf, oc = country_features.get(team, {}), country_features.get(opponent, {})

    # ── Existing derived quantities ──
    form_win_rate = sum(1 for f in form if f == 1) / len(form)
    form_draw_rate = sum(1 for f in form if f == 0.5) / len(form)
    opp_form_draw_rate = sum(1 for f in opp_form if f == 0.5) / len(opp_form)
    elo_diff = s["elo"] - o["elo"]

    # ── NEW FEATURES ──
    # 1. Elo momentum: is the team trending up or down recently?
    elo_momentum_5 = _elo_momentum(s, 5)
    elo_momentum_10 = _elo_momentum(s, 10)
    opp_elo_momentum_5 = _elo_momentum(o, 5)
    elo_momentum_diff = elo_momentum_5 - opp_elo_momentum_5

    # 2. Attacking / defensive trend: recent 3 vs baseline 10.
    attack_trend = float(np.mean(gf3) - np.mean(gf10))
    defense_trend = float(np.mean(ga3) - np.mean(ga10))  # positive = conceding more lately

    # 3. Opponent-strength-weighted form (quality of recent results).
    weighted_form = _opp_weighted_form(s)
    opp_weighted_form = _opp_weighted_form(o)
    weighted_form_diff = weighted_form - opp_weighted_form

    # 4. Fatigue / congestion: matches in the last 30 and 90 days.
    fatigue_30 = _matches_in_window(s, match_date, 30)
    fatigue_90 = _matches_in_window(s, match_date, 90)
    opp_fatigue_30 = _matches_in_window(o, match_date, 30)
    fatigue_diff_30 = fatigue_30 - opp_fatigue_30

    # 5. Draw-propensity signals (the model's documented weakness is missing draws).
    #    Closely-matched, low-scoring, draw-prone sides are the classic draw setup.
    elo_parity = 1.0 / (1.0 + abs(elo_diff) / 100.0)  # ~1 when evenly matched, ->0 apart
    combined_draw_rate = (form_draw_rate + opp_form_draw_rate) / 2.0
    expected_total_goals = float(np.mean(gf5) + np.mean(ga5)
                                 + np.mean(o["goals_for"][-5:] or [0])
                                 + np.mean(o["goals_against"][-5:] or [0])) / 2.0
    low_scoring_indicator = 1.0 / (1.0 + expected_total_goals)

    # 6. Bookmaker implied probabilities (single best predictor in the football
    #    literature). NaN when the match has no odds; XGBoost handles missing.
    if odds_row is None:
        odds_row = {col: np.nan for col in ODDS_FEATURE_COLUMNS}

    # 7. Transfermarkt squad market values (log-scaled). Uses the most recent WC
    #    edition's values at or before the match year. Missing teams/years stay
    #    NaN so XGBoost can learn a dedicated "no valuation" split rather than
    #    treating an absent value as 0 EUR.
    home_val = squad_value_for_team(squad_values, team, match_date)["total_squad_value_eur"]
    away_val = squad_value_for_team(squad_values, opponent, match_date)["total_squad_value_eur"]
    squad_value = np.log1p(home_val) if np.isfinite(home_val) else np.nan
    opp_squad_value = np.log1p(away_val) if np.isfinite(away_val) else np.nan
    if np.isfinite(home_val) and np.isfinite(away_val):
        squad_value_diff = np.log1p(home_val) - np.log1p(away_val)
        # Ratio of raw values (home / away); +1 guards against a zero-value squad.
        squad_value_ratio = (home_val + 1.0) / (away_val + 1.0)
    else:
        squad_value_diff = np.nan
        squad_value_ratio = np.nan

    return {
        "elo": s["elo"], "elo_opponent": o["elo"],
        "elo_diff": elo_diff, "elo_sum": s["elo"] + o["elo"],
        "form_win_rate": form_win_rate,
        "form_draw_rate": form_draw_rate,
        "form_loss_rate": sum(1 for f in form if f == 0) / len(form),
        "avg_goals_scored_5": np.mean(gf5), "avg_goals_conceded_5": np.mean(ga5),
        "avg_goals_scored_10": np.mean(gf10), "avg_goals_conceded_10": np.mean(ga10),
        "rest_days": rest,
        "h2h_matches": h["matches"], "h2h_win_rate": h["wins"] / hm,
        "h2h_draw_rate": h["draws"] / hm, "h2h_avg_goals_for": h["gf"] / hm,
        "h2h_avg_goals_against": h["ga"] / hm,
        "wc_participations": s["wc_participations"], "wc_titles": s["wc_titles"],
        "wc_win_rate": s["wc_wins"] / max(s["wc_matches"], 1),
        "stage": stage_num, "neutral": int(bool(neutral)), "is_home": int(bool(is_home)),
        "gdp_per_capita": cf.get("gdp_per_capita", np.nan),
        "population": cf.get("population", np.nan),
        "life_expectancy": cf.get("life_expectancy", np.nan),
        "urbanization_pct": cf.get("urbanization_pct", np.nan),
        "health_spending_pct_gdp": cf.get("health_spending_pct_gdp", np.nan),
        "elo_pre_tournament": cf.get("elo_rating", s["elo"]),
        "fifa_rank": cf.get("fifa_rank", 100),
        "football_power_index": cf.get("football_power_index", 0),
        "football_tradition": cf.get("football_tradition", 0),
        "opp_elo_pre_tournament": oc.get("elo_rating", o["elo"]),
        "opp_football_power_index": oc.get("football_power_index", 0),
        "opp_football_tradition": oc.get("football_tradition", 0),
        "elo_diff_pre": cf.get("elo_rating", s["elo"]) - oc.get("elo_rating", o["elo"]),
        "power_diff": cf.get("football_power_index", 0) - oc.get("football_power_index", 0),
        "tradition_diff": cf.get("football_tradition", 0) - oc.get("football_tradition", 0),
        # ── NEW engineered features ──
        "elo_momentum_5": elo_momentum_5,
        "elo_momentum_10": elo_momentum_10,
        "elo_momentum_diff": elo_momentum_diff,
        "attack_trend": attack_trend,
        "defense_trend": defense_trend,
        "weighted_form": weighted_form,
        "weighted_form_diff": weighted_form_diff,
        "fatigue_30": fatigue_30,
        "fatigue_90": fatigue_90,
        "fatigue_diff_30": fatigue_diff_30,
        "elo_parity": elo_parity,
        "combined_draw_rate": combined_draw_rate,
        "expected_total_goals": expected_total_goals,
        "low_scoring_indicator": low_scoring_indicator,
        # ── Bookmaker implied probabilities (NaN when no odds available) ──
        "implied_home_prob": odds_row.get("implied_home_prob", np.nan),
        "implied_draw_prob": odds_row.get("implied_draw_prob", np.nan),
        "implied_away_prob": odds_row.get("implied_away_prob", np.nan),
        "odds_overround": odds_row.get("odds_overround", np.nan),
        # ── Transfermarkt squad market values (log-scaled; NaN when unavailable) ──
        "squad_value": squad_value,
        "opp_squad_value": opp_squad_value,
        "squad_value_diff": squad_value_diff,
        "squad_value_ratio": squad_value_ratio,
    }


def finalize_feature_frame(rows: Sequence[dict]) -> pd.DataFrame:
    """Turn feature dicts into a model-ready frame.

    Most columns are filled with 0 (the historical behaviour), but the
    bookmaker-odds and squad-value columns are *deliberately left as NaN* when
    missing so that XGBoost can learn dedicated "missing odds" / "no valuation"
    splits instead of treating an absent market as a 0% implied probability or an
    unvalued squad as 0 EUR. ``predict_proba`` at inference time must use the same
    convention (see ``prepare_prediction_frame``).
    """
    X = pd.DataFrame(rows)
    keep_nan = set(ODDS_FEATURE_COLUMNS) | set(SQUAD_VALUE_FEATURE_COLUMNS)
    non_nan = [c for c in X.columns if c not in keep_nan]
    if non_nan:
        X[non_nan] = X[non_nan].fillna(0)
    return X


def prepare_prediction_frame(feat: dict, feature_names: Sequence[str]) -> pd.DataFrame:
    """Build a single-row frame aligned to ``feature_names`` for prediction.

    Mirrors :func:`finalize_feature_frame`: odds and squad-value columns keep
    their NaN so the model sees the same missing-value encoding it was trained on.
    """
    X = pd.DataFrame([feat])
    for name in feature_names:
        if name not in X.columns:
            X[name] = np.nan
    X = X[list(feature_names)]
    keep_nan = set(ODDS_FEATURE_COLUMNS) | set(SQUAD_VALUE_FEATURE_COLUMNS)
    non_nan = [c for c in X.columns if c not in keep_nan]
    if non_nan:
        X[non_nan] = X[non_nan].fillna(0)
    return X


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _odds_key(date, home: str, away: str) -> Tuple[str, str, str]:
    """Build the merge key used to attach bookmaker odds to a match.

    Keyed by (ISO date string, canonical home, canonical away) so the lookup is
    robust to the various name spellings used in the odds source.
    """
    try:
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
    except Exception:
        date_str = str(date)
    return (date_str, harmonize_country(home), harmonize_country(away))


def load_betting_odds(
    path: Path = DATA_DIR / "betting_odds.csv",
) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """Load bookmaker implied probabilities keyed by (date, home, away).

    Returns a mapping from ``(date, canonical_home, canonical_away)`` to the
    ``ODDS_FEATURE_COLUMNS`` values. ``odds_overround`` is derived from the three
    implied probabilities (their sum minus 1). The orientation-swapped key is also
    stored (with home/away implied probs flipped) so a match recorded in the
    opposite orientation still finds its odds.

    Missing/unparseable files yield an empty mapping, so callers degrade
    gracefully to "no odds" (all-NaN) rather than crashing.
    """
    odds: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    if not Path(path).exists():
        return odds
    try:
        df = pd.read_csv(path)
    except Exception:
        return odds

    required = {"date", "home_team", "away_team",
                "implied_home_prob", "implied_draw_prob", "implied_away_prob"}
    if not required.issubset(df.columns):
        return odds

    for _, row in df.iterrows():
        try:
            ph = float(row["implied_home_prob"])
            pd_ = float(row["implied_draw_prob"])
            pa = float(row["implied_away_prob"])
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite([ph, pd_, pa])):
            continue
        overround = ph + pd_ + pa - 1.0
        forward = {
            "implied_home_prob": ph,
            "implied_draw_prob": pd_,
            "implied_away_prob": pa,
            "odds_overround": overround,
        }
        reverse = {
            "implied_home_prob": pa,
            "implied_draw_prob": pd_,
            "implied_away_prob": ph,
            "odds_overround": overround,
        }
        key = _odds_key(row["date"], row["home_team"], row["away_team"])
        # Forward orientation takes precedence; only add the swapped orientation
        # if that exact (date, home, away) pair is not already populated.
        odds.setdefault(key, forward)
        swapped = (key[0], key[2], key[1])
        odds.setdefault(swapped, reverse)
    return odds


def odds_features_for_match(
    odds: Optional[Dict[Tuple[str, str, str], Dict[str, float]]],
    date,
    home: str,
    away: str,
) -> Dict[str, float]:
    """Return the odds feature dict for a match, or all-NaN when unavailable."""
    nan_row = {col: np.nan for col in ODDS_FEATURE_COLUMNS}
    if not odds:
        return nan_row
    return dict(odds.get(_odds_key(date, home, away), nan_row))


def load_squad_values(
    path: Path = DATA_DIR / "squad_values.csv",
) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Load Transfermarkt squad values keyed by (canonical_team, year).

    The CSV uses the same raw team spellings as ``results.csv`` (e.g. "Iran",
    "South Korea", "Ivory Coast"), so each team is harmonized through
    :data:`NAME_ALIASES` on load to match the canonical names used in the rolling
    state dict ("IR Iran", "Korea Republic", "Cote d'Ivoire" -> "Côte d'Ivoire").

    Returns a mapping ``(team, year) -> {total_squad_value_eur,
    avg_player_value_eur}``. Missing/unparseable files yield an empty mapping so
    callers degrade gracefully to all-NaN squad-value features.
    """
    values: Dict[Tuple[str, int], Dict[str, float]] = {}
    if not Path(path).exists():
        return values
    try:
        df = pd.read_csv(path)
    except Exception:
        return values

    required = {"team", "year", "total_squad_value_eur", "avg_player_value_eur"}
    if not required.issubset(df.columns):
        return values

    for _, row in df.iterrows():
        try:
            year = int(row["year"])
        except (TypeError, ValueError):
            continue
        team = harmonize_country(row["team"])
        if pd.isna(team):
            continue

        def _num(col: str) -> float:
            try:
                v = float(row[col])
            except (TypeError, ValueError):
                return np.nan
            return v if np.isfinite(v) else np.nan

        values[(str(team), year)] = {
            "total_squad_value_eur": _num("total_squad_value_eur"),
            "avg_player_value_eur": _num("avg_player_value_eur"),
        }
    return values


def _squad_value_year(match_year: int) -> Optional[int]:
    """Most recent World Cup edition with squad values at or before ``match_year``.

    Implements the documented year lag: matches in 2023 look up 2022 squads, 2026
    fixtures look up 2026 values, etc. Returns ``None`` when the match predates the
    earliest available edition.
    """
    usable = [y for y in SQUAD_VALUE_YEARS if y <= match_year]
    return max(usable) if usable else None


def squad_value_for_team(
    squad_values: Optional[Dict[Tuple[str, int], Dict[str, float]]],
    team: str,
    match_date,
) -> Dict[str, float]:
    """Return the squad-value entry for ``team`` at the lagged WC year, or NaNs.

    ``team`` must already be the canonical (harmonized) name. ``match_date`` may be
    anything ``pd.to_datetime`` understands; its calendar year drives the year-lag
    lookup via :func:`_squad_value_year`.
    """
    nan_entry = {"total_squad_value_eur": np.nan, "avg_player_value_eur": np.nan}
    if not squad_values:
        return dict(nan_entry)
    try:
        match_year = int(pd.to_datetime(match_date).year)
    except Exception:
        return dict(nan_entry)
    lookup_year = _squad_value_year(match_year)
    if lookup_year is None:
        return dict(nan_entry)
    return dict(squad_values.get((harmonize_country(team), lookup_year), nan_entry))


def load_country_feature_history(
    path: Path = DATA_DIR / "world_cup_predictors_dataset.csv",
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Load country features keyed by canonical team and data year.

    The source column is named ``wc_year`` because the original dataset was
    World-Cup-edition based, but callers should treat it as a feature vintage.
    If newer non-WC rows are added later (for example 2023/2024 country data),
    they will be eligible for 2026 lookups automatically.
    """
    df = pd.read_csv(path)
    history: Dict[str, Dict[int, Dict[str, float]]] = {}
    for _, row in df.iterrows():
        team = harmonize_country(row["country"])
        year = int(row["wc_year"])
        entry = {
            col: row.get(col, np.nan) for col in COUNTRY_FEATURE_COLUMNS
        }
        entry["feature_year"] = year
        history.setdefault(team, {})[year] = entry
    return history


def country_features_for_year(
    history: Dict[str, Dict[int, Dict[str, float]]], year: int
) -> Dict[str, Dict[str, float]]:
    """Return latest known country features at or before ``year`` for each team.

    Feature vintages older than ``COUNTRY_FEATURE_STALE_YEARS`` are retained but
    explicitly flagged via ``country_features_stale`` and ``feature_age_years``.
    That keeps predictions possible while making cases such as 2026 Norway using
    1998 country data visible to reports and CLIs.
    """
    features: Dict[str, Dict[str, float]] = {}
    for team, by_year in history.items():
        usable = [wc_year for wc_year in by_year if wc_year <= year]
        if not usable:
            continue
        feature_year = max(usable)
        entry = dict(by_year[feature_year])
        age = int(year) - int(feature_year)
        entry["feature_year"] = feature_year
        entry["feature_age_years"] = age
        entry["country_features_stale"] = int(age > COUNTRY_FEATURE_STALE_YEARS)
        features[team] = entry
    return features


def country_feature_staleness_warnings(
    features: Dict[str, Dict[str, float]],
    teams: Iterable[str],
    target_year: int,
) -> List[str]:
    """Human-readable warnings for stale country feature vintages."""
    warnings: List[str] = []
    for team in sorted({harmonize_country(t) for t in teams}):
        entry = features.get(team)
        if not entry:
            warnings.append(f"{team}: no country feature row available for {target_year}.")
            continue
        if int(entry.get("country_features_stale", 0)):
            fy = entry.get("feature_year", "unknown")
            age = entry.get("feature_age_years", "unknown")
            warnings.append(f"{team}: country features are from {fy} ({age} years old for {target_year}).")
    return warnings


def wc_calibration_buckets(
    path: Path = ROOT / "backtest_walkforward_results.csv",
    bucket_size: float = 0.10,
) -> List[dict]:
    """Compute World-Cup-only top-prediction calibration buckets.

    Returns one dict per confidence bucket with ``n``, average confidence,
    empirical accuracy and smoothed accuracy. The smoothed value is conservative:
    sparse buckets are pulled toward their own average confidence instead of
    overreacting to one or two historical examples.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    required = {"tournament", "confidence", "correct"}
    if not required.issubset(df.columns):
        return []
    wc = df[df["tournament"].astype(str).eq("FIFA World Cup")].copy()
    wc["confidence"] = pd.to_numeric(wc["confidence"], errors="coerce")
    wc = wc.dropna(subset=["confidence"])
    if wc.empty:
        return []
    buckets: List[dict] = []
    edges = np.arange(0.0, 1.0 + bucket_size, bucket_size)
    prior_strength = 12.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            part = wc[(wc["confidence"] >= lo) & (wc["confidence"] <= hi)]
        else:
            part = wc[(wc["confidence"] >= lo) & (wc["confidence"] < hi)]
        if part.empty:
            continue
        correct = part["correct"].map(parse_bool).astype(float)
        avg_conf = float(part["confidence"].mean())
        accuracy = float(correct.mean())
        smoothed = float((correct.sum() + prior_strength * avg_conf) / (len(part) + prior_strength))
        buckets.append({
            "lo": float(lo),
            "hi": float(hi),
            "n": int(len(part)),
            "avg_confidence": avg_conf,
            "accuracy": accuracy,
            "smoothed_accuracy": smoothed,
        })
    return buckets


def apply_wc_knockout_calibration(
    p_home: float,
    p_away: float,
    buckets: Sequence[dict],
    *,
    min_bucket_n: int = 5,
) -> Tuple[float, float, str]:
    """Calibrate a two-way knockout favorite probability using WC buckets."""
    total = float(p_home) + float(p_away)
    if total <= 0 or not np.isfinite(total):
        return 0.5, 0.5, "WC calibration skipped: invalid knockout probabilities."
    p_home = float(p_home) / total
    p_away = float(p_away) / total
    home_favored = p_home >= p_away
    favorite_prob = p_home if home_favored else p_away
    bucket = None
    for candidate in buckets or []:
        lo, hi = float(candidate.get("lo", 0.0)), float(candidate.get("hi", 1.0))
        if lo <= favorite_prob < hi or (hi >= 1.0 and favorite_prob <= hi):
            bucket = candidate
            break
    if not bucket or int(bucket.get("n", 0)) < min_bucket_n:
        return p_home, p_away, "WC calibration skipped: no sufficiently populated bucket."
    calibrated_favorite = float(bucket.get("smoothed_accuracy", favorite_prob))
    calibrated_favorite = float(np.clip(calibrated_favorite, 0.50, 0.98))
    calibrated_underdog = 1.0 - calibrated_favorite
    note = (
        f"WC bucket {bucket['lo']:.0%}-{bucket['hi']:.0%}: n={bucket['n']}, "
        f"avg_conf={bucket['avg_confidence']:.1%}, acc={bucket['accuracy']:.1%}, "
        f"smoothed={calibrated_favorite:.1%}"
    )
    if home_favored:
        return calibrated_favorite, calibrated_underdog, note
    return calibrated_underdog, calibrated_favorite, note


def stage_counts_for_year(year: int, total_matches: int) -> Tuple[int, int, int, int]:
    """Return counts for group, R16, QF, SF before final/tail handling."""
    if year == 1930:
        return (15, 0, 0, 2)
    if year == 1934:
        return (0, 9, 4, 2)
    if year == 1938:
        return (0, 10, 4, 2)
    if year == 1950:
        return (16, 0, 0, 0)
    if year == 1954:
        return (18, 0, 4, 2)
    if year == 1958:
        return (27, 0, 4, 2)
    if year in {1962, 1966, 1970}:
        return (24, 0, 4, 2)
    if year in {1974, 1978}:
        return (24, 12, 0, 0)
    if year == 1982:
        return (36, 12, 0, 2)
    if year in {1986, 1990, 1994}:
        return (36, 8, 4, 2)
    if total_matches >= 64:
        return (48, 8, 4, 2)
    if total_matches >= 52:
        return (36, 8, 4, 2)
    if total_matches >= 32:
        return (24, 0, 4, 2)
    return (max(total_matches - 3, 0), 0, 0, 2)


def select_final_match_index(wc_matches: pd.DataFrame, year: int) -> int:
    """Select the likely final row within a canonicalized World Cup match frame."""
    winner = harmonize_country(WC_WINNERS.get(int(year), ""))
    df = wc_matches.sort_values(["date", "home_team", "away_team"]).reset_index()
    if df.empty:
        return -1
    latest_date = df["date"].max()
    same_day = df[df["date"] == latest_date]
    involving_winner = same_day[(same_day["home_team"] == winner) | (same_day["away_team"] == winner)]
    if not involving_winner.empty:
        return int(involving_winner.index[0])
    if int(year) == 1950:
        end_slice = df.tail(min(6, len(df)))
        candidates = end_slice[(end_slice["home_team"] == winner) | (end_slice["away_team"] == winner)]
        if not candidates.empty:
            return int(candidates.index[-1])
    return int(df.index[-1])


def assign_wc_stage_map(wc_matches: pd.DataFrame, year: int) -> Dict[int, int]:
    """Assign stage code per original dataframe index using historical WC schedule shapes."""
    if wc_matches.empty:
        return {}
    df = wc_matches.sort_values(["date", "home_team", "away_team"]).reset_index()
    total = len(df)
    stage_labels = ["group"] * total
    group_count, r16_count, qf_count, sf_count = stage_counts_for_year(year, total)
    idx = 0
    for label, count in (
        ("group", group_count),
        ("round_of_16", r16_count),
        ("quarterfinal", qf_count),
        ("semifinal", sf_count),
    ):
        for _ in range(count):
            if idx < total:
                stage_labels[idx] = label
                idx += 1
    while idx < total:
        stage_labels[idx] = "semifinal"
        idx += 1
    final_idx = select_final_match_index(wc_matches, year)
    if 0 <= final_idx < total:
        stage_labels[final_idx] = "final"
    return {int(df.iloc[i]["index"]): STAGE_TO_INT[stage_labels[i]] for i in range(total)}


def infer_world_cup_stage_map(results_df: pd.DataFrame) -> Dict[int, int]:
    """Infer historical World Cup stage labels for rows in a results dataframe."""
    if results_df.empty or "tournament" not in results_df.columns or "date" not in results_df.columns:
        return {}
    df = harmonize_columns(results_df.copy(), ["home_team", "away_team"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    wc = df[df["tournament"].astype(str).eq("FIFA World Cup")].copy()
    if wc.empty:
        return {}
    stage_by_index: Dict[int, int] = {}
    for year, part in wc.groupby(wc["date"].dt.year):
        if pd.isna(year):
            continue
        stage_by_index.update(assign_wc_stage_map(part, int(year)))
    return stage_by_index


# Half-life (in days) for recency weighting of training rows. A 4-year half-life
# means a friendly from 4 years ago counts half as much as a match today, so the
# model leans on recent squad/era information without discarding deep history.
TIME_DECAY_HALF_LIFE_DAYS = 365 * 4

# Multiplier applied to draw rows (label == 1) so the softmax stops collapsing
# draws into home wins (the model's documented weakness). Combined with isotonic
# calibration, this targets log-loss/Brier on the under-predicted draw class.
DRAW_CLASS_WEIGHT = 1.6


def time_decay_weights(dates: Sequence, half_life_days: int = TIME_DECAY_HALF_LIFE_DAYS) -> np.ndarray:
    """Recency weights ``0.5 ** (age_days / half_life_days)`` for each row."""
    d = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    max_date = d.max()
    age_days = (max_date - d).dt.days.fillna(0).clip(lower=0)
    return np.asarray(0.5 ** (age_days / float(half_life_days)), dtype=float)


def sample_weights(
    y: np.ndarray,
    dates: Sequence | None = None,
    *,
    time_decay: bool = True,
    draw_weight: float = DRAW_CLASS_WEIGHT,
) -> np.ndarray:
    """Combine recency decay and draw up-weighting into one sample-weight vector."""
    y = np.asarray(y)
    w = np.ones(len(y), dtype=float)
    if time_decay and dates is not None and len(dates) == len(y):
        w = w * time_decay_weights(dates)
    if draw_weight and draw_weight != 1.0:
        w = w * np.where(y == 1, draw_weight, 1.0)
    return w


class IsotonicProbabilityCalibrator:
    """Per-class isotonic recalibration of a fitted multiclass classifier.

    ``CalibratedClassifierCV(method="isotonic", cv="prefit")`` is the canonical
    tool, but it refits on the whole validation set and does not always preserve
    the exact 3-class ordering the rest of the pipeline assumes. This thin wrapper
    fits one ``IsotonicRegression`` per class on the chronological holdout, then
    renormalizes, which keeps the [home, draw, away] column order intact and lets
    callers keep using ``predict_proba``/``predict`` unchanged.
    """

    def __init__(self, base_model, classes):
        self.base_model = base_model
        self.classes_ = np.asarray(classes)
        self._iso = {}

    def fit(self, X_val, y_val, sample_weight=None):
        from sklearn.isotonic import IsotonicRegression

        probs = self.base_model.predict_proba(X_val)
        y_val = np.asarray(y_val)
        for j, cls in enumerate(self.classes_):
            target = (y_val == cls).astype(float)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            try:
                iso.fit(probs[:, j], target, sample_weight=sample_weight)
                self._iso[j] = iso
            except Exception:
                self._iso[j] = None
        return self

    def predict_proba(self, X):
        probs = np.asarray(self.base_model.predict_proba(X), dtype=float)
        out = np.empty_like(probs)
        for j in range(probs.shape[1]):
            iso = self._iso.get(j)
            out[:, j] = iso.predict(probs[:, j]) if iso is not None else probs[:, j]
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def fit_xgb_with_validation(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    label: str = "model",
    dates: Sequence | None = None,
    sample_weight: Sequence | None = None,
    calibrate: bool = False,
):
    """Fit an XGBoost classifier and report a chronological holdout when dates are available.

    When ``sample_weight`` is provided it is applied to the training split (and
    the early-stopping eval set), so time-decay and draw up-weighting flow into
    the fit. When ``calibrate=True`` the fitted model is wrapped in an isotonic
    calibrator fitted on the chronological holdout; the returned object still
    exposes ``predict``/``predict_proba`` and the original estimator stays
    available as ``.base_model``.
    """
    sw = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
    if dates is not None and len(dates) == len(X):
        order = pd.Series(pd.to_datetime(dates, errors="coerce")).sort_values().index
        split = max(1, int(len(order) * 0.8))
        if split >= len(order):
            split = len(order) - 1
        train_idx, val_idx = order[:split], order[split:]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        sw_train = sw[train_idx] if sw is not None else None
        sw_val = sw[val_idx] if sw is not None else None
        split_label = "chronological holdout"
    else:
        stratify = y if min(np.bincount(y.astype(int))) >= 2 else None
        if sw is not None:
            idx = np.arange(len(X))
            tr, va = train_test_split(idx, test_size=0.2, random_state=42, stratify=stratify)
            X_train, X_val = X.iloc[tr], X.iloc[va]
            y_train, y_val = y[tr], y[va]
            sw_train, sw_val = sw[tr], sw[va]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
            sw_train = sw_val = None
        split_label = "random holdout"

    fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
    if sw_train is not None:
        fit_kwargs["sample_weight"] = sw_train
    try:
        model.fit(X_train, y_train, early_stopping_rounds=30, **fit_kwargs)
    except TypeError:
        try:
            model.set_params(early_stopping_rounds=30)
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            if sw_train is not None:
                model.fit(X_train, y_train, sample_weight=sw_train)
            else:
                model.fit(X_train, y_train)

    estimator = model
    if calibrate:
        classes = getattr(model, "classes_", np.array(sorted(np.unique(y).astype(int).tolist())))
        try:
            estimator = IsotonicProbabilityCalibrator(model, classes).fit(X_val, y_val, sample_weight=sw_val)
        except Exception as exc:  # pragma: no cover - calibration is best-effort
            print(f"  {label} calibration failed ({exc}); using uncalibrated model")
            estimator = model

    val_probs = estimator.predict_proba(X_val)
    val_pred = np.argmax(val_probs, axis=1)
    labels = sorted(np.unique(y).astype(int).tolist())
    acc = accuracy_score(y_val, val_pred)
    loss = log_loss(y_val, val_probs, labels=labels)
    cal_tag = " (isotonic-calibrated)" if calibrate else ""
    print(f"  {label} {split_label}{cal_tag} accuracy={acc:.3f}, log-loss={loss:.3f} ({len(X_val)} matches)")
    return estimator, {"accuracy": float(acc), "log_loss": float(loss), "n_val": int(len(X_val)), "split": split_label}


class DixonColesModel:
    """Dixon-Coles bivariate-Poisson goal model.

    Estimates per-team attack/defense strengths plus a global home advantage and
    a low-score dependence parameter ``rho``. W/D/L probabilities come from a
    truncated scoreline grid, which naturally produces realistic *draw* rates
    (the XGBoost model's documented weakness). Strengths are fit by weighted
    Poisson MLE; recent matches are up-weighted via the shared time-decay.
    """

    def __init__(self, attack, defense, home_adv, rho, base, teams):
        self.attack = attack
        self.defense = defense
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self.base = float(base)
        self.teams = set(teams)

    def _lambdas(self, home: str, away: str, neutral: bool = True):
        home = harmonize_country(home)
        away = harmonize_country(away)
        ah = self.attack.get(home, 0.0)
        dh = self.defense.get(home, 0.0)
        aa = self.attack.get(away, 0.0)
        da = self.defense.get(away, 0.0)
        adv = 0.0 if neutral else self.home_adv
        lam_home = np.exp(self.base + adv + ah - da)
        lam_away = np.exp(self.base + aa - dh)
        # Guard against pathological strengths producing huge expected goals.
        return float(np.clip(lam_home, 1e-3, 8.0)), float(np.clip(lam_away, 1e-3, 8.0))

    @staticmethod
    def _tau(i, j, lam, mu, rho):
        if i == 0 and j == 0:
            return 1.0 - lam * mu * rho
        if i == 0 and j == 1:
            return 1.0 + lam * rho
        if i == 1 and j == 0:
            return 1.0 + mu * rho
        if i == 1 and j == 1:
            return 1.0 - rho
        return 1.0

    def outcome_probs(self, home: str, away: str, neutral: bool = True, max_goals: int = 10) -> np.ndarray:
        """Return [P(home win), P(draw), P(away win)] from the scoreline grid."""
        lam, mu = self._lambdas(home, away, neutral)
        i = np.arange(0, max_goals + 1)
        # Poisson PMFs.
        from math import lgamma

        log_fact = np.array([lgamma(k + 1) for k in i])
        ph = np.exp(i * np.log(lam) - lam - log_fact)
        pa = np.exp(i * np.log(mu) - mu - log_fact)
        grid = np.outer(ph, pa)
        # Dixon-Coles low-score correction on the four 0/1 cells.
        for a in (0, 1):
            for b in (0, 1):
                grid[a, b] *= self._tau(a, b, lam, mu, self.rho)
        grid = np.clip(grid, 0.0, None)
        total = grid.sum()
        if total <= 0:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        grid /= total
        p_home = np.tril(grid, -1).sum()   # home goals > away goals
        p_away = np.triu(grid, 1).sum()    # away goals > home goals
        p_draw = np.trace(grid)
        return np.array([p_home, p_draw, p_away])


def fit_dixon_coles(
    results_df: pd.DataFrame,
    *,
    max_year: int | None = None,
    half_life_days: int = TIME_DECAY_HALF_LIFE_DAYS,
    min_matches: int = 8,
    exclude_2026_wc: bool = True,
) -> "DixonColesModel":
    """Fit a Dixon-Coles model by weighted Poisson MLE on historical results.

    Falls back to a closed-form attack/defense estimate (no ``rho``) when SciPy
    is unavailable, so the ensemble still works without the optional dependency.
    """
    try:
        from scipy.optimize import minimize
        _have_scipy = True
    except Exception:
        minimize = None
        _have_scipy = False

    df = harmonize_columns(results_df.copy(), ["home_team", "away_team"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if exclude_2026_wc:
        df = df[~((df["date"].dt.year == 2026) & (df["tournament"] == "FIFA World Cup"))]
    df = df.dropna(subset=["home_score", "away_score", "date"])
    if max_year is not None:
        df = df[df["date"].dt.year <= max_year]
    # Keep recent history meaningful; ~last 12 years covers several WC cycles
    # while keeping the parameter count (2 * n_teams + 3) tractable for L-BFGS-B.
    cutoff = df["date"].max() - pd.Timedelta(days=365 * 12)
    df = df[df["date"] >= cutoff].copy()

    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    teams = sorted(counts[counts >= min_matches].index.tolist())
    df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)].reset_index(drop=True)
    if df.empty or len(teams) < 2:
        return DixonColesModel({}, {}, 0.0, 0.0, 0.0, teams)

    idx = {t: k for k, t in enumerate(teams)}
    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    hg = df["home_score"].to_numpy(dtype=float)
    ag = df["away_score"].to_numpy(dtype=float)
    neutral = df.get("neutral")
    is_neutral = (neutral.map(parse_bool).to_numpy() if neutral is not None
                  else np.ones(len(df), dtype=bool))
    w = time_decay_weights(df["date"], half_life_days)

    n = len(teams)

    if not _have_scipy:
        # Closed-form weighted estimate: attack ~ log(weighted GF / mean), and
        # defense ~ -log(weighted GA / mean). No low-score (rho) correction.
        gf_sum = np.zeros(n); ga_sum = np.zeros(n); wt = np.zeros(n)
        for k in range(len(df)):
            gf_sum[hi[k]] += w[k] * hg[k]; ga_sum[hi[k]] += w[k] * ag[k]; wt[hi[k]] += w[k]
            gf_sum[ai[k]] += w[k] * ag[k]; ga_sum[ai[k]] += w[k] * hg[k]; wt[ai[k]] += w[k]
        wt = np.clip(wt, 1e-9, None)
        gf_rate = gf_sum / wt
        ga_rate = ga_sum / wt
        base = float(np.log(np.clip(np.average(gf_rate, weights=wt), 1e-3, None)))
        attack = np.log(np.clip(gf_rate, 1e-3, None)) - base
        attack = attack - attack.mean()
        defense = -(np.log(np.clip(ga_rate, 1e-3, None)) - base)
        defense = defense - defense.mean()
        home_adv = float(np.log(np.clip(hg.mean(), 1e-3, None) / np.clip(ag.mean(), 1e-3, None)))
        return DixonColesModel(
            attack={t: float(attack[idx[t]]) for t in teams},
            defense={t: float(defense[idx[t]]) for t in teams},
            home_adv=max(0.0, home_adv), rho=0.0, base=base, teams=teams,
        )
    # Params: attack[n], defense[n], home_adv, rho, base. Identifiability: mean
    # attack fixed to 0 via a soft penalty.
    def unpack(p):
        return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2]

    def neg_log_like(p):
        attack, defense, home_adv, rho, base = unpack(p)
        adv = np.where(is_neutral, 0.0, home_adv)
        log_lh = base + adv + attack[hi] - defense[ai]
        log_la = base + attack[ai] - defense[hi]
        lam = np.exp(np.clip(log_lh, -4, 3))
        mu = np.exp(np.clip(log_la, -4, 3))
        ll = hg * np.log(lam) - lam + ag * np.log(mu) - mu
        # Dixon-Coles low-score correction (vectorized over the 0/1 cells).
        tau = np.ones(len(df))
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.clip(tau, 1e-6, None)
        ll = ll + np.log(tau)
        penalty = 1e3 * (attack.mean() ** 2)  # anchor mean attack at 0
        return -np.sum(w * ll) + penalty

    p0 = np.zeros(2 * n + 3)
    p0[2 * n] = 0.25   # home advantage
    p0[2 * n + 1] = -0.05  # rho
    p0[2 * n + 2] = 0.0    # base
    bounds = [(-3, 3)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2), (-2.0, 2.0)]
    try:
        res = minimize(neg_log_like, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200})
        p = res.x
    except Exception:
        p = p0
    attack, defense, home_adv, rho, base = unpack(p)
    return DixonColesModel(
        attack={t: float(attack[idx[t]]) for t in teams},
        defense={t: float(defense[idx[t]]) for t in teams},
        home_adv=float(home_adv), rho=float(rho), base=float(base), teams=teams,
    )


def blend_probabilities(p_xgb: np.ndarray, p_poisson: np.ndarray, alpha: float) -> np.ndarray:
    """Convex blend ``alpha * xgb + (1 - alpha) * poisson``, renormalized."""
    p = alpha * np.asarray(p_xgb, dtype=float) + (1.0 - alpha) * np.asarray(p_poisson, dtype=float)
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def empty_group_table() -> Dict[str, Dict[str, int]]:
    return {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "played": 0}


def group_for_match(home: str, away: str) -> Optional[str]:
    home = harmonize_country(home)
    away = harmonize_country(away)
    for group, teams in GROUP_2026_TEAMS.items():
        if home in teams and away in teams:
            return group
    return None


def apply_group_result(groups, group: str, home: str, away: str, home_score: int, away_score: int) -> None:
    home = harmonize_country(home)
    away = harmonize_country(away)
    for team, gf, ga in [(home, home_score, away_score), (away, away_score, home_score)]:
        groups[group][team]["played"] += 1
        groups[group][team]["gf"] += gf
        groups[group][team]["ga"] += ga
        groups[group][team]["gd"] += gf - ga
    if home_score > away_score:
        groups[group][home]["pts"] += 3
    elif home_score < away_score:
        groups[group][away]["pts"] += 3
    else:
        groups[group][home]["pts"] += 1
        groups[group][away]["pts"] += 1


def build_2026_group_state(results: pd.DataFrame):
    """Build 2026 tables from completed CSV rows and list only unplayed fixtures."""
    groups = {
        group: {team: empty_group_table() for team in teams}
        for group, teams in GROUP_2026_TEAMS.items()
    }
    wc26 = results.copy()
    wc26["date"] = pd.to_datetime(wc26["date"])
    wc26 = wc26[(wc26["tournament"] == "FIFA World Cup") & (wc26["date"].dt.year == 2026)].copy()
    wc26 = harmonize_columns(wc26, ["home_team", "away_team"])
    remaining = []
    for _, row in wc26.sort_values("date").iterrows():
        group = group_for_match(row["home_team"], row["away_team"])
        if group is None:
            continue
        if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
            apply_group_result(groups, group, row["home_team"], row["away_team"], int(row["home_score"]), int(row["away_score"]))
        else:
            remaining.append((row["date"], row["home_team"], row["away_team"], group))
    return groups, remaining


def sorted_group_standings(group_data: Dict[str, Dict[str, int]]) -> List[Tuple[str, int, int, int]]:
    return sorted(
        [(team, data["pts"], data["gd"], data["gf"]) for team, data in group_data.items()],
        key=lambda item: (-item[1], -item[2], -item[3], item[0]),
    )


def rank_third_place_teams(standings_by_group: Dict[str, List[Tuple[str, int, int, int]]]):
    thirds = [
        (group, table[2][0], table[2][1], table[2][2], table[2][3])
        for group, table in standings_by_group.items()
    ]
    return sorted(thirds, key=lambda item: (-item[2], -item[3], -item[4], item[0]))


def build_round_of_32(gw: Dict[str, str], gr: Dict[str, str], best_thirds: Sequence[Tuple[str, str, int, int, int]]):
    """Build the 2026 R32, replacing all third-place placeholders with actual teams."""
    r32_base = [
        gr["A"], gr["B"],
        gw["E"], None,
        gw["F"], gr["C"],
        gw["C"], gr["F"],
        gw["I"], None,
        gr["E"], gr["I"],
        gw["A"], None,
        gw["L"], None,
        gw["D"], None,
        gw["G"], None,
        gr["K"], gr["L"],
        gw["H"], gr["J"],
        gw["B"], None,
        gw["J"], gr["H"],
        gw["K"], None,
        gr["D"], gr["G"],
    ]

    third_by_group = {group: team for group, team, *_ in best_thirds}
    combo = tuple(sorted(third_by_group))
    try:
        slot_by_group = THIRD_PLACE_ALLOCATION_MATRIX[combo]
    except KeyError as exc:
        raise ValueError(f"Unsupported third-place group combination: {','.join(combo)}") from exc

    for group, slot in slot_by_group.items():
        r32_base[slot] = third_by_group[group]

    missing = [slot for slot in THIRD_SLOTS if r32_base[slot] is None]
    if missing:
        raise ValueError(f"Could not assign third-place teams to R32 slots: {missing}")

    labels = [f"Match {match_num}" for match_num in range(73, 89)]
    return [
        (labels[i // 2], r32_base[i], r32_base[i + 1])
        for i in range(0, len(r32_base), 2)
    ]
