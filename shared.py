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
        "h2h": defaultdict(lambda: {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0}),
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


def compute_match_features(team, opponent, state, country_features, stage_num, match_date, neutral=True, is_home=False):
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
    }


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def load_country_feature_history(
    path: Path = DATA_DIR / "world_cup_predictors_dataset.csv",
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Load country features keyed by canonical team and World Cup year."""
    df = pd.read_csv(path)
    history: Dict[str, Dict[int, Dict[str, float]]] = {}
    for _, row in df.iterrows():
        team = harmonize_country(row["country"])
        year = int(row["wc_year"])
        history.setdefault(team, {})[year] = {
            col: row.get(col, np.nan) for col in COUNTRY_FEATURE_COLUMNS
        }
    return history


def country_features_for_year(
    history: Dict[str, Dict[int, Dict[str, float]]], year: int
) -> Dict[str, Dict[str, float]]:
    """Return latest known country features at or before ``year`` for each team."""
    features: Dict[str, Dict[str, float]] = {}
    for team, by_year in history.items():
        usable = [wc_year for wc_year in by_year if wc_year <= year]
        if not usable:
            continue
        features[team] = by_year[max(usable)]
    return features


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


def fit_xgb_with_validation(model, X: pd.DataFrame, y: np.ndarray, label: str = "model", dates: Sequence | None = None):
    """Fit an XGBoost classifier and report a chronological holdout when dates are available."""
    if dates is not None and len(dates) == len(X):
        order = pd.Series(pd.to_datetime(dates, errors="coerce")).sort_values().index
        split = max(1, int(len(order) * 0.8))
        if split >= len(order):
            split = len(order) - 1
        train_idx, val_idx = order[:split], order[split:]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        split_label = "chronological holdout"
    else:
        stratify = y if min(np.bincount(y.astype(int))) >= 2 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        split_label = "random holdout"
    fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
    try:
        model.fit(X_train, y_train, early_stopping_rounds=30, **fit_kwargs)
    except TypeError:
        try:
            model.set_params(early_stopping_rounds=30)
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)
    val_pred = np.argmax(val_probs, axis=1)
    labels = sorted(np.unique(y).astype(int).tolist())
    acc = accuracy_score(y_val, val_pred)
    loss = log_loss(y_val, val_probs, labels=labels)
    print(f"  {label} {split_label} accuracy={acc:.3f}, log-loss={loss:.3f} ({len(X_val)} matches)")
    return model, {"accuracy": float(acc), "log_loss": float(loss), "n_val": int(len(X_val)), "split": split_label}


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
