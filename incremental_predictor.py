#!/usr/bin/env python3
"""Incremental World Cup predictor with strict temporal ordering.

This script implements INCREMENTAL_SPEC.md:
- Train an expanding-window multiclass XGBoost model before each World Cup.
- Walk through actual FIFA World Cup matches chronologically.
- For each match, build pre-match features only from prior data.
- Predict match outcome, then update match state with actual result.
- Backtest winner prediction strategies across all World Cups (1930-2022).
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared import harmonize_country


def ensure_xgboost():
    """Import xgboost, installing it inline if missing."""
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
        from xgboost import XGBClassifier  # type: ignore
    return XGBClassifier


XGBClassifier = ensure_xgboost()


RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "incremental"

RESULTS_PATH = DATA_DIR / "results.csv"
GOALSCORERS_PATH = DATA_DIR / "goalscorers.csv"
SHOOTOUTS_PATH = DATA_DIR / "shootouts.csv"
COUNTRY_DATASET_PATH = DATA_DIR / "world_cup_predictors_dataset.csv"

WC_YEARS = [
    1930,
    1934,
    1938,
    1950,
    1954,
    1958,
    1962,
    1966,
    1970,
    1974,
    1978,
    1982,
    1986,
    1990,
    1994,
    1998,
    2002,
    2006,
    2010,
    2014,
    2018,
    2022,
]

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

STAGE_TO_INT = {"group": 0, "round_of_16": 1, "quarterfinal": 2, "semifinal": 3, "final": 4}
INT_TO_STAGE = {v: k for k, v in STAGE_TO_INT.items()}


@dataclass
class TeamState:
    elo: float = 1500.0
    outcomes: Deque[int] = field(default_factory=lambda: deque(maxlen=10))  # 1,0,-1
    goals_for: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    goals_against: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    last_date: Optional[pd.Timestamp] = None


@dataclass
class H2HState:
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    goals_a: float = 0.0
    goals_b: float = 0.0
    matches: int = 0


def canonical_team(name: object) -> object:
    if pd.isna(name):
        return name
    return harmonize_country(name)


def canonicalize(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(canonical_team)
    return out


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def outcome_class(home_score: float, away_score: float) -> int:
    if home_score > away_score:
        return 0
    if home_score < away_score:
        return 2
    return 1


def expected_home_score(home_elo: float, away_elo: float, home_advantage: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + home_advantage)) / 400.0))


def elo_margin_multiplier(goal_diff: float, elo_diff: float) -> float:
    diff = max(goal_diff, 1.0)
    return (math.log(diff + 1.0) * 2.2) / (2.2 + 0.001 * abs(elo_diff))


def safe_mean(values: Deque[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(np.mean(values))


def team_snapshot(team: str, states: Dict[str, TeamState], date: pd.Timestamp) -> Dict[str, float]:
    state = states.get(team, TeamState())
    wins = sum(1 for x in state.outcomes if x == 1)
    draws = sum(1 for x in state.outcomes if x == 0)
    losses = sum(1 for x in state.outcomes if x == -1)
    n = max(len(state.outcomes), 1)

    gf5 = float(np.mean(list(state.goals_for)[-5:])) if state.goals_for else 0.0
    ga5 = float(np.mean(list(state.goals_against)[-5:])) if state.goals_against else 0.0
    gf10 = safe_mean(state.goals_for, 0.0)
    ga10 = safe_mean(state.goals_against, 0.0)

    if state.last_date is None:
        rest_days = 30.0
    else:
        rest_days = float(max((date - state.last_date).days, 0))

    return {
        "elo": float(state.elo),
        "form_win_rate_10": wins / n if n > 0 else 0.0,
        "form_draw_rate_10": draws / n if n > 0 else 0.0,
        "form_loss_rate_10": losses / n if n > 0 else 0.0,
        "goals_for_avg_5": gf5,
        "goals_against_avg_5": ga5,
        "goals_for_avg_10": gf10,
        "goals_against_avg_10": ga10,
        "rest_days": rest_days,
        "form_sample_n": float(len(state.outcomes)),
    }


def h2h_snapshot(h2h: Dict[Tuple[str, str], H2HState], home: str, away: str) -> Dict[str, float]:
    key = tuple(sorted((home, away)))
    state = h2h.get(key, H2HState())

    if key[0] == home:
        home_wins, away_wins = state.a_wins, state.b_wins
        home_goals, away_goals = state.goals_a, state.goals_b
    else:
        home_wins, away_wins = state.b_wins, state.a_wins
        home_goals, away_goals = state.goals_b, state.goals_a

    n = max(state.matches, 1)
    return {
        "h2h_matches": float(state.matches),
        "h2h_home_win_rate": home_wins / n,
        "h2h_draw_rate": state.draws / n,
        "h2h_away_win_rate": away_wins / n,
        "h2h_home_goals_avg": home_goals / n,
        "h2h_home_goals_against_avg": away_goals / n,
    }


def update_h2h(
    h2h: Dict[Tuple[str, str], H2HState], home: str, away: str, home_score: float, away_score: float
) -> None:
    key = tuple(sorted((home, away)))
    state = h2h.setdefault(key, H2HState())

    if key[0] == home:
        state.goals_a += home_score
        state.goals_b += away_score
        if home_score > away_score:
            state.a_wins += 1
        elif away_score > home_score:
            state.b_wins += 1
        else:
            state.draws += 1
    else:
        state.goals_a += away_score
        state.goals_b += home_score
        if home_score > away_score:
            state.b_wins += 1
        elif away_score > home_score:
            state.a_wins += 1
        else:
            state.draws += 1
    state.matches += 1


def update_team_states(
    states: Dict[str, TeamState],
    date: pd.Timestamp,
    home: str,
    away: str,
    home_score: float,
    away_score: float,
    neutral: bool,
) -> None:
    home_state = states.setdefault(home, TeamState())
    away_state = states.setdefault(away, TeamState())

    if home_score > away_score:
        home_state.outcomes.append(1)
        away_state.outcomes.append(-1)
        actual_home = 1.0
    elif home_score < away_score:
        home_state.outcomes.append(-1)
        away_state.outcomes.append(1)
        actual_home = 0.0
    else:
        home_state.outcomes.append(0)
        away_state.outcomes.append(0)
        actual_home = 0.5

    home_state.goals_for.append(float(home_score))
    home_state.goals_against.append(float(away_score))
    away_state.goals_for.append(float(away_score))
    away_state.goals_against.append(float(home_score))
    home_state.last_date = date
    away_state.last_date = date

    home_adv = 0.0 if neutral else 50.0
    exp_home = expected_home_score(home_state.elo, away_state.elo, home_adv)
    margin = elo_margin_multiplier(abs(home_score - away_score), home_state.elo - away_state.elo)
    delta = 32.0 * margin * (actual_home - exp_home)
    home_state.elo += delta
    away_state.elo -= delta


def lookup_shootout_winner(
    shootout_lookup: Dict[Tuple[str, str, str], str], date: pd.Timestamp, home: str, away: str
) -> Optional[str]:
    key = (date.strftime("%Y-%m-%d"), home, away)
    if key in shootout_lookup:
        return shootout_lookup[key]
    rev = (date.strftime("%Y-%m-%d"), away, home)
    if rev in shootout_lookup:
        return shootout_lookup[rev]
    return None


def select_final_match_index(
    wc_matches: pd.DataFrame, year: int, shootout_lookup: Dict[Tuple[str, str, str], str]
) -> int:
    actual_winner = canonical_team(WC_WINNERS[year])
    df = wc_matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    latest_date = df["date"].max()
    same_day = df[df["date"] == latest_date]
    involving_winner = same_day[(same_day["home_team"] == actual_winner) | (same_day["away_team"] == actual_winner)]
    if not involving_winner.empty:
        return int(involving_winner.index[0])

    if year == 1950:
        end_slice = df.tail(min(6, len(df)))
        candidates = end_slice[(end_slice["home_team"] == actual_winner) | (end_slice["away_team"] == actual_winner)]
        if not candidates.empty:
            return int(candidates.index[-1])

    return int(df.index[-1])


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


def assign_wc_stage_map(
    wc_matches: pd.DataFrame, year: int, shootout_lookup: Dict[Tuple[str, str, str], str]
) -> Dict[int, int]:
    """Assign stage per WC match uid using year-specific schedule templates."""
    df = wc_matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    total = len(df)
    stage_labels = ["group"] * total

    g, r16, qf, sf = stage_counts_for_year(year, total)
    idx = 0
    for _ in range(g):
        if idx < total:
            stage_labels[idx] = "group"
            idx += 1
    for _ in range(r16):
        if idx < total:
            stage_labels[idx] = "round_of_16"
            idx += 1
    for _ in range(qf):
        if idx < total:
            stage_labels[idx] = "quarterfinal"
            idx += 1
    for _ in range(sf):
        if idx < total:
            stage_labels[idx] = "semifinal"
            idx += 1
    while idx < total:
        stage_labels[idx] = "semifinal"
        idx += 1

    final_idx = select_final_match_index(df, year, shootout_lookup)
    if 0 <= final_idx < total:
        stage_labels[final_idx] = "final"

    return {int(df.iloc[i]["match_uid"]): STAGE_TO_INT[stage_labels[i]] for i in range(total)}


def build_country_feature_lookup(country_df: pd.DataFrame) -> Tuple[Dict[Tuple[str, int], Dict[str, float]], List[str]]:
    numeric_cols = [c for c in country_df.columns if pd.api.types.is_numeric_dtype(country_df[c])]
    excluded = {
        "wc_year",
        "won_wc",
        "runner_up",
        "semifinalist",
        "finalist",
        "top4",
        "is_winner",
        "gdp_per_capita_vs_winner",
        "population_vs_winner",
        "total_goals_in_tournament",
        "avg_goals_per_match",
    }
    feature_cols = [c for c in numeric_cols if c not in excluded]

    lookup: Dict[Tuple[str, int], Dict[str, float]] = {}
    for row in country_df.itertuples(index=False):
        team = canonical_team(getattr(row, "country"))
        year = int(getattr(row, "wc_year"))
        vals = {col: float(getattr(row, col)) if not pd.isna(getattr(row, col)) else np.nan for col in feature_cols}
        lookup[(str(team), year)] = vals
    return lookup, feature_cols


def get_country_vector(
    team: str,
    match_year: int,
    country_lookup: Dict[Tuple[str, int], Dict[str, float]],
    country_years_by_team: Dict[str, List[int]],
    country_feature_cols: List[str],
) -> Dict[str, float]:
    years = country_years_by_team.get(team, [])
    prior = [y for y in years if y < match_year]
    if not prior:
        return {f"country_{c}": np.nan for c in country_feature_cols}
    selected_year = max(prior)
    base = country_lookup.get((team, selected_year), {})
    return {f"country_{c}": base.get(c, np.nan) for c in country_feature_cols}


def build_wc_experience_snapshots(
    wc_matches_all: pd.DataFrame, shootout_lookup: Dict[Tuple[str, str, str], str]
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Build prior-WC experience snapshot for each WC year (strictly previous tournaments)."""
    snapshots: Dict[int, Dict[str, Dict[str, float]]] = {}

    participations = defaultdict(int)
    titles = defaultdict(int)
    last_title_year: Dict[str, int] = {}
    last_final_year: Dict[str, int] = {}
    wc_wins = defaultdict(int)
    wc_matches = defaultdict(int)

    for year in WC_YEARS:
        year_matches = wc_matches_all[wc_matches_all["year"] == year].copy()
        teams_in_year = set(year_matches["home_team"]).union(set(year_matches["away_team"]))
        all_known = set(participations.keys()).union(teams_in_year)

        year_snapshot: Dict[str, Dict[str, float]] = {}
        for team in all_known:
            part = participations[team]
            title_count = titles[team]
            ys_title = float(year - last_title_year[team]) if team in last_title_year else 99.0
            ys_final = float(year - last_final_year[team]) if team in last_final_year else 99.0
            prior_matches = wc_matches[team]
            win_rate = (wc_wins[team] / prior_matches) if prior_matches > 0 else 0.0
            year_snapshot[team] = {
                "wc_prior_participations": float(part),
                "wc_prior_titles": float(title_count),
                "wc_years_since_title": ys_title,
                "wc_years_since_final": ys_final,
                "wc_prior_win_rate": float(win_rate),
            }
        snapshots[year] = year_snapshot

        if year_matches.empty:
            continue

        participants = set(year_matches["home_team"]).union(set(year_matches["away_team"]))
        for team in participants:
            participations[team] += 1

        winner = str(canonical_team(WC_WINNERS[year]))
        titles[winner] += 1
        last_title_year[winner] = year

        final_idx = select_final_match_index(year_matches, year, shootout_lookup)
        final_match = year_matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True).iloc[final_idx]
        f_home = str(final_match["home_team"])
        f_away = str(final_match["away_team"])
        f_hs = float(final_match["home_score"])
        f_as = float(final_match["away_score"])
        if f_hs > f_as:
            finalist = f_away
        elif f_as > f_hs:
            finalist = f_home
        else:
            so_winner = lookup_shootout_winner(shootout_lookup, final_match["date"], f_home, f_away)
            if so_winner == f_home:
                finalist = f_away
            elif so_winner == f_away:
                finalist = f_home
            else:
                finalist = f_away if winner == f_home else f_home

        last_final_year[winner] = year
        last_final_year[finalist] = year

        for row in year_matches.itertuples(index=False):
            hs = float(row.home_score)
            a_s = float(row.away_score)
            home = str(row.home_team)
            away = str(row.away_team)
            wc_matches[home] += 1
            wc_matches[away] += 1
            if hs > a_s:
                wc_wins[home] += 1
            elif a_s > hs:
                wc_wins[away] += 1

    return snapshots


def progress_snapshot(progress: Dict[str, Dict[str, float]], team: str) -> Dict[str, float]:
    row = progress.get(team, {"played": 0.0, "points": 0.0, "gf": 0.0, "ga": 0.0, "wins": 0.0})
    played = max(row["played"], 1.0)
    return {
        "played": row["played"],
        "points": row["points"],
        "goals_for": row["gf"],
        "goals_against": row["ga"],
        "goal_diff": row["gf"] - row["ga"],
        "points_per_match": row["points"] / played,
        "wins": row["wins"],
    }


def update_progress(
    progress: Dict[str, Dict[str, float]], home: str, away: str, home_score: float, away_score: float
) -> None:
    h = progress.setdefault(home, {"played": 0.0, "points": 0.0, "gf": 0.0, "ga": 0.0, "wins": 0.0})
    a = progress.setdefault(away, {"played": 0.0, "points": 0.0, "gf": 0.0, "ga": 0.0, "wins": 0.0})
    h["played"] += 1.0
    a["played"] += 1.0
    h["gf"] += home_score
    h["ga"] += away_score
    a["gf"] += away_score
    a["ga"] += home_score
    if home_score > away_score:
        h["points"] += 3.0
        h["wins"] += 1.0
    elif away_score > home_score:
        a["points"] += 3.0
        a["wins"] += 1.0
    else:
        h["points"] += 1.0
        a["points"] += 1.0


def build_feature_row(
    row: pd.Series,
    states: Dict[str, TeamState],
    h2h: Dict[Tuple[str, str], H2HState],
    wc_progress: Dict[str, Dict[str, float]],
    wc_experience_snapshots: Dict[int, Dict[str, Dict[str, float]]],
    country_lookup: Dict[Tuple[str, int], Dict[str, float]],
    country_years_by_team: Dict[str, List[int]],
    country_feature_cols: List[str],
    stage_map: Dict[int, int],
) -> Dict[str, float]:
    date = row["date"]
    year = int(date.year)
    home = str(row["home_team"])
    away = str(row["away_team"])
    neutral = parse_bool(row["neutral"])
    host_country = canonical_team(row["country"])
    is_wc = str(row["tournament"]) == "FIFA World Cup"
    stage = stage_map.get(int(row["match_uid"]), STAGE_TO_INT["group"]) if is_wc else STAGE_TO_INT["group"]

    home_dyn = team_snapshot(home, states, date)
    away_dyn = team_snapshot(away, states, date)
    pair = h2h_snapshot(h2h, home, away)

    exp_map = wc_experience_snapshots.get(year, {})
    home_exp = exp_map.get(
        home,
        {
            "wc_prior_participations": 0.0,
            "wc_prior_titles": 0.0,
            "wc_years_since_title": 99.0,
            "wc_years_since_final": 99.0,
            "wc_prior_win_rate": 0.0,
        },
    )
    away_exp = exp_map.get(
        away,
        {
            "wc_prior_participations": 0.0,
            "wc_prior_titles": 0.0,
            "wc_years_since_title": 99.0,
            "wc_years_since_final": 99.0,
            "wc_prior_win_rate": 0.0,
        },
    )

    home_country = get_country_vector(home, year, country_lookup, country_years_by_team, country_feature_cols)
    away_country = get_country_vector(away, year, country_lookup, country_years_by_team, country_feature_cols)

    home_prog = progress_snapshot(wc_progress, home) if is_wc else progress_snapshot({}, home)
    away_prog = progress_snapshot(wc_progress, away) if is_wc else progress_snapshot({}, away)

    features: Dict[str, float] = {
        "neutral": float(1 if neutral else 0),
        "stage": float(stage),
        "home_is_host": float(1 if (not neutral and host_country == home) else 0),
    }

    for k, v in home_dyn.items():
        features[f"home_{k}"] = float(v)
    for k, v in away_dyn.items():
        features[f"away_{k}"] = float(v)
    for k, v in pair.items():
        features[k] = float(v)
    for k, v in home_exp.items():
        features[f"home_{k}"] = float(v)
    for k, v in away_exp.items():
        features[f"away_{k}"] = float(v)
    for k, v in home_prog.items():
        features[f"home_wc_prog_{k}"] = float(v)
    for k, v in away_prog.items():
        features[f"away_wc_prog_{k}"] = float(v)

    for k in country_feature_cols:
        hk = f"country_{k}"
        features[f"home_{hk}"] = float(home_country.get(hk, np.nan))
        features[f"away_{hk}"] = float(away_country.get(hk, np.nan))
        features[f"diff_{hk}"] = features[f"home_{hk}"] - features[f"away_{hk}"]

    features["elo_diff"] = features["home_elo"] - features["away_elo"]
    features["elo_sum"] = features["home_elo"] + features["away_elo"]
    features["form_win_rate_diff"] = features["home_form_win_rate_10"] - features["away_form_win_rate_10"]
    features["form_draw_rate_diff"] = features["home_form_draw_rate_10"] - features["away_form_draw_rate_10"]
    features["goals_for_5_diff"] = features["home_goals_for_avg_5"] - features["away_goals_for_avg_5"]
    features["goals_against_5_diff"] = features["home_goals_against_avg_5"] - features["away_goals_against_avg_5"]
    features["rest_days_diff"] = features["home_rest_days"] - features["away_rest_days"]
    features["h2h_advantage"] = features["h2h_home_win_rate"] - features["h2h_away_win_rate"]
    features["wc_points_diff"] = features["home_wc_prog_points"] - features["away_wc_prog_points"]
    features["wc_goal_diff_diff"] = features["home_wc_prog_goal_diff"] - features["away_wc_prog_goal_diff"]

    return features


def build_global_feature_table(
    results: pd.DataFrame,
    stage_map: Dict[int, int],
    wc_experience_snapshots: Dict[int, Dict[str, Dict[str, float]]],
    country_lookup: Dict[Tuple[str, int], Dict[str, float]],
    country_years_by_team: Dict[str, List[int]],
    country_feature_cols: List[str],
) -> pd.DataFrame:
    states: Dict[str, TeamState] = {}
    h2h: Dict[Tuple[str, str], H2HState] = {}
    rows: List[Dict[str, object]] = []
    current_wc_year: Optional[int] = None
    wc_progress: Dict[str, Dict[str, float]] = {}

    sorted_results = results.sort_values(["date", "home_team", "away_team", "match_uid"]).reset_index(drop=True)
    for r in sorted_results.itertuples(index=False):
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            continue

        if r.tournament == "FIFA World Cup":
            year = int(r.date.year)
            if current_wc_year != year:
                wc_progress = {}
                current_wc_year = year
        else:
            current_wc_year = None
            wc_progress = {}

        row_series = pd.Series(r._asdict())
        feat = build_feature_row(
            row=row_series,
            states=states,
            h2h=h2h,
            wc_progress=wc_progress,
            wc_experience_snapshots=wc_experience_snapshots,
            country_lookup=country_lookup,
            country_years_by_team=country_years_by_team,
            country_feature_cols=country_feature_cols,
            stage_map=stage_map,
        )

        hs = float(r.home_score)
        a_s = float(r.away_score)
        out = {
            "match_uid": int(r.match_uid),
            "date": r.date,
            "year": int(r.date.year),
            "home_team": r.home_team,
            "away_team": r.away_team,
            "tournament": r.tournament,
            "target": outcome_class(hs, a_s),
            "home_score": hs,
            "away_score": a_s,
        }
        out.update(feat)
        rows.append(out)

        update_team_states(states, r.date, r.home_team, r.away_team, hs, a_s, parse_bool(r.neutral))
        update_h2h(h2h, r.home_team, r.away_team, hs, a_s)
        if r.tournament == "FIFA World Cup":
            update_progress(wc_progress, r.home_team, r.away_team, hs, a_s)

    return pd.DataFrame(rows)


def train_xgb(train_df: pd.DataFrame, feature_cols: List[str]):
    x = train_df[feature_cols].astype(float)
    y = train_df["target"].astype(int)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=260,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2.0,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    model.fit(x, y)
    return model


def build_state_until_date(results: pd.DataFrame, cutoff: pd.Timestamp) -> Tuple[Dict[str, TeamState], Dict[Tuple[str, str], H2HState]]:
    states: Dict[str, TeamState] = {}
    h2h: Dict[Tuple[str, str], H2HState] = {}
    subset = results[(results["date"] < cutoff) & results["home_score"].notna() & results["away_score"].notna()].copy()
    subset = subset.sort_values(["date", "home_team", "away_team", "match_uid"])
    for r in subset.itertuples(index=False):
        hs = float(r.home_score)
        a_s = float(r.away_score)
        update_team_states(states, r.date, r.home_team, r.away_team, hs, a_s, parse_bool(r.neutral))
        update_h2h(h2h, r.home_team, r.away_team, hs, a_s)
    return states, h2h


def run_incremental_backtest(
    results: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_cols: List[str],
    stage_map: Dict[int, int],
    wc_experience_snapshots: Dict[int, Dict[str, Dict[str, float]]],
    country_lookup: Dict[Tuple[str, int], Dict[str, float]],
    country_years_by_team: Dict[str, List[int]],
    country_feature_cols: List[str],
    shootout_lookup: Dict[Tuple[str, str, str], str],
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    wc_results = results[(results["tournament"] == "FIFA World Cup")].copy()
    wc_results["year"] = wc_results["date"].dt.year.astype(int)

    wc_rows: List[Dict[str, object]] = []
    match_rows: List[Dict[str, object]] = []
    auc_rows: List[Dict[str, float]] = []
    final_model = None

    for year in WC_YEARS:
        year_matches = wc_results[wc_results["year"] == year].copy()
        if year_matches.empty:
            continue
        year_matches = year_matches.sort_values(["date", "home_team", "away_team", "match_uid"]).reset_index(drop=True)
        wc_start = year_matches["date"].min()

        train_df = features_df[features_df["date"] < wc_start].copy()
        if train_df.empty or train_df["target"].nunique() < 3:
            continue

        # Time-based validation window: years [year-4, year)
        val_start = pd.Timestamp(year=year - 4, month=1, day=1)
        val_df = train_df[train_df["date"] >= val_start].copy()
        train_core = train_df[train_df["date"] < val_start].copy()

        val_acc = float("nan")
        val_ll = float("nan")
        if len(train_core) > 1000 and len(val_df) > 100 and train_core["target"].nunique() == 3:
            val_model = train_xgb(train_core, feature_cols)
            val_probs = val_model.predict_proba(val_df[feature_cols].astype(float))
            val_pred = np.argmax(val_probs, axis=1)
            val_acc = float(accuracy_score(val_df["target"].astype(int), val_pred))
            val_ll = float(log_loss(val_df["target"].astype(int), val_probs, labels=[0, 1, 2]))

        model = train_xgb(train_df, feature_cols)
        final_model = model

        states, h2h = build_state_until_date(results, wc_start)
        wc_progress: Dict[str, Dict[str, float]] = {}

        pred_win_count = defaultdict(int)
        agg_prob_sum = defaultdict(float)
        y_true: List[int] = []
        y_prob: List[np.ndarray] = []
        final_predicted_winner = None

        final_uid = None
        stage_for_year = assign_wc_stage_map(year_matches, year, shootout_lookup)
        for uid, st in stage_for_year.items():
            if st == STAGE_TO_INT["final"]:
                final_uid = uid
                break

        for m in year_matches.itertuples(index=False):
            m_series = pd.Series(m._asdict())
            feat = build_feature_row(
                row=m_series,
                states=states,
                h2h=h2h,
                wc_progress=wc_progress,
                wc_experience_snapshots=wc_experience_snapshots,
                country_lookup=country_lookup,
                country_years_by_team=country_years_by_team,
                country_feature_cols=country_feature_cols,
                stage_map=stage_map,
            )

            x = pd.DataFrame([feat])[feature_cols].astype(float)
            probs = model.predict_proba(x)[0]
            probs = np.clip(probs, 1e-9, 1.0)
            probs = probs / probs.sum()
            pred_class = int(np.argmax(probs))

            hs = float(m.home_score)
            a_s = float(m.away_score)
            actual_class = outcome_class(hs, a_s)
            y_true.append(actual_class)
            y_prob.append(probs.copy())

            if pred_class == 0:
                predicted_winner = m.home_team
                pred_win_count[m.home_team] += 1
            elif pred_class == 2:
                predicted_winner = m.away_team
                pred_win_count[m.away_team] += 1
            else:
                predicted_winner = "draw"

            agg_prob_sum[m.home_team] += float(probs[0])
            agg_prob_sum[m.away_team] += float(probs[2])

            if int(m.match_uid) == final_uid:
                if pred_class == 0:
                    final_predicted_winner = m.home_team
                elif pred_class == 2:
                    final_predicted_winner = m.away_team
                else:
                    final_predicted_winner = m.home_team if probs[0] >= probs[2] else m.away_team

            if hs > a_s:
                actual_winner = m.home_team
            elif a_s > hs:
                actual_winner = m.away_team
            else:
                actual_winner = lookup_shootout_winner(shootout_lookup, m.date, m.home_team, m.away_team)

            rec: Dict[str, object] = {
                "year": year,
                "date": m.date,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "stage": INT_TO_STAGE.get(int(feat["stage"]), "group"),
                "p_home_win": float(probs[0]),
                "p_draw": float(probs[1]),
                "p_away_win": float(probs[2]),
                "predicted_class": ["home_win", "draw", "away_win"][pred_class],
                "predicted_winner": predicted_winner,
                "actual_class": ["home_win", "draw", "away_win"][actual_class],
                "actual_winner_or_shootout": actual_winner if actual_winner is not None else "draw",
                "is_correct": int(pred_class == actual_class),
            }
            rec.update(feat)
            match_rows.append(rec)

            update_team_states(states, m.date, m.home_team, m.away_team, hs, a_s, parse_bool(m.neutral))
            update_h2h(h2h, m.home_team, m.away_team, hs, a_s)
            update_progress(wc_progress, m.home_team, m.away_team, hs, a_s)

        actual_winner = str(canonical_team(WC_WINNERS[year]))
        teams = sorted(set(year_matches["home_team"]).union(set(year_matches["away_team"])))
        for team in teams:
            auc_rows.append(
                {
                    "year": float(year),
                    "team": team,
                    "score": float(agg_prob_sum.get(team, 0.0)),
                    "is_winner": float(1.0 if team == actual_winner else 0.0),
                }
            )

        most_wins_sorted = sorted(
            teams,
            key=lambda t: (pred_win_count.get(t, 0), agg_prob_sum.get(t, 0.0)),
            reverse=True,
        )
        agg_sorted = sorted(teams, key=lambda t: agg_prob_sum.get(t, 0.0), reverse=True)
        pred_most_wins = most_wins_sorted[0] if most_wins_sorted else None
        pred_agg = agg_sorted[0] if agg_sorted else None
        if final_predicted_winner is None:
            final_predicted_winner = pred_agg

        rank_lookup = {team: i + 1 for i, team in enumerate(agg_sorted)}
        actual_rank = rank_lookup.get(actual_winner, len(agg_sorted) + 1)

        probs_arr = np.array(y_prob)
        match_acc = float(accuracy_score(y_true, np.argmax(probs_arr, axis=1)))
        match_ll = float(log_loss(y_true, probs_arr, labels=[0, 1, 2]))

        print(
            f"{year}: matches={len(year_matches)} "
            f"acc={match_acc:.3f} pred_most_wins={pred_most_wins} pred_final={final_predicted_winner} pred_agg={pred_agg}"
        )

        wc_rows.append(
            {
                "year": year,
                "actual_winner": actual_winner,
                "predicted_most_wins": pred_most_wins,
                "predicted_final_winner": final_predicted_winner,
                "predicted_aggregate": pred_agg,
                "correct_most_wins": int(pred_most_wins == actual_winner),
                "correct_final_winner": int(final_predicted_winner == actual_winner),
                "correct_aggregate": int(pred_agg == actual_winner),
                "actual_rank_aggregate": int(actual_rank),
                "in_top3_aggregate": int(actual_rank <= 3),
                "in_top5_aggregate": int(actual_rank <= 5),
                "match_accuracy": match_acc,
                "match_logloss": match_ll,
                "val_accuracy_year_minus_4_to_year": val_acc,
                "val_logloss_year_minus_4_to_year": val_ll,
            }
        )

    return pd.DataFrame(wc_rows), pd.DataFrame(match_rows), pd.DataFrame(auc_rows), final_model


def save_feature_importance(model, feature_cols: List[str], path: Path) -> None:
    if model is None or not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    if len(importances) != len(feature_cols):
        return
    s = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(30)
    plt.figure(figsize=(12, 9))
    s.sort_values().plot(kind="barh")
    plt.title("Top 30 Feature Importances (XGBoost)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_summary(
    wc_backtest: pd.DataFrame,
    match_predictions: pd.DataFrame,
    auc_rows: pd.DataFrame,
    summary_path: Path,
) -> Dict[str, float]:
    n_wc = len(wc_backtest)
    metrics: Dict[str, float] = {}

    exact_most = float(wc_backtest["correct_most_wins"].mean()) if n_wc else float("nan")
    exact_final = float(wc_backtest["correct_final_winner"].mean()) if n_wc else float("nan")
    exact_agg = float(wc_backtest["correct_aggregate"].mean()) if n_wc else float("nan")
    top3 = float(wc_backtest["in_top3_aggregate"].mean()) if n_wc else float("nan")
    top5 = float(wc_backtest["in_top5_aggregate"].mean()) if n_wc else float("nan")
    match_acc = float(match_predictions["is_correct"].mean()) if len(match_predictions) else float("nan")
    overall_logloss = float(
        log_loss(
            match_predictions["actual_class"].map({"home_win": 0, "draw": 1, "away_win": 2}),
            match_predictions[["p_home_win", "p_draw", "p_away_win"]].values,
            labels=[0, 1, 2],
        )
    )

    auc = float("nan")
    if not auc_rows.empty and len(auc_rows["is_winner"].unique()) > 1:
        auc = float(roc_auc_score(auc_rows["is_winner"], auc_rows["score"]))

    metrics["exact_most_wins"] = exact_most
    metrics["exact_final"] = exact_final
    metrics["exact_aggregate"] = exact_agg
    metrics["top3_aggregate"] = top3
    metrics["top5_aggregate"] = top5
    metrics["match_accuracy"] = match_acc
    metrics["overall_logloss"] = overall_logloss
    metrics["winner_auc"] = auc

    lines = [
        "# Incremental Predictor Summary",
        "",
        "## Setup",
        "- Strategy: strict chronological incremental prediction (no bracket simulation).",
        "- Model: XGBoost multiclass (`home_win`, `draw`, `away_win`).",
        "- Dynamic state: Elo (K=32, +50 home advantage, margin multiplier), form, rolling goals, rest days, H2H.",
        "- Context: stage, neutral/home-host indicators, tournament progress in current WC, WC experience, country-level features from closest preceding WC year.",
        "",
        "## Backtest Results (1930-2022)",
        f"- World Cups evaluated: {n_wc}",
        f"- Exact winner accuracy (most predicted wins): {exact_most:.3f}",
        f"- Exact winner accuracy (predicted final winner): {exact_final:.3f}",
        f"- Exact winner accuracy (aggregate probability): {exact_agg:.3f}",
        f"- Top-3 accuracy (aggregate): {top3:.3f}",
        f"- Top-5 accuracy (aggregate): {top5:.3f}",
        f"- Match-level accuracy (all WC matches): {match_acc:.3f}",
        f"- Match-level log-loss (all WC matches): {overall_logloss:.3f}",
        f"- Winner-vs-non-winner AUC (aggregate score): {auc:.3f}",
        "",
        "## Historical Baseline Comparison",
        "- Country-level model reference: exact 0.409, top-3 0.636",
        "- Monte Carlo simulator reference: exact 0.318, top-3 0.500",
        "",
        "## Per-WC Snapshot",
    ]
    for row in wc_backtest.itertuples(index=False):
        lines.append(
            f"- {row.year}: actual={row.actual_winner}, most_wins={row.predicted_most_wins}, "
            f"final={row.predicted_final_winner}, aggregate={row.predicted_aggregate}, "
            f"acc={row.match_accuracy:.3f}"
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RESULTS_PATH}")
    if not GOALSCORERS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {GOALSCORERS_PATH}")
    if not SHOOTOUTS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {SHOOTOUTS_PATH}")
    if not COUNTRY_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing file: {COUNTRY_DATASET_PATH}")

    results = pd.read_csv(RESULTS_PATH)
    goalscorers = pd.read_csv(GOALSCORERS_PATH)
    shootouts = pd.read_csv(SHOOTOUTS_PATH)
    country = pd.read_csv(COUNTRY_DATASET_PATH)

    results["date"] = pd.to_datetime(results["date"])
    goalscorers["date"] = pd.to_datetime(goalscorers["date"])
    shootouts["date"] = pd.to_datetime(shootouts["date"])
    country["wc_year"] = country["wc_year"].astype(int)

    results = canonicalize(results, ["home_team", "away_team", "country"])
    goalscorers = canonicalize(goalscorers, ["home_team", "away_team", "team"])
    shootouts = canonicalize(shootouts, ["home_team", "away_team", "winner"])
    country = canonicalize(country, ["country"])

    results = results.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    results["match_uid"] = np.arange(len(results), dtype=int)
    return results, goalscorers, shootouts, country


def main() -> None:
    np.random.seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results, _goalscorers, shootouts, country = load_data()
    shootout_lookup = {
        (r.date.strftime("%Y-%m-%d"), str(r.home_team), str(r.away_team)): str(r.winner)
        for r in shootouts.itertuples(index=False)
    }

    wc_matches_all = results[
        (results["tournament"] == "FIFA World Cup")
        & (results["date"].dt.year.isin(WC_YEARS))
        & results["home_score"].notna()
        & results["away_score"].notna()
    ].copy()
    wc_matches_all["year"] = wc_matches_all["date"].dt.year.astype(int)

    stage_map: Dict[int, int] = {}
    for year in WC_YEARS:
        yr = wc_matches_all[wc_matches_all["year"] == year].copy()
        if not yr.empty:
            stage_map.update(assign_wc_stage_map(yr, year, shootout_lookup))

    wc_experience_snapshots = build_wc_experience_snapshots(wc_matches_all, shootout_lookup)
    country_lookup, country_feature_cols = build_country_feature_lookup(country)
    country_years_by_team = defaultdict(list)
    for team, year in country_lookup.keys():
        country_years_by_team[team].append(year)
    for team in country_years_by_team:
        country_years_by_team[team] = sorted(set(country_years_by_team[team]))

    print("Building leakage-safe global feature table...")
    features_df = build_global_feature_table(
        results=results,
        stage_map=stage_map,
        wc_experience_snapshots=wc_experience_snapshots,
        country_lookup=country_lookup,
        country_years_by_team=country_years_by_team,
        country_feature_cols=country_feature_cols,
    )

    non_feature_cols = {
        "match_uid",
        "date",
        "year",
        "home_team",
        "away_team",
        "tournament",
        "target",
        "home_score",
        "away_score",
    }
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols]

    print("Running incremental World Cup backtest...")
    wc_backtest, match_predictions, auc_rows, final_model = run_incremental_backtest(
        results=results,
        features_df=features_df,
        feature_cols=feature_cols,
        stage_map=stage_map,
        wc_experience_snapshots=wc_experience_snapshots,
        country_lookup=country_lookup,
        country_years_by_team=country_years_by_team,
        country_feature_cols=country_feature_cols,
        shootout_lookup=shootout_lookup,
    )

    backtest_path = OUTPUT_DIR / "backtest_results.csv"
    matches_path = OUTPUT_DIR / "match_predictions.csv"
    importance_path = OUTPUT_DIR / "feature_importance.png"
    summary_path = OUTPUT_DIR / "summary.md"

    wc_backtest.to_csv(backtest_path, index=False)
    match_predictions.to_csv(matches_path, index=False)
    save_feature_importance(final_model, feature_cols, importance_path)
    metrics = write_summary(wc_backtest, match_predictions, auc_rows, summary_path)

    print("=" * 88)
    print("INCREMENTAL PREDICTOR COMPLETE")
    print("=" * 88)
    print(f"Backtests: {len(wc_backtest)} World Cups")
    print(
        "Exact winner accuracy -> "
        f"most_wins={metrics['exact_most_wins']:.3f}, "
        f"final={metrics['exact_final']:.3f}, "
        f"aggregate={metrics['exact_aggregate']:.3f}"
    )
    print(
        "Top-k aggregate -> "
        f"top3={metrics['top3_aggregate']:.3f}, "
        f"top5={metrics['top5_aggregate']:.3f}, "
        f"AUC={metrics['winner_auc']:.3f}"
    )
    print(
        "Match-level -> "
        f"accuracy={metrics['match_accuracy']:.3f}, "
        f"logloss={metrics['overall_logloss']:.3f}"
    )
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 88)


if __name__ == "__main__":
    main()
