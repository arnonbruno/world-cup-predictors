"""Shared utilities for World Cup predictor scripts."""

from __future__ import annotations

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
    "Soviet Union": "Russia",
    "USSR": "Russia",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Czechoslovakia": "Czech Republic",
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

THIRD_SLOT_BY_GROUP = {
    "D": 3,   # Match 74
    "F": 9,   # Match 77
    "E": 13,  # Match 79
    "I": 15,  # Match 80
    "B": 17,  # Match 81
    "A": 19,  # Match 82
    "G": 25,  # Match 85
    "L": 29,  # Match 87
}
THIRD_SLOTS = [3, 9, 13, 15, 17, 19, 25, 29]
INITIAL_ELO = 1500
K_FACTOR = 32


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
    ea = expected_score(elo_a, elo_b)
    sa = 1 if score_a > score_b else (0.5 if score_a == score_b else 0)
    margin = abs(score_a - score_b)
    multiplier = np.log(max(margin, 1) + 1)
    return (
        elo_a + K_FACTOR * multiplier * (sa - ea),
        elo_b + K_FACTOR * multiplier * ((1 - sa) - (1 - ea)),
    )


def compute_match_features(team, opponent, state, country_features, stage_num, match_date):
    s, o = state[team], state[opponent]
    form = s["form"][-10:] if s["form"] else [0.5]
    gf5 = s["goals_for"][-5:] or [0]
    ga5 = s["goals_against"][-5:] or [0]
    gf10 = s["goals_for"][-10:] or [0]
    ga10 = s["goals_against"][-10:] or [0]
    rest = min((match_date - s["last_match"]).days if s["last_match"] else 30, 60)
    h2h_key = tuple(sorted([team, opponent]))
    h = s["h2h"].get(h2h_key, {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0})
    hm = max(h["matches"], 1)
    cf, oc = country_features.get(team, {}), country_features.get(opponent, {})
    return {
        "elo": s["elo"], "elo_opponent": o["elo"],
        "elo_diff": s["elo"] - o["elo"], "elo_sum": s["elo"] + o["elo"],
        "form_win_rate": sum(1 for f in form if f == 1) / len(form),
        "form_draw_rate": sum(1 for f in form if f == 0.5) / len(form),
        "form_loss_rate": sum(1 for f in form if f == 0) / len(form),
        "avg_goals_scored_5": np.mean(gf5), "avg_goals_conceded_5": np.mean(ga5),
        "avg_goals_scored_10": np.mean(gf10), "avg_goals_conceded_10": np.mean(ga10),
        "rest_days": rest,
        "h2h_matches": h["matches"], "h2h_win_rate": h["wins"] / hm,
        "h2h_draw_rate": h["draws"] / hm, "h2h_avg_goals_for": h["gf"] / hm,
        "h2h_avg_goals_against": h["ga"] / hm,
        "wc_participations": s["wc_participations"], "wc_titles": s["wc_titles"],
        "wc_win_rate": s["wc_wins"] / max(s["wc_matches"], 1),
        "stage": stage_num, "neutral": 1, "is_home": 0,
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


def fit_xgb_with_validation(model, X: pd.DataFrame, y: np.ndarray, label: str = "model"):
    """Fit an XGBoost classifier with a validation split and print basic metrics."""
    stratify = y if min(np.bincount(y.astype(int))) >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
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
    print(f"  {label} holdout accuracy={acc:.3f}, log-loss={loss:.3f} ({len(X_val)} matches)")
    return model, {"accuracy": float(acc), "log_loss": float(loss), "n_val": int(len(X_val))}


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

    placed_groups = set()
    for group, team, *_ in best_thirds:
        slot = THIRD_SLOT_BY_GROUP.get(group)
        if slot is not None and r32_base[slot] is None:
            r32_base[slot] = team
            placed_groups.add(group)

    open_slots = [slot for slot in THIRD_SLOTS if r32_base[slot] is None]
    unplaced = [(group, team) for group, team, *_ in best_thirds if group not in placed_groups]
    for slot, (_group, team) in zip(open_slots, unplaced):
        r32_base[slot] = team

    missing = [slot for slot in THIRD_SLOTS if r32_base[slot] is None]
    if missing:
        raise ValueError(f"Could not assign third-place teams to R32 slots: {missing}")

    labels = [f"Match {match_num}" for match_num in range(73, 89)]
    return [
        (labels[i // 2], r32_base[i], r32_base[i + 1])
        for i in range(0, len(r32_base), 2)
    ]
