#!/usr/bin/env python3
"""World Cup match-level predictor, simulator, and meta-model backtest.

This script implements MATCH_SPEC.md requirements:
- Leakage-safe match-level feature engineering (strict chronological order)
- Multi-class XGBoost match outcome model (home win / draw / away win)
- Time-based evaluation splits
- Historical World Cup Monte Carlo simulations (1930-2022)
- Meta-model that combines simulation outputs with country-level features
- Backtest metrics and output artifacts under output/match_predictor/
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 42
N_SIMULATIONS = 1000

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "match_predictor"
FIG_DIR = OUTPUT_DIR / "figures"

RESULTS_PATH = DATA_DIR / "results.csv"
GOALSCORERS_PATH = DATA_DIR / "goalscorers.csv"
SHOOTOUTS_PATH = DATA_DIR / "shootouts.csv"
COUNTRY_DATASET_PATH = DATA_DIR / "world_cup_predictors_dataset.csv"

WORLD_CUP_WINNERS = {
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

NAME_ALIASES = {
    "West Germany": "Germany",
    "German DR": "East Germany",
    "East Germany": "East Germany",
    "Soviet Union": "Russia",
    "USSR": "Russia",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Czechoslovakia": "Czech Republic",
    "Zaire": "DR Congo",
    "Dutch Guyana": "Suriname",
    "Burma": "Myanmar",
    "Curaçao": "Curacao",
    "Vietnam Republic": "Vietnam",
    "United Arab Republic": "Egypt",
    "IR Iran": "Iran",
}


@dataclass
class TeamState:
    """Rolling team state used for leakage-safe features."""

    elo: float = 1500.0
    form_points: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    goals_for: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    goals_against: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    scorer_diversity: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    scorer_minute: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    penalty_share: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    own_goal_share: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    last_date: Optional[pd.Timestamp] = None


def canonicalize_team(name: str) -> str:
    if pd.isna(name):
        return name
    return NAME_ALIASES.get(str(name), str(name))


def canonicalize_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].map(canonicalize_team)
    return out


def tournament_type(tournament_name: str) -> str:
    name = str(tournament_name).lower()
    if name == "fifa world cup":
        return "world_cup"
    if "qualification" in name or "qualifier" in name:
        return "qualifier"
    if "friendly" in name:
        return "friendly"
    if any(k in name for k in ["copa", "euro", "nations", "cup", "championship"]):
        return "continental"
    return "other"


def elo_k_factor(match_tournament: str) -> float:
    ttype = tournament_type(match_tournament)
    if ttype == "world_cup":
        return 60.0
    if ttype == "qualifier":
        return 40.0
    if ttype == "continental":
        return 35.0
    if ttype == "friendly":
        return 20.0
    return 30.0


def safe_mean(values: Deque[float], default: float) -> float:
    if len(values) == 0:
        return default
    return float(np.mean(values))


def team_feature_snapshot(team: str, states: Dict[str, TeamState], now: pd.Timestamp) -> Dict[str, float]:
    state = states.get(team, TeamState())
    rest_days = 30.0 if state.last_date is None else max(float((now - state.last_date).days), 0.0)
    win_rate = 0.0
    if len(state.form_points) > 0:
        win_rate = float(np.mean([1.0 if x >= 2.99 else 0.0 for x in state.form_points]))
    return {
        "elo": float(state.elo),
        "form_points_avg": safe_mean(state.form_points, 1.0),
        "form_win_rate": win_rate,
        "goals_for_avg": safe_mean(state.goals_for, 1.0),
        "goals_against_avg": safe_mean(state.goals_against, 1.0),
        "rest_days": rest_days,
        "matches_in_form_window": float(len(state.form_points)),
        "squad_scorer_diversity": safe_mean(state.scorer_diversity, 0.0),
        "squad_scorer_minute_avg": safe_mean(state.scorer_minute, 45.0),
        "squad_penalty_share": safe_mean(state.penalty_share, 0.0),
        "squad_own_goal_share": safe_mean(state.own_goal_share, 0.0),
    }


def h2h_snapshot(h2h: Dict[Tuple[str, str], Dict[str, float]], home: str, away: str) -> Dict[str, float]:
    key = tuple(sorted([home, away]))
    data = h2h.get(
        key,
        {
            "a_wins": 0.0,
            "b_wins": 0.0,
            "draws": 0.0,
            "a_goals": 0.0,
            "b_goals": 0.0,
            "matches": 0.0,
        },
    )
    team_a, team_b = key
    if home == team_a:
        home_wins = data["a_wins"]
        away_wins = data["b_wins"]
        home_goals = data["a_goals"]
        away_goals = data["b_goals"]
    else:
        home_wins = data["b_wins"]
        away_wins = data["a_wins"]
        home_goals = data["b_goals"]
        away_goals = data["a_goals"]
    matches = max(data["matches"], 1.0)
    return {
        "h2h_matches": float(data["matches"]),
        "h2h_home_win_rate": float(home_wins / matches),
        "h2h_draw_rate": float(data["draws"] / matches),
        "h2h_goal_diff_avg": float((home_goals - away_goals) / matches),
    }


def update_h2h(
    h2h: Dict[Tuple[str, str], Dict[str, float]],
    home: str,
    away: str,
    home_score: float,
    away_score: float,
) -> None:
    key = tuple(sorted([home, away]))
    if key not in h2h:
        h2h[key] = {
            "a_wins": 0.0,
            "b_wins": 0.0,
            "draws": 0.0,
            "a_goals": 0.0,
            "b_goals": 0.0,
            "matches": 0.0,
        }
    data = h2h[key]
    team_a, team_b = key
    if home == team_a:
        data["a_goals"] += float(home_score)
        data["b_goals"] += float(away_score)
        if home_score > away_score:
            data["a_wins"] += 1.0
        elif away_score > home_score:
            data["b_wins"] += 1.0
        else:
            data["draws"] += 1.0
    else:
        data["a_goals"] += float(away_score)
        data["b_goals"] += float(home_score)
        if home_score > away_score:
            data["b_wins"] += 1.0
        elif away_score > home_score:
            data["a_wins"] += 1.0
        else:
            data["draws"] += 1.0
    data["matches"] += 1.0


def get_outcome_class(home_score: float, away_score: float) -> int:
    if home_score > away_score:
        return 0
    if home_score < away_score:
        return 2
    return 1


def scorer_defaults(goals_for: float) -> Tuple[float, float, float, float]:
    if goals_for <= 0:
        return (0.0, 45.0, 0.0, 0.0)
    approx_diversity = min(float(goals_for), 3.0)
    return (approx_diversity, 45.0, 0.0, 0.0)


def build_scorer_map(goalscorers: pd.DataFrame) -> Dict[Tuple[str, str, str, str], Tuple[float, float, float, float]]:
    goals = goalscorers.copy()
    goals["date"] = pd.to_datetime(goals["date"])
    goals = canonicalize_columns(goals, ["home_team", "away_team", "team"])
    goals["minute"] = goals["minute"].fillna(45.0).clip(lower=0.0, upper=130.0)
    grouped = (
        goals.groupby(["date", "home_team", "away_team", "team"], as_index=False)
        .agg(
            goals=("scorer", "size"),
            scorer_diversity=("scorer", "nunique"),
            scorer_minute_avg=("minute", "mean"),
            penalty_share=("penalty", "mean"),
            own_goal_share=("own_goal", "mean"),
        )
        .fillna(0.0)
    )
    out: Dict[Tuple[str, str, str, str], Tuple[float, float, float, float]] = {}
    for row in grouped.itertuples(index=False):
        key = (row.date.strftime("%Y-%m-%d"), row.home_team, row.away_team, row.team)
        out[key] = (
            float(row.scorer_diversity),
            float(row.scorer_minute_avg),
            float(row.penalty_share),
            float(row.own_goal_share),
        )
    return out


def apply_match_update(
    states: Dict[str, TeamState],
    h2h: Dict[Tuple[str, str], Dict[str, float]],
    date: pd.Timestamp,
    home: str,
    away: str,
    home_score: float,
    away_score: float,
    tournament: str,
    home_scorer_stats: Tuple[float, float, float, float],
    away_scorer_stats: Tuple[float, float, float, float],
    neutral: bool,
) -> None:
    if home not in states:
        states[home] = TeamState()
    if away not in states:
        states[away] = TeamState()
    home_state = states[home]
    away_state = states[away]

    if home_score > away_score:
        home_points, away_points = 3.0, 0.0
    elif home_score < away_score:
        home_points, away_points = 0.0, 3.0
    else:
        home_points, away_points = 1.0, 1.0

    home_state.form_points.append(home_points)
    away_state.form_points.append(away_points)
    home_state.goals_for.append(float(home_score))
    home_state.goals_against.append(float(away_score))
    away_state.goals_for.append(float(away_score))
    away_state.goals_against.append(float(home_score))

    home_state.scorer_diversity.append(home_scorer_stats[0])
    home_state.scorer_minute.append(home_scorer_stats[1])
    home_state.penalty_share.append(home_scorer_stats[2])
    home_state.own_goal_share.append(home_scorer_stats[3])
    away_state.scorer_diversity.append(away_scorer_stats[0])
    away_state.scorer_minute.append(away_scorer_stats[1])
    away_state.penalty_share.append(away_scorer_stats[2])
    away_state.own_goal_share.append(away_scorer_stats[3])
    home_state.last_date = date
    away_state.last_date = date

    home_advantage = 80.0 if not neutral else 0.0
    expected_home = 1.0 / (1.0 + 10.0 ** ((away_state.elo - (home_state.elo + home_advantage)) / 400.0))
    score_home = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)
    goal_margin_multiplier = 1.0 + max(abs(home_score - away_score) - 1.0, 0.0) * 0.5
    k = elo_k_factor(tournament) * goal_margin_multiplier
    delta = k * (score_home - expected_home)
    home_state.elo += delta
    away_state.elo -= delta

    update_h2h(h2h, home, away, home_score, away_score)


def build_match_feature_dataset(
    results: pd.DataFrame, scorer_map: Dict[Tuple[str, str, str, str], Tuple[float, float, float, float]]
) -> pd.DataFrame:
    matches = results.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = canonicalize_columns(matches, ["home_team", "away_team", "country"])
    matches = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    matches = matches[matches["home_score"].notna() & matches["away_score"].notna()].copy()

    rows: List[Dict[str, object]] = []
    states: Dict[str, TeamState] = {}
    h2h: Dict[Tuple[str, str], Dict[str, float]] = {}

    for row in matches.itertuples(index=False):
        date = row.date
        home = row.home_team
        away = row.away_team
        home_score = float(row.home_score)
        away_score = float(row.away_score)
        neutral = bool(row.neutral)

        home_feat = team_feature_snapshot(home, states, date)
        away_feat = team_feature_snapshot(away, states, date)
        h2h_feat = h2h_snapshot(h2h, home, away)

        feature_row: Dict[str, object] = {
            "date": date,
            "year": int(date.year),
            "home_team": home,
            "away_team": away,
            "tournament": row.tournament,
            "tournament_type": tournament_type(row.tournament),
            "neutral": 1 if neutral else 0,
            "target": get_outcome_class(home_score, away_score),
            "home_score": home_score,
            "away_score": away_score,
        }

        for key, value in home_feat.items():
            feature_row[f"home_{key}"] = value
        for key, value in away_feat.items():
            feature_row[f"away_{key}"] = value
        for key, value in h2h_feat.items():
            feature_row[key] = value

        feature_row["elo_diff"] = feature_row["home_elo"] - feature_row["away_elo"]
        feature_row["form_points_diff"] = feature_row["home_form_points_avg"] - feature_row["away_form_points_avg"]
        feature_row["goals_for_diff"] = feature_row["home_goals_for_avg"] - feature_row["away_goals_for_avg"]
        feature_row["goals_against_diff"] = feature_row["home_goals_against_avg"] - feature_row["away_goals_against_avg"]
        feature_row["rest_days_diff"] = feature_row["home_rest_days"] - feature_row["away_rest_days"]
        feature_row["squad_diversity_diff"] = (
            feature_row["home_squad_scorer_diversity"] - feature_row["away_squad_scorer_diversity"]
        )
        rows.append(feature_row)

        date_key = date.strftime("%Y-%m-%d")
        home_scorer_stats = scorer_map.get((date_key, home, away, home), scorer_defaults(home_score))
        away_scorer_stats = scorer_map.get((date_key, home, away, away), scorer_defaults(away_score))
        apply_match_update(
            states=states,
            h2h=h2h,
            date=date,
            home=home,
            away=away,
            home_score=home_score,
            away_score=away_score,
            tournament=row.tournament,
            home_scorer_stats=home_scorer_stats,
            away_scorer_stats=away_scorer_stats,
            neutral=neutral,
        )

    return pd.DataFrame(rows)


def build_state_until(
    results: pd.DataFrame,
    scorer_map: Dict[Tuple[str, str, str, str], Tuple[float, float, float, float]],
    cutoff_date: pd.Timestamp,
) -> Tuple[Dict[str, TeamState], Dict[Tuple[str, str], Dict[str, float]]]:
    played = results.copy()
    played["date"] = pd.to_datetime(played["date"])
    played = canonicalize_columns(played, ["home_team", "away_team", "country"])
    played = played.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    played = played[
        (played["date"] < cutoff_date) & played["home_score"].notna() & played["away_score"].notna()
    ].copy()

    states: Dict[str, TeamState] = {}
    h2h: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in played.itertuples(index=False):
        home_score = float(row.home_score)
        away_score = float(row.away_score)
        date_key = row.date.strftime("%Y-%m-%d")
        home_scorer_stats = scorer_map.get((date_key, row.home_team, row.away_team, row.home_team), scorer_defaults(home_score))
        away_scorer_stats = scorer_map.get((date_key, row.home_team, row.away_team, row.away_team), scorer_defaults(away_score))
        apply_match_update(
            states=states,
            h2h=h2h,
            date=row.date,
            home=row.home_team,
            away=row.away_team,
            home_score=home_score,
            away_score=away_score,
            tournament=row.tournament,
            home_scorer_stats=home_scorer_stats,
            away_scorer_stats=away_scorer_stats,
            neutral=bool(row.neutral),
        )
    return states, h2h


def build_model_matrix(
    df: pd.DataFrame, feature_cols: List[str], template_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    x = df[feature_cols].copy()
    x = pd.get_dummies(x, columns=["tournament_type"], prefix="tt")
    if template_cols is not None:
        x = x.reindex(columns=template_cols, fill_value=0.0)
        return x, template_cols
    cols = x.columns.tolist()
    return x, cols


def train_match_model(train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[XGBClassifier, List[str]]:
    x_train, model_cols = build_model_matrix(train_df, feature_cols)
    y_train = train_df["target"].astype(int)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2.0,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        n_jobs=4,
    )
    model.fit(x_train, y_train)
    return model, model_cols


def elo_baseline_probs(elo_diff: np.ndarray) -> np.ndarray:
    draw_prob = 0.23
    home_no_draw = 1.0 / (1.0 + np.exp(-elo_diff / 250.0))
    home_prob = (1.0 - draw_prob) * home_no_draw
    away_prob = (1.0 - draw_prob) * (1.0 - home_no_draw)
    return np.vstack([home_prob, np.full_like(home_prob, draw_prob), away_prob]).T


def evaluate_time_splits(feature_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = feature_df.sort_values("date").reset_index(drop=True)
    n = len(df)
    fold_points = [0.55, 0.65, 0.75, 0.85]
    window = max(int(0.08 * n), 1500)
    rows = []
    for i, frac in enumerate(fold_points, start=1):
        train_end = int(frac * n)
        test_end = min(train_end + window, n)
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        if len(train_df) < 5000 or len(test_df) < 500:
            continue
        model, model_cols = train_match_model(train_df, feature_cols)
        x_test, _ = build_model_matrix(test_df, feature_cols, model_cols)
        y_test = test_df["target"].astype(int).values
        probs = model.predict_proba(x_test)
        preds = np.argmax(probs, axis=1)
        baseline_probs = elo_baseline_probs(test_df["elo_diff"].values.astype(float))
        baseline_preds = np.argmax(baseline_probs, axis=1)
        rows.append(
            {
                "fold": i,
                "train_start": train_df["date"].min(),
                "train_end": train_df["date"].max(),
                "test_start": test_df["date"].min(),
                "test_end": test_df["date"].max(),
                "samples_train": len(train_df),
                "samples_test": len(test_df),
                "xgb_accuracy": accuracy_score(y_test, preds),
                "xgb_logloss": log_loss(y_test, probs, labels=[0, 1, 2]),
                "elo_accuracy": accuracy_score(y_test, baseline_preds),
                "elo_logloss": log_loss(y_test, baseline_probs, labels=[0, 1, 2]),
            }
        )
    return pd.DataFrame(rows)


def make_matchup_row(
    match_date: pd.Timestamp,
    home_team: str,
    away_team: str,
    neutral: bool,
    tournament: str,
    states: Dict[str, TeamState],
    h2h: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[str, object]:
    home_feat = team_feature_snapshot(home_team, states, match_date)
    away_feat = team_feature_snapshot(away_team, states, match_date)
    h2h_feat = h2h_snapshot(h2h, home_team, away_team)
    row: Dict[str, object] = {
        "tournament_type": tournament_type(tournament),
        "neutral": 1 if neutral else 0,
    }
    for key, value in home_feat.items():
        row[f"home_{key}"] = value
    for key, value in away_feat.items():
        row[f"away_{key}"] = value
    row.update(h2h_feat)
    row["elo_diff"] = row["home_elo"] - row["away_elo"]
    row["form_points_diff"] = row["home_form_points_avg"] - row["away_form_points_avg"]
    row["goals_for_diff"] = row["home_goals_for_avg"] - row["away_goals_for_avg"]
    row["goals_against_diff"] = row["home_goals_against_avg"] - row["away_goals_against_avg"]
    row["rest_days_diff"] = row["home_rest_days"] - row["away_rest_days"]
    row["squad_diversity_diff"] = row["home_squad_scorer_diversity"] - row["away_squad_scorer_diversity"]
    return row


def resolve_actual_winner(
    row: pd.Series, shootout_lookup: Dict[Tuple[str, str, str], str]
) -> Optional[str]:
    home_team = row["home_team"]
    away_team = row["away_team"]
    home_score = row["home_score"]
    away_score = row["away_score"]
    if home_score > away_score:
        return home_team
    if away_score > home_score:
        return away_team
    key = (row["date"].strftime("%Y-%m-%d"), home_team, away_team)
    return shootout_lookup.get(key, None)


def build_knockout_tree(
    wc_matches: pd.DataFrame, actual_winner: str, shootout_lookup: Dict[Tuple[str, str, str], str]
) -> Tuple[int, Dict[int, Dict[str, Optional[int]]], Dict[int, Optional[str]]]:
    data = wc_matches.copy()
    data["actual_winner"] = data.apply(resolve_actual_winner, axis=1, shootout_lookup=shootout_lookup)
    data = data.sort_values(["date", "match_id"]).reset_index(drop=True)
    candidates = data[data["actual_winner"] == actual_winner]
    if candidates.empty:
        final_id = int(data.iloc[-1]["match_id"])
    else:
        final_id = int(candidates.sort_values(["date", "match_id"]).iloc[-1]["match_id"])

    id_to_row = {int(r.match_id): r for r in data.itertuples(index=False)}
    sources: Dict[int, Dict[str, Optional[int]]] = {}
    seen = {final_id}
    queue = [final_id]
    used_children = set()

    while queue:
        node = queue.pop(0)
        row = id_to_row[node]
        sources[node] = {"home_child": None, "away_child": None}
        for side in ["home", "away"]:
            team = getattr(row, f"{side}_team")
            prev = data[(data["date"] < row.date) & (data["actual_winner"] == team)]
            if prev.empty:
                continue
            prev = prev[~prev["match_id"].isin(used_children)]
            if prev.empty:
                continue
            child_id = int(prev.sort_values(["date", "match_id"]).iloc[-1]["match_id"])
            sources[node][f"{side}_child"] = child_id
            used_children.add(child_id)
            if child_id not in seen:
                seen.add(child_id)
                queue.append(child_id)

    return final_id, sources, {int(r.match_id): r.actual_winner for r in data.itertuples(index=False)}


def predict_match_prob(
    model: XGBClassifier,
    model_cols: List[str],
    feature_cols: List[str],
    row_dict: Dict[str, object],
) -> np.ndarray:
    x, _ = build_model_matrix(pd.DataFrame([row_dict]), feature_cols, model_cols)
    probs = model.predict_proba(x)[0]
    probs = np.clip(probs.astype(float), 1e-8, 1.0)
    probs = probs / probs.sum()
    return probs


def run_tournament_simulation(
    wc_matches: pd.DataFrame,
    model: XGBClassifier,
    model_cols: List[str],
    feature_cols: List[str],
    states: Dict[str, TeamState],
    h2h: Dict[Tuple[str, str], Dict[str, float]],
    final_id: int,
    sources: Dict[int, Dict[str, Optional[int]]],
    n_sims: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    matches = wc_matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    champ_counts = defaultdict(int)
    probability_cache: Dict[Tuple[int, str, str], np.ndarray] = {}

    per_match_rows = []
    for row in matches.itertuples(index=False):
        matchup_row = make_matchup_row(
            match_date=row.date,
            home_team=row.home_team,
            away_team=row.away_team,
            neutral=bool(row.neutral),
            tournament=row.tournament,
            states=states,
            h2h=h2h,
        )
        probs = predict_match_prob(model, model_cols, feature_cols, matchup_row)
        per_match_rows.append(
            {
                "year": int(row.date.year),
                "date": row.date,
                "match_id": int(row.match_id),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "p_home_win": probs[0],
                "p_draw": probs[1],
                "p_away_win": probs[2],
                "predicted_class": ["home_win", "draw", "away_win"][int(np.argmax(probs))],
            }
        )
        probability_cache[(int(row.match_id), row.home_team, row.away_team)] = probs

    for _ in range(n_sims):
        winners_by_match: Dict[int, str] = {}
        for row in matches.itertuples(index=False):
            match_id = int(row.match_id)
            if match_id in sources:
                source = sources[match_id]
                home_team = (
                    winners_by_match[source["home_child"]]
                    if source.get("home_child") is not None
                    else row.home_team
                )
                away_team = (
                    winners_by_match[source["away_child"]]
                    if source.get("away_child") is not None
                    else row.away_team
                )
            else:
                home_team = row.home_team
                away_team = row.away_team

            cache_key = (match_id, home_team, away_team)
            if cache_key in probability_cache:
                probs = probability_cache[cache_key]
            else:
                matchup_row = make_matchup_row(
                    match_date=row.date,
                    home_team=home_team,
                    away_team=away_team,
                    neutral=bool(row.neutral),
                    tournament=row.tournament,
                    states=states,
                    h2h=h2h,
                )
                probs = predict_match_prob(model, model_cols, feature_cols, matchup_row)
                probability_cache[cache_key] = probs

            sampled_class = int(rng.choice([0, 1, 2], p=probs))
            if sampled_class == 0:
                winner = home_team
            elif sampled_class == 2:
                winner = away_team
            else:
                non_draw = probs[0] + probs[2]
                if non_draw <= 1e-8:
                    winner = home_team if rng.random() < 0.5 else away_team
                else:
                    winner = home_team if rng.random() < (probs[0] / non_draw) else away_team
            winners_by_match[match_id] = winner

        champion = winners_by_match.get(final_id)
        if champion is not None:
            champ_counts[champion] += 1

    teams = sorted(set(matches["home_team"]).union(set(matches["away_team"])))
    champion_probs = {team: champ_counts[team] / float(n_sims) for team in teams}
    return champion_probs, pd.DataFrame(per_match_rows)


def compute_topk_and_auc(result_rows: List[Dict[str, object]], prob_rows: pd.DataFrame) -> Dict[str, float]:
    n = len(result_rows)
    exact = float(sum(1 for r in result_rows if bool(r["correct"])))
    top3 = float(sum(1 for r in result_rows if int(r["actual_rank"]) <= 3))
    top5 = float(sum(1 for r in result_rows if int(r["actual_rank"]) <= 5))
    auc = float("nan")
    if len(prob_rows["is_winner"].unique()) > 1:
        auc = float(roc_auc_score(prob_rows["is_winner"], prob_rows["win_prob"]))
    return {
        "tournaments": n,
        "exact_winner_accuracy": exact / n if n else float("nan"),
        "top3_accuracy": top3 / n if n else float("nan"),
        "top5_accuracy": top5 / n if n else float("nan"),
        "pooled_auc": auc,
    }


def run_match_level_backtest(
    feature_df: pd.DataFrame,
    results: pd.DataFrame,
    scorer_map: Dict[Tuple[str, str, str, str], Tuple[float, float, float, float]],
    shootout_lookup: Dict[Tuple[str, str, str], str],
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_wc = results.copy()
    results_wc["date"] = pd.to_datetime(results_wc["date"])
    results_wc = canonicalize_columns(results_wc, ["home_team", "away_team", "country"])
    results_wc = results_wc[
        (results_wc["tournament"] == "FIFA World Cup")
        & results_wc["home_score"].notna()
        & results_wc["away_score"].notna()
    ].copy()
    results_wc["year"] = results_wc["date"].dt.year.astype(int)

    summary_rows: List[Dict[str, object]] = []
    team_prob_rows: List[Dict[str, object]] = []
    all_match_prob_rows: List[pd.DataFrame] = []

    for year in sorted(WORLD_CUP_WINNERS):
        wc_matches = results_wc[results_wc["year"] == year].copy()
        if wc_matches.empty:
            continue
        wc_matches = wc_matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
        wc_matches["match_id"] = np.arange(len(wc_matches), dtype=int)

        start_date = wc_matches["date"].min()
        train_df = feature_df[feature_df["date"] < start_date].copy()
        if train_df["target"].nunique() < 3:
            continue

        model, model_cols = train_match_model(train_df, feature_cols)
        states, h2h = build_state_until(results, scorer_map, start_date)
        actual_winner = canonicalize_team(WORLD_CUP_WINNERS[year])
        final_id, sources, _ = build_knockout_tree(wc_matches, actual_winner, shootout_lookup)
        champion_probs, match_probs = run_tournament_simulation(
            wc_matches=wc_matches,
            model=model,
            model_cols=model_cols,
            feature_cols=feature_cols,
            states=states,
            h2h=h2h,
            final_id=final_id,
            sources=sources,
            n_sims=N_SIMULATIONS,
        )
        match_probs["year"] = year
        all_match_prob_rows.append(match_probs)

        ranking = sorted(champion_probs.items(), key=lambda kv: kv[1], reverse=True)
        predicted_winner = ranking[0][0]
        rank_lookup = {team: idx + 1 for idx, (team, _) in enumerate(ranking)}
        actual_rank = rank_lookup.get(actual_winner, len(ranking) + 1)
        summary_rows.append(
            {
                "year": year,
                "actual_winner": actual_winner,
                "predicted_winner": predicted_winner,
                "predicted_winner_prob": ranking[0][1],
                "actual_winner_prob": champion_probs.get(actual_winner, 0.0),
                "actual_rank": actual_rank,
                "correct": int(predicted_winner == actual_winner),
            }
        )
        for team, prob in ranking:
            team_prob_rows.append(
                {
                    "year": year,
                    "team": team,
                    "win_prob": prob,
                    "is_winner": int(team == actual_winner),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(team_prob_rows), pd.concat(all_match_prob_rows, ignore_index=True)


def run_meta_model_backtest(match_team_probs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    country = pd.read_csv(COUNTRY_DATASET_PATH)
    country = canonicalize_columns(country, ["country"])
    country["wc_year"] = country["wc_year"].astype(int)
    prob_map = {
        (int(r.year), r.team): float(r.win_prob)
        for r in match_team_probs.itertuples(index=False)
    }
    country["match_win_prob"] = country.apply(
        lambda r: prob_map.get((int(r["wc_year"]), canonicalize_team(r["country"])), 0.0), axis=1
    )

    drop_cols = {
        "won_wc",
        "runner_up",
        "semifinalist",
        "finalist",
        "top4",
        "is_winner",
        "country",
        "iso3",
        "confederation",
        "gdp_per_capita_vs_winner",
        "population_vs_winner",
        "total_goals_in_tournament",
        "avg_goals_per_match",
    }
    numeric_cols = [c for c in country.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(country[c])]
    numeric_cols = [c for c in numeric_cols if country[c].notna().mean() >= 0.35]
    feature_cols = sorted(set(numeric_cols + ["match_win_prob"]) - {"wc_year"})

    rows = []
    prob_rows = []
    for year in sorted(WORLD_CUP_WINNERS):
        train = country[country["wc_year"] < year].copy()
        test = country[country["wc_year"] == year].copy()
        if train.empty or test.empty:
            continue
        y_train = train["won_wc"].astype(int).values
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue
        fold_feature_cols = [c for c in feature_cols if train[c].notna().any()]
        if "match_win_prob" not in fold_feature_cols:
            fold_feature_cols.append("match_win_prob")
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_train = imp.fit_transform(train[fold_feature_cols])
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(imp.transform(test[fold_feature_cols]))
        model = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            C=0.5,
        )
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        tmp = test[["country", "won_wc"]].copy()
        tmp["pred_prob"] = probs
        tmp = tmp.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        tmp["rank"] = np.arange(1, len(tmp) + 1)
        actual_winner = canonicalize_team(WORLD_CUP_WINNERS[year])
        actual_row = tmp[tmp["country"] == actual_winner]
        actual_rank = int(actual_row["rank"].iloc[0]) if not actual_row.empty else len(tmp) + 1
        predicted_winner = str(tmp.iloc[0]["country"])
        rows.append(
            {
                "year": year,
                "actual_winner": actual_winner,
                "predicted_winner": predicted_winner,
                "predicted_winner_prob": float(tmp.iloc[0]["pred_prob"]),
                "actual_winner_prob": float(actual_row["pred_prob"].iloc[0]) if not actual_row.empty else 0.0,
                "actual_rank": actual_rank,
                "correct": int(predicted_winner == actual_winner),
            }
        )
        for r in tmp.itertuples(index=False):
            prob_rows.append(
                {
                    "year": year,
                    "team": r.country,
                    "win_prob": float(r.pred_prob),
                    "is_winner": int(r.won_wc),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(prob_rows)


def save_figures(
    time_split_df: pd.DataFrame,
    match_metrics: Dict[str, float],
    meta_metrics: Dict[str, float],
    match_summary: pd.DataFrame,
    meta_summary: pd.DataFrame,
) -> None:
    if not time_split_df.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(time_split_df["fold"], time_split_df["xgb_accuracy"], marker="o", label="XGBoost")
        plt.plot(time_split_df["fold"], time_split_df["elo_accuracy"], marker="o", label="Elo baseline")
        plt.xlabel("Time split fold")
        plt.ylabel("Accuracy")
        plt.title("Time-based split accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "time_split_accuracy.png", dpi=150)
        plt.close()

    metric_names = ["exact_winner_accuracy", "top3_accuracy", "top5_accuracy", "pooled_auc"]
    plt.figure(figsize=(9, 5))
    x = np.arange(len(metric_names))
    w = 0.35
    plt.bar(x - w / 2, [match_metrics[m] for m in metric_names], width=w, label="Match simulator")
    plt.bar(x + w / 2, [meta_metrics[m] for m in metric_names], width=w, label="Meta-model")
    plt.xticks(x, ["Exact", "Top-3", "Top-5", "AUC"])
    plt.ylim(0, 1)
    plt.title("Backtest performance comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "backtest_comparison.png", dpi=150)
    plt.close()

    if not match_summary.empty and not meta_summary.empty:
        merged = match_summary[["year", "actual_rank"]].merge(
            meta_summary[["year", "actual_rank"]], on="year", suffixes=("_match", "_meta")
        )
        plt.figure(figsize=(10, 5))
        plt.plot(merged["year"], merged["actual_rank_match"], marker="o", label="Match simulator rank")
        plt.plot(merged["year"], merged["actual_rank_meta"], marker="o", label="Meta-model rank")
        plt.axhline(3, color="gray", linestyle="--", linewidth=1)
        plt.axhline(5, color="gray", linestyle=":", linewidth=1)
        plt.gca().invert_yaxis()
        plt.xlabel("World Cup year")
        plt.ylabel("Actual winner rank (lower is better)")
        plt.title("Actual winner rank by tournament")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "winner_rank_over_time.png", dpi=150)
        plt.close()


def write_summary(
    time_split_df: pd.DataFrame,
    match_metrics: Dict[str, float],
    meta_metrics: Dict[str, float],
    match_summary: pd.DataFrame,
    meta_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Match-Level World Cup Predictor Summary",
        "",
        "## Data and leakage controls",
        "- Match-level features are generated strictly in chronological order.",
        "- Every feature for a match uses only data available before kickoff.",
        "- Name harmonization includes Germany/West Germany, USSR/Russia, Yugoslavia/Serbia, and additional aliases.",
        "",
        "## Match model",
        "- Model: XGBoost multi-class (`home_win`, `draw`, `away_win`).",
        "- Features: Elo, recent form, rolling goals, H2H, rest days, scorer-based squad proxies, neutral/tournament context.",
        "",
        "## Time-based split performance",
    ]
    if time_split_df.empty:
        lines.append("- No time splits were generated.")
    else:
        lines.append(f"- Mean XGBoost accuracy: {time_split_df['xgb_accuracy'].mean():.4f}")
        lines.append(f"- Mean Elo baseline accuracy: {time_split_df['elo_accuracy'].mean():.4f}")
        lines.append(f"- Mean XGBoost log-loss: {time_split_df['xgb_logloss'].mean():.4f}")
    lines += [
        "",
        "## Historical tournament simulation (1930-2022, 1000 Monte Carlo runs each)",
        f"- Exact winner accuracy: {match_metrics['exact_winner_accuracy']:.4f}",
        f"- Top-3 accuracy: {match_metrics['top3_accuracy']:.4f}",
        f"- Top-5 accuracy: {match_metrics['top5_accuracy']:.4f}",
        f"- Pooled AUC: {match_metrics['pooled_auc']:.4f}",
        "",
        "## Meta-model (country-level + match win probability)",
        f"- Exact winner accuracy: {meta_metrics['exact_winner_accuracy']:.4f}",
        f"- Top-3 accuracy: {meta_metrics['top3_accuracy']:.4f}",
        f"- Top-5 accuracy: {meta_metrics['top5_accuracy']:.4f}",
        f"- Pooled AUC: {meta_metrics['pooled_auc']:.4f}",
        "",
        "## Best predicted winners by year",
    ]
    if not meta_summary.empty:
        for row in meta_summary.itertuples(index=False):
            mark = "OK" if row.correct else "MISS"
            lines.append(f"- {row.year}: actual={row.actual_winner}, predicted={row.predicted_winner} ({mark})")
    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    np.random.seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_PATH.exists() or not GOALSCORERS_PATH.exists() or not SHOOTOUTS_PATH.exists():
        raise FileNotFoundError("Required files are missing in data/.")

    results = pd.read_csv(RESULTS_PATH)
    goalscorers = pd.read_csv(GOALSCORERS_PATH)
    shootouts = pd.read_csv(SHOOTOUTS_PATH)

    results = canonicalize_columns(results, ["home_team", "away_team", "country"])
    goalscorers = canonicalize_columns(goalscorers, ["home_team", "away_team", "team"])
    shootouts = canonicalize_columns(shootouts, ["home_team", "away_team", "winner"])

    shootouts["date"] = pd.to_datetime(shootouts["date"])
    shootout_lookup = {
        (r.date.strftime("%Y-%m-%d"), r.home_team, r.away_team): r.winner
        for r in shootouts.itertuples(index=False)
    }

    scorer_map = build_scorer_map(goalscorers)
    feature_df = build_match_feature_dataset(results, scorer_map)
    feature_df = feature_df.sort_values("date").reset_index(drop=True)
    feature_df.to_csv(OUTPUT_DIR / "match_features.csv", index=False)

    non_feature_cols = {
        "date",
        "year",
        "home_team",
        "away_team",
        "tournament",
        "home_score",
        "away_score",
        "target",
    }
    feature_cols = [c for c in feature_df.columns if c not in non_feature_cols]
    if "tournament_type" not in feature_cols:
        raise RuntimeError("Expected feature `tournament_type` was not generated.")

    time_split_df = evaluate_time_splits(feature_df, feature_cols)
    time_split_df.to_csv(OUTPUT_DIR / "time_split_metrics.csv", index=False)

    match_summary, match_team_probs, match_probs_per_game = run_match_level_backtest(
        feature_df=feature_df,
        results=results,
        scorer_map=scorer_map,
        shootout_lookup=shootout_lookup,
        feature_cols=feature_cols,
    )
    match_summary.to_csv(OUTPUT_DIR / "historical_simulation_summary.csv", index=False)
    match_team_probs.to_csv(OUTPUT_DIR / "historical_team_win_probs.csv", index=False)
    match_probs_per_game.to_csv(OUTPUT_DIR / "historical_match_probs.csv", index=False)

    meta_summary, meta_prob_rows = run_meta_model_backtest(match_team_probs)
    meta_summary.to_csv(OUTPUT_DIR / "meta_backtest_summary.csv", index=False)
    meta_prob_rows.to_csv(OUTPUT_DIR / "meta_backtest_team_probs.csv", index=False)

    match_metrics = compute_topk_and_auc(match_summary.to_dict("records"), match_team_probs)
    meta_metrics = compute_topk_and_auc(meta_summary.to_dict("records"), meta_prob_rows)

    save_figures(time_split_df, match_metrics, meta_metrics, match_summary, meta_summary)
    write_summary(time_split_df, match_metrics, meta_metrics, match_summary, meta_summary)

    metrics_df = pd.DataFrame(
        [
            {"model": "match_simulator", **match_metrics},
            {"model": "meta_model", **meta_metrics},
        ]
    )
    metrics_df.to_csv(OUTPUT_DIR / "backtest_metrics.csv", index=False)

    print("=" * 88)
    print("MATCH PREDICTOR COMPLETE")
    print("=" * 88)
    print(f"Engineered matches: {len(feature_df):,}")
    if not time_split_df.empty:
        print(f"Time-split XGBoost accuracy (mean): {time_split_df['xgb_accuracy'].mean():.4f}")
        print(f"Time-split Elo baseline accuracy (mean): {time_split_df['elo_accuracy'].mean():.4f}")
    print(
        f"Match simulator backtest: exact={match_metrics['exact_winner_accuracy']:.4f}, "
        f"top3={match_metrics['top3_accuracy']:.4f}, top5={match_metrics['top5_accuracy']:.4f}, "
        f"auc={match_metrics['pooled_auc']:.4f}"
    )
    print(
        f"Meta-model backtest: exact={meta_metrics['exact_winner_accuracy']:.4f}, "
        f"top3={meta_metrics['top3_accuracy']:.4f}, top5={meta_metrics['top5_accuracy']:.4f}, "
        f"auc={meta_metrics['pooled_auc']:.4f}"
    )
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 88)


if __name__ == "__main__":
    main()
