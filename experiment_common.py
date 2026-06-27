#!/usr/bin/env python3
"""Reusable harness for World Cup model experiments.

Each experiment in this repo should use this module so the 2026 walk-forward
backtest stays identical across model families.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from shared import (
    DATA_DIR,
    GROUP_2026_TEAMS,
    apply_match_to_state,
    blend_probabilities,
    compute_match_features,
    country_features_for_year,
    finalize_feature_frame,
    finalize_world_cup_history,
    fit_dixon_coles,
    harmonize_country,
    infer_world_cup_stage_map,
    load_betting_odds,
    load_country_feature_history,
    load_squad_values,
    make_team_state,
    odds_features_for_match,
    parse_bool,
    prepare_prediction_frame,
    sample_weights,
)
from backtest_2026_wc import actual_result, stage_from_tournament_round


TARGET_ACCURACY = 0.645
TARGET_LOG_LOSS = 0.8858
TARGET_BRIER = 0.1791
RESULT_LABELS = ["Home win", "Draw", "Away win"]
RESULTS_CSV = Path(__file__).resolve().parent / "experiments_results.csv"
RESULTS_JSONL = Path(__file__).resolve().parent / "experiments_results.jsonl"


@dataclass
class TrainingData:
    X: pd.DataFrame
    y: np.ndarray
    dates: list[pd.Timestamp]
    match_meta: list[tuple[str, str, bool]]
    state: Any
    feature_names: list[str]


@dataclass
class ModelBundle:
    model: Any
    state: Any
    feature_names: list[str]
    poisson_model: Any | None = None
    alpha: float = 1.0
    odds: dict | None = None
    squad_values: dict | None = None
    country_history: dict | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    name: str
    accuracy: float
    log_loss: float
    brier: float
    ece: float
    mce: float
    correct: int
    total: int
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "experiment": self.name,
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "brier": self.brier,
            "ece": self.ece,
            "mce": self.mce,
            "correct": self.correct,
            "total": self.total,
            "beats_accuracy": self.accuracy > TARGET_ACCURACY,
            "beats_log_loss": self.log_loss < TARGET_LOG_LOSS,
            "beats_brier": self.brier < TARGET_BRIER,
            "beats_all": (
                self.accuracy > TARGET_ACCURACY
                and self.log_loss < TARGET_LOG_LOSS
                and self.brier < TARGET_BRIER
            ),
            "notes": self.notes,
        }


class NoopFeatureAdapter:
    """Default feature adapter used by most experiments."""

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
        return X.copy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()


class EnhancedFeatureAdapter(NoopFeatureAdapter):
    """Feature additions from Approach 6, kept deterministic for backtests."""

    def _add(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        elo = out.get("elo_diff", pd.Series(0.0, index=out.index)).astype(float)
        form = out.get("weighted_form_diff", pd.Series(0.0, index=out.index)).astype(float)
        squad = out.get("squad_value_diff", pd.Series(np.nan, index=out.index)).astype(float)
        draw_rate = out.get("combined_draw_rate", pd.Series(0.0, index=out.index)).astype(float)
        goals = out.get("expected_total_goals", pd.Series(0.0, index=out.index)).astype(float)

        out["elo_x_weighted_form"] = elo * form
        out["elo_x_draw_rate"] = elo.abs() * draw_rate
        out["elo_diff_sq"] = elo ** 2
        out["elo_diff_abs"] = elo.abs()
        out["squad_value_diff_sq"] = squad ** 2
        out["draw_low_goal_interaction"] = draw_rate / (1.0 + goals.clip(lower=0.0))
        out["coin_flip_elo"] = (elo.abs() <= 75).astype(float)
        out["slight_favorite_elo"] = ((elo.abs() > 75) & (elo.abs() <= 175)).astype(float)
        out["big_favorite_elo"] = (elo.abs() > 175).astype(float)
        out["home_is_elo_favorite"] = (elo > 0).astype(float)
        out["away_is_elo_favorite"] = (elo < 0).astype(float)

        # Approximation from available WC participation count: first-timers get a
        # larger value, frequent participants are treated as recently present.
        wc_part = out.get("wc_participations", pd.Series(0.0, index=out.index)).astype(float)
        out["approx_years_since_wc"] = np.where(wc_part <= 0, 99.0, 4.0 / wc_part.clip(lower=1.0))
        return out

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
        return self._add(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._add(X)


def load_inputs() -> tuple[pd.DataFrame, dict, dict, dict]:
    results_df = pd.read_csv(DATA_DIR / "results.csv")
    country_history = load_country_feature_history()
    odds = load_betting_odds()
    squad_values = load_squad_values()
    return results_df, country_history, odds, squad_values


def build_training_data(
    results_df: pd.DataFrame,
    country_history: dict,
    odds: dict | None,
    squad_values: dict | None,
    *,
    feature_adapter: NoopFeatureAdapter | None = None,
) -> TrainingData:
    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[~((df["date"].dt.year == 2026) & (df["tournament"] == "FIFA World Cup"))]
    df = df.sort_values("date").reset_index(drop=True)

    state = defaultdict(make_team_state)
    rows: list[dict] = []
    labels: list[int] = []
    dates: list[pd.Timestamp] = []
    match_meta: list[tuple[str, str, bool]] = []
    country_feature_cache: dict[int, dict] = {}
    active_wc_year: int | None = None
    active_wc_teams: set[str] = set()
    wc_stage_by_index = infer_world_cup_stage_map(df)

    for _, r in df.iterrows():
        home = harmonize_country(r["home_team"])
        away = harmonize_country(r["away_team"])
        hs, aw = r["home_score"], r["away_score"]
        if pd.isna(hs) or pd.isna(aw):
            continue

        hs, aw = int(hs), int(aw)
        match_date = r["date"]
        feature_year = int(match_date.year)
        if feature_year not in country_feature_cache:
            country_feature_cache[feature_year] = country_features_for_year(country_history, feature_year)

        is_world_cup = r["tournament"] == "FIFA World Cup"
        if active_wc_year is not None and (not is_world_cup or feature_year != active_wc_year):
            finalize_world_cup_history(state, active_wc_year, active_wc_teams)
            active_wc_year = None
            active_wc_teams = set()

        neutral = parse_bool(r.get("neutral", True))
        stage = wc_stage_by_index.get(int(r.name), 0) if is_world_cup else 0
        odds_row = odds_features_for_match(odds, match_date, home, away)
        rows.append(
            compute_match_features(
                home,
                away,
                state,
                country_feature_cache[feature_year],
                stage,
                match_date,
                neutral=neutral,
                is_home=not neutral,
                odds_row=odds_row,
                squad_values=squad_values,
            )
        )
        labels.append(0 if hs > aw else (1 if hs == aw else 2))
        dates.append(match_date)
        match_meta.append((home, away, neutral))

        if is_world_cup:
            if active_wc_year is None:
                active_wc_year = feature_year
            active_wc_teams.update([home, away])

        apply_match_to_state(
            state,
            home,
            away,
            hs,
            aw,
            match_date,
            neutral=neutral,
            is_world_cup=is_world_cup,
        )

    if active_wc_year is not None:
        finalize_world_cup_history(state, active_wc_year, active_wc_teams)

    y = np.asarray(labels, dtype=int)
    adapter = feature_adapter or NoopFeatureAdapter()
    X = adapter.fit_transform(finalize_feature_frame(rows), y)
    return TrainingData(X=X, y=y, dates=dates, match_meta=match_meta, state=state, feature_names=list(X.columns))


def chronological_holdout_indices(dates: Sequence[pd.Timestamp], train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    order = pd.Series(pd.to_datetime(dates, errors="coerce")).sort_values().index.to_numpy()
    split = max(1, int(len(order) * train_frac))
    if split >= len(order):
        split = len(order) - 1
    return order[:split], order[split:]


def fit_dixon_and_alpha(
    results_df: pd.DataFrame,
    model: Any,
    training: TrainingData,
    *,
    alpha_grid: Iterable[float] | None = None,
) -> tuple[Any, float, float]:
    poisson_model = fit_dixon_coles(results_df)
    _, val_idx = chronological_holdout_indices(training.dates)
    p_model = np.asarray(model.predict_proba(training.X.iloc[val_idx]), dtype=float)
    p_poisson = np.asarray(
        [
            poisson_model.outcome_probs(training.match_meta[i][0], training.match_meta[i][1], neutral=training.match_meta[i][2])
            for i in val_idx
        ],
        dtype=float,
    )
    best_alpha, best_loss = 1.0, np.inf
    for alpha in alpha_grid or np.linspace(0.0, 1.0, 21):
        blended = np.asarray(
            [blend_probabilities(p_model[i], p_poisson[i], float(alpha)) for i in range(len(p_model))],
            dtype=float,
        )
        try:
            loss = log_loss(training.y[val_idx], blended, labels=[0, 1, 2])
        except Exception:
            continue
        if loss < best_loss:
            best_alpha = float(alpha)
            best_loss = float(loss)
    return poisson_model, best_alpha, best_loss


def prepare_2026_state(state: Any) -> Any:
    for teams in GROUP_2026_TEAMS.values():
        for team in teams:
            state[harmonize_country(team)]["wc_participations"] += 1
    return state


def align_prediction_frame(
    feature_dict: dict,
    feature_names: Sequence[str],
    feature_adapter: NoopFeatureAdapter | None = None,
) -> pd.DataFrame:
    base_names = list(feature_names)
    adapter = feature_adapter or NoopFeatureAdapter()
    if isinstance(adapter, EnhancedFeatureAdapter):
        # Build with original columns first, then let the adapter add derived columns.
        original_names = [name for name in base_names if name not in {
            "elo_x_weighted_form",
            "elo_x_draw_rate",
            "elo_diff_sq",
            "elo_diff_abs",
            "squad_value_diff_sq",
            "draw_low_goal_interaction",
            "coin_flip_elo",
            "slight_favorite_elo",
            "big_favorite_elo",
            "home_is_elo_favorite",
            "away_is_elo_favorite",
            "approx_years_since_wc",
        }]
        X = prepare_prediction_frame(feature_dict, original_names)
        X = adapter.transform(X)
        for name in base_names:
            if name not in X.columns:
                X[name] = np.nan
        return X[base_names]
    return prepare_prediction_frame(feature_dict, base_names)


def predict_default(
    bundle: ModelBundle,
    home: str,
    away: str,
    state: Any,
    cf: dict,
    stage: int,
    date: pd.Timestamp,
    *,
    neutral: bool,
    is_home: bool,
    feature_adapter: NoopFeatureAdapter | None = None,
) -> np.ndarray:
    odds_row = odds_features_for_match(bundle.odds, date, home, away)
    feature_dict = compute_match_features(
        home,
        away,
        state,
        cf,
        stage,
        date,
        neutral=neutral,
        is_home=is_home,
        odds_row=odds_row,
        squad_values=bundle.squad_values,
    )
    X = align_prediction_frame(feature_dict, bundle.feature_names, feature_adapter)
    probs = np.asarray(bundle.model.predict_proba(X)[0], dtype=float)
    if bundle.poisson_model is not None and bundle.alpha < 1.0:
        probs = blend_probabilities(probs, bundle.poisson_model.outcome_probs(home, away, neutral=neutral), bundle.alpha)
    return normalize_probs(probs)


def normalize_probs(probs: Sequence[float]) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    arr = np.nan_to_num(arr, nan=1.0 / 3.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 1e-12, None)
    return arr / arr.sum()


def calibration_errors(probs: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == actuals).astype(float)
    ece = 0.0
    mce = 0.0
    for lo in np.linspace(0.0, 1.0, n_bins, endpoint=False):
        hi = lo + 1.0 / n_bins
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if not np.any(mask):
            continue
        gap = abs(float(correct[mask].mean()) - float(conf[mask].mean()))
        ece += float(mask.mean()) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def score_probabilities(probs: np.ndarray, actuals: np.ndarray) -> tuple[float, float, float, int]:
    pred = probs.argmax(axis=1)
    correct = int((pred == actuals).sum())
    accuracy = float(accuracy_score(actuals, pred))
    loss = float(log_loss(actuals, probs, labels=[0, 1, 2]))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(actuals)), actuals] = 1.0
    brier = float(np.mean((probs - one_hot) ** 2))
    return accuracy, loss, brier, correct


def run_walk_forward_backtest(
    name: str,
    bundle: ModelBundle,
    *,
    predict_fn: Callable[..., np.ndarray] = predict_default,
    feature_adapter: NoopFeatureAdapter | None = None,
    verbose: bool = True,
) -> ExperimentResult:
    results_df, country_history, odds, squad_values = load_inputs()
    bundle.odds = bundle.odds if bundle.odds is not None else odds
    bundle.squad_values = bundle.squad_values if bundle.squad_values is not None else squad_values
    bundle.country_history = bundle.country_history if bundle.country_history is not None else country_history

    state = prepare_2026_state(copy.deepcopy(bundle.state))
    cf = country_features_for_year(bundle.country_history, 2022)

    results_df["date"] = pd.to_datetime(results_df["date"])
    wc26 = results_df[
        (results_df["tournament"] == "FIFA World Cup") & (results_df["date"].dt.year == 2026)
    ].copy()
    completed = wc26.sort_values("date").reset_index(drop=True)
    completed = completed[completed["home_score"].notna() & completed["away_score"].notna()].copy()

    probs_list: list[np.ndarray] = []
    actuals: list[int] = []
    match_rows: list[dict[str, Any]] = []
    for _, r in completed.iterrows():
        home = harmonize_country(r["home_team"])
        away = harmonize_country(r["away_team"])
        hs = int(r["home_score"])
        aw = int(r["away_score"])
        date = r["date"]
        stage = stage_from_tournament_round(r.get("tournament", ""), home, away)
        neutral = parse_bool(r.get("neutral", True))
        probs = normalize_probs(
            predict_fn(
                bundle,
                home,
                away,
                state,
                cf,
                stage,
                date,
                neutral=neutral,
                is_home=not neutral,
                feature_adapter=feature_adapter,
            )
        )
        if stage > 0:
            total = probs[0] + probs[2]
            probs = np.array([probs[0] / total, 0.0, probs[2] / total]) if total > 0 else np.array([0.5, 0.0, 0.5])

        actual = actual_result(hs, aw)
        probs_list.append(probs)
        actuals.append(actual)
        match_rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "match": f"{home} vs {away}",
                "score": f"{hs}-{aw}",
                "predicted": RESULT_LABELS[int(np.argmax(probs))],
                "actual": RESULT_LABELS[actual],
                "confidence": float(np.max(probs)),
                "p_home": float(probs[0]),
                "p_draw": float(probs[1]),
                "p_away": float(probs[2]),
            }
        )

        apply_match_to_state(state, home, away, hs, aw, date, neutral=neutral, is_world_cup=True)

    probs_arr = np.asarray(probs_list, dtype=float)
    actuals_arr = np.asarray(actuals, dtype=int)
    accuracy, loss, brier, correct = score_probabilities(probs_arr, actuals_arr)
    ece, mce = calibration_errors(probs_arr, actuals_arr)
    result = ExperimentResult(
        name=name,
        accuracy=accuracy,
        log_loss=loss,
        brier=brier,
        ece=ece,
        mce=mce,
        correct=correct,
        total=len(actuals),
        notes="; ".join(bundle.notes),
    )
    if verbose:
        print_result(result)
    save_result(result)
    save_match_log(name, match_rows)
    return result


def print_result(result: ExperimentResult) -> None:
    row = result.as_row()
    print("\n" + "=" * 72)
    print(f"Experiment: {result.name}")
    print("=" * 72)
    print(f"Accuracy:  {result.accuracy:.1%} ({result.correct}/{result.total})")
    print(f"Log-loss:  {result.log_loss:.4f}")
    print(f"Brier:     {result.brier:.4f}")
    print(f"ECE/MCE:   {result.ece:.4f} / {result.mce:.4f}")
    print(
        "Beats target: "
        f"accuracy={row['beats_accuracy']} log_loss={row['beats_log_loss']} "
        f"brier={row['beats_brier']} all={row['beats_all']}"
    )
    if result.notes:
        print(f"Notes: {result.notes}")


def save_result(result: ExperimentResult) -> None:
    row = result.as_row()
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        existing = existing[existing["experiment"] != result.name]
        out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])
    out = out.sort_values(["beats_all", "log_loss", "accuracy"], ascending=[False, True, False])
    out.to_csv(RESULTS_CSV, index=False)
    json_line = pd.DataFrame([row]).to_json(orient="records", lines=True)
    with RESULTS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json_line)


def save_match_log(name: str, rows: Sequence[dict[str, Any]]) -> None:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name.lower())
    path = Path(__file__).resolve().parent / f"experiments_{safe}_matches.csv"
    pd.DataFrame(rows).to_csv(path, index=False)


def make_bundle(
    model: Any,
    training: TrainingData,
    *,
    results_df: pd.DataFrame,
    country_history: dict,
    odds: dict | None,
    squad_values: dict | None,
    tune_poisson_blend: bool = True,
    notes: Optional[list[str]] = None,
) -> ModelBundle:
    poisson_model = None
    alpha = 1.0
    if tune_poisson_blend:
        poisson_model, alpha, blend_loss = fit_dixon_and_alpha(results_df, model, training)
        notes = list(notes or []) + [f"Dixon-Coles blend alpha={alpha:.2f} (holdout log-loss {blend_loss:.4f})"]
    return ModelBundle(
        model=model,
        state=training.state,
        feature_names=training.feature_names,
        poisson_model=poisson_model,
        alpha=alpha,
        odds=odds,
        squad_values=squad_values,
        country_history=country_history,
        notes=list(notes or []),
    )


def sample_weight_array(training: TrainingData, draw_weight: float = 1.6) -> np.ndarray:
    return sample_weights(training.y, training.dates, draw_weight=draw_weight)
