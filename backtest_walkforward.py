#!/usr/bin/env python3
"""Comprehensive chronological walk-forward backtest.

This validates every completed match from 2014 onward. For each target match the
feature row is computed from state built only from earlier results, the model is
trained on the expanding set of earlier feature rows, and the actual result is
applied only after scoring the prediction.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

from shared import (
    DATA_DIR,
    ODDS_FEATURE_COLUMNS,
    SQUAD_VALUE_FEATURE_COLUMNS,
    apply_match_to_state,
    blend_probabilities,
    compute_match_features,
    country_features_for_year,
    finalize_feature_frame,
    finalize_world_cup_history,
    fit_dixon_coles,
    fit_xgb_with_validation,
    harmonize_country,
    infer_world_cup_stage_map,
    load_betting_odds,
    load_country_feature_history,
    load_squad_values,
    make_team_state,
    odds_features_for_match,
    parse_bool,
    sample_weights,
)


RESULT_LABELS = ["Home win", "Draw", "Away win"]
XGB_BLEND_WEIGHT = 0.25
RETRAIN_EVERY = 1000
ROLLING_WINDOW = 500


@dataclass
class PreparedMatches:
    matches: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    dates: list[pd.Timestamp]


@dataclass
class FittedModels:
    xgb_model: Any
    dixon_coles: Any
    feature_names: list[str]
    train_matches: int


def actual_result(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def tournament_type(tournament: str) -> str:
    text = str(tournament).lower()
    if text == "fifa world cup":
        return "World Cup"
    if "qualification" in text or "qualifier" in text:
        return "Qualifier"
    if "friendly" in text:
        return "Friendly"
    continental_terms = (
        "africa cup",
        "african cup",
        "asian cup",
        "copa america",
        "copa américa",
        "euro",
        "gold cup",
        "nations cup",
        "oceania",
        "uefa",
        "concacaf",
        "caf",
        "afc",
        "ofc",
        "conmebol",
    )
    if any(term in text for term in continental_terms):
        return "Continental"
    return "Other"


def normalize_probs(probs: Iterable[float]) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    arr = np.nan_to_num(arr, nan=1.0 / 3.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 1e-12, None)
    return arr / arr.sum()


def load_inputs(*, use_external_data: bool) -> tuple[pd.DataFrame, dict, dict | None, dict | None]:
    results = pd.read_csv(DATA_DIR / "results.csv")
    country_history = load_country_feature_history()
    odds = load_betting_odds() if use_external_data else None
    squad_values = load_squad_values() if use_external_data else None
    return results, country_history, odds, squad_values


def build_chronological_feature_set(
    results_df: pd.DataFrame,
    country_history: dict,
    *,
    odds: dict | None,
    squad_values: dict | None,
) -> PreparedMatches:
    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    wc_stage_by_index = infer_world_cup_stage_map(df)

    state = defaultdict(make_team_state)
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    dates: list[pd.Timestamp] = []
    match_rows: list[dict[str, Any]] = []
    country_feature_cache: dict[int, dict] = {}
    active_wc_year: int | None = None
    active_wc_teams: set[str] = set()

    for idx, row in df.iterrows():
        home = harmonize_country(row["home_team"])
        away = harmonize_country(row["away_team"])
        match_date = row["date"]
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        year = int(match_date.year)
        tournament = row.get("tournament", "")
        is_world_cup = tournament == "FIFA World Cup"

        if active_wc_year is not None and (not is_world_cup or year != active_wc_year):
            finalize_world_cup_history(state, active_wc_year, active_wc_teams)
            active_wc_year = None
            active_wc_teams = set()

        if year not in country_feature_cache:
            country_feature_cache[year] = country_features_for_year(country_history, year)

        neutral = parse_bool(row.get("neutral", True))
        stage = wc_stage_by_index.get(int(idx), 0) if is_world_cup else 0
        odds_row = odds_features_for_match(odds, match_date, home, away)
        feature_row = compute_match_features(
            home,
            away,
            state,
            country_feature_cache[year],
            stage,
            match_date,
            neutral=neutral,
            is_home=not neutral,
            odds_row=odds_row,
            squad_values=squad_values,
        )
        if not odds:
            for col in ODDS_FEATURE_COLUMNS:
                feature_row[col] = np.nan
        if not squad_values:
            for col in SQUAD_VALUE_FEATURE_COLUMNS:
                feature_row[col] = np.nan

        rows.append(feature_row)
        labels.append(actual_result(home_score, away_score))
        dates.append(match_date)
        match_rows.append(
            {
                "date": match_date,
                "year": year,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": tournament,
                "tournament_type": tournament_type(tournament),
                "neutral": neutral,
                "stage": stage,
            }
        )

        if is_world_cup:
            if active_wc_year is None:
                active_wc_year = year
            active_wc_teams.update([home, away])

        apply_match_to_state(
            state,
            home,
            away,
            home_score,
            away_score,
            match_date,
            neutral=neutral,
            is_world_cup=is_world_cup,
        )

    if active_wc_year is not None:
        finalize_world_cup_history(state, active_wc_year, active_wc_teams)

    return PreparedMatches(
        matches=pd.DataFrame(match_rows),
        X=finalize_feature_frame(rows),
        y=np.asarray(labels, dtype=int),
        dates=dates,
    )


def fit_models(prepared: PreparedMatches, train_idx: np.ndarray) -> FittedModels:
    X_train = prepared.X.iloc[train_idx].copy()
    y_train = prepared.y[train_idx]
    train_dates = [prepared.dates[int(i)] for i in train_idx]

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    weights = sample_weights(y_train, train_dates)
    model, _ = fit_xgb_with_validation(
        model,
        X_train,
        y_train,
        label="XGBoost",
        dates=train_dates,
        sample_weight=weights,
        calibrate=True,
    )

    train_matches = prepared.matches.iloc[train_idx][
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "neutral",
        ]
    ].rename(columns={"home_team": "home_team", "away_team": "away_team"})
    dixon_coles = fit_dixon_coles(train_matches, exclude_2026_wc=False)
    return FittedModels(
        xgb_model=model,
        dixon_coles=dixon_coles,
        feature_names=list(X_train.columns),
        train_matches=len(train_idx),
    )


def predict_row(prepared: PreparedMatches, models: FittedModels, row_idx: int) -> np.ndarray:
    match = prepared.matches.iloc[row_idx]
    X_row = prepared.X.iloc[[row_idx]][models.feature_names]
    p_xgb = np.asarray(models.xgb_model.predict_proba(X_row)[0], dtype=float)
    p_dc = np.asarray(
        models.dixon_coles.outcome_probs(
            match["home_team"],
            match["away_team"],
            neutral=bool(match["neutral"]),
        ),
        dtype=float,
    )
    return normalize_probs(blend_probabilities(p_xgb, p_dc, XGB_BLEND_WEIGHT))


def score_rows(results: pd.DataFrame) -> dict[str, float | int]:
    probs = results[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    actuals = results["actual_idx"].to_numpy(dtype=int)
    predicted = results["predicted_idx"].to_numpy(dtype=int)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(actuals)), actuals] = 1.0
    return {
        "matches": int(len(results)),
        "correct": int((predicted == actuals).sum()),
        "accuracy": float(accuracy_score(actuals, predicted)),
        "log_loss": float(log_loss(actuals, probs, labels=[0, 1, 2])),
        "brier": float(np.mean((probs - one_hot) ** 2)),
    }


def grouped_metrics(results: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, part in results.groupby(group_col, dropna=False):
        metrics = score_rows(part)
        metrics[group_col] = value
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out[[group_col, "matches", "correct", "accuracy", "log_loss", "brier"]].sort_values(group_col)


def calibration_table(results: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    rows = []
    ece = 0.0
    mce = 0.0
    total = len(results)
    for lo, hi in [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        mask = (results["confidence"] >= lo) & (
            (results["confidence"] < hi) if hi < 1.0 else (results["confidence"] <= hi)
        )
        part = results[mask]
        if part.empty:
            rows.append(
                {
                    "bucket": f"{int(lo * 100)}-{int(hi * 100)}%",
                    "matches": 0,
                    "avg_confidence": np.nan,
                    "accuracy": np.nan,
                    "gap": np.nan,
                }
            )
            continue
        avg_conf = float(part["confidence"].mean())
        acc = float(part["correct"].mean())
        gap = abs(acc - avg_conf)
        ece += (len(part) / total) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bucket": f"{int(lo * 100)}-{int(hi * 100)}%",
                "matches": int(len(part)),
                "avg_confidence": avg_conf,
                "accuracy": acc,
                "gap": gap,
            }
        )
    return pd.DataFrame(rows), float(ece), float(mce)


def rolling_accuracy(results: pd.DataFrame, *, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    rows = []
    for start in range(0, len(results), window):
        part = results.iloc[start:start + window]
        if part.empty:
            continue
        rows.append(
            {
                "segment": f"{start + 1}-{start + len(part)}",
                "matches": int(len(part)),
                "start_date": part.iloc[0]["date"],
                "end_date": part.iloc[-1]["date"],
                "accuracy": float(part["correct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def format_metric_table(df: pd.DataFrame, label_col: str) -> str:
    if df.empty:
        return "(none)"
    out = df.copy()
    for col in ("accuracy", "log_loss", "brier"):
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return markdown_table(out)


def format_calibration(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in ("avg_confidence", "accuracy", "gap"):
        out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return markdown_table(out)


def format_rolling(df: pd.DataFrame) -> str:
    out = df.copy()
    out["accuracy"] = out["accuracy"].map(lambda x: f"{x:.4f}")
    return markdown_table(out)


def write_summary(
    path: Path,
    *,
    title: str,
    start_year: int,
    use_external_data: bool,
    results: pd.DataFrame,
    overall: dict[str, float | int],
    by_year: pd.DataFrame,
    by_tournament: pd.DataFrame,
    calibration: pd.DataFrame,
    ece: float,
    mce: float,
    rolling: pd.DataFrame,
) -> str:
    text = f"""# {title}

## Configuration

| Setting | Value |
| --- | --- |
| Validation start | {start_year} |
| Validation matches | {overall['matches']} |
| Retrain interval | {RETRAIN_EVERY} matches |
| Blend | {int((1 - XGB_BLEND_WEIGHT) * 100)}% Dixon-Coles / {int(XGB_BLEND_WEIGHT * 100)}% XGBoost |
| Betting odds and squad values | {"enabled when available" if use_external_data else "disabled (all NaN)"} |

## Overall Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | {overall['accuracy']:.4f} |
| Correct | {overall['correct']} / {overall['matches']} |
| Log-loss | {overall['log_loss']:.4f} |
| Brier score | {overall['brier']:.4f} |
| ECE | {ece:.4f} |
| MCE | {mce:.4f} |

## Per-Year Metrics

{format_metric_table(by_year, "year")}

## Per-Tournament-Type Metrics

{format_metric_table(by_tournament, "tournament_type")}

## Calibration

{format_calibration(calibration)}

## Rolling {ROLLING_WINDOW}-Match Accuracy

{format_rolling(rolling)}
"""
    path.write_text(text, encoding="utf-8")
    return text


def print_report(
    *,
    title: str,
    overall: dict[str, float | int],
    by_year: pd.DataFrame,
    by_tournament: pd.DataFrame,
    calibration: pd.DataFrame,
    ece: float,
    mce: float,
    rolling: pd.DataFrame,
) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Matches:   {overall['matches']}")
    print(f"Correct:   {overall['correct']}")
    print(f"Accuracy:  {overall['accuracy']:.4f}")
    print(f"Log-loss:  {overall['log_loss']:.4f}")
    print(f"Brier:     {overall['brier']:.4f}")
    print(f"ECE/MCE:   {ece:.4f} / {mce:.4f}")

    print("\nPer-tournament-type metrics:")
    print(format_metric_table(by_tournament, "tournament_type"))
    print("\nPer-year metrics:")
    print(format_metric_table(by_year, "year"))
    print("\nCalibration:")
    print(format_calibration(calibration))
    print(f"\nRolling {ROLLING_WINDOW}-match accuracy:")
    print(format_rolling(rolling))


def run_backtest(
    *,
    start_year: int = 2014,
    use_external_data: bool = True,
    results_path: Path = Path("backtest_walkforward_results.csv"),
    summary_path: Path = Path("backtest_walkforward_summary.md"),
    title: str = "Walk-Forward Backtest (2014+)",
    retrain_every: int = RETRAIN_EVERY,
) -> pd.DataFrame:
    print("=" * 80)
    print(title)
    print("=" * 80)
    print("Loading data...")
    results_df, country_history, odds, squad_values = load_inputs(use_external_data=use_external_data)
    print(f"Results rows: {len(results_df):,}")
    if use_external_data:
        print(f"Betting odds fixtures: {len(odds or {}) // 2:,}")
        print(f"Squad value team-years: {len(squad_values or {}):,}")
    else:
        print("Betting odds and squad values disabled; external features stay NaN.")

    print("Precomputing chronological feature rows...")
    prepared = build_chronological_feature_set(
        results_df,
        country_history,
        odds=odds,
        squad_values=squad_values,
    )
    validation_idx = prepared.matches.index[prepared.matches["year"] >= start_year].to_numpy()
    if len(validation_idx) == 0:
        raise ValueError(f"No completed matches found from {start_year} onward")

    print(f"Feature columns: {len(prepared.X.columns)}")
    print(f"Completed matches with features: {len(prepared.matches):,}")
    print(f"Validation matches from {start_year}: {len(validation_idx):,}")
    print(f"Retraining every {retrain_every:,} validation matches.")

    current_models: FittedModels | None = None
    next_retrain_at = 0
    rows: list[dict[str, Any]] = []

    for position, row_idx in enumerate(validation_idx):
        if current_models is None or position >= next_retrain_at:
            train_idx = np.arange(0, row_idx, dtype=int)
            if len(train_idx) < 100:
                raise ValueError(f"Only {len(train_idx)} prior matches available before first validation row")
            match = prepared.matches.iloc[row_idx]
            print(
                f"\nRetraining at validation match {position + 1:,}/{len(validation_idx):,} "
                f"({match['date'].date()}), training rows={len(train_idx):,}"
            )
            current_models = fit_models(prepared, train_idx)
            next_retrain_at = position + retrain_every

        if (position + 1) % 1000 == 0:
            print(f"Scored {position + 1:,}/{len(validation_idx):,} matches...")

        match = prepared.matches.iloc[row_idx]
        probs = predict_row(prepared, current_models, int(row_idx))
        predicted_idx = int(np.argmax(probs))
        actual_idx = int(prepared.y[row_idx])
        score = f"{int(match['home_score'])}-{int(match['away_score'])}"
        rows.append(
            {
                "date": match["date"].strftime("%Y-%m-%d"),
                "year": int(match["year"]),
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "score": score,
                "home_score": int(match["home_score"]),
                "away_score": int(match["away_score"]),
                "tournament": match["tournament"],
                "tournament_type": match["tournament_type"],
                "neutral": bool(match["neutral"]),
                "stage": int(match["stage"]),
                "predicted": RESULT_LABELS[predicted_idx],
                "actual": RESULT_LABELS[actual_idx],
                "predicted_idx": predicted_idx,
                "actual_idx": actual_idx,
                "correct": bool(predicted_idx == actual_idx),
                "confidence": float(probs[predicted_idx]),
                "actual_prob": float(probs[actual_idx]),
                "p_home": float(probs[0]),
                "p_draw": float(probs[1]),
                "p_away": float(probs[2]),
                "train_matches": int(current_models.train_matches),
            }
        )

    results = pd.DataFrame(rows)
    results_path.write_text(results.to_csv(index=False), encoding="utf-8")

    overall = score_rows(results)
    by_year = grouped_metrics(results, "year")
    by_tournament = grouped_metrics(results, "tournament_type")
    calibration, ece, mce = calibration_table(results)
    rolling = rolling_accuracy(results)
    write_summary(
        summary_path,
        title=title,
        start_year=start_year,
        use_external_data=use_external_data,
        results=results,
        overall=overall,
        by_year=by_year,
        by_tournament=by_tournament,
        calibration=calibration,
        ece=ece,
        mce=mce,
        rolling=rolling,
    )
    print_report(
        title=title,
        overall=overall,
        by_year=by_year,
        by_tournament=by_tournament,
        calibration=calibration,
        ece=ece,
        mce=mce,
        rolling=rolling,
    )
    print(f"\nSaved detailed predictions to {results_path}")
    print(f"Saved summary to {summary_path}")
    return results


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        run_backtest()


if __name__ == "__main__":
    main()
