#!/usr/bin/env python3
"""Explain a single World Cup match prediction.

This script is intentionally self-contained, but it expects to be run from the
World Cup prediction project root so it can import the project-specific
``shared.py`` and, when present, ``predict_2026.py``.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except Exception as exc:  # pragma: no cover - runtime dependency check
    shap = None
    SHAP_IMPORT_ERROR = exc
else:
    SHAP_IMPORT_ERROR = None

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover - runtime dependency check
    xgb = None
    XGB_IMPORT_ERROR = exc
else:
    XGB_IMPORT_ERROR = None

try:
    import joblib
except Exception:  # pragma: no cover - optional loader
    joblib = None


OUTCOME_LABELS = ["Home win", "Draw", "Away win"]
CLASS_ALIASES = {
    "home": {"home", "home_win", "h", "1", 1, "Home win"},
    "draw": {"draw", "d", "x", "0", 0, "Draw"},
    "away": {"away", "away_win", "a", "2", 2, -1, "Away win"},
}
FACTOR_KEYWORDS = {
    "Team Strength": (
        "elo",
        "rank",
        "rating",
        "strength",
        "power",
        "fifa",
        "tradition",
        "quality",
    ),
    "Form": (
        "form",
        "recent",
        "last",
        "streak",
        "goal",
        "scored",
        "conced",
        "clean",
        "win_rate",
        "points",
    ),
    "Head-to-Head": ("h2h", "head_to_head", "head2head", "mutual", "vs_record"),
    "Experience": (
        "world_cup",
        "wc_",
        "particip",
        "title",
        "champion",
        "knockout",
        "stage",
        "experience",
    ),
    "Economic/Demographic": (
        "gdp",
        "population",
        "pop_",
        "income",
        "econom",
        "area",
        "development",
    ),
    "Contextual": (
        "host",
        "home_adv",
        "neutral",
        "venue",
        "rest",
        "travel",
        "stage",
        "confederation",
    ),
}
MODEL_PATTERNS = (
    "models/*.json",
    "models/*.ubj",
    "models/*.pkl",
    "models/*.joblib",
    "output/**/*.json",
    "output/**/*.ubj",
    "output/**/*.pkl",
    "output/**/*.joblib",
    "artifacts/**/*.json",
    "artifacts/**/*.ubj",
    "artifacts/**/*.pkl",
    "artifacts/**/*.joblib",
)
MATCH_DATA_PATTERNS = (
    "data/*match*.csv",
    "data/*result*.csv",
    "data/*game*.csv",
    "data/*.csv",
    "output/*match*.csv",
    "output/*result*.csv",
    "output/*prediction*.csv",
    "artifacts/*match*.csv",
)


@dataclass
class RuntimeNotes:
    warnings: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def detail(self, message: str) -> None:
        if message not in self.details:
            self.details.append(message)


@dataclass
class ModelBundle:
    model: Any
    feature_names: list[str]
    class_labels: list[Any]
    state: Any | None = None
    country_features: dict[str, dict[str, float]] | None = None
    train_X: pd.DataFrame | None = None
    train_y: pd.Series | np.ndarray | None = None
    train_probabilities: np.ndarray | None = None
    squad_values: dict[tuple[str, int], dict[str, float]] | None = None
    odds: dict | None = None
    poisson_model: Any | None = None
    alpha: float = 1.0
    wc_calibration_buckets: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MatchColumns:
    date: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    home_score: str | None = None
    away_score: str | None = None
    tournament: str | None = None
    stage: str | None = None
    neutral: str | None = None
    home_elo: str | None = None
    away_elo: str | None = None


@dataclass
class ExplanationContext:
    home: str
    away: str
    year: int
    stage: int
    match_date: pd.Timestamp
    root: Path
    output_dir: Path
    notes: RuntimeNotes
    shared: Any | None = None
    predict_2026: Any | None = None
    country_history: Any | None = None
    matches: pd.DataFrame | None = None


def import_optional_module(name: str, notes: RuntimeNotes) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        notes.warn(f"Could not import `{name}`: {exc}")
        return None


def require_project_functions(shared: Any | None, notes: RuntimeNotes) -> None:
    required = [
        "load_country_feature_history",
        "country_features_for_year",
        "compute_match_features",
        "harmonize_country",
        "fit_xgb_with_validation",
    ]
    if shared is None:
        notes.warn("Project `shared.py` is not importable from this directory.")
        return
    missing = [name for name in required if not hasattr(shared, name)]
    if missing:
        notes.warn(f"`shared.py` is missing expected functions: {', '.join(missing)}")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "team"


def fmt_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    if abs(number) >= 100:
        return f"{number:,.0f}"
    if abs(number) >= 10:
        return f"{number:,.1f}"
    return f"{number:,.{digits}f}"


def normalize_feature_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def flexible_call(func: Callable[..., Any], attempts: Sequence[tuple[tuple[Any, ...], dict[str, Any]]]) -> Any:
    errors: list[str] = []
    for args, kwargs in attempts:
        try:
            return func(*args, **kwargs)
        except TypeError as exc:
            errors.append(str(exc))
            continue
    raise TypeError("; ".join(errors[-3:]))


def call_harmonize(shared: Any | None, team: str) -> str:
    if shared is None or not hasattr(shared, "harmonize_country"):
        return team
    try:
        return str(shared.harmonize_country(team))
    except Exception:
        return team


def load_country_history(shared: Any | None, root: Path, notes: RuntimeNotes) -> Any | None:
    if shared is None or not hasattr(shared, "load_country_feature_history"):
        return None
    loader = shared.load_country_feature_history
    candidates = [
        ((), {}),
        (((root / "data"),), {}),
        (((root / "data" / "country_features.csv"),), {}),
        ((str(root / "data"),), {}),
    ]
    try:
        history = flexible_call(loader, candidates)
        notes.detail("Loaded country feature history through `shared.load_country_feature_history()`.")
        return history
    except Exception as exc:
        notes.warn(f"Could not load country feature history: {exc}")
        return None


def country_feature_map_for_year(
    shared: Any | None,
    history: Any,
    year: int,
    notes: RuntimeNotes,
) -> dict[str, dict[str, float]]:
    if shared is None or not hasattr(shared, "country_features_for_year") or history is None:
        return {}
    feature_year = int(year)
    try:
        features = shared.country_features_for_year(history, feature_year)
    except Exception as exc:
        notes.warn(f"Could not load country feature map for {feature_year}: {exc}")
        return {}
    notes.detail(f"Loaded country feature map for {feature_year} through `shared.country_features_for_year()`.")
    return dict(features or {})


def build_match_features_from_state(
    ctx: ExplanationContext,
    bundle: ModelBundle,
    country_features: Mapping[str, Mapping[str, float]],
    match_date: pd.Timestamp,
    *,
    state: Any | None = None,
    neutral: bool = True,
    is_home: bool = False,
) -> pd.DataFrame:
    if ctx.shared is None or not hasattr(ctx.shared, "compute_match_features"):
        raise RuntimeError("Project `shared.compute_match_features()` is required for explanations.")
    active_state = state if state is not None else bundle.state
    if active_state is None:
        raise RuntimeError("Prediction state is required for match explanations.")
    odds_row = None
    if bundle.odds is not None and hasattr(ctx.shared, "odds_features_for_match"):
        odds_row = ctx.shared.odds_features_for_match(bundle.odds, match_date, ctx.home, ctx.away)
    features = ctx.shared.compute_match_features(
        ctx.home,
        ctx.away,
        active_state,
        country_features,
        ctx.stage,
        match_date,
        neutral,
        is_home,
        odds_row=odds_row,
        squad_values=bundle.squad_values,
    )
    frame = coerce_feature_frame(features)
    if frame is None or frame.empty:
        raise RuntimeError("Project feature builder returned no features.")
    ctx.notes.detail("Built match features through strict `shared.compute_match_features()` call.")
    return frame


def coerce_feature_frame(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame().T
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    if isinstance(value, tuple):
        for item in value:
            frame = coerce_feature_frame(item)
            if frame is not None:
                return frame
    return None


def is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(number)


def discover_first(root: Path, patterns: Sequence[str], predicate: Callable[[Path], bool] | None = None) -> Path | None:
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and (predicate is None or predicate(path)):
                return path
    return None


def looks_like_xgb_model(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ("xgb", "xgboost", "model", "world", "wc", "2026"))


def load_model_from_file(path: Path, notes: RuntimeNotes) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".json", ".ubj"} and xgb is not None:
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        notes.detail(f"Loaded XGBoost model from `{path}`.")
        return model
    if suffix in {".pkl", ".joblib"} and joblib is not None:
        model = joblib.load(path)
        notes.detail(f"Loaded serialized model from `{path}`.")
        return model
    raise RuntimeError(f"Unsupported model file: {path}")


def bundle_from_training_parts(
    model: Any,
    state: Any,
    feature_names: Sequence[str],
    notes: RuntimeNotes,
    train_X: pd.DataFrame | None = None,
    train_y: Any | None = None,
    country_features: dict[str, dict[str, float]] | None = None,
    squad_values: dict[tuple[str, int], dict[str, float]] | None = None,
    odds: dict | None = None,
    poisson_model: Any | None = None,
    alpha: float = 1.0,
    wc_buckets: list[dict[str, Any]] | None = None,
) -> ModelBundle:
    class_labels = list(getattr(model, "classes_", OUTCOME_LABELS))
    if len(class_labels) == 0:
        class_labels = OUTCOME_LABELS
    return ModelBundle(
        model=model,
        state=state,
        feature_names=list(feature_names),
        class_labels=class_labels,
        train_X=train_X,
        train_y=train_y,
        country_features=country_features,
        squad_values=squad_values,
        odds=odds,
        poisson_model=poisson_model,
        alpha=alpha,
        wc_calibration_buckets=list(wc_buckets or []),
    )


def load_model_from_predict_module(module: Any | None, notes: RuntimeNotes) -> ModelBundle | None:
    if module is None:
        return None
    try:
        import shared as _shared
        results_path = _shared.DATA_DIR / "results.csv"
        results_df = pd.read_csv(results_path)
        country_history = _shared.load_country_feature_history()
    except Exception as exc:
        notes.warn(f"Could not load training inputs for `predict_2026`: {exc}")
        return None

    def prepare_state(state: Any) -> Any:
        if hasattr(module, "prepare_2026_state") and callable(getattr(module, "prepare_2026_state")):
            try:
                return module.prepare_2026_state(results_df, state)
            except Exception as exc:
                notes.warn(f"Could not apply 2026 prediction state updates: {exc}")
        return state

    if hasattr(module, "train_model_bundle") and callable(getattr(module, "train_model_bundle")):
        try:
            result = module.train_model_bundle(results_df, country_history)
            model = getattr(result, "model", None)
            state = getattr(result, "state", None)
            feature_names = getattr(result, "feature_names", None)
            if has_predict_proba(model) and state is not None and feature_names:
                notes.detail("Trained model bundle through `predict_2026.train_model_bundle()`.")
                return bundle_from_training_parts(
                    model=model,
                    state=prepare_state(state),
                    feature_names=list(feature_names),
                    notes=notes,
                    train_X=getattr(result, "train_X", None),
                    train_y=getattr(result, "train_y", None),
                    country_features=getattr(result, "country_features", None),
                    squad_values=getattr(result, "squad_values", None),
                    odds=getattr(result, "odds", None),
                    poisson_model=getattr(result, "poisson_model", None),
                    alpha=getattr(result, "alpha", 1.0),
                    wc_buckets=_shared.wc_calibration_buckets() if hasattr(_shared, "wc_calibration_buckets") else [],
                )
        except Exception as exc:
            notes.warn(f"Could not train via `predict_2026.train_model_bundle()`: {exc}")

    if hasattr(module, "train_model") and callable(getattr(module, "train_model")):
        try:
            result = module.train_model(results_df, country_history)
            if isinstance(result, tuple) and len(result) >= 3 and has_predict_proba(result[0]):
                model, state, feature_names = result[:3]
                train_X = result[3] if len(result) >= 4 and isinstance(result[3], pd.DataFrame) else None
                train_y = result[4] if len(result) >= 5 else None
                notes.detail("Trained model/state/schema through `predict_2026.train_model()`.")
                return bundle_from_training_parts(
                    model=model,
                    state=prepare_state(state),
                    feature_names=list(feature_names),
                    notes=notes,
                    train_X=train_X,
                    train_y=train_y,
                    squad_values=getattr(module, "_SQUAD_VALUES", None),
                    odds=getattr(module, "_ODDS", None),
                    poisson_model=getattr(module, "_POISSON", None),
                    alpha=getattr(module, "_ALPHA", 1.0),
                    wc_buckets=_shared.wc_calibration_buckets() if hasattr(_shared, "wc_calibration_buckets") else [],
                )
            if has_predict_proba(result):
                notes.warn("`predict_2026.train_model()` returned a bare model without state/schema; ignoring it.")
        except Exception as exc:
            notes.warn(f"Could not train via `predict_2026.train_model()`: {exc}")
    return None


def has_predict_proba(model: Any) -> bool:
    return model is not None and callable(getattr(model, "predict_proba", None))


def discover_match_data(root: Path, explicit: str | None, notes: RuntimeNotes, shared: Any | None = None) -> pd.DataFrame | None:
    default_path = root / "data" / "results.csv"
    path = Path(explicit).expanduser() if explicit else (default_path if default_path.exists() else discover_first(root, MATCH_DATA_PATTERNS))
    if path is None:
        notes.warn("No match history CSV was found; historical sections will be limited.")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        notes.warn(f"Could not read match history `{path}`: {exc}")
        return None
    if shared is not None and hasattr(shared, "harmonize_columns"):
        try:
            df = shared.harmonize_columns(df, ["home_team", "away_team", "country"])
        except Exception as exc:
            notes.warn(f"Could not harmonize match history country names: {exc}")
    notes.detail(f"Loaded match history from `{path}`.")
    return df


def infer_match_columns(df: pd.DataFrame | None) -> MatchColumns:
    cols = MatchColumns()
    if df is None:
        return cols
    names = {normalize_feature_name(col): col for col in df.columns}

    def pick(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in names:
                return names[candidate]
        for candidate in candidates:
            for norm, original in names.items():
                if candidate in norm:
                    return original
        return None

    cols.date = pick("date", "match_date", "game_date")
    cols.home_team = pick("home_team", "home", "team1", "team_a")
    cols.away_team = pick("away_team", "away", "team2", "team_b")
    cols.home_score = pick("home_score", "home_goals", "score_home", "goals_home")
    cols.away_score = pick("away_score", "away_goals", "score_away", "goals_away")
    cols.tournament = pick("tournament", "competition", "event")
    cols.stage = pick("stage", "round", "phase")
    cols.neutral = pick("neutral", "is_neutral")
    cols.home_elo = pick("home_elo", "elo_home", "home_rating")
    cols.away_elo = pick("away_elo", "elo_away", "away_rating")
    return cols


def get_feature_names(model: Any, train_X: pd.DataFrame | None, row: pd.DataFrame) -> list[str]:
    if train_X is not None:
        return list(train_X.columns)
    booster = getattr(model, "get_booster", lambda: None)()
    if booster is not None:
        names = getattr(booster, "feature_names", None)
        if names:
            return list(names)
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return list(row.columns)


def prepare_X(frame: pd.DataFrame, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    frame = frame.copy()
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if feature_names:
        missing = [name for name in feature_names if name not in frame.columns]
        if missing:
            preview = ", ".join(missing[:25])
            suffix = " ..." if len(missing) > 25 else ""
            raise RuntimeError(f"Feature schema mismatch. Missing trained features: {preview}{suffix}")
        frame = frame[list(feature_names)]
    keep_nan = set()
    try:
        import shared as _shared
        keep_nan |= set(getattr(_shared, "ODDS_FEATURE_COLUMNS", []))
        keep_nan |= set(getattr(_shared, "SQUAD_VALUE_FEATURE_COLUMNS", []))
    except Exception:
        pass
    non_nan = [col for col in frame.columns if col not in keep_nan]
    if non_nan:
        frame[non_nan] = frame[non_nan].fillna(0.0)
    return frame


def renormalize_knockout(probs: np.ndarray) -> np.ndarray:
    """Remove draw probability and renormalize for knockout matches."""
    if len(probs) < 3:
        return probs
    home, away = probs[0], probs[2]
    total = home + away
    if total > 0:
        return np.array([home / total, 0.0, away / total])
    return np.array([0.5, 0.0, 0.5])


def train_model_with_shared(
    shared: Any | None,
    matches: pd.DataFrame | None,
    notes: RuntimeNotes,
) -> tuple[Any | None, pd.DataFrame | None, pd.Series | None]:
    notes.warn("Direct training through `shared.fit_xgb_with_validation()` is disabled; use `predict_2026.train_model_bundle()`.")
    return None, None, None


def build_model_bundle(
    ctx: ExplanationContext,
    model_path: str | None,
) -> ModelBundle:
    model = None
    if model_path:
        model = load_model_from_file(Path(model_path).expanduser(), ctx.notes)
    bundle = load_model_from_predict_module(ctx.predict_2026, ctx.notes)
    if bundle is not None:
        if model is not None:
            bundle.model = model
            bundle.class_labels = list(getattr(model, "classes_", bundle.class_labels))
            ctx.notes.detail("Using explicit model file with state/schema from `predict_2026` training.")
        return bundle
    if model_path is None:
        candidate = discover_first(ctx.root, MODEL_PATTERNS, looks_like_xgb_model)
        if candidate is not None:
            try:
                model = load_model_from_file(candidate, ctx.notes)
            except Exception as exc:
                ctx.notes.warn(f"Could not load discovered model `{candidate}`: {exc}")
    if model is None:
        raise RuntimeError(
            "No usable model was found. Run this from the project root or pass `--model-path`."
        )
    raise RuntimeError("A model file alone is not enough to explain this project; prediction state and feature schema are required.")


def class_index(labels: Sequence[Any], kind: str, fallback: int) -> int:
    aliases = CLASS_ALIASES[kind]
    for idx, label in enumerate(labels):
        if label in aliases or str(label).lower() in {str(v).lower() for v in aliases}:
            return idx
    return min(fallback, max(len(labels) - 1, 0))


def predict_probabilities(
    bundle: ModelBundle,
    X: pd.DataFrame,
    home: str | None = None,
    away: str | None = None,
    *,
    neutral: bool = True,
    stage: int = 0,
) -> np.ndarray:
    probs = np.asarray(bundle.model.predict_proba(X))[0]
    if probs.ndim != 1:
        probs = probs.reshape(-1)
    if len(probs) == 2 and len(bundle.class_labels) == 2:
        labels = bundle.class_labels
        if class_index(labels, "home", 1) == 1:
            return np.array([probs[1], np.nan, probs[0]])
    if len(probs) == 3:
        h = class_index(bundle.class_labels, "home", 0)
        d = class_index(bundle.class_labels, "draw", 1)
        a = class_index(bundle.class_labels, "away", 2)
        out = np.array([probs[h], probs[d], probs[a]])
        if home is not None and away is not None and bundle.poisson_model is not None:
            try:
                import shared as _shared
                alpha = getattr(_shared, "KNOCKOUT_ALPHA", 0.50) if stage > 0 else bundle.alpha
                if alpha < 1.0:
                    p_dc = bundle.poisson_model.outcome_probs(home, away, neutral=neutral)
                    out = _shared.blend_probabilities(out, p_dc, alpha)
            except Exception:
                pass
        return out
    padded = np.full(3, np.nan)
    padded[: min(3, len(probs))] = probs[:3]
    return padded


def compute_shap_contrast(
    bundle: ModelBundle,
    X: pd.DataFrame,
    notes: RuntimeNotes,
) -> tuple[np.ndarray | None, float, Any | None]:
    if shap is None:
        notes.warn(f"SHAP is unavailable: {SHAP_IMPORT_ERROR}")
        return None, 0.0, None
    try:
        explainer = shap.TreeExplainer(bundle.model)
        raw = explainer.shap_values(X)
    except Exception as exc:
        notes.warn(f"Could not compute SHAP values: {exc}")
        return None, 0.0, None
    h = class_index(bundle.class_labels, "home", 0)
    a = class_index(bundle.class_labels, "away", 2)
    base = getattr(explainer, "expected_value", 0.0)
    values: np.ndarray
    if isinstance(raw, list):
        if len(raw) >= 3:
            values = np.asarray(raw[h][0]) - np.asarray(raw[a][0])
            base_value = np.asarray(base)[h] - np.asarray(base)[a] if np.ndim(base) else 0.0
        else:
            values = np.asarray(raw[-1][0])
            base_value = float(np.asarray(base).reshape(-1)[-1]) if np.size(base) else 0.0
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            values = arr[0, :, h] - arr[0, :, a]
            base_arr = np.asarray(base).reshape(-1)
            base_value = base_arr[h] - base_arr[a] if len(base_arr) > max(h, a) else 0.0
        elif arr.ndim == 3 and arr.shape[1] >= 3:
            values = arr[0, h, :] - arr[0, a, :]
            base_arr = np.asarray(base).reshape(-1)
            base_value = base_arr[h] - base_arr[a] if len(base_arr) > max(h, a) else 0.0
        elif arr.ndim == 2:
            values = arr[0]
            base_value = float(np.asarray(base).reshape(-1)[0]) if np.size(base) else 0.0
        else:
            values = arr.reshape(-1)
            base_value = 0.0
    try:
        margin = np.asarray(bundle.model.predict(X, output_margin=True))
        expected_margin_diff = float(margin[0, h] - margin[0, a]) if margin.ndim == 2 and margin.shape[1] > max(h, a) else None
        actual_margin_diff = float(base_value + values.sum())
        if expected_margin_diff is not None and not np.isclose(expected_margin_diff, actual_margin_diff, atol=1e-3):
            notes.warn("SHAP additivity check did not match the model margin exactly; interpret local contributions cautiously.")
    except Exception:
        pass
    notes.detail(
        "Computed SHAP contrast on the model's raw multiclass margin: home-win margin contribution minus away-win margin contribution."
    )
    return values.astype(float), float(base_value), explainer


def feature_plain_english(feature: str, value: Any, home: str, away: str) -> str:
    name = normalize_feature_name(feature)
    side = None
    if name.startswith("home_"):
        side = home
    elif name.startswith("away_"):
        side = away
    if "elo" in name:
        return "Elo summarizes team strength from historical results; larger values usually indicate a stronger side."
    if "rank" in name or "fifa" in name:
        return "Ranking-style features capture broad team quality compared with the international field."
    if "form" in name or "recent" in name or "last" in name:
        return "Recent-form features describe how the team has been performing in its latest matches."
    if "goal" in name or "scored" in name:
        return "Goal features capture attacking output and finishing strength."
    if "conced" in name or "clean" in name or "defen" in name:
        return "Defensive features capture how often a team prevents chances and goals."
    if "h2h" in name or "head_to_head" in name:
        return f"Head-to-head features measure the historical matchup between {home} and {away}."
    if "world_cup" in name or "wc_" in name or "title" in name or "particip" in name:
        return "World Cup experience features proxy for tournament pedigree and exposure to high-pressure matches."
    if "gdp" in name or "population" in name:
        return "Economic and demographic features are broad background proxies for football infrastructure and talent pool."
    if "host" in name or "home_adv" in name or "venue" in name or "neutral" in name:
        return "Contextual venue features describe whether either side receives location or hosting advantage."
    if "stage" in name:
        return "Tournament stage features let the model adjust expectations for group and knockout match dynamics."
    if side:
        return f"This is a team-specific input for {side}; the model compares it with the opponent's context."
    return "This engineered input is part of the model's matchup profile for the two teams."


def top_shap_rows(
    feature_names: Sequence[str],
    values: np.ndarray | None,
    X: pd.DataFrame,
    home: str,
    away: str,
    limit: int = 15,
) -> list[dict[str, Any]]:
    if values is None:
        return []
    rows = []
    for idx, value in enumerate(values[: len(feature_names)]):
        feature = feature_names[idx]
        feature_value = X.iloc[0][feature] if feature in X.columns else np.nan
        rows.append(
            {
                "feature": feature,
                "value": feature_value,
                "shap": float(value),
                "direction": home if value >= 0 else away,
                "explanation": feature_plain_english(feature, feature_value, home, away),
            }
        )
    rows.sort(key=lambda item: abs(item["shap"]), reverse=True)
    return rows[:limit]


def save_shap_plots(
    ctx: ExplanationContext,
    X: pd.DataFrame,
    shap_values: np.ndarray | None,
    base_value: float,
) -> tuple[Path | None, Path | None]:
    if shap is None or shap_values is None:
        return None, None
    stem = f"{slugify(ctx.home)}_vs_{slugify(ctx.away)}"
    waterfall_path = ctx.output_dir / f"{stem}_shap.png"
    force_path = ctx.output_dir / f"{stem}_force.html"
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=X.iloc[0].values,
            feature_names=list(X.columns),
        )
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        ctx.notes.warn(f"Could not save SHAP waterfall plot: {exc}")
        waterfall_path = None
    try:
        force = shap.force_plot(base_value, shap_values, X.iloc[0], feature_names=list(X.columns), matplotlib=False)
        shap.save_html(str(force_path), force)
    except Exception as exc:
        ctx.notes.warn(f"Could not save SHAP force plot: {exc}")
        force_path = None
    return waterfall_path, force_path


def team_matches(
    df: pd.DataFrame | None,
    cols: MatchColumns,
    team: str,
    match_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if df is None or not cols.home_team or not cols.away_team:
        return pd.DataFrame()
    mask = df[cols.home_team].astype(str).eq(team) | df[cols.away_team].astype(str).eq(team)
    out = df.loc[mask].copy()
    if cols.date:
        out[cols.date] = pd.to_datetime(out[cols.date], errors="coerce")
        if match_date is not None:
            out = out[out[cols.date] < match_date]
        out = out.sort_values(cols.date)
    if cols.home_score and cols.away_score:
        out = out[out[cols.home_score].notna() & out[cols.away_score].notna()]
    return out


def result_for_row(row: pd.Series, cols: MatchColumns, team: str) -> dict[str, Any]:
    is_home = str(row.get(cols.home_team)) == team if cols.home_team else False
    opponent = row.get(cols.away_team if is_home else cols.home_team, "Unknown")
    gf = row.get(cols.home_score if is_home else cols.away_score, np.nan)
    ga = row.get(cols.away_score if is_home else cols.home_score, np.nan)
    outcome = "?"
    if is_number(gf) and is_number(ga):
        gf_f, ga_f = float(gf), float(ga)
        outcome = "W" if gf_f > ga_f else "D" if gf_f == ga_f else "L"
    date = row.get(cols.date, "")
    if pd.notna(date) and not isinstance(date, str):
        date = pd.to_datetime(date).date().isoformat()
    return {"date": date, "opponent": opponent, "gf": gf, "ga": ga, "outcome": outcome}


def recent_form(df: pd.DataFrame, cols: MatchColumns, team: str, n: int = 10) -> dict[str, Any]:
    recent = df.tail(n)
    results = [result_for_row(row, cols, team) for _, row in recent.iterrows()]
    played = [r for r in results if is_number(r["gf"]) and is_number(r["ga"])]
    if not played:
        return {"matches": results, "win_rate": None, "gf": None, "ga": None, "clean_sheets": None}
    wins = sum(1 for r in played if r["outcome"] == "W")
    clean = sum(1 for r in played if float(r["ga"]) == 0)
    return {
        "matches": results,
        "win_rate": wins / len(played),
        "gf": sum(float(r["gf"]) for r in played) / len(played),
        "ga": sum(float(r["ga"]) for r in played) / len(played),
        "clean_sheets": clean,
    }


def elo_trajectory(df: pd.DataFrame, cols: MatchColumns, team: str, n: int = 20) -> list[tuple[str, float]]:
    if df.empty or not cols.home_elo or not cols.away_elo:
        return []
    points: list[tuple[str, float]] = []
    for _, row in df.tail(n).iterrows():
        is_home = str(row.get(cols.home_team)) == team if cols.home_team else False
        col = cols.home_elo if is_home else cols.away_elo
        if is_number(row.get(col)):
            date = row.get(cols.date, "")
            if pd.notna(date) and not isinstance(date, str):
                date = pd.to_datetime(date).date().isoformat()
            points.append((str(date), float(row.get(col))))
    return points


def h2h_record(
    df: pd.DataFrame | None,
    cols: MatchColumns,
    home: str,
    away: str,
    match_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    empty = {"all_time": None, "last_10_years": None, "matches": pd.DataFrame()}
    if df is None or not cols.home_team or not cols.away_team:
        return empty
    mask = (
        (df[cols.home_team].astype(str).eq(home) & df[cols.away_team].astype(str).eq(away))
        | (df[cols.home_team].astype(str).eq(away) & df[cols.away_team].astype(str).eq(home))
    )
    matches = df.loc[mask].copy()
    if matches.empty or not cols.home_score or not cols.away_score:
        return {**empty, "matches": matches}
    if cols.date:
        matches[cols.date] = pd.to_datetime(matches[cols.date], errors="coerce")
        if match_date is not None:
            matches = matches[matches[cols.date] < match_date]
        matches = matches.sort_values(cols.date)
    if matches.empty:
        return {**empty, "matches": matches}

    def summarize(part: pd.DataFrame) -> dict[str, int]:
        home_wins = away_wins = draws = 0
        for _, row in part.iterrows():
            hs, as_ = row.get(cols.home_score), row.get(cols.away_score)
            if not (is_number(hs) and is_number(as_)):
                continue
            team_home = str(row.get(cols.home_team)) == home
            h_goals = float(hs) if team_home else float(as_)
            a_goals = float(as_) if team_home else float(hs)
            if h_goals > a_goals:
                home_wins += 1
            elif h_goals < a_goals:
                away_wins += 1
            else:
                draws += 1
        return {"home_wins": home_wins, "draws": draws, "away_wins": away_wins, "total": home_wins + draws + away_wins}

    last_10 = matches
    if cols.date and not matches[cols.date].isna().all():
        cutoff = matches[cols.date].max() - pd.DateOffset(years=10)
        last_10 = matches[matches[cols.date] >= cutoff]
    return {"all_time": summarize(matches), "last_10_years": summarize(last_10), "matches": matches}


def world_cup_history(team: str, through_year: int = 2026) -> dict[str, Any]:
    try:
        from collect_data import WC_PARTICIPANTS
        from shared import GROUP_2026_TEAMS, WC_WINNERS, harmonize_country
    except Exception:
        return {"available": False}
    canonical = harmonize_country(team)
    years = [
        int(year)
        for year, participants in WC_PARTICIPANTS.items()
        if int(year) <= through_year and canonical in {harmonize_country(t) for t in participants}
    ]
    if through_year >= 2026:
        wc26_teams = {harmonize_country(t) for teams in GROUP_2026_TEAMS.values() for t in teams}
        if canonical in wc26_teams and 2026 not in years:
            years.append(2026)
    years = sorted(years)
    titles = sum(
        1
        for year, winner in WC_WINNERS.items()
        if int(year) <= through_year and harmonize_country(winner) == canonical
    )
    return {
        "available": True,
        "participations": len(years),
        "years": years[-8:],
        "titles": titles,
        "recent": [],
    }


def factor_decomposition(
    top_rows: list[dict[str, Any]],
    X: pd.DataFrame,
    home: str,
    away: str,
) -> list[dict[str, Any]]:
    out = []
    for factor, keywords in FACTOR_KEYWORDS.items():
        related = [
            row
            for row in top_rows
            if any(keyword in normalize_feature_name(row["feature"]) for keyword in keywords)
        ]
        raw_features = [
            col for col in X.columns if any(keyword in normalize_feature_name(col) for keyword in keywords)
        ][:8]
        shap_sum = float(sum(row["shap"] for row in related)) if related else 0.0
        out.append(
            {
                "factor": factor,
                "shap_sum": shap_sum,
                "favors": home if shap_sum > 0 else away if shap_sum < 0 else "Neither",
                "raw_features": [(feature, X.iloc[0].get(feature)) for feature in raw_features],
                "drivers": related[:5],
            }
        )
    return out


def elo_baseline(X: pd.DataFrame, home: str, away: str) -> tuple[np.ndarray | None, str]:
    cols = {normalize_feature_name(c): c for c in X.columns}
    diff = None
    for name, col in cols.items():
        if "elo" in name and "diff" in name:
            diff = float(X.iloc[0][col])
            break
    if diff is None:
        home_elo = next((col for name, col in cols.items() if "home" in name and "elo" in name), None)
        away_elo = next((col for name, col in cols.items() if "away" in name and "elo" in name), None)
        if home_elo and away_elo:
            diff = float(X.iloc[0][home_elo]) - float(X.iloc[0][away_elo])
    if diff is None or not np.isfinite(diff):
        return None, "No Elo feature was available."
    home_no_draw = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    draw = max(0.15, min(0.30, 0.27 - abs(diff) / 3000.0))
    home_prob = (1.0 - draw) * home_no_draw
    away_prob = (1.0 - draw) * (1.0 - home_no_draw)
    return np.array([home_prob, draw, away_prob]), f"Elo difference: {home} minus {away} = {diff:.0f}."


def confidence_summary(probs: np.ndarray) -> dict[str, Any]:
    valid = probs[np.isfinite(probs)]
    if len(valid) == 0:
        return {"confidence": None, "margin": None, "entropy": None}
    sorted_probs = np.sort(valid)[::-1]
    entropy = -sum(float(p) * math.log(float(p) + 1e-12) for p in valid) / math.log(len(valid))
    return {
        "confidence": float(sorted_probs[0]),
        "margin": float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else None,
        "entropy": float(entropy),
    }


def historical_accuracy_bucket(
    bundle: ModelBundle,
    X_current: pd.DataFrame,
    probs: np.ndarray,
) -> str:
    if bundle.train_X is None or bundle.train_y is None or not has_predict_proba(bundle.model):
        return "Historical calibration bucket unavailable because training features/labels were not returned."
    try:
        train_X = prepare_X(bundle.train_X, bundle.feature_names)
        train_probs_raw = np.asarray(bundle.model.predict_proba(train_X))
    except Exception:
        return "Historical calibration bucket unavailable because training predictions could not be recomputed."
    train_probs = []
    for row in train_probs_raw:
        if len(row) == 3:
            train_probs.append(max(row))
        else:
            train_probs.append(float(np.max(row)))
    current_conf = float(np.nanmax(probs))
    lo, hi = current_conf - 0.05, current_conf + 0.05
    idx = [i for i, p in enumerate(train_probs) if lo <= p <= hi]
    if not idx:
        return f"No historical training predictions were within +/-5 percentage points of {fmt_pct(current_conf)} confidence."
    y = pd.Series(bundle.train_y).reset_index(drop=True)
    correct = 0
    used = 0
    for i in idx:
        pred = int(np.argmax(train_probs_raw[i]))
        if i < len(y):
            used += 1
            correct += int(str(y.iloc[i]) == str(bundle.class_labels[pred]) or y.iloc[i] == pred)
    if not used:
        return "Calibration bucket found similar probabilities, but labels could not be aligned."
    return f"Among {used} training cases with similar confidence, the top prediction was correct {fmt_pct(correct / used)} of the time."


def brier_note(bundle: ModelBundle) -> str:
    if bundle.train_X is None or bundle.train_y is None:
        return "Brier decomposition unavailable because training features/labels were not returned."
    try:
        probs = np.asarray(bundle.model.predict_proba(prepare_X(bundle.train_X, bundle.feature_names)))
        labels = list(bundle.class_labels)
        y = pd.Series(bundle.train_y).reset_index(drop=True)
        onehot = np.zeros_like(probs, dtype=float)
        for i, label in enumerate(y):
            if label in labels:
                onehot[i, labels.index(label)] = 1.0
            elif is_number(label) and int(label) < probs.shape[1]:
                onehot[i, int(label)] = 1.0
        brier = np.mean(np.sum((probs - onehot) ** 2, axis=1))
    except Exception:
        return "Brier decomposition unavailable because labels/probabilities could not be aligned."
    return f"Multiclass Brier score on available training data: {brier:.4f}. Full reliability/resolution decomposition needs held-out bins."


def reverse_h2h_state(state: Any, home: str, away: str) -> None:
    key = tuple(sorted([home, away]))
    for team in (home, away):
        record = state[team]["h2h"].get(key)
        if not record:
            continue
        record["wins"], record["losses"] = record["losses"], record["wins"]
        record["gf"], record["ga"] = record["ga"], record["gf"]


def counterfactuals(
    ctx: ExplanationContext,
    bundle: ModelBundle,
    X: pd.DataFrame,
    probs: np.ndarray,
    country_features: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    favorite_idx = int(np.nanargmax(probs))
    home_is_underdog = favorite_idx == 2
    if favorite_idx == 0:
        underdog = ctx.away
    elif favorite_idx == 2:
        underdog = ctx.home
    else:
        underdog = ctx.home if float(X.iloc[0].get("elo_diff", 0.0)) < 0 else ctx.away
    scenarios = [
        ("underdog_elo_plus_100", "Underdog Elo +100"),
        ("h2h_reversed", "Head-to-head record reversed"),
        ("home_venue_advantage", f"{ctx.home} has venue advantage"),
    ]
    out = []
    for key, label in scenarios:
        try:
            cf_state = copy.deepcopy(bundle.state)
            neutral = True
            is_home = False
            changed = ""
            if key == "underdog_elo_plus_100":
                cf_state[underdog]["elo"] += 100
                changed = f"{underdog} rolling Elo +100; dependent Elo features recomputed"
            elif key == "h2h_reversed":
                reverse_h2h_state(cf_state, ctx.home, ctx.away)
                changed = "H2H wins/losses and goals reversed in rolling state; H2H features recomputed"
            elif key == "home_venue_advantage":
                neutral = False
                is_home = True
                changed = f"{ctx.home} marked as non-neutral home side; venue features recomputed"
            frame = build_match_features_from_state(
                ctx,
                bundle,
                country_features,
                ctx.match_date,
                state=cf_state,
                neutral=neutral,
                is_home=is_home,
            )
            X_cf = prepare_X(frame, bundle.feature_names)
            cf_probs = predict_probabilities(
                bundle, X_cf, ctx.home, ctx.away, neutral=neutral, stage=ctx.stage,
            )
            if ctx.stage > 0:
                cf_probs = renormalize_knockout(cf_probs)
        except Exception as exc:
            cf_probs = np.array([np.nan, np.nan, np.nan])
            changed = f"Counterfactual unavailable: {exc}"
        out.append({"label": label, "changed": changed, "probs": cf_probs})
    return out


def single_feature_flip_search(
    bundle: ModelBundle,
    X: pd.DataFrame,
    probs: np.ndarray,
    shap_rows: list[dict[str, Any]],
) -> str:
    target = 2 if int(np.nanargmax(probs)) == 0 else 0
    base_target = float(probs[target]) if np.isfinite(probs[target]) else 0.0
    best = None
    train_std = bundle.train_X.std(numeric_only=True) if bundle.train_X is not None else pd.Series(dtype=float)
    for row in shap_rows[:15]:
        feature = row["feature"]
        if feature not in X or not is_number(X.iloc[0][feature]):
            continue
        current = float(X.iloc[0][feature])
        scale = float(train_std.get(feature, np.nan))
        if not np.isfinite(scale) or scale == 0:
            scale = max(1.0, abs(current) * 0.25)
        for sign in (-1, 1):
            X_try = X.copy()
            X_try[feature] = current + sign * scale
            try:
                p_try = predict_probabilities(bundle, X_try)
            except Exception:
                continue
            gain = float(p_try[target] - base_target) if np.isfinite(p_try[target]) else -np.inf
            if best is None or gain > best["gain"]:
                best = {"feature": feature, "new": current + sign * scale, "gain": gain, "probs": p_try}
    if best is None:
        return "No numeric high-impact feature could be perturbed for flip analysis."
    return (
        f"Changing `{best['feature']}` to {fmt_num(best['new'])} moves the opposing win probability "
        f"by {fmt_pct(best['gain'])}; new probabilities are {format_probs(best['probs'])}."
    )


def causal_insights(root: Path) -> list[str]:
    paths: list[Path] = []
    for pattern in (
        "output/**/*sota*",
        "output/**/*causal*",
        "output/**/*lowco*",
        "artifacts/**/*sota*",
        "artifacts/**/*causal*",
        "artifacts/**/*lowco*",
        "*sota*.md",
        "*causal*.md",
        "*lowco*.md",
    ):
        paths.extend([p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in {".md", ".txt", ".json", ".csv"}])
    snippets: list[str] = []
    for path in sorted(set(paths))[:8]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = [
            line.strip()
            for line in text.splitlines()
            if any(k in line.lower() for k in ("host", "did", "world cup experience", "lowco", "causal", "importance", "predictor"))
        ]
        if lines:
            try:
                display_path = path.relative_to(root)
            except ValueError:
                display_path = path
            snippets.append(f"`{display_path}`: " + " ".join(lines[:3])[:500])
    if snippets:
        return snippets
    return [
        "No SOTA analysis output files were found. When `sota_analysis.py` outputs are present, this section will surface hosting DiD estimates, World Cup experience effects, and LOWCO-validated predictors."
    ]


def format_probs(probs: np.ndarray) -> str:
    return f"Home {fmt_pct(probs[0])}, Draw {fmt_pct(probs[1])}, Away {fmt_pct(probs[2])}"


def report_lines(
    ctx: ExplanationContext,
    probs: np.ndarray,
    raw_probs: np.ndarray,
    conditional_probs: np.ndarray | None,
    calibration_note: str,
    odds_missing: bool,
    shap_rows: list[dict[str, Any]],
    factors: list[dict[str, Any]],
    waterfall_path: Path | None,
    force_path: Path | None,
    bundle: ModelBundle,
    X: pd.DataFrame,
    country_features: Mapping[str, Mapping[str, float]],
) -> list[str]:
    cols = infer_match_columns(ctx.matches)
    home_df = team_matches(ctx.matches, cols, ctx.home, ctx.match_date)
    away_df = team_matches(ctx.matches, cols, ctx.away, ctx.match_date)
    home_form = recent_form(home_df, cols, ctx.home)
    away_form = recent_form(away_df, cols, ctx.away)
    h2h = h2h_record(ctx.matches, cols, ctx.home, ctx.away, ctx.match_date)
    elo_probs, elo_msg = elo_baseline(X, ctx.home, ctx.away)
    if ctx.stage > 0 and elo_probs is not None:
        elo_probs = renormalize_knockout(elo_probs)
    conf = confidence_summary(probs)
    cfs = counterfactuals(ctx, bundle, X, probs, country_features)
    flip = single_feature_flip_search(bundle, X, probs, shap_rows)
    lines: list[str] = []
    lines.append(f"# Match Explanation: {ctx.home} vs {ctx.away}")
    lines.append("")
    lines.append(f"Stage code: `{ctx.stage}` | Year: `{ctx.year}` | Match date: `{ctx.match_date.date().isoformat()}`")
    lines.append("")
    lines.append("## Prediction")
    lines.append("")
    lines.append(f"- Final reported probabilities: **{format_probs(probs)}**")
    lines.append(f"- Raw blended 3-way probabilities: **{format_probs(raw_probs)}**")
    if ctx.stage > 0:
        if conditional_probs is not None:
            lines.append(f"- Knockout no-draw probabilities before WC calibration: **{format_probs(conditional_probs)}**")
        lines.append(f"- *Knockout stage: draw excluded for advancement-style reporting, then WC-specific calibration is applied to the favorite confidence.*")
        if calibration_note:
            lines.append(f"- WC calibration: {calibration_note}")
    if odds_missing:
        lines.append("- Warning: bookmaker odds are unavailable for this fixture, so the prediction is less market-informed.")
    lines.append(f"- Most likely outcome: **{OUTCOME_LABELS[int(np.nanargmax(probs))]}**")
    lines.append(f"- Confidence: **{fmt_pct(conf['confidence'])}**; top-two margin: **{fmt_pct(conf['margin'])}**; entropy: **{fmt_num(conf['entropy'])}**")
    if elo_probs is not None:
        lines.append(f"- Elo-only baseline: **{format_probs(elo_probs)}** ({elo_msg})")
    else:
        lines.append(f"- Elo-only baseline: {elo_msg}")
    lines.append(f"- Similar-confidence history: {historical_accuracy_bucket(bundle, X, probs)}")
    lines.append(f"- Brier/calibration note: {brier_note(bundle)}")
    lines.append("")
    lines.append("## Top SHAP Drivers")
    lines.append("")
    lines.append(
        "SHAP values below are raw multiclass margin contrasts: home-win margin contribution minus away-win margin contribution. Signs show local direction; magnitudes are not probability points."
    )
    lines.append("")
    if shap_rows:
        lines.append("| Feature | Value | Contribution | Favors | Plain-English meaning |")
        lines.append("|---|---:|---:|---|---|")
        for row in shap_rows:
            lines.append(
                f"| `{row['feature']}` | {fmt_num(row['value'])} | {row['shap']:+.4f} | {row['direction']} | {row['explanation']} |"
            )
    else:
        lines.append("SHAP values were not available for this run.")
    if waterfall_path:
        lines.append(f"\nWaterfall plot: `{waterfall_path}`")
    if force_path:
        lines.append(f"Force plot: `{force_path}`")
    lines.append("")
    lines.append("## Historical Context")
    lines.append("")
    append_team_history(lines, ctx.home, home_form, elo_trajectory(home_df, cols, ctx.home), world_cup_history(ctx.home, ctx.year))
    append_team_history(lines, ctx.away, away_form, elo_trajectory(away_df, cols, ctx.away), world_cup_history(ctx.away, ctx.year))
    lines.append("### Head-To-Head")
    lines.append("")
    lines.append(format_h2h(h2h, ctx.home, ctx.away))
    lines.append("")
    lines.append("## Match Factor Decomposition")
    lines.append("")
    for factor in factors:
        lines.append(f"### {factor['factor']}")
        lines.append("")
        lines.append(f"- Net SHAP contribution: **{factor['shap_sum']:+.4f}**, favors **{factor['favors']}**.")
        if factor["raw_features"]:
            raw = ", ".join(f"`{name}`={fmt_num(value)}" for name, value in factor["raw_features"][:6])
            lines.append(f"- Raw feature values: {raw}")
        else:
            lines.append("- Raw feature values: no related engineered features were present.")
        if factor["drivers"]:
            drivers = ", ".join(f"`{item['feature']}` ({item['shap']:+.3f})" for item in factor["drivers"])
            lines.append(f"- Main local drivers: {drivers}")
        lines.append("")
    lines.append("## Counterfactuals")
    lines.append("")
    for cf in cfs:
        lines.append(f"- {cf['label']}: {format_probs(cf['probs'])}. Changed: {cf['changed']}.")
    lines.append(f"- Single-feature sensitivity search: {flip}")
    lines.append("")
    lines.append("## Causal And SOTA Context")
    lines.append("")
    for item in causal_insights(ctx.root):
        lines.append(f"- {item}")
    if ctx.notes.warnings:
        lines.append("")
        lines.append("## Data Availability Notes")
        lines.append("")
        for warning in ctx.notes.warnings:
            lines.append(f"- {warning}")
    if ctx.notes.details:
        lines.append("")
        lines.append("## Runtime Trace")
        lines.append("")
        for detail in ctx.notes.details:
            lines.append(f"- {detail}")
    return lines


def append_team_history(
    lines: list[str],
    team: str,
    form: Mapping[str, Any],
    elo_points: list[tuple[str, float]],
    wc: Mapping[str, Any],
) -> None:
    lines.append(f"### {team}")
    lines.append("")
    lines.append(
        f"- Recent form: win rate {fmt_pct(form.get('win_rate'))}, "
        f"goals for {fmt_num(form.get('gf'))}/match, goals against {fmt_num(form.get('ga'))}/match, "
        f"clean sheets {form.get('clean_sheets', 'n/a')}."
    )
    matches = form.get("matches", [])
    if matches:
        compact = []
        for match in matches[-10:]:
            compact.append(
                f"{match.get('date', '')}: {match.get('outcome', '?')} {fmt_num(match.get('gf'))}-{fmt_num(match.get('ga'))} vs {match.get('opponent')}"
            )
        lines.append("- Last 10 results: " + "; ".join(compact))
    else:
        lines.append("- Last 10 results: unavailable.")
    if elo_points:
        start = elo_points[0][1]
        end = elo_points[-1][1]
        lines.append(f"- Elo trajectory: {fmt_num(start)} -> {fmt_num(end)} over the last {len(elo_points)} recorded matches.")
    else:
        lines.append("- Elo trajectory: unavailable.")
    if wc.get("available"):
        lines.append(
            f"- World Cup history: participations {wc.get('participations', 'n/a')}, "
            f"titles {wc.get('titles', 'n/a')}, recent years {wc.get('years', [])}."
        )
    else:
        lines.append("- World Cup history: unavailable from local match data.")
    lines.append("")


def format_h2h(h2h: Mapping[str, Any], home: str, away: str) -> str:
    def one(record: Mapping[str, Any] | None) -> str:
        if not record or record.get("total", 0) == 0:
            return "unavailable"
        return (
            f"{home} {record['home_wins']} wins, {record['draws']} draws, "
            f"{away} {record['away_wins']} wins ({record['total']} matches)"
        )

    return f"- All-time: {one(h2h.get('all_time'))}\n- Last 10 years: {one(h2h.get('last_10_years'))}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home", nargs="?", help="Home/first team, e.g. Brazil")
    parser.add_argument("away", nargs="?", help="Away/second team, e.g. Japan")
    parser.add_argument("--teams", help='Comma-separated teams, e.g. "Brazil,Japan"')
    parser.add_argument("--stage", type=int, default=0, help="Tournament stage code: 0 group, 1 R16, etc.")
    parser.add_argument("--year", type=int, default=2026, help="Feature year to use for country features.")
    parser.add_argument("--match-date", default="2026-06-29", help="Match date used for rest-days and historical cutoffs.")
    parser.add_argument("--model-path", help="Optional path to a saved XGBoost/joblib model.")
    parser.add_argument("--matches-data", help="Optional path to match history CSV.")
    parser.add_argument("--output-dir", default="output/explain", help="Directory for plots and Markdown report.")
    parser.add_argument("--no-interactive", action="store_true", help="Fail instead of prompting when teams are omitted.")
    return parser.parse_args(argv)


def resolve_teams(args: argparse.Namespace) -> tuple[str, str]:
    if args.teams:
        parts = [part.strip() for part in args.teams.split(",") if part.strip()]
        if len(parts) != 2:
            raise SystemExit("--teams must contain exactly two comma-separated team names.")
        return parts[0], parts[1]
    if args.home and args.away:
        return args.home, args.away
    if args.no_interactive or not sys.stdin.isatty():
        raise SystemExit("Provide teams as `python explain_match.py Brazil Japan` or `--teams Brazil,Japan`.")
    home = input("Home/first team: ").strip()
    away = input("Away/second team: ").strip()
    if not home or not away:
        raise SystemExit("Both teams are required.")
    return home, away


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    notes = RuntimeNotes()
    root = Path.cwd()
    match_date = pd.Timestamp(args.match_date)
    shared = import_optional_module("shared", notes)
    predict_2026 = import_optional_module("predict_2026", notes)
    require_project_functions(shared, notes)
    home_raw, away_raw = resolve_teams(args)
    home = call_harmonize(shared, home_raw)
    away = call_harmonize(shared, away_raw)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    ctx = ExplanationContext(
        home=home,
        away=away,
        year=args.year,
        stage=args.stage,
        match_date=match_date,
        root=root,
        output_dir=output_dir,
        notes=notes,
        shared=shared,
        predict_2026=predict_2026,
    )
    ctx.country_history = load_country_history(shared, root, notes)
    ctx.matches = discover_match_data(root, args.matches_data, notes, shared)
    bundle = build_model_bundle(ctx, args.model_path)
    country_features = country_feature_map_for_year(shared, ctx.country_history, args.year, notes)
    if not country_features and bundle.country_features:
        country_features = bundle.country_features
    match_features = build_match_features_from_state(ctx, bundle, country_features, match_date)
    X = prepare_X(match_features, bundle.feature_names)
    raw_probs = predict_probabilities(bundle, X, home, away, neutral=True, stage=ctx.stage)
    probs = raw_probs.copy()
    is_knockout = ctx.stage > 0
    conditional_probs = None
    calibration_note = ""
    if is_knockout:
        conditional_probs = renormalize_knockout(raw_probs)
        probs = conditional_probs.copy()
        if shared is not None and hasattr(shared, "apply_wc_knockout_calibration"):
            probs[0], probs[2], calibration_note = shared.apply_wc_knockout_calibration(
                probs[0], probs[2], bundle.wc_calibration_buckets,
            )
        probs[1] = 0.0
    odds_missing = False
    if shared is not None and hasattr(shared, "ODDS_FEATURE_COLUMNS"):
        odds_cols = getattr(shared, "ODDS_FEATURE_COLUMNS", [])
        odds_missing = any(
            col in X.columns and not np.isfinite(float(X.iloc[0][col]))
            for col in odds_cols
        )
    shap_values, base_value, _explainer = compute_shap_contrast(bundle, X, notes)
    shap_rows = top_shap_rows(bundle.feature_names, shap_values, X, home, away)
    waterfall_path, force_path = save_shap_plots(ctx, X, shap_values, base_value)
    factors = factor_decomposition(shap_rows, X, home, away)
    lines = report_lines(
        ctx, probs, raw_probs, conditional_probs, calibration_note, odds_missing,
        shap_rows, factors, waterfall_path, force_path, bundle, X, country_features,
    )
    report = "\n".join(lines) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{slugify(home)}_vs_{slugify(away)}_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved Markdown report to {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
