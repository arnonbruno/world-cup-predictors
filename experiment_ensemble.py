#!/usr/bin/env python3
"""Approach 4: stacking and dynamic model blending."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment_common import (
    ModelBundle,
    align_prediction_frame,
    build_training_data,
    chronological_holdout_indices,
    fit_dixon_and_alpha,
    load_inputs,
    make_bundle,
    normalize_probs,
    predict_default,
    run_walk_forward_backtest,
    sample_weight_array,
)
from shared import blend_probabilities, compute_match_features, fit_dixon_coles, odds_features_for_match


def _fit_xgb(training, weights, train_idx, val_idx):
    import xgboost as xgb

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
    fit_kwargs = {
        "eval_set": [(training.X.iloc[val_idx], training.y[val_idx])],
        "verbose": False,
        "sample_weight": weights[train_idx],
    }
    try:
        model.fit(training.X.iloc[train_idx], training.y[train_idx], early_stopping_rounds=30, **fit_kwargs)
    except TypeError:
        model.fit(training.X.iloc[train_idx], training.y[train_idx], **fit_kwargs)
    return model


def _base_meta_features(models, poisson_model, training, indices):
    chunks = []
    for model in models:
        chunks.append(np.asarray(model.predict_proba(training.X.iloc[indices]), dtype=float))
    chunks.append(
        np.asarray(
            [
                poisson_model.outcome_probs(training.match_meta[i][0], training.match_meta[i][1], neutral=training.match_meta[i][2])
                for i in indices
            ],
            dtype=float,
        )
    )
    return np.hstack(chunks)


def train_stacking_bundle():
    results_df, country_history, odds, squad_values = load_inputs()
    training = build_training_data(results_df, country_history, odds, squad_values)
    weights = sample_weight_array(training)
    train_idx, val_idx = chronological_holdout_indices(training.dates)

    xgb_model = _fit_xgb(training, weights, train_idx, val_idx)
    rf_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42,
                    class_weight={0: 1.0, 1: 1.6, 2: 1.0},
                ),
            ),
        ]
    )
    logit_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    max_iter=1000,
                    
                    class_weight={0: 1.0, 1: 1.6, 2: 1.0},
                    random_state=42,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_model.fit(training.X.iloc[train_idx], training.y[train_idx])
        logit_model.fit(training.X.iloc[train_idx], training.y[train_idx])

    poisson_model = fit_dixon_coles(results_df)
    base_models = [xgb_model, rf_model, logit_model]
    Z_val = _base_meta_features(base_models, poisson_model, training, val_idx)
    meta = LogisticRegression(max_iter=1000,  random_state=42)
    meta.fit(Z_val, training.y[val_idx], sample_weight=weights[val_idx])

    bundle = ModelBundle(
        model=xgb_model,
        state=training.state,
        feature_names=training.feature_names,
        poisson_model=poisson_model,
        alpha=1.0,
        odds=odds,
        squad_values=squad_values,
        country_history=country_history,
        notes=["Meta-learner trained on chronological holdout base probabilities"],
    )
    bundle.base_models = base_models
    bundle.meta_model = meta
    return bundle


def predict_stacked(bundle, home, away, state, cf, stage, date, *, neutral, is_home, feature_adapter=None):
    odds_row = odds_features_for_match(bundle.odds, date, home, away)
    feat = compute_match_features(
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
    X = align_prediction_frame(feat, bundle.feature_names, feature_adapter)
    chunks = [np.asarray(model.predict_proba(X), dtype=float) for model in bundle.base_models]
    chunks.append(np.asarray([bundle.poisson_model.outcome_probs(home, away, neutral=neutral)], dtype=float))
    return normalize_probs(bundle.meta_model.predict_proba(np.hstack(chunks))[0])


def train_dynamic_bundle():
    import xgboost as xgb
    from shared import fit_xgb_with_validation

    results_df, country_history, odds, squad_values = load_inputs()
    training = build_training_data(results_df, country_history, odds, squad_values)
    weights = sample_weight_array(training)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = fit_xgb_with_validation(
            model,
            training.X,
            training.y,
            label="XGBoost dynamic base",
            dates=training.dates,
            sample_weight=weights,
            calibrate=True,
        )
    bundle = make_bundle(
        model,
        training,
        results_df=results_df,
        country_history=country_history,
        odds=odds,
        squad_values=squad_values,
        tune_poisson_blend=True,
        notes=["Dynamic blend lowers ML weight when draw-context features are high"],
    )
    return bundle


def predict_dynamic(bundle, home, away, state, cf, stage, date, *, neutral, is_home, feature_adapter=None):
    odds_row = odds_features_for_match(bundle.odds, date, home, away)
    feat = compute_match_features(
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
    X = align_prediction_frame(feat, bundle.feature_names, feature_adapter)
    p_model = np.asarray(bundle.model.predict_proba(X)[0], dtype=float)
    p_pois = bundle.poisson_model.outcome_probs(home, away, neutral=neutral)
    draw_context = float(
        np.mean(
            [
                feat.get("elo_parity", 0.5),
                feat.get("combined_draw_rate", 0.25) * 2.0,
                feat.get("low_scoring_indicator", 0.25) * 2.0,
            ]
        )
    )
    alpha = float(np.clip(bundle.alpha - 0.20 * (draw_context - 0.5), 0.05, 0.95))
    probs = blend_probabilities(p_model, p_pois, alpha)
    if draw_context > 0.65:
        probs = probs.copy()
        probs[1] *= 1.0 + min(0.20, draw_context - 0.65)
        probs = normalize_probs(probs)
    return probs


def main():
    stacking = train_stacking_bundle()
    run_walk_forward_backtest("Stacking: XGB + RF + Logit + Dixon-Coles", stacking, predict_fn=predict_stacked)

    dynamic = train_dynamic_bundle()
    run_walk_forward_backtest("Dynamic Dixon-Coles blend", dynamic, predict_fn=predict_dynamic)


if __name__ == "__main__":
    main()
