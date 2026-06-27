#!/usr/bin/env python3
"""Approach 5: draw-specific decomposition model."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from experiment_common import (
    build_training_data,
    chronological_holdout_indices,
    load_inputs,
    make_bundle,
    normalize_probs,
    predict_default,
    run_walk_forward_backtest,
    sample_weight_array,
)


class DrawDecompositionModel:
    """P(draw) model + P(home|not draw) model composed into 3 classes."""

    def __init__(self, draw_model, win_model):
        self.draw_model = draw_model
        self.win_model = win_model
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        p_draw = self.draw_model.predict_proba(X)[:, 1]
        p_home_given_not_draw = self.win_model.predict_proba(X)[:, 1]
        p_not_draw = 1.0 - p_draw
        probs = np.column_stack(
            [
                p_not_draw * p_home_given_not_draw,
                p_draw,
                p_not_draw * (1.0 - p_home_given_not_draw),
            ]
        )
        return np.asarray([normalize_probs(row) for row in probs], dtype=float)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def _fit_models(training, weights):
    draw_y = (training.y == 1).astype(int)
    draw_weights = np.where(draw_y == 1, weights * 1.35, weights)
    draw_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.04,
        max_iter=350,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )
    non_draw = training.y != 1
    win_y = (training.y[non_draw] == 0).astype(int)
    win_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=43,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draw_model.fit(training.X, draw_y, sample_weight=draw_weights)
        win_model.fit(training.X.loc[non_draw], win_y, sample_weight=weights[non_draw])
    return DrawDecompositionModel(draw_model, win_model)


def train_bundle():
    results_df, country_history, odds, squad_values = load_inputs()
    training = build_training_data(results_df, country_history, odds, squad_values)
    weights = sample_weight_array(training)
    model = _fit_models(training, weights)
    return make_bundle(
        model,
        training,
        results_df=results_df,
        country_history=country_history,
        odds=odds,
        squad_values=squad_values,
        tune_poisson_blend=True,
        notes=["Binary draw model composed with home-vs-away non-draw model"],
    ), training


def tune_draw_threshold(bundle, training):
    _, val_idx = chronological_holdout_indices(training.dates)
    probs = np.asarray(bundle.model.predict_proba(training.X.iloc[val_idx]), dtype=float)
    y_val = training.y[val_idx]
    best_threshold, best_loss = 0.33, np.inf
    for threshold in np.linspace(0.22, 0.46, 25):
        adjusted = probs.copy()
        mask = adjusted[:, 1] >= threshold
        if np.any(mask):
            adjusted[mask, 1] = np.maximum(adjusted[mask, 1], adjusted[mask].max(axis=1) + 0.01)
            adjusted[mask] = np.asarray([normalize_probs(row) for row in adjusted[mask]], dtype=float)
        loss = log_loss(y_val, adjusted, labels=[0, 1, 2])
        if loss < best_loss:
            best_loss = float(loss)
            best_threshold = float(threshold)
    return best_threshold, best_loss


def predict_threshold(bundle, *args, feature_adapter=None, **kwargs):
    probs = predict_default(bundle, *args, feature_adapter=feature_adapter, **kwargs)
    threshold = getattr(bundle, "draw_threshold", 0.33)
    if probs[1] >= threshold:
        probs = probs.copy()
        probs[1] = max(probs[1], probs.max() + 0.01)
        probs = normalize_probs(probs)
    return probs


def main():
    bundle, training = train_bundle()
    run_walk_forward_backtest("Draw decomposition + Dixon-Coles", bundle)

    threshold, holdout_loss = tune_draw_threshold(bundle, training)
    bundle.draw_threshold = threshold
    bundle.notes = [f"Draw override threshold={threshold:.2f} (holdout log-loss {holdout_loss:.4f})"]
    run_walk_forward_backtest("Draw threshold override", bundle, predict_fn=predict_threshold)


if __name__ == "__main__":
    main()
