#!/usr/bin/env python3
"""Approach 1: LightGBM + Bayesian hyperparameter optimization."""

from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.metrics import log_loss

from experiment_common import (
    build_training_data,
    chronological_holdout_indices,
    load_inputs,
    make_bundle,
    run_walk_forward_backtest,
    sample_weight_array,
)


EXPERIMENT_NAME = "LightGBM + Optuna + Dixon-Coles"


def _fit_lgbm(model, X_train, y_train, X_val=None, y_val=None, sample_weight=None, sample_weight_val=None):
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_metric"] = "multi_logloss"
        if sample_weight_val is not None:
            fit_kwargs["eval_sample_weight"] = [sample_weight_val]
        try:
            import lightgbm as lgb

            fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        except Exception:
            pass
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        fit_kwargs.pop("callbacks", None)
        model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_bundle(n_trials: int | None = None):
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise SystemExit(f"SKIP: LightGBM is unavailable ({exc}). Install `lightgbm` to run this experiment.")

    results_df, country_history, odds, squad_values = load_inputs()
    training = build_training_data(results_df, country_history, odds, squad_values)
    weights = sample_weight_array(training)
    train_idx, val_idx = chronological_holdout_indices(training.dates)
    X_train, X_val = training.X.iloc[train_idx], training.X.iloc[val_idx]
    y_train, y_val = training.y[train_idx], training.y[val_idx]
    w_train, w_val = weights[train_idx], weights[val_idx]
    notes: list[str] = []

    base_params = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": 600,
        "random_state": 42,
        "verbosity": -1,
        "class_weight": None,
    }
    best_params = {
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.04,
        "min_child_samples": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }

    try:
        import optuna
    except Exception as exc:
        notes.append(f"Optuna unavailable ({exc}); used conservative LightGBM defaults")
    else:
        n_trials = int(n_trials or os.getenv("LGBM_OPTUNA_TRIALS", "50"))

        def objective(trial):
            params = {
                **base_params,
                "num_leaves": trial.suggest_int("num_leaves", 16, 160),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
                "subsample": trial.suggest_float("subsample", 0.55, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            model = LGBMClassifier(**params)
            _fit_lgbm(model, X_train, y_train, X_val, y_val, w_train, w_val)
            probs = np.asarray(model.predict_proba(X_val), dtype=float)
            return log_loss(y_val, probs, labels=[0, 1, 2])

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params.update(study.best_params)
        notes.append(f"Optuna trials={n_trials}; best holdout log-loss={study.best_value:.4f}")

    model = LGBMClassifier(**base_params, **best_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fit_lgbm(model, training.X, training.y, sample_weight=weights)
    return make_bundle(
        model,
        training,
        results_df=results_df,
        country_history=country_history,
        odds=odds,
        squad_values=squad_values,
        tune_poisson_blend=True,
        notes=notes,
    )


def main():
    bundle = train_bundle()
    run_walk_forward_backtest(EXPERIMENT_NAME, bundle)


if __name__ == "__main__":
    main()
