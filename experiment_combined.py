#!/usr/bin/env python3
"""Final combined experiment candidate.

Uses engineered features, LightGBM when available (XGBoost fallback),
Dixon-Coles blending, and isotonic calibration fitted on the ensemble output.
"""

from __future__ import annotations

import warnings

import numpy as np

from experiment_calibration import EnsembleIsotonicCalibrator
from experiment_common import (
    EnhancedFeatureAdapter,
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
from shared import blend_probabilities


EXPERIMENT_NAME = "Combined: engineered + boosted trees + DC + isotonic"


def _fit_lightgbm_or_xgb(training, weights, notes):
    train_idx, val_idx = chronological_holdout_indices(training.dates)
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        import xgboost as xgb
        from shared import fit_xgb_with_validation

        notes.append(f"LightGBM unavailable ({exc}); fell back to XGBoost")
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
            n_estimators=350,
            max_depth=5,
            learning_rate=0.045,
            subsample=0.85,
            colsample_bytree=0.85,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, _ = fit_xgb_with_validation(
                model,
                training.X,
                training.y,
                label="Combined XGBoost fallback",
                dates=training.dates,
                sample_weight=weights,
                calibrate=False,
            )
        return model

    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=650,
        num_leaves=48,
        max_depth=8,
        learning_rate=0.035,
        min_child_samples=45,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=42,
        verbosity=-1,
    )
    fit_kwargs = {
        "sample_weight": weights[train_idx],
        "eval_set": [(training.X.iloc[val_idx], training.y[val_idx])],
        "eval_metric": "multi_logloss",
        "eval_sample_weight": [weights[val_idx]],
    }
    try:
        import lightgbm as lgb

        fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(training.X.iloc[train_idx], training.y[train_idx], **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("callbacks", None)
            model.fit(training.X.iloc[train_idx], training.y[train_idx], **fit_kwargs)
    notes.append("Used LightGBM base with conservative tuned-style parameters")
    return model


def _fit_ensemble_calibrator(bundle, training, weights):
    _, val_idx = chronological_holdout_indices(training.dates)
    p_model = np.asarray(bundle.model.predict_proba(training.X.iloc[val_idx]), dtype=float)
    if bundle.poisson_model is not None and bundle.alpha < 1.0:
        p_pois = np.asarray(
            [
                bundle.poisson_model.outcome_probs(training.match_meta[i][0], training.match_meta[i][1], neutral=training.match_meta[i][2])
                for i in val_idx
            ],
            dtype=float,
        )
        probs = np.asarray([blend_probabilities(p_model[k], p_pois[k], bundle.alpha) for k in range(len(val_idx))], dtype=float)
    else:
        probs = p_model
    return EnsembleIsotonicCalibrator().fit(probs, training.y[val_idx], sample_weight=weights[val_idx])


def predict_combined(bundle, *args, feature_adapter=None, **kwargs):
    probs = predict_default(bundle, *args, feature_adapter=feature_adapter, **kwargs)
    calibrator = getattr(bundle, "calibrator", None)
    if calibrator is not None:
        probs = normalize_probs(calibrator.predict_proba(np.asarray([probs], dtype=float))[0])
    return probs


def train_bundle():
    adapter = EnhancedFeatureAdapter()
    results_df, country_history, odds, squad_values = load_inputs()
    training = build_training_data(
        results_df,
        country_history,
        odds,
        squad_values,
        feature_adapter=adapter,
    )
    weights = sample_weight_array(training)
    notes: list[str] = []
    model = _fit_lightgbm_or_xgb(training, weights, notes)
    poisson_model, alpha, blend_loss = fit_dixon_and_alpha(results_df, model, training)
    bundle = make_bundle(
        model,
        training,
        results_df=results_df,
        country_history=country_history,
        odds=odds,
        squad_values=squad_values,
        tune_poisson_blend=False,
        notes=notes + [f"Dixon-Coles alpha={alpha:.2f} (holdout log-loss {blend_loss:.4f})"],
    )
    bundle.poisson_model = poisson_model
    bundle.alpha = alpha
    bundle.calibrator = _fit_ensemble_calibrator(bundle, training, weights)
    bundle.notes.append("Applied per-class isotonic calibration to blended ensemble probabilities")
    return bundle, adapter


def main():
    bundle, adapter = train_bundle()
    run_walk_forward_backtest(EXPERIMENT_NAME, bundle, predict_fn=predict_combined, feature_adapter=adapter)


if __name__ == "__main__":
    main()
