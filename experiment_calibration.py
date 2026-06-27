#!/usr/bin/env python3
"""Approach 3: improved calibration on ensemble probabilities."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from experiment_common import (
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


def _logit_features(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


class PlattCalibrator:
    name = "Platt scaling"

    def fit(self, probs: np.ndarray, y: np.ndarray, sample_weight=None):
        self.model = LogisticRegression(max_iter=1000,  random_state=42)
        self.model.fit(_logit_features(probs), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit_features(probs))


class TemperatureCalibrator:
    name = "Temperature scaling"

    def fit(self, probs: np.ndarray, y: np.ndarray, sample_weight=None):
        logits = np.log(np.clip(probs, 1e-12, 1.0))

        def transform(temp: float) -> np.ndarray:
            z = logits / max(temp, 1e-3)
            z = z - z.max(axis=1, keepdims=True)
            out = np.exp(z)
            return out / out.sum(axis=1, keepdims=True)

        best_t, best_loss = 1.0, np.inf
        for temp in np.linspace(0.5, 3.0, 101):
            loss = log_loss(y, transform(float(temp)), labels=[0, 1, 2], sample_weight=sample_weight)
            if loss < best_loss:
                best_t, best_loss = float(temp), float(loss)
        self.temperature = best_t
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(probs, 1e-12, 1.0)) / self.temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        out = np.exp(logits)
        return out / out.sum(axis=1, keepdims=True)


class EnsembleIsotonicCalibrator:
    name = "Isotonic on ensemble"

    def fit(self, probs: np.ndarray, y: np.ndarray, sample_weight=None):
        self.models = []
        for cls in [0, 1, 2]:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs[:, cls], (y == cls).astype(float), sample_weight=sample_weight)
            self.models.append(iso)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        out = np.column_stack([model.predict(probs[:, i]) for i, model in enumerate(self.models)])
        return np.asarray([normalize_probs(row) for row in out], dtype=float)


class BetaCalibrator:
    name = "Beta calibration"

    def fit(self, probs: np.ndarray, y: np.ndarray, sample_weight=None):
        p = np.clip(probs, 1e-6, 1.0 - 1e-6)
        self.models = []
        for cls in [0, 1, 2]:
            X = np.column_stack([np.log(p[:, cls]), np.log1p(-p[:, cls])])
            target = (y == cls).astype(int)
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X, target, sample_weight=sample_weight)
            self.models.append(lr)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        p = np.clip(probs, 1e-6, 1.0 - 1e-6)
        cols = []
        for cls, model in enumerate(self.models):
            X = np.column_stack([np.log(p[:, cls]), np.log1p(-p[:, cls])])
            cols.append(model.predict_proba(X)[:, 1])
        out = np.column_stack(cols)
        return np.asarray([normalize_probs(row) for row in out], dtype=float)


def fit_base_bundle():
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
            label="XGBoost calibration base",
            dates=training.dates,
            sample_weight=weights,
            calibrate=False,
        )
    poisson_model, alpha, blend_loss = fit_dixon_and_alpha(results_df, model, training)
    bundle = make_bundle(
        model,
        training,
        results_df=results_df,
        country_history=country_history,
        odds=odds,
        squad_values=squad_values,
        tune_poisson_blend=False,
        notes=[f"Base Dixon-Coles alpha={alpha:.2f} (holdout log-loss {blend_loss:.4f})"],
    )
    bundle.poisson_model = poisson_model
    bundle.alpha = alpha
    return bundle, training, weights


def _ensemble_probs(bundle, training, indices):
    p_model = np.asarray(bundle.model.predict_proba(training.X.iloc[indices]), dtype=float)
    if bundle.poisson_model is None or bundle.alpha >= 1.0:
        return p_model
    p_pois = np.asarray(
        [
            bundle.poisson_model.outcome_probs(training.match_meta[i][0], training.match_meta[i][1], neutral=training.match_meta[i][2])
            for i in indices
        ],
        dtype=float,
    )
    from shared import blend_probabilities

    return np.asarray([blend_probabilities(p_model[k], p_pois[k], bundle.alpha) for k in range(len(indices))], dtype=float)


def calibrated_predict(bundle, *args, feature_adapter=None, **kwargs):
    probs = predict_default(bundle, *args, feature_adapter=feature_adapter, **kwargs)
    calibrator = getattr(bundle, "calibrator", None)
    if calibrator is None:
        return probs
    return normalize_probs(calibrator.predict_proba(np.asarray([probs], dtype=float))[0])


def main():
    bundle, training, weights = fit_base_bundle()
    _, val_idx = chronological_holdout_indices(training.dates)
    val_probs = _ensemble_probs(bundle, training, val_idx)
    y_val = training.y[val_idx]
    w_val = weights[val_idx]

    run_walk_forward_backtest("Calibration baseline ensemble", bundle, predict_fn=calibrated_predict)

    for calibrator in [
        PlattCalibrator(),
        TemperatureCalibrator(),
        EnsembleIsotonicCalibrator(),
        BetaCalibrator(),
    ]:
        fitted = calibrator.fit(val_probs, y_val, sample_weight=w_val)
        bundle.calibrator = fitted
        loss = log_loss(y_val, fitted.predict_proba(val_probs), labels=[0, 1, 2], sample_weight=w_val)
        bundle.notes = [f"{fitted.name}; weighted holdout log-loss={loss:.4f}"]
        run_walk_forward_backtest(f"Calibration: {fitted.name}", bundle, predict_fn=calibrated_predict)


if __name__ == "__main__":
    main()
