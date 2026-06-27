#!/usr/bin/env python3
"""Approach 6: extra feature engineering."""

from __future__ import annotations

import warnings

from experiment_common import (
    EnhancedFeatureAdapter,
    build_training_data,
    load_inputs,
    make_bundle,
    run_walk_forward_backtest,
    sample_weight_array,
)


EXPERIMENT_NAME = "XGBoost + engineered feature additions"


def train_bundle():
    import xgboost as xgb
    from shared import fit_xgb_with_validation

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
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = fit_xgb_with_validation(
            model,
            training.X,
            training.y,
            label="XGBoost engineered features",
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
        notes=[
            "Added Elo/form interactions, Elo polynomials/bins, squad-value polynomial, draw-low-goal interaction",
            "Target encoding was not added because current shared features expose no stable categorical confederation column",
        ],
    )
    return bundle, adapter


def main():
    bundle, adapter = train_bundle()
    run_walk_forward_backtest(EXPERIMENT_NAME, bundle, feature_adapter=adapter)


if __name__ == "__main__":
    main()
