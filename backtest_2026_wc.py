"""
Walk-forward backtest of the World Cup 2026 prediction model.

For each completed 2026 WC match in chronological order:
1. Use current state (Elo, form, H2H, etc.) to predict
2. Compare prediction against actual result
3. Update state with the real result
4. Move to next match

Reports: accuracy, log-loss, Brier score, per-stage breakdown, and per-match details.

Use ``--compare`` to evaluate XGBoost vs LightGBM (Hyperopt) with/without tradition features.
"""
from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared import (
    KNOCKOUT_ALPHA,
    DATA_DIR,
    DEFAULT_GBT_MODEL,
    TRADITION_FEATURE_COLUMNS,
    GROUP_2026_TEAMS,
    WC2026_STAGE_TO_TRAIN,
    analyze_tradition_correlation,
    apply_match_to_state,
    blend_probabilities,
    compute_match_features,
    country_features_for_year,
    drop_feature_columns,
    fit_dixon_coles,
    fit_gbt_with_validation,
    finalize_feature_frame,
    finalize_world_cup_history,
    get_gbt_feature_importance,
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

# ─── Constants ───────────────────────────────────────────────────────────────
RESULT_LABELS = ["Home win", "Draw", "Away win"]
WC_TEAM_SET = {harmonize_country(t) for teams in GROUP_2026_TEAMS.values() for t in teams}


@dataclass
class BacktestConfig:
    model_type: str = DEFAULT_GBT_MODEL
    exclude_tradition: bool = False
    hyperopt_trials: int = 0
    lgbm_params: dict | None = None
    label: str = ""

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        model = self.model_type.upper()
        trad = "no-tradition" if self.exclude_tradition else "tradition"
        return f"{model}+{trad}"


@dataclass
class BacktestMetrics:
    name: str
    accuracy: float
    log_loss: float
    brier: float
    correct: int
    total: int
    stage_metrics: dict
    calibration: list[dict]


def make_initial_state():
    return defaultdict(make_team_state)


def _tune_blend_alpha(gbt_model, poisson_model, X_val, y_val, val_meta):
    """Pick the GBT/Poisson blend weight that minimizes holdout log-loss."""
    from sklearn.metrics import log_loss as _ll

    p_gbt = gbt_model.predict_proba(X_val)
    p_pois = np.array([
        poisson_model.outcome_probs(h, a, neutral=neu)
        for (h, a, neu) in val_meta
    ])
    best_alpha, best_loss = 1.0, np.inf
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = np.array([
            blend_probabilities(p_gbt[i], p_pois[i], alpha) for i in range(len(p_gbt))
        ])
        try:
            loss = _ll(y_val, blended, labels=[0, 1, 2])
        except Exception:
            continue
        if loss < best_loss:
            best_loss, best_alpha = loss, alpha
    return float(best_alpha), float(best_loss)


def build_training_matrix(results_df, country_history, odds=None, squad_values=None):
    """Build pre-2026-WC feature matrix (shared by analysis and training)."""
    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[~((df["date"].dt.year == 2026) & (df["tournament"] == "FIFA World Cup"))]
    df = df.sort_values("date").reset_index(drop=True)

    state = make_initial_state()
    rows, labels, feature_dates = [], [], []
    match_meta = []
    country_feature_cache = {}
    active_wc_year = None
    active_wc_teams = set()
    wc_stage_by_index = infer_world_cup_stage_map(df)

    for _, r in df.iterrows():
        ht = harmonize_country(r["home_team"])
        at = harmonize_country(r["away_team"])
        hs, aw = r["home_score"], r["away_score"]
        if pd.isna(hs) or pd.isna(aw):
            continue
        hs, aw = int(hs), int(aw)

        feature_year = int(r["date"].year)
        if feature_year not in country_feature_cache:
            country_feature_cache[feature_year] = country_features_for_year(country_history, feature_year)

        is_world_cup = r["tournament"] == "FIFA World Cup"
        if active_wc_year is not None and (not is_world_cup or feature_year != active_wc_year):
            finalize_world_cup_history(state, active_wc_year, active_wc_teams)
            active_wc_year, active_wc_teams = None, set()

        neutral = parse_bool(r.get("neutral", True))
        is_home = not neutral
        stage = wc_stage_by_index.get(int(r.name), 0) if is_world_cup else 0
        odds_row = odds_features_for_match(odds, r["date"], ht, at)
        rows.append(compute_match_features(
            ht, at, state, country_feature_cache[feature_year], stage, r["date"],
            neutral=neutral, is_home=is_home, odds_row=odds_row,
            squad_values=squad_values,
        ))
        labels.append(0 if hs > aw else (1 if hs == aw else 2))
        feature_dates.append(r["date"])
        match_meta.append((ht, at, neutral))

        if is_world_cup:
            if active_wc_year is None:
                active_wc_year = feature_year
            active_wc_teams.update([ht, at])

        apply_match_to_state(state, ht, at, hs, aw, r["date"],
                             neutral=neutral, is_world_cup=is_world_cup)

    if active_wc_year is not None:
        finalize_world_cup_history(state, active_wc_year, active_wc_teams)

    X = finalize_feature_frame(rows)
    y = np.array(labels)
    return X, y, feature_dates, match_meta, state


def train_model(
    results_df,
    country_history,
    odds=None,
    squad_values=None,
    *,
    config: BacktestConfig | None = None,
):
    """Train GBT + Dixon-Coles on all pre-2026-WC matches."""
    config = config or BacktestConfig()
    X, y, feature_dates, match_meta, state = build_training_matrix(
        results_df, country_history, odds=odds, squad_values=squad_values,
    )
    if config.exclude_tradition:
        X = drop_feature_columns(X, TRADITION_FEATURE_COLUMNS)

    weights = sample_weights(y, feature_dates)
    gbt_label = "LightGBM" if config.model_type == "lgbm" else "XGBoost"
    trials = config.hyperopt_trials if config.model_type == "lgbm" else 0
    model, _ = fit_gbt_with_validation(
        config.model_type, X, y,
        dates=feature_dates,
        sample_weight=weights,
        calibrate=True,
        lgbm_params=config.lgbm_params,
        hyperopt_trials=trials,
        label=gbt_label,
    )

    print("  Fitting Dixon-Coles Poisson goal model...")
    poisson_model = fit_dixon_coles(results_df)
    order = pd.Series(pd.to_datetime(feature_dates, errors="coerce")).sort_values().index
    split = max(1, int(len(order) * 0.8))
    val_idx = order[split:]
    X_val = X.iloc[val_idx]
    y_val = y[val_idx]
    val_meta = [match_meta[i] for i in val_idx]
    alpha, blend_loss = _tune_blend_alpha(model, poisson_model, X_val, y_val, val_meta)
    print(f"  Tuned blend alpha ({gbt_label} weight) = {alpha:.2f}  (holdout log-loss {blend_loss:.3f})")
    return model, state, X.columns.tolist(), poisson_model, alpha


def prepare_2026_state(state, results_df):
    """Update state with completed 2026 WC group matches before backtest starts."""
    for teams in GROUP_2026_TEAMS.values():
        for team in teams:
            ht = harmonize_country(team)
            state[ht]["wc_participations"] += 1
    return state


def predict_match(model, feature_names, home, away, state, cf, stage, date,
                  neutral=True, is_home=False, odds=None,
                  poisson_model=None, alpha=1.0, squad_values=None):
    """Predict a single match. Returns (predicted_label_idx, probs[3])."""
    odds_row = odds_features_for_match(odds, date, home, away)
    feat = compute_match_features(home, away, state, cf, stage, date,
                                  neutral=neutral, is_home=is_home, odds_row=odds_row,
                                  squad_values=squad_values)
    X = prepare_prediction_frame(feat, feature_names)
    probs = np.asarray(model.predict_proba(X)[0], dtype=float)

    blend_alpha = KNOCKOUT_ALPHA if stage > 0 else alpha
    if poisson_model is not None and blend_alpha < 1.0:
        p_pois = poisson_model.outcome_probs(home, away, neutral=neutral)
        probs = blend_probabilities(probs, p_pois, blend_alpha)

    if stage > 0:
        total = probs[0] + probs[2]
        if total > 0:
            probs = np.array([probs[0] / total, 0.0, probs[2] / total])

    predicted = int(np.argmax(probs))
    return predicted, probs


def actual_result(home_score, away_score):
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def stage_from_tournament_round(tournament_str, home_team, away_team):
    t = str(tournament_str).lower()
    if "group" in t:
        return WC2026_STAGE_TO_TRAIN["group"]
    if "round of 32" in t or "r32" in t:
        return WC2026_STAGE_TO_TRAIN["round_of_32"]
    if "round of 16" in t or "r16" in t:
        return WC2026_STAGE_TO_TRAIN["round_of_16"]
    if "quarter" in t or "qf" in t:
        return WC2026_STAGE_TO_TRAIN["quarterfinal"]
    if "semi" in t or "sf" in t:
        return WC2026_STAGE_TO_TRAIN["semifinal"]
    if "third" in t or "3rd" in t:
        return WC2026_STAGE_TO_TRAIN["third_place"]
    if "final" in t and "semi" not in t:
        return WC2026_STAGE_TO_TRAIN["final"]
    return 0


def _compute_metrics(results: list[dict]) -> BacktestMetrics:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    eps = 1e-15
    log_losses = []
    brier = 0.0
    for r in results:
        p = max(min(r["actual_prob"], 1 - eps), eps)
        log_losses.append(-np.log(p))
        one_hot = np.zeros(3)
        one_hot[r["actual_idx"]] = 1.0
        probs = np.array([r["p_home"], r["p_draw"], r["p_away"]])
        brier += np.mean((probs - one_hot) ** 2)
    brier /= max(total, 1)

    stage_results: dict[int, dict] = {}
    for r in results:
        s = r["stage"]
        if s not in stage_results:
            stage_results[s] = {"correct": 0, "total": 0, "log_losses": []}
        stage_results[s]["total"] += 1
        if r["correct"]:
            stage_results[s]["correct"] += 1
        p = max(min(r["actual_prob"], 1 - eps), eps)
        stage_results[s]["log_losses"].append(-np.log(p))

    calibration = []
    for lo, hi in [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        bucket = [r for r in results if lo <= r["confidence"] < hi]
        if bucket:
            calibration.append({
                "range": f"{lo:.0%}-{hi:.0%}",
                "n": len(bucket),
                "avg_conf": float(np.mean([r["confidence"] for r in bucket])),
                "accuracy": sum(1 for r in bucket if r["correct"]) / len(bucket),
            })

    return BacktestMetrics(
        name="",
        accuracy=correct / max(total, 1),
        log_loss=float(np.mean(log_losses)) if log_losses else float("nan"),
        brier=float(brier),
        correct=correct,
        total=total,
        stage_metrics=stage_results,
        calibration=calibration,
    )


def run_backtest(config: BacktestConfig | None = None, *, verbose: bool = True) -> tuple[list[dict], BacktestMetrics]:
    config = config or BacktestConfig()
    if verbose:
        print("=" * 70)
        print(f"  WORLD CUP 2026 WALK-FORWARD BACKTEST — {config.name}")
        print("=" * 70)
        print()

    results_df = pd.read_csv(DATA_DIR / "results.csv")
    country_history = load_country_feature_history()
    odds = load_betting_odds()
    squad_values = load_squad_values()
    if verbose:
        print(f"Loaded bookmaker odds for {len(odds) // 2} fixtures.")
        print(f"Loaded squad values for {len(squad_values)} team-years.")
        print("Training model on historical data (excluding 2026 WC)...")

    model, state, feature_names, poisson_model, alpha = train_model(
        results_df, country_history, odds=odds, squad_values=squad_values, config=config,
    )
    state = prepare_2026_state(state, results_df)
    cf = country_features_for_year(country_history, 2026)

    results_df["date"] = pd.to_datetime(results_df["date"])
    wc26 = results_df[
        (results_df["tournament"] == "FIFA World Cup") & (results_df["date"].dt.year == 2026)
    ].copy()
    completed = wc26[wc26["home_score"].notna() & wc26["away_score"].notna()].sort_values("date")

    if verbose:
        print(f"Backtesting {len(completed)} completed matches...")
        print()

    results = []
    for _, r in completed.iterrows():
        home = harmonize_country(r["home_team"])
        away = harmonize_country(r["away_team"])
        hs, aw = int(r["home_score"]), int(r["away_score"])
        date = r["date"]
        stage = stage_from_tournament_round(r.get("tournament", ""), home, away)
        neutral = parse_bool(r.get("neutral", True))
        is_home = not neutral

        predicted_idx, probs = predict_match(
            model, feature_names, home, away, state, cf, stage, date,
            neutral=neutral, is_home=is_home,
            odds=odds, poisson_model=poisson_model, alpha=alpha,
            squad_values=squad_values,
        )
        actual_idx = actual_result(hs, aw)
        is_correct = predicted_idx == actual_idx
        results.append({
            "date": date.strftime("%Y-%m-%d"),
            "home": home,
            "away": away,
            "score": f"{hs}-{aw}",
            "stage": stage,
            "predicted": RESULT_LABELS[predicted_idx],
            "actual": RESULT_LABELS[actual_idx],
            "predicted_idx": predicted_idx,
            "actual_idx": actual_idx,
            "correct": is_correct,
            "confidence": float(probs[predicted_idx]),
            "actual_prob": float(probs[actual_idx]),
            "p_home": float(probs[0]),
            "p_draw": float(probs[1]),
            "p_away": float(probs[2]),
        })
        apply_match_to_state(state, home, away, hs, aw, date,
                             neutral=neutral, is_world_cup=True)

    metrics = _compute_metrics(results)
    metrics.name = config.name

    if verbose:
        _print_report(results, metrics)
    return results, metrics


def _print_report(results: list[dict], metrics: BacktestMetrics) -> None:
    stage_names = {0: "Group Stage", 1: "Round of 32", 2: "Round of 16",
                   3: "Quarterfinals", 4: "Semifinals", 5: "Third Place", 6: "Final"}
    print()
    print("=" * 70)
    print(f"  RESULTS SUMMARY — {metrics.name}")
    print("=" * 70)
    print()
    print(f"  Total matches:  {metrics.total}")
    print(f"  Correct:        {metrics.correct}")
    print(f"  Accuracy:       {metrics.accuracy:.1%}")
    print(f"  Log-loss:       {metrics.log_loss:.4f}")
    print(f"  Brier score:    {metrics.brier:.4f}")
    print()
    print("  PER-STAGE BREAKDOWN:")
    print(f"  {'Stage':<20s} {'Correct':>8s} {'Total':>6s} {'Acc':>8s} {'LogLoss':>9s}")
    print("  " + "-" * 55)
    for s in sorted(metrics.stage_metrics.keys()):
        sr = metrics.stage_metrics[s]
        acc = sr["correct"] / sr["total"] if sr["total"] > 0 else 0
        ll = np.mean(sr["log_losses"]) if sr["log_losses"] else 0
        name = stage_names.get(s, f"Stage {s}")
        print(f"  {name:<20s} {sr['correct']:>8d} {sr['total']:>6d} {acc:>8.1%} {ll:>9.4f}")
    print()
    print("  CALIBRATION (predicted confidence vs actual accuracy):")
    for row in metrics.calibration:
        print(f"    {row['range']}: n={row['n']:3d}, avg_conf={row['avg_conf']:.1%}, actual_acc={row['accuracy']:.1%}")
    print()
    print(f"Final accuracy: {metrics.accuracy:.1%} ({metrics.correct}/{metrics.total})")
    print(f"Log-loss: {metrics.log_loss:.4f} | Brier: {metrics.brier:.4f}")


def run_tradition_analysis(X: pd.DataFrame, y: np.ndarray, feature_dates: list) -> None:
    """Print correlation check and XGBoost feature importance for tradition features."""
    print("\n" + "=" * 70)
    print("  TRADITION FEATURE ANALYSIS")
    print("=" * 70)
    corr = analyze_tradition_correlation(X)
    print(f"\n  tradition_diff vs elo_diff:")
    print(f"    Pearson r  = {corr['pearson']:.4f}")
    print(f"    Spearman r = {corr['spearman']:.4f}")
    if abs(corr["pearson"]) > 0.7 or abs(corr["spearman"]) > 0.7:
        print("    → HIGH correlation (>0.7): tradition features likely redundant with Elo")
    else:
        print("    → Moderate correlation: tradition may carry independent signal")

    from shared import create_xgb_classifier, fit_xgb_with_validation

    weights = sample_weights(y, feature_dates)
    model = create_xgb_classifier()
    model, _ = fit_xgb_with_validation(
        model, X, y, label="XGBoost (importance)", dates=feature_dates,
        sample_weight=weights, calibrate=False,
    )
    imp = get_gbt_feature_importance(model, X.columns, top_n=20)
    print("\n  Top 20 features by gain (normalized):")
    for _, row in imp.iterrows():
        marker = " ← tradition" if row["feature"] in TRADITION_FEATURE_COLUMNS else ""
        print(f"    {row['feature']:<30s} {row['importance']:.4f}{marker}")

    trad_in_top15 = imp.head(15)["feature"].isin(TRADITION_FEATURE_COLUMNS).any()
    print(f"\n  Tradition features in top 15: {'YES' if trad_in_top15 else 'NO'}")


def run_comparison(hyperopt_trials: int = 100) -> pd.DataFrame:
    """Run all four model variants and print a comparison table."""
    results_df = pd.read_csv(DATA_DIR / "results.csv")
    country_history = load_country_feature_history()
    odds = load_betting_odds()
    squad_values = load_squad_values()

    X, y, feature_dates, _, _ = build_training_matrix(
        results_df, country_history, odds=odds, squad_values=squad_values,
    )
    run_tradition_analysis(X, y, feature_dates)

    # Hyperopt once per feature set (with / without tradition).
    from shared import tune_lgbm_hyperopt

    print("\n  Tuning LightGBM (with tradition features)...")
    lgbm_params_full = tune_lgbm_hyperopt(
        X, y, dates=feature_dates, sample_weight=sample_weights(y, feature_dates),
        n_trials=hyperopt_trials, label="LightGBM+tradition",
    )
    X_no_trad = drop_feature_columns(X, TRADITION_FEATURE_COLUMNS)
    print("\n  Tuning LightGBM (without tradition features)...")
    lgbm_params_no_trad = tune_lgbm_hyperopt(
        X_no_trad, y, dates=feature_dates, sample_weight=sample_weights(y, feature_dates),
        n_trials=hyperopt_trials, label="LightGBM-no-tradition",
    )

    variants = [
        BacktestConfig(model_type="xgb", exclude_tradition=False, hyperopt_trials=0,
                       label="XGBoost + tradition"),
        BacktestConfig(model_type="xgb", exclude_tradition=True, hyperopt_trials=0,
                       label="XGBoost − tradition"),
        BacktestConfig(model_type="lgbm", exclude_tradition=False, hyperopt_trials=0,
                       lgbm_params=lgbm_params_full, label="LightGBM + tradition"),
        BacktestConfig(model_type="lgbm", exclude_tradition=True, hyperopt_trials=0,
                       lgbm_params=lgbm_params_no_trad, label="LightGBM − tradition"),
    ]

    rows = []
    all_metrics: list[BacktestMetrics] = []
    for cfg in variants:
        print("\n" + "#" * 70)
        _, metrics = run_backtest(cfg, verbose=False)
        all_metrics.append(metrics)
        rows.append({
            "variant": metrics.name,
            "accuracy": metrics.accuracy,
            "log_loss": metrics.log_loss,
            "brier": metrics.brier,
            "correct": metrics.correct,
            "total": metrics.total,
        })

    table = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON (2026 WC walk-forward)")
    print("=" * 70)
    print()
    print(f"  {'Variant':<28s} {'Acc':>7s} {'LogLoss':>9s} {'Brier':>8s} {'Correct':>10s}")
    print("  " + "-" * 65)
    for _, r in table.iterrows():
        print(f"  {r['variant']:<28s} {r['accuracy']:>7.1%} {r['log_loss']:>9.4f} "
              f"{r['brier']:>8.4f} {int(r['correct']):>4d}/{int(r['total']):<4d}")

    best_acc = table.loc[table["accuracy"].idxmax()]
    best_ll = table.loc[table["log_loss"].idxmin()]
    best_brier = table.loc[table["brier"].idxmin()]
    print()
    print(f"  Best accuracy:  {best_acc['variant']} ({best_acc['accuracy']:.1%})")
    print(f"  Best log-loss:  {best_ll['variant']} ({best_ll['log_loss']:.4f})")
    print(f"  Best Brier:     {best_brier['variant']} ({best_brier['brier']:.4f})")

    # Per-stage for best log-loss variant
    best_metrics = next(m for m in all_metrics if m.name == best_ll["variant"])
    stage_names = {0: "Group", 1: "R32", 2: "R16", 3: "QF", 4: "SF", 5: "3rd", 6: "Final"}
    print(f"\n  Per-stage metrics for best log-loss ({best_ll['variant']}):")
    for s in sorted(best_metrics.stage_metrics.keys()):
        sr = best_metrics.stage_metrics[s]
        acc = sr["correct"] / sr["total"] if sr["total"] else 0
        ll = np.mean(sr["log_losses"]) if sr["log_losses"] else 0
        print(f"    {stage_names.get(s, s):<8s}  acc={acc:.1%}  log-loss={ll:.4f}  n={sr['total']}")

    return table


def parse_args():
    parser = argparse.ArgumentParser(description="2026 WC walk-forward backtest")
    parser.add_argument("--compare", action="store_true",
                        help="Compare XGBoost vs LightGBM with/without tradition features")
    parser.add_argument("--model", choices=("xgb", "lgbm"), default=DEFAULT_GBT_MODEL)
    parser.add_argument("--exclude-tradition", action="store_true")
    parser.add_argument("--hyperopt-trials", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        if args.compare:
            run_comparison(hyperopt_trials=args.hyperopt_trials)
        else:
            cfg = BacktestConfig(
                model_type=args.model,
                exclude_tradition=args.exclude_tradition,
                hyperopt_trials=args.hyperopt_trials,
            )
            run_backtest(cfg)
