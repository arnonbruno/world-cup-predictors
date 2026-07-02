#!/usr/bin/env python3
"""Ablation: LGBM baseline vs DL (PyTorch GPU) vs SMOTE/ADASYN augmentation."""
import sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from shared import (
    IsotonicProbabilityCalibrator, sample_weights,
    chronological_train_val_split, fit_dixon_coles, harmonize_country,
    KNOCKOUT_ALPHA, DATA_DIR, GROUP_2026_TEAMS, TRADITION_FEATURE_COLUMNS,
    WC2026_STAGE_TO_TRAIN, apply_match_to_state, blend_probabilities,
    compute_match_features, country_features_for_year, drop_feature_columns,
    finalize_feature_frame, finalize_world_cup_history,
    infer_world_cup_stage_map, load_betting_odds, load_country_feature_history,
    load_squad_values, make_team_state, odds_features_for_match,
    parse_neutral_flag, prepare_prediction_frame,
    wc2026_penalty_winner, wc2026_stage_for_match,
)
from backtest_2026_wc import (
    build_training_matrix, prepare_2026_state, predict_match,
    stage_from_tournament_round, actual_result,
    _compute_metrics, _print_report, BacktestConfig,
    BacktestMetrics, RESULT_LABELS,
)

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
import lightgbm as lgb
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ─── PyTorch MLP Model ─────────────────────────────────────────────────────────

class TorchMLP(nn.Module):
    def __init__(self, n_features, n_classes=3, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchClassifier:
    """Sklearn-compatible wrapper for PyTorch MLP on GPU."""
    def __init__(self, n_features, epochs=200, lr=1e-3, weight_decay=1e-4,
                 hidden=(256, 128, 64), dropout=0.3, batch_size=512):
        self.n_features = n_features
        self.epochs = epochs
        self.lr = lr
        self.wd = weight_decay
        self.hidden = hidden
        self.dropout = dropout
        self.bs = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = None
        self.classes_ = np.array([0, 1, 2])

    def _impute(self, X):
        if self.imputer is not None:
            return self.imputer.transform(np.asarray(X, dtype=np.float32))
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X).astype(np.float32)
        self.model = TorchMLP(self.n_features, 3, self.hidden, self.dropout).to(DEVICE)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xs, device=DEVICE)
        yt = torch.tensor(y, device=DEVICE, dtype=torch.long)
        wt = torch.tensor(np.asarray(sample_weight, dtype=np.float32), device=DEVICE) if sample_weight is not None else None
        n = len(Xt)
        best_vl = float('inf')
        best_state = None
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, self.bs):
                idx = perm[i:i+self.bs]
                xb, yb = Xt[idx], yt[idx]
                logits = self.model(xb)
                if wt is not None:
                    loss = (crit(logits, yb) * wt[idx]).mean()
                else:
                    loss = crit(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    Xv = torch.tensor(self.scaler.transform(np.asarray(X_val, dtype=np.float32)).astype(np.float32), device=DEVICE)
                    yv = torch.tensor(np.asarray(y_val), device=DEVICE, dtype=torch.long)
                    vl = crit(self.model(Xv), yv).item()
                if vl < best_vl:
                    best_vl = vl
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                if ep > 50 and vl > best_vl * 1.5:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_imp = self._impute(X)
        Xs = self.scaler.transform(X_imp).astype(np.float32)
        Xt = torch.tensor(Xs, device=DEVICE)
        with torch.no_grad():
            return torch.softmax(self.model(Xt), dim=1).cpu().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ─── Model Factory ────────────────────────────────────────────────────────────

LGBM_PARAMS = dict(
    objective="multiclass", num_class=3, n_estimators=400,
    learning_rate=0.05, max_depth=5, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.5, random_state=42, verbose=-1,
)

def make_lgbm(X, y, weights, X_val, y_val, w_val):
    """Standard LGBM with isotonic calibration — our baseline."""
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(X, y, sample_weight=weights)
    cal = IsotonicProbabilityCalibrator(m, classes=np.array([0, 1, 2]))
    cal.fit(X_val, y_val, sample_weight=w_val)
    return cal

def make_lgbm_smote(X, y, weights, X_val, y_val, w_val, sampler="smote"):
    """LGBM trained on SMOTE/ADASYN augmented training data."""
    from sklearn.impute import SimpleImputer
    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y)
    # Impute NaNs before SMOTE (SMOTE can't handle NaN)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_np)
    if sampler == "smote":
        s = SMOTE(random_state=42, k_neighbors=5)
    elif sampler == "adasyn":
        s = ADASYN(random_state=42, n_neighbors=5)
    elif sampler == "borderline":
        s = BorderlineSMOTE(random_state=42, k_neighbors=5, kind="borderline-1")
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    X_res, y_res = s.fit_resample(X_imp, y_np)
    print(f"  SMOTE({sampler}): {len(y_np)} → {len(y_res)} samples "
          f"(H:{sum(y_np==0)} D:{sum(y_np==1)} A:{sum(y_np==2)} → "
          f"H:{sum(y_res==0)} D:{sum(y_res==1)} A:{sum(y_res==2)})")
    # LGBM handles NaN natively, so train on the imputed+augmented data
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(X_res, y_res)
    cal = IsotonicProbabilityCalibrator(m, classes=np.array([0, 1, 2]))
    cal.fit(X_val, y_val, sample_weight=w_val)
    return cal

def make_dl(X, y, weights, X_val, y_val, w_val):
    """Deep learning MLP on GPU with isotonic calibration."""
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(np.asarray(X, dtype=np.float32))
    X_val_imp = imp.transform(np.asarray(X_val, dtype=np.float32))
    n_feat = X.shape[1]
    clf = TorchClassifier(n_features=n_feat, epochs=200, lr=1e-3,
                          hidden=(256, 128, 64), dropout=0.3, batch_size=512)
    clf.imputer = imp
    clf.fit(X_imp, y, sample_weight=weights, X_val=X_val_imp, y_val=y_val)
    cal = IsotonicProbabilityCalibrator(clf, classes=np.array([0, 1, 2]))
    cal.fit(X_val, y_val, sample_weight=w_val)
    return cal

def make_dl_smote(X, y, weights, X_val, y_val, w_val, sampler="smote"):
    """DL trained on SMOTE/ADASYN augmented data."""
    from sklearn.impute import SimpleImputer
    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_np)
    if sampler == "smote":
        s = SMOTE(random_state=42, k_neighbors=5)
    elif sampler == "adasyn":
        s = ADASYN(random_state=42, n_neighbors=5)
    else:
        s = BorderlineSMOTE(random_state=42, k_neighbors=5, kind="borderline-1")
    X_res, y_res = s.fit_resample(X_imp, y_np)
    print(f"  DL+{sampler}: {len(y_np)} → {len(y_res)} samples")
    n_feat = X.shape[1]
    X_val_imp = imp.transform(np.asarray(X_val, dtype=np.float32))
    clf = TorchClassifier(n_features=n_feat, epochs=200, lr=1e-3,
                          hidden=(256, 128, 64), dropout=0.3, batch_size=512)
    clf.imputer = imp
    clf.fit(X_res, y_res, X_val=X_val_imp, y_val=y_val)
    cal = IsotonicProbabilityCalibrator(clf, classes=np.array([0, 1, 2]))
    cal.fit(X_val, y_val, sample_weight=w_val)
    return cal


# ─── Backtest Runner ──────────────────────────────────────────────────────────

def run_model_backtest(model, poisson_model, alpha, feature_names,
                      results_df, country_history, odds, squad_values,
                      label="model"):
    """Run walk-forward backtest with a custom model."""
    state = make_team_state()
    # Rebuild state from training data (same as build_training_matrix does)
    X, y, dates, meta, state = build_training_matrix(
        results_df, country_history, odds=odds, squad_values=squad_values)
    state = prepare_2026_state(state, results_df)
    cf = country_features_for_year(country_history, 2026)

    results_df["date"] = pd.to_datetime(results_df["date"])
    wc26 = results_df[
        (results_df["tournament"] == "FIFA World Cup") &
        (results_df["date"].dt.year == 2026)
    ].copy()
    completed = wc26[wc26["home_score"].notna() & wc26["away_score"].notna()].sort_values("date")

    results = []
    for _, r in completed.iterrows():
        home = harmonize_country(r["home_team"])
        away = harmonize_country(r["away_team"])
        hs, aw = int(r["home_score"]), int(r["away_score"])
        date = r["date"]
        stage = stage_from_tournament_round(r.get("tournament", ""), home, away, date=date)
        neutral = parse_neutral_flag(r.get("neutral", True))
        is_home = not neutral
        city = r.get("city") if pd.notna(r.get("city")) else None

        predicted_idx, probs = predict_match(
            model, feature_names, home, away, state, cf, stage, date,
            neutral=neutral, is_home=is_home,
            odds=odds, poisson_model=poisson_model, alpha=alpha,
            squad_values=squad_values, city=city,
        )
        actual_idx = actual_result(hs, aw, home, away, stage)
        is_correct = predicted_idx == actual_idx
        results.append({
            "date": date.strftime("%Y-%m-%d"), "home": home, "away": away,
            "score": f"{hs}-{aw}", "stage": stage,
            "predicted": RESULT_LABELS[predicted_idx],
            "actual": RESULT_LABELS[actual_idx],
            "predicted_idx": predicted_idx, "actual_idx": actual_idx,
            "correct": is_correct,
            "confidence": float(probs[predicted_idx]),
            "actual_prob": float(probs[actual_idx]),
            "p_home": float(probs[0]), "p_draw": float(probs[1]),
            "p_away": float(probs[2]),
        })
        apply_match_to_state(state, home, away, hs, aw, date,
                             neutral=neutral, is_world_cup=True,
                             city=city, is_competitive=True)

    metrics = _compute_metrics(results)
    metrics.name = label
    return results, metrics


# ─── Blend alpha tuning (reuse from backtest) ─────────────────────────────────

def _tune_blend_alpha(model, poisson_model, X_val, y_val, val_meta):
    """Find best LGBM/DC blend weight on validation set."""
    best_alpha, best_loss = 0.5, float("inf")
    probs_gbt = np.asarray(model.predict_proba(X_val), dtype=float)
    for a in np.arange(0.0, 1.01, 0.05):
        total_loss = 0.0
        for i in range(len(y_val)):
            ht, at, neutral = val_meta[i] if i < len(val_meta) else ("", "", True)
            p_dc = poisson_model.outcome_probs(ht, at, neutral=neutral)
            blended = blend_probabilities(probs_gbt[i], p_dc, a)
            p = max(min(blended[y_val[i]], 1-1e-15), 1e-15)
            total_loss += -np.log(p)
        avg = total_loss / max(len(y_val), 1)
        if avg < best_loss:
            best_loss = avg
            best_alpha = a
    return best_alpha, best_loss


# ─── Main Ablation ───────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MODEL ABLATION: LGBM vs DL vs SMOTE/ADASYN")
    print("=" * 70)

    results_df = pd.read_csv(DATA_DIR / "results.csv")
    country_history = load_country_feature_history()
    odds = load_betting_odds()
    squad_values = load_squad_values()

    # Build training matrix once
    print("\nBuilding training matrix...")
    X, y, feature_dates, match_meta, state = build_training_matrix(
        results_df, country_history, odds=odds, squad_values=squad_values)
    print(f"Training matrix: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Label distribution: H={sum(y==0)} D={sum(y==1)} A={sum(y==2)}")

    weights = sample_weights(y, feature_dates)

    # Chronological split for validation
    order = pd.Series(pd.to_datetime(feature_dates, errors="coerce")).sort_values().index
    split = max(1, int(len(order) * 0.8))
    train_idx = order[:split]
    val_idx = order[split:]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train, w_val = weights[train_idx], weights[val_idx]
    val_meta = [match_meta[i] for i in val_idx]

    print(f"Train: {len(y_train)} | Val: {len(y_val)}")

    # Fit Dixon-Coles once
    print("\nFitting Dixon-Coles model...")
    poisson_model = fit_dixon_coles(results_df)

    feature_names = X.columns.tolist()

    # ─── Define experiments ──────────────────────────────────────────────────
    experiments = [
        ("LGBM baseline",         make_lgbm,          {}),
        ("LGBM + SMOTE",          make_lgbm_smote,    {"sampler": "smote"}),
        ("LGBM + ADASYN",         make_lgbm_smote,    {"sampler": "adasyn"}),
        ("LGBM + BorderlineSMOTE",make_lgbm_smote,    {"sampler": "borderline"}),
        ("DL (PyTorch GPU)",      make_dl,            {}),
        ("DL + SMOTE",            make_dl_smote,      {"sampler": "smote"}),
        ("DL + ADASYN",           make_dl_smote,      {"sampler": "adasyn"}),
    ]

    all_results = []

    for name, factory, kwargs in experiments:
        print(f"\n{'─'*60}")
        print(f"  Training: {name}")
        print(f"{'─'*60}")
        t0 = time.time()
        model = factory(X_train, y_train, w_train, X_val, y_val, w_val, **kwargs)
        train_time = time.time() - t0

        # Tune blend alpha
        alpha, blend_loss = _tune_blend_alpha(
            model, poisson_model, X_val, y_val, val_meta)
        print(f"  Blend alpha={alpha:.2f} (val log-loss={blend_loss:.4f})")

        # Run backtest
        t1 = time.time()
        results, metrics = run_model_backtest(
            model, poisson_model, alpha, feature_names,
            results_df, country_history, odds, squad_values, label=name)
        bt_time = time.time() - t1

        stage_names = {0: "Group", 1: "R32", 2: "R16", 3: "QF", 4: "SF", 5: "3rd", 6: "Final"}
        print(f"\n  Results for {name}:")
        print(f"    Accuracy: {metrics.accuracy:.1%} ({metrics.correct}/{metrics.total})")
        print(f"    Log-loss: {metrics.log_loss:.4f}")
        print(f"    Brier:    {metrics.brier:.4f}")
        print(f"    Train: {train_time:.1f}s | Backtest: {bt_time:.1f}s")
        for s in sorted(metrics.stage_metrics.keys()):
            sr = metrics.stage_metrics[s]
            acc = sr["correct"] / sr["total"] if sr["total"] > 0 else 0
            sn = stage_names.get(s, f"S{s}")
            print(f"      {sn:<8}: {sr['correct']}/{sr['total']} ({acc:.0%})")

        all_results.append({
            "name": name, "metrics": metrics, "results": results,
            "train_time": train_time, "bt_time": bt_time,
        })

    # ─── Summary table ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  ABLATION SUMMARY")
    print(f"{'='*80}")
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'LogLoss':>10} {'Brier':>10} {'Train':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for r in all_results:
        m = r["metrics"]
        print(f"  {r['name']:<25} {m.accuracy:>10.1%} {m.log_loss:>10.4f} {m.brier:>10.4f} {r['train_time']:>7.1f}s")

    # Per-stage comparison
    print(f"\n  Per-stage accuracy:")
    stage_names = {0: "Group", 1: "R32", 2: "R16", 3: "QF", 4: "SF", 5: "3rd", 6: "Final"}
    for s in sorted(all_results[0]["metrics"].stage_metrics.keys()):
        sn = stage_names.get(s, f"S{s}")
        vals = []
        for r in all_results:
            sr = r["metrics"].stage_metrics.get(s, {"correct": 0, "total": 0})
            vals.append(f"{sr['correct']}/{sr['total']}")
        print(f"    {sn:<8}: " + "  ".join(f"{v:>5}" for v in vals))

    print()


if __name__ == "__main__":
    main()


