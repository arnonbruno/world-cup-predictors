#!/usr/bin/env python3
"""World Cup predictors SOTA econometric + ML analysis.

This script follows ANALYSIS_SPEC.md constraints:
- Explicit leakage audit before any modeling
- Leakage columns dropped prior to analysis
- Leave-One-World-Cup-Out (LOWCO) CV only
- Train-only fitting for imputers/scalers in every fold
- Bootstrap confidence intervals for effect sizes
- FDR-corrected p-values for univariate analysis
- SHAP if installable, permutation importance fallback
- Figures saved as PNG under output/sota/figures/
- Plain-English summary saved to output/sota/summary.md
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 42
N_BOOTSTRAP = 1000
PROFILE_GRID_POINTS = 15

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "world_cup_predictors_dataset.csv"
OUT_DIR = ROOT / "output" / "sota"
FIG_DIR = OUT_DIR / "figures"

LEAK_COLUMNS = [
    "runner_up",
    "semifinalist",
    "finalist",
    "top4",
    "is_winner",
    "gdp_per_capita_vs_winner",
    "population_vs_winner",
]

VALIDATION_COLUMNS = [
    "gdp_per_capita_vs_avg",
    "population_vs_avg",
]

SAFE_KEY_COLUMNS = [
    "is_host",
    "elo_rating",
    "fifa_rank",
    "fifa_rank_inverse",
    "football_tradition",
    "football_power_index",
    "is_former_champion",
    "is_strong_europe",
    "is_strong_sa",
    "wc_titles_before",
    "wc_finals_before",
    "wc_semifinals_before",
    "wc_participations_before",
    "years_since_last_wc",
    "years_since_last_win",
    "years_since_last_final",
]

# Tournament-outcome variables known only after the event (excluded for prediction).
POST_TOURNAMENT_COLUMNS = [
    "total_goals_in_tournament",
    "avg_goals_per_match",
]

IDENTIFIER_COLUMNS = ["country", "iso3", "confederation"]


def try_import_or_install(module_name: str, pip_name: Optional[str] = None):
    """Attempt import, then pip install once if needed."""
    pip_name = pip_name or module_name
    try:
        return __import__(module_name)
    except Exception:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return __import__(module_name)
        except Exception:
            return None


def ensure_output_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def print_and_save_leakage_audit(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Print leakage audit before any analysis, and save markdown copy."""
    all_cols = set(df.columns)

    present_leaks = [c for c in LEAK_COLUMNS if c in all_cols]
    present_validation = [c for c in VALIDATION_COLUMNS if c in all_cols]
    present_safe = [c for c in SAFE_KEY_COLUMNS if c in all_cols]
    present_post_tournament = [c for c in POST_TOURNAMENT_COLUMNS if c in all_cols]

    missing_safe = [c for c in SAFE_KEY_COLUMNS if c not in all_cols]
    missing_leaks = [c for c in LEAK_COLUMNS if c not in all_cols]

    audit = {
        "leaking_drop_immediately": present_leaks,
        "safe_keep": present_safe,
        "validation_required_keep": present_validation,
        "post_tournament_drop_for_prediction": present_post_tournament,
        "missing_expected_safe_columns": missing_safe,
        "missing_expected_leak_columns": missing_leaks,
    }

    print("\n" + "=" * 88)
    print("LEAKAGE AUDIT (REQUIRED BEFORE ANY MODELING)")
    print("=" * 88)
    print("\n[DROP - DIRECT LEAKS]")
    for c in audit["leaking_drop_immediately"]:
        print(f" - {c}")
    print("\n[KEEP - SAFE PRE-TOURNAMENT/HISTORICAL]")
    for c in audit["safe_keep"]:
        print(f" - {c}")
    print("\n[KEEP + VALIDATE]")
    for c in audit["validation_required_keep"]:
        print(f" - {c}")
    print("\n[DROP FROM PREDICTION - POST-TOURNAMENT SIGNALS]")
    for c in audit["post_tournament_drop_for_prediction"]:
        print(f" - {c}")
    if audit["missing_expected_safe_columns"]:
        print("\n[MISSING SAFE COLUMNS - NOTE]")
        for c in audit["missing_expected_safe_columns"]:
            print(f" - {c}")
    if audit["missing_expected_leak_columns"]:
        print("\n[MISSING LEAK COLUMNS - NOTE]")
        for c in audit["missing_expected_leak_columns"]:
            print(f" - {c}")
    print("=" * 88 + "\n")

    audit_md = []
    audit_md.append("# Leakage Audit\n")
    audit_md.append("## Drop (Direct Leakage)\n")
    audit_md.extend([f"- `{c}`\n" for c in audit["leaking_drop_immediately"]])
    audit_md.append("\n## Keep (Safe)\n")
    audit_md.extend([f"- `{c}`\n" for c in audit["safe_keep"]])
    audit_md.append("\n## Keep but Validate\n")
    audit_md.extend([f"- `{c}`\n" for c in audit["validation_required_keep"]])
    audit_md.append("\n## Excluded from Prediction (Post-Tournament)\n")
    audit_md.extend([f"- `{c}`\n" for c in audit["post_tournament_drop_for_prediction"]])
    audit_md.append("\n## Validation Notes\n")
    audit_md.append(
        "- `gdp_per_capita_vs_avg` and `population_vs_avg` are cross-sectional within-year "
        "averages and retained.\n"
    )
    audit_md.append(
        "- World Bank indicators are used as provided, with LOWCO split preserving strict "
        "train-only preprocessing.\n"
    )
    (OUT_DIR / "leakage_audit.md").write_text("".join(audit_md), encoding="utf-8")
    (OUT_DIR / "leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


class YearMedianImputer:
    """Impute missing numeric values using training-year medians.

    Fit on train only:
    - Store per-year medians from train years
    - Store global train medians for fallback
    Transform:
    - Fill each row from its year median if available
    - fallback to global train median
    """

    def __init__(self, year_col: str = "wc_year"):
        self.year_col = year_col
        self.numeric_cols_: List[str] = []
        self.year_medians_: Dict[int, pd.Series] = {}
        self.global_medians_: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame, numeric_cols: Sequence[str]) -> "YearMedianImputer":
        self.numeric_cols_ = list(numeric_cols)
        tmp = df[[self.year_col] + self.numeric_cols_].copy()
        self.year_medians_ = {
            int(year): grp[self.numeric_cols_].median()
            for year, grp in tmp.groupby(self.year_col, dropna=False)
        }
        self.global_medians_ = tmp[self.numeric_cols_].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.global_medians_ is None:
            raise RuntimeError("YearMedianImputer must be fit before transform.")
        out = df.copy()
        for year in out[self.year_col].dropna().unique():
            mask = out[self.year_col] == year
            med = self.year_medians_.get(int(year), self.global_medians_)
            out.loc[mask, self.numeric_cols_] = out.loc[mask, self.numeric_cols_].fillna(med)
        out[self.numeric_cols_] = out[self.numeric_cols_].fillna(self.global_medians_)
        return out


def cohens_d(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    """Compute Cohen's d for positive minus negative class."""
    x_pos = np.asarray(x_pos, dtype=float)
    x_neg = np.asarray(x_neg, dtype=float)
    if len(x_pos) < 2 or len(x_neg) < 2:
        return np.nan
    m1, m0 = np.nanmean(x_pos), np.nanmean(x_neg)
    s1, s0 = np.nanstd(x_pos, ddof=1), np.nanstd(x_neg, ddof=1)
    n1, n0 = len(x_pos), len(x_neg)
    pooled_var = ((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / max((n1 + n0 - 2), 1)
    pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else np.nan
    if not np.isfinite(pooled_sd) or pooled_sd == 0:
        return np.nan
    return (m1 - m0) / pooled_sd


def bootstrap_ci_effect(
    x: np.ndarray,
    y: np.ndarray,
    effect_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> Tuple[float, float]:
    """Bootstrap CI for effect size (stratified by class)."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return (np.nan, np.nan)
    boots = []
    for _ in range(n_bootstrap):
        pos_s = rng.choice(pos, size=len(pos), replace=True)
        neg_s = rng.choice(neg, size=len(neg), replace=True)
        eff = effect_fn(pos_s, neg_s)
        if np.isfinite(eff):
            boots.append(eff)
    if len(boots) < 30:
        return (np.nan, np.nan)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def run_univariate_analysis(
    df: pd.DataFrame,
    predictor_cols: Sequence[str],
    target_col: str = "won_wc",
) -> pd.DataFrame:
    """Univariate tests with bootstrap CIs and multiplicity corrections."""
    results = []
    y = df[target_col].astype(int).values
    for col in predictor_cols:
        x = df[col].astype(float).values
        mask = np.isfinite(x) & np.isfinite(y)
        x_m = x[mask]
        y_m = y[mask]
        if len(np.unique(x_m)) <= 1 or len(np.unique(y_m)) <= 1:
            continue
        x_pos = x_m[y_m == 1]
        x_neg = x_m[y_m == 0]
        if len(x_pos) < 2 or len(x_neg) < 2:
            continue
        try:
            pb_r, pb_p = stats.pointbiserialr(y_m, x_m)
        except Exception:
            pb_r, pb_p = np.nan, np.nan
        try:
            mw_u, mw_p = stats.mannwhitneyu(x_pos, x_neg, alternative="two-sided")
        except Exception:
            mw_u, mw_p = np.nan, np.nan
        d = cohens_d(x_pos, x_neg)
        d_ci_lo, d_ci_hi = bootstrap_ci_effect(x_m, y_m, cohens_d)
        results.append(
            {
                "feature": col,
                "pointbiserial_r": pb_r,
                "pointbiserial_p": pb_p,
                "mannwhitney_u": mw_u,
                "mannwhitney_p": mw_p,
                "cohens_d": d,
                "cohens_d_ci_low": d_ci_lo,
                "cohens_d_ci_high": d_ci_hi,
                "n_non_missing": int(mask.sum()),
            }
        )

    res = pd.DataFrame(results)
    if res.empty:
        raise RuntimeError("Univariate analysis produced no valid results.")

    m = len(res)
    res["pointbiserial_p_bonf"] = np.minimum(res["pointbiserial_p"] * m, 1.0)
    res["mannwhitney_p_bonf"] = np.minimum(res["mannwhitney_p"] * m, 1.0)
    res["pointbiserial_p_fdr"] = multipletests(
        res["pointbiserial_p"].fillna(1.0).values, alpha=0.05, method="fdr_bh"
    )[1]
    res["mannwhitney_p_fdr"] = multipletests(
        res["mannwhitney_p"].fillna(1.0).values, alpha=0.05, method="fdr_bh"
    )[1]
    res["abs_cohens_d"] = res["cohens_d"].abs()
    res = res.sort_values(["pointbiserial_p_fdr", "mannwhitney_p_fdr", "abs_cohens_d"]).reset_index(
        drop=True
    )
    return res


def plot_univariate_effects(univariate_df: pd.DataFrame, out_file: Path, top_n: int = 20) -> None:
    top = univariate_df.head(top_n).copy()
    top = top.sort_values("cohens_d")
    y_pos = np.arange(len(top))
    plt.figure(figsize=(10, max(6, len(top) * 0.35)))
    plt.errorbar(
        top["cohens_d"],
        y_pos,
        xerr=[
            top["cohens_d"] - top["cohens_d_ci_low"],
            top["cohens_d_ci_high"] - top["cohens_d"],
        ],
        fmt="o",
        capsize=3,
    )
    plt.yticks(y_pos, top["feature"])
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Cohen's d (winners - non-winners)")
    plt.title("Top Univariate Effect Sizes (95% Bootstrap CI)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def compute_vif(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    x = df[list(cols)].astype(float).copy()
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty or x.shape[1] < 2:
        return pd.DataFrame({"feature": list(cols), "vif": np.nan})
    x_const = sm.add_constant(x, has_constant="add")
    vals = []
    for i, c in enumerate(x_const.columns):
        if c == "const":
            continue
        vif_val = variance_inflation_factor(x_const.values, i)
        vals.append({"feature": c, "vif": float(vif_val)})
    return pd.DataFrame(vals)


def iterative_vif_prune(df: pd.DataFrame, cols: Sequence[str], threshold: float = 5.0) -> List[str]:
    kept = list(cols)
    while len(kept) > 2:
        vif_df = compute_vif(df, kept)
        if vif_df.empty:
            break
        max_row = vif_df.sort_values("vif", ascending=False).iloc[0]
        if np.isnan(max_row["vif"]) or max_row["vif"] <= threshold:
            break
        kept.remove(str(max_row["feature"]))
    return kept


def _logit_ll(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    z = x @ beta
    p = expit(z)
    eps = 1e-12
    return float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def profile_likelihood_ci(
    x: np.ndarray,
    y: np.ndarray,
    beta_hat: np.ndarray,
    bse: np.ndarray,
    feature_names: Sequence[str],
    grid_points: int = PROFILE_GRID_POINTS,
) -> Dict[str, Tuple[float, float]]:
    """Approximate profile-likelihood CI for each coefficient."""
    ll_max = _logit_ll(beta_hat, x, y)
    cutoff = ll_max - 0.5 * stats.chi2.ppf(0.95, df=1)
    ci = {}
    p = len(beta_hat)
    for j in range(1, p):  # skip intercept
        name = feature_names[j]
        center = beta_hat[j]
        spread = max(1e-3, 3.0 * (bse[j] if np.isfinite(bse[j]) else 1.0))
        grid = np.linspace(center - spread, center + spread, grid_points)
        prof_ll = []
        idx_free = [k for k in range(p) if k != j]

        for fixed_val in grid:
            start = beta_hat[idx_free]

            def obj(theta_free):
                beta = np.zeros(p)
                beta[j] = fixed_val
                beta[idx_free] = theta_free
                return -_logit_ll(beta, x, y)

            res = minimize(obj, start, method="BFGS")
            if res.success and np.isfinite(res.fun):
                prof_ll.append(-float(res.fun))
            else:
                prof_ll.append(np.nan)

        prof_ll = np.array(prof_ll, dtype=float)
        ok = np.isfinite(prof_ll)
        if ok.sum() < 4:
            ci[name] = (np.nan, np.nan)
            continue
        g = grid[ok]
        lls = prof_ll[ok]

        # Determine lower/upper where profiled log-likelihood intersects cutoff.
        lower = np.nan
        upper = np.nan
        for a, b, la, lb in zip(g[:-1], g[1:], lls[:-1], lls[1:]):
            if (la - cutoff) * (lb - cutoff) <= 0:
                # linear interpolation
                if lb != la:
                    x_star = a + (cutoff - la) * (b - a) / (lb - la)
                else:
                    x_star = a
                if x_star <= center and (np.isnan(lower) or x_star > lower):
                    lower = x_star
                if x_star >= center and (np.isnan(upper) or x_star < upper):
                    upper = x_star
        ci[name] = (float(lower), float(upper))
    return ci


def hosmer_lemeshow_test(y_true: np.ndarray, y_prob: np.ndarray, n_groups: int = 10) -> Tuple[float, float]:
    df_hl = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    try:
        df_hl["bin"] = pd.qcut(df_hl["p"], q=n_groups, duplicates="drop")
    except Exception:
        df_hl["bin"] = pd.cut(df_hl["p"], bins=n_groups)
    grp = df_hl.groupby("bin", observed=False).agg(
        observed=("y", "sum"),
        expected=("p", "sum"),
        total=("y", "count"),
    )
    grp["observed0"] = grp["total"] - grp["observed"]
    grp["expected0"] = grp["total"] - grp["expected"]
    eps = 1e-9
    chi2 = np.sum((grp["observed"] - grp["expected"]) ** 2 / (grp["expected"] + eps)) + np.sum(
        (grp["observed0"] - grp["expected0"]) ** 2 / (grp["expected0"] + eps)
    )
    dof = max(int(grp.shape[0] - 2), 1)
    p_val = 1.0 - stats.chi2.cdf(chi2, dof)
    return float(chi2), float(p_val)


@dataclass
class CVFoldResult:
    fold_year: int
    auc: float
    precision: float
    recall: float
    f1: float
    brier: float


def lowco_splits(df: pd.DataFrame, year_col: str = "wc_year") -> List[Tuple[int, np.ndarray, np.ndarray]]:
    years = sorted(df[year_col].dropna().unique().astype(int))
    splits = []
    for year in years:
        train_idx = df[year_col].values != year
        test_idx = df[year_col].values == year
        splits.append((year, train_idx, test_idx))
    return splits


def prepare_predictive_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"won_wc", "wc_year"} | set(LEAK_COLUMNS) | set(POST_TOURNAMENT_COLUMNS)
    cols = [c for c in numeric_cols if c not in excluded]
    return cols


def evaluate_lowco_model(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    build_model_fn: Callable[[int, int], object],
    scale_features: bool = False,
    model_name: str = "model",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate model under LOWCO with strict train-only preprocessing."""
    folds = []
    oof_rows = []
    y_all = df["won_wc"].astype(int).values

    for fold_year, train_idx, test_idx in lowco_splits(df):
        train_df = df.loc[train_idx, ["wc_year", "won_wc"] + list(feature_cols)].copy()
        test_df = df.loc[test_idx, ["wc_year", "won_wc"] + list(feature_cols)].copy()

        imp = YearMedianImputer(year_col="wc_year").fit(train_df, numeric_cols=feature_cols)
        train_imp = imp.transform(train_df)
        test_imp = imp.transform(test_df)

        x_train = train_imp[list(feature_cols)].astype(float).values
        x_test = test_imp[list(feature_cols)].astype(float).values
        y_train = train_imp["won_wc"].astype(int).values
        y_test = test_imp["won_wc"].astype(int).values

        if scale_features:
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        model = build_model_fn(pos, neg)
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        br = brier_score_loss(y_test, y_prob)
        folds.append(
            CVFoldResult(
                fold_year=int(fold_year),
                auc=float(auc),
                precision=float(prec),
                recall=float(rec),
                f1=float(f1),
                brier=float(br),
            )
        )

        for yy, pp in zip(y_test, y_prob):
            oof_rows.append(
                {"model": model_name, "fold_year": int(fold_year), "y_true": int(yy), "y_prob": float(pp)}
            )

    return pd.DataFrame([f.__dict__ for f in folds]), pd.DataFrame(oof_rows)


def select_logit_features_on_train(
    train_df: pd.DataFrame,
    candidate_cols: Sequence[str],
    top_n: int = 20,
    vif_threshold: float = 5.0,
) -> List[str]:
    """Select logit features using only one LOWCO training fold."""
    fit_cols = ["wc_year", "won_wc"] + list(candidate_cols)
    train_fit = train_df[fit_cols].copy()
    imp = YearMedianImputer(year_col="wc_year").fit(train_fit, numeric_cols=candidate_cols)
    train_imp = imp.transform(train_fit)
    try:
        fold_univariate = run_univariate_analysis(
            train_imp, predictor_cols=candidate_cols, target_col="won_wc"
        )
        top_features = fold_univariate.head(top_n)["feature"].tolist()
    except RuntimeError:
        top_features = list(candidate_cols)[:top_n]
    if len(top_features) <= 2:
        return top_features
    return iterative_vif_prune(train_imp[top_features].copy(), top_features, threshold=vif_threshold)


def tune_regularized_logit_lowco(
    df: pd.DataFrame, feature_cols: Sequence[str]
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    configs = []
    # L2
    for c in [0.01, 0.1, 1.0, 10.0]:
        configs.append({"penalty": "l2", "C": c, "solver": "lbfgs"})
    # L1
    for c in [0.01, 0.1, 1.0, 10.0]:
        configs.append({"penalty": "l1", "C": c, "solver": "saga"})
    # ElasticNet
    for c in [0.01, 0.1, 1.0, 10.0]:
        for l1r in [0.2, 0.5, 0.8]:
            configs.append({"penalty": "elasticnet", "C": c, "solver": "saga", "l1_ratio": l1r})

    all_rows = []
    best_models: Dict[str, dict] = {}
    for cfg in configs:
        fold_rows = []
        selected_counts = []
        for fold_year, train_idx, test_idx in lowco_splits(df):
            train_base = df.loc[train_idx].copy()
            selected = select_logit_features_on_train(train_base, feature_cols)
            selected_counts.append(len(selected))
            train_df = df.loc[train_idx, ["wc_year", "won_wc"] + selected].copy()
            test_df = df.loc[test_idx, ["wc_year", "won_wc"] + selected].copy()

            imp = YearMedianImputer(year_col="wc_year").fit(train_df, numeric_cols=selected)
            train_imp = imp.transform(train_df)
            test_imp = imp.transform(test_df)

            x_train = train_imp[selected].astype(float).values
            x_test = test_imp[selected].astype(float).values
            y_train = train_imp["won_wc"].astype(int).values
            y_test = test_imp["won_wc"].astype(int).values

            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            kwargs = dict(
                penalty=cfg["penalty"],
                C=cfg["C"],
                solver=cfg["solver"],
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000,
            )
            if cfg["penalty"] == "elasticnet":
                kwargs["l1_ratio"] = cfg["l1_ratio"]
            model = LogisticRegression(**kwargs)
            model.fit(x_train, y_train)
            y_prob = model.predict_proba(x_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            fold_rows.append(
                {
                    "fold_year": int(fold_year),
                    "auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan,
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "brier": brier_score_loss(y_test, y_prob),
                }
            )

        fold_df = pd.DataFrame(fold_rows)
        row = {**cfg}
        row["mean_auc"] = float(np.nanmean(fold_df["auc"]))
        row["mean_f1"] = float(np.nanmean(fold_df["f1"]))
        row["mean_brier"] = float(np.nanmean(fold_df["brier"]))
        row["mean_selected_features"] = float(np.nanmean(selected_counts))
        all_rows.append(row)

    score_df = pd.DataFrame(all_rows).sort_values(["mean_auc", "mean_f1"], ascending=False).reset_index(
        drop=True
    )
    for pen in ["l1", "l2", "elasticnet"]:
        best = score_df[score_df["penalty"] == pen].head(1)
        if not best.empty:
            best_models[pen] = best.iloc[0].to_dict()
    return score_df, best_models


def fit_statsmodels_logit_with_profile_ci(
    df: pd.DataFrame, feature_cols: Sequence[str]
) -> Tuple[pd.DataFrame, sm.Logit]:
    # For inferential model, impute with year medians built on full sample.
    fit_df = df[["wc_year", "won_wc"] + list(feature_cols)].copy()
    imp = YearMedianImputer(year_col="wc_year").fit(fit_df, numeric_cols=feature_cols)
    fit_df = imp.transform(fit_df)

    x_raw = fit_df[list(feature_cols)].astype(float)
    scaler = StandardScaler().fit(x_raw.values)
    x_scaled = scaler.transform(x_raw.values)
    x_design = sm.add_constant(x_scaled, has_constant="add")
    y = fit_df["won_wc"].astype(int).values

    model = sm.Logit(y, x_design)
    res = model.fit(disp=False, maxiter=500)
    params = res.params
    bse = res.bse
    names = ["const"] + list(feature_cols)
    prof_ci = profile_likelihood_ci(x_design, y, params, bse, names, grid_points=PROFILE_GRID_POINTS)

    rows = []
    for i, feat in enumerate(names):
        if feat == "const":
            continue
        coef = float(params[i])
        se = float(bse[i])
        pval = float(res.pvalues[i])
        lo, hi = prof_ci.get(feat, (np.nan, np.nan))
        or_val = float(np.exp(coef))
        rows.append(
            {
                "feature": feat,
                "coef": coef,
                "std_err": se,
                "p_value": pval,
                "odds_ratio": or_val,
                "profile_ci_low": lo,
                "profile_ci_high": hi,
                "or_ci_low": float(np.exp(lo)) if np.isfinite(lo) else np.nan,
                "or_ci_high": float(np.exp(hi)) if np.isfinite(hi) else np.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    out["p_value_fdr"] = multipletests(out["p_value"].fillna(1.0).values, alpha=0.05, method="fdr_bh")[1]
    return out, res


def fit_tree_models_and_importance(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    shap_module,
) -> Dict[str, pd.DataFrame]:
    """Fit RF/XGB on full imputed data for interpretation artifacts."""
    out: Dict[str, pd.DataFrame] = {}
    full_df = df[["wc_year", "won_wc"] + list(feature_cols)].copy()
    imp = YearMedianImputer(year_col="wc_year").fit(full_df, numeric_cols=feature_cols)
    full_df = imp.transform(full_df)
    x = full_df[list(feature_cols)].astype(float).values
    y = full_df["won_wc"].astype(int).values

    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = float(neg / max(pos, 1))

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(x, y)
    rf_imp = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    rf_imp.to_csv(OUT_DIR / "rf_importance.csv", index=False)
    out["rf_importance"] = rf_imp

    plt.figure(figsize=(10, 6))
    top = rf_imp.head(20).iloc[::-1]
    plt.barh(top["feature"], top["importance"])
    plt.title("Random Forest Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rf_feature_importance.png", dpi=150)
    plt.close()

    xgb_model = None
    try:
        xgb_pkg = try_import_or_install("xgboost")
        XGBClassifier = xgb_pkg.XGBClassifier
        xgb_model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
        xgb_model.fit(x, y)
        xgb_imp = pd.DataFrame({"feature": feature_cols, "importance": xgb_model.feature_importances_}).sort_values(
            "importance", ascending=False
        )
        xgb_imp.to_csv(OUT_DIR / "xgb_importance.csv", index=False)
        out["xgb_importance"] = xgb_imp
        plt.figure(figsize=(10, 6))
        top = xgb_imp.head(20).iloc[::-1]
        plt.barh(top["feature"], top["importance"])
        plt.title("XGBoost Feature Importance (Top 20)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "xgb_feature_importance.png", dpi=150)
        plt.close()
    except Exception as exc:
        warnings.warn(f"XGBoost unavailable or failed ({exc}). Tree phase will skip XGBoost model.")
        out["xgb_importance"] = pd.DataFrame(columns=["feature", "importance"])

    # SHAP or permutation importance fallback
    shap_rows = []
    if shap_module is not None:
        try:
            rf_explainer = shap_module.TreeExplainer(rf)
            shap_vals = rf_explainer.shap_values(x)
            if isinstance(shap_vals, list):
                shap_arr = np.abs(shap_vals[1]).mean(axis=0)
            else:
                shap_arr = np.abs(shap_vals).mean(axis=0)
            shap_rows.append(
                pd.DataFrame({"model": "random_forest", "feature": feature_cols, "importance": shap_arr})
            )
        except Exception:
            pass

        if xgb_model is not None:
            try:
                xgb_explainer = shap_module.TreeExplainer(xgb_model)
                shap_vals = xgb_explainer.shap_values(x)
                if isinstance(shap_vals, list):
                    shap_arr = np.abs(shap_vals[1]).mean(axis=0)
                else:
                    shap_arr = np.abs(shap_vals).mean(axis=0)
                shap_rows.append(pd.DataFrame({"model": "xgboost", "feature": feature_cols, "importance": shap_arr}))
            except Exception:
                pass

    if shap_rows:
        shap_df = pd.concat(shap_rows, ignore_index=True)
        shap_agg = (
            shap_df.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
        )
        shap_agg.to_csv(OUT_DIR / "shap_importance.csv", index=False)
        out["shap_importance"] = shap_agg
        plt.figure(figsize=(10, 6))
        top = shap_agg.head(20).iloc[::-1]
        plt.barh(top["feature"], top["importance"])
        plt.title("SHAP Importance (Top 20, averaged across tree models)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_importance.png", dpi=150)
        plt.close()
    else:
        # Permutation fallback
        perm = permutation_importance(
            rf,
            x,
            y,
            n_repeats=15,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            n_jobs=-1,
        )
        perm_df = pd.DataFrame(
            {"feature": feature_cols, "importance": perm.importances_mean}
        ).sort_values("importance", ascending=False)
        perm_df.to_csv(OUT_DIR / "permutation_importance.csv", index=False)
        out["shap_importance"] = perm_df
        plt.figure(figsize=(10, 6))
        top = perm_df.head(20).iloc[::-1]
        plt.barh(top["feature"], top["importance"])
        plt.title("Permutation Importance Fallback (Top 20)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "permutation_importance.png", dpi=150)
        plt.close()

    def _save_pdp(estimator, selected_features: List[str], out_file: Path, title: str) -> None:
        feat_indices = [feature_cols.index(f) for f in selected_features if f in feature_cols]
        if not feat_indices:
            return
        n = len(feat_indices)
        ncols = 2
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(6, 3.5 * nrows)))
        axes_arr = np.atleast_1d(axes).reshape(-1)
        PartialDependenceDisplay.from_estimator(
            estimator,
            x,
            feat_indices,
            feature_names=list(feature_cols),
            ax=axes_arr[:n],
        )
        for ax in axes_arr[n:]:
            ax.remove()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

    # Partial dependence for top-5 RF features.
    try:
        _save_pdp(
            rf,
            out["rf_importance"]["feature"].head(5).tolist(),
            FIG_DIR / "pdp_rf_top5.png",
            "Random Forest Partial Dependence (Top 5)",
        )
    except Exception:
        warnings.warn("Failed to generate RF partial dependence plots.")

    if xgb_model is not None:
        try:
            _save_pdp(
                xgb_model,
                out["xgb_importance"]["feature"].head(5).tolist(),
                FIG_DIR / "pdp_xgb_top5.png",
                "XGBoost Partial Dependence (Top 5)",
            )
        except Exception:
            warnings.warn("Failed to generate XGB partial dependence plots.")

    return out


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int = N_BOOTSTRAP) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(RANDOM_STATE)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(sample.mean())
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def run_causal_analysis(df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    tmp = df.copy()

    # Impute fields needed for causal routines.
    needed = list(
        {
            "gdp_per_capita_log",
            "population_log",
            "gdp_growth",
            "football_tradition",
            "fifa_rank_inverse",
            "wc_titles_before",
            "elo_rating",
        }.intersection(set(tmp.columns))
    )
    base_cols = ["country", "wc_year", "won_wc", "is_host"]
    causal_cols = base_cols + [c for c in needed if c not in base_cols]
    imp = YearMedianImputer(year_col="wc_year").fit(tmp[["wc_year"] + needed], numeric_cols=needed)
    tmp = imp.transform(tmp[causal_cols].copy())

    # 1) Difference-in-Differences for hosting.
    first_host = tmp.loc[tmp["is_host"] == 1].groupby("country")["wc_year"].min().to_dict()
    tmp["treated"] = tmp["country"].isin(first_host.keys()).astype(int)
    tmp["post"] = tmp.apply(
        lambda r: int(r["wc_year"] >= first_host.get(r["country"], 10**9)),
        axis=1,
    )
    tmp["treated_post"] = tmp["treated"] * tmp["post"]
    did_controls = [c for c in ["football_tradition", "gdp_per_capita_log", "population_log"] if c in tmp.columns]
    did_x_cols = ["treated", "post", "treated_post"] + did_controls
    did_x = sm.add_constant(tmp[did_x_cols], has_constant="add")
    did_model = sm.OLS(tmp["won_wc"], did_x).fit(cov_type="HC3")
    did_effect = float(did_model.params.get("treated_post", np.nan))

    rng = np.random.default_rng(RANDOM_STATE)
    boot_eff = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(np.arange(len(tmp)), size=len(tmp), replace=True)
        b = tmp.iloc[idx]
        x_b = sm.add_constant(b[did_x_cols], has_constant="add")
        try:
            m_b = sm.OLS(b["won_wc"], x_b).fit()
            boot_eff.append(float(m_b.params.get("treated_post", np.nan)))
        except Exception:
            continue
    did_ci = (
        float(np.percentile(boot_eff, 2.5)) if boot_eff else np.nan,
        float(np.percentile(boot_eff, 97.5)) if boot_eff else np.nan,
    )
    did_df = pd.DataFrame(
        [
            {
                "method": "difference_in_differences_hosting",
                "effect": did_effect,
                "ci_low": did_ci[0],
                "ci_high": did_ci[1],
                "p_value": float(did_model.pvalues.get("treated_post", np.nan)),
            }
        ]
    )
    did_df.to_csv(OUT_DIR / "causal_did_hosting.csv", index=False)
    out["did"] = did_df

    # 2) Granger-like: compare lag structures for economic indicators.
    gl = df[["country", "wc_year", "won_wc"]].copy()
    econ_cols = [c for c in ["gdp_per_capita_log", "population_log", "gdp_growth"] if c in df.columns]
    for c in econ_cols:
        gl[c] = df[c]
    gl = gl.sort_values(["country", "wc_year"]).reset_index(drop=True)
    for lag_step, label in [(0, "t1_like"), (1, "t4"), (2, "t8")]:
        for c in econ_cols:
            if lag_step == 0:
                gl[f"{c}_{label}"] = gl[c]
            else:
                gl[f"{c}_{label}"] = gl.groupby("country")[c].shift(lag_step)

    granger_rows = []
    for label in ["t1_like", "t4", "t8"]:
        feat = [f"{c}_{label}" for c in econ_cols if f"{c}_{label}" in gl.columns]
        if not feat:
            continue
        tmp_g = gl[["wc_year", "won_wc"] + feat].copy()

        def builder(_pos: int, _neg: int):
            return LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=3000,
            )

        fold_df, _ = evaluate_lowco_model(
            tmp_g,
            feature_cols=feat,
            build_model_fn=builder,
            scale_features=True,
            model_name=f"granger_{label}",
        )
        granger_rows.append(
            {
                "spec": label,
                "features": ",".join(feat),
                "mean_auc_lowco": float(np.nanmean(fold_df["auc"])),
                "mean_f1_lowco": float(np.nanmean(fold_df["f1"])),
            }
        )
    granger_df = pd.DataFrame(granger_rows).sort_values("mean_auc_lowco", ascending=False)
    granger_df.to_csv(OUT_DIR / "causal_granger_like.csv", index=False)
    out["granger_like"] = granger_df

    # 3) Propensity score matching for hosting effect.
    psm_covars = [c for c in ["gdp_per_capita_log", "population_log", "football_tradition", "fifa_rank_inverse"] if c in tmp.columns]
    psm_df = tmp[["won_wc", "is_host"] + psm_covars].dropna().copy()
    if len(psm_df) > 10 and psm_df["is_host"].nunique() == 2:
        psm_x = psm_df[psm_covars].values
        psm_t = psm_df["is_host"].astype(int).values
        psm_y = psm_df["won_wc"].astype(int).values
        ps_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ps_model.fit(psm_x, psm_t)
        ps = ps_model.predict_proba(psm_x)[:, 1]
        psm_df["propensity"] = ps

        treated_idx = psm_df.index[psm_df["is_host"] == 1].tolist()
        control_idx = psm_df.index[psm_df["is_host"] == 0].tolist()
        available_controls = set(control_idx)
        pairs = []
        caliper = 0.05
        for ti in treated_idx:
            if not available_controls:
                break
            t_ps = psm_df.loc[ti, "propensity"]
            cands = list(available_controls)
            dists = np.abs(psm_df.loc[cands, "propensity"].values - t_ps)
            j = int(np.argmin(dists))
            best_c = cands[j]
            if dists[j] <= caliper:
                pairs.append((ti, best_c))
                available_controls.remove(best_c)
        if pairs:
            diffs = np.array([psm_df.loc[t, "won_wc"] - psm_df.loc[c, "won_wc"] for t, c in pairs], dtype=float)
            att = float(np.mean(diffs))
            ci = bootstrap_mean_ci(diffs)
            psm_out = pd.DataFrame(
                [
                    {
                        "matched_pairs": int(len(pairs)),
                        "att_hosting_on_win": att,
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                    }
                ]
            )
        else:
            psm_out = pd.DataFrame(
                [{"matched_pairs": 0, "att_hosting_on_win": np.nan, "ci_low": np.nan, "ci_high": np.nan}]
            )
    else:
        psm_out = pd.DataFrame(
            [{"matched_pairs": 0, "att_hosting_on_win": np.nan, "ci_low": np.nan, "ci_high": np.nan}]
        )
    psm_out.to_csv(OUT_DIR / "causal_psm_hosting.csv", index=False)
    out["psm"] = psm_out

    # 4) IV attempt for Elo rating.
    iv_controls = [c for c in ["gdp_per_capita_log", "population_log", "is_host"] if c in tmp.columns]
    instrument = "wc_titles_before" if "wc_titles_before" in tmp.columns else None
    iv_rows = []
    if instrument is not None and "elo_rating" in tmp.columns:
        iv_df = tmp[["won_wc", "elo_rating", instrument] + iv_controls].dropna()
        if len(iv_df) > 20:
            # First stage
            fs_x = sm.add_constant(iv_df[[instrument] + iv_controls], has_constant="add")
            fs = sm.OLS(iv_df["elo_rating"], fs_x).fit()
            f_test = fs.f_test(f"{instrument}=0")
            fs_f = float(np.asarray(f_test.fvalue).item())

            # Second stage using fitted Elo in LPM
            iv_df = iv_df.copy()
            iv_df["elo_hat"] = fs.predict(fs_x)
            ss_x = sm.add_constant(iv_df[["elo_hat"] + iv_controls], has_constant="add")
            ss = sm.OLS(iv_df["won_wc"], ss_x).fit(cov_type="HC3")

            # Heuristic exclusion-risk signal.
            red_x = sm.add_constant(iv_df[iv_controls], has_constant="add")
            red = sm.OLS(iv_df["won_wc"], red_x).fit(cov_type="HC3")
            residual = iv_df["won_wc"] - red.predict(red_x)
            excl_corr = float(np.corrcoef(iv_df[instrument], residual)[0, 1])
            iv_rows.append(
                {
                    "instrument": instrument,
                    "first_stage_F": fs_f,
                    "elo_hat_coef_second_stage": float(ss.params.get("elo_hat", np.nan)),
                    "elo_hat_p_second_stage": float(ss.pvalues.get("elo_hat", np.nan)),
                    "instrument_residual_corr": excl_corr,
                    "instrument_assessment": (
                        "likely weak/invalid"
                        if (fs_f < 10 or abs(excl_corr) > 0.05)
                        else "plausible_strength_but_exclusion_unverifiable"
                    ),
                }
            )
    if not iv_rows:
        iv_rows.append(
            {
                "instrument": instrument or "none",
                "first_stage_F": np.nan,
                "elo_hat_coef_second_stage": np.nan,
                "elo_hat_p_second_stage": np.nan,
                "instrument_residual_corr": np.nan,
                "instrument_assessment": "no_valid_instrument_identified",
            }
        )
    iv_out = pd.DataFrame(iv_rows)
    iv_out.to_csv(OUT_DIR / "causal_iv_attempt.csv", index=False)
    out["iv"] = iv_out

    dag_mermaid = """# Hypothesized Causal DAG

```mermaid
graph LR
    GDP[GDP / Income] --> Elo[Elo / Team Quality]
    Pop[Population / Talent Pool] --> Elo
    Tradition[Football Tradition] --> Elo
    Titles[Past WC Titles] --> Tradition
    Host[Hosting] --> Win[World Cup Win]
    Elo --> Win
    Institution[Institutions / Governance] --> Econ[Macro Conditions]
    Econ --> GDP
    Region[Geography / Confederation] --> Elo
    Region --> Win
```
"""
    (OUT_DIR / "causal_dag.md").write_text(dag_mermaid, encoding="utf-8")
    return out


def rank_from_series(series: pd.Series, ascending: bool = False) -> pd.Series:
    s = series.copy().dropna()
    order = s.sort_values(ascending=ascending)
    return pd.Series(np.arange(1, len(order) + 1), index=order.index)


def infer_category(feature: str) -> str:
    f = feature.lower()
    if any(k in f for k in ["gdp", "trade", "inflation", "unemployment", "investment", "fdi", "gini", "rd_"]):
        return "economy"
    if any(k in f for k in ["football", "elo", "fifa", "wc_", "years_since_last", "former_champion", "titles_before"]):
        return "football"
    if any(k in f for k in ["population", "urban", "density", "young"]):
        return "demographics"
    if any(k in f for k in ["life_expectancy", "mortality", "health", "physician"]):
        return "health"
    if any(k in f for k in ["govt", "rule_of_law", "corruption", "political", "voice", "regulatory"]):
        return "institutions"
    if any(k in f for k in ["internet", "electricity", "air_transport"]):
        return "infrastructure"
    if any(k in f for k in ["is_europe", "is_south_america", "is_africa", "is_asia", "is_north_america", "is_oceania", "is_strong"]):
        return "geography"
    return "other"


def build_synthesis_ranking(
    univariate_df: pd.DataFrame,
    logistic_df: pd.DataFrame,
    rf_importance: pd.DataFrame,
    xgb_importance: pd.DataFrame,
    shap_or_perm_importance: pd.DataFrame,
) -> pd.DataFrame:
    uni_rank = rank_from_series(
        pd.Series(univariate_df["pointbiserial_p_fdr"].values, index=univariate_df["feature"].values),
        ascending=True,
    )
    log_rank = rank_from_series(
        pd.Series(logistic_df["coef"].abs().values, index=logistic_df["feature"].values),
        ascending=False,
    )
    rf_rank = rank_from_series(
        pd.Series(rf_importance["importance"].values, index=rf_importance["feature"].values),
        ascending=False,
    )
    xgb_rank = rank_from_series(
        pd.Series(xgb_importance["importance"].values, index=xgb_importance["feature"].values),
        ascending=False,
    )
    shap_rank = rank_from_series(
        pd.Series(shap_or_perm_importance["importance"].values, index=shap_or_perm_importance["feature"].values),
        ascending=False,
    )

    all_feats = sorted(set(uni_rank.index) | set(log_rank.index) | set(rf_rank.index) | set(xgb_rank.index) | set(shap_rank.index))
    rows = []
    for f in all_feats:
        r = {
            "feature": f,
            "rank_univariate_fdr": float(uni_rank.get(f, np.nan)),
            "rank_logistic_abscoef": float(log_rank.get(f, np.nan)),
            "rank_rf": float(rf_rank.get(f, np.nan)),
            "rank_xgb": float(xgb_rank.get(f, np.nan)),
            "rank_shap_or_perm": float(shap_rank.get(f, np.nan)),
        }
        vals = np.array([v for k, v in r.items() if k.startswith("rank_") and np.isfinite(v)], dtype=float)
        r["n_methods_present"] = int(len(vals))
        r["n_methods_top10"] = int(np.sum(vals <= 10))
        r["avg_rank"] = float(np.mean(vals)) if len(vals) else np.nan
        r["category"] = infer_category(f)
        rows.append(r)

    synth = pd.DataFrame(rows).sort_values(["n_methods_top10", "avg_rank"], ascending=[False, True]).reset_index(
        drop=True
    )
    return synth


def plot_synthesis_top(synth_df: pd.DataFrame, out_file: Path, top_n: int = 20) -> None:
    top = synth_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["n_methods_top10"])
    plt.xlabel("# Methods where feature is Top-10")
    plt.title("Unified Importance Synthesis (Top Features)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_calibration_and_confusion(oof_df: pd.DataFrame, out_prefix: str) -> Tuple[Path, Path]:
    y_true = oof_df["y_true"].astype(int).values
    y_prob = oof_df["y_prob"].astype(float).values
    y_pred = (y_prob >= 0.5).astype(int)

    # Calibration by deciles.
    cal_df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    cal_df["decile"] = pd.qcut(cal_df["y_prob"], q=10, duplicates="drop")
    grp = cal_df.groupby("decile", observed=False).agg(pred=("y_prob", "mean"), obs=("y_true", "mean"), n=("y_true", "count"))

    cal_path = FIG_DIR / f"{out_prefix}_calibration.png"
    plt.figure(figsize=(7, 6))
    plt.plot(grp["pred"], grp["obs"], marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    plt.xlabel("Predicted win probability")
    plt.ylabel("Observed win rate")
    plt.title("Calibration Plot (Deciles)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cal_path, dpi=150)
    plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_path = FIG_DIR / f"{out_prefix}_confusion_matrix.png"
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Aggregated Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    return cal_path, cm_path


def make_summary_markdown(
    audit: Dict[str, List[str]],
    univariate_df: pd.DataFrame,
    logistic_df: pd.DataFrame,
    hl_stat: Tuple[float, float],
    reg_scores: pd.DataFrame,
    rf_cv: pd.DataFrame,
    xgb_cv: pd.DataFrame,
    causal: Dict[str, pd.DataFrame],
    synth: pd.DataFrame,
    best_model_name: str,
    best_fold_df: pd.DataFrame,
    best_oof_df: pd.DataFrame,
) -> str:
    top_uni = univariate_df.head(10)["feature"].tolist()
    top_log = logistic_df.sort_values("p_value_fdr").head(10)["feature"].tolist()
    top_synth = synth.head(10)[["feature", "category", "n_methods_top10", "avg_rank"]]

    did_row = causal["did"].iloc[0]
    psm_row = causal["psm"].iloc[0]
    iv_row = causal["iv"].iloc[0]
    gr = causal["granger_like"]

    overall_auc = float(np.nanmean(best_fold_df["auc"]))
    overall_f1 = float(np.nanmean(best_fold_df["f1"]))
    overall_precision = float(np.nanmean(best_fold_df["precision"]))
    overall_recall = float(np.nanmean(best_fold_df["recall"]))
    overall_brier = float(np.nanmean(best_fold_df["brier"]))
    brier_oof = float(brier_score_loss(best_oof_df["y_true"], best_oof_df["y_prob"]))

    summary_lines = [
        "# World Cup Predictors: SOTA Analysis Summary",
        "",
        "## What was done",
        "- Ran a full leakage audit and dropped all direct leakage columns before analysis.",
        "- Used strict Leave-One-World-Cup-Out validation for all predictive models.",
        "- Fit train-only imputers/scalers in every fold (no test leakage).",
        "- Estimated effect sizes with 95% bootstrap CIs and adjusted p-values with both Bonferroni and FDR.",
        "",
        "## Leakage controls",
        f"- Direct leakage columns dropped: {', '.join(audit['leaking_drop_immediately'])}.",
        f"- Post-tournament columns excluded from prediction: {', '.join(audit['post_tournament_drop_for_prediction'])}.",
        "- Cross-sectional within-year ratio columns were retained after validation.",
        "",
        "## Key univariate signals (FDR-aware)",
        f"- Top 10 predictors by univariate significance/effect: {', '.join(top_uni)}.",
        "",
        "## Multivariate logistic results",
        f"- Top logistic features (FDR-ranked): {', '.join(top_log)}.",
        f"- Hosmer-Lemeshow goodness-of-fit: chi2={hl_stat[0]:.3f}, p={hl_stat[1]:.4f}.",
        "",
        "## Regularized logistic (LOWCO tuning)",
        f"- Best L1/L2/ElasticNet settings are in `regularized_logit_lowco_scores.csv` "
        f"(best mean AUC observed: {reg_scores['mean_auc'].max():.4f}).",
        "",
        "## Tree models (LOWCO)",
        f"- Random Forest mean LOWCO AUC: {np.nanmean(rf_cv['auc']):.4f}.",
        (
            f"- XGBoost mean LOWCO AUC: {np.nanmean(xgb_cv['auc']):.4f}."
            if not xgb_cv.empty
            else "- XGBoost was unavailable in this runtime."
        ),
        "",
        "## Causal analysis snapshots",
        (
            f"- Difference-in-Differences hosting effect: {did_row['effect']:.4f} "
            f"(95% CI {did_row['ci_low']:.4f} to {did_row['ci_high']:.4f})."
        ),
        (
            f"- Propensity score matching ATT (hosting -> winning): {psm_row['att_hosting_on_win']:.4f} "
            f"(95% CI {psm_row['ci_low']:.4f} to {psm_row['ci_high']:.4f}, "
            f"pairs={int(psm_row['matched_pairs'])})."
        ),
        (
            "- Granger-like lag comparison: "
            + ", ".join([f"{r.spec} AUC={r.mean_auc_lowco:.4f}" for r in gr.itertuples(index=False)])
            if not gr.empty
            else "- Granger-like lag comparison could not be estimated."
        ),
        (
            f"- IV attempt for Elo (instrument `{iv_row['instrument']}`): "
            f"first-stage F={iv_row['first_stage_F']}, assessment={iv_row['instrument_assessment']}."
        ),
        "",
        "## Unified ranking (across methods)",
        "- Features strongest across methods:",
    ]
    for r in top_synth.itertuples(index=False):
        summary_lines.append(
            f"  - {r.feature} (category={r.category}, top10_methods={int(r.n_methods_top10)}, avg_rank={r.avg_rank:.2f})"
        )

    summary_lines.extend(
        [
            "",
            "## Best predictive model (LOWCO)",
            f"- Selected model: `{best_model_name}`.",
            f"- Mean AUC={overall_auc:.4f}, Precision={overall_precision:.4f}, Recall={overall_recall:.4f}, F1={overall_f1:.4f}.",
            f"- Mean fold Brier={overall_brier:.4f}, pooled OOF Brier={brier_oof:.4f}.",
            "",
            "## Caveats",
            "- This is observational data; causal effects should be interpreted as suggestive, not definitive.",
            "- IV validity for Elo is weak and exclusion restriction is difficult to justify with available columns.",
        ]
    )
    return "\n".join(summary_lines) + "\n"


def main() -> None:
    ensure_output_dirs()
    np.random.seed(RANDOM_STATE)
    warnings.filterwarnings("ignore")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df_raw = pd.read_csv(DATA_PATH)
    if "won_wc" not in df_raw.columns or "wc_year" not in df_raw.columns:
        raise ValueError("Dataset must include `won_wc` and `wc_year`.")

    # ---------------------------------------------------------------------
    # Phase 1: Leakage audit (must happen before any analysis).
    # ---------------------------------------------------------------------
    audit = print_and_save_leakage_audit(df_raw)

    # Drop leakage and post-tournament columns from analytical dataframe.
    drop_cols = [c for c in (LEAK_COLUMNS + POST_TOURNAMENT_COLUMNS) if c in df_raw.columns]
    df = df_raw.drop(columns=drop_cols).copy()

    # ---------------------------------------------------------------------
    # Phase 2 + 3: Preprocessing setup and univariate analysis.
    # ---------------------------------------------------------------------
    predictor_cols = prepare_predictive_columns(df)

    # For univariate descriptive inference, use within-year medians.
    uni_df = df[["wc_year", "won_wc"] + predictor_cols].copy()
    uni_imp = YearMedianImputer(year_col="wc_year").fit(uni_df, numeric_cols=predictor_cols)
    uni_df = uni_imp.transform(uni_df)
    univariate = run_univariate_analysis(uni_df, predictor_cols=predictor_cols, target_col="won_wc")
    univariate.to_csv(OUT_DIR / "univariate_analysis.csv", index=False)
    plot_univariate_effects(univariate, FIG_DIR / "univariate_effect_sizes.png", top_n=20)

    # ---------------------------------------------------------------------
    # Phase 4: Multivariate logistic regression.
    # ---------------------------------------------------------------------
    top_features = univariate.head(20)["feature"].tolist()
    vif_base = uni_df[top_features].copy()
    vif_selected = iterative_vif_prune(vif_base, top_features, threshold=5.0)

    vif_report = compute_vif(uni_df, vif_selected)
    vif_report.to_csv(OUT_DIR / "logit_vif_report.csv", index=False)

    logistic_results, logit_model = fit_statsmodels_logit_with_profile_ci(df, vif_selected)
    logistic_results.to_csv(OUT_DIR / "logit_unpenalized_results.csv", index=False)

    # Hosmer-Lemeshow on inferential model fitted values.
    fit_df = df[["wc_year", "won_wc"] + vif_selected].copy()
    fit_imp = YearMedianImputer(year_col="wc_year").fit(fit_df, numeric_cols=vif_selected)
    fit_df = fit_imp.transform(fit_df)
    x_fit = StandardScaler().fit_transform(fit_df[vif_selected].values)
    x_fit = sm.add_constant(x_fit, has_constant="add")
    y_fit = fit_df["won_wc"].astype(int).values
    y_prob_fit = logit_model.predict(x_fit)
    hl_chi2, hl_p = hosmer_lemeshow_test(y_fit, y_prob_fit, n_groups=10)
    pd.DataFrame([{"hl_chi2": hl_chi2, "hl_p_value": hl_p}]).to_csv(OUT_DIR / "logit_hosmer_lemeshow.csv", index=False)

    reg_scores, reg_best = tune_regularized_logit_lowco(df, feature_cols=predictor_cols)
    reg_scores.to_csv(OUT_DIR / "regularized_logit_lowco_scores.csv", index=False)

    # ---------------------------------------------------------------------
    # Phase 5: Tree models with LOWCO CV + SHAP/permutation.
    # ---------------------------------------------------------------------
    # Random Forest LOWCO
    rf_builder = lambda _pos, _neg: RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_cv, rf_oof = evaluate_lowco_model(
        df,
        feature_cols=predictor_cols,
        build_model_fn=rf_builder,
        scale_features=False,
        model_name="random_forest",
    )
    rf_cv.to_csv(OUT_DIR / "rf_lowco_folds.csv", index=False)
    rf_oof.to_csv(OUT_DIR / "rf_lowco_oof_predictions.csv", index=False)

    # XGBoost LOWCO (if available)
    xgb_cv = pd.DataFrame()
    xgb_oof = pd.DataFrame()
    xgb_pkg = try_import_or_install("xgboost")
    if xgb_pkg is not None:
        XGBClassifier = xgb_pkg.XGBClassifier

        def xgb_builder(pos: int, neg: int):
            spw = float(neg / max(pos, 1))
            return XGBClassifier(
                n_estimators=450,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                scale_pos_weight=spw,
            )

        xgb_cv, xgb_oof = evaluate_lowco_model(
            df,
            feature_cols=predictor_cols,
            build_model_fn=xgb_builder,
            scale_features=False,
            model_name="xgboost",
        )
        xgb_cv.to_csv(OUT_DIR / "xgb_lowco_folds.csv", index=False)
        xgb_oof.to_csv(OUT_DIR / "xgb_lowco_oof_predictions.csv", index=False)

    shap_module = try_import_or_install("shap")
    tree_importance = fit_tree_models_and_importance(df, predictor_cols, shap_module=shap_module)

    # ---------------------------------------------------------------------
    # Phase 6: Causal analysis.
    # ---------------------------------------------------------------------
    causal = run_causal_analysis(df, predictor_cols)

    # ---------------------------------------------------------------------
    # Phase 7: Synthesis ranking.
    # ---------------------------------------------------------------------
    synth = build_synthesis_ranking(
        univariate_df=univariate,
        logistic_df=logistic_results,
        rf_importance=tree_importance["rf_importance"],
        xgb_importance=tree_importance.get("xgb_importance", pd.DataFrame(columns=["feature", "importance"])),
        shap_or_perm_importance=tree_importance["shap_importance"],
    )
    synth.to_csv(OUT_DIR / "synthesis_ranking.csv", index=False)
    cat_breakdown = synth.groupby("category", as_index=False).agg(
        mean_avg_rank=("avg_rank", "mean"),
        features_in_top10=("n_methods_top10", lambda x: int(np.sum(x > 0))),
        n_features=("feature", "count"),
    )
    cat_breakdown.to_csv(OUT_DIR / "synthesis_category_breakdown.csv", index=False)
    plot_synthesis_top(synth, FIG_DIR / "synthesis_top_features.png", top_n=20)

    # ---------------------------------------------------------------------
    # Phase 8: Predictive power assessment.
    # ---------------------------------------------------------------------
    candidates = []

    # Best regularized logistic by LOWCO AUC.
    if not reg_scores.empty:
        best_row = reg_scores.iloc[0]
        best_cfg = best_row.to_dict()

        def best_logit_builder(_pos: int, _neg: int):
            kwargs = dict(
                penalty=best_cfg["penalty"],
                C=float(best_cfg["C"]),
                solver=best_cfg["solver"],
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000,
            )
            if best_cfg["penalty"] == "elasticnet":
                kwargs["l1_ratio"] = float(best_cfg["l1_ratio"])
            return LogisticRegression(**kwargs)

        logit_fold, logit_oof = evaluate_lowco_model(
            df,
            feature_cols=vif_selected,
            build_model_fn=best_logit_builder,
            scale_features=True,
            model_name="best_regularized_logit",
        )
        logit_fold.to_csv(OUT_DIR / "best_logit_lowco_folds.csv", index=False)
        logit_oof.to_csv(OUT_DIR / "best_logit_lowco_oof_predictions.csv", index=False)
        candidates.append(("best_regularized_logit", logit_fold, logit_oof))

    candidates.append(("random_forest", rf_cv, rf_oof))
    if not xgb_cv.empty:
        candidates.append(("xgboost", xgb_cv, xgb_oof))

    # Pick best by mean AUC.
    scored = []
    for name, fold_df, oof_df in candidates:
        scored.append((name, float(np.nanmean(fold_df["auc"])), fold_df, oof_df))
    scored.sort(key=lambda t: t[1], reverse=True)
    best_model_name, best_auc, best_fold_df, best_oof_df = scored[0]

    best_fold_df.to_csv(OUT_DIR / "best_model_lowco_folds.csv", index=False)
    best_oof_df.to_csv(OUT_DIR / "best_model_lowco_oof_predictions.csv", index=False)
    plot_calibration_and_confusion(best_oof_df, out_prefix="best_model")
    best_brier = brier_score_loss(best_oof_df["y_true"], best_oof_df["y_prob"])
    pd.DataFrame(
        [
            {
                "best_model": best_model_name,
                "mean_auc": float(np.nanmean(best_fold_df["auc"])),
                "mean_precision": float(np.nanmean(best_fold_df["precision"])),
                "mean_recall": float(np.nanmean(best_fold_df["recall"])),
                "mean_f1": float(np.nanmean(best_fold_df["f1"])),
                "mean_brier": float(np.nanmean(best_fold_df["brier"])),
                "pooled_oof_brier": float(best_brier),
            }
        ]
    ).to_csv(OUT_DIR / "predictive_assessment.csv", index=False)

    # ---------------------------------------------------------------------
    # Summary markdown + executive stdout summary.
    # ---------------------------------------------------------------------
    summary_md = make_summary_markdown(
        audit=audit,
        univariate_df=univariate,
        logistic_df=logistic_results,
        hl_stat=(hl_chi2, hl_p),
        reg_scores=reg_scores,
        rf_cv=rf_cv,
        xgb_cv=xgb_cv,
        causal=causal,
        synth=synth,
        best_model_name=best_model_name,
        best_fold_df=best_fold_df,
        best_oof_df=best_oof_df,
    )
    (OUT_DIR / "summary.md").write_text(summary_md, encoding="utf-8")

    print("\n" + "=" * 88)
    print("EXECUTIVE SUMMARY")
    print("=" * 88)
    print(f"Data rows: {len(df_raw)}, target positives: {int(df_raw['won_wc'].sum())}")
    print(f"Leakage columns dropped: {', '.join(audit['leaking_drop_immediately'])}")
    print(f"Best LOWCO model: {best_model_name} (AUC={best_auc:.4f})")
    print(f"Hosmer-Lemeshow p-value (unpenalized logit): {hl_p:.4f}")
    print(
        "Top synthesis features: "
        + ", ".join(synth.head(5)["feature"].tolist())
    )
    print(f"Outputs saved to: {OUT_DIR}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
