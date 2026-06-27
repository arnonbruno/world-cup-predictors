"""
World Cup Predictors - Comprehensive Econometric & ML Analysis
==============================================================
Goal: Identify which variables actually predict World Cup winners.

Analyzes:
1. Univariate statistics (t-tests, effect sizes)
2. Correlation analysis (point-biserial, Spearman)
3. Logistic regression (univariate + multivariate)
4. Random Forest feature importance
5. XGBoost feature importance
6. LASSO feature selection
7. Panel data econometrics (fixed effects)
8. Marginal effects analysis
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'world_cup_predictors_dataset.csv'))
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Won WC: {df['won_wc'].sum()} out of {len(df)} ({df['won_wc'].mean()*100:.1f}%)")
    return df


# ============================================================
# ANALYSIS 1: UNIVARIATE COMPARISON (Winners vs Others)
# ============================================================

def univariate_analysis(df):
    """Compare winners vs non-winners on each variable."""
    print("\n" + "="*80)
    print("ANALYSIS 1: UNIVARIATE COMPARISON (Winners vs Non-Winners)")
    print("="*80)

    winners = df[df['won_wc'] == 1]
    others = df[df['won_wc'] == 0]

    # All numeric potential predictors
    exclude = ['won_wc', 'runner_up', 'semifinalist', 'finalist', 'top4',
               'is_winner', 'iso3', 'country', 'wc_year', 'decade']
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    results = []
    for col in numeric_cols:
        w_vals = winners[col].dropna()
        o_vals = others[col].dropna()

        if len(w_vals) < 3 or len(o_vals) < 10:
            continue

        # T-test
        t_stat, p_val = stats.ttest_ind(w_vals, o_vals, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(w_vals)-1)*w_vals.std()**2 + (len(o_vals)-1)*o_vals.std()**2) /
                            (len(w_vals) + len(o_vals) - 2))
        cohens_d = (w_vals.mean() - o_vals.mean()) / pooled_std if pooled_std > 0 else 0

        # Point-biserial correlation
        valid_mask = df[col].notna() & df['won_wc'].notna()
        if valid_mask.sum() > 20:
            r_pb, p_pb = stats.pointbiserialr(df.loc[valid_mask, 'won_wc'],
                                             df.loc[valid_mask, col])
        else:
            r_pb, p_pb = np.nan, np.nan

        results.append({
            'variable': col,
            'winners_mean': w_vals.mean(),
            'others_mean': o_vals.mean(),
            'ratio': w_vals.mean() / o_vals.mean() if o_vals.mean() != 0 else np.nan,
            'pct_diff': ((w_vals.mean() - o_vals.mean()) / abs(o_vals.mean()) * 100
                        if o_vals.mean() != 0 else np.nan),
            'cohens_d': cohens_d,
            't_stat': t_stat,
            'p_value': p_val,
            'pointbiserial_r': r_pb,
            'pointbiserial_p': p_pb,
            'n_winners': len(w_vals),
            'n_others': len(o_vals),
            'significant_005': '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else '')),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')

    print("\nTop 30 predictors by p-value:")
    print("-" * 100)
    print(f"{'Variable':30s} {'Win Mean':>10s} {'Oth Mean':>10s} {'Ratio':>7s} {'Cohen d':>8s} {'t-stat':>8s} {'p-val':>10s} {'r_pb':>7s} {'Sig':>4s}")
    print("-" * 100)
    for _, row in results_df.head(30).iterrows():
        print(f"{row['variable']:30s} {row['winners_mean']:10.2f} {row['others_mean']:10.2f} "
              f"{row['ratio']:7.2f} {row['cohens_d']:8.3f} {row['t_stat']:8.2f} "
              f"{row['p_value']:10.4f} {row['pointbiserial_r']:7.3f} {row['significant_005']:>4s}")

    # Save
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'univariate_analysis.csv'), index=False)
    return results_df


# ============================================================
# ANALYSIS 2: CORRELATION MATRIX
# ============================================================

def correlation_analysis(df):
    """Full correlation matrix of all predictors with won_wc."""
    print("\n" + "="*80)
    print("ANALYSIS 2: CORRELATION WITH WINNING")
    print("="*80)

    exclude = ['won_wc', 'runner_up', 'semifinalist', 'finalist', 'top4',
               'is_winner', 'iso3', 'country', 'wc_year', 'decade']
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    correlations = {}
    for col in numeric_cols:
        valid = df[[col, 'won_wc']].dropna()
        if len(valid) > 20:
            r_s, p_s = stats.spearmanr(valid[col], valid['won_wc'])
            r_p, p_p = stats.pearsonr(valid[col], valid['won_wc'])
            correlations[col] = {
                'spearman_r': r_s, 'spearman_p': p_s,
                'pearson_r': r_p, 'pearson_p': p_p,
                'n': len(valid)
            }

    corr_df = pd.DataFrame(correlations).T
    corr_df = corr_df.sort_values('spearman_p')

    print("\nTop 20 by Spearman correlation with won_wc:")
    print("-" * 80)
    print(f"{'Variable':30s} {'Spearman r':>10s} {'p-val':>10s} {'Pearson r':>10s} {'N':>6s}")
    print("-" * 80)
    for var, row in corr_df.head(20).iterrows():
        print(f"{var:30s} {row['spearman_r']:10.4f} {row['spearman_p']:10.6f} {row['pearson_r']:10.4f} {int(row['n']):6d}")

    corr_df.to_csv(os.path.join(OUTPUT_DIR, 'correlation_analysis.csv'))
    return corr_df


# ============================================================
# ANALYSIS 3: LOGISTIC REGRESSION
# ============================================================

def logistic_regression_analysis(df):
    """Univariate and multivariate logistic regression."""
    print("\n" + "="*80)
    print("ANALYSIS 3: LOGISTIC REGRESSION")
    print("="*80)

    exclude = ['won_wc', 'runner_up', 'semifinalist', 'finalist', 'top4',
               'is_winner', 'iso3', 'country', 'wc_year', 'decade',
               'confederation']
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    # Filter to columns with enough data
    valid_cols = []
    for col in numeric_cols:
        valid_mask = df[col].notna() & df['won_wc'].notna()
        if valid_mask.sum() > 50:
            valid_cols.append(col)

    print(f"\nAnalyzing {len(valid_cols)} variables with sufficient data")

    # --- Univariate logistic regressions ---
    print("\n--- UNIVARIATE LOGISTIC REGRESSIONS ---")
    univariate_results = []
    for col in valid_cols:
        subset = df[[col, 'won_wc']].dropna()
        X = subset[[col]].values
        y = subset['won_wc'].values

        if len(np.unique(y)) < 2:
            continue

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LogisticRegression(max_iter=1000, solver='lbfgs')
            model.fit(X_scaled, y)

            coef = model.coef_[0][0]
            odds_ratio = np.exp(coef)
            y_pred = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, y_pred)

            # p-value via Wald test
            from sklearn.linear_model import LogisticRegression as LR
            # Approximate p-value using z-score
            n = len(y)
            se = np.sqrt(coef**2 / (auc * (1 - auc) * n)) if auc > 0 and auc < 1 else np.nan
            z = coef / se if se > 0 else 0
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))

            univariate_results.append({
                'variable': col,
                'coef': coef,
                'odds_ratio': odds_ratio,
                'auc': auc,
                'p_value': p_val,
                'n': n,
            })
        except Exception:
            pass

    uni_df = pd.DataFrame(univariate_results).sort_values('auc', ascending=False)
    print(f"\nTop 20 by AUC:")
    print("-" * 90)
    print(f"{'Variable':30s} {'Coef':>8s} {'Odds Ratio':>10s} {'AUC':>8s} {'p-val':>10s} {'N':>6s}")
    print("-" * 90)
    for _, row in uni_df.head(20).iterrows():
        print(f"{row['variable']:30s} {row['coef']:8.4f} {row['odds_ratio']:10.4f} "
              f"{row['auc']:8.4f} {row['p_value']:10.6f} {int(row['n']):6d}")

    # --- Multivariate logistic regression ---
    print("\n--- MULTIVARIATE LOGISTIC REGRESSION ---")

    # Select top features from univariate (AUC > 0.7 or p < 0.05)
    top_features = uni_df[(uni_df['auc'] > 0.65) | (uni_df['p_value'] < 0.1)]['variable'].tolist()
    if len(top_features) > 15:
        top_features = uni_df.head(15)['variable'].tolist()

    print(f"Using {len(top_features)} features: {top_features}")

    # Prepare data
    df_clean = df[top_features + ['won_wc', 'wc_year']].dropna()
    X = df_clean[top_features].values
    y = df_clean['won_wc'].values
    groups = df_clean['wc_year'].values

    if len(df_clean) > 50 and len(np.unique(y)) > 1:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=2000, solver='lbfgs', C=0.1)
        model.fit(X_scaled, y)

        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)

        print(f"\nMultivariate AUC: {auc:.4f}")
        print(f"\nFeature coefficients (standardized):")
        coef_df = pd.DataFrame({
            'feature': top_features,
            'coef': model.coef_[0],
            'odds_ratio': np.exp(model.coef_[0]),
            'abs_coef': np.abs(model.coef_[0])
        }).sort_values('abs_coef', ascending=False)

        for _, row in coef_df.iterrows():
            direction = "+" if row['coef'] > 0 else "-"
            print(f"  {row['feature']:30s}: {direction}{abs(row['coef']):.4f} (OR={row['odds_ratio']:.4f})")

        # Cross-validation (leave-one-WC-out)
        logo = LeaveOneGroupOut()
        cv_scores = cross_val_score(model, X_scaled, y, cv=logo.split(X_scaled, y, groups),
                                    scoring='roc_auc')
        print(f"\nLeave-One-WC-Out CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    uni_df.to_csv(os.path.join(OUTPUT_DIR, 'logistic_regression_univariate.csv'), index=False)
    return uni_df


# ============================================================
# ANALYSIS 4: RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

def random_forest_analysis(df):
    """Random Forest feature importance analysis."""
    print("\n" + "="*80)
    print("ANALYSIS 4: RANDOM FOREST FEATURE IMPORTANCE")
    print("="*80)

    exclude = ['won_wc', 'runner_up', 'semifinalist', 'finalist', 'top4',
               'is_winner', 'iso3', 'country', 'wc_year', 'decade',
               'confederation']
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    # Use only columns with >60% completeness
    valid_cols = [c for c in numeric_cols if df[c].notna().mean() > 0.6]
    print(f"Using {len(valid_cols)} features with >60% completeness")

    df_clean = df[valid_cols + ['won_wc']].dropna()
    X = df_clean[valid_cols].values
    y = df_clean['won_wc'].values

    print(f"Clean dataset: {len(df_clean)} rows, {y.sum()} winners")

    if len(df_clean) > 30 and y.sum() >= 5:
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=2,
            class_weight='balanced', random_state=42
        )
        rf.fit(X, y)

        importance_df = pd.DataFrame({
            'feature': valid_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 20 features by Random Forest importance:")
        print("-" * 60)
        for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
            bar = "█" * int(row['importance'] * 200)
            print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f} {bar}")

        # Cross-validation
        groups = df_clean.index.map(lambda i: df.loc[i, 'wc_year'])
        logo = LeaveOneGroupOut()
        try:
            cv_scores = cross_val_score(rf, X, y, cv=logo.split(X, y, groups),
                                        scoring='roc_auc')
            print(f"\nLeave-One-WC-Out CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception as e:
            print(f"\nCV error: {e}")

        importance_df.to_csv(os.path.join(OUTPUT_DIR, 'random_forest_importance.csv'), index=False)
        return importance_df


# ============================================================
# ANALYSIS 5: XGBOOST FEATURE IMPORTANCE
# ============================================================

def xgboost_analysis(df):
    """XGBoost feature importance analysis."""
    print("\n" + "="*80)
    print("ANALYSIS 5: XGBOOST FEATURE IMPORTANCE")
    print("="*80)

    exclude = ['won_wc', 'runner_up', 'semifinalist', 'finalist', 'top4',
               'is_winner', 'iso3', 'country', 'wc_year', 'decade',
               'confederation']
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    valid_cols = [c for c in numeric_cols if df[c].notna().mean() > 0.6]
    df_clean = df[valid_cols + ['won_wc']].dropna()
    X = df_clean[valid_cols].values
    y = df_clean['won_wc'].values

    if len(df_clean) > 30 and y.sum() >= 5:
        scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)

        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos, random_state=42,
            eval_metric='auc', use_label_encoder=False
        )
        xgb_model.fit(X, y)

        importance_df = pd.DataFrame({
            'feature': valid_cols,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 20 features by XGBoost importance:")
        print("-" * 60)
        for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
            bar = "█" * int(row['importance'] * 200)
            print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f} {bar}")

        importance_df.to_csv(os.path.join(OUTPUT_DIR, 'xgboost_importance.csv'), index=False)
        return importance_df


# ============================================================
# ANALYSIS 6: LASSO FEATURE SELECTION
# ============================================================

def lasso_analysis(df):
    """LASSO regularization for feature selection."""
    print("\n" + "="*80)
    print("ANALYSIS 6: LASSO FEATURE SELECTION")
    print("="*80)

    exclude = ['won_wc', 'runner_up', 'semifinalist', 'finalist', 'top4',
               'is_winner', 'iso3', 'country', 'wc_year', 'decade',
               'confederation']
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    valid_cols = [c for c in numeric_cols if df[c].notna().mean() > 0.6]
    df_clean = df[valid_cols + ['won_wc']].dropna()
    X = df_clean[valid_cols].values
    y = df_clean['won_wc'].values

    if len(df_clean) > 30:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Try different regularization strengths
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.1]:
            lasso = LogisticRegression(penalty='l1', solver='saga', C=1/alpha,
                                       max_iter=5000, random_state=42)
            lasso.fit(X_scaled, y)

            coefs = lasso.coef_[0]
            selected = [(valid_cols[i], coefs[i]) for i in range(len(coefs)) if coefs[i] != 0]
            selected.sort(key=lambda x: abs(x[1]), reverse=True)

            print(f"\nC={1/alpha:.1f} (alpha={alpha}): {len(selected)} features selected")
            if selected:
                for name, coef in selected[:10]:
                    direction = "+" if coef > 0 else "-"
                    print(f"    {direction} {name}: {coef:.4f}")

        # Final model with best C
        lasso_final = LogisticRegression(penalty='l1', solver='saga', C=0.01,
                                         max_iter=5000, random_state=42)
        lasso_final.fit(X_scaled, y)

        coefs = lasso_final.coef_[0]
        selected_df = pd.DataFrame({
            'feature': valid_cols,
            'coef': coefs,
            'abs_coef': np.abs(coefs)
        })
        selected_df = selected_df[selected_df['coef'] != 0].sort_values('abs_coef', ascending=False)

        selected_df.to_csv(os.path.join(OUTPUT_DIR, 'lasso_selected_features.csv'), index=False)
        return selected_df


# ============================================================
# ANALYSIS 7: SYNTHESIS RANKING
# ============================================================

def synthesis_ranking(df, uni_df, rf_imp, xgb_imp, lasso_df, corr_df):
    """Combine all methods into a unified ranking."""
    print("\n" + "="*80)
    print("ANALYSIS 7: SYNTHESIS - UNIFIED VARIABLE IMPORTANCE RANKING")
    print("="*80)

    # Collect rankings from each method
    all_vars = set()

    # Univariate (by p-value rank)
    if uni_df is not None:
        uni_ranks = {row['variable']: i+1 for i, (_, row) in enumerate(uni_df.iterrows())}
        all_vars.update(uni_ranks.keys())

    # Random Forest
    rf_ranks = {}
    if rf_imp is not None:
        rf_ranks = {row['feature']: i+1 for i, (_, row) in enumerate(rf_imp.iterrows())}
        all_vars.update(rf_ranks.keys())

    # XGBoost
    xgb_ranks = {}
    if xgb_imp is not None:
        xgb_ranks = {row['feature']: i+1 for i, (_, row) in enumerate(xgb_imp.iterrows())}
        all_vars.update(xgb_ranks.keys())

    # Correlation
    corr_ranks = {}
    if corr_df is not None:
        corr_ranks = {var: i+1 for i, var in enumerate(corr_df.index)}
        all_vars.update(corr_ranks.keys())

    # LASSO
    lasso_vars = set()
    if lasso_df is not None:
        lasso_vars = set(lasso_df['feature'].tolist())
        all_vars.update(lasso_vars)

    # Compute synthesis score (lower = better)
    synthesis = []
    for var in all_vars:
        ranks = []
        methods_present = []

        if var in uni_ranks:
            ranks.append(uni_ranks[var])
            methods_present.append('univariate')
        if var in rf_ranks:
            ranks.append(rf_ranks[var])
            methods_present.append('random_forest')
        if var in xgb_ranks:
            ranks.append(xgb_ranks[var])
            methods_present.append('xgboost')
        if var in corr_ranks:
            ranks.append(corr_ranks[var])
            methods_present.append('correlation')

        avg_rank = np.mean(ranks) if ranks else 999
        in_lasso = var in lasso_vars

        synthesis.append({
            'variable': var,
            'avg_rank': avg_rank,
            'univariate_rank': uni_ranks.get(var, '-'),
            'rf_rank': rf_ranks.get(var, '-'),
            'xgb_rank': xgb_ranks.get(var, '-'),
            'corr_rank': corr_ranks.get(var, '-'),
            'in_lasso': in_lasso,
            'n_methods': len(methods_present),
            'methods': ', '.join(methods_present),
        })

    synth_df = pd.DataFrame(synthesis).sort_values('avg_rank')

    print("\nTOP 25 PREDICTORS OF WORLD CUP WINNERS")
    print("=" * 100)
    print(f"{'Rank':>4s} {'Variable':30s} {'Avg Rank':>9s} {'Univar':>7s} {'RF':>5s} {'XGB':>5s} {'Corr':>5s} {'LASSO':>6s} {'Methods':>3s}")
    print("-" * 100)
    for i, (_, row) in enumerate(synth_df.head(25).iterrows()):
        lasso_mark = "  ✓" if row['in_lasso'] else ""
        print(f"{i+1:4d} {row['variable']:30s} {row['avg_rank']:9.1f} "
              f"{str(row['univariate_rank']):>7s} {str(row['rf_rank']):>5s} "
              f"{str(row['xgb_rank']):>5s} {str(row['corr_rank']):>5s}"
              f"{lasso_mark:>6s} {int(row['n_methods']):3d}")

    synth_df.to_csv(os.path.join(OUTPUT_DIR, 'synthesis_ranking.csv'), index=False)

    # Category breakdown
    print("\n--- BY CATEGORY ---")
    categories = {
        'Economy': ['gdp_per_capita', 'gdp_total', 'gdp_growth', 'trade_pct_gdp',
                    'inflation', 'unemployment', 'investment_pct_gdp', 'fdi_pct_gdp',
                    'govt_spending_pct_gdp', 'gdp_per_capita_log', 'gdp_total_log',
                    'gdp_per_capita_vs_avg', 'gdp_per_capita_vs_winner', 'gdp_per_capita_rank', 'gdp_total_rank'],
        'Population': ['population', 'population_log', 'population_growth', 'urbanization_pct',
                       'pop_density', 'pop_pct_young', 'pop_pct_0_14', 'pop_pct_15_64', 'pop_pct_65_plus',
                       'fertility_rate', 'population_rank', 'population_vs_avg', 'population_vs_winner'],
        'Health': ['life_expectancy', 'infant_mortality', 'under5_mortality',
                   'health_spending_pct_gdp', 'physicians_per_1000'],
        'Education': ['literacy_rate', 'edu_spending_pct_gdp', 'secondary_enrollment', 'tertiary_enrollment'],
        'Governance': ['govt_effectiveness', 'rule_of_law', 'control_corruption',
                       'political_stability', 'voice_accountability', 'regulatory_quality'],
        'Infrastructure': ['internet_users_pct', 'electricity_access_pct', 'air_transport_passengers'],
        'Football': ['elo_rating', 'fifa_rank', 'fifa_rank_inverse', 'football_tradition',
                     'wc_titles_before', 'wc_finals_before', 'wc_semifinals_before',
                     'wc_participations_before', 'years_since_last_wc', 'years_since_last_win',
                     'years_since_last_final', 'football_power_index', 'is_former_champion',
                     'is_strong_europe', 'is_strong_sa', 'is_host'],
        'Military': ['military_spending_pct_gdp', 'military_personnel'],
        'Inequality': ['gini_index'],
        'Innovation': ['rd_spending_pct_gdp'],
        'Regional': ['is_europe', 'is_south_america', 'is_africa', 'is_asia', 'is_north_america', 'is_oceania'],
    }

    for cat_name, cat_vars in categories.items():
        cat_items = synth_df[synth_df['variable'].isin(cat_vars)].head(5)
        if not cat_items.empty:
            best = cat_items.iloc[0]
            print(f"\n  {cat_name}:")
            for _, row in cat_items.iterrows():
                print(f"    {row['variable']:30s} (avg rank: {row['avg_rank']:.1f})")

    return synth_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*80)
    print("WORLD CUP PREDICTORS - COMPREHENSIVE ANALYSIS")
    print("="*80)

    df = load_data()

    # Run all analyses
    uni_df = univariate_analysis(df)
    corr_df = correlation_analysis(df)
    logistic_regression_analysis(df)
    rf_imp = random_forest_analysis(df)
    xgb_imp = xgboost_analysis(df)
    lasso_df = lasso_analysis(df)
    synth_df = synthesis_ranking(df, uni_df, rf_imp, xgb_imp, lasso_df, corr_df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
