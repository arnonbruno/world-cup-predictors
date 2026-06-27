"""
World Cup Backtest: Leave-One-World-Cup-Out Predictions
"""
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Actual WC winners by year
WC_WINNERS = {
    1930: 'Uruguay', 1934: 'Italy', 1938: 'Italy',
    1950: 'Uruguay', 1954: 'Germany', 1958: 'Brazil',
    1962: 'Brazil', 1966: 'England', 1970: 'Brazil',
    1974: 'Germany', 1978: 'Argentina', 1982: 'Italy',
    1986: 'Argentina', 1990: 'Germany', 1994: 'Brazil',
    1998: 'France', 2002: 'Brazil', 2006: 'Italy',
    2010: 'Spain', 2014: 'Germany', 2018: 'France',
    2022: 'Argentina',
}

DROP_COLS = ['runner_up', 'semifinalist', 'finalist', 'top4', 'is_winner',
             'gdp_per_capita_vs_winner', 'population_vs_winner',
             'total_goals_in_tournament', 'avg_goals_per_match',
             'country', 'iso3', 'confederation']
TARGET = 'won_wc'

df = pd.read_csv(f'{DATA_DIR}/world_cup_predictors_dataset.csv')
feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET, 'wc_year']]
wc_years = sorted(df['wc_year'].unique())

print("="*95)
print("WORLD CUP WINNER BACKTEST — Chronological")
print("="*95)
print(f"Dataset: {len(df)} rows, {len(feature_cols)} features, {len(wc_years)} WCs")
print(f"Model: L2 Logistic Regression (C=0.1, balanced class weights)\n")

results = []
all_probs = []
all_truths = []

for test_year in wc_years:
    actual_winner = WC_WINNERS[test_year]

    test_mask = df['wc_year'] == test_year
    train_mask = df['wc_year'] < test_year

    if not train_mask.any():
        print(f"{test_year} skipped: no prior World Cups available for training")
        continue

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, TARGET].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, TARGET].values
    teams = df.loc[test_mask, 'country'].values

    # Impute + scale on train only (no leakage)
    imp = SimpleImputer(strategy='median')
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(penalty='l2', C=0.1, max_iter=2000,
                               class_weight='balanced', solver='lbfgs')
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    all_probs.extend(probs)
    all_truths.extend(y_test)

    rank_df = pd.DataFrame({
        'team': teams, 'win_prob': probs
    }).sort_values('win_prob', ascending=False).reset_index(drop=True)
    rank_df['rank'] = range(1, len(rank_df) + 1)

    predicted_winner = rank_df.iloc[0]['team']
    predicted_prob = rank_df.iloc[0]['win_prob']

    # Find actual winner in ranking
    actual_row = rank_df[rank_df['team'] == actual_winner]
    if not actual_row.empty:
        actual_rank = actual_row['rank'].values[0]
        actual_prob = actual_row['win_prob'].values[0]
    else:
        actual_rank = 99
        actual_prob = 0.0

    correct = (predicted_winner == actual_winner)
    in_top3 = actual_rank <= 3
    in_top5 = actual_rank <= 5

    results.append({
        'year': test_year,
        'actual': actual_winner,
        'predicted': predicted_winner,
        'pred_prob': predicted_prob,
        'correct': correct,
        'actual_rank': actual_rank,
        'actual_prob': actual_prob,
        'in_top3': in_top3,
        'in_top5': in_top5,
    })

    icon = "✅" if correct else ("🟡" if in_top3 else ("🟠" if in_top5 else "❌"))
    top3_str = " | ".join([f"{r.team}({r.win_prob:.0%})" for _, r in rank_df.head(3).iterrows()])
    print(f"{test_year} {icon}  Winner: {actual_winner:<16s} → Predicted: {predicted_winner:<16s} "
          f"({predicted_prob:.0%}) | Actual #{actual_rank} ({actual_prob:.0%})")
    if not correct:
        print(f"       Top 3: {top3_str}")

# Summary
n = len(results)
total_correct = sum(r['correct'] for r in results)
total_top3 = sum(r['in_top3'] for r in results)
total_top5 = sum(r['in_top5'] for r in results)

print("\n" + "="*95)
print("RESULTS SUMMARY")
print("="*95)
print(f"  Exact winner predicted:    {total_correct}/{n} ({total_correct/n*100:.1f}%)")
print(f"  Winner in Top 3:           {total_top3}/{n} ({total_top3/n*100:.1f}%)")
print(f"  Winner in Top 5:           {total_top5}/{n} ({total_top5/n*100:.1f}%)")

auc = roc_auc_score(all_truths, all_probs)
brier = brier_score_loss(all_truths, all_probs)
print(f"  Pooled AUC:                {auc:.4f}")
print(f"  Pooled Brier Score:        {brier:.4f}")

# Per-era
print("\n--- PER-ERA ---")
for name, start, end in [("Pre-1970 (13-16 teams)", 1930, 1966),
                          ("1970-1990 (expansion era)", 1970, 1990),
                          ("1994-2022 (32-team era)", 1994, 2022)]:
    era = [r for r in results if start <= r['year'] <= end]
    c = sum(r['correct'] for r in era)
    t3 = sum(r['in_top3'] for r in era)
    t5 = sum(r['in_top5'] for r in era)
    ne = len(era)
    print(f"  {name}: Exact={c}/{ne} | Top3={t3}/{ne} | Top5={t5}/{ne}")

# Correct picks
print("\n--- CORRECT PICKS ---")
for r in results:
    if r['correct']:
        print(f"  {r['year']}: {r['actual']} (confidence: {r['pred_prob']:.0%})")

# Biggest misses
print("\n--- BIGGEST MISSES ---")
misses = sorted(results, key=lambda r: r['actual_rank'], reverse=True)[:5]
for r in misses:
    print(f"  {r['year']}: {r['actual']} ranked #{r['actual_rank']} ({r['actual_prob']:.0%}) → Model picked {r['predicted']}")

# Surprises (actual winner had low probability but model got it right)
print("\n--- SURPRISE PICKS (correct but low confidence) ---")
for r in sorted(results, key=lambda r: r['actual_prob']):
    if r['correct'] and r['actual_prob'] < 0.8:
        print(f"  {r['year']}: {r['actual']} with only {r['actual_prob']:.0%} confidence")

# Confidence calibration
print("\n--- CONFIDENCE CALIBRATION ---")
for bucket_name, lo, hi in [("Very High (>80%)", 0.8, 1.0),
                              ("High (50-80%)", 0.5, 0.8),
                              ("Medium (20-50%)", 0.2, 0.5),
                              ("Low (<20%)", 0.0, 0.2)]:
    bucket = [r for r in results if lo <= r['pred_prob'] < hi]
    if bucket:
        c = sum(r['correct'] for r in bucket)
        print(f"  {bucket_name}: {c}/{len(bucket)} correct ({c/len(bucket)*100:.0f}%)")