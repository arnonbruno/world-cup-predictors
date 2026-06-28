#!/usr/bin/env python3
"""
Daily World Cup 2026 update script.
Fetches latest results from Wikipedia, updates results.csv, runs predictions.
Outputs a summary for the agent to use in README update.
"""

import pandas as pd
import urllib.request
import re
import subprocess
import sys
import os
from datetime import datetime

WORKDIR = '/var/mnt/DATA/Hermes/workspace/world-cup-predictors'
os.chdir(WORKDIR)

def fetch_group_results(group_letter):
    """Fetch all match results for a group from Wikipedia."""
    url = f'https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{group_letter}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = urllib.request.urlopen(req, timeout=15).read().decode()
    
    # Find all footballbox sections
    pattern = r'class="fhome[^"]*"[^>]*>(.*?)</td>.*?class="fscore"[^>]*>(.*?)</td>.*?class="faway[^"]*"[^>]*>(.*?)</td>'
    matches = re.findall(pattern, data, re.DOTALL)
    
    # Find dates
    dates = re.findall(r'class="fdate"[^>]*>(.*?)</(?:div|span)', data, re.DOTALL)
    
    results = []
    for i, (home_html, score_html, away_html) in enumerate(matches):
        # Clean team names
        home = re.sub(r'<[^>]+>', '', home_html).strip()
        away = re.sub(r'<[^>]+>', '', away_html).strip()
        home = re.sub(r'&#160;', ' ', home).strip()
        away = re.sub(r'&#160;', ' ', away).strip()
        
        # Clean score - look for X-Y or X–Y pattern
        score_clean = re.sub(r'<[^>]+>', ' ', score_html)
        score_clean = re.sub(r'&#160;', '', score_clean)
        score_clean = re.sub(r'&#8211;', '-', score_clean)
        
        # Try to find score pattern
        score_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', score_clean)
        if not score_match:
            continue
        
        s1, s2 = int(score_match.group(1)), int(score_match.group(2))
        
        # Get date
        date_str = ''
        if i < len(dates):
            date_clean = re.sub(r'<[^>]+>', '', dates[i]).strip()
            date_clean = re.sub(r'&#160;', ' ', date_clean)
            # Extract date
            dm = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date_clean)
            if dm:
                day, month, year = dm.group(1), dm.group(2), dm.group(3)
                try:
                    dt = datetime.strptime(f'{day} {month} {year}', '%d %B %Y')
                    date_str = dt.strftime('%Y-%m-%d')
                except:
                    pass
        
        results.append({
            'date': date_str,
            'home_team': home,
            'away_team': away,
            'home_score': s1,
            'away_score': s2,
            'group': group_letter
        })
    
    return results


def update_results_csv(new_results):
    """Update results.csv with new match results."""
    df = pd.read_csv('data/results.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    updated = 0
    added = 0
    
    for r in new_results:
        if not r['date']:
            continue
        
        match_date = pd.to_datetime(r['date'])
        
        # Check if this match exists (by date + teams)
        mask = (
            (df['date'] == match_date) &
            (df['home_team'] == r['home_team']) &
            (df['away_team'] == r['away_team'])
        )
        
        existing = df[mask]
        
        if len(existing) > 0:
            # Update if NaN or different
            idx = existing.index[0]
            if pd.isna(df.loc[idx, 'home_score']) or df.loc[idx, 'home_score'] != r['home_score']:
                df.loc[idx, 'home_score'] = r['home_score']
                df.loc[idx, 'away_score'] = r['away_score']
                updated += 1
        else:
            # Add new row
            new_row = pd.DataFrame([{
                'date': match_date,
                'home_team': r['home_team'],
                'away_team': r['away_team'],
                'home_score': r['home_score'],
                'away_score': r['away_score'],
                'tournament': 'FIFA World Cup',
                'city': '',
                'country': '',
                'neutral': True
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            added += 1
    
    if updated > 0 or added > 0:
        df = df.sort_values('date').reset_index(drop=True)
        df.to_csv('data/results.csv', index=False)
    
    return updated, added


def run_backtest():
    """Run the walk-forward backtest and return accuracy for recent matches."""
    result = subprocess.run(
        ['python3', 'backtest_2026_wc.py'],
        capture_output=True, text=True, timeout=300
    )
    output = result.stdout + result.stderr
    
    # Parse final accuracy
    acc_match = re.search(r'Final accuracy: ([\d.]+)% \((\d+)/(\d+)\)', output)
    ll_match = re.search(r'Log-loss: ([\d.]+)', output)
    br_match = re.search(r'Brier: ([\d.]+)', output)
    
    accuracy = acc_match.group(1) if acc_match else '?'
    correct = acc_match.group(2) if acc_match else '?'
    total = acc_match.group(3) if acc_match else '?'
    logloss = ll_match.group(1) if ll_match else '?'
    brier = br_match.group(1) if br_match else '?'
    
    # Extract last day matches
    last_day_matches = []
    lines = output.split('\n')
    for line in lines:
        if '2026-06-2' in line and ('✅' in line or '❌' in line):
            last_day_matches.append(line.strip())
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'logloss': logloss,
        'brier': brier,
        'last_day_matches': last_day_matches,
        'full_output': output
    }


def run_predictions():
    """Run predict_2026.py and capture output."""
    result = subprocess.run(
        ['python3', 'predict_2026.py'],
        capture_output=True, text=True, timeout=300
    )
    output = result.stdout + result.stderr
    
    # Extract key predictions
    lines = output.split('\n')
    
    champion = '?'
    runner_up = '?'
    third = '?'
    fourth = '?'
    
    for line in lines:
        if 'CHAMPION' in line:
            m = re.search(r'CHAMPION:\s+(\w[\w\s]+)', line)
            if m: champion = m.group(1).strip()
        elif 'Runner-up' in line:
            m = re.search(r'Runner-up:\s+(\w[\w\s]+)', line)
            if m: runner_up = m.group(1).strip()
        elif 'Third' in line:
            m = re.search(r'Third:\s+(\w[\w\s]+)', line)
            if m: third = m.group(1).strip()
        elif '4th place' in line:
            m = re.search(r'4th place:\s+(\w[\w\s]+)', line)
            if m: fourth = m.group(1).strip()
    
    # Extract R32 predictions
    r32_matches = []
    in_r32 = False
    for line in lines:
        if 'ROUND OF 32' in line:
            in_r32 = True
            continue
        if 'ROUND OF 16' in line:
            in_r32 = False
            continue
        if in_r32 and '→' in line:
            r32_matches.append(line.strip())
    
    # Extract full bracket path for top 4
    paths = []
    for line in lines:
        if any(name in line for name in [champion, runner_up, third, fourth]):
            if '→' in line and ':' in line:
                paths.append(line.strip())
    
    return {
        'champion': champion,
        'runner_up': runner_up,
        'third': third,
        'fourth': fourth,
        'r32_matches': r32_matches,
        'paths': paths,
        'full_output': output
    }


def main():
    print("=" * 60)
    print(f"WORLD CUP 2026 DAILY UPDATE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Step 1: Fetch latest results from Wikipedia
    print("\n[1/4] Fetching latest results from Wikipedia...")
    all_new = []
    for g in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        try:
            results = fetch_group_results(g)
            all_new.extend(results)
            print(f"  Group {g}: {len(results)} matches found")
        except Exception as e:
            print(f"  Group {g}: ERROR - {e}")
    
    # Step 2: Update results.csv
    print(f"\n[2/4] Updating results.csv ({len(all_new)} matches from Wikipedia)...")
    updated, added = update_results_csv(all_new)
    print(f"  Updated: {updated}, Added: {added}")
    
    # Step 3: Run backtest
    print("\n[3/4] Running backtest...")
    backtest = run_backtest()
    print(f"  Overall: {backtest['accuracy']}% ({backtest['correct']}/{backtest['total']})")
    print(f"  Log-loss: {backtest['logloss']}, Brier: {backtest['brier']}")
    
    # Step 4: Run predictions
    print("\n[4/4] Running predictions...")
    preds = run_predictions()
    print(f"  Champion: {preds['champion']}")
    print(f"  Runner-up: {preds['runner_up']}")
    print(f"  Third: {preds['third']}")
    print(f"  Fourth: {preds['fourth']}")
    
    # Output summary for agent
    print("\n" + "=" * 60)
    print("SUMMARY FOR README UPDATE")
    print("=" * 60)
    print(f"DATE: {datetime.now().strftime('%B %d, %Y')}")
    print(f"CHAMPION: {preds['champion']}")
    print(f"RUNNER_UP: {preds['runner_up']}")
    print(f"THIRD: {preds['third']}")
    print(f"FOURTH: {preds['fourth']}")
    print(f"OVERALL_ACCURACY: {backtest['accuracy']}% ({backtest['correct']}/{backtest['total']})")
    print(f"LOG_LOSS: {backtest['logloss']}")
    print(f"BRIER: {backtest['brier']}")
    
    if backtest['last_day_matches']:
        print(f"\nLAST_DAY_MATCHES ({len(backtest['last_day_matches'])}):")
        for m in backtest['last_day_matches']:
            print(f"  {m}")
    
    if preds['r32_matches']:
        print(f"\nR32_PREDICTIONS ({len(preds['r32_matches'])}):")
        for m in preds['r32_matches']:
            print(f"  {m}")
    
    # Count last day accuracy
    correct_last = sum(1 for m in backtest['last_day_matches'] if '✅' in m)
    total_last = len(backtest['last_day_matches'])
    if total_last > 0:
        print(f"\nLAST_DAY_ACCURACY: {correct_last}/{total_last} ({100*correct_last/total_last:.1f}%)")


if __name__ == '__main__':
    main()
