#!/usr/bin/env python3
"""Compare per-match backtest results between two git states."""
import subprocess, sys, json, os

def run_backtest_capture():
    """Run backtest and capture per-match results."""
    from backtest_2026_wc import run_backtest, BacktestConfig
    config = BacktestConfig(label="comparison", hyperopt_trials=0)
    results, metrics = run_backtest(config=config, verbose=False)
    return results, metrics

def run_on_branch(branch_name):
    """Stash if needed, checkout branch, run backtest, return results."""
    # Ensure clean state
    subprocess.run(["git", "stash"], capture_output=True)
    subprocess.run(["git", "checkout", branch_name], capture_output=True, text=True)
    print(f"\n{'='*60}")
    print(f"  Running backtest on: {branch_name}")
    print(f"{'='*60}")
    
    # Force reimport
    import importlib
    import shared
    import backtest_2026_wc
    importlib.reload(shared)
    importlib.reload(backtest_2026_wc)
    
    results, metrics = run_backtest_capture()
    return results, metrics

if __name__ == "__main__":
    # Run on current branch first
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    print(f"Current branch: {branch}")
    
    results_branch, metrics_branch = run_on_branch(branch)
    
    # Now checkout main
    subprocess.run(["git", "stash"], capture_output=True)
    subprocess.run(["git", "checkout", "main"], capture_output=True, text=True)
    print(f"\n{'='*60}")
    print(f"  Running backtest on: main")
    print(f"{'='*60}")
    
    import importlib
    import shared
    import backtest_2026_wc
    importlib.reload(shared)
    importlib.reload(backtest_2026_wc)
    
    results_main, metrics_main = run_backtest_capture()
    
    # Checkout back to original branch
    subprocess.run(["git", "checkout", branch], capture_output=True)
    subprocess.run(["git", "stash", "pop"], capture_output=True)
    
    # Compare per-match
    print(f"\n{'='*80}")
    print("  PER-MATCH COMPARISON")
    print(f"{'='*80}")
    stage_names = {0: "Group", 1: "R32", 2: "R16", 3: "QF", 4: "SF", 5: "3rd", 6: "Final"}
    
    print(f"\n{'Date':<12} {'Home':<16} {'Away':<16} {'Score':<6} {'Stg':<5} {'Main':<6} {'Branch':<6} {'Same?':<6} {'MainProb':>8} {'BranchProb':>10}")
    print("-" * 100)
    
    flips = []
    for rm, rb in zip(results_main, results_branch):
        same = "✓" if rm["correct"] == rb["correct"] else "✗ FLIP"
        main_mark = "✓" if rm["correct"] else "✗"
        branch_mark = "✓" if rb["correct"] else "✗"
        stg = stage_names.get(rm["stage"], "?")
        
        if rm["correct"] != rb["correct"]:
            flips.append({
                "match": f"{rm['home']} vs {rm['away']}",
                "score": rm["score"],
                "stage": stg,
                "main_correct": rm["correct"],
                "branch_correct": rb["correct"],
                "main_pred": rm["predicted"],
                "branch_pred": rb["predicted"],
                "actual": rm["actual"],
                "main_probs": (rm["p_home"], rm["p_draw"], rm["p_away"]),
                "branch_probs": (rb["p_home"], rb["p_draw"], rb["p_away"]),
            })
        
        print(f"{rm['date']:<12} {rm['home']:<16} {rm['away']:<16} {rm['score']:<6} {stg:<5} {main_mark:<6} {branch_mark:<6} {same:<6} {rm['actual_prob']:>8.1%} {rb['actual_prob']:>10.1%}")
    
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"  Main:   Accuracy={metrics_main.accuracy:.1%} ({metrics_main.correct}/{metrics_main.total})  LogLoss={metrics_main.log_loss:.4f}  Brier={metrics_main.brier:.4f}")
    print(f"  Branch: Accuracy={metrics_branch.accuracy:.1%} ({metrics_branch.correct}/{metrics_branch.total})  LogLoss={metrics_branch.log_loss:.4f}  Brier={metrics_branch.brier:.4f}")
    
    print(f"\n  Per-stage:")
    for s in sorted(set(list(metrics_main.stage_metrics.keys()) + list(metrics_branch.stage_metrics.keys()))):
        sm = metrics_main.stage_metrics.get(s, {"correct":0,"total":0,"log_losses":[]})
        sb = metrics_branch.stage_metrics.get(s, {"correct":0,"total":0,"log_losses":[]})
        name = stage_names.get(s, f"Stage {s}")
        print(f"    {name:<10}: Main {sm['correct']}/{sm['total']}  vs  Branch {sb['correct']}/{sb['total']}")
    
    print(f"\n  Flipped matches ({len(flips)}):")
    for f in flips:
        print(f"    {f['match']} ({f['score']}, {f['stage']})")
        print(f"      Main:   {f['main_pred']:>5}  probs=({f['main_probs'][0]:.1%}, {f['main_probs'][1]:.1%}, {f['main_probs'][2]:.1%})  {'CORRECT' if f['main_correct'] else 'WRONG'}")
        print(f"      Branch: {f['branch_pred']:>5}  probs=({f['branch_probs'][0]:.1%}, {f['branch_probs'][1]:.1%}, {f['branch_probs'][2]:.1%})  {'CORRECT' if f['branch_correct'] else 'WRONG'}")
        print(f"      Actual: {f['actual']}")
