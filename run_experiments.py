#!/usr/bin/env python3
"""Run all model experiments and continue past optional dependency skips."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [
    "experiment_lgbm.py",
    "experiment_nn.py",
    "experiment_calibration.py",
    "experiment_ensemble.py",
    "experiment_draw_model.py",
    "experiment_features.py",
    "experiment_combined.py",
]


def main() -> int:
    failures: list[tuple[str, int]] = []
    for script in EXPERIMENTS:
        print("\n" + "=" * 80)
        print(f"Running {script}")
        print("=" * 80)
        proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=False)
        if proc.returncode != 0:
            failures.append((script, proc.returncode))
            print(f"{script} exited with code {proc.returncode}; continuing.")

    print("\n" + "=" * 80)
    print("Experiment suite complete")
    print("=" * 80)
    if failures:
        print("Failures/skips:")
        for script, code in failures:
            print(f"  {script}: exit code {code}")
    results_path = ROOT / "experiments_results.csv"
    if results_path.exists():
        print(f"\nComparison table: {results_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
