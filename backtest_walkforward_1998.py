#!/usr/bin/env python3
"""Lighter 1998+ walk-forward backtest without external market/value data."""

from __future__ import annotations

import warnings
from pathlib import Path

from backtest_walkforward import run_backtest


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        run_backtest(
            start_year=1998,
            use_external_data=False,
            results_path=Path("backtest_walkforward_1998_results.csv"),
            summary_path=Path("backtest_walkforward_1998_summary.md"),
            title="Walk-Forward Backtest (1998+, No Odds/Squad Values)",
        )


if __name__ == "__main__":
    main()
