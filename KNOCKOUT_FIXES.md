# Knockout Prediction Fixes

## Summary

Implemented conservative knockout-specific fixes without changing the Dixon-Coles model itself:

- Knockout matches now use `KNOCKOUT_ALPHA = 0.50`, a 50/50 XGBoost and Dixon-Coles blend in this codebase's alpha convention.
- Group-stage predictions continue to use the existing tuned blend alpha.
- Head-to-head features now use only the last 15 years and apply exponential time decay with a 10-year half-life.
- Country feature lookups now request the 2026 feature vintage and flag stale rows older than 8 years.
- World Cup calibration buckets are computed from `backtest_walkforward_results.csv` and applied to knockout favorite confidence.
- Knockout output now prints raw blended 3-way probabilities, no-draw conditional probabilities, WC-calibrated probabilities, and missing-odds warnings.
- `explain_match.py` now follows the same knockout blend path and reports raw probabilities separately.
- `backtest_2026_wc.py` uses the same knockout blend rule for consistency.

## Before

| Match | Production knockout | Production calibrated | Known XGBoost-only reference | Notes |
| --- | ---: | ---: | ---: | --- |
| Brazil vs Japan | 83.2% Brazil | 86.0% Brazil | 63.4% Brazil | Missing knockout odds; Brazil-Japan H2H was using ancient matches at full weight. |
| Brazil vs Norway | 89.7% Brazil | 92.0% Brazil | unavailable | Missing knockout odds; Norway country features were from 1998. |

## After Implementation

The new runtime report distinguishes:

- `raw 3-way`: blended regulation-time home/draw/away probabilities before draw removal.
- `knockout no-draw`: conditional home/away probabilities after draw removal.
- `WC-calibrated`: final displayed knockout probabilities after World Cup-specific calibration.
- `WARNING: betting odds are unavailable`: shown when odds-derived features are NaN.

Expected directional impact:

- Brazil vs Japan should move down from the prior 83-86% range because Dixon-Coles no longer dominates the knockout blend and old H2H is decayed.
- Brazil vs Norway should move down from the prior 90% range because Norway is no longer silently treated as having current 1998 country features; stale country data is explicitly flagged.
- Group-stage probabilities should remain on the same tuned alpha path.

## Validation Status

Python execution was blocked in this environment, including `py_compile`, `predict_2026.py`, and `backtest_2026_wc.py`, so new numeric Brazil probabilities could not be honestly recomputed here.

Commands attempted:

```bash
python3 -m py_compile shared.py predict_2026.py explain_match.py backtest_2026_wc.py
python3 predict_2026.py
python3 backtest_2026_wc.py
```

The code paths are set up so a successful `python3 predict_2026.py` run will print the requested raw and calibrated values for Brazil vs Japan and Brazil vs Norway directly in the knockout bracket output.

