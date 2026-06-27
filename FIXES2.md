# Fixes 2 Summary

- Replaced fixed 2026 third-place R32 placement with a full 8-from-12 allocation matrix keyed by the qualifying third-place groups.
- Added missing historical country aliases and ISO coverage, including `Algeria` (`DZA`) and `Dutch East Indies` via `Indonesia`.
- Updated the 2026 training loops to finalize World Cup participation and title history after each completed tournament.
- Excluded winner-relative exploratory columns from legacy `analysis.py` modeling paths.
- Moved the final SOTA regularized-logit LOWCO feature selection inside each held-out tournament fold.
