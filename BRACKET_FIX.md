# 2026 R32 Bracket Fix

Updated: June 27, 2026

## What Changed

`shared.py` now uses the corrected FIFA 2026 Round of 32 structure:

- Match 77 is `1I vs 2H`.
- Match 86 is `1J vs 3C/3D/3F/3G/3H`.
- The third-place placeholder formerly attached to Match 77 was moved to Match 86.
- Third-place allocation now uses explicit candidate ordering instead of alphabetical tie-breaking, so the current Group H third-place qualifier maps to Match 86.

This fixes the incorrect `Argentina vs Saudi Arabia` R32 matchup. With Group J winner Argentina and Group H third-place qualifier Cape Verde, Match 86 is now `Argentina vs Cape Verde`.

## Correct Round of 32 Matchups

Using the June 27 standings projection:

| Match | Teams |
|-------|-------|
| Match 73 | South Africa vs Canada |
| Match 74 | Germany vs Paraguay |
| Match 75 | Netherlands vs Morocco |
| Match 76 | Brazil vs Japan |
| Match 77 | France vs Saudi Arabia |
| Match 78 | Côte d'Ivoire vs Norway |
| Match 79 | Mexico vs Sweden |
| Match 80 | England vs Ecuador |
| Match 81 | USA vs Bosnia and Herzegovina |
| Match 82 | Belgium vs Korea Republic |
| Match 83 | Portugal vs Croatia |
| Match 84 | Spain vs Algeria |
| Match 85 | Switzerland vs IR Iran |
| Match 86 | Argentina vs Cape Verde |
| Match 87 | Colombia vs Ghana |
| Match 88 | Australia vs Egypt |

## Verification

Required commands:

```bash
python3 -m py_compile shared.py predict_2026.py
python3 predict_2026.py
```

In this session, Python execution was rejected by the command runner, so these checks still need to be run locally.
