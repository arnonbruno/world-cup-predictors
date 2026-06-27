Fix the 2026 World Cup R32 bracket structure in shared.py.

## PROBLEM
The current R32 bracket in `build_round_of_32()` is incorrect. As of June 27, the model produces wrong matchups. The user confirmed that Argentina (1J) should play Cape Verde (best third from Group H), not Saudi Arabia (2H).

## YOUR TASK

### 1. Research the correct 2026 WC bracket
Look up the actual 2026 FIFA World Cup Round of 32 bracket. The tournament uses a 48-team format with 12 groups of 4. The top 2 from each group (24 teams) plus the 8 best third-place teams (8 teams) form a 32-team knockout bracket.

Search the web for "2026 FIFA World Cup bracket" or "2026 World Cup Round of 32 matchups" to find the official bracket structure. FIFA published the full bracket before the tournament started.

### 2. Fix shared.py
Update `build_round_of_32()` to use the correct bracket pairings. The base R32 list should match the actual FIFA bracket.

### 3. Also fix the THIRD_PLACE_ALLOCATION_MATRIX if needed
The matrix determines which third-place teams go to which slots. Verify it matches the FIFA allocation rules.

### 4. Run py_compile on shared.py and predict_2026.py
Verify both files parse clean.

### 5. Run predict_2026.py
Verify the R32 matchups are now correct. Argentina should play Cape Verde in R32.

### 6. Update README.md
After verifying predictions are correct, update README.md with:
- Correct R32 matchups
- Updated prediction (1st, 2nd, 3rd) with date (June 27, 2026)
- Any other bracket changes

### 7. Write BRACKET_FIX.md
Document what changed and the new R32 matchups.

## CURRENT GROUP STANDINGS (as of June 27)
Group A: 1st Mexico, 2nd South Africa, 3rd Korea Republic
Group B: 1st Switzerland, 2nd Canada, 3rd Bosnia and Herzegovina
Group C: 1st Brazil, 2nd Morocco, 3rd Scotland
Group D: 1st USA, 2nd Australia, 3rd Paraguay
Group E: 1st Germany, 2nd Cote dIvoire, 3rd Ecuador
Group F: 1st Netherlands, 2nd Japan, 3rd Sweden
Group G: 1st Belgium, 2nd Egypt, 3rd IR Iran
Group H: 1st Spain, 2nd Saudi Arabia, 3rd Cape Verde
Group I: 1st France, 2nd Norway, 3rd Senegal
Group J: 1st Argentina, 2nd Algeria, 3rd Austria
Group K: 1st Colombia, 2nd Portugal, 3rd Uzbekistan
Group L: 1st England, 2nd Croatia, 3rd Ghana

Best 8 thirds from groups: A, B, D, E, F, G, J, L (Korea Republic, Bosnia, Paraguay, Ecuador, Sweden, IR Iran, Austria, Ghana)

## CONSTRAINTS
- Read shared.py FIRST before making any changes
- The bracket structure must match the official FIFA 2026 WC bracket exactly
- All files must parse clean (python3 -m py_compile)
- Run predict_2026.py to verify the output shows correct R32 matchups
- Do NOT change anything else in the prediction pipeline, only the bracket structure
