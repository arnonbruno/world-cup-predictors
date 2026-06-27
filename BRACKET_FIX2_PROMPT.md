Fix the 2026 World Cup R32 bracket in shared.py based on the OFFICIAL FIFA bracket from Wikipedia.

## OFFICIAL R32 BRACKET (from Wikipedia - confirmed)
- M73: South Africa vs Canada (2A vs 1B... no, South Africa is 2A and Canada is 2B. Actually: 2A vs 2B is wrong. Let me write the group positions.)

Group winners (1st): A=Mexico, B=Switzerland, C=Brazil, D=USA, E=Germany, F=Netherlands, G=Belgium, H=Spain, I=France, J=Argentina, K=Colombia, L=England
Group runners-up (2nd): A=South Africa, B=Canada, C=Morocco, D=Australia, E=Cote dIvoire, F=Japan, G=Egypt, H=Cape Verde, I=Norway, J=Algeria, K=Portugal, L=Croatia
Best 8 thirds: A=Korea Republic, B=Bosnia and Herzegovina, D=Paraguay, E=Ecuador, F=Sweden, G=IR Iran, J=Austria, L=Ghana

## OFFICIAL R32 MATCHUPS
Match 73: 2A (South Africa) vs 2B (Canada)
Match 74: 1E (Germany) vs 3D (Paraguay)
Match 75: 1F (Netherlands) vs 2C (Morocco)
Match 76: 1C (Brazil) vs 2F (Japan)
Match 77: 1I (France) vs 2F... wait no. Let me be precise from the Wikipedia tables:

From Wikipedia match tables:
- M73: South Africa vs Canada
- M74: Germany vs Paraguay
- M75: Netherlands vs Morocco
- M76: Brazil vs Japan
- M77: France vs Sweden
- M78: Ivory Coast (Cote dIvoire) vs Norway
- M79: Mexico vs 3rd Group C/E
- M80: Winner Group L (England) vs 3rd Group I/J/K
- M81: United States vs Bosnia and Herzegovina
- M82: Belgium vs 3rd Group A/I/J
- M83: Runner-up Group K (Portugal) vs Runner-up Group L (Croatia)
- M84: Spain vs Runner-up Group J (Algeria)
- M85: Switzerland vs 3rd Group G/J
- M86: Argentina vs Cape Verde (CONFIRMED by user and Wikipedia)
- M87: Winner Group K (Colombia) vs 3rd Group E/I/L
- M88: Australia vs Egypt

## YOUR TASK

### 1. Read shared.py first
Focus on the `build_round_of_32()` function and the `THIRD_PLACE_ALLOCATION_MATRIX`.

### 2. Fix the bracket in shared.py
Update `r32_base` list in `build_round_of_32()` to match the official bracket above.

The r32_base is a flat list of 32 team slots (16 pairs = 16 matches):
Index 0,1 = M73, Index 2,3 = M74, ... Index 30,31 = M88

So:
r32_base[0] = gr["A"] (South Africa)
r32_base[1] = gr["B"] (Canada)
r32_base[2] = gw["E"] (Germany)
r32_base[3] = 3rd D (Paraguay) -- this is a third-place slot
r32_base[4] = gw["F"] (Netherlands)
r32_base[5] = gr["C"] (Morocco)
r32_base[6] = gw["C"] (Brazil)
r32_base[7] = gr["F"] (Japan)
r32_base[8] = gw["I"] (France)
r32_base[9] = 3rd F (Sweden) -- third-place slot
r32_base[10] = gr["E"] (Ivory Coast)
r32_base[11] = gr["I"] (Norway)
r32_base[12] = gw["A"] (Mexico)
r32_base[13] = 3rd C or E -- third-place slot
r32_base[14] = gw["L"] (England)
r32_base[15] = 3rd I/J/K -- third-place slot
r32_base[16] = gw["D"] (USA)
r32_base[17] = 3rd B (Bosnia) -- third-place slot
r32_base[18] = gw["G"] (Belgium)
r32_base[19] = 3rd A/I/J -- third-place slot
r32_base[20] = gr["K"] (Portugal)
r32_base[21] = gr["L"] (Croatia)
r32_base[22] = gw["H"] (Spain)
r32_base[23] = gr["J"] (Algeria)
r32_base[24] = gw["B"] (Switzerland)
r32_base[25] = 3rd G/J -- third-place slot
r32_base[26] = gw["J"] (Argentina)
r32_base[27] = 3rd H (Cape Verde) -- third-place slot
r32_base[28] = gw["K"] (Colombia)
r32_base[29] = 3rd E/I/L -- third-place slot
r32_base[30] = gr["D"] (Australia)
r32_base[31] = gr["G"] (Egypt)

### 3. Fix the THIRD_PLACE_ALLOCATION_MATRIX
Update the matrix so that for the combination (A,B,D,E,F,G,J,L):
- D -> slot 3 (M74 third slot)
- F -> slot 9 (M77 third slot)  
- B -> slot 17 (M81 third slot)
- A -> slot 19 (M82 third slot) -- actually A/I/J, so A goes here
- G -> slot 25 (M85 third slot) -- G/J, so G goes here
- H -> slot 27 (M86 third slot) -- BUT H is NOT in best 8 thirds! Cape Verde is 3rd H but H is not in the qualifying combination.

Wait -- this is the issue. The best 8 thirds combination is (A,B,D,E,F,G,J,L), but Group H (Cape Verde) needs to go to slot 27. The matrix must handle this.

Actually, looking again: the third-place allocation for the official bracket uses ALL groups including H. But the FIFA rules say only 8 third-place teams qualify. If the qualifying combination is (A,B,D,E,F,G,J,L), then Group H third (Cape Verde) did NOT qualify.

But the Wikipedia bracket says Argentina vs Cape Verde in M86. This means either:
1. Cape Verde qualified as 2nd place (not 3rd), OR
2. The third-place qualifying combination includes H instead of one of A,B,D,E,F,G,J,L

Let me check: with the updated Group H results (Cape Verde 3pts as 2nd place, not 3rd), Cape Verde is the RUNNER-UP of Group H, not a third-place team.

YES! That is the fix. Cape Verde finished 2nd in Group H. So in M86, Argentina (1J) plays Cape Verde (2H). The bracket already has gr["H"] in the code for that slot, but the issue was that the old data had Saudi Arabia as 2nd in Group H.

So the fix is:
1. M86 = gw["J"] vs gr["H"] (Argentina vs Cape Verde) -- this was ALREADY in the code!
2. The issue was only that the results.csv had wrong Group H standings because the June 26 match results were missing.

Since we already updated the results.csv (Cape Verde 1-1 Saudi Arabia, Uruguay 0-2 Spain), running predict_2026.py should now show Cape Verde as 2H and Argentina vs Cape Verde in M86.

BUT there is also a third-place issue: Saudi Arabia would now be 3rd in Group H (not qualifying as best 8). The best 8 thirds would now include a different combination.

### 4. Also fix any other bracket mismatches
Verify all 16 R32 matchups match the official FIFA bracket.

### 5. Run py_compile and predict_2026.py
Verify everything works and Argentina plays Cape Verde in R32.

### 6. Update README.md with correct bracket and predictions
Update the 2026 predictions section with correct matchups. Date: June 27, 2026.

## CONSTRAINTS
- Read shared.py FIRST before making changes
- All files must parse clean
- Run predict_2026.py to verify R32 matchups are correct
- Argentina MUST play Cape Verde in R32
