# HorseCash accuracy report

All numbers are out of sample: every prediction comes from models fit only on earlier years (yearly walk-forward).
Log-loss = mean of -log(probability given to the winner); lower is better. Top-1 = the model's most likely horse won.

## Key findings (France hold-out 2025-01 -> 2026-05, 13,760 races)

- **Usable 30 minutes before the off:** the end model's top pick wins 30.9%. The favourite in the market at that time wins 30.4% (+0.5%, CI +0.0% to +1.0%). Its probabilities are clearly sharper (log-loss 1.979 vs 2.023).
- **Against the final pool odds** (known only at the off): the final favourite wins 33.0%, because money placed in the last 30 minutes is informed. Given the final odds as input too, the model matches that hit rate (33.0%) and still has better probabilities (log-loss -0.0107, CI -0.0132 to -0.0080). So the model holds information the final market lacks.
- **The fundamental model alone** (never sees this race's odds) picks the winner 28.0% of the time against 8.3% for a random pick in a 12-horse field.
- **Accuracy is not profit.** The pool keeps about 16.5 %. Betting the end model's top pick in every race returns -11.3% at an average price of 3.43, while the market favourite returns -10.6%. The best value rule chosen on 2018-24 (bet when the model sees more than 30% expected value at the 30-minute odds) returns -3.5% on 3,422 bets: a much smaller loss than the ~16.5 % the pool takes on average, but still a loss.
- **Hong Kong**, a much tighter market, gives the same picture with smaller numbers. The model adds almost nothing to the final odds there.

## France (PMU, 2014-2026)

### Hold-out: 2025-01-01 -> 2026-05-05 (never used for any choice)

| Model | Races | Top-1 (winner picked) | Winner in top 3 | Log-loss |
|---|---|---|---|---|
| Market, final odds (known only at the off) | 13,760 | 33.0% | 64.8% | 1.9321 |
| Market, reference odds (~30 min before) | 13,760 | 30.4% | 61.8% | 2.0230 |
| Market, reference odds, longshot bias corrected | 13,760 | 30.4% | 61.8% | 2.0199 |
| Fundamental model (no odds of the race) | 13,760 | 28.0% | 59.4% | 2.0378 |
| Fundamental + reference odds (conditional logit) | 13,760 | 30.5% | 62.8% | 1.9883 |
| **END MODEL**: fundamental + reference odds + residual booster | 13,760 | 30.9% | 62.9% | 1.9794 |
| Fundamental + final odds + residual booster | 13,760 | 33.0% | 65.0% | 1.9214 |

Against the final market:

| Model vs benchmark | Log-loss diff [95 % CI] | Top-1 diff [95 % CI] |
|---|---|---|
| Fundamental model (no odds of the race) | +0.1056 [+0.0976, +0.1135] | -4.94% [-5.66%, -4.21%] |
| **END MODEL**: fundamental + reference odds + residual booster | +0.0472 [+0.0416, +0.0530] | -2.05% [-2.61%, -1.46%] |
| Fundamental + final odds + residual booster | -0.0107 [-0.0132, -0.0080] | +0.03% [-0.35%, +0.40%] |
| Fundamental + reference odds (conditional logit) | +0.0562 [+0.0504, +0.0619] | -2.50% [-3.10%, -1.95%] |

Against the reference market (what is known 30 minutes before the off):

| Model vs benchmark | Log-loss diff [95 % CI] | Top-1 diff [95 % CI] |
|---|---|---|
| Fundamental model (no odds of the race) | +0.0148 [+0.0063, +0.0229] | -2.39% [-3.08%, -1.72%] |
| **END MODEL**: fundamental + reference odds + residual booster | -0.0436 [-0.0476, -0.0395] | +0.50% [+0.04%, +0.99%] |
| Market, reference odds, longshot bias corrected | -0.0030 [-0.0043, -0.0018] | +0.00% [+0.00%, +0.00%] |

### Development years 2018-2024

| Model | Races | Top-1 (winner picked) | Winner in top 3 | Log-loss |
|---|---|---|---|---|
| Market, final odds (known only at the off) | 70,279 | 31.8% | 63.8% | 1.9494 |
| Market, reference odds (~30 min before) | 70,279 | 30.0% | 61.3% | 2.0223 |
| Market, reference odds, longshot bias corrected | 70,279 | 30.0% | 61.3% | 2.0186 |
| Fundamental model (no odds of the race) | 70,279 | 27.7% | 58.6% | 2.0527 |
| Fundamental + reference odds (conditional logit) | 70,279 | 30.3% | 62.0% | 1.9905 |
| **END MODEL**: fundamental + reference odds + residual booster | 70,279 | 30.3% | 62.1% | 1.9831 |
| Fundamental + final odds + residual booster | 70,279 | 31.8% | 64.1% | 1.9394 |

### By discipline (hold-out)

| Discipline | Races | Final market | Ref market | Fundamental | End model (ref) | End model (final) |
|---|---|---|---|---|---|---|
| OBSTACLE | 1,540 | 32.3% / 1.879 | 29.7% / 1.954 | 29.2% / 1.954 | 31.7% / 1.913 | 32.6% / 1.870 |
| PLAT | 5,160 | 29.3% / 2.012 | 26.7% / 2.103 | 25.1% / 2.109 | 26.8% / 2.066 | 29.6% / 2.004 |
| TROT_ATTELE | 5,908 | 35.8% / 1.892 | 33.1% / 1.989 | 29.6% / 2.012 | 33.5% / 1.936 | 35.6% / 1.877 |
| TROT_MONTE | 1,152 | 36.0% / 1.854 | 34.7% / 1.931 | 31.4% / 1.963 | 35.0% / 1.901 | 35.5% / 1.846 |

(top-1 / log-loss)

### Per year

| Year | Races | Final market | Ref market | Fundamental | End model (ref) | End model (final) |
|---|---|---|---|---|---|---|
| 2018 | 10,878 | 31.6% / 1.960 | 30.2% / 2.024 | 27.9% / 2.058 | 30.5% / 1.983 | 31.4% / 1.956 |
| 2019 | 9,691 | 31.2% / 1.970 | 30.0% / 2.040 | 27.0% / 2.077 | 30.1% / 2.003 | 31.5% / 1.962 |
| 2020 | 8,754 | 32.4% / 1.938 | 30.0% / 2.016 | 28.2% / 2.054 | 30.3% / 1.976 | 32.2% / 1.928 |
| 2021 | 10,083 | 31.8% / 1.952 | 29.6% / 2.023 | 27.3% / 2.058 | 30.2% / 1.987 | 31.9% / 1.944 |
| 2022 | 10,206 | 31.8% / 1.933 | 30.9% / 2.003 | 28.5% / 2.027 | 30.6% / 1.962 | 32.0% / 1.920 |
| 2023 | 10,330 | 31.8% / 1.941 | 29.5% / 2.018 | 28.1% / 2.040 | 30.4% / 1.977 | 32.0% / 1.927 |
| 2024 | 10,337 | 32.1% / 1.950 | 29.5% / 2.032 | 26.7% / 2.056 | 29.6% / 1.994 | 31.6% / 1.938 |
| 2025 | 10,334 | 32.3% / 1.946 | 29.7% / 2.035 | 27.5% / 2.047 | 30.4% / 1.992 | 32.4% / 1.936 |
| 2026 | 3,426 | 35.1% / 1.891 | 32.5% / 1.986 | 29.6% / 2.010 | 32.3% / 1.942 | 34.8% / 1.878 |

### Calibration of the end model (hold-out)

| Predicted | Runners | Mean predicted | Actual win rate |
|---|---|---|---|
| (0.0, 0.02] | 36,425 | 0.009 | 0.009 |
| (0.02, 0.05] | 39,478 | 0.034 | 0.032 |
| (0.05, 0.1] | 38,732 | 0.072 | 0.072 |
| (0.1, 0.15] | 19,935 | 0.122 | 0.123 |
| (0.15, 0.2] | 10,940 | 0.173 | 0.179 |
| (0.2, 0.3] | 10,169 | 0.241 | 0.243 |
| (0.3, 0.4] | 3,633 | 0.342 | 0.350 |
| (0.4, 0.6] | 2,180 | 0.474 | 0.462 |
| (0.6, 1.0] | 339 | 0.666 | 0.625 |

### Top pick vs odds vs profit (hold-out)

One unit on each model's top pick in every race, paid at the final pool odds (in a pari-mutuel pool every bet gets the final odds).
Break-even hit rate = average of 1/odds of the picks: the hit rate needed to make money at those prices.

| Model | Top-pick hit rate | Avg odds of the pick | Break-even hit rate | ROI betting the pick every race | Same pick as market favourite | When it differs: races / hit / odds / ROI |
|---|---|---|---|---|---|---|
| Market, final odds (known only at the off) | 33.0% | 3.06 | 36.6% | -10.6% | 100% | - |
| Market, reference odds (~30 min before) | 30.4% | 3.50 | 34.3% | -12.4% | 72% | 3,805 / 17.6% / 5.24 / -16.8% |
| Fundamental model (no odds of the race) | 28.0% | 4.16 | 31.4% | -11.8% | 58% | 5,743 / 16.5% / 6.09 / -13.7% |
| Fundamental + reference odds (conditional logit) | 30.5% | 3.42 | 34.5% | -13.4% | 74% | 3,602 / 17.4% / 5.02 / -20.0% |
| **END MODEL**: fundamental + reference odds + residual booster | 30.9% | 3.43 | 34.5% | -11.3% | 73% | 3,676 / 18.3% / 5.05 / -15.1% |
| Fundamental + final odds + residual booster | 33.0% | 3.12 | 36.2% | -9.2% | 88% | 1,707 / 21.3% / 4.53 / -10.1% |

End model's top pick, split by its odds and by its own probability:

| Pick's final odds | Races | Hit rate | Break-even | ROI |
|---|---|---|---|---|
| 1-2 | 1,857 | 53.7% | 60.3% | -11.2% |
| 2-3 | 4,291 | 38.4% | 41.3% | -7.0% |
| 3-5 | 5,703 | 23.3% | 27.0% | -14.2% |
| 5-10 | 1,856 | 14.9% | 16.7% | -10.9% |
| 10-20 | 52 | 1.9% | 9.0% | -78.8% |
| 20+ | 1 | 0.0% | 4.8% | -100.0% |

| Model probability of its pick | Races | Hit rate | Avg odds | ROI |
|---|---|---|---|---|
| 0.00-0.15 | 428 | 14.7% | 6.68 | -12.1% |
| 0.15-0.25 | 4,840 | 21.4% | 4.44 | -13.5% |
| 0.25-0.35 | 4,638 | 30.7% | 3.15 | -10.3% |
| 0.35-0.50 | 2,825 | 41.2% | 2.31 | -9.9% |
| 0.50-0.70 | 949 | 53.9% | 1.73 | -9.4% |
| 0.70-1.00 | 80 | 65.0% | 1.40 | -11.1% |

Development years 2018-2024:

| Model | Top-pick hit rate | Avg odds of the pick | Break-even hit rate | ROI betting the pick every race | Same pick as market favourite | When it differs: races / hit / odds / ROI |
|---|---|---|---|---|---|---|
| Market, final odds (known only at the off) | 31.8% | 3.09 | 36.1% | -13.2% | 100% | - |
| Market, reference odds (~30 min before) | 30.0% | 3.44 | 34.3% | -14.2% | 75% | 17,343 / 18.0% / 5.13 / -16.5% |
| Fundamental model (no odds of the race) | 27.7% | 4.34 | 30.7% | -10.2% | 56% | 30,625 / 17.1% / 6.33 / -9.0% |
| **END MODEL**: fundamental + reference odds + residual booster | 30.3% | 3.44 | 34.3% | -12.5% | 75% | 17,819 / 18.8% / 5.07 / -12.3% |
| Fundamental + final odds + residual booster | 31.8% | 3.19 | 35.5% | -11.2% | 85% | 10,814 / 21.9% / 4.58 / -5.8% |

End model's top pick, split by its odds and by its own probability:

| Pick's final odds | Races | Hit rate | Break-even | ROI |
|---|---|---|---|---|
| 1-2 | 8,850 | 56.4% | 59.9% | -6.2% |
| 2-3 | 22,248 | 36.2% | 41.2% | -12.3% |
| 3-5 | 29,804 | 23.2% | 27.1% | -14.4% |
| 5-10 | 9,047 | 14.1% | 16.6% | -14.2% |
| 10-20 | 325 | 10.2% | 9.2% | +11.1% |
| 20+ | 5 | 20.0% | 3.2% | +740.0% |

| Model probability of its pick | Races | Hit rate | Avg odds | ROI |
|---|---|---|---|---|
| 0.00-0.15 | 2,425 | 13.4% | 6.55 | -19.0% |
| 0.15-0.25 | 23,882 | 20.6% | 4.46 | -15.3% |
| 0.25-0.35 | 24,480 | 29.3% | 3.16 | -12.4% |
| 0.35-0.50 | 14,185 | 41.1% | 2.31 | -9.8% |
| 0.50-0.70 | 4,704 | 55.7% | 1.75 | -5.2% |
| 0.70-1.00 | 603 | 69.8% | 1.41 | -3.1% |

### Betting check (secondary)

Decide 30 minutes before the off (end-model probability x reference odds - 1 > threshold), get paid the final pool odds.
Threshold chosen on 2018-24: 0.3. Hold-out: 3,422 bets, ROI -3.5% (market-only rule, same threshold: 265 bets, ROI -29.2%).

## Hong Kong (HKJC, 1979-2018): replication on a second market

Only final odds exist here, so the end model uses them.

### Hold-out: 2017-2018

| Model | Races | Top-1 (winner picked) | Winner in top 3 | Log-loss |
|---|---|---|---|---|
| Market, final odds (known only at the off) | 1,270 | 29.9% | 62.3% | 2.0013 |
| Market, final odds, longshot bias corrected | 1,270 | 29.9% | 62.3% | 1.9997 |
| Fundamental model (no odds of the race) | 1,270 | 25.6% | 52.7% | 2.1705 |
| Fundamental + final odds (conditional logit) | 1,270 | 30.2% | 61.6% | 2.0020 |
| **END MODEL**: fundamental + final odds + residual booster | 1,270 | 29.4% | 62.3% | 1.9990 |

| Model vs benchmark | Log-loss diff [95 % CI] | Top-1 diff [95 % CI] |
|---|---|---|
| Fundamental model (no odds of the race) | +0.1692 [+0.1377, +0.2020] | -4.33% [-6.83%, -1.83%] |
| Fundamental + final odds (conditional logit) | +0.0007 [-0.0019, +0.0034] | +0.24% [-0.55%, +1.02%] |
| **END MODEL**: fundamental + final odds + residual booster | -0.0023 [-0.0071, +0.0027] | -0.55% [-1.68%, +0.48%] |
| Market, final odds, longshot bias corrected | -0.0016 [-0.0030, -0.0002] | +0.00% [+0.00%, +0.00%] |

### Development years 1996-2016

| Model | Races | Top-1 (winner picked) | Winner in top 3 | Log-loss |
|---|---|---|---|---|
| Market, final odds (known only at the off) | 13,980 | 29.3% | 59.3% | 2.0506 |
| Market, final odds, longshot bias corrected | 13,980 | 29.3% | 59.3% | 2.0503 |
| Fundamental model (no odds of the race) | 13,980 | 24.5% | 53.3% | 2.1897 |
| Fundamental + final odds (conditional logit) | 13,980 | 29.4% | 59.7% | 2.0499 |
| **END MODEL**: fundamental + final odds + residual booster | 13,980 | 29.4% | 59.7% | 2.0480 |

| Model vs benchmark | Log-loss diff [95 % CI] | Top-1 diff [95 % CI] |
|---|---|---|
| Fundamental model (no odds of the race) | +0.1390 [+0.1294, +0.1478] | -4.76% [-5.47%, -3.99%] |
| Fundamental + final odds (conditional logit) | -0.0007 [-0.0020, +0.0007] | +0.06% [-0.22%, +0.34%] |
| **END MODEL**: fundamental + final odds + residual booster | -0.0026 [-0.0051, -0.0000] | +0.15% [-0.23%, +0.54%] |
| Market, final odds, longshot bias corrected | -0.0003 [-0.0006, -0.0001] | +0.00% [+0.00%, +0.00%] |

### Top pick vs odds vs profit (hold-out 2017-2018)

Paid at the final odds. The end model here uses the final odds to choose, so its ROI is an upper bound; the fundamental model never sees odds.

| Model | Top-pick hit rate | Avg odds of the pick | Break-even hit rate | ROI betting the pick every race | Same pick as market favourite | When it differs: races / hit / odds / ROI |
|---|---|---|---|---|---|---|
| Market, final odds (known only at the off) | 29.9% | 3.04 | 36.1% | -19.4% | 100% | - |
| Fundamental model (no odds of the race) | 25.5% | 4.30 | 30.4% | -18.8% | 55% | 572 / 16.6% / 6.17 / -17.7% |
| Fundamental + final odds (conditional logit) | 30.1% | 3.04 | 36.1% | -18.6% | 96% | 55 / 21.8% / 4.18 / -18.2% |
| **END MODEL**: fundamental + final odds + residual booster | 29.3% | 3.07 | 35.9% | -21.1% | 91% | 110 / 15.4% / 4.29 / -39.5% |

End model's top pick, split by its odds and by its own probability:

| Pick's final odds | Races | Hit rate | Break-even | ROI |
|---|---|---|---|---|
| 1-2 | 143 | 55.9% | 61.3% | -8.6% |
| 2-3 | 451 | 33.7% | 40.8% | -17.6% |
| 3-5 | 634 | 21.8% | 27.9% | -22.8% |
| 5-10 | 41 | 4.9% | 18.6% | -74.2% |

| Model probability of its pick | Races | Hit rate | Avg odds | ROI |
|---|---|---|---|---|
| 0.00-0.15 | 6 | 0.0% | 5.60 | -100.0% |
| 0.15-0.25 | 450 | 19.6% | 3.97 | -26.5% |
| 0.25-0.35 | 492 | 27.4% | 2.92 | -21.3% |
| 0.35-0.50 | 258 | 42.2% | 2.09 | -12.7% |
| 0.50-0.70 | 60 | 63.3% | 1.53 | -5.8% |
| 0.70-1.00 | 3 | 66.7% | 1.13 | -26.7% |

### By period

| Period | Races | Market | Fundamental | End model |
|---|---|---|---|---|
| 1996-2000 | 3,097 | 26.0% / 2.157 | 21.4% / 2.279 | 25.8% / 2.156 |
| 2001-2005 | 2,665 | 31.3% / 1.946 | 27.1% / 2.069 | 31.4% / 1.939 |
| 2006-2010 | 3,570 | 29.4% / 2.046 | 24.6% / 2.196 | 30.1% / 2.044 |
| 2011-2016 | 4,648 | 30.3% / 2.043 | 25.1% / 2.195 | 30.2% / 2.042 |
| 2017-2018 | 1,270 | 29.9% / 2.001 | 25.6% / 2.171 | 29.4% / 1.999 |

## Combination bets: couplé (first two) and trio (first three)

Order of finish: a Plackett-Luce model. The win probability comes from the end model. The 2nd and 3rd places use their own strengths (a conditional logit plus a small residual LightGBM per stage, walk-forward by year: train on 2018..Y-1, predict Y). Harville (2nd/3rd = the win probability renormalised) is the textbook baseline. Every combination of every race is enumerated. Dev = 2019-2024, test = 2025-01 -> 2026-05. *ref* = decided 30 minutes before the off (reference odds); *final* = with the final win odds (upper bound, not usable to bet). Payouts are the official PMU dividends per 1 EUR (paid at the final pool). Races where nobody held the winning combination (the pool then pays consolation combinations) or that were refunded are left out of the payout analysis (about 2 % of couplé races, a conservative choice for long-shot strategies).

**Summary**

- **The model prices combinations much better than the textbook method.** Against Harville on the 30-minute market, the stage models cut log-loss by 0.082 (quinella) and 0.131 (trio); the combination probabilities are well calibrated (table below). For the quinella most of the gain comes from the better win model; for the trio about two thirds come from modelling 2nd and 3rd place separately. Harville overrates favourites for the minor places: the fitted discount exponent is about 0.8 for 2nd and 0.7 for 3rd (lower in trotting).
- **But the combination pools are sharper still.** On the test races where the pool ran, the trio pool's own implied log-loss is 4.274, against 4.455 for the model 30 minutes out and 4.359 even with the final win odds. For the quinella the model with final odds roughly ties the pool (3.366 vs 3.378). Exotic bettors do not price like Harville on the win odds.
- **No profit.** Betting the model's top combination loses 22-46 % (30-minute version). No value rule chosen on dev makes money on test with any confidence: the usable (30-minute) rules lose on test, and the final-odds rules that look positive rest on 2 to 14 winning tickets, with confidence intervals spanning -60 % to +100 % or more. The takeout on combinations (22-25 %) is larger than on win bets (about 16.5 %).
- **Hong Kong** gives the same picture: the model adds nothing to the quinella pool and only matches the tierce pool, and every model-edge ROI bin loses.

### Accuracy of the combination probabilities (test, all races with a clear 1-2-3)

Log-loss of the actual combination (lower is better) / hit rate of the model's single most likely combination.

| Pool | Races | Harville on the 30-min market | End model + Harville | **End model + stage models (ref)** | Harville on the final market | End model + stage models (final) |
|---|---|---|---|---|---|---|
| Couplé ordre (exacta) | 13,710 | 4.123 / 7.3% | 4.057 / 7.6% | **4.026 / 7.9%** | 3.976 / 8.6% | 3.937 / 8.9% |
| Couplé gagnant (quinella) | 13,710 | 3.453 / 12.6% | 3.389 / 12.9% | **3.371 / 13.0%** | 3.310 / 14.5% | 3.289 / 14.8% |
| Trio ordre (trifecta) | 13,675 | 6.219 / 2.1% | 6.168 / 1.9% | **6.050 / 1.9%** | 6.054 / 2.3% | 5.941 / 2.4% |
| Trio (any order) | 13,675 | 4.493 / 7.5% | 4.450 / 7.8% | **4.362 / 7.5%** | 4.344 / 8.7% | 4.271 / 8.5% |

Stage models vs Harville on the same market (test; bootstrap by race day):

| Pool | ref: log-loss diff [95 % CI] | ref: top-combination hit diff | final: log-loss diff [95 % CI] |
|---|---|---|---|
| Couplé ordre (exacta) | -0.096 [-0.103, -0.090] | +0.54% [+0.17%, +0.91%] | -0.039 [-0.043, -0.035] |
| Couplé gagnant (quinella) | -0.082 [-0.088, -0.076] | +0.45% [+0.07%, +0.85%] | -0.021 [-0.025, -0.018] |
| Trio ordre (trifecta) | -0.169 [-0.178, -0.159] | -0.12% [-0.37%, +0.13%] | -0.112 [-0.120, -0.105] |
| Trio (any order) | -0.131 [-0.139, -0.122] | +0.04% [-0.30%, +0.37%] | -0.073 [-0.079, -0.066] |

Calibration of the end model's combination probabilities (test, ref): mean predicted vs actual frequency.

| Predicted | Quinella combos | Quinella predicted / actual | Trio combos | Trio predicted / actual |
|---|---|---|---|---|
| 0-0.0005 | 101,365 | 0.0002 / 0.0001 | 1,176,214 | 0.0002 / 0.0002 |
| 0.0005-0.001 | 59,666 | 0.0007 / 0.0008 | 466,089 | 0.0007 / 0.0007 |
| 0.001-0.002 | 87,502 | 0.0015 / 0.0017 | 527,917 | 0.0014 / 0.0014 |
| 0.002-0.005 | 171,990 | 0.0034 / 0.0032 | 640,600 | 0.0032 / 0.0031 |
| 0.005-0.01 | 162,056 | 0.0072 / 0.0071 | 339,718 | 0.0070 / 0.0069 |
| 0.01-0.02 | 153,838 | 0.0143 / 0.0142 | 197,779 | 0.0139 / 0.0135 |
| 0.02-0.05 | 134,663 | 0.0311 / 0.0303 | 104,866 | 0.0298 / 0.0303 |
| 0.05-0.1 | 44,432 | 0.0683 / 0.0686 | 20,164 | 0.0666 / 0.0685 |
| 0.1-0.2 | 13,878 | 0.1326 / 0.1381 | 4,411 | 0.1314 / 0.1342 |
| 0.2-0.3 | 1,741 | 0.2354 / 0.2372 | 549 | 0.2345 / 0.2532 |
| 0.3-1.01 | 420 | 0.3673 / 0.3500 | 202 | 0.4104 / 0.3861 |

### Against the combination pool itself (test, races where the pool ran)

The pool's probability for the winning combination = return to players / payout (returns assumed: couplé 74.8 %, trio 77.9 %, OFDT/PMU averages; the ordre variants assumed equal). *Growth* = mean log return of spreading the bankroll over every combination in proportion to the model's probability: it does not depend on the assumed takeout, and > 0 would mean the model beats the pool after takeout.

| Pool | Races | Median payout | Pool log-loss | Model (ref) log-loss / growth | Harville final market log-loss / growth | Model (final) log-loss / growth |
|---|---|---|---|---|---|---|
| Couplé ordre (exacta) | 989 | 13.7 | 2.908 | 2.967 / -0.349 | 2.943 / -0.326 | 2.907 / -0.290 |
| Couplé gagnant (quinella) | 12,753 | 19.8 | 3.378 | 3.451 / -0.364 | 3.388 / -0.300 | 3.366 / -0.279 |
| Trio ordre (trifecta) | 979 | 41.2 | 4.019 | 4.257 / -0.487 | 4.271 / -0.501 | 4.191 / -0.421 |
| Trio (any order) | 12,093 | 51.4 | 4.274 | 4.455 / -0.431 | 4.431 / -0.406 | 4.359 / -0.334 |

**The pools already price most of what the model knows.** Regressing the log payout of the winning combination on the market-Harville probability H and the model's edge P/H (dev winners) gives an edge coefficient of Couplé ordre -0.69, Couplé gagnant -0.70, Trio ordre -0.80, Trio -0.79 (0 = the pool ignores what the model knows, -1 = the pool prices all of it). So, 30 minutes before the off, roughly 70-80 % of the model's apparent edge over Harville is already in the final combination payouts.

### Top combination vs payout vs profit (test, 1 unit per race)

| Pool | Model | Hit rate | Avg payout when hit | ROI [95 % CI] | Same as market's top | When different: bets / hit / ROI |
|---|---|---|---|---|---|---|
| Couplé ordre (exacta) | mkt_harv_ref | 13.6% | 5.08 | -31.2% [-41.8%, -18.7%] | 100% | - |
| Couplé ordre (exacta) | end_stage_ref | 14.0% | 4.95 | -31.0% [-42.0%, -19.2%] | 66% | 332 / 10.2% / -36.2% |
| Couplé ordre (exacta) | end_stage_final | 14.1% | 4.71 | -33.8% [-44.4%, -22.9%] | 80% | 200 / 10.5% / -34.1% |
| Couplé gagnant (quinella) | mkt_harv_ref | 11.8% | 6.47 | -23.4% [-27.8%, -19.4%] | 100% | - |
| Couplé gagnant (quinella) | end_stage_ref | 12.3% | 6.37 | -21.8% [-25.8%, -17.6%] | 68% | 4,130 / 9.9% / -16.7% |
| Couplé gagnant (quinella) | end_stage_final | 14.0% | 6.30 | -11.7% [-16.0%, -7.4%] | 82% | 2,240 / 10.4% / -3.6% |
| Trio ordre (trifecta) | mkt_harv_ref | 6.3% | 8.99 | -43.1% [-58.0%, -26.5%] | 100% | - |
| Trio ordre (trifecta) | end_stage_ref | 6.0% | 8.97 | -45.9% [-61.0%, -29.9%] | 48% | 513 / 4.1% / -54.2% |
| Trio ordre (trifecta) | end_stage_final | 6.7% | 10.66 | -28.2% [-50.7%, -3.5%] | 64% | 353 / 5.1% / -19.9% |
| Trio (any order) | mkt_harv_ref | 6.5% | 9.05 | -40.8% [-45.6%, -36.0%] | 100% | - |
| Trio (any order) | end_stage_ref | 6.6% | 8.99 | -40.9% [-45.5%, -35.9%] | 58% | 5,068 / 5.0% / -36.1% |
| Trio (any order) | end_stage_final | 7.5% | 9.27 | -30.0% [-35.9%, -24.3%] | 73% | 3,232 / 5.2% / -24.1% |

### Where the money goes: ROI of betting every combination, by price and by model edge (ref)

By the market-Harville probability of the combination (the exotic favourite-longshot bias):

| Market prob of the combination | Quinella dev | Quinella test | Trio dev | Trio test |
|---|---|---|---|---|
| 0-0.0005 | -87% (139,452) | -72% (34,644) | -60% (4,453,666) | -57% (964,411) |
| 0.0005-0.001 | -69% (216,294) | -59% (47,046) | -31% (2,204,127) | -32% (482,567) |
| 0.001-0.002 | -60% (416,214) | -55% (90,216) | -28% (2,362,079) | -27% (535,465) |
| 0.002-0.005 | -45% (920,312) | -45% (206,658) | -29% (2,588,900) | -30% (602,686) |
| 0.005-0.01 | -37% (852,566) | -35% (197,415) | -30% (1,232,857) | -28% (291,210) |
| 0.01-0.02 | -32% (715,002) | -29% (167,594) | -28% (676,743) | -28% (160,903) |
| 0.02-0.05 | -28% (517,051) | -29% (122,117) | -27% (365,058) | -28% (87,733) |
| 0.05-0.1 | -23% (147,495) | -20% (34,777) | -29% (77,586) | -31% (19,179) |
| 0.1-0.2 | -19% (45,821) | -16% (11,356) | -32% (16,958) | -36% (4,270) |
| 0.2-0.3 | -18% (5,910) | -21% (1,486) | -33% (1,449) | -32% (369) |
| 0.3-1.01 | -15% (1,258) | -5% (324) | -38% (204) | +36% (50) |

By the model's edge (model probability / market-Harville probability):

| Model / market | Quinella dev | Quinella test | Trio dev | Trio test |
|---|---|---|---|---|
| 0-0.5 | -72% (842,078) | -72% (193,671) | -74% (2,655,351) | -74% (679,095) |
| 0.5-0.67 | -49% (483,861) | -45% (104,126) | -45% (1,684,084) | -55% (340,326) |
| 0.67-0.8 | -41% (433,454) | -36% (99,420) | -38% (1,379,730) | -45% (289,031) |
| 0.8-0.9 | -37% (354,851) | -36% (82,094) | -37% (1,045,543) | -18% (222,701) |
| 0.9-1.1 | -30% (658,811) | -28% (154,001) | -30% (1,875,738) | -30% (407,409) |
| 1.1-1.25 | -28% (383,889) | -25% (89,738) | -31% (1,142,756) | -23% (252,202) |
| 1.25-1.5 | -26% (410,897) | -26% (95,494) | -24% (1,397,842) | -28% (310,732) |
| 1.5-2 | -27% (306,255) | -20% (70,492) | -23% (1,474,829) | -15% (333,410) |
| 2-3 | -22% (91,169) | -12% (21,470) | -18% (897,531) | -8% (208,607) |
| 3+ | -26% (12,110) | -6% (3,127) | -26% (426,223) | +2% (105,330) |

### Value rule (bet when estimated EV > threshold; chosen on dev, applied once to test)

EV = model probability x expected payout. Expected payout comes from the regression above (fit on dev winners). Threshold and minimum model probability are picked on dev, best ROI with at least 1,000 dev bets (300 for the small ordre pools).

| Pool | Variant | Rule chosen on dev | Dev | Test |
|---|---|---|---|---|
| Couplé ordre (exacta) | ref | p>=0.001,ev>-0.1 | 1,662 bets, 318 hits, ROI -13.5%  | 284 bets, 62 hits, ROI -15.3% [-34.1%, +3.0%] |
| Couplé ordre (exacta) | final | p>=0.02,ev>0 | 464 bets, 64 hits, ROI -11.3%  | 51 bets, 11 hits, ROI +35.3% [-56.9%, +180.6%] |
| Couplé gagnant (quinella) | ref | p>=0.001,ev>-0.1 | 24,752 bets, 3,007 hits, ROI -11.3%  | 5,751 bets, 725 hits, ROI -7.0% [-14.3%, +0.7%] |
| Couplé gagnant (quinella) | final | p>=0.005,ev>0.3 | 1,135 bets, 32 hits, ROI +23.1%  | 52 bets, 3 hits, ROI +41.5% [-100.0%, +260.4%] |
| Trio ordre (trifecta) | ref | p>=0.001,ev>-0.1 | 552 bets, 64 hits, ROI +2.1%  | 105 bets, 15 hits, ROI -21.6% [-55.7%, +23.7%] |
| Trio ordre (trifecta) | final | p>=0.001,ev>0 | 340 bets, 8 hits, ROI +105.9%  | 126 bets, 2 hits, ROI +147.0% [-100.0%, +437.1%] |
| Trio (any order) | ref | p>=0.02,ev>-0.1 | 10,491 bets, 454 hits, ROI -17.7%  | 2,978 bets, 122 hits, ROI -23.5% [-37.8%, -7.5%] |
| Trio (any order) | final | p>=0.001,ev>0.3 | 7,177 bets, 48 hits, ROI +26.0%  | 1,302 bets, 14 hits, ROI +10.3% [-59.5%, +95.2%] |

Realised ROI by estimated EV (test, ref; is the EV estimate right?):

| Estimated EV | Quinella | Trio |
|---|---|---|
| -0.3..-0.2 | -22.8% (161,509) | -14.7% (768,723) |
| -0.2..-0.1 | -19.6% (53,296) | -4.7% (149,018) |
| -0.1..0 | -7.8% (5,559) | -21.6% (12,411) |
| 0..0.1 | +17.9% (187) | +4.4% (1,290) |
| 0.1..0.3 | -100.0% (5) | -38.4% (164) |
| 0.3+ | - | -100.0% (4) |

### Hong Kong replication (final odds only; dividends exist for 2016-09 -> 2018-06)

Same order models (conditional logits per stage, walk-forward by year). The value rule is France's (final variant), not re-tuned; the payout model is fit on the 239 HK races of 2016.

| Pool | Test races | Harville on market | Discounted Harville on market | End model + stages | Pool log-loss (takeout 17.5 % / 25 %) | Model growth | Top combo ROI (model) | France rule on HK test |
|---|---|---|---|---|---|---|---|---|
| Quinella | 1,269 | 3.406 / 13.6% | 3.408 / 13.6% | 3.407 / 13.7% | 3.398 | -0.199 | -8.3% | p>=0.005,ev>0.3: 0 bets |
| Tierce (trifecta) | 1,269 | 6.177 / 1.6% | 6.149 / 1.6% | 6.146 / 1.4% | 6.145 | -0.281 | -54.3% | p>=0.001,ev>0: 1,115 bets, 25 hits, ROI +32.8% [-33.6%, +108.1%] |
