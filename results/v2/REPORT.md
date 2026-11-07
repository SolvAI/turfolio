# HorseCash accuracy report

All numbers are out of sample: every prediction comes from models fit only on earlier years (yearly walk-forward).
Log-loss = mean of -log(probability given to the winner); lower is better. Top-1 = the model's most likely horse won.

## Key findings (France hold-out 2025-01 -> 2026-05, 13,760 races; model v2)

- **The fundamental model alone now beats the 30-minute market.** Without seeing this race's odds, model v2 (a neural race model that reads each horse's last 10 runs, plus opponent-adjusted ratings) scores log-loss 1.984 against 2.023 for the reference odds (-0.039, CI -0.047 to -0.031). v1 (gradient boosting) scored 2.038.
- **Usable 30 minutes before the off:** the end model's top pick wins 31.1% against 30.4% for the favourite at that time (+0.7%, CI +0.2% to +1.3%); log-loss 1.954 vs 2.023. The final market (1.932) is still sharper: the money of the last 30 minutes is informed.
- **The model beats the closing line.** Given the final odds too, it beats the final market: log-loss -0.0168 (CI -0.0195 to -0.0139), in every pool size, most in thin pools.
- **Profit is a question of liquidity and timing.** Betting the top pick in every race returns -10.7% (the pool keeps about 16.5 %). The naive 30-minute value rule returns -1.5% on 8,357 bets. Rules that predict the final odds first return +2.8% (1,161 bets, CI -15% to +23%) to +12.0% (1,185 bets) on the hold-out, depending on the payout model: break-even to modestly positive, never significant, and the best returns sit in pools under 20k EUR where a few euros move the price. Against a locked 30-minute price the same decisions would return +28 % to +74 %; betting at the final odds would also be clearly profitable. Both are out of reach in a pari-mutuel pool (see 'Win bets').
- **Place bets** (simple placé) 30 minutes out: the rule chosen on dev returns +2.6% on 725 hold-out bets (CI -2.8% to +7.9%); positive on dev too, not significant.
- **Combination bets** (couplé, trio): the model prices them far better than Harville, but the pools are about as sharp and keep 22-25 %; no rule is reliably profitable.
- **Hong Kong**, a much bigger and tighter market: the network improves the fundamental model as much as in France, but the end model adds little to the final odds (-0.0039, CI -0.0101 to +0.0027).

## France (PMU, 2014-2026)

### Hold-out: 2025-01-01 -> 2026-05-05 (never used for any choice)

| Model | Races | Top-1 (winner picked) | Winner in top 3 | Log-loss |
|---|---|---|---|---|
| Market, final odds (known only at the off) | 13,760 | 33.0% | 64.8% | 1.9321 |
| Market, reference odds (~30 min before) | 13,760 | 30.4% | 61.8% | 2.0230 |
| Market, reference odds, longshot bias corrected | 13,760 | 30.4% | 61.8% | 2.0199 |
| Fundamental model (no odds of the race) | 13,760 | 29.9% | 61.7% | 1.9837 |
| Fundamental + reference odds (conditional logit) | 13,760 | 30.9% | 63.6% | 1.9612 |
| **END MODEL**: fundamental + reference odds + residual booster | 13,760 | 31.1% | 63.5% | 1.9545 |
| Fundamental + final odds + residual booster | 13,760 | 32.8% | 65.1% | 1.9153 |

Against the final market:

| Model vs benchmark | Log-loss diff [95 % CI] | Top-1 diff [95 % CI] |
|---|---|---|
| Fundamental model (no odds of the race) | +0.0515 [+0.0449, +0.0580] | -3.02% [-3.73%, -2.32%] |
| **END MODEL**: fundamental + reference odds + residual booster | +0.0224 [+0.0166, +0.0278] | -1.82% [-2.40%, -1.19%] |
| Fundamental + final odds + residual booster | -0.0168 [-0.0195, -0.0139] | -0.21% [-0.62%, +0.18%] |
| Fundamental + reference odds (conditional logit) | +0.0291 [+0.0236, +0.0346] | -2.09% [-2.67%, -1.50%] |

Against the reference market (what is known 30 minutes before the off):

| Model vs benchmark | Log-loss diff [95 % CI] | Top-1 diff [95 % CI] |
|---|---|---|
| Fundamental model (no odds of the race) | -0.0393 [-0.0472, -0.0309] | -0.47% [-1.21%, +0.21%] |
| **END MODEL**: fundamental + reference odds + residual booster | -0.0685 [-0.0735, -0.0636] | +0.73% [+0.17%, +1.28%] |
| Market, reference odds, longshot bias corrected | -0.0030 [-0.0043, -0.0018] | +0.00% [+0.00%, +0.00%] |

### Development years 2018-2024

| Model | Races | Top-1 (winner picked) | Winner in top 3 | Log-loss |
|---|---|---|---|---|
| Market, final odds (known only at the off) | 70,279 | 31.8% | 63.8% | 1.9494 |
| Market, reference odds (~30 min before) | 70,279 | 30.0% | 61.3% | 2.0223 |
| Market, reference odds, longshot bias corrected | 70,279 | 30.0% | 61.3% | 2.0186 |
| Fundamental model (no odds of the race) | 70,279 | 29.5% | 61.1% | 1.9974 |
| Fundamental + reference odds (conditional logit) | 70,279 | 30.9% | 62.7% | 1.9669 |
| **END MODEL**: fundamental + reference odds + residual booster | 70,279 | 30.9% | 62.8% | 1.9626 |
| Fundamental + final odds + residual booster | 70,279 | 32.1% | 64.3% | 1.9309 |

### By discipline (hold-out)

| Discipline | Races | Final market | Ref market | Fundamental | End model (ref) | End model (final) |
|---|---|---|---|---|---|---|
| OBSTACLE | 1,540 | 32.3% / 1.879 | 29.7% / 1.954 | 31.9% / 1.905 | 31.8% / 1.885 | 33.2% / 1.860 |
| PLAT | 5,160 | 29.3% / 2.012 | 26.7% / 2.103 | 26.3% / 2.072 | 27.0% / 2.045 | 28.8% / 2.003 |
| TROT_ATTELE | 5,908 | 35.8% / 1.892 | 33.1% / 1.989 | 32.1% / 1.941 | 34.1% / 1.908 | 35.5% / 1.867 |
| TROT_MONTE | 1,152 | 36.0% / 1.854 | 34.7% / 1.931 | 32.6% / 1.910 | 33.7% / 1.884 | 35.4% / 1.841 |

(top-1 / log-loss)

### Per year

| Year | Races | Final market | Ref market | Fundamental | End model (ref) | End model (final) |
|---|---|---|---|---|---|---|
| 2018 | 10,878 | 31.6% / 1.960 | 30.2% / 2.024 | 29.5% / 2.000 | 30.8% / 1.964 | 31.8% / 1.941 |
| 2019 | 9,691 | 31.2% / 1.970 | 30.0% / 2.040 | 28.5% / 2.020 | 30.0% / 1.983 | 31.0% / 1.953 |
| 2020 | 8,754 | 32.4% / 1.938 | 30.0% / 2.016 | 29.5% / 1.998 | 31.0% / 1.957 | 32.5% / 1.920 |
| 2021 | 10,083 | 31.8% / 1.952 | 29.6% / 2.023 | 29.9% / 2.000 | 31.1% / 1.967 | 32.4% / 1.935 |
| 2022 | 10,206 | 31.8% / 1.933 | 30.9% / 2.003 | 30.7% / 1.978 | 31.8% / 1.942 | 32.4% / 1.913 |
| 2023 | 10,330 | 31.8% / 1.941 | 29.5% / 2.018 | 29.8% / 1.985 | 30.9% / 1.954 | 32.6% / 1.921 |
| 2024 | 10,337 | 32.1% / 1.950 | 29.5% / 2.032 | 28.8% / 2.001 | 30.5% / 1.971 | 31.9% / 1.932 |
| 2025 | 10,334 | 32.3% / 1.946 | 29.7% / 2.035 | 29.5% / 1.995 | 30.6% / 1.967 | 32.2% / 1.930 |
| 2026 | 3,426 | 35.1% / 1.891 | 32.5% / 1.986 | 31.1% / 1.950 | 32.9% / 1.917 | 34.4% / 1.872 |

### Calibration of the end model (hold-out)

| Predicted | Runners | Mean predicted | Actual win rate |
|---|---|---|---|
| (0.0, 0.02] | 39,369 | 0.009 | 0.008 |
| (0.02, 0.05] | 38,224 | 0.034 | 0.034 |
| (0.05, 0.1] | 37,205 | 0.072 | 0.071 |
| (0.1, 0.15] | 19,447 | 0.122 | 0.123 |
| (0.15, 0.2] | 10,696 | 0.173 | 0.175 |
| (0.2, 0.3] | 10,215 | 0.242 | 0.245 |
| (0.3, 0.4] | 3,849 | 0.342 | 0.352 |
| (0.4, 0.6] | 2,375 | 0.472 | 0.455 |
| (0.6, 1.0] | 451 | 0.666 | 0.620 |

### Top pick vs odds vs profit (hold-out)

One unit on each model's top pick in every race, paid at the final pool odds (in a pari-mutuel pool every bet gets the final odds).
Break-even hit rate = average of 1/odds of the picks: the hit rate needed to make money at those prices.

| Model | Top-pick hit rate | Avg odds of the pick | Break-even hit rate | ROI betting the pick every race | Same pick as market favourite | When it differs: races / hit / odds / ROI |
|---|---|---|---|---|---|---|
| Market, final odds (known only at the off) | 33.0% | 3.06 | 36.6% | -10.6% | 100% | - |
| Market, reference odds (~30 min before) | 30.4% | 3.50 | 34.3% | -12.4% | 72% | 3,805 / 17.6% / 5.24 / -16.8% |
| Fundamental model (no odds of the race) | 29.9% | 3.82 | 32.7% | -8.2% | 63% | 5,025 / 18.8% / 5.58 / -6.2% |
| Fundamental + reference odds (conditional logit) | 30.9% | 3.40 | 34.6% | -11.5% | 74% | 3,513 / 18.5% / 5.00 / -13.7% |
| **END MODEL**: fundamental + reference odds + residual booster | 31.1% | 3.40 | 34.6% | -10.7% | 74% | 3,529 / 18.9% / 5.00 / -12.6% |
| Fundamental + final odds + residual booster | 32.8% | 3.14 | 36.1% | -9.6% | 86% | 1,861 / 20.4% / 4.57 / -12.6% |

End model's top pick, split by its odds and by its own probability:

| Pick's final odds | Races | Hit rate | Break-even | ROI |
|---|---|---|---|---|
| 1-2 | 1,856 | 53.9% | 60.3% | -10.8% |
| 2-3 | 4,344 | 38.1% | 41.2% | -7.7% |
| 3-5 | 5,735 | 23.6% | 27.1% | -13.4% |
| 5-10 | 1,780 | 15.4% | 16.7% | -7.5% |
| 10-20 | 44 | 2.3% | 9.3% | -77.3% |
| 20+ | 1 | 0.0% | 4.8% | -100.0% |

| Model probability of its pick | Races | Hit rate | Avg odds | ROI |
|---|---|---|---|---|
| 0.00-0.15 | 345 | 16.2% | 6.50 | -4.3% |
| 0.15-0.25 | 4,407 | 20.4% | 4.53 | -15.2% |
| 0.25-0.35 | 4,825 | 29.7% | 3.22 | -9.9% |
| 0.35-0.50 | 3,044 | 41.1% | 2.35 | -7.8% |
| 0.50-0.70 | 1,038 | 55.8% | 1.74 | -5.2% |
| 0.70-1.00 | 101 | 63.4% | 1.40 | -15.2% |

Development years 2018-2024:

| Model | Top-pick hit rate | Avg odds of the pick | Break-even hit rate | ROI betting the pick every race | Same pick as market favourite | When it differs: races / hit / odds / ROI |
|---|---|---|---|---|---|---|
| Market, final odds (known only at the off) | 31.8% | 3.09 | 36.1% | -13.2% | 100% | - |
| Market, reference odds (~30 min before) | 30.0% | 3.44 | 34.3% | -14.2% | 75% | 17,343 / 18.0% / 5.13 / -16.5% |
| Fundamental model (no odds of the race) | 29.5% | 3.96 | 32.0% | -6.8% | 62% | 26,636 / 19.1% / 5.83 / -2.5% |
| **END MODEL**: fundamental + reference odds + residual booster | 30.9% | 3.45 | 34.2% | -9.8% | 74% | 18,160 / 20.0% / 5.10 / -6.2% |
| Fundamental + final odds + residual booster | 32.1% | 3.21 | 35.4% | -9.5% | 84% | 11,357 / 22.6% / 4.66 / -1.6% |

End model's top pick, split by its odds and by its own probability:

| Pick's final odds | Races | Hit rate | Break-even | ROI |
|---|---|---|---|---|
| 1-2 | 8,827 | 56.3% | 59.9% | -6.3% |
| 2-3 | 22,148 | 36.6% | 41.2% | -11.3% |
| 3-5 | 29,689 | 24.1% | 27.1% | -10.9% |
| 5-10 | 9,283 | 15.3% | 16.6% | -6.8% |
| 10-20 | 326 | 9.8% | 9.1% | +7.1% |
| 20+ | 6 | 16.7% | 4.0% | +600.0% |

| Model probability of its pick | Races | Hit rate | Avg odds | ROI |
|---|---|---|---|---|
| 0.00-0.15 | 2,088 | 12.4% | 6.71 | -24.9% |
| 0.15-0.25 | 22,838 | 20.9% | 4.55 | -11.0% |
| 0.25-0.35 | 24,356 | 29.3% | 3.23 | -10.3% |
| 0.35-0.50 | 15,070 | 40.5% | 2.38 | -7.8% |
| 0.50-0.70 | 5,244 | 56.3% | 1.77 | -2.9% |
| 0.70-1.00 | 683 | 70.0% | 1.42 | -1.6% |

### Betting check (secondary)

Decide 30 minutes before the off (end-model probability x reference odds - 1 > threshold), get paid the final pool odds.
Threshold chosen on 2018-24: 0.3. Hold-out: 8,357 bets, ROI -1.5% (market-only rule, same threshold: 265 bets, ROI -29.2%).

## Hong Kong (HKJC, 1979-2018): replication on a second market

Only final odds exist here, so the end model uses them.

### Hold-out: 2017-2018

| Model | Races | Top-1 (winner picked) | Winner in top 3 | Log-loss |
|---|---|---|---|---|
| Market, final odds (known only at the off) | 1,270 | 29.9% | 62.3% | 2.0013 |
| Market, final odds, longshot bias corrected | 1,270 | 29.9% | 62.3% | 1.9997 |
| Fundamental model (no odds of the race) | 1,270 | 25.5% | 57.2% | 2.1134 |
| Fundamental + final odds (conditional logit) | 1,270 | 30.1% | 61.5% | 2.0016 |
| **END MODEL**: fundamental + final odds + residual booster | 1,270 | 29.2% | 62.2% | 1.9974 |

| Model vs benchmark | Log-loss diff [95 % CI] | Top-1 diff [95 % CI] |
|---|---|---|
| Fundamental model (no odds of the race) | +0.1121 [+0.0839, +0.1402] | -4.41% [-6.75%, -2.05%] |
| Fundamental + final odds (conditional logit) | +0.0003 [-0.0037, +0.0046] | +0.16% [-0.63%, +0.88%] |
| **END MODEL**: fundamental + final odds + residual booster | -0.0039 [-0.0101, +0.0027] | -0.71% [-1.99%, +0.55%] |
| Market, final odds, longshot bias corrected | -0.0016 [-0.0030, -0.0002] | +0.00% [+0.00%, +0.00%] |

### Development years 1996-2016

| Model | Races | Top-1 (winner picked) | Winner in top 3 | Log-loss |
|---|---|---|---|---|
| Market, final odds (known only at the off) | 13,980 | 29.3% | 59.3% | 2.0506 |
| Market, final odds, longshot bias corrected | 13,980 | 29.3% | 59.3% | 2.0503 |
| Fundamental model (no odds of the race) | 13,980 | 25.5% | 54.9% | 2.1475 |
| Fundamental + final odds (conditional logit) | 13,980 | 29.3% | 59.6% | 2.0487 |
| **END MODEL**: fundamental + final odds + residual booster | 13,980 | 29.5% | 59.8% | 2.0477 |

| Model vs benchmark | Log-loss diff [95 % CI] | Top-1 diff [95 % CI] |
|---|---|---|
| Fundamental model (no odds of the race) | +0.0969 [+0.0886, +0.1047] | -3.75% [-4.42%, -3.05%] |
| Fundamental + final odds (conditional logit) | -0.0020 [-0.0036, -0.0001] | +0.06% [-0.27%, +0.36%] |
| **END MODEL**: fundamental + final odds + residual booster | -0.0030 [-0.0063, +0.0003] | +0.17% [-0.22%, +0.60%] |
| Market, final odds, longshot bias corrected | -0.0003 [-0.0006, -0.0001] | +0.00% [+0.00%, +0.00%] |

### Top pick vs odds vs profit (hold-out 2017-2018)

Paid at the final odds. The end model here uses the final odds to choose, so its ROI is an upper bound; the fundamental model never sees odds.

| Model | Top-pick hit rate | Avg odds of the pick | Break-even hit rate | ROI betting the pick every race | Same pick as market favourite | When it differs: races / hit / odds / ROI |
|---|---|---|---|---|---|---|
| Market, final odds (known only at the off) | 29.9% | 3.04 | 36.1% | -19.4% | 100% | - |
| Fundamental model (no odds of the race) | 25.4% | 3.88 | 31.7% | -25.3% | 60% | 508 / 14.6% / 5.52 / -34.2% |
| Fundamental + final odds (conditional logit) | 30.0% | 3.05 | 36.0% | -19.0% | 95% | 69 / 20.3% / 4.17 / -27.8% |
| **END MODEL**: fundamental + final odds + residual booster | 29.2% | 3.09 | 35.8% | -20.9% | 89% | 137 / 17.5% / 4.31 / -28.8% |

End model's top pick, split by its odds and by its own probability:

| Pick's final odds | Races | Hit rate | Break-even | ROI |
|---|---|---|---|---|
| 1-2 | 143 | 55.9% | 61.3% | -8.6% |
| 2-3 | 448 | 33.9% | 40.8% | -17.1% |
| 3-5 | 631 | 21.1% | 27.8% | -24.9% |
| 5-10 | 47 | 10.6% | 18.4% | -42.3% |

| Model probability of its pick | Races | Hit rate | Avg odds | ROI |
|---|---|---|---|---|
| 0.00-0.15 | 9 | 11.1% | 5.53 | -33.3% |
| 0.15-0.25 | 449 | 19.1% | 3.99 | -27.0% |
| 0.25-0.35 | 484 | 26.7% | 2.94 | -23.1% |
| 0.35-0.50 | 257 | 42.8% | 2.12 | -9.8% |
| 0.50-0.70 | 69 | 62.3% | 1.54 | -6.5% |
| 0.70-1.00 | 1 | 100.0% | 1.10 | +10.0% |

### By period

| Period | Races | Market | Fundamental | End model |
|---|---|---|---|---|
| 1996-2000 | 3,097 | 26.0% / 2.157 | 22.6% / 2.244 | 25.7% / 2.157 |
| 2001-2005 | 2,665 | 31.3% / 1.946 | 28.6% / 2.030 | 31.4% / 1.938 |
| 2006-2010 | 3,570 | 29.4% / 2.046 | 25.6% / 2.150 | 30.3% / 2.044 |
| 2011-2016 | 4,648 | 30.3% / 2.043 | 25.7% / 2.149 | 30.2% / 2.041 |
| 2017-2018 | 1,270 | 29.9% / 2.001 | 25.5% / 2.113 | 29.2% / 1.997 |

## Model v2: opponent-adjusted ratings and a neural race model

v1 used one LightGBM fundamental model (157 hand-made features, at most 3 past runs per horse). v2 adds two things, and the end model, the combination models and the place study were all rebuilt on top of it.
- **Ratings** (`src/fr_exp.py ratings`): a Plackett-Luce rating for every horse, driver/jockey and trainer, updated after each race day from the full finishing order (gradient step on the finishing-order likelihood, step size shrinking with experience, slow decay). Beating a strong field counts more than beating a weak one. Features of day D only use results of earlier days.
- **A neural race model** (`src/fr_nn.py`): each runner is its feature vector plus the sequence of its last 10 starts (result, speed vs the field, beaten lengths, market opinion then, the field's mean rating = opponent strength, distance / class / track / driver changes relative to today). An attention pool summarises the history, a transformer runs over the field, and the loss is Plackett-Luce over the first 3 places. Walk-forward by year like the GBM (2 seeds, 10 epochs, chosen on 2022).
- The v2 fundamental score is a walk-forward blend (conditional logit, no odds) of the GBM with ratings and the network (`src/fr_blend.py`). Weights in recent years: about 0.85 network, 0.22 GBM.

### Development experiments (France 2018-2024; the hold-out was not used for any choice)

Race log-loss of the winner, walk-forward; difference vs v1 with a day-bootstrap 95 % CI.

| Fundamental variant | Alone (no odds) | Stacked with 30-min odds | Stacked with final odds |
|---|---|---|---|
| v1 GBM | 2.0526 | 1.9905 | 1.9433 |
| GBM + ratings | 2.0412 (-0.0113 [-0.0132, -0.0095]) | 1.9858 (-0.0047 [-0.0056, -0.0038]) | 1.9419 (-0.0014 [-0.0018, -0.0010]) |
| Neural race model | 2.0018 (-0.0508 [-0.0539, -0.0479]) | 1.9669 (-0.0235 [-0.0252, -0.0220]) | 1.9347 (-0.0086 [-0.0094, -0.0077]) |
| All three | - | 1.9667 (-0.0238 [-0.0254, -0.0224]) | 1.9344 (-0.0089 [-0.0098, -0.0080]) |

Hong Kong (dev 2000-2016, final odds only):

| Fundamental variant | Alone | Stacked with final odds |
|---|---|---|
| v1 GBM | 2.1738 | 2.0295 |
| GBM + ratings | 2.1627 (-0.0112 [-0.0152, -0.0068]) | 2.0291 (-0.0004 [-0.0012, +0.0003]) |
| GBM + ratings, top-3 Plackett-Luce objective | 2.1536 (-0.0202 [-0.0248, -0.0154]) | 2.0287 (-0.0008 [-0.0018, +0.0001]) |
| Neural race model | 2.1314 (-0.0424 [-0.0491, -0.0355]) | 2.0283 (-0.0012 [-0.0025, +0.0000]) |
| All | - | 2.0281 (-0.0014 [-0.0028, -0.0001]) |

### Where the gain comes from (France, train 2015-2021, validate 2022, fundamental only)

| Variant | 2022 race log-loss |
|---|---|
| v1 GBM (157 features) | 2.0400 |
| GBM + top-3 Plackett-Luce objective | 2.0353 |
| GBM + ratings | 2.0241 |
| GBM + ratings + top-3 objective | 2.0255 |
| Network (10-run history + field transformer) | 2.0019 |
| Network without the history | 2.0250 |
| Network without the field transformer | 2.0015 |
| Network with a 20-run history | 1.9994 |
| Network, 20-run history, wider | 1.9996 |
| Network given this race's 30-min odds too (end-to-end; compare with the stack: 1.9425 on the same 2022 races, the end-to-end network 1.9565) | 1.9743 |

The gain is the history sequence: without it the network is no better than the GBM with ratings; the transformer over the field adds nothing; 20 runs instead of 10 add a little (-0.0025, not used in v2).

**The gain grows with the length of the history** (dev 2018-2024, network minus v1 GBM, race log-loss):

| Winner's earlier starts in the data | 0-1 | 1-3 | 3-6 | 6-10 | 10-20 | 20-1000 |
|---|---|---|---|---|---|---|
| Races | 5,298 | 10,361 | 12,188 | 11,975 | 18,198 | 28,072 |
| Network vs GBM | +0.022 | -0.023 | -0.026 | -0.038 | -0.044 | -0.075 |
| GBM + ratings vs GBM | -0.006 | -0.009 | -0.013 | -0.016 | -0.010 | -0.013 |

The same holds by the field's average history, and in every discipline (obstacle -0.045, plat -0.044, trot_attele -0.044, trot_monte -0.047). For debutants the network is slightly worse; for horses with 20+ earlier starts it gains 0.07. Three hand-made lags and a few aggregates cannot use the detail of the last 10 runs (who was beaten, how fast, in what company, under what conditions); attention over them can.

Leakage checks for a gain this large: the history holds only strictly earlier starts of the same horse (of 1.76M consecutive same-horse pairs, none share a race and 175 share a start time); every input of the current race is pre-race (the same features as the GBM); the gain disappears when the history is masked; and against the final odds it shrinks to -0.009, the size one expects from better information rather than from a leaked result (which would beat the final market by far more).

### v1 vs v2 on the France hold-out (2025-01 -> 2026-05)

| | v1 | v2 |
|---|---|---|
| Fundamental alone: log-loss | 2.0378 | 1.9837 |
| Fundamental alone: top pick wins | 28.0% | 29.9% |
| End model, 30 min out: log-loss | 1.9794 | 1.9545 |
| End model, 30 min out: top pick wins | 30.9% | 31.1% |
| End model, 30 min out, vs 30-min market | -0.0436 | -0.0685 |
| End model with final odds, vs final market | -0.0107 | -0.0168 |
| Value rule at 30 min (naive, chosen on dev): ROI | -3.5% | -1.5% |
| Market: 30-min / final odds log-loss | 2.0230 / 1.9321 | same |

Hong Kong hold-out 2017-18 (1,270 races): fundamental 2.1705 -> 2.1134; end model vs final market -0.0023 -> -0.0039 [-0.0101, +0.0027]. The network improves the HK fundamental model as much as in France, but the HK final odds already contain it.

## Win bets: late money, pool size and the closing line (France, v2)

In a pari-mutuel pool every bet is paid at the final odds. Late money moves toward what the model sees, so a horse the model likes at the 30-minute odds usually starts shorter. `src/fr_value.py` first predicts the final odds (regression on dev runners: log final odds = 1.00 log ref odds -0.33 log(model p / market p) + ...; i.e. the final odds shrink by edge^0.33), then bets when p x predicted final odds - 1 exceeds a threshold chosen on 2018-24.

| Rule (decided 30 min out, paid at final odds) | Chosen on dev | Dev 2018-24 | Hold-out 2025-26 |
|---|---|---|---|
| EV at the 30-min odds | p>=0.1,ev>0.5 | 9,788 bets, +6.2% | 2,059 bets, +0.1% [-9.9%, +10.1%] |
| EV at the predicted final odds (fit on all runners) | p>=0,ev>0.5 | 5,550 bets, +26.8% | 1,185 bets, +12.0% [-8.3%, +34.6%] |
| **EV at the predicted final odds of a winner** (fit on winners) | p>=0,ev>0.15 | 4,276 bets, +16.0% | 1,161 bets, +2.5% [-16.1%, +23.4%] |

What pays is the final odds of a winner, and the late money is informed: winners shorten more than losers (edge exponent -0.54 for winners against -0.33 for all runners). The winner fit is the correct expected payout, and its estimated EV matches the realised return on dev. It was adopted after the all-runner rule had been run once on the hold-out, so the two hold-out numbers should be read together: +2 % to +12 %, every interval spanning zero. The tables below use the winner fit.

Same rule (p>=0,ev>0.15) paid with the official PMU win dividends: dev 4,276 bets, +15.0% [+6.1%, +24.2%]; hold-out 1,161 bets, +2.8% [-14.7%, +22.9%]. Per year: 2018 +18%, 2019 +13%, 2020 +14%, 2021 +46%, 2022 +18%, 2023 +1%, 2024 +5%, 2025 +11%, 2026 -11%.

**Where the profit is: by size of the win pool** (net of takeout, rebuilt from the winner's dividend x winning stakes). Own stake = the bet's effect on its own payout.

| Net win pool | Dev bets | Dev ROI (no stake effect / 10 EUR / 50 EUR) | Hold-out bets | Hold-out ROI (no stake effect / 10 EUR / 50 EUR) |
|---|---|---|---|---|
| 0-20,000 EUR | 557 | +90% / +54% / +14% | 95 | +43% / +38% / +23% |
| 20,000-50,000 EUR | 205 | -24% / -24% / -25% | 17 | +49% / +49% / +48% |
| 50,000-100,000 EUR | 1,231 | +16% / +15% / +15% | 197 | +27% / +27% / +26% |
| 100,000-300,000 EUR | 2,198 | +0% / -0% / -0% | 716 | -11% / -11% / -12% |
| 300,000-max EUR | 85 | -3% / -3% / -3% | 136 | +9% / +8% / +8% |

Pools under 20k EUR (2.5% of the hold-out races, mostly regional meetings late in the programme) carry the largest returns, and that is where a stake of a few euros already moves the price. Outside them the returns scatter around zero from one band to the next (see the liquid-pool rule below). Dev returns are optimistic: the threshold was chosen on them.

**Liquid pools only** (net win pool >= 50,000 EUR, threshold re-chosen on dev): p>=0.05,ev>0.15; dev 3,418 bets, +7.8%; hold-out 892 bets, -1.5% [-16.1%, +13.6%].

**Accuracy by pool size, and the closing line.** The last column bets at the final odds with the final-odds model (EV > 20 %): an upper bound, since nobody can bet at the exact final odds (the French data has no snapshot between 30 minutes and the off to measure how much the odds still move).

| Net win pool | Races (hold-out) | End model (final odds) vs final market [CI] | End model (30 min) vs 30-min market | Bet at the final odds, EV > 20 %: dev | hold-out |
|---|---|---|---|---|---|
| 0-20,000 EUR | 349 | -0.0704 [-0.0907, -0.0493] | -0.1918 | 1,092 bets, +107.5% | 135 bets, +43.9% |
| 20,000-50,000 EUR | 292 | -0.0123 [-0.0293, +0.0077] | -0.0387 | 912 bets, +23.2% | 27 bets, -51.8% |
| 50,000-100,000 EUR | 3,018 | -0.0168 [-0.0226, -0.0108] | -0.0651 | 5,145 bets, +20.1% | 325 bets, +58.8% |
| 100,000-200,000 EUR | 8,480 | -0.0151 [-0.0185, -0.0115] | -0.0675 | 9,578 bets, +18.3% | 934 bets, +19.1% |
| 200,000-max EUR | 1,621 | -0.0151 [-0.0228, -0.0076] | -0.0586 | 1,787 bets, -22.1% | 260 bets, -6.0% |

The model beats the final market in every pool size, most in thin pools. Given the final odds it would be profitable in pools up to 200k EUR on dev, and on the hold-out except one 27-bet cell; 30 minutes out, most of that edge is taken by the late money.

### The opening edge: what if the 30-minute price could be locked?

In a pari-mutuel pool an early bet does not lock its price: it is paid at the final odds. If the 30-minute PMU price could be locked (as with a fixed-odds bookmaker or an exchange quoting the same prices), the same 30-minute decisions would return:

| Model EV at the 30-min odds | v1: locked 30-min price (hold-out) | v2: locked 30-min price (hold-out) | v2: paid at final odds (reality) |
|---|---|---|---|
| > 0% | 28,092 bets, +24% [+19%, +28%] | 34,255 bets, +28% [+24%, +32%] | -3.2% |
| > 5% | 20,589 bets, +29% [+24%, +36%] | 27,716 bets, +33% [+28%, +38%] | -1.8% |
| > 10% | 14,709 bets, +35% [+27%, +43%] | 22,196 bets, +39% [+34%, +45%] | -0.1% |
| > 20% | 7,240 bets, +47% [+34%, +60%] | 13,922 bets, +47% [+40%, +54%] | +0.1% |
| > 30% | 3,422 bets, +55% [+38%, +69%] | 8,357 bets, +53% [+43%, +63%] | -1.5% |
| > 50% | 788 bets, +64% [+29%, +104%] | 2,895 bets, +74% [+57%, +92%] | +2.4% |

The model knows 30 minutes early where the late money will go. Against a locked early price that is worth +28 % to +74 % (even v1: +24 % to +65 %); in the pari-mutuel pool the late money takes it, and the same bets are about break-even.

### How late is the late money? (Hong Kong live odds, 1,356 races 2016-18)

Market log-loss of the winner k minutes before the odds freeze (the off), against the final odds (2.0356). Snapshots near the off are about 4 minutes apart.

| Minutes before the off | 30 | 15 | 10 | 5 | 3 | last snapshot before the off |
|---|---|---|---|---|---|---|
| Log-loss | 2.1044 | 2.1024 | 2.0997 | 2.0940 | 2.0717 | 2.0712 |
| Share of the 30-min -> final gain still missing | 100% | 97% | 93% | 85% | 52% | 52% |

In Hong Kong about half of the late information reaches the odds only in the last few minutes. If France is similar, the closing-line edge is only reachable by betting in the final seconds, and only in part.

## Place bets (simple placé, France, v2)

A place bet wins if the horse finishes in the first 3 (first 2 with 4-7 starters). The pool is split equally between the placed horses: payout = 1 + (R x pool - stakes on the placed) / (m x stake on the horse), rounded down to 0.10, minimum 1.10 (1.05 in some races since 2021). PMU publishes each placed horse's payout and number of winning stakes, so (payout - 1) x stakes is equal across the placed horses of a race (checked: median deviation 2.5 %). `src/fr_place.py` prices every horse with the order models of the combination study (P(place) from Plackett-Luce with stage-2/3 strengths) and a structural model of the place pool.

- **The place pool is flatter than the win pool:** stakes rise as (win-market share)^0.72 in trotting and ^0.86 in flat/jumps. Estimated return rate 84.4% (trot) / 85.5% (flat).
- **The payouts already price most of the 30-minute edge:** regressing log(payout - 1) of placed horses on the structural prediction and log(model / market P(place)) gives -0.72 at 30 min and -0.20 with the final odds.

| P(place) model (hold-out) | Log-loss per horse | Top place pick places | Top place pick ROI |
|---|---|---|---|
| Harville, 30-min market | 0.4875 | 57.4% | -9.7% |
| Discounted Harville, 30-min market | 0.4866 | 57.4% | -9.7% |
| End model + stages, 30 min | 0.4702 | 58.8% | -7.6% |
| Harville, final market | 0.4737 | 60.4% | -7.3% |
| Discounted Harville, final market | 0.4709 | 60.3% | -7.5% |
| End model + stages, final odds | 0.4649 | 60.6% | -6.0% |

ROI of every place bet by the horse's final-odds win probability (hold-out): 0-0.02 -48%, 0.02-0.05 -26%, 0.05-0.1 -20%, 0.1-0.2 -14%, 0.2-0.3 -8%, 0.3-0.4 -3%, 0.4-0.5 -6%, 0.5-0.6 -2%, 0.6-0.7 -8%, 0.7-1.01 +6%.
Heavy favourites, where the minimum payout binds, come close to break-even but do not beat it.

| Value rule (chosen on dev) | Chosen | Dev | Hold-out |
|---|---|---|---|
| 30 min out | p>=0.6,ev>0.1 | 2,747 bets, +3.1% | 725 bets, +2.6% [-2.8%, +7.9%] |
| final odds (upper bound) | p>=0,ev>0.2 | 1,668 bets, +17.9% | 93 bets, +51.4% [+6.2%, +98.2%] |

The 30-minute rule by size of the place pool (gross payout pool = return rate x pool, which does not depend on the result; bands = quintiles of dev races):

| Place pool (EUR) | Dev | Hold-out |
|---|---|---|
| 0-54,322 | 587 bets, +9.1% | 138 bets, +6.2% |
| 54,322-76,018 | 461 bets, +2.8% | 126 bets, -7.6% |
| 76,018-96,852 | 513 bets, -1.5% | 177 bets, +12.0% |
| 96,852-123,564 | 583 bets, -4.3% | 162 bets, -2.6% |
| 123,564-max | 603 bets, +8.8% | 122 bets, +2.4% |

No pool-size pattern. (A first version filtered on the *net* place pool, R x pool minus the stakes on the placed horses. That quantity shrinks when favourites place, so filtering on it selects races where favourites placed: it showed a spurious +15 % on dev and hold-out alike for these favourite bets. With the outcome-free measure the effect is gone: the cap chosen on dev gives 264 bets, -0.4% [-10.0%, +8.6%] on the hold-out.)

Realised ROI by estimated EV (hold-out): 30 min: -0.05-0 -6% (10,093), 0-0.05 -5% (5,884), 0.05-0.1 -2% (2,833), 0.1-0.2 -2% (1,470), 0.2-1e+09 +7% (149); final: -0.05-0 -6% (11,242), 0-0.05 +2% (5,529), 0.05-0.1 -2% (2,155), 0.1-0.2 +6% (918), 0.2-1e+09 +51% (93)

## Combination bets: couplé (first two) and trio (first three)

Order of finish: a Plackett-Luce model. The win probability comes from the end model. The 2nd and 3rd places use their own strengths (a conditional logit plus a small residual LightGBM per stage, walk-forward by year: train on 2018..Y-1, predict Y). Harville (2nd/3rd = the win probability renormalised) is the textbook baseline. Every combination of every race is enumerated. Dev = 2019-2024, test = 2025-01 -> 2026-05. *ref* = decided 30 minutes before the off (reference odds); *final* = with the final win odds (upper bound, not usable to bet). Payouts are the official PMU dividends per 1 EUR (paid at the final pool). Races where nobody held the winning combination (the pool then pays consolation combinations) or that were refunded are left out of the payout analysis (about 2 % of couplé races, a conservative choice for long-shot strategies).

**Summary**

- **The model prices combinations much better than the textbook method.** Against Harville on the 30-minute market, the stage models cut log-loss by 0.124 (quinella) and 0.188 (trio); the combination probabilities are well calibrated (table below). For the quinella most of the gain comes from the better win model; for the trio about two thirds come from modelling 2nd and 3rd place separately. Harville overrates favourites for the minor places: the fitted discount exponent is about 0.8 for 2nd and 0.7 for 3rd (lower in trotting).
- **But the combination pools are sharper still.** On the test races where the pool ran, the trio pool's own implied log-loss is 4.274, against 4.396 for the model 30 minutes out and 4.338 even with the final win odds. For the quinella the model with final odds beats the pool (3.354 vs 3.378), but 30 minutes out it is worse (3.408). Exotic bettors do not price like Harville on the win odds.
- **No reliable profit.** Betting the model's top combination returns -32% to -17% (30-minute version). The value rules chosen on dev give, on test: Couplé ordre 30 min +2% (52 hits); Couplé ordre final odds -20% (4 hits); Couplé gagnant 30 min -21% (44 hits); Couplé gagnant final odds +41% (19 hits); Trio ordre 30 min -16% (63 hits); Trio ordre final odds +116% (4 hits); Trio 30 min +32% (26 hits); Trio final odds -14% (13 hits). Every confidence interval spans zero widely: the rules rest on 4 to 63 winning tickets. The takeout on combinations (22-25 %) is larger than on win bets (about 16.5 %).
- **Hong Kong** gives the same picture: the model adds nothing to the quinella pool and only matches the tierce pool, and every model-edge ROI bin loses.

### Accuracy of the combination probabilities (test, all races with a clear 1-2-3)

Log-loss of the actual combination (lower is better) / hit rate of the model's single most likely combination.

| Pool | Races | Harville on the 30-min market | End model + Harville | **End model + stage models (ref)** | Harville on the final market | End model + stage models (final) |
|---|---|---|---|---|---|---|
| Couplé ordre (exacta) | 13,710 | 4.123 / 7.3% | 4.013 / 7.9% | **3.982 / 8.2%** | 3.976 / 8.6% | 3.924 / 8.8% |
| Couplé gagnant (quinella) | 13,710 | 3.453 / 12.6% | 3.346 / 13.5% | **3.329 / 13.5%** | 3.310 / 14.5% | 3.276 / 14.6% |
| Trio ordre (trifecta) | 13,675 | 6.219 / 2.1% | 6.106 / 2.1% | **5.987 / 2.1%** | 6.054 / 2.3% | 5.919 / 2.3% |
| Trio (any order) | 13,675 | 4.493 / 7.5% | 4.392 / 8.1% | **4.305 / 8.0%** | 4.344 / 8.7% | 4.250 / 8.7% |

Stage models vs Harville on the same market (test; bootstrap by race day):

| Pool | ref: log-loss diff [95 % CI] | ref: top-combination hit diff | final: log-loss diff [95 % CI] |
|---|---|---|---|
| Couplé ordre (exacta) | -0.140 [-0.148, -0.133] | +0.90% [+0.50%, +1.32%] | -0.052 [-0.057, -0.046] |
| Couplé gagnant (quinella) | -0.124 [-0.130, -0.116] | +0.96% [+0.47%, +1.39%] | -0.034 [-0.038, -0.030] |
| Trio ordre (trifecta) | -0.231 [-0.242, -0.221] | +0.03% [-0.23%, +0.27%] | -0.135 [-0.144, -0.125] |
| Trio (any order) | -0.188 [-0.198, -0.178] | +0.48% [+0.08%, +0.87%] | -0.094 [-0.102, -0.086] |

Calibration of the end model's combination probabilities (test, ref): mean predicted vs actual frequency.

| Predicted | Quinella combos | Quinella predicted / actual | Trio combos | Trio predicted / actual |
|---|---|---|---|---|
| 0-0.0005 | 127,632 | 0.0002 / 0.0002 | 1,282,361 | 0.0002 / 0.0002 |
| 0.0005-0.001 | 61,159 | 0.0007 / 0.0006 | 443,146 | 0.0007 / 0.0008 |
| 0.001-0.002 | 86,596 | 0.0015 / 0.0016 | 491,897 | 0.0014 / 0.0015 |
| 0.002-0.005 | 164,738 | 0.0034 / 0.0036 | 601,832 | 0.0032 / 0.0031 |
| 0.005-0.01 | 152,003 | 0.0073 / 0.0073 | 328,483 | 0.0070 / 0.0069 |
| 0.01-0.02 | 145,486 | 0.0143 / 0.0144 | 196,185 | 0.0139 / 0.0135 |
| 0.02-0.05 | 131,320 | 0.0311 / 0.0300 | 107,314 | 0.0298 / 0.0301 |
| 0.05-0.1 | 45,225 | 0.0684 / 0.0682 | 21,432 | 0.0666 / 0.0683 |
| 0.1-0.2 | 14,821 | 0.1329 / 0.1374 | 4,935 | 0.1319 / 0.1291 |
| 0.2-0.3 | 2,035 | 0.2359 / 0.2324 | 673 | 0.2358 / 0.2526 |
| 0.3-1.01 | 536 | 0.3761 / 0.3619 | 251 | 0.4099 / 0.3825 |

### Against the combination pool itself (test, races where the pool ran)

The pool's probability for the winning combination = return to players / payout (returns assumed: couplé 74.8 %, trio 77.9 %, OFDT/PMU averages; the ordre variants assumed equal). *Growth* = mean log return of spreading the bankroll over every combination in proportion to the model's probability: it does not depend on the assumed takeout, and > 0 would mean the model beats the pool after takeout.

| Pool | Races | Median payout | Pool log-loss | Model (ref) log-loss / growth | Harville final market log-loss / growth | Model (final) log-loss / growth |
|---|---|---|---|---|---|---|
| Couplé ordre (exacta) | 989 | 13.7 | 2.908 | 2.936 / -0.319 | 2.943 / -0.326 | 2.896 / -0.279 |
| Couplé gagnant (quinella) | 12,753 | 19.8 | 3.378 | 3.408 / -0.321 | 3.388 / -0.300 | 3.354 / -0.267 |
| Trio ordre (trifecta) | 979 | 41.2 | 4.019 | 4.217 / -0.447 | 4.271 / -0.501 | 4.175 / -0.406 |
| Trio (any order) | 12,093 | 51.4 | 4.274 | 4.396 / -0.372 | 4.431 / -0.406 | 4.338 / -0.314 |

**The pools already price most of what the model knows.** Regressing the log payout of the winning combination on the market-Harville probability H and the model's edge P/H (dev winners) gives an edge coefficient of Couplé ordre -0.60, Couplé gagnant -0.60, Trio ordre -0.72, Trio -0.68 (0 = the pool ignores what the model knows, -1 = the pool prices all of it). So, 30 minutes before the off, roughly 70-80 % of the model's apparent edge over Harville is already in the final combination payouts.

### Top combination vs payout vs profit (test, 1 unit per race)

| Pool | Model | Hit rate | Avg payout when hit | ROI [95 % CI] | Same as market's top | When different: bets / hit / ROI |
|---|---|---|---|---|---|---|
| Couplé ordre (exacta) | mkt_harv_ref | 13.6% | 5.08 | -31.2% [-41.8%, -18.7%] | 100% | - |
| Couplé ordre (exacta) | end_stage_ref | 13.6% | 5.03 | -31.8% [-44.2%, -19.2%] | 59% | 408 / 10.3% / -31.2% |
| Couplé ordre (exacta) | end_stage_final | 15.4% | 5.09 | -21.8% [-33.9%, -8.0%] | 75% | 245 / 13.1% / -8.9% |
| Couplé gagnant (quinella) | mkt_harv_ref | 11.8% | 6.47 | -23.4% [-27.7%, -18.9%] | 100% | - |
| Couplé gagnant (quinella) | end_stage_ref | 12.8% | 6.48 | -16.8% [-21.2%, -12.2%] | 58% | 5,374 / 10.3% / -12.0% |
| Couplé gagnant (quinella) | end_stage_final | 13.9% | 6.38 | -11.4% [-15.8%, -7.0%] | 77% | 2,972 / 9.9% / -6.4% |
| Trio ordre (trifecta) | mkt_harv_ref | 6.3% | 8.99 | -43.1% [-58.6%, -26.7%] | 100% | - |
| Trio ordre (trifecta) | end_stage_ref | 6.2% | 11.13 | -30.6% [-53.3%, -2.7%] | 39% | 601 / 5.0% / -22.6% |
| Trio ordre (trifecta) | end_stage_final | 6.4% | 11.04 | -28.9% [-48.7%, -4.7%] | 56% | 426 / 4.2% / -23.4% |
| Trio (any order) | mkt_harv_ref | 6.5% | 9.05 | -40.8% [-45.8%, -35.8%] | 100% | - |
| Trio (any order) | end_stage_ref | 7.0% | 9.80 | -31.3% [-36.8%, -25.4%] | 49% | 6,218 / 5.7% / -21.5% |
| Trio (any order) | end_stage_final | 7.7% | 10.00 | -23.1% [-28.7%, -17.0%] | 66% | 4,150 / 5.8% / -10.7% |

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
| 0-0.5 | -73% (1,137,030) | -73% (261,681) | -72% (3,967,445) | -75% (946,530) |
| 0.5-0.67 | -44% (457,892) | -43% (103,254) | -44% (1,546,587) | -53% (325,574) |
| 0.67-0.8 | -40% (371,197) | -33% (84,930) | -41% (1,139,191) | -44% (241,374) |
| 0.8-0.9 | -33% (281,432) | -28% (64,461) | -35% (819,603) | -22% (175,289) |
| 0.9-1.1 | -30% (506,019) | -29% (115,589) | -24% (1,441,012) | -23% (310,549) |
| 1.1-1.25 | -25% (307,164) | -20% (70,129) | -25% (891,039) | -18% (193,095) |
| 1.25-1.5 | -21% (366,634) | -17% (83,929) | -20% (1,147,258) | -21% (252,250) |
| 1.5-2 | -18% (353,337) | -15% (82,186) | -16% (1,378,887) | -11% (309,387) |
| 2-3 | -11% (163,327) | -8% (39,022) | -13% (1,045,748) | -2% (243,430) |
| 3+ | +0% (33,343) | +11% (8,452) | -7% (602,857) | +23% (151,365) |

### Value rule (bet when estimated EV > threshold; chosen on dev, applied once to test)

EV = model probability x expected payout. Expected payout comes from the regression above (fit on dev winners). Threshold and minimum model probability are picked on dev, best ROI with at least 1,000 dev bets (300 for the small ordre pools).

| Pool | Variant | Rule chosen on dev | Dev | Test |
|---|---|---|---|---|
| Couplé ordre (exacta) | ref | p>=0.001,ev>0 | 1,921 bets, 194 hits, ROI -5.3%  | 413 bets, 52 hits, ROI +2.0% [-37.6%, +47.2%] |
| Couplé ordre (exacta) | final | p>=0.005,ev>0.1 | 755 bets, 51 hits, ROI +7.8%  | 50 bets, 4 hits, ROI -19.6% [-95.7%, +145.4%] |
| Couplé gagnant (quinella) | ref | p>=0.005,ev>0.2 | 1,854 bets, 161 hits, ROI +17.4%  | 545 bets, 44 hits, ROI -20.6% [-48.3%, +11.8%] |
| Couplé gagnant (quinella) | final | p>=0.005,ev>0.3 | 5,806 bets, 220 hits, ROI +44.7%  | 432 bets, 19 hits, ROI +41.5% [-59.2%, +183.3%] |
| Trio ordre (trifecta) | ref | p>=0.02,ev>-0.1 | 4,467 bets, 299 hits, ROI -6.5%  | 971 bets, 63 hits, ROI -16.4% [-41.2%, +9.3%] |
| Trio ordre (trifecta) | final | p>=0.005,ev>0.1 | 1,012 bets, 17 hits, ROI +8.6%  | 223 bets, 4 hits, ROI +116.3% [-72.9%, +328.1%] |
| Trio (any order) | ref | p>=0.001,ev>0.3 | 5,899 bets, 99 hits, ROI +4.5%  | 1,863 bets, 26 hits, ROI +32.4% [-36.1%, +123.7%] |
| Trio (any order) | final | p>=0.001,ev>0.5 | 8,509 bets, 63 hits, ROI +38.0%  | 1,264 bets, 13 hits, ROI -14.4% [-65.9%, +58.4%] |

Realised ROI by estimated EV (test, ref; is the EV estimate right?):

| Estimated EV | Quinella | Trio |
|---|---|---|
| -0.3..-0.2 | -20.7% (138,418) | -14.2% (560,979) |
| -0.2..-0.1 | -11.9% (79,139) | +2.5% (312,643) |
| -0.1..0 | -10.6% (30,618) | +12.0% (116,191) |
| 0..0.1 | -8.5% (8,414) | +12.2% (35,293) |
| 0.1..0.3 | -1.9% (2,248) | -11.7% (13,826) |
| 0.3+ | -21.1% (111) | +18.0% (2,090) |

### Hong Kong replication (final odds only; dividends exist for 2016-09 -> 2018-06)

Same order models (conditional logits per stage, walk-forward by year). The value rule is France's (final variant), not re-tuned; the payout model is fit on the 239 HK races of 2016.

| Pool | Test races | Harville on market | Discounted Harville on market | End model + stages | Pool log-loss (takeout 17.5 % / 25 %) | Model growth | Top combo ROI (model) | France rule on HK test |
|---|---|---|---|---|---|---|---|---|
| Quinella | 1,269 | 3.406 / 13.6% | 3.408 / 13.6% | 3.405 / 14.0% | 3.398 | -0.198 | -6.1% | p>=0.005,ev>0.3: 4 bets, 2 hits, ROI +67.5%  |
| Tierce (trifecta) | 1,269 | 6.177 / 1.6% | 6.149 / 1.6% | 6.141 / 1.4% | 6.145 | -0.276 | -55.3% | p>=0.001,ev>0: 11,240 bets, 97 hits, ROI -2.1% [-32.0%, +33.5%] |
