# HorseCash accuracy report

All numbers are out of sample: every prediction comes from models fit only on earlier years (yearly walk-forward).
Log-loss = mean of -log(probability given to the winner); lower is better. Top-1 = the model's most likely horse won.

## Key findings (France hold-out 2025-01 -> 2026-05, 13,760 races; model v3)

- **The fundamental model alone now beats the 30-minute market.** Without seeing this race's odds, model v3 (a neural race model that reads each horse's last 20 runs and its form string, plus opponent-adjusted ratings) scores log-loss 1.980 against 2.023 for the reference odds (-0.043, CI -0.051 to -0.035). v1 (gradient boosting) scored 2.038.
- **Usable 30 minutes before the off:** the end model's top pick wins 31.1% against 30.4% for the favourite at that time (+0.7%, CI +0.2% to +1.2%); log-loss 1.952 vs 2.023. The final market (1.932) is still sharper: the money of the last 30 minutes is informed.
- **The model beats the closing line.** Given the final odds too, it beats the final market: log-loss -0.0174 (CI -0.0201 to -0.0145), in every pool size, most in thin pools.
- **Profit is a question of liquidity and timing.** Betting the top pick in every race returns -10.7% (the pool keeps about 16.5 %). The naive 30-minute value rule returns +2.4% on 8,812 bets. Rules that predict the final odds first return +6.1% (1,299 bets, CI -11% to +27%) to +12.9% (1,341 bets) on the hold-out, depending on the payout model: break-even to modestly positive, never significant, and the best returns sit in pools under 20k EUR where a few euros move the price. Against a locked 30-minute price the same decisions would return +27% to +65%; betting at the final odds would also be clearly profitable. Both are out of reach in a pari-mutuel pool (see 'Win bets').
- **Place bets** (simple placé) 30 minutes out: the rule chosen on dev returns +4.3% on 755 hold-out bets (CI -0.6% to +9.2%); positive on dev too, not significant.
- **Combination bets** (couplé, trio): the model prices them far better than Harville, but the pools are about as sharp and keep 22-25 %; no rule is reliably profitable.
- **Hong Kong**, a much bigger and tighter market: the network improves the fundamental model as much as in France, but the end model adds little to the final odds (-0.0039, CI -0.0101 to +0.0027).

## France (PMU, 2014-2026)

### Hold-out: 2025-01-01 -> 2026-05-05 (never used for any choice)

| Model | Races | Top-1 (winner picked) | Winner in top 3 | Log-loss |
|---|---|---|---|---|
| Market, final odds (known only at the off) | 13,760 | 33.0% | 64.8% | 1.9321 |
| Market, reference odds (~30 min before) | 13,760 | 30.4% | 61.8% | 2.0230 |
| Market, reference odds, longshot bias corrected | 13,760 | 30.4% | 61.8% | 2.0199 |
| Fundamental model (no odds of the race) | 13,760 | 30.2% | 61.8% | 1.9796 |
| Fundamental + reference odds (conditional logit) | 13,760 | 30.9% | 63.6% | 1.9588 |
| **END MODEL**: fundamental + reference odds + residual booster | 13,760 | 31.1% | 63.8% | 1.9521 |
| Fundamental + final odds + residual booster | 13,760 | 32.8% | 65.2% | 1.9147 |

Against the final market:

| Model vs benchmark | Log-loss diff [95 % CI] | Top-1 diff [95 % CI] |
|---|---|---|
| Fundamental model (no odds of the race) | +0.0475 [+0.0410, +0.0540] | -2.81% [-3.48%, -2.11%] |
| **END MODEL**: fundamental + reference odds + residual booster | +0.0200 [+0.0142, +0.0255] | -1.85% [-2.41%, -1.26%] |
| Fundamental + final odds + residual booster | -0.0174 [-0.0201, -0.0145] | -0.16% [-0.56%, +0.27%] |
| Fundamental + reference odds (conditional logit) | +0.0267 [+0.0212, +0.0322] | -2.10% [-2.68%, -1.48%] |

Against the reference market (what is known 30 minutes before the off):

| Model vs benchmark | Log-loss diff [95 % CI] | Top-1 diff [95 % CI] |
|---|---|---|
| Fundamental model (no odds of the race) | -0.0434 [-0.0512, -0.0353] | -0.25% [-0.92%, +0.43%] |
| **END MODEL**: fundamental + reference odds + residual booster | -0.0708 [-0.0762, -0.0659] | +0.70% [+0.17%, +1.23%] |
| Market, reference odds, longshot bias corrected | -0.0030 [-0.0043, -0.0018] | +0.00% [+0.00%, +0.00%] |

### Development years 2018-2024

| Model | Races | Top-1 (winner picked) | Winner in top 3 | Log-loss |
|---|---|---|---|---|
| Market, final odds (known only at the off) | 70,279 | 31.8% | 63.8% | 1.9494 |
| Market, reference odds (~30 min before) | 70,279 | 30.0% | 61.3% | 2.0223 |
| Market, reference odds, longshot bias corrected | 70,279 | 30.0% | 61.3% | 2.0186 |
| Fundamental model (no odds of the race) | 70,279 | 29.6% | 61.3% | 1.9929 |
| Fundamental + reference odds (conditional logit) | 70,279 | 30.9% | 62.8% | 1.9645 |
| **END MODEL**: fundamental + reference odds + residual booster | 70,279 | 30.9% | 62.8% | 1.9600 |
| Fundamental + final odds + residual booster | 70,279 | 32.1% | 64.4% | 1.9296 |

### By discipline (hold-out)

| Discipline | Races | Final market | Ref market | Fundamental | End model (ref) | End model (final) |
|---|---|---|---|---|---|---|
| OBSTACLE | 1,540 | 32.3% / 1.879 | 29.7% / 1.954 | 31.8% / 1.901 | 31.8% / 1.882 | 32.8% / 1.859 |
| PLAT | 5,160 | 29.3% / 2.012 | 26.7% / 2.103 | 26.2% / 2.070 | 27.2% / 2.043 | 29.0% / 2.004 |
| TROT_ATTELE | 5,908 | 35.8% / 1.892 | 33.1% / 1.989 | 32.6% / 1.935 | 33.9% / 1.905 | 35.6% / 1.866 |
| TROT_MONTE | 1,152 | 36.0% / 1.854 | 34.7% / 1.931 | 33.3% / 1.909 | 33.6% / 1.882 | 35.3% / 1.841 |

(top-1 / log-loss)

### Per year

| Year | Races | Final market | Ref market | Fundamental | End model (ref) | End model (final) |
|---|---|---|---|---|---|---|
| 2018 | 10,878 | 31.6% / 1.960 | 30.2% / 2.024 | 29.8% / 1.995 | 31.1% / 1.961 | 31.8% / 1.939 |
| 2019 | 9,691 | 31.2% / 1.970 | 30.0% / 2.040 | 28.5% / 2.016 | 30.2% / 1.981 | 31.2% / 1.952 |
| 2020 | 8,754 | 32.4% / 1.938 | 30.0% / 2.016 | 29.7% / 1.995 | 31.0% / 1.956 | 32.8% / 1.919 |
| 2021 | 10,083 | 31.8% / 1.952 | 29.6% / 2.023 | 29.9% / 1.995 | 31.2% / 1.964 | 32.3% / 1.933 |
| 2022 | 10,206 | 31.8% / 1.933 | 30.9% / 2.003 | 30.6% / 1.974 | 31.7% / 1.940 | 32.3% / 1.913 |
| 2023 | 10,330 | 31.8% / 1.941 | 29.5% / 2.018 | 29.8% / 1.981 | 30.8% / 1.952 | 32.4% / 1.920 |
| 2024 | 10,337 | 32.1% / 1.950 | 29.5% / 2.032 | 29.2% / 1.996 | 30.3% / 1.967 | 32.0% / 1.931 |
| 2025 | 10,334 | 32.3% / 1.946 | 29.7% / 2.035 | 29.9% / 1.991 | 30.5% / 1.965 | 32.1% / 1.929 |
| 2026 | 3,426 | 35.1% / 1.891 | 32.5% / 1.986 | 31.1% / 1.945 | 32.9% / 1.914 | 34.9% / 1.871 |

### Calibration of the end model (hold-out)

| Predicted | Runners | Mean predicted | Actual win rate |
|---|---|---|---|
| (0.0, 0.02] | 39,653 | 0.009 | 0.008 |
| (0.02, 0.05] | 38,004 | 0.034 | 0.034 |
| (0.05, 0.1] | 37,056 | 0.072 | 0.071 |
| (0.1, 0.15] | 19,491 | 0.122 | 0.122 |
| (0.15, 0.2] | 10,684 | 0.173 | 0.178 |
| (0.2, 0.3] | 10,255 | 0.242 | 0.242 |
| (0.3, 0.4] | 3,853 | 0.342 | 0.352 |
| (0.4, 0.6] | 2,378 | 0.473 | 0.457 |
| (0.6, 1.0] | 457 | 0.667 | 0.618 |

### Top pick vs odds vs profit (hold-out)

One unit on each model's top pick in every race, paid at the final pool odds (in a pari-mutuel pool every bet gets the final odds).
Break-even hit rate = average of 1/odds of the picks: the hit rate needed to make money at those prices.

| Model | Top-pick hit rate | Avg odds of the pick | Break-even hit rate | ROI betting the pick every race | Same pick as market favourite | When it differs: races / hit / odds / ROI |
|---|---|---|---|---|---|---|
| Market, final odds (known only at the off) | 33.0% | 3.06 | 36.6% | -10.6% | 100% | - |
| Market, reference odds (~30 min before) | 30.4% | 3.50 | 34.3% | -12.4% | 72% | 3,805 / 17.6% / 5.24 / -16.8% |
| Fundamental model (no odds of the race) | 30.2% | 3.81 | 32.8% | -7.9% | 64% | 4,950 / 18.7% / 5.60 / -6.2% |
| Fundamental + reference odds (conditional logit) | 30.9% | 3.40 | 34.7% | -11.8% | 75% | 3,501 / 18.4% / 5.00 / -14.2% |
| **END MODEL**: fundamental + reference odds + residual booster | 31.1% | 3.41 | 34.6% | -10.7% | 74% | 3,527 / 18.6% / 5.02 / -13.2% |
| Fundamental + final odds + residual booster | 32.8% | 3.14 | 36.1% | -9.4% | 86% | 1,872 / 21.1% / 4.59 / -9.2% |

End model's top pick, split by its odds and by its own probability:

| Pick's final odds | Races | Hit rate | Break-even | ROI |
|---|---|---|---|---|
| 1-2 | 1,857 | 54.0% | 60.3% | -10.8% |
| 2-3 | 4,348 | 38.1% | 41.3% | -7.7% |
| 3-5 | 5,730 | 23.5% | 27.1% | -13.8% |
| 5-10 | 1,773 | 15.5% | 16.7% | -6.8% |
| 10-20 | 51 | 3.9% | 9.2% | -60.8% |
| 20+ | 1 | 0.0% | 4.8% | -100.0% |

| Model probability of its pick | Races | Hit rate | Avg odds | ROI |
|---|---|---|---|---|
| 0.00-0.15 | 335 | 15.5% | 6.77 | -5.0% |
| 0.15-0.25 | 4,408 | 20.4% | 4.53 | -15.3% |
| 0.25-0.35 | 4,782 | 29.9% | 3.22 | -9.2% |
| 0.35-0.50 | 3,079 | 40.6% | 2.35 | -8.7% |
| 0.50-0.70 | 1,056 | 55.6% | 1.75 | -5.5% |
| 0.70-1.00 | 100 | 61.0% | 1.40 | -18.3% |

Development years 2018-2024:

| Model | Top-pick hit rate | Avg odds of the pick | Break-even hit rate | ROI betting the pick every race | Same pick as market favourite | When it differs: races / hit / odds / ROI |
|---|---|---|---|---|---|---|
| Market, final odds (known only at the off) | 31.8% | 3.09 | 36.1% | -13.2% | 100% | - |
| Market, reference odds (~30 min before) | 30.0% | 3.44 | 34.3% | -14.2% | 75% | 17,343 / 18.0% / 5.13 / -16.5% |
| Fundamental model (no odds of the race) | 29.6% | 3.95 | 32.0% | -6.5% | 62% | 26,421 / 19.2% / 5.81 / -1.7% |
| **END MODEL**: fundamental + reference odds + residual booster | 30.9% | 3.45 | 34.2% | -9.4% | 74% | 18,199 / 20.3% / 5.10 / -4.8% |
| Fundamental + final odds + residual booster | 32.1% | 3.21 | 35.4% | -9.3% | 84% | 11,447 / 22.7% / 4.66 / -0.8% |

End model's top pick, split by its odds and by its own probability:

| Pick's final odds | Races | Hit rate | Break-even | ROI |
|---|---|---|---|---|
| 1-2 | 8,827 | 56.3% | 59.9% | -6.3% |
| 2-3 | 22,136 | 36.6% | 41.2% | -11.2% |
| 3-5 | 29,661 | 24.1% | 27.1% | -11.0% |
| 5-10 | 9,338 | 15.7% | 16.6% | -4.1% |
| 10-20 | 312 | 10.3% | 9.1% | +11.2% |
| 20+ | 5 | 20.0% | 3.7% | +740.0% |

| Model probability of its pick | Races | Hit rate | Avg odds | ROI |
|---|---|---|---|---|
| 0.00-0.15 | 2,050 | 12.6% | 6.73 | -21.9% |
| 0.15-0.25 | 22,702 | 20.9% | 4.56 | -10.3% |
| 0.25-0.35 | 24,237 | 29.2% | 3.24 | -10.2% |
| 0.35-0.50 | 15,270 | 40.4% | 2.38 | -7.7% |
| 0.50-0.70 | 5,320 | 56.0% | 1.78 | -3.3% |
| 0.70-1.00 | 700 | 70.7% | 1.42 | -0.5% |

### Betting check (secondary)

Decide 30 minutes before the off (end-model probability x reference odds - 1 > threshold), get paid the final pool odds.
Threshold chosen on 2018-24: 0.3. Hold-out: 8,812 bets, ROI +2.4% (market-only rule, same threshold: 265 bets, ROI -29.2%).

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

## Model v2 / v3: opponent-adjusted ratings and a neural race model

v1 used one LightGBM fundamental model (157 hand-made features, at most 3 past runs per horse). v2 adds two things, and the end model, the combination models and the place study were all rebuilt on top of it.
- **Ratings** (`src/fr_exp.py ratings`): a Plackett-Luce rating for every horse, driver/jockey and trainer, updated after each race day from the full finishing order (gradient step on the finishing-order likelihood, step size shrinking with experience, slow decay). Beating a strong field counts more than beating a weak one. Features of day D only use results of earlier days.
- **A neural race model** (`src/fr_nn.py`): each runner is its feature vector plus the sequence of its last 10 starts (result, speed vs the field, beaten lengths, market opinion then, the field's mean rating = opponent strength, distance / class / track / driver changes relative to today). An attention pool summarises the history, a transformer runs over the field, and the loss is Plackett-Luce over the first 3 places. Walk-forward by year like the GBM (2 seeds, 10 epochs, chosen on 2022).
- The fundamental score is a walk-forward blend (conditional logit, no odds) of the GBM with ratings and the network(s) (`src/fr_blend.py`); the network gets most of the weight.
- **v3** (current; chosen on dev only): the network with a 20-run history, and a second one that also reads the form string (the last 12 placings, including races outside the data), each averaged over 4 seeds, blended with the GBM with ratings. It improves the 30-minute stack by a further 0.003 on dev.

### Development experiments (France 2018-2024; the hold-out was not used for any choice)

Race log-loss of the winner, walk-forward; difference vs v1 with a day-bootstrap 95 % CI.

| Fundamental variant | Alone (no odds) | Stacked with 30-min odds | Stacked with final odds |
|---|---|---|---|
| v1 GBM | 2.0526 | 1.9905 | 1.9433 |
| GBM + ratings | 2.0412 (-0.0113 [-0.0132, -0.0095]) | 1.9858 (-0.0047 [-0.0056, -0.0038]) | 1.9419 (-0.0014 [-0.0018, -0.0010]) |
| Neural race model (10 runs, 2 seeds) = v2 network | 2.0018 (-0.0508 [-0.0539, -0.0479]) | 1.9669 (-0.0235 [-0.0252, -0.0220]) | 1.9347 (-0.0086 [-0.0094, -0.0077]) |
| v2 blend (all three) | - | 1.9667 (-0.0238 [-0.0254, -0.0224]) | 1.9344 (-0.0089 [-0.0098, -0.0080]) |
| Network, 20 runs, 4 seeds | 1.9970 (-0.0556 [-0.0587, -0.0527]) | 1.9644 (-0.0260 [-0.0277, -0.0245]) | 1.9337 (-0.0095 [-0.0105, -0.0087]) |
| Network, 20 runs + form string, 4 seeds | 1.9966 (-0.0560 [-0.0590, -0.0530]) | 1.9643 (-0.0262 [-0.0279, -0.0246]) | 1.9337 (-0.0096 [-0.0106, -0.0088]) |
| **v3 blend** (GBM + ratings and both) | - | 1.9640 (-0.0265 [-0.0280, -0.0249]) | 1.9332 (-0.0101 [-0.0111, -0.0092]) |

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
| Network + form-string tokens | 2.0003 |
| Network, 20-run history + form string | 1.9983 |
| Network given this race's 30-min odds too (end-to-end; compare with the stack: 1.9425 on the same 2022 races, the end-to-end network 1.9565) | 1.9743 |

The gain is the history sequence: without it the network is no better than the GBM with ratings; the transformer over the field adds nothing; 20 runs instead of 10 and the form string add a little (both used in v3).

**The gain grows with the length of the history** (dev 2018-2024, network minus v1 GBM, race log-loss):

| Winner's earlier starts in the data | 0-1 | 1-3 | 3-6 | 6-10 | 10-20 | 20-1000 |
|---|---|---|---|---|---|---|
| Races | 5,298 | 10,361 | 12,188 | 11,975 | 18,198 | 28,072 |
| Network vs GBM | +0.022 | -0.023 | -0.026 | -0.038 | -0.044 | -0.075 |
| GBM + ratings vs GBM | -0.006 | -0.009 | -0.013 | -0.016 | -0.010 | -0.013 |

The same holds by the field's average history, and in every discipline (obstacle -0.045, plat -0.044, trot_attele -0.044, trot_monte -0.047). For debutants the network is slightly worse; for horses with 20+ earlier starts it gains 0.07. Three hand-made lags and a few aggregates cannot use the detail of the last 10 runs (who was beaten, how fast, in what company, under what conditions); attention over them can.

Leakage checks for a gain this large: the history holds only strictly earlier starts of the same horse (of 1.76M consecutive same-horse pairs, none share a race and 175 share a start time); every input of the current race is pre-race (the same features as the GBM); the gain disappears when the history is masked; and against the final odds it shrinks to -0.009, the size one expects from better information rather than from a leaked result (which would beat the final market by far more).

### v1 vs v2 vs v3 on the France hold-out (2025-01 -> 2026-05)

| | v1 | v2 | v3 (current) |
|---|---|---|---|
| Fundamental alone: log-loss | 2.0378 | 1.9837 | 1.9796 |
| Fundamental alone: top pick wins | 28.0% | 29.9% | 30.2% |
| End model, 30 min out: log-loss | 1.9794 | 1.9545 | 1.9521 |
| End model, 30 min out: top pick wins | 30.9% | 31.1% | 31.1% |
| End model, 30 min out, vs 30-min market | -0.0436 | -0.0685 | -0.0708 |
| End model with final odds, vs final market | -0.0107 | -0.0168 | -0.0174 |
| Value rule at 30 min (naive, chosen on dev): ROI | -3.5% | -1.5% | +2.4% |
| Market: 30-min / final odds log-loss | 2.0230 / 1.9321 | same | same |

The hold-out was evaluated once per version; every choice between versions was made on dev.

Hong Kong hold-out 2017-18 (1,270 races): fundamental 2.1705 -> 2.1134; end model vs final market -0.0023 -> -0.0039 [-0.0101, +0.0027]. The network improves the HK fundamental model as much as in France, but the HK final odds already contain it. The v3 changes (20 runs, 4 seeds) improve the HK fundamental a little more but not the market stack (dev -0.0012 either way), so HK stays on the v2 recipe.

## Win bets: late money, pool size and the closing line (France, v3)

In a pari-mutuel pool every bet is paid at the final odds. Late money moves toward what the model sees, so a horse the model likes at the 30-minute odds usually starts shorter. `src/fr_value.py` first predicts the final odds (regression on dev runners: log final odds = 1.00 log ref odds -0.33 log(model p / market p) + ...; i.e. the final odds shrink by edge^0.33), then bets when p x predicted final odds - 1 exceeds a threshold chosen on 2018-24.

| Rule (decided 30 min out, paid at final odds) | Chosen on dev | Dev 2018-24 | Hold-out 2025-26 |
|---|---|---|---|
| EV at the 30-min odds | p>=0.1,ev>0.5 | 10,534 bets, +5.3% | 2,249 bets, +2.5% [-6.8%, +12.2%] |
| EV at the predicted final odds (fit on all runners) | p>=0,ev>0.5 | 6,102 bets, +21.0% | 1,341 bets, +12.9% [-7.8%, +36.0%] |
| **EV at the predicted final odds of a winner** (fit on winners) | p>=0,ev>0.15 | 4,937 bets, +23.9% | 1,299 bets, +5.7% [-10.3%, +24.6%] |

What pays is the final odds of a winner, and the late money is informed: winners shorten more than losers (edge exponent -0.54 for winners against -0.33 for all runners). The winner fit is the correct expected payout, and its estimated EV matches the realised return on dev. It was adopted after the all-runner rule had been run once on the hold-out, so the two hold-out numbers should be read together: +2 % to +12 %, every interval spanning zero. The tables below use the winner fit.

Same rule (p>=0,ev>0.15) paid with the official PMU win dividends: dev 4,936 bets, +23.6% [+12.3%, +38.5%]; hold-out 1,299 bets, +6.1% [-11.2%, +26.8%]. Per year: 2018 +19%, 2019 +28%, 2020 +20%, 2021 +87%, 2022 +23%, 2023 -1%, 2024 +5%, 2025 +12%, 2026 -5%.

**Where the profit is: by size of the win pool** (net of takeout, rebuilt from the winner's dividend x winning stakes). Own stake = the bet's effect on its own payout.

| Net win pool | Dev bets | Dev ROI (no stake effect / 10 EUR / 50 EUR) | Hold-out bets | Hold-out ROI (no stake effect / 10 EUR / 50 EUR) |
|---|---|---|---|---|
| 0-20,000 EUR | 599 | +132% / +60% / +15% | 111 | +63% / +54% / +31% |
| 20,000-50,000 EUR | 229 | -10% / -10% / -11% | 20 | +1% / +1% / -0% |
| 50,000-100,000 EUR | 1,393 | +13% / +13% / +13% | 212 | +24% / +24% / +23% |
| 100,000-300,000 EUR | 2,611 | +8% / +8% / +8% | 816 | -8% / -8% / -8% |
| 300,000-max EUR | 104 | +5% / +5% / +5% | 140 | +17% / +17% / +16% |

Pools under 20k EUR (2.5% of the hold-out races, mostly regional meetings late in the programme) carry the largest returns, and that is where a stake of a few euros already moves the price. Outside them the returns scatter around zero from one band to the next (see the liquid-pool rule below). Dev returns are optimistic: the threshold was chosen on them.

**Liquid pools only** (net win pool >= 50,000 EUR, threshold re-chosen on dev): p>=0.02,ev>0.15; dev 4,102 bets, +9.8%; hold-out 1,103 bets, +6.7% [-11.5%, +27.4%].

**Accuracy by pool size, and the closing line.** The last column bets at the final odds with the final-odds model (EV > 20 %): an upper bound, since nobody can bet at the exact final odds (the French data has no snapshot between 30 minutes and the off to measure how much the odds still move).

| Net win pool | Races (hold-out) | End model (final odds) vs final market [CI] | End model (30 min) vs 30-min market | Bet at the final odds, EV > 20 %: dev | hold-out |
|---|---|---|---|---|---|
| 0-20,000 EUR | 349 | -0.0778 [-0.0982, -0.0575] | -0.2030 | 1,148 bets, +101.3% | 161 bets, +78.1% |
| 20,000-50,000 EUR | 292 | -0.0128 [-0.0317, +0.0082] | -0.0396 | 956 bets, +20.5% | 34 bets, +16.5% |
| 50,000-100,000 EUR | 3,018 | -0.0183 [-0.0241, -0.0124] | -0.0691 | 5,493 bets, +23.1% | 364 bets, +68.8% |
| 100,000-200,000 EUR | 8,480 | -0.0152 [-0.0186, -0.0116] | -0.0690 | 10,182 bets, +19.6% | 1,039 bets, +14.2% |
| 200,000-max EUR | 1,621 | -0.0150 [-0.0230, -0.0073] | -0.0608 | 1,883 bets, -15.7% | 281 bets, +6.8% |

The model beats the final market in every pool size, most in thin pools. Given the final odds it would be profitable in pools up to 200k EUR on dev, and on the hold-out except one 27-bet cell; 30 minutes out, most of that edge is taken by the late money.

### The opening edge: what if the 30-minute price could be locked?

In a pari-mutuel pool an early bet does not lock its price: it is paid at the final odds. If the 30-minute PMU price could be locked (as with a fixed-odds bookmaker or an exchange quoting the same prices), the same 30-minute decisions would return:

| Model EV at the 30-min odds | v1: locked 30-min price (hold-out) | v3: locked 30-min price (hold-out) | v3: paid at final odds (reality) |
|---|---|---|---|
| > 0% | 28,092 bets, +24% [+19%, +28%] | 34,855 bets, +27% [+23%, +31%] | -3.5% |
| > 5% | 20,589 bets, +29% [+24%, +36%] | 28,422 bets, +33% [+29%, +38%] | -1.0% |
| > 10% | 14,709 bets, +35% [+27%, +43%] | 22,967 bets, +39% [+33%, +44%] | +0.3% |
| > 20% | 7,240 bets, +47% [+34%, +60%] | 14,455 bets, +47% [+40%, +54%] | +1.2% |
| > 30% | 3,422 bets, +55% [+38%, +69%] | 8,812 bets, +56% [+46%, +65%] | +2.4% |
| > 50% | 788 bets, +64% [+29%, +104%] | 3,149 bets, +65% [+49%, +84%] | -1.3% |

The model knows 30 minutes early where the late money will go. Against a locked early price that is worth +27% to +65% (even v1: +24% to +64%); in the pari-mutuel pool the late money takes it, and the same bets are about break-even.

### How late is the late money? (Hong Kong live odds, 1,356 races 2016-18)

Market log-loss of the winner k minutes before the odds freeze (the off), against the final odds (2.0356). Snapshots near the off are about 4 minutes apart.

| Minutes before the off | 30 | 15 | 10 | 5 | 3 | last snapshot before the off |
|---|---|---|---|---|---|---|
| Log-loss | 2.1044 | 2.1024 | 2.0997 | 2.0940 | 2.0717 | 2.0712 |
| Share of the 30-min -> final gain still missing | 100% | 97% | 93% | 85% | 52% | 52% |

In Hong Kong about half of the late information reaches the odds only in the last few minutes. If France is similar, the closing-line edge is only reachable by betting in the final seconds, and only in part.

## Place bets (simple placé, France, v3)

A place bet wins if the horse finishes in the first 3 (first 2 with 4-7 starters). The pool is split equally between the placed horses: payout = 1 + (R x pool - stakes on the placed) / (m x stake on the horse), rounded down to 0.10, minimum 1.10 (1.05 in some races since 2021). PMU publishes each placed horse's payout and number of winning stakes, so (payout - 1) x stakes is equal across the placed horses of a race (checked: median deviation 2.5 %). `src/fr_place.py` prices every horse with the order models of the combination study (P(place) from Plackett-Luce with stage-2/3 strengths) and a structural model of the place pool.

- **The place pool is flatter than the win pool:** stakes rise as (win-market share)^0.72 in trotting and ^0.86 in flat/jumps. Estimated return rate 84.4% (trot) / 85.5% (flat).
- **The payouts already price most of the 30-minute edge:** regressing log(payout - 1) of placed horses on the structural prediction and log(model / market P(place)) gives -0.72 at 30 min and -0.19 with the final odds.

| P(place) model (hold-out) | Log-loss per horse | Top place pick places | Top place pick ROI |
|---|---|---|---|
| Harville, 30-min market | 0.4875 | 57.4% | -9.7% |
| Discounted Harville, 30-min market | 0.4866 | 57.4% | -9.7% |
| End model + stages, 30 min | 0.4696 | 58.9% | -7.5% |
| Harville, final market | 0.4737 | 60.4% | -7.3% |
| Discounted Harville, final market | 0.4709 | 60.3% | -7.5% |
| End model + stages, final odds | 0.4647 | 60.5% | -6.2% |

ROI of every place bet by the horse's final-odds win probability (hold-out): 0-0.02 -48%, 0.02-0.05 -26%, 0.05-0.1 -20%, 0.1-0.2 -14%, 0.2-0.3 -8%, 0.3-0.4 -3%, 0.4-0.5 -6%, 0.5-0.6 -2%, 0.6-0.7 -8%, 0.7-1.01 +6%.
Heavy favourites, where the minimum payout binds, come close to break-even but do not beat it.

| Value rule (chosen on dev) | Chosen | Dev | Hold-out |
|---|---|---|---|
| 30 min out | p>=0.6,ev>0.1 | 2,841 bets, +4.0% | 755 bets, +4.3% [-0.6%, +9.2%] |
| final odds (upper bound) | p>=0.2,ev>0.2 | 1,067 bets, +28.9% | 92 bets, +53.9% [+10.5%, +98.5%] |

The 30-minute rule by size of the place pool (gross payout pool = return rate x pool, which does not depend on the result; bands = quintiles of dev races):

| Place pool (EUR) | Dev | Hold-out |
|---|---|---|
| 0-54,322 | 597 bets, +10.4% | 136 bets, +10.4% |
| 54,322-76,018 | 484 bets, +3.0% | 122 bets, -7.7% |
| 76,018-96,852 | 538 bets, -1.5% | 194 bets, +9.3% |
| 96,852-123,564 | 597 bets, +0.2% | 175 bets, +5.0% |
| 123,564-max | 625 bets, +7.0% | 128 bets, +0.7% |

No pool-size pattern. (A first version filtered on the *net* place pool, R x pool minus the stakes on the placed horses. That quantity shrinks when favourites place, so filtering on it selects races where favourites placed: it showed a spurious +15 % on dev and hold-out alike for these favourite bets. With the outcome-free measure the effect is gone: the cap chosen on dev gives 258 bets, +1.8% [-7.8%, +11.6%] on the hold-out.)

Realised ROI by estimated EV (hold-out): 30 min: -0.05-0 -5% (10,217), 0-0.05 -5% (6,008), 0.05-0.1 -2% (2,918), 0.1-0.2 +0% (1,557), 0.2-1e+09 +3% (183); final: -0.05-0 -4% (11,549), 0-0.05 -2% (5,854), 0.05-0.1 +3% (2,385), 0.1-0.2 +4% (1,043), 0.2-1e+09 +57% (125)

## Combination bets: couplé (first two) and trio (first three)

Order of finish: a Plackett-Luce model. The win probability comes from the end model. The 2nd and 3rd places use their own strengths (a conditional logit plus a small residual LightGBM per stage, walk-forward by year: train on 2018..Y-1, predict Y). Harville (2nd/3rd = the win probability renormalised) is the textbook baseline. Every combination of every race is enumerated. Dev = 2019-2024, test = 2025-01 -> 2026-05. *ref* = decided 30 minutes before the off (reference odds); *final* = with the final win odds (upper bound, not usable to bet). Payouts are the official PMU dividends per 1 EUR (paid at the final pool). Races where nobody held the winning combination (the pool then pays consolation combinations) or that were refunded are left out of the payout analysis (about 2 % of couplé races, a conservative choice for long-shot strategies).

**Summary**

- **The model prices combinations much better than the textbook method.** Against Harville on the 30-minute market, the stage models cut log-loss by 0.128 (quinella) and 0.193 (trio); the combination probabilities are well calibrated (table below). For the quinella most of the gain comes from the better win model; for the trio about two thirds come from modelling 2nd and 3rd place separately. Harville overrates favourites for the minor places: the fitted discount exponent is about 0.8 for 2nd and 0.7 for 3rd (lower in trotting).
- **But the combination pools are sharper still.** On the test races where the pool ran, the trio pool's own implied log-loss is 4.274, against 4.391 for the model 30 minutes out and 4.336 even with the final win odds. For the quinella the model with final odds beats the pool (3.352 vs 3.378), but 30 minutes out it is worse (3.404). Exotic bettors do not price like Harville on the win odds.
- **No reliable profit.** Betting the model's top combination returns -31% to -15% (30-minute version). The value rules chosen on dev give, on test: Couplé ordre 30 min +3% (48 hits); Couplé ordre final odds -11% (7 hits); Couplé gagnant 30 min -4% (55 hits); Couplé gagnant final odds -86% (1 hits); Trio ordre 30 min -16% (62 hits); Trio ordre final odds +26% (24 hits); Trio 30 min +1% (31 hits); Trio final odds -35% (13 hits). Every confidence interval spans zero widely: the rules rest on 1 to 62 winning tickets. The takeout on combinations (22-25 %) is larger than on win bets (about 16.5 %).
- **Hong Kong** gives the same picture: the model adds nothing to the quinella pool and only matches the tierce pool, and every model-edge ROI bin loses.

### Accuracy of the combination probabilities (test, all races with a clear 1-2-3)

Log-loss of the actual combination (lower is better) / hit rate of the model's single most likely combination.

| Pool | Races | Harville on the 30-min market | End model + Harville | **End model + stage models (ref)** | Harville on the final market | End model + stage models (final) |
|---|---|---|---|---|---|---|
| Couplé ordre (exacta) | 13,710 | 4.123 / 7.3% | 4.009 / 8.0% | **3.978 / 8.2%** | 3.976 / 8.6% | 3.922 / 9.0% |
| Couplé gagnant (quinella) | 13,710 | 3.453 / 12.6% | 3.342 / 13.7% | **3.325 / 13.8%** | 3.310 / 14.5% | 3.275 / 14.8% |
| Trio ordre (trifecta) | 13,675 | 6.219 / 2.1% | 6.099 / 2.1% | **5.982 / 2.1%** | 6.054 / 2.3% | 5.917 / 2.3% |
| Trio (any order) | 13,675 | 4.493 / 7.5% | 4.386 / 8.3% | **4.299 / 7.9%** | 4.344 / 8.7% | 4.248 / 8.6% |

Stage models vs Harville on the same market (test; bootstrap by race day):

| Pool | ref: log-loss diff [95 % CI] | ref: top-combination hit diff | final: log-loss diff [95 % CI] |
|---|---|---|---|
| Couplé ordre (exacta) | -0.144 [-0.152, -0.137] | +0.90% [+0.51%, +1.31%] | -0.053 [-0.058, -0.048] |
| Couplé gagnant (quinella) | -0.128 [-0.135, -0.119] | +1.18% [+0.70%, +1.66%] | -0.035 [-0.040, -0.031] |
| Trio ordre (trifecta) | -0.237 [-0.249, -0.226] | +0.04% [-0.23%, +0.29%] | -0.137 [-0.145, -0.128] |
| Trio (any order) | -0.193 [-0.203, -0.183] | +0.43% [+0.04%, +0.81%] | -0.096 [-0.103, -0.088] |

Calibration of the end model's combination probabilities (test, ref): mean predicted vs actual frequency.

| Predicted | Quinella combos | Quinella predicted / actual | Trio combos | Trio predicted / actual |
|---|---|---|---|---|
| 0-0.0005 | 130,369 | 0.0002 / 0.0002 | 1,293,111 | 0.0002 / 0.0002 |
| 0.0005-0.001 | 60,847 | 0.0007 / 0.0008 | 439,087 | 0.0007 / 0.0007 |
| 0.001-0.002 | 86,158 | 0.0015 / 0.0016 | 488,303 | 0.0014 / 0.0014 |
| 0.002-0.005 | 163,982 | 0.0034 / 0.0035 | 598,351 | 0.0032 / 0.0032 |
| 0.005-0.01 | 150,971 | 0.0072 / 0.0072 | 327,847 | 0.0070 / 0.0068 |
| 0.01-0.02 | 144,994 | 0.0143 / 0.0144 | 196,525 | 0.0139 / 0.0136 |
| 0.02-0.05 | 131,632 | 0.0312 / 0.0301 | 107,959 | 0.0298 / 0.0302 |
| 0.05-0.1 | 45,281 | 0.0685 / 0.0681 | 21,436 | 0.0665 / 0.0682 |
| 0.1-0.2 | 14,728 | 0.1331 / 0.1384 | 4,957 | 0.1315 / 0.1281 |
| 0.2-0.3 | 2,039 | 0.2354 / 0.2320 | 675 | 0.2351 / 0.2548 |
| 0.3-1.01 | 550 | 0.3753 / 0.3545 | 258 | 0.4085 / 0.3953 |

### Against the combination pool itself (test, races where the pool ran)

The pool's probability for the winning combination = return to players / payout (returns assumed: couplé 74.8 %, trio 77.9 %, OFDT/PMU averages; the ordre variants assumed equal). *Growth* = mean log return of spreading the bankroll over every combination in proportion to the model's probability: it does not depend on the assumed takeout, and > 0 would mean the model beats the pool after takeout.

| Pool | Races | Median payout | Pool log-loss | Model (ref) log-loss / growth | Harville final market log-loss / growth | Model (final) log-loss / growth |
|---|---|---|---|---|---|---|
| Couplé ordre (exacta) | 989 | 13.7 | 2.908 | 2.935 / -0.318 | 2.943 / -0.326 | 2.896 / -0.278 |
| Couplé gagnant (quinella) | 12,753 | 19.8 | 3.378 | 3.404 / -0.317 | 3.388 / -0.300 | 3.352 / -0.265 |
| Trio ordre (trifecta) | 979 | 41.2 | 4.019 | 4.214 / -0.445 | 4.271 / -0.501 | 4.173 / -0.404 |
| Trio (any order) | 12,093 | 51.4 | 4.274 | 4.391 / -0.366 | 4.431 / -0.406 | 4.336 / -0.312 |

**The pools already price most of what the model knows.** Regressing the log payout of the winning combination on the market-Harville probability H and the model's edge P/H (dev winners) gives an edge coefficient of Couplé ordre -0.61, Couplé gagnant -0.59, Trio ordre -0.73, Trio -0.68 (0 = the pool ignores what the model knows, -1 = the pool prices all of it). So, 30 minutes before the off, roughly 70-80 % of the model's apparent edge over Harville is already in the final combination payouts.

### Top combination vs payout vs profit (test, 1 unit per race)

| Pool | Model | Hit rate | Avg payout when hit | ROI [95 % CI] | Same as market's top | When different: bets / hit / ROI |
|---|---|---|---|---|---|---|
| Couplé ordre (exacta) | mkt_harv_ref | 13.6% | 5.08 | -31.2% [-41.8%, -18.7%] | 100% | - |
| Couplé ordre (exacta) | end_stage_ref | 14.6% | 5.05 | -26.5% [-38.5%, -13.5%] | 57% | 427 / 11.2% / -29.0% |
| Couplé ordre (exacta) | end_stage_final | 15.4% | 4.83 | -25.7% [-36.8%, -13.9%] | 75% | 251 / 12.8% / -22.4% |
| Couplé gagnant (quinella) | mkt_harv_ref | 11.8% | 6.47 | -23.4% [-27.7%, -19.1%] | 100% | - |
| Couplé gagnant (quinella) | end_stage_ref | 13.0% | 6.51 | -15.4% [-20.0%, -11.0%] | 57% | 5,450 / 10.5% / -10.9% |
| Couplé gagnant (quinella) | end_stage_final | 14.0% | 6.38 | -10.9% [-15.3%, -6.7%] | 77% | 2,995 / 10.4% / -3.0% |
| Trio ordre (trifecta) | mkt_harv_ref | 6.3% | 8.99 | -43.1% [-57.4%, -27.0%] | 100% | - |
| Trio ordre (trifecta) | end_stage_ref | 6.8% | 12.09 | -17.2% [-43.7%, +14.6%] | 39% | 601 / 5.5% / -4.3% |
| Trio ordre (trifecta) | end_stage_final | 6.6% | 10.37 | -31.1% [-51.6%, -5.6%] | 58% | 415 / 4.8% / -25.4% |
| Trio (any order) | mkt_harv_ref | 6.5% | 9.05 | -40.8% [-45.7%, -36.1%] | 100% | - |
| Trio (any order) | end_stage_ref | 7.0% | 9.84 | -31.1% [-36.7%, -24.8%] | 48% | 6,316 / 5.6% / -24.0% |
| Trio (any order) | end_stage_final | 7.7% | 9.89 | -24.2% [-30.4%, -18.3%] | 66% | 4,158 / 5.6% / -16.1% |

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
| 0-0.5 | -73% (1,151,359) | -73% (266,713) | -72% (4,019,734) | -76% (975,734) |
| 0.5-0.67 | -44% (457,555) | -47% (102,239) | -50% (1,538,871) | -47% (319,045) |
| 0.67-0.8 | -40% (366,101) | -31% (83,035) | -35% (1,127,726) | -42% (235,123) |
| 0.8-0.9 | -34% (276,600) | -31% (62,677) | -28% (809,348) | -25% (170,238) |
| 0.9-1.1 | -30% (496,490) | -26% (112,680) | -29% (1,420,079) | -25% (300,416) |
| 1.1-1.25 | -25% (301,600) | -17% (68,652) | -28% (875,962) | -30% (187,672) |
| 1.25-1.5 | -20% (363,474) | -20% (83,355) | -17% (1,132,865) | -15% (245,790) |
| 1.5-2 | -17% (357,274) | -13% (83,527) | -16% (1,371,946) | -8% (306,673) |
| 2-3 | -11% (170,474) | -8% (41,383) | -13% (1,057,389) | -4% (247,634) |
| 3+ | +4% (36,448) | +12% (9,372) | -5% (625,707) | +27% (160,518) |

### Value rule (bet when estimated EV > threshold; chosen on dev, applied once to test)

EV = model probability x expected payout. Expected payout comes from the regression above (fit on dev winners). Threshold and minimum model probability are picked on dev, best ROI with at least 1,000 dev bets (300 for the small ordre pools).

| Pool | Variant | Rule chosen on dev | Dev | Test |
|---|---|---|---|---|
| Couplé ordre (exacta) | ref | p>=0.02,ev>0 | 1,902 bets, 203 hits, ROI -3.5%  | 386 bets, 48 hits, ROI +3.2% [-38.5%, +52.8%] |
| Couplé ordre (exacta) | final | p>=0.005,ev>0.1 | 935 bets, 67 hits, ROI +15.6%  | 69 bets, 7 hits, ROI -11.5% [-83.5%, +80.0%] |
| Couplé gagnant (quinella) | ref | p>=0.02,ev>0.2 | 2,060 bets, 184 hits, ROI +12.9%  | 602 bets, 55 hits, ROI -3.6% [-39.6%, +43.5%] |
| Couplé gagnant (quinella) | final | p>=0.005,ev>0.5 | 1,140 bets, 48 hits, ROI +107.5%  | 49 bets, 1 hits, ROI -85.7%  |
| Trio ordre (trifecta) | ref | p>=0.02,ev>-0.1 | 4,393 bets, 284 hits, ROI -5.9%  | 976 bets, 62 hits, ROI -15.8% [-40.0%, +10.2%] |
| Trio ordre (trifecta) | final | p>=0.001,ev>0 | 5,618 bets, 97 hits, ROI +12.1%  | 1,132 bets, 24 hits, ROI +26.4% [-32.7%, +105.8%] |
| Trio (any order) | ref | p>=0.001,ev>0.3 | 7,888 bets, 112 hits, ROI +25.1%  | 2,451 bets, 31 hits, ROI +1.1% [-45.7%, +76.3%] |
| Trio (any order) | final | p>=0.001,ev>0.5 | 10,362 bets, 73 hits, ROI +60.0%  | 1,582 bets, 13 hits, ROI -34.9% [-77.5%, +15.7%] |

Realised ROI by estimated EV (test, ref; is the EV estimate right?):

| Estimated EV | Quinella | Trio |
|---|---|---|
| -0.3..-0.2 | -20.9% (139,304) | -11.8% (554,604) |
| -0.2..-0.1 | -11.2% (81,678) | +0.9% (324,878) |
| -0.1..0 | -9.2% (32,520) | +13.0% (129,117) |
| 0..0.1 | -9.5% (9,420) | +12.3% (41,928) |
| 0.1..0.3 | -0.6% (2,736) | -3.0% (17,323) |
| 0.3+ | -42.6% (172) | -9.1% (2,727) |

### Hong Kong replication (final odds only; dividends exist for 2016-09 -> 2018-06)

Same order models (conditional logits per stage, walk-forward by year). The value rule is France's (final variant), not re-tuned; the payout model is fit on the 239 HK races of 2016.

| Pool | Test races | Harville on market | Discounted Harville on market | End model + stages | Pool log-loss (takeout 17.5 % / 25 %) | Model growth | Top combo ROI (model) | France rule on HK test |
|---|---|---|---|---|---|---|---|---|
| Quinella | 1,269 | 3.406 / 13.6% | 3.408 / 13.6% | 3.405 / 14.0% | 3.398 | -0.198 | -6.1% | p>=0.005,ev>0.3: 4 bets, 2 hits, ROI +67.5%  |
| Tierce (trifecta) | 1,269 | 6.177 / 1.6% | 6.149 / 1.6% | 6.141 / 1.4% | 6.145 | -0.276 | -55.3% | p>=0.001,ev>0: 11,240 bets, 97 hits, ROI -2.1% [-32.0%, +33.5%] |
