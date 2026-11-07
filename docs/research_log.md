> Working notes written during the research (2026-11-03/04, when the project was called HorseCash). Later work and the final numbers
> are in the main README and `fresh/README.md`. Paths below are relative to the repository root.

# HorseCash: can a model beat the betting market's win probabilities? (research)

This is the follow-up to UFCash. There, a 70 % model only matched the closing odds, and a 1,500-fight hold-out was too small
to tell a real improvement from luck. Horse racing gives about 20× more events, each ranking a whole field, with the market's
probabilities available for every runner.

Status: **model v3 built and evaluated** (night of 2026-11-03/04): opponent-adjusted ratings and a neural race model that reads each
horse's last 20 runs and its form string. Win, place and combination bets were all re-evaluated on it. The full report is in
[`results/REPORT.md`](../results/REPORT.md); v1 and v2 results are kept in `results/v1/` and `results/v2/`.

## Results (France hold-out 2025-01-01 → 2026-05-05, 13,760 races, never used for any choice)

| Model | Top pick wins | Winner in top 3 | Log-loss | Betting the top pick every race (paid at final odds) |
|---|---|---|---|---|
| Random pick (12-horse field) | 8.3 % | 25 % | 2.48 | about −16.5 % (pool cut) |
| Market, reference odds (30 min before the off) | 30.4 % | 61.8 % | 2.023 | −12.4 % |
| Fundamental model v1 (gradient boosting, never sees this race's odds) | 28.0 % | 59.4 % | 2.038 | −11.8 % |
| **Fundamental model v3** (neural race model + ratings, never sees this race's odds) | 30.2 % | 61.8 % | **1.980** | −7.9 % |
| **End model v3** (fundamental + reference odds), usable 30 min before | **31.1 %** | 63.8 % | **1.952** | −10.7 % |
| Market, final odds (only known at the off) | 33.0 % | 64.8 % | 1.932 | −10.6 % |
| End model v3 with the final odds (upper bound) | 32.8 % | 65.2 % | **1.915** | −9.4 % |

- **Without seeing the race's odds, the model now beats the 30-minute market** (log-loss −0.043, CI −0.051 to −0.035). v1 could not.
- **The end model beats the market available when betting by a wide margin** (−0.071, CI −0.076 to −0.066; top pick +0.7 points).
- **With the final odds as input it beats the closing market** (−0.017, CI −0.020 to −0.015), in every pool size and most in thin pools.
- **Where the gain comes from:** encoding each horse's recent runs as a sequence (with the field's rating as opponent strength) is worth
  about −0.05 log-loss on its own. Without the history the network is no better than gradient boosting; attention across the field adds nothing.
  The gain grows with the length of the history: +0.02 (worse) for horses with no earlier start in the data, −0.07 for 20+ earlier starts.
  Giving the network the 30-minute odds directly (end-to-end) is worse than stacking it with the market (1.957 vs 1.943 on 2022).
- **Versions** (hold-out log-loss of the end model 30 min out): v1 1.979, v2 (10-run network) 1.955, v3 (20 runs + form string, 4 seeds) 1.952.
  Each version was chosen on 2018-24 only.
- **Hong Kong:** the network improves the fundamental model there too (−0.04), but the end model adds only −0.004 to the final odds
  (not significant). That market is bigger and tighter; the v3 network changes nothing there (HK stays on the v2 recipe).

## Betting (France hold-out, paid at the official PMU dividends)

| Bet, decided 30 minutes before the off | Hold-out result (v3) |
|---|---|
| Win, naive value rule (EV at the 30-min odds) | +2.4 % (8,812 bets); v1 −3.5 %, v2 −1.5 % |
| Win, rule that predicts the final odds of a winner first (late money moves toward what the model sees) | +6.1 % (1,299 bets, CI −11 % to +27 %) |
| ... same with the first payout model (fit on all runners, overstates the payout) | +12.9 % (1,341 bets, CI −8 % to +36 %) |
| ... winner fit, liquid pools only (net win pool ≥ 50k EUR) | +6.7 % (1,103 bets, CI −11 % to +27 %) |
| Place (simple placé), rule chosen on dev | +4.3 % (755 bets, CI −0.6 % to +9.2 %) |
| Couplé gagnant / trio, rules chosen on dev | −3.6 % (55 winning tickets) / +1.1 % (31 winning tickets): noise |

- **Accuracy has reached roughly break-even against a 16.5 % takeout, but nothing is significant.** Every 30-minute rule is positive on the
  hold-out (+2 % to +7 %), with intervals that include zero. A year or two more of hold-out would be needed to tell.
- **The "opening edge" is real, but a pari-mutuel pool does not let you keep it.** If the 30-minute PMU price could be locked (fixed odds
  or an exchange at the same prices), the same decisions would return +27 % (EV > 0, 35k bets) to +65 % on the hold-out, all
  significant. The model sees where the late money will go; in the pool that late money is exactly what removes the edge.
- **The edge is at the close.** Deciding with the final odds, the model would make +7 % to +78 % per pool-size band at EV > 20 % on the hold-out.
  Nobody can bet at the exact final odds. In Hong Kong's live odds, about half of the late information arrives only in the last few minutes before the off.
- **A trap worth knowing:** filtering place bets on the *net* place pool (pool minus the stakes on the placed horses) gave +15 % on dev
  and hold-out alike. That quantity shrinks when favourites place, so the filter used the result. With an outcome-free pool measure the
  effect disappears.
- **Pools:** the place pool returns about 84-85 %. Its stakes are flatter than the win pool's (share^0.72 in trot, ^0.86 in flat), and its
  payouts already price most of the 30-minute edge. The combination pools (couplé 74.8 %, trio 77.9 % returned) are about as sharp as the model.

## Combination bets (couplé / trio), v3

Log-loss of the winning combination on the test races where the pool ran (lower is better). The pool's own probability is the return to
players divided by the payout, at the official average takeout (couplé 25.2 %, trio 22.1 %).

| Pool (test) | Races | Harville on the 30-min market | **Model, 30 min out** | Model with final odds | Pool's own prices | Top combination ROI (30 min) |
|---|---|---|---|---|---|---|
| Couplé gagnant (first two, any order) | 12,753 | 3.533 | **3.404** | 3.352 | 3.378 | −15.4 % |
| Trio (first three, any order) | 12,093 | 4.585 | **4.391** | 4.336 | 4.274 | −31.1 % |

- The model beats the textbook method (Harville) by a wide margin; with the final odds it now beats the quinella pool, not the trio pool.
- No combination rule is reliably profitable: every rule's confidence interval spans zero widely.

## Data

| | France (PMU) | Hong Kong (HKJC) |
|---|---|---|
| Source | HF `annaelmoussa/horse-racing-france`, scraped from PMU, license "other" (as-is) | GitHub `eprochasson/horserace_data`, scraped, no guarantee |
| Period | 2014-01 → 2026-05 | 1979 → 2018-06 |
| Size | 157k races, 1.96M declared runners (1.91M starters) | 24k races, 274k runners |
| Runners with odds | 85–99 % per year from 2015 (2014: 51 %) | 100 % |
| Odds | **final** pool odds (snapshot a median 1.8 min after the scheduled off) and **reference** odds (median 31 min before the off) | final pool odds. Pre-race snapshots for 1,519 races (2016–18, median 171 per race) |
| Disciplines | harness trot 57k races, ridden trot 11k, flat 43k, hurdles 8k, steeplechase 5k (races with full odds) | flat, Sha Tin and Happy Valley |
| Runner info | form string, career starts/wins/places/earnings (this and last year), age, sex, driver/jockey, trainer, owner, breeder, sire/dam/dam-sire, draw, weight/handicap, distance handicap, shoes, blinkers, trainer's opinion, claiming price | official rating, weight carried, body weight, draw, gear, jockey, trainer |
| Race info | distance, prize money, conditions, class, going (penetrometer), track type, start time | class, distance, course, going |
| Results | finishing order, time, reduction per km (trot), margin, incidents, post-race comment | placing, finish time, lengths behind, running positions, sectionals (2008+) |

Raw files are in `data/raw/{fr,hk}`. The flattened tables are `data/processed/fr_runners.parquet` (91 columns), `fr_races.parquet` and `hk_runners.parquet`.
Logs of the checks are in `results/`.

## Checks (no modelling)

**France** (`src/fr_checks.py`)
- **The final odds field is the price that paid out.** Against the official win payout of the winner, the ratio has a median of 1.000 and is within 5 % for 94 % of winners.
- **The reference odds are an earlier price.** They are within 5 % of the payout for only 11 % of winners, and are taken about 30 minutes before the off.
- **Career stats are as of the race, not a later snapshot.** Between a horse's consecutive starts, `nombreCourses` rises by exactly 1 in 84 % of cases and by more than 1 in 15 % (races outside the data). It stays flat in only 0.6 %.
  The next start's form string begins with this race's placing 87 % of the time. Earnings never decrease in 99.3 % of cases.
- **Pool margin:** the sum of 1/odds has a median of 1.197 (about 16.5 % takeout).
- **Favourite-longshot bias is strong.**
  - Runners priced at over 40 % win 51.5 % against 47 % implied.
  - Runners priced under 2 % win 0.84 % against 1.18 % implied.
  - After the pool's cut, even favourites lose money.
- **Late information matters.** Market log-loss is 1.99 at the final odds and 2.04 at the reference odds 30 minutes earlier (harness trot).
  A uniform guess scores 2.57. Favourites win 33 % (trot) and 29 % (flat).
- **Post-race fields, never to be used as features:**
  - runners: `raw.ordreArrivee`, `raw.tempsObtenu`, `raw.reductionKilometrique`, `raw.distanceChevalPrecedent.*`, `raw.commentaireApresCourse.*`, `incident`;
  - races: `ordreArrivee`, `arriveeDefinitive`, `incidents`, `raw.dureeCourse`.
  The final odds (`dernierRapportDirect.*`) are market input only.

**Hong Kong** (`src/hk_checks.py`)
- 96 % of races have exactly one recorded winner (3.8 % have none, 0.2 % dead heats). 23,042 races are clean.
- The sum of 1/odds has a median of 1.227, consistent with the 17.5 % takeout plus rounding, so these are final pool odds.
- **`rating` is the rating carried into the race.** The rating change after race t correlates −0.63 with the placing in race t.
  Winners rise by +6.7 on average.
- Favourites win 27.5 % (1979–99) and 30 % (2000–18). Market log-loss is about 2.03, against 2.4–2.5 for a uniform guess.
  The favourite-longshot bias is milder than in France.
- The last live snapshot equals the final odds for 89 % of runners. The first snapshots, taken the day before, are far off
  (median |log ratio| 1.3), because the pools are still nearly empty.

## Model

- **Fundamental** (`src/fr_model.py`): LightGBM trained with a race-level softmax objective, so every horse is scored against its own field.
  - 157 pre-race features (`src/fr_features.py`): programme data, the form string, the horse's earlier races in the data (speed per km vs the field, beaten lengths, class/distance changes, past market opinion, actual/expected), and exponentially weighted driver/jockey, trainer, sire and owner statistics. All are computed from strictly earlier races.
  - Retrained every year; trained from 2015.
- **End model** (`src/fr_eval.py`, `src/fr_final.py`):
  - A conditional logit on [log reference-odds probability, fundamental score] (× trot).
  - A small LightGBM boosted from that score, which learns where the combination is still wrong.
- **v2/v3 fundamental** (`src/fr_exp.py`, `src/fr_nn.py`, `src/fr_blend.py`): Plackett-Luce ratings of horse, driver and trainer (updated
  after each race day); the GBM with those ratings; a neural race model (each runner's features + an attention pool over its recent starts +
  a transformer over the field, Plackett-Luce loss over the first 3 places). v2: 10 starts, 2 seeds. v3: 20 starts, plus a variant that also
  reads the form string, 4 seeds each. All blended walk-forward into one score.
- **Hong Kong** (`src/hk_pipeline.py`, `src/hk_exp.py`, `src/hk_nn.py`): the same method with HK-specific features. The final odds are
  the only market there.

## Usage

```
./run_all.sh                                  # everything, from the raw files (about 1.5 h on 28 cores)
python -I src/fr_final.py predict 2026-05-05  # score a race day with the v1 fundamental model (the v2 network is not wired in here)
```
`NTHREADS=12` limits LightGBM threads when the machine is shared.

## Layout

```
src/fr_extract.py   PMU archive -> fr_runners / fr_races parquet
src/fr_checks.py    France data checks (odds semantics, as-of stats, market, post-race fields)
src/hk_checks.py    Hong Kong data checks -> hk_runners.parquet
src/fr_features.py  France pre-race features
src/racemodel.py    race-softmax LightGBM objective, conditional logit, metrics, top-pick analysis
src/fr_model.py     fundamental model: dev rounds + walk-forward out-of-fold
src/fr_eval.py      stackers, evaluation, top-pick vs odds vs profit, betting check -> results/fr_report.json
src/fr_final.py     end model on all data (models/) + prediction for a race day
src/hk_pipeline.py  Hong Kong features, model, evaluation -> results/hk_report.json
src/exotics.py      combination engine (Plackett-Luce with stage strengths, every combination of every race, numba)
src/fr_exotics.py   France 2nd/3rd-place models + couplé / trio evaluation -> results/exotics_report.json
src/hk_exotics.py   Hong Kong quinella / tierce replication -> results/hk_exotics_report.json
src/fr_exp.py       v2: Plackett-Luce ratings, GBM variants (ratings, top-3 objective), dev stack comparisons
src/fr_nn.py        v2: neural race model (history attention + field transformer, GPU), tuning, walk-forward, ablations
src/fr_blend.py     v2: walk-forward blend of fundamental variants -> fr_oof.parquet (v1 kept as fr_oof_v1.parquet)
src/hk_exp.py       Hong Kong ratings / GBM variants / stack / blend;   src/hk_nn.py  Hong Kong neural race model
src/fr_place.py     simple placé: P(place) from the order models, place-pool structure, value rule -> results/place_report.json
src/fr_value.py     win value rules: predicted final odds, official dividends, own-stake effect, pool size -> results/value_report.json
src/hk_timing.py    Hong Kong live odds: how much information the market still lacks k minutes before the off -> results/hk_timing.json
src/make_report.py  results/REPORT.md
```

## Protocol

- **Target:** the win probability of every runner (multinomial over the field).
- **Score:** log-loss of the winner's probability and top-pick hit rate, each against the market. Both final odds and the
  30-minute reference odds are benchmarks.
- **France:** select on 2016–2024, then a single hold-out on 2025-01-01 → 2026-05 (about 17k races).
- **Hong Kong:** select on ≤ 2016, hold out 2017–18. A method has to work in both markets to count.
- **Splits and features:** yearly walk-forward. Every feature is computed from earlier races only (strictly earlier dates).
