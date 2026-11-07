# Turfolio

**Can a model built by an LLM agent beat the French horse-racing betting pools (PMU)?**

At prediction, yes. The model gives the winner a higher probability than the market's own odds 30 minutes before the off, and it still does on six
months of races that nothing in the project ever touched. Simply betting on its picks loses money, because pari-mutuel prices move at the
last minute and PMU keeps about 16 %. One rule, **place bets where the model says a horse is much likelier to finish in the paid places than
the place pool implies**, made money in every out-of-sample test: **+4.6 % on 1,069 bets** (95 % CI +0.4 % to +8.8 %). On the one fully
pre-registered window alone it made +5.9 %, with a CI that still includes zero. A small, plausible edge, not a proven one.

This is research code, not betting advice. Nothing here was bet with real money.

## Headline results

Winner log-loss (lower is better) and top-pick hit rate, French races. "30-min market" = PMU's reference odds about 30 minutes before the off,
margin removed; "final market" = the final pool odds (known only at the off). 95 % CIs by bootstrap over race days.

| | Hold-out 2025-01 → 2026-05-05 (13,760 races) | Fresh window 2026-05-06 → 11-05 (5,488 races) |
|---|---|---|
| 30-min market | 2.023 · top pick 30.4 % | 1.984 · 30.3 % |
| Model without any odds (`p_fundamental`) | 1.980, **−0.043** [−0.051, −0.035] | 1.942, **−0.042** [−0.056, −0.029] |
| Model + 30-min odds (`p_win`) | 1.952, **−0.071** [−0.076, −0.066] · 31.1 % | 1.914, **−0.071** [−0.079, −0.062] · 31.8 % |
| Final market | 1.932 · 33.0 % | 1.884 · 33.5 % |
| Model + final odds (upper bound) | 1.915, **−0.017** [−0.020, −0.015] vs final | 1.868, **−0.015** [−0.020, −0.011] vs final |

Betting, flat stakes, paid at PMU's **official** dividends (thresholds always chosen on earlier years):

| Rule | Hold-out 2025-01 → 2026-05 | Fresh window (pre-registered) |
|---|---|---|
| Win: value rule (edge vs predicted final odds) | +6.1 % (1,299 bets), **−1.3 % when the pipeline is retrained** | −5.9 % (738 bets) |
| Win: back the top pick when the model gives it ≥ 60 % | −5.4 % (457) | −13.1 % (199) [−23 %, −2 %] |
| Win: value rule + "late-move" network head | −10.3 % (591) | −24.5 % (647) |
| Place: back the most likely placer when ≥ 80 % (`rule A`) | −1.9 % (859) | −3.5 % (373) |
| **Place: P(place) ≥ 0.6 and estimated EV > 0.1 (`rule B`)** | **+4.3 %** (755) [−0.9 %, +9.4 %] | **+6.1 %** (320) [−2.0 %, +14.3 %]; +5.9 % with our own €10 in the pool |

Rule B is the only rule that was positive in every period: +4.0 % on 2019–2024 where it was chosen, +4.3 % and +6.1 % after. Retraining the whole
pipeline changes its hold-out result by 0.02 points (the win value rule moved by 7 points). Paper bank on the fresh window, €10 flat stakes:
€1,000 → **€1,189** (4 of 7 months positive). Both out-of-sample windows pooled: +4.6 % [+0.4 %, +8.8 %] with our own stake in the pool.

## What did not work, and why

- **Confidence is not value.** The model's most confident picks are always the market favourite, and their price already reflects it.
  At a 62 % hit rate you need an average payout of 1.62; these pay about 1.53. Every confidence threshold lost on every period.
- **Late money.** The 30-min odds are beaten easily, the final odds barely. Odds keep moving to the off and winners shorten most:
  from the 30-min to the final odds the winner's price falls by 7 % on average in 2023 and 12 % in mid-2026, so the late market gets sharper every year.
  Even the main-pool price within one minute of the off (from a public minute-level capture) is clearly worse than the final market
  (+0.024 [0.008, 0.044] log-loss), and model + that price does not close the gap. The final odds cannot be had at any bettable moment.
- **Win value bets** looked good on development years (+24 %) and +6.1 % on the hold-out, then the same code retrained with other seeds gave −1.3 %
  on that same hold-out, and −6 % on the fresh window. The payout model fitted on 2018–24 overstates what winners pay in 2026 (late money again).
- **Quarter-hour internet-pool prices** (Canalturf, 2022+) add almost nothing to the model: the information arrives after the last sample.
- **PMU's internet pool** pays more on non-favourites (+2–9 %) but choosing the pool did not make the value rule significant.
- **Learning from the final market.** Distilling it into the network hurt; a second head predicting the late move improved accuracy
  (−0.004 log-loss, replicated on both test windows) but its betting rule lost.
- **Combination bets** (couplé, trio): PMU's combination pools are sharper than the model built on the win odds.
- **Hong Kong** (1979–2018): the model improves on fundamentals, but the market stack is level with the final odds (−0.004 [−0.010, +0.003]).

## Method

- **Data.** PMU races 2014-01 → 2026-05 from the Hugging Face archive `annaelmoussa/horse-racing-france` (157k races, 1.96M runners, with
  final and 30-min reference win odds), then 2026-05-06 → 11-05 rebuilt from PMU's public JSON API by `src/fr_fresh.py` in exactly the archive's
  format (checked field by field on two overlapping weeks: every field the pipeline reads agrees ≥ 99.9 %).
- **No leakage.** Every feature uses only earlier races; career figures are as of the race. Every model is **walk-forward**: to predict year Y,
  everything is trained on years before Y. The fundamental model never sees the race's odds; odds enter only in the stack on top.
- **Model (v3).** LightGBM with Plackett–Luce ratings of horse, driver and trainer, plus two neural networks (attention over each horse's last
  20 starts and its form string, a transformer over the field, 4 seeds each), blended by a conditional logit. The end model stacks
  the 30-min odds on top (conditional logit + residual LightGBM). Place probabilities come from stage-2/3 order models (Plackett–Luce
  with stage strengths); expected place payouts from a place-pool model fitted on 2019–2024.
- **Protocol.** Every choice was made on data up to 2024. The 2025-01 → 2026-05 hold-out was looked at once per rule. Then a **fresh window**, 6 months
  downloaded after every rule was frozen and written down (`results/*_preregistration.json`), run once.
- **Statistics.** Paired bootstrap over race days. Returns at official dividends; the effect of our own stake on the pool is included
  where stated (a €10 bet moves a place payout by about 0.0003 per euro: a median €22,800 is already on the horse).

## Model weights and prediction

`models/v3/` (34 MB) holds the v3 model fitted on every race up to 2026-11-05 (149,266 races; `manifest.json` lists the parts):
the LightGBM model, 8 network checkpoints (2 networks × 4 seeds), the blend and stack weights, the stage-2/3 place models and the
frozen place-pool parameters. The same training code with a 2026-05-05 cutoff reproduces the walk-forward fresh-window results
(winner log-loss 1.915 vs 1.914, correlation 0.999), so the shipped weights are the model behind the numbers above.

```bash
pip install -r requirements.txt            # the networks need a CUDA GPU
# 1. get the archive (see Data) and build the pipeline's tables: src/fr_extract.py, src/fr_features.py, src/fr_exp.py ratings
# 2. add a race day from PMU's API, rebuild features, predict:
python -I src/fr_fresh.py download 2026-11-20 2026-11-20 data/raw/fr_fresh
python -I src/fr_fresh.py build data/raw/fr_fresh .
python -I src/fr_features.py && python -I src/fr_exp.py ratings
python -I src/fr_v3_final.py predict 2026-11-20         # -> results/v3_predictions_2026-11-20_2026-11-20.csv
```

Columns: `p_fundamental` (no odds), `p_win` (with the 30-min odds when available), `p_place`, `est_place_payout`, `place_ev`, and `rule_B`.
The model needs the 30-min reference odds for `p_win`, `p_place` and rule B, so run it 15–30 minutes before the off.
This path was tested end to end on the real programme of 2026-11-07 (39 races, before any odds were out: `p_fundamental` only).

`predictions/walkforward_predictions.parquet` has the out-of-sample predictions of the fresh-window rerun for every race 2018 → 2026-11-05
(model outputs and race keys only: no odds and no results, join them from your own copy of the data).

## Reproduce

`run_all.sh` runs the whole pipeline from the archive (a few hours on one RTX 4090), `fresh/run_fresh.sh` the fresh-window test.
`fresh/README.md` has every fresh-window number. `results/` holds the reports (`REPORT.md`, `fr_report.json`, `place_report.json`, ...).

## Caveats

- Rule B's edge is small (≈ €0.50 per €10 bet) and needs about 3,000–4,000 more fresh bets to be told apart from zero.
- Several rules were tried on the hold-out; rule B was the best of them. The fresh window is the only clean replication.
- PMU changes things: the win pool's margin dropped by about one point in the week of 2025-09-29, and in mid-October 2026 the dividend labels
  changed to "Simple Gagnant" / "Simple Placé" (the code matches them case-insensitively).
- Bets are decided 30 minutes out and paid at the final dividend, which is what a real bettor gets; it was never tested with real money.

## How this was made

The research was carried out by an LLM agent (Claude, Anthropic) working in a terminal with a GPU. SolvAI set the question ("can a SOTA LLM
break new ground in ML on its own?"), the hold-out rule and the constraints, and asked for the follow-ups. Data search, pipeline,
models, pre-registrations and write-ups are the agent's. The project was called HorseCash while it ran, so logs, `docs/research_log.md` and some docstrings use that name.
The sibling project on UFC is [SolvAI/ufcash](https://github.com/SolvAI/ufcash).

## License and data

Code and model weights: MIT (see `LICENSE`). **No data is included.** Get it from its sources and respect their terms:
- [annaelmoussa/horse-racing-france](https://huggingface.co/datasets/annaelmoussa/horse-racing-france) (license "other"): the PMU archive.
- PMU's public JSON API (`src/fr_fresh.py`, `src/fr_epool.py`, `src/fr_live_probe.py`): programmes, odds and dividends.
- Canalturf (`src/fr_canalturf.py`): its terms of use (art. 7) claim rights over its database; the script is for personal research only.
- [SlyZouuu/turf-data](https://github.com/SlyZouuu/turf-data): minute-level odds capture used by `src/fr_nearoff.py`; it has no license.
- Hong Kong: [eprochasson/horserace_data](https://github.com/eprochasson/horserace_data) (`src/hk_*.py`).
