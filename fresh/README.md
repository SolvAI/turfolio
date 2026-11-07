# Fresh hold-out run (2026-05-06 -> 2026-11-05)

A copy of the frozen v3 pipeline (src/ copied 2026-11-06, tuned rounds/epochs copied from ../results) run on the archive plus the races
after its end, downloaded from PMU's API by ../src/fr_fresh.py and rebuilt in the archive's format (validated on 2026-04-22..05-05:
every runner field the pipeline reads agrees with the archive). No choice of any kind is made on these months.
data/raw/fr and data/raw/pmu_internet link to the main project's raw data; data/raw/fr/rapports-fresh.parquet holds the fresh dividends.

## Results (run 2026-11-06; nothing below was used for any choice)

Accuracy, 5,488 races 2026-05-06..11-05 (results/fresh_eval_2026-05-06_2026-12-31.json), winner log-loss, 95% day-bootstrap CI:
- fundamental model (no odds) vs 30-min market: 1.942 vs 1.984, -0.042 [-0.056, -0.029]
- end model at 30 min vs 30-min market: 1.914, -0.071 [-0.079, -0.062]; top pick 31.8% vs 30.3%
- end model with final odds vs final market: 1.868 vs 1.884, -0.015 [-0.020, -0.011]
- late-move head (nn_L20_mus_mt head 2) added to 30-min market + v3: -0.0041 [-0.0061, -0.0021] (hold-out in this run: -0.0036)

Betting, same window, official main-pool dividends (re-run 2026-11-07 after the label fix below; first figures in brackets):
- v3 value rule (this run's dev choice ev>0.2): -5.9% on 738 bets [-34%, +28%] (-7.0% on 710); at the original run's ev>0.15: -8.2% on 1,019 bets
- pre-registered late-head rule (results/value_late_test.json): -24.5% on 647 bets [-49%, +8%] (-26.5% on 628)
- internet pool for the value rule (results/pools_test_*.json): -3.5% vs main -9.4% on 1,005 bets (-4.7% vs -10.5% on 950)
- confidence rule, top pick if end_ref >= 0.60 (../results/confidence_test.json): -13.1% on 199 bets [-23%, -2%] (-8.5% on 174)

Label change: from mid-October 2026 PMU writes "Simple Gagnant" / "Simple Placé" (capitalised; one place label for every field size).
The first fresh runs matched the old lower-case labels only and so dropped the official dividends of 2026-10-15..11-05.
Matching is now case-insensitive (fr_value, fr_value_late, fr_pools, fr_confidence, fr_place); the archive has only the old labels,
so dev and hold-out results are unchanged (place_report.json identical byte for byte).

Reproducibility: rerunning the pipeline changes end_ref by 0.06 mean |log ratio| (seeds, threads). On the 2025-01..2026-05 hold-out
the v3 value rule at ev>0.15 gives -1.7% here against +5.7% in the original run: the betting results are within rerun noise.

Drift seen in this run: winners shorten more between the 30-min and final odds each year (mean log -0.076 in 2023, -0.127 in 2026Q3),
so the payout model fitted on 2018-24 overstates winners' final odds (-4.1% residual in the fresh window) and the rule bets 3x more often.
The main pool's overround stepped from 1.200 to 1.185 in the week of 2025-09-29.

Near-off test (../src/fr_nearoff.py, ../results/nearoff_minute_capture.json; public minute capture, 183-298 races Oct-Nov 2026):
the last main-pool price at most 1 min before the off is still +0.024 [0.008, 0.044] worse than the final market, model + that price +0.037.

Place bets (simple placé, main pool, official dividends; ../src/fr_place_conf.py, pre-registered in ../results/place_preregistration.json):
- rule B, place value rule (P(place) >= 0.6 and EV > 0.1, chosen on 2019-24 by fr_place.py, re-chosen identically in this run):
  fresh window +6.1% on 320 bets [-2.0%, +14.3%]; with our own EUR 10 in the pool +5.9% [-2.1%, +13.5%]; paper bank EUR 1000 -> 1189.
  Reproducible across reruns (2025-01..2026-05 hold-out: +4.28% here vs +4.30% in the original run).
  Both out-of-sample windows pooled (2025-01..2026-11-05, 1,069 bets): +4.8% [+0.7%, +9.5%]; with own EUR 10 +4.6% [+0.4%, +8.8%];
  with own EUR 50 +4.0% [+0.1%, +8.1%].
- rule A, confidence only (top P(place) >= 0.80): fresh -3.5% on 373 bets [-7.8%, +0.9%]; paper bank -> 871.
- Own-stake fix (2026-11-07): the first version re-derived each payout from the equal-split formula and rounded it down again, which cut
  0.10 off about 30% of winning bets even with no extra stake (fresh: +6.1% -> +4.1% at X = 0). Our EUR 10 actually moves the unrounded
  payout by a median 0.0003 (median EUR 22,800 already on the horse). fr_place.py's small-pool own-stake figures had the same flaw, now fixed.

## Re-running and the final-model check

`fresh/setup_fresh.sh` downloads the fresh races and builds this folder (code copy, links to the raw archive); `fresh/run_fresh.sh` then
runs everything above in order. The last steps check the shipped training code (`src/fr_v3_final.py`): trained on races up to 2026-05-05
and scored on this window, it gives a winner log-loss of 1.915 (`p_win`) and 1.943 (`p_fundamental`), against 1.914 and 1.942 for the
walk-forward models, with a correlation of 0.999. Its rule-B bets (275) overlap 249 of the walk-forward 320; they are a code check, not a
separate result. The shipped model (`models/v3`) is the same code trained on every race up to 2026-11-05.
