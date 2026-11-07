"""Place bets (simple placé, main pool) at the official dividend, from the per-horse rows saved by fr_place.py (data/processed/fr_place_rows.parquet;
P = P(place) of the end_stage_ref order model, decided 30 min out; EV = P x corrected expected payout - 1).
  rule A  confidence only: in each race back the horse with the highest P if P >= c (c chosen on dev 2019-2024: best ROI, >= 2000 bets)
  rule B  the place value rule of fr_place.py as chosen there on dev: P >= 0.6 and EV > 0.1
  dev     ROI of rule A by c on 2019-2024 -> results/place_conf_dev.json + results/place_preregistration.json
  test    both rules once on the fresh window 2026-05-06..11-05 (fresh/ run; the headline), and on 2025-01..2026-05-05 for reference
          (already seen: results/place_report.json showed test ROI by model P(place) and rule B's test result).
          Paper bank EUR 1000, flat EUR 10; also with our own EUR 10 in the pool:
          official payout - [net / (m S) - (net + (R - 1) X) / (m (S + X))], >= the minimum payout (fr_place.py), R = 0.85
Run: python -I src/fr_place_conf.py dev | test
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_exotics import roi_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PRE = RES / "place_preregistration.json"
GRID = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]


def rows(root):
    r = pd.read_parquet(root / "data/processed/fr_place_rows.parquet").sort_values(["date", "rid", "numPmu"]).reset_index(drop=True)
    r["is_top"] = r.index.isin(r.groupby("rid").P.idxmax())
    return r


def with_stake(r, X=10.0):
    """Official payout minus the drop our own stake X causes in the unrounded payout (expected effect of the 0.10 rounding: the same drop).
    Earlier version re-derived the payout from the equal-split formula and rounded it down again, which cut 0.10 off about 30% of
    winning bets even with X = 0 (fixed 2026-11-07)."""
    S = np.where(r.ret.values > 0, r.stake_eur.fillna(0).values, 1.0)
    m, net = r.places.values, r.net_pool.values
    drop = net / (m * S) - (net + (0.85 - 1) * X) / (m * (S + X))
    return np.where(r.ret.values > 0, np.maximum(r.ret.values - drop, r.minpay.values), 0.0)


def bank(r, ret, stake=10.0, start=1000.0):
    pnl = stake * (ret - 1)
    b = start + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[start, b])[1:]
    mon = pd.Series(pnl, index=pd.to_datetime(r.date.values).to_period("M")).groupby(level=0).sum()
    return {"bets": int(len(ret)), "end": round(float(b[-1]), 2) if len(b) else start, "min_bank": round(float(b.min()), 2) if len(b) else start,
            "max_drawdown": round(float((peak - b).max()), 2) if len(b) else 0.0, "months_positive": f"{int((mon > 0).sum())}/{len(mon)}",
            "by_month": {str(k): round(float(v), 2) for k, v in mon.items()}}


def main(mode):
    r = rows(ROOT)
    yr = r.date.dt.year.values
    dev = (yr >= 2019) & (yr <= 2024)
    if mode == "dev":
        cv = {}
        for c in GRID:
            k = dev & r.is_top.values & (r.P.values >= c)
            cv[f"{c:.2f}"] = {**roi_summary(r.ret.values[k]), "share_of_races": round(float(k.sum() / (dev & r.is_top.values).sum()), 3),
                              "mean_P": round(float(r.P.values[k].mean()), 3) if k.any() else None}
            x = cv[f"{c:.2f}"]
            print(f"P(place)>={c:.2f}: {x['bets']:6d} bets ({x['share_of_races']:.1%} of races), mean P {x['mean_P']}, placed {x.get('hit_rate', 0):.3f}, "
                  f"avg payout {x.get('avg_payout_when_hit')}, ROI {x.get('roi', 0):+.4f}", flush=True)
        ok = {c: x["roi"] for c, x in cv.items() if x["bets"] >= 2000}
        best = max(ok, key=ok.get)
        kB = dev & (r.P.values >= 0.6) & (r.EV.values > 0.1)
        json.dump({"rule_A_dev_2019_2024": cv, "chosen_c": best, "rule_B_dev": roi_summary(r.ret.values[kB])}, open(RES / "place_conf_dev.json", "w"), indent=1)
        json.dump({"written": datetime.now().isoformat(), "choice_made_on": "2019-2024 only (results/place_conf_dev.json; rule B from fr_place.py's dev choice)",
                   "rule_A": f"each race: back the horse with the highest P(place) (end_stage_ref, 30 min out) if P >= {best}; else skip",
                   "threshold_A": float(best),
                   "rule_B": "P(place) >= 0.6 and EV > 0.1 (fr_place.py value rule as chosen on dev; payout model refit by the same code in the fresh/ run)",
                   "bet": "main-pool simple placé at the official dividend, flat EUR 10, paper bank EUR 1000; also with our own EUR 10 in the pool",
                   "headline": "ROI with 95% day-bootstrap CI on the fresh window 2026-05-06..2026-11-05 (fresh/ run), rules A and B; "
                               "2025-01..2026-05-05 reported for reference only (already seen)"}, open(PRE, "w"), indent=1)
        print(f"chosen c = {best}; rule B on dev: {roi_summary(r.ret.values[kB])}")
        return
    pre = json.load(open(PRE))
    c = pre["threshold_A"]
    f = rows(ROOT / "fresh")
    rep = {"preregistration": pre}
    for nm, (x, w) in {"fresh_2026_05_06..11_05 (headline)": (f, (f.date >= "2026-05-06").values),
                       "2025_01..2026_05_05 (seen before, original run)": (r, (r.date >= "2025-01-01").values),
                       "2025_01..2026_05_05 (fresh rerun)": (f, ((f.date >= "2025-01-01") & (f.date < "2026-05-06")).values)}.items():
        rs = with_stake(x)
        out = {"races": int((w & x.is_top.values).sum())}
        for rule, k in (("A_confidence", w & x.is_top.values & (x.P.values >= c)), ("B_value", w & (x.P.values >= 0.6) & (x.EV.values > 0.1))):
            day = x.date.values[k]
            out[rule] = {"no_stake_effect": roi_summary(x.ret.values[k], day), "own_10eur_in_pool": roi_summary(rs[k], day),
                         "paper_bank_own_stake": bank(x[k], rs[k])}
            a, b = out[rule]["no_stake_effect"], out[rule]["own_10eur_in_pool"]
            print(f"{nm} {rule}: {a['bets']} bets, placed {a.get('hit_rate')}, avg payout {a.get('avg_payout_when_hit')}, ROI {a.get('roi')} "
                  f"{a.get('roi_ci95')} | with own stake {b.get('roi')} {b.get('roi_ci95')} | bank {out[rule]['paper_bank_own_stake']['end']}", flush=True)
        out["A_curve_descriptive"] = {f"{cc:.2f}": roi_summary(x.ret.values[w & x.is_top.values & (x.P.values >= cc)]) for cc in GRID}
        rep[nm] = out
    json.dump(rep, open(RES / "place_conf_test.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
