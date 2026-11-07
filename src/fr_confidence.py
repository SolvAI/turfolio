"""Confidence rule: in each race, back the model's top pick (highest end_ref = v3 end model, decided 30 min before the off) only when its
win probability is at least c; otherwise skip the race. Win bet in the main PMU pool, paid at the official dividend (rapports-definitifs).
  dev   ROI by threshold on 2018-2024 (original run); choose c = best ROI with >= 2000 bets; write results/confidence_preregistration.json
  test  the pre-registered c, once, on the hold-out 2025-01..2026-05-05 (original run) and the fresh window 2026-05-06..11-05 (fresh/ run),
        with a paper bank: EUR 1000 start, flat EUR 10 per bet
Run: python -I src/fr_confidence.py dev | test
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
GRID = [round(x, 2) for x in np.arange(0.20, 0.701, 0.05)]
PRE = RES / "confidence_preregistration.json"


def top_picks(root):
    d = pd.read_parquet(root / "data/processed/fr_preds.parquet")
    d = d.merge(pd.read_parquet(root / "data/processed/fr_feats.parquet", columns=["rid", "numPmu", "raceKey"]), on=["rid", "numPmu"])
    d = d[d.odf_rapport > 1]  # runners that ran
    d["mkt_rank"] = d.groupby("rid").mkt_ref.rank(ascending=False, method="first")
    t = d.loc[d.groupby("rid").end_ref.idxmax()].copy()
    rp = pd.concat([pd.read_parquet(q, columns=["raceKey", "typePari", "libelle", "combinaison", "dividendePourUnEuro"])
                    for q in sorted((root / "data/raw/fr").glob("rapports-*.parquet"))], ignore_index=True)
    sg = rp[(rp.typePari == "SIMPLE_GAGNANT") & (rp.libelle.astype(str).str.lower() == "simple gagnant")].copy()  # PMU wrote "Simple Gagnant" from mid-Oct 2026
    sg["numPmu"] = pd.to_numeric(sg.combinaison, errors="coerce")
    sg["div"] = sg.dividendePourUnEuro / 100
    t = t[t.raceKey.isin(set(sg.raceKey))]  # races with official dividends
    t = t.merge(sg[["raceKey", "numPmu", "div"]].drop_duplicates(["raceKey", "numPmu"]), on=["raceKey", "numPmu"], how="left")
    t["ret"] = t["div"].fillna(0.0)  # official payout per EUR 1 (0 if the pick lost)
    t["is_mkt_fav"] = t.mkt_rank == 1
    return t.sort_values(["date", "rid"]).reset_index(drop=True)


def curve(t, m):
    return {f"{c:.2f}": {**roi_summary(t.ret.values[m & (t.end_ref.values >= c)]),
                         "share_of_races": round(float((m & (t.end_ref.values >= c)).sum() / max(1, m.sum())), 3)} for c in GRID}


def bank(t, stake=10.0, start=1000.0):
    pnl = stake * (t.ret.values - 1)
    b = start + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[start, b])[1:]
    mon = pd.Series(pnl, index=t.date.dt.to_period("M")).groupby(level=0).sum()
    return {"start": start, "stake": stake, "bets": int(len(t)), "end": round(float(b[-1]), 2) if len(b) else start,
            "profit": round(float(pnl.sum()), 2), "min_bank": round(float(b.min()), 2) if len(b) else start,
            "max_drawdown": round(float((peak - b).max()), 2) if len(b) else 0.0,
            "months_positive": f"{int((mon > 0).sum())}/{len(mon)}", "by_month": {str(k): round(float(v), 2) for k, v in mon.items()}}


def main(mode):
    t = top_picks(ROOT)
    yr = t.date.dt.year.values
    dev = (yr >= 2018) & (yr <= 2024)
    if mode == "dev":
        cv = curve(t, dev)
        for c, r in cv.items():
            print(f"p>={c}: {r['bets']:6d} bets ({r['share_of_races']:.1%} of races), hit {r.get('hit_rate', 0):.3f}, "
                  f"avg payout {r.get('avg_payout_when_hit')}, ROI {r.get('roi', 0):+.4f}", flush=True)
        ok = {c: r["roi"] for c, r in cv.items() if r["bets"] >= 2000}
        best = max(ok, key=ok.get)
        by_year = {int(Y): roi_summary(t.ret.values[(yr == Y) & (t.end_ref.values >= float(best))]) for Y in range(2018, 2025)}
        json.dump({"dev_2018_2024": cv, "chosen": best, "chosen_by_year": by_year}, open(RES / "confidence_dev.json", "w"), indent=1)
        json.dump({"written": datetime.now().isoformat(), "choice_made_on": "2018-2024 only (results/confidence_dev.json)",
                   "rule": f"each race: back the top pick of end_ref (v3 end model, 30 min out) if its probability >= {best}; else skip. "
                           "Main-pool win bet at the official dividend; flat EUR 10, paper bank EUR 1000",
                   "threshold": float(best), "selection": "best dev ROI among thresholds with >= 2000 dev bets",
                   "headline": "ROI with 95% day-bootstrap CI on the hold-out 2025-01..2026-05-05 and, separately, the fresh window 2026-05-06..11-05"},
                  open(PRE, "w"), indent=1)
        print(f"chosen p>={best}; by year:", json.dumps({y: (r["bets"], r["roi"]) for y, r in by_year.items()}))
        return
    c = json.load(open(PRE))["threshold"]
    f = top_picks(ROOT / "fresh")
    windows = {"holdout_2025_01..2026_05_05": (t, (t.date >= "2025-01-01").values),
               "fresh_2026_05_06..11_05": (f, (f.date >= "2026-05-06").values)}
    rep = {"threshold": c}
    for nm, (x, w) in windows.items():
        k = w & (x.end_ref.values >= c)
        rep[nm] = {"rule": roi_summary(x.ret.values[k], x.date.values[k]), "share_of_races_bet": round(float(k.sum() / w.sum()), 3),
                   "picks_that_are_the_30min_favourite": round(float(x.is_mkt_fav.values[k].mean()), 3),
                   "paper_bank": bank(x[k]),
                   "every_race_top_pick": roi_summary(x.ret.values[w], x.date.values[w]),
                   "descriptive_curve_not_for_choice": curve(x, w)}
        r = rep[nm]["rule"]
        print(f"{nm}: {r['bets']} bets ({rep[nm]['share_of_races_bet']:.1%} of races), hit {r['hit_rate']:.3f}, ROI {r['roi']:+.4f} "
              f"{r.get('roi_ci95')}; paper bank {rep[nm]['paper_bank']['end']} (min {rep[nm]['paper_bank']['min_bank']})", flush=True)
    json.dump(rep, open(RES / "confidence_test.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
