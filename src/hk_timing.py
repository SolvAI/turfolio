"""Hong Kong: how fast does the win market learn in the last minutes? (live odds snapshots, 2016-09 -> 2018)

For each race with live snapshots, the market probability (1/odds, normalised) of the winner is taken from the last snapshot at least
k minutes before the off (= the first snapshot after which the odds never change), and compared with the final odds. Tells how much of the late
information arrives in the last minutes, i.e. how close to the off a bettor must be to get near-final prices.
Run: python -I src/hk_timing.py -> results/hk_timing.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, RES = ROOT / "data" / "raw" / "hk", ROOT / "data" / "processed", ROOT / "results"
KS = [120, 60, 30, 20, 15, 10, 5, 3, 2, 1, 0.5, 0]


def main():
    lo = pd.read_csv(RAW / "live_odds.csv.gz", parse_dates=["capture_time"])
    lo["race_date"] = pd.to_datetime(lo.race_date)
    lo = lo[lo.race_country == "HK"]
    r = pd.read_parquet(PROC / "hk_runners.parquet", columns=["race_date", "race_no", "horse_no", "final_placing", "winning_odds"])
    r["race_date"] = pd.to_datetime(r.race_date)
    r = r[r.horse_no.notna()]
    win = r[r.final_placing == 1].groupby(["race_date", "race_no"]).filter(lambda g: len(g) == 1)
    wk = {(a, b): int(c) for a, b, c in zip(win.race_date, win.race_no, win.horse_no)}
    fin = {}
    for (a, b), g in r.groupby(["race_date", "race_no"]):
        q = 1 / g.winning_odds.values
        if np.isfinite(q).all() and (a, b) in wk:
            fin[(a, b)] = float(q[g.horse_no.values == wk[(a, b)]][0] / q.sum())
    rows = []
    for (a, b), g in lo.groupby(["race_date", "race_no"]):
        if (a, b) not in wk:
            continue
        g = g.sort_values("capture_time")
        # the off = first snapshot after which the win odds never change again (the scraper kept running after betting closed)
        wins = [json.dumps(json.loads(x).get("win", {}), sort_keys=True) for x in g.data.values]
        j = len(wins) - 1
        while j > 0 and wins[j - 1] == wins[-1]:
            j -= 1
        tl = g.capture_time.iloc[j]
        rec_gap = (g.capture_time.iloc[j] - g.capture_time.iloc[j - 1]).total_seconds() if j > 0 else np.nan
        w = str(wk[(a, b)])
        rec = {"race": f"{a.date()}_{b}", "n_snap": len(g), "final": fin.get((a, b)), "gap_s": rec_gap}
        for k in KS:
            h = g[g.capture_time <= tl - pd.Timedelta(minutes=k)]
            if h.empty:
                rec[k] = np.nan
                continue
            od = json.loads(h.data.iloc[-1]).get("win", {})
            try:
                q = {kk: 1 / float(v) for kk, v in od.items() if float(v) > 1}
            except ValueError:
                q = {}
            rec[k] = q[w] / sum(q.values()) if w in q and len(q) > 1 else np.nan
        rows.append(rec)
    t = pd.DataFrame(rows)
    ok = t[[k for k in KS] + ["final"]].notna().all(1) & (t.n_snap >= 50)
    t = t[ok]
    out = {"races": int(len(t)), "median_snapshots": int(t.n_snap.median()), "median_seconds_between_last_two_snapshots_before_freeze": float(t.gap_s.median()), "final_odds_logloss": round(float(-np.log(t.final).mean()), 4),
           "logloss_k_min_before_last_snapshot": {str(k): round(float(-np.log(t[k]).mean()), 4) for k in KS}}
    ll30, llf = out["logloss_k_min_before_last_snapshot"]["30"], out["final_odds_logloss"]
    out["share_of_30min_to_final_gain_still_missing"] = {str(k): round((out["logloss_k_min_before_last_snapshot"][str(k)] - llf) / (ll30 - llf), 3)
                                                         for k in KS if k <= 30}
    json.dump(out, open(RES / "hk_timing.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
