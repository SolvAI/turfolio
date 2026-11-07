"""Late internet-pool prices from the Canalturf archive, with real capture times (dev only).

The live check of 2026-11-05 showed: Canalturf samples PMU's internet pool on the quarter-hour, its page stops updating ~20-30 min
after the off, and the snapshot time 'Cote a HHhMM' (snap) is the real time of the newest sample. So point k (0..11) of an archived
series was captured at about snap - 15 min x (11 - k). A point is usable for a bet only if captured before the off; the off is taken
as the stamp of the main pool's final odds (dernierRapportDirect.dateRapport), minus a 1-min margin.
  1) validation (where the internet-pool final odds are known): post-off points must equal the e-pool final odds, pre-off ones not;
  2) value: winner log-loss of the model, the 30-min market, and conditional-logit blends of the model with the last pre-off
     internet price, by how many minutes before the off that price was captured (2-fold by date inside the range).
Run: python -I src/fr_late_ct.py [from YYYY-MM-DD] [to YYYY-MM-DD]
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_pools import load_epool  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RAW, RES = ROOT / "data" / "processed", ROOT / "data" / "raw", ROOT / "results"
Z = ZoneInfo("Europe/Paris")


def ct_points(lo, hi):
    rows = []
    for l in open(RAW / "canalturf" / "odds.jsonl"):
        r = json.loads(l)
        if not (lo <= r["date"] <= hi) or not r.get("snap") or not r.get("horses"):
            continue
        try:
            snap = datetime.fromisoformat(f"{r['date']}T{r['snap'].replace('h', ':')}").replace(tzinfo=Z)
        except ValueError:
            continue
        for h in r["horses"]:
            s = h.get("series") or []
            for k, (_, v) in enumerate(s):
                if v:
                    rows.append((r["raceKey"], h["num"], k, len(s), snap - timedelta(minutes=15 * (len(s) - 1 - k)), v))
    return pd.DataFrame(rows, columns=["raceKey", "numPmu", "k", "npts", "t", "v"])


def main(lo="2022-06-01", hi="2024-12-31"):
    assert hi < "2025-01-01", "dev only"
    s = ct_points(lo, hi)
    u = pd.read_parquet(PROC / "fr_runners.parquet", columns=["raceKey", "numPmu", "dernierRapportDirect.dateRapport"])
    off = (pd.to_datetime(u.groupby("raceKey")["dernierRapportDirect.dateRapport"].max(), unit="ms", utc=True)
           .dt.tz_convert(Z).rename("off"))
    s = s.merge(off, left_on="raceKey", right_index=True)
    s["t"] = pd.to_datetime(s.t.astype(str), utc=True).dt.tz_convert(Z)
    s["mins_before_off"] = (s.off - s.t).dt.total_seconds() / 60
    rep = {"range": [lo, hi], "races_with_points": int(s.raceKey.nunique())}
    # 1) validation against the internet pool's final odds
    e, _ = load_epool()
    m = s.merge(e[["raceKey", "numPmu", "e_fin"]], on=["raceKey", "numPmu"])
    m = m[m.e_fin > 1]
    m["mb"] = pd.cut(m.mins_before_off, [-60, -15, -5, -1, 0, 1, 5, 10, 15, 20, 30, 60, 1e4])
    m["same"] = (np.log(m.v / m.e_fin).abs() < 0.02)
    rep["validation_share_equal_to_epool_final_by_minutes_before_off"] = {
        str(k): dict(n=int(len(x)), share_equal=round(float(x.same.mean()), 3), median_abs_log=round(float(np.log(x.v / x.e_fin).abs().median()), 3))
        for k, x in m.groupby("mb", observed=True)}
    # 2) value of the last pre-off point
    pre = s[s.mins_before_off >= 1].sort_values("k").groupby(["raceKey", "numPmu"]).tail(1)
    d = pd.read_parquet(PROC / "fr_preds.parquet")
    d = d.merge(pd.read_parquet(PROC / "fr_feats.parquet", columns=["rid", "numPmu", "raceKey"]), on=["rid", "numPmu"])
    d = d[d.date.astype(str).str[:10].between(lo, hi) & (d.odr_rapport > 1) & (d.odf_rapport > 1)]
    d = d.merge(pre[["raceKey", "numPmu", "v", "mins_before_off"]], on=["raceKey", "numPmu"], how="left")
    full = d.v.gt(1).groupby(d.rid).transform("all") & (d.y > 0).groupby(d.rid).transform("any")
    d = d[full].copy()
    d["lag"] = d.groupby("rid").mins_before_off.transform("median")
    rid = np.unique(d.rid.values, return_inverse=True)[1]
    won = (d.y > 0).values
    nrm = lambda q: q / np.bincount(rid, q)[rid]
    F = {"model": np.log(nrm(d.end_ref.values)), "ref": np.log(nrm(1 / d.odr_rapport.values)), "late_e": np.log(nrm(1 / d.v.values)),
         "final": np.log(nrm(1 / d.odf_rapport.values))}

    def nll(w, cols, msk):
        sc = sum(wi * F[c] for wi, c in zip(w, cols))
        ex = np.exp(sc - sc.max())
        den = np.bincount(rid, ex)[rid]
        k = won & msk
        return float(-(sc[k] - sc.max() - np.log(den[k])).mean())

    a = d.date.values.astype("int64") <= np.median(d.date.values.astype("int64"))
    out = {}
    lagb = pd.cut(d.lag, [1, 4, 8, 12, 16, 60]).astype(str).values
    for cols in (["ref"], ["late_e"], ["final"], ["model"], ["model", "late_e"], ["model", "ref", "late_e"]):
        pred = np.zeros(len(d))
        for tr in (a, ~a):
            w = minimize(lambda w: nll(w, cols, tr), np.ones(len(cols)) / len(cols), method="Nelder-Mead").x
            sc = sum(wi * F[c] for wi, c in zip(w, cols))
            ex = np.exp(sc - sc.max())
            pred[~tr] = (ex / np.bincount(rid, ex)[rid])[~tr]
        lw = -np.log(pred[won])
        out["+".join(cols)] = dict(all=round(float(lw.mean()), 4),
                                   by_minutes_before_off={b: round(float(lw[lagb[won] == b].mean()), 4) for b in sorted(set(lagb[won]))})
    rep["races_used"] = int(won.sum())
    rep["races_by_lag"] = {b: int((lagb[won] == b).sum()) for b in sorted(set(lagb[won]))}
    rep["winner_logloss_2fold"] = out
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(RES / f"late_ct_{lo}_{hi}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
