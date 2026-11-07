"""Quality check of the Canalturf odds series against our PMU data, on a chosen date range (default: the 2024 dev part; never the hold-out).

Joins data/raw/canalturf/odds.jsonl to our runners by (raceKey, numPmu) and reports:
  coverage, timing of the last point vs the scheduled start, agreement with our 30-min reference odds at the same time,
  agreement of the last point with the final odds, how much of the ref -> final move is already in the last point,
  duplicated series inside a race (copy errors), and winner log-loss of the market at ref / each series point / final.
Run: python -I src/fr_late_check.py [from YYYY-MM-DD] [to YYYY-MM-DD]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC, RAW = ROOT / "data" / "processed", ROOT / "data" / "raw" / "canalturf"


def load_series(lo, hi):
    rows, races = [], []
    for l in open(RAW / "odds.jsonl"):
        r = json.loads(l)
        if not (lo <= r["date"] <= hi):
            continue
        ser = [tuple(map(tuple, h["series"])) if h["series"] else None for h in r["horses"]]
        races.append(dict(raceKey=r["raceKey"], snap=r["snap"], n=len(r["horses"]),
                          dup=len([s for s in ser if s]) - len({s for s in ser if s})))
        for h in r["horses"]:
            for k, (t, v) in enumerate(h["series"] or []):
                rows.append((r["raceKey"], h["num"], h["name"], k, t, v))
    s = pd.DataFrame(rows, columns=["raceKey", "numPmu", "name", "k", "hhmm", "v"])
    return s, pd.DataFrame(races)


def main(lo="2024-01-01", hi="2024-12-31"):
    assert hi < "2025-01-01", "quality checks stay on dev data"
    s, races = load_series(lo, hi)
    u = pd.read_parquet(PROC / "fr_runners.parquet", columns=["raceKey", "numPmu", "nom", "dernierRapportReference.rapport",
                                                             "dernierRapportReference.dateRapport", "dernierRapportDirect.rapport"])
    u = u[u.raceKey.isin(races.raceKey)].rename(columns={"dernierRapportReference.rapport": "ref", "dernierRapportDirect.rapport": "fin",
                                                         "dernierRapportReference.dateRapport": "ref_t"})
    rc = pd.read_parquet(PROC / "fr_races.parquet", columns=["raceKey", "heureDepart", "date"])
    rc = rc[rc.date.astype(str).str[:10].between(lo, hi)]
    f = pd.read_parquet(PROC / "fr_feats.parquet", columns=["date", "numReunion", "numCourse", "numPmu", "won", "n_winners"])
    f = f[f.date.astype(str).between(lo, hi + "z")]
    f["raceKey"] = f.date.dt.strftime("%Y-%m-%d") + "_R" + f.numReunion.astype(str) + "_C" + f.numCourse.astype(str)
    u = u.merge(f[["raceKey", "numPmu", "won", "n_winners"]], on=["raceKey", "numPmu"], how="left")
    off = pd.to_datetime(rc.set_index("raceKey").heureDepart, utc=True).dt.tz_convert("Europe/Paris")
    rep = {}
    ours = rc.raceKey.nunique()
    rep["coverage"] = dict(our_races=int(ours), with_page=int(races.raceKey.nunique()),
                           with_series=int(s.raceKey.nunique()), share_with_series=round(s.raceKey.nunique() / ours, 3))
    # names agree?
    nm = s[s.k == 0].merge(u[["raceKey", "numPmu", "nom"]], on=["raceKey", "numPmu"], how="left")
    rep["runner_name_match"] = round(float((nm.name.str.upper().str.strip() == nm.nom.str.upper().str.strip()).mean()), 4)
    # time of points relative to the scheduled start
    s["t"] = pd.to_datetime(s.raceKey.str[:10] + " " + s.hhmm).dt.tz_localize("Europe/Paris", ambiguous="NaT", nonexistent="NaT")
    s["mins_to_off"] = (off.reindex(s.raceKey).values - s.t.values) / np.timedelta64(1, "m")
    last = s.sort_values("k").groupby(["raceKey", "numPmu"]).tail(1)
    rep["last_point_minutes_before_scheduled_start"] = {q: float(last.mins_to_off.quantile(q)) for q in (.05, .25, .5, .75, .95)}
    rep["points_per_horse"] = {int(k): int(v) for k, v in s.groupby(["raceKey", "numPmu"]).size().value_counts().head(5).items()}
    rep["missing_values_share"] = round(float(s.v.isna().mean()), 4)
    rep["races_with_duplicated_series"] = round(float((races.dup > 0).mean()), 4)
    # agreement with our reference odds at the reference time
    u["ref_t"] = pd.to_datetime(u.ref_t, unit="ms", utc=True).dt.tz_convert("Europe/Paris")
    m = s.merge(u, on=["raceKey", "numPmu"])
    m = m[m.v.notna() & m.ref.notna() & m.fin.notna() & (m.v > 1)]
    m["dt_ref"] = (m.t - m.ref_t).dt.total_seconds().abs() / 60
    at_ref = m[m.dt_ref <= 5]
    lr = np.log(at_ref.v) - np.log(at_ref.ref)
    rep["at_ref_time"] = dict(n=int(len(at_ref)), median_abs_log_diff=round(float(lr.abs().median()), 4),
                              share_within_10pct=round(float((lr.abs() < np.log(1.1)).mean()), 3),
                              corr_log=round(float(np.corrcoef(np.log(at_ref.v), np.log(at_ref.ref))[0, 1]), 4))
    lm = m.merge(last[["raceKey", "numPmu", "k"]], on=["raceKey", "numPmu", "k"])
    d_fin, d_last = np.log(lm.fin) - np.log(lm.ref), np.log(lm.v) - np.log(lm.ref)
    rep["last_point_vs_final"] = dict(n=int(len(lm)), median_abs_log_diff_last_final=round(float((np.log(lm.v) - np.log(lm.fin)).abs().median()), 4),
                                      median_abs_log_diff_ref_final=round(float(d_fin.abs().median()), 4),
                                      share_of_ref_to_final_move_in_last_point=round(float((d_last * d_fin).sum() / (d_fin ** 2).sum()), 3))
    # winner log-loss of the market at each stage, on races where every runner has the value
    def ll(df, col):
        df = df[df.n_winners >= 1]
        g = df.groupby("raceKey")
        ok = g[col].transform(lambda x: x.notna().all() and (x > 1).all())
        df = df[ok.astype(bool)].copy()
        df["q"] = 1 / df[col]
        df["p"] = df.q / df.groupby("raceKey").q.transform("sum")
        w = df[df.won == 1]
        return float(-np.log(w.p).mean()), int(w.raceKey.nunique()), set(w.raceKey)
    base = u[u.won.notna()].copy()
    ll_ref, n_ref, R_ref = ll(base, "ref")
    ll_fin, n_fin, R_fin = ll(base, "fin")
    stage = {}
    for k in sorted(s.k.unique()):
        sk = s[s.k == k][["raceKey", "numPmu", "v", "mins_to_off"]]
        bk = base.merge(sk, on=["raceKey", "numPmu"], how="left")
        val, n, Rk = ll(bk, "v")
        common = R_ref & R_fin & Rk
        a = ll(bk[bk.raceKey.isin(common)], "v")[0]
        rf = ll(base[base.raceKey.isin(common)], "ref")[0]
        fi = ll(base[base.raceKey.isin(common)], "fin")[0]
        stage[int(k)] = dict(median_mins_to_off=float(sk.mins_to_off.median()), races=len(common), canalturf=round(a, 4), ref=round(rf, 4), final=round(fi, 4))
    rep["winner_logloss_by_point"] = stage
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(ROOT / "results" / "canalturf_check.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
