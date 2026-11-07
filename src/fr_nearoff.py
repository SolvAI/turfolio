"""Near-off test on a public minute-level capture (github.com/SlyZouuu/turf-data, donnees/cotes/*.jsonl, cloned to
data/raw/minute_capture_github; no license: private research use only). Main-pool ("OFFLINE") win odds polled each minute from about
15 min before to 3 min after the scheduled start, Oct-Nov 2026 (inside the fresh window, so the v3 predictions come from the fresh/ run).

The off = the last dateRapport of the main pool's final odds (PMU API, data/raw/fr_fresh/participants.jsonl).
For each lag L before the off: the last capture taken at or before off - L (kept only if it is at most 1 min older than that).
Question: how close does model + the market L minutes out get to the final market?  Winner log-loss on the same races for each L,
stack weights by 2-fold over days (odd/even day of month), paired day-bootstrap CI of (stack - final market).
Run: python -I src/fr_nearoff.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from racemodel import fit_clogit, group_softmax, starts_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "data" / "raw" / "minute_capture_github" / "donnees" / "cotes"
FRESH = ROOT / "data" / "raw" / "fr_fresh"
PRED = ROOT / "fresh" / "data" / "processed"
LAGS = [0, 0.5, 1, 2, 3, 5, 8, 12]


def captures():
    rows = []
    for f in sorted(CAP.glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("spec") != "OFFLINE":
                continue
            day = pd.Timestamp(r["depart"], unit="s", tz="UTC").tz_convert("Europe/Paris").strftime("%Y-%m-%d")
            R, C = r["course"][1:].split("C")
            for k, v in (r.get("cotes") or {}).items():
                if v:
                    rows.append((f"{day}_R{R}_C{C}", int(k), float(r["t"]), float(r["depart"]), float(v)))
    return pd.DataFrame(rows, columns=["raceKey", "numPmu", "t", "depart", "odds"])


def offs(keys):
    off, sched = {}, {}
    for line in open(FRESH / "participants.jsonl"):
        r = json.loads(line)
        if r["raceKey"] in keys and r.get("dernierRapportDirect"):
            off[r["raceKey"]] = max(off.get(r["raceKey"], 0), r["dernierRapportDirect"]["dateRapport"] / 1000)
    for line in open(FRESH / "courses.jsonl"):
        r = json.loads(line)
        if r["raceKey"] in keys:
            sched[r["raceKey"]] = r["raw"]["heureDepart"] / 1000
    return pd.DataFrame({"raceKey": list(off), "off": list(off.values())}).assign(sched=lambda x: x.raceKey.map(sched))


def main():
    cap = captures()
    keys = set(cap.raceKey)
    o = offs(keys)
    cap = cap.merge(o, on="raceKey")
    print(f"{cap.raceKey.nunique()} races captured; scheduled start agrees with the API: "
          f"{float((np.abs(cap.groupby('raceKey').depart.first() - cap.groupby('raceKey').sched.first()) < 61).mean()):.3f}")
    od = o.assign(d=(o.off - o.sched) / 60).d
    print(f"off - scheduled start (min): median {od.median():.1f}, 90% {od.quantile(.9):.1f}")
    last = cap.groupby("raceKey").t.max() - o.set_index("raceKey").off
    print(f"last capture - off (min): median {last.median() / 60:.1f}; share of races captured after the off {float((last > 0).mean()):.2f}")

    p = pd.read_parquet(PRED / "fr_preds.parquet").merge(
        pd.read_parquet(PRED / "fr_feats.parquet", columns=["rid", "numPmu", "raceKey", "n_winners"]), on=["rid", "numPmu"])
    p = p[p.raceKey.isin(keys) & (p.n_winners >= 1) & p.mkt_ref.notna() & p.mkt_final.notna()].copy()
    p["t"] = p.specialite.astype(str).str.startswith("TROT").astype(float)
    snaps = {}
    for L in LAGS:
        c = cap[cap.t <= cap.off - 60 * L]
        tl = c.groupby("raceKey").t.transform("max")
        c = c[(c.t == tl) & (c.off - 60 * L - c.t <= 60)]
        snaps[L] = c.set_index(["raceKey", "numPmu"]).odds
    # check: odds captured after the off against the official final odds
    a = cap[cap.t > cap.off + 30]
    a = a[a.t == a.groupby("raceKey").t.transform("max")].merge(p[["raceKey", "numPmu", "odf_rapport"]], on=["raceKey", "numPmu"])
    print(f"captures >30 s after the off vs final odds: {a.raceKey.nunique()} races, "
          f"median |log ratio| {np.median(np.abs(np.log(a.odds / a.odf_rapport))):.4f}")

    out = {"races_captured": int(len(keys))}
    rng = np.random.default_rng(0)
    for L in LAGS:
        q = p.merge(snaps[L].rename("late").reset_index(), on=["raceKey", "numPmu"], how="left")
        ok = q.late.notna().groupby(q.rid).transform("all")
        q = q[ok].sort_values(["date", "rid", "numPmu"]).reset_index(drop=True)
        if q.rid.nunique() < 50:
            continue
        st, sz = starts_of(q.rid.values)
        y = (q.y.values > 0) / np.maximum(1, np.add.reduceat(q.y.values > 0, st)[np.repeat(np.arange(len(st)), sz)])
        gsum = lambda v: np.repeat(np.add.reduceat(v, st), sz)
        late = (1 / q.late.values) / gsum(1 / q.late.values)
        lmr, lmf, ll, le = np.log(q.mkt_ref.values), np.log(q.mkt_final.values), np.log(late), np.log(q.end_ref.values)
        half = (q.date.dt.day.values % 2 == 0)
        day = q.date.values[st]
        res = {}
        for name, xs in (("final_market", [lmf]), ("market_at_lag", [ll]), ("end_ref_model", [le]), ("model+lag", [le, ll]),
                         ("model+ref+lag", [le, lmr, ll]), ("final+model", [lmf, le])):
            Z = np.column_stack(xs)
            s = np.zeros(len(q))
            for f in (half, ~half):
                w = fit_clogit(Z[f], y[f], q.rid.values[f])
                s[~f] = Z[~f] @ w
            pr = group_softmax(s, st, sz)
            res[name] = -np.add.reduceat(y * np.log(np.clip(pr, 1e-12, 1)), st)
        u, inv = np.unique(day, return_inverse=True)
        idx = rng.integers(0, len(u), (2000, len(u)))
        cnt = np.bincount(inv)

        def ci(a, b):
            dl = res[a] - res[b]
            bs = np.bincount(inv, dl)[idx].sum(1) / cnt[idx].sum(1)
            return [round(float(dl.mean()), 4), [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]]

        out[f"lag_{L}min"] = {"races": int(len(st)), "days": int(len(u)),
                              "logloss": {k: round(float(v.mean()), 4) for k, v in res.items()},
                              "model+lag_minus_final": ci("model+lag", "final_market"),
                              "model+lag_minus_lag": ci("model+lag", "market_at_lag"),
                              "lag_minus_final": ci("market_at_lag", "final_market"),
                              "median_abs_log_odds_vs_final": round(float(np.median(np.abs(np.log(q.late / q.odf_rapport)))), 4)}
        print(f"L={L}", json.dumps(out[f"lag_{L}min"]), flush=True)
    json.dump(out, open(ROOT / "results" / "nearoff_minute_capture.json", "w"), indent=1)


if __name__ == "__main__":
    main()
