"""Main win pool (SIMPLE_GAGNANT) vs PMU's separate internet pool (E_SIMPLE_GAGNANT): is there value in choosing the pool?

A PMU.fr customer can back a horse in either pool; both pay pari-mutuel dividends, and their odds differ (live check 2026-11-05:
final odds of the two pools are a median 13% apart). Data: our main-pool odds (30-min reference + final) and official dividends,
and data/raw/pmu_internet/epool.jsonl (internet-pool reference + final odds, official e-dividends, from src/fr_epool.py).
Reports, on a chosen date range (dev only unless --test is given AFTER every choice is frozen):
  pool sizes and overrounds, the gap log(e odds / main odds) by odds level, which pool is sharper (winner log-loss),
  whether the gap at 30 min predicts the gap at the off, and the official-dividend return of pool-choice strategies,
  over all runners and over the value-rule bets of src/fr_value.py (p>=0, EV>0.15 with the winners payout model), with our own stake.
Run: python -I src/fr_pools.py [from YYYY-MM-DD] [to YYYY-MM-DD] [--test]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_exotics import roi_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, RAW = ROOT / "data" / "processed", ROOT / "results", ROOT / "data" / "raw"
DEV = (2018, 2024)


def load_epool():
    rows, divs, recs = [], [], {}
    for name in ("epool.jsonl", "epool_fix.jsonl"):  # a refetched record (fr_epool.py --refetch) overrides a failed one
        q = RAW / "pmu_internet" / name
        for l in (open(q) if q.exists() else []):
            r = json.loads(l)
            if r["raceKey"] not in recs or None not in r["st"]:
                recs[r["raceKey"]] = r
    for rk, r in recs.items():
        for x in r.get("runners") or []:
            rows.append((rk, x["num"], x["statut"], (x["direct"] or {}).get("rapport"), (x["ref"] or {}).get("rapport"),
                         (x["direct"] or {}).get("dateRapport"), (x["ref"] or {}).get("dateRapport")))
        for b in r.get("rapports") or []:
            if b["typePari"] == "E_SIMPLE_GAGNANT":
                for y in b["rapports"]:
                    divs.append((rk, y.get("combinaison"), y.get("libelle"), y.get("dividendePourUnEuro"), y.get("nombreGagnants"),
                                 b.get("miseBase"), b.get("rembourse")))
    e = pd.DataFrame(rows, columns=["raceKey", "numPmu", "statut", "e_fin", "e_ref", "e_fin_t", "e_ref_t"])
    dv = pd.DataFrame(divs, columns=["raceKey", "comb", "libelle", "div", "nwin", "miseBase", "remb"])
    return e, dv


def official(dv, lib):
    """Official win dividends -> per (raceKey, numPmu) dividend per 1 euro, and the net win pool (dividend x winning stakes)."""
    bad = set(dv.raceKey[dv.libelle.astype(str).str.lower().str.contains("rembours")])
    dv = dv[(dv.libelle.astype(str).str.lower() == lib.lower()) & ~dv.raceKey.isin(bad)].copy()  # PMU capitalised its labels in mid-Oct 2026
    dv["numPmu"] = pd.to_numeric(dv.comb, errors="coerce")
    dv["d"] = dv["div"] / 100
    dv["net_pool"] = dv.d * dv.nwin * dv.miseBase / 100
    return dv.dropna(subset=["numPmu"])[["raceKey", "numPmu", "d", "net_pool"]]


def ll(p, won, rid):
    """mean winner log-loss of normalised probabilities p over races"""
    s = pd.Series(p).groupby(rid).transform("sum").values
    q = p / s
    return float(-np.log(q[won]).mean())


def main(lo="2018-01-01", hi="2024-12-31", *flags):
    if "--test" not in flags:
        assert hi < "2025-01-01", "dev only: choices are made on dev; pass --test once they are frozen"
    d = pd.read_parquet(PROC / "fr_preds.parquet")
    f = pd.read_parquet(PROC / "fr_feats.parquet", columns=["rid", "numPmu", "raceKey", "field", "quinte"])
    d = d.merge(f, on=["rid", "numPmu"])
    d["year"] = d.date.dt.year
    d["t"] = d.specialite.astype(str).str.startswith("TROT").astype(float)
    # value-rule EV exactly as in fr_value (payout model fitted on dev winners 2018-2024 over the whole dev set)
    d = d[(d.odr_rapport > 1) & (d.odf_rapport > 1)].copy()
    p = d.end_ref.values
    lo_ = np.log(d.odr_rapport.values)
    X = np.c_[np.ones(len(d)), lo_, np.log(p / d.mkt_ref.values), d.t, np.log(d.field), lo_ * d.t]
    wn = d.year.between(*DEV).values & (d.y.values > 0)
    bw, *_ = np.linalg.lstsq(X[wn], np.log(d.odf_rapport.values[wn]), rcond=None)
    d["ev"] = p * np.exp(X @ bw) * float(np.mean(np.exp(np.log(d.odf_rapport.values[wn]) - X[wn] @ bw))) - 1
    d["rule"] = (d.ev > 0.15) & (d.ev < 1.0)
    d = d[d.date.astype(str).str[:10].between(lo, hi)]
    # pools
    e, edv = load_epool()
    rp = pd.concat([pd.read_parquet(q, columns=["raceKey", "typePari", "libelle", "combinaison", "dividendePourUnEuro", "nombreGagnants", "miseBase"])
                    for q in sorted((RAW / "fr").glob("rapports-*.parquet"))], ignore_index=True)
    rp = rp[rp.typePari == "SIMPLE_GAGNANT"].rename(columns={"combinaison": "comb", "dividendePourUnEuro": "div", "nombreGagnants": "nwin"})
    mo, eo = official(rp, "Simple gagnant"), official(edv, "e-Simple Gagnant")
    d = d.merge(e[["raceKey", "numPmu", "e_fin", "e_ref"]], on=["raceKey", "numPmu"], how="inner")
    d = d.merge(mo.rename(columns={"d": "m_div", "net_pool": "m_pool"}), on=["raceKey", "numPmu"], how="left")
    d = d.merge(eo.rename(columns={"d": "e_div", "net_pool": "e_pool"}), on=["raceKey", "numPmu"], how="left")
    # keep races where every runner has both pools' ref and final odds, both pools have an official dividend on the same winner(s)
    ok = (d.e_fin > 1) & (d.e_ref > 1)
    d = d[ok.groupby(d.rid).transform("all")]
    has = d.groupby("rid").agg(mw=("m_div", lambda s: s.notna().sum()), ew=("e_div", lambda s: s.notna().sum()), yw=("y", lambda s: (s > 0).sum()))
    good = has[(has.mw >= 1) & (has.mw == has.ew) & (has.yw >= 1)].index
    d = d[d.rid.isin(good)].copy()
    d["m_ret"], d["e_ret"] = d.m_div.fillna(0.0), d.e_div.fillna(0.0)
    d["g_ref"], d["g_fin"] = np.log(d.e_ref / d.odr_rapport), np.log(d.e_fin / d.odf_rapport)
    rid, won = d.rid.values, (d.y > 0).values
    pools = d.groupby("rid")[["m_pool", "e_pool"]].max()
    rep = {"range": [lo, hi], "races": int(d.rid.nunique()), "runners": int(len(d)),
           "net_win_pool_eur_median": {"main": round(float(pools.m_pool.median())), "internet": round(float(pools.e_pool.median())),
                                       "internet_share_median": round(float((pools.e_pool / (pools.e_pool + pools.m_pool)).median()), 3)}}
    ov = lambda c: float(d.groupby("rid")[c].apply(lambda s: (1 / s).sum()).median())
    rep["overround_median"] = {"main_ref": round(ov("odr_rapport"), 3), "main_final": round(ov("odf_rapport"), 3),
                               "e_ref": round(ov("e_ref"), 3), "e_final": round(ov("e_fin"), 3)}
    rep["winner_logloss"] = {k: round(ll(1 / d[c].values, won, rid), 4) for k, c in
                             (("main_ref", "odr_rapport"), ("e_ref", "e_ref"), ("main_final", "odf_rapport"), ("e_final", "e_fin"))}
    rep["winner_logloss"]["both_ref_geo_mean"] = round(ll(1 / np.sqrt(d.odr_rapport.values * d.e_ref.values), won, rid), 4)
    rep["winner_logloss"]["both_final_geo_mean"] = round(ll(1 / np.sqrt(d.odf_rapport.values * d.e_fin.values), won, rid), 4)
    rep["winner_logloss"]["model_end_ref"] = round(ll(d.end_ref.values, won, rid), 4)
    # gap by odds level (main ref odds)
    bins = [1, 2, 3, 5, 8, 13, 21, 40, 1e9]
    d["ob"] = pd.cut(d.odr_rapport, bins, right=False)
    g = d.groupby("ob", observed=True)
    rep["gap_by_main_ref_odds"] = [dict(odds=str(k), n=int(len(x)), mean_log_gap_final=round(float(x.g_fin.mean()), 3),
                                        share_e_pays_more_final=round(float((x.g_fin > 0).mean()), 3), mean_log_gap_ref=round(float(x.g_ref.mean()), 3),
                                        winners=int((x.y > 0).sum()), mean_log_official_gap_winners=round(float(np.log(x.e_div / x.m_div)[x.y > 0].mean()), 3))
                                   for k, x in g]
    rep["gap_ref_predicts_gap_final"] = dict(corr=round(float(np.corrcoef(d.g_ref, d.g_fin)[0, 1]), 3),
                                             sign_agreement=round(float((np.sign(d.g_ref) == np.sign(d.g_fin)).mean()), 3))
    # strategies (per 1 euro on each runner selected): which pool to bet in
    sel = {"all_runners": np.ones(len(d), bool), "value_rule": d.rule.values,
           "favourite_main_ref": (d.odr_rapport == d.groupby("rid").odr_rapport.transform("min")).values}
    strat = {"main": d.m_ret.values, "internet": d.e_ret.values,
             "higher_at_30min": np.where(d.g_ref > 0, d.e_ret, d.m_ret),
             "oracle_higher_final_odds": np.where(d.g_fin > 0, d.e_ret, d.m_ret)}
    day = d.date.values
    rep["strategies"] = {s: {k: roi_summary(v[m], day[m]) for k, v in strat.items()} for s, m in sel.items()}
    # own stake X euros on the selected horse in the chosen pool: (net pool + r X) / (stake on horse + X), r = pool return rate
    out = {}
    for s in ("value_rule",):
        m = sel[s]
        for X_ in (2, 10, 50):
            res = {}
            for k, (ret, pool) in {"main": (d.m_ret, d.m_pool), "internet": (d.e_ret, d.e_pool)}.items():
                ret, pool = ret.values[m], d.groupby("rid")[pool.name].transform("max").values[m]
                S = np.where(ret > 0, pool / np.maximum(ret, 1e-9), np.nan)
                res[k] = roi_summary(np.where(ret > 0, (pool + 0.85 * X_) / (S + X_), 0.0), day[m])
            out[f"{X_}eur"] = res
    rep["value_rule_with_own_stake"] = out
    m = sel["value_rule"]
    rep["value_rule_by_year"] = {int(Y): {k: roi_summary(v[m & (d.year.values == Y)]) for k, v in (("main", d.m_ret.values), ("internet", d.e_ret.values))}
                                 for Y in sorted(d.year.unique())}
    print(json.dumps(rep, indent=1, default=str))
    tag = "test" if "--test" in flags else "dev"
    json.dump(rep, open(RES / f"pools_{tag}_{lo}_{hi}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(*sys.argv[1:])
