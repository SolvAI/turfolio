"""Hong Kong replication of the combination-bet study: quinella (first two, any order) and tierce (first three in order).

Dividends exist only from 2016-09-28 (1,522 races), so: order models walk-forward by year on 1996-2018 finishing positions,
accuracy dev = 2010-2016, test = 2017-2018; pool / profit analysis dev = 2016-09..12, test = 2017-2018.
The value rule is NOT re-tuned here: it reuses the rule chosen on France dev (final-odds variant), with a payout model fit on the
2016 HK winners only. Only final odds exist in HK, so this is the analogue of France's "final" upper bound.
Run: python -I src/hk_exotics.py -> results/hk_exotics_report.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exotics import PB, RB, combo_table, run_pool  # noqa: E402
from fr_exotics import day_boot, roi_summary  # noqa: E402
from racemodel import fit_clogit, starts_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, RAW = ROOT / "data" / "processed", ROOT / "results", ROOT / "data" / "raw" / "hk"
# HKJC deductions: quinella 17.5 %, tierce 25 %
POOLS = {"QIN": ("quinella", 2, False, .825, "CG"), "TCE": ("tierce", 3, True, .75, "TO")}
NAMES = {"QIN": "Quinella (first two, any order)", "TCE": "Tierce (first three in order)"}
MODELS = ["mkt_harv", "mkt_disc", "end_harv", "end_disc", "end_full"]


def load():
    p = pd.read_parquet(PROC / "hk_preds.parquet")
    r = pd.read_parquet(PROC / "hk_runners.parquet", columns=["rid", "horse_id", "race_no", "horse_no", "final_placing"])
    d = p.merge(r, on=["rid", "horse_id"], how="left")
    d = d.sort_values(["race_date", "rid", "horse_id"]).reset_index(drop=True)
    d["year"] = d.race_date.dt.year
    d["lp"], d["lm"], d["lf"] = np.log(d.end), np.log(d.mkt_final), np.log(np.clip(d.fund, 1e-9, 1))
    for k in (1, 2, 3):
        d[f"n{k}"] = (d.final_placing == k).groupby(d.rid).transform("sum")
    return d


def stage_scores(d):
    Z = {"mdisc": lambda g: np.c_[g.lm], "disc": lambda g: np.c_[g.lp], "full": lambda g: np.c_[g.lp, g.lf, g.lm]}
    out = {f"{k}_{s}": np.full(len(d), np.nan) for k in Z for s in (2, 3)}
    w_last = {}
    for s in (2, 3):
        ok = ((d.n1 == 1) & (d.n2 == 1) & ((d.n3 == 1) if s == 3 else True)) & ~d.final_placing.isin(range(1, s))
        for yr in range(1998, 2019):
            g = d[ok & (d.year < yr)]
            te = (d.year == yr).values
            y = (g.final_placing == s).astype(float).values
            for k, f in Z.items():
                w = fit_clogit(f(g), y, g.rid.values)
                out[f"{k}_{s}"][te] = f(d[te]) @ w
                w_last[f"{k}_{s}"] = np.round(w, 3).tolist()
    print("stage weights (last fold):", w_last, flush=True)
    return out, w_last


def dividends():
    dv = pd.read_csv(RAW / "all_dividends.csv.gz")
    dv["race_date"] = pd.to_datetime(dv.race_date)
    rows = []
    for rd, rn, js in zip(dv.race_date, dv.race_no, dv.dividends):
        j = json.loads(js)
        for key in ("quinella", "tierce"):
            for e in j.get(key) or []:
                rows.append((rd, int(rn), key, [int(x) for x in e["combination"]], float(e["dividend"]) / 10.0))
    return pd.DataFrame(rows, columns=["race_date", "race_no", "pool", "combo", "pay"])


def main():
    d = load()
    sc, w_last = stage_scores(d)
    for k, v in sc.items():
        d[k] = v
    d = d[d.year >= 2010].reset_index(drop=True)
    st, sz = starts_of(d.rid.values)
    d["li"] = np.arange(len(d)) - np.repeat(st, sz)
    nr = len(st)
    yr, day = d.year.values[st], d.race_date.values[st]
    grp = np.where(yr <= 2016, 0, 1).astype(np.int64)
    act = np.full((nr, 3), -1, np.int64)
    rr = np.repeat(np.arange(nr), sz)
    for k in (1, 2, 3):
        ok = (d.final_placing == k).values & (d[f"n{k}"] == 1).values
        act[rr[ok], k - 1] = d.li.values[ok]
    xr = np.c_[np.zeros(nr), np.log(sz)]

    def norm(x):
        return np.exp(x - np.repeat(np.maximum.reduceat(x, st), sz))
    p_end, p_mkt = d.end.values, d.mkt_final.values
    S = {"mkt_harv": (p_mkt, p_mkt, p_mkt), "end_harv": (p_end, p_end, p_end),
         "mkt_disc": (p_mkt, norm(d.mdisc_2.values), norm(d.mdisc_3.values)), "end_disc": (p_end, norm(d.disc_2.values), norm(d.disc_3.values)),
         "end_full": (p_end, norm(d.full_2.values), norm(d.full_3.values))}
    dv = dividends()
    key = pd.Series(np.arange(nr), index=pd.MultiIndex.from_arrays([d.race_date.values[st], d.race_no.values[st].astype(int)]))
    li = {(r_, int(h)): l for r_, h, l in zip(rr, d.horse_no.fillna(-1).values, d.li.values)}
    fr = json.load(open(RES / "exotics_report.json"))
    rep = {"stage_weights_last_fold": w_last, "takeout_assumed_return_to_players": {p: POOLS[p][3] for p in POOLS}}
    nocoef = np.array([0., 0, 0, 0, 0, 1])
    for pool, (pkey, k, ordered, trj, fr_pool) in POOLS.items():
        C, off = combo_table(k, ordered)
        W = np.full((nr, 4, 3), -1, np.int64)
        PAY, NW, valid = np.zeros((nr, 4)), np.zeros(nr, np.int64), np.zeros(nr, bool)
        for (rd, rn), g in dv[dv.pool == pkey].groupby(["race_date", "race_no"]):
            if (rd, rn) not in key.index:
                continue
            r = int(key[(rd, rn)])
            ok = True
            for w, (cmb, pay) in enumerate(zip(g.combo, g.pay)):
                if w >= 4 or any((r, h) not in li for h in cmb) or len(cmb) != k:
                    ok = False
                    break
                loc = [li[(r, h)] for h in cmb]
                W[r, w, :k] = loc if ordered else sorted(loc)
                PAY[r, w] = pay
                NW[r] = w + 1
            valid[r] = ok and NW[r] > 0 and pay > 0
        gpay = np.where(valid, grp, -1)
        R = {"races_with_valid_payout": {"dev": int((gpay == 0).sum()), "test": int((gpay == 1).sum())}, "accuracy": {}, "vs_market": {}}
        per = {}
        for m in MODELS:
            o = run_pool(st, sz, *S[m], *S["mkt_harv"], k, ordered, C, off, act, W, PAY, NW, grp, gpay, nocoef, xr, np.inf, 1.0, 1)
            per[m] = o
            R["accuracy"][m] = {}
            for lab, gi in (("dev", 0), ("test", 1)):
                sel = (grp == gi) & np.isfinite(o[0])
                R["accuracy"][m][lab] = {"races": int(sel.sum()), "logloss": round(float(-np.log(o[0][sel]).mean()), 4),
                                         "top_combo_hit": round(float(o[5][sel].mean()), 4)}
        base = per["mkt_harv"]
        for m in MODELS[1:]:
            o = per[m]
            sel = (grp == 1) & np.isfinite(o[0]) & np.isfinite(base[0])
            dl = -np.log(o[0][sel]) + np.log(base[0][sel])
            dh = o[5][sel].astype(float) - base[5][sel]
            R["vs_market"][m] = {"logloss_diff": round(float(dl.mean()), 4), "ll_ci95": day_boot(dl, day[sel]),
                                 "top_hit_diff": round(float(dh.mean()), 4), "top_ci95": day_boot(dh, day[sel])}
        R["pool"] = {}
        for lab, gi in (("dev", 0), ("test", 1)):
            sel = gpay == gi
            pay = PAY[sel, 0]
            blk = {"races": int(sel.sum()), "median_payout": round(float(np.median(pay)), 2),
                   "pool_logloss_at_assumed_takeout": round(float(-np.log(trj / pay).mean()), 4)}
            for m in ("mkt_harv", "end_full"):
                Pw = per[m][2][sel]
                okk = np.isfinite(Pw)
                g_ = np.log(Pw[okk] * pay[okk])
                blk[m] = {"logloss": round(float(-np.log(Pw[okk]).mean()), 4), "growth_bet_all_combos_in_proportion": round(float(g_.mean()), 4),
                          "growth_ci95": day_boot(g_, day[sel][okk]) if okk.sum() > 50 else None,
                          "implied_return_to_players_ipw": round(float(1 / np.mean(1 / (Pw[okk] * pay[okk]))), 4)}
            R["pool"][lab] = blk
        R["top_combo"] = {}
        for m in ("mkt_harv", "end_full"):
            o = per[m]
            sel = gpay == 1
            ret, ag = o[6][sel], o[9][sel].astype(bool)
            R["top_combo"][m] = {"test": {"all": roi_summary(ret, day[sel]), "same_as_market_top": roi_summary(ret[ag]),
                                          "differs_from_market_top": roi_summary(ret[~ag]), "share_same": round(float(ag.mean()), 4)}}
        o = per["end_full"]
        R["roi_by_market_prob"] = {"test": [{"market_prob": f"{PB[i]:g}-{PB[i + 1]:g}", "combos": int(o[11][1, i, 0]),
                                             "roi": round(float(o[11][1, i, 1] / o[11][1, i, 0] - 1), 4)} for i in range(len(PB) - 1) if o[11][1, i, 0] > 0]}
        R["roi_by_edge"] = {"test": [{"model_over_market": f"{RB[i]:g}-{RB[i + 1]:g}", "combos": int(o[12][1, i, 0]),
                                      "roi": round(float(o[12][1, i, 1] / o[12][1, i, 0] - 1), 4)} for i in range(len(RB) - 1) if o[12][1, i, 0] > 0]}
        # value rule: France's rule (final variant) with an HK payout model fit on the 2016 winners
        sel = (gpay == 0) & np.isfinite(o[2])
        Pw, Hw, pay = o[2][sel], o[3][sel], PAY[sel, 0]
        X = np.c_[np.ones(sel.sum()), np.log(Hw), np.log(Pw / Hw), np.log(sz[sel])]
        beta, *_ = np.linalg.lstsq(X, np.log(pay), rcond=None)
        res = np.log(pay) - X @ beta
        coef = np.r_[beta[:3], 0.0, beta[3], np.mean(np.exp(res))]
        oc = run_pool(st, sz, *S["end_full"], *S["mkt_harv"], k, ordered, C, off, act, W, PAY, NW, grp, gpay, coef, xr, 0.0, 1e-4, 20_000_000)
        cr, cp, ch, cev, cret = oc[13:18]
        rule = fr[fr_pool]["value_rule"]["final"]["chosen_on_dev"]
        test_all = {}
        for key_ in fr[fr_pool]["value_rule"]["final"]["test_all"]:
            pm, tau = (float(x.split(">=")[-1].split(">")[-1]) for x in key_.split(","))
            mm = (gpay[cr] == 1) & (cp >= pm) & (cev > tau)
            test_all[key_] = roi_summary(cret[mm], day[cr][mm])
        R["value_rule"] = {"payout_model_coef_const_logH_logEdge_logField": [round(float(x), 4) for x in beta], "rule_from_france_dev": rule,
                           "test_rule_from_france": test_all.get(rule), "test_all": test_all}
        rep[pool] = R
        a = R["accuracy"]
        print(f"{pool}: test log-loss mkt_harv {a['mkt_harv']['test']['logloss']} mkt_disc {a['mkt_disc']['test']['logloss']} end_full "
              f"{a['end_full']['test']['logloss']}; top hit {a['mkt_harv']['test']['top_combo_hit']} vs {a['end_full']['test']['top_combo_hit']}", flush=True)
        print(f"  pool test {json.dumps(R['pool']['test'])}\n  value rule {rule}: {test_all.get(rule)}", flush=True)
    json.dump(rep, open(RES / "hk_exotics_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
