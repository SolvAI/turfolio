"""France: simple placé (place bets: the horse finishes in the first 3, or first 2 with 4-7 starters).

The place pool is split in equal parts between the placed horses:  payout_i = 1 + (R * pool - stakes on the placed) / (m * stake_i),
rounded down to 0.10, never below the guaranteed minimum (1.10; 1.05 in some races since 2021). PMU publishes for every placed
horse its payout and the number of winning stakes, so (payout_i - 1) * stake_i is equal across the placed horses of a race (checked:
median deviation 2.5 %). This lets us measure how the place pool is spread compared with the win pool.

  P(place)       from the order models of fr_exotics (Plackett-Luce with stage-2/3 strengths), m = 2 or 3 places
  pool model     place-pool share s_i proportional to (win-market prob)^gamma (gamma by trot/flat, fitted on dev placed pairs),
                 return rate R fitted on dev dividends; E[payout_i | i placed] by enumerating every co-placed set with the model
  value rule     EV = P(place) * E[payout | placed] (corrected by a log-payout regression on dev) - 1; threshold chosen on 2019-24
Run: python -I src/fr_place.py   -> results/place_report.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_exotics import DEV, RAW, RES, PROC, TEST, YEARS, day_boot, load, roi_summary, strengths  # noqa: E402
from racemodel import starts_of  # noqa: E402

MODELS = {"ref": ["mkt_harv_ref", "mkt_disc_ref", "end_harv_ref", "end_stage_ref"],
          "final": ["mkt_harv_final", "mkt_disc_final", "end_harv_final", "end_stage_final"]}
WIN_BINS = [0, .02, .05, .1, .2, .3, .4, .5, .6, .7, 1.01]
PL_BINS = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
RATIO_BINS = [0, .67, .8, .9, .95, 1.05, 1.1, 1.25, 1.5, 2, 1e9]


@njit(cache=True)
def place_ev(st, sz, p, a, b, m, s, R, minpay, want_ev):
    """P(i placed) and E[payout_i | i placed] under the order model (p, a, b), m[r] places, place-pool shares s, return rate R[r]."""
    n_all = len(p)
    P = np.zeros(n_all)
    E = np.zeros(n_all)
    for r in range(len(st)):
        o, n, k = st[r], sz[r], m[r]
        if k < 2 or n < k + 1:
            continue
        S2 = 0.0
        S3 = 0.0
        for i in range(n):
            S2 += a[o + i]
            S3 += b[o + i]
        for i in range(n):
            pi = p[o + i]
            for j in range(n):
                if j == i:
                    continue
                pij = pi * a[o + j] / (S2 - a[o + i])
                if k == 2:
                    if want_ev:
                        sa = s[o + i] + s[o + j]
                        for x in (i, j):
                            v = 1.0 + (R[r] - sa) / (2.0 * s[o + x])
                            v = np.floor(v * 10.0 + 1e-9) / 10.0
                            E[o + x] += pij * max(v, minpay[r])
                    P[o + i] += pij
                    P[o + j] += pij
                    continue
                for l in range(n):
                    if l == i or l == j:
                        continue
                    pijl = pij * b[o + l] / (S3 - b[o + i] - b[o + j])
                    P[o + i] += pijl
                    P[o + j] += pijl
                    P[o + l] += pijl
                    if want_ev:
                        sa = s[o + i] + s[o + j] + s[o + l]
                        for x in (i, j, l):
                            v = 1.0 + (R[r] - sa) / (3.0 * s[o + x])
                            v = np.floor(v * 10.0 + 1e-9) / 10.0
                            E[o + x] += pijl * max(v, minpay[r])
    for t in range(n_all):
        if P[t] > 0:
            E[t] = E[t] / P[t]
    return P, E


def place_dividends(keys, field=None):
    rp = pd.concat([pd.read_parquet(p, columns=["raceKey", "typePari", "libelle", "combinaison", "dividendePourUnEuro", "nombreGagnants", "miseBase"])
                    for p in sorted(RAW.glob("rapports-*.parquet"))], ignore_index=True)
    rp = rp[(rp.typePari == "SIMPLE_PLACE") & rp.raceKey.isin(keys)]
    bad = set(rp.raceKey[rp.libelle.str.lower().str.contains("rembours")]) | set(rp.raceKey[rp.dividendePourUnEuro <= 0])
    # PMU relabelled in mid-October 2026: "Simple Placé" for every field size (no separate "4 à 7 partants" label); the archive has none
    m = rp[rp.libelle.str.lower().isin(["simple placé", "simple placé 4 à 7 partants"]) & ~rp.raceKey.isin(bad)].copy()
    m["miseBase"] = rp.loc[m.index, "miseBase"] if "miseBase" in rp.columns else 100
    m["numPmu"] = pd.to_numeric(m.combinaison, errors="coerce")
    m = m[m.numPmu.notna()]
    m["numPmu"] = m.numPmu.astype(int)
    m["div"] = m.dividendePourUnEuro / 100.0
    m["places"] = np.where(m.libelle.str.contains("4 à 7"), 2, 3)
    new = (m.libelle == "Simple Placé").values  # new label: 2 places with 4-7 starters, 3 with 8 or more
    if new.any():
        m.loc[new, "places"] = np.where(m.raceKey[new].map(field).values <= 7, 2, 3)
    # place-pool size that does NOT depend on the result: gross payout = sum over placed horses of payout x stake = R x pool
    # (the net profit pool R x pool - stakes on the placed would shrink when favourites place: an outcome-dependent filter)
    m["stake_eur"] = m.nombreGagnants * m.miseBase / 100
    m["net_place_pool"] = (m["div"] * m.stake_eur).groupby(m.raceKey).transform("sum")
    return m[["raceKey", "numPmu", "div", "nombreGagnants", "places", "net_place_pool", "stake_eur"]]


def bin_roi(x, ret, edges, sel_groups, label):
    out = {}
    for lab_g, sel in sel_groups.items():
        bb = np.digitize(x[sel], edges) - 1
        rows = []
        for i in range(len(edges) - 1):
            mm = bb == i
            if mm.sum() == 0:
                continue
            r_ = ret[sel][mm]
            rows.append({label: f"{edges[i]:g}-{min(edges[i + 1], 1e9):g}", "bets": int(mm.sum()), "hit_rate": round(float((r_ > 0).mean()), 4),
                         "avg_payout_when_hit": round(float(r_[r_ > 0].mean()), 3) if (r_ > 0).any() else None, "roi": round(float(r_.mean() - 1), 4)})
        out[lab_g] = rows
    return out


def bll(P, y):
    P = np.clip(P, 1e-6, 1 - 1e-6)
    return -(y * np.log(P) + (1 - y) * np.log(1 - P))


def main():
    t0 = time.time()
    d = load()
    d = d.merge(pd.read_parquet(PROC / "fr_stage.parquet"), on=["rid", "numPmu"], how="left")
    d = d[d.year >= YEARS[0]]
    dv = place_dividends(set(d.raceKey), d.groupby("raceKey").size())
    d = d.merge(dv.drop(columns=["net_place_pool"], errors="ignore"), on=["raceKey", "numPmu"], how="left")
    d["placed"] = d["div"].notna().astype(float)
    # a race is valid when its dividends exist, every placed horse is among our starters, and at least m horses placed
    rk = dv.groupby("raceKey").agg(np_div=("numPmu", "size"), places=("places", "first"), net_place_pool=("net_place_pool", "first"))
    dv = dv.drop(columns=["net_place_pool"])
    d = d.drop(columns=["places", "net_place_pool"], errors="ignore").merge(rk, left_on="raceKey", right_index=True, how="left")
    ours = d.groupby("rid").placed.transform("sum")
    d = d[d.np_div.notna() & (ours == d.np_div) & (d.np_div >= d.places)]
    d = d.sort_values(["ts", "rid", "numPmu"]).reset_index(drop=True)
    st, sz = starts_of(d.rid.values)
    nr = len(st)
    yr, day = d.year.values[st], d.date.values[st]
    grp = np.where((yr >= DEV[0]) & (yr <= DEV[1]), 0, np.where(yr >= TEST[0], 1, -1))
    g_row = np.repeat(grp, sz)
    day_row = np.repeat(day, sz)
    m = d.places.values[st].astype(np.int64)
    trot = d.t.values
    y = d.placed.values
    ret = np.where(y > 0, d["div"].fillna(0).values, 0.0)
    print(f"{nr:,} races with valid place dividends ({d.date.min().date()} .. {d.date.max().date()}), {len(d):,} starters; "
          f"dev {(grp == 0).sum():,} test {(grp == 1).sum():,}", flush=True)
    rep = {"races": {"dev": int((grp == 0).sum()), "test": int((grp == 1).sum())}}
    zero = np.zeros(len(d))
    one = np.ones(nr)
    # ------------------------------------------------------------------ accuracy of P(place)
    PP = {}
    for v, models in MODELS.items():
        for mn in models:
            p, a, b = strengths(d, mn)
            PP[mn], _ = place_ev(st, sz, p, a, b, m, zero, one, one, False)
    chk = np.add.reduceat(PP["end_stage_ref"], st) / m
    print(f"  P(place) sums to m: min {chk.min():.4f} max {chk.max():.4f}", flush=True)
    acc = {}
    for mn, P in PP.items():
        acc[mn] = {}
        for lab_g, gi in (("dev", 0), ("test", 1)):
            sel = g_row == gi
            # top place pick: the horse with the highest P(place) in each race
            stg, szg = starts_of(d.rid.values[sel])
            Pg, yg = P[sel], y[sel]
            rr = np.repeat(np.arange(len(stg)), szg)
            top = np.lexsort((-Pg, rr))[stg]
            acc[mn][lab_g] = {"logloss_per_horse": round(float(bll(Pg, yg).mean()), 5), "brier": round(float(((Pg - yg) ** 2).mean()), 5),
                              "top_pick_places": round(float(yg[top].mean()), 4),
                              "top_pick_roi": round(float(ret[sel][top].mean() - 1), 4)}
    rep["accuracy"] = acc
    rep["vs_market"] = {}
    for v in MODELS:
        base = PP[f"mkt_disc_{v}"]
        for mn in MODELS[v]:
            sel = g_row == 1
            dl = bll(PP[mn][sel], y[sel]) - bll(base[sel], y[sel])
            per_race = np.add.reduceat(dl, starts_of(d.rid.values[sel])[0])
            rep["vs_market"][mn] = {"logloss_diff_per_horse": round(float(dl.mean()), 5),
                                    "per_race_ci95": day_boot(per_race, day[grp == 1])}
    # calibration of the main model
    P = PP["end_stage_ref"]
    rep["calibration_end_stage_ref"] = {}
    for lab_g, gi in (("dev", 0), ("test", 1)):
        sel = g_row == gi
        bb = np.digitize(P[sel], PL_BINS) - 1
        rep["calibration_end_stage_ref"][lab_g] = [{"bin": f"{PL_BINS[i]:g}-{min(PL_BINS[i + 1], 1):g}", "n": int((bb == i).sum()),
                                                    "predicted": round(float(P[sel][bb == i].mean()), 4), "actual": round(float(y[sel][bb == i].mean()), 4)}
                                                   for i in range(len(PL_BINS) - 1) if (bb == i).any()]
    for k in rep["accuracy"]:
        print(f"  {k:16s} test {rep['accuracy'][k]['test']}", flush=True)
    groups = {"dev": g_row == 0, "test": g_row == 1}
    # ------------------------------------------------------------------ where the money goes: ROI of every place bet by bins
    rep["roi_by_win_prob_final"] = bin_roi(d.mkt_final.values, ret, WIN_BINS, groups, "win_prob_final")
    rep["roi_by_win_prob_ref"] = bin_roi(d.mkt_ref.values, ret, WIN_BINS, groups, "win_prob_ref")
    for v in MODELS:
        rep[f"roi_by_model_place_prob_{v}"] = bin_roi(PP[f"end_stage_{v}"], ret, PL_BINS, groups, "model_place_prob")
        rep[f"roi_by_edge_{v}"] = bin_roi(PP[f"end_stage_{v}"] / PP[f"mkt_disc_{v}"], ret, RATIO_BINS, groups, "model_over_market")
    # ------------------------------------------------------------------ place-pool structure (dev placed horses)
    pool = {}
    plc = (y > 0) & (g_row == 0)
    rid = d.rid.values
    for v in MODELS:
        lq = np.log(d[f"mkt_{v}"].values)
        lN = np.log(d.nombreGagnants.values)
        gam = {}
        for tt in (1.0, 0.0):
            sel = plc & (trot == tt)
            h = pd.DataFrame({"rid": rid[sel], "lq": lq[sel], "lN": lN[sel]})
            dq = h.lq - h.groupby("rid").lq.transform("mean")
            dN = h.lN - h.groupby("rid").lN.transform("mean")
            gam[tt] = float(np.polyfit(dq, dN, 1)[0])
        gvec = np.where(trot == 1, gam[1.0], gam[0.0])
        e = np.exp(gvec * lq)
        s = e / np.repeat(np.add.reduceat(e, st), sz)
        # return rate per race implied by each placed horse: R = (payout - 1 + 0.05 rounding) * m * s_i + sum of placed shares
        sA = np.add.reduceat(np.where(y > 0, s, 0.0), st)
        Ri = (d["div"].values - 1 + 0.05) * np.repeat(m, sz) * s + np.repeat(sA, sz)
        okR = plc & (d["div"].values >= 1.3)
        Rt = {tt: float(np.median(Ri[okR & (trot == tt)])) for tt in (1.0, 0.0)}
        Rr = np.where(trot[st] == 1, Rt[1.0], Rt[0.0])
        minpay = np.where(yr >= 2021, 1.05, 1.10)
        p, a, b = strengths(d, f"end_stage_{v}")
        _, Ehat = place_ev(st, sz, p, a, b, m, s, Rr, minpay, True)
        # correct the structural payout with a log regression on dev placed horses (also tells how much of the model edge is priced)
        lr = np.log(PP[f"end_stage_{v}"] / PP[f"mkt_disc_{v}"])
        X = np.c_[np.ones(len(d)), np.log(Ehat - 1 + 1e-3), lr, trot, np.log(np.repeat(sz, sz))]
        okf = plc & (Ehat > 1)
        beta, *_ = np.linalg.lstsq(X[okf], np.log(d["div"].values[okf] - 1 + 1e-3), rcond=None)
        res = np.log(d["div"].values[okf] - 1 + 1e-3) - X[okf] @ beta
        smear = float(np.mean(np.exp(res)))
        Ecor = 1 + np.exp(X @ beta) * smear
        Ecor = np.maximum(Ecor, np.repeat(minpay, sz))
        for lab_g, gi in (("dev", 0), ("test", 1)):
            ok = (y > 0) & (g_row == gi) & (Ehat > 1)
            pool.setdefault(v, {})[f"payout_fit_{lab_g}"] = {
                "structural_log_ratio_mean": round(float(np.mean(np.log(d["div"].values[ok] / Ehat[ok]))), 4),
                "corrected_log_ratio_mean": round(float(np.mean(np.log(d["div"].values[ok] / Ecor[ok]))), 4),
                "corrected_log_ratio_sd": round(float(np.std(np.log(d["div"].values[ok] / Ecor[ok]))), 4)}
        exact = json.load(open(RES / "place_pool_params.json")) if (RES / "place_pool_params.json").exists() else {}
        exact[v] = {"gamma_trot": gam[1.0], "gamma_flat": gam[0.0], "R_trot": Rt[1.0], "R_flat": Rt[0.0], "beta": [float(x_) for x_ in beta],
                    "smear": smear, "fitted_on": f"dev {DEV[0]}-{DEV[1]} placed horses"}
        json.dump(exact, open(RES / "place_pool_params.json", "w"), indent=1)  # unrounded, for fr_v3_final.py
        pool[v].update({"gamma_trot": round(gam[1.0], 4), "gamma_flat": round(gam[0.0], 4), "return_rate_trot": round(Rt[1.0], 4),
                        "return_rate_flat": round(Rt[0.0], 4),
                        "payout_regression_const_logEhatm1_logEdge_trot_logField": [round(float(x_), 4) for x_ in beta], "smearing": round(smear, 4)})
        print(f"  pool {v}: {pool[v]}", flush=True)
        # -------------------------------------------------------------- value rule
        P = PP[f"end_stage_{v}"]
        EV = P * Ecor - 1
        if v == "ref":  # per-horse rows for fr_place_conf.py (confidence rule, window splits, own-stake effect)
            plc_st = pd.Series(np.where(y > 0, d.stake_eur.fillna(0).values, 0.0)).groupby(d.rid.values).transform("sum").values
            pd.DataFrame({"rid": d.rid.values, "numPmu": d.numPmu.values, "raceKey": d.raceKey.values, "date": d.date.values,
                          "P": P, "EV": EV, "ret": ret, "placed": y, "places": np.repeat(m, sz), "stake_eur": d.stake_eur.values,
                          "net_pool": d.net_place_pool.values - plc_st, "minpay": np.repeat(minpay, sz),
                          "P_mkt": PP["mkt_disc_ref"]}).to_parquet(PROC / "fr_place_rows.parquet", index=False)
        eb = [-1, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 1e9]
        rep.setdefault("value_rule", {})[v] = {"roi_by_estimated_ev": bin_roi(EV, ret, eb, groups, "est_ev")}
        grid = [(pm, tau) for pm in (0.0, 0.2, 0.4, 0.6) for tau in (-0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3)]
        dev_res, test_res = {}, {}
        for pm, tau in grid:
            key = f"p>={pm:g},ev>{tau:g}"
            mm = (g_row == 0) & (P >= pm) & (EV > tau)
            dev_res[key] = roi_summary(ret[mm])
            mm = (g_row == 1) & (P >= pm) & (EV > tau)
            test_res[key] = roi_summary(ret[mm], day_row[mm])
        okk = [(k_, r_) for k_, r_ in dev_res.items() if r_["bets"] >= 1000]
        best = max(okk, key=lambda kv: kv[1]["roi"])[0] if okk else None
        rep["value_rule"][v].update({"dev": dev_res, "chosen_on_dev": best, "test_chosen": test_res.get(best), "test_all": test_res})
        print(f"  value rule {v}: chosen {best}; dev {dev_res.get(best)}; test {test_res.get(best)}", flush=True)
        if best:
            pm_, tau_ = [float(x.split(">")[-1].lstrip("=")) for x in best.split(",")]
            sel = (P >= pm_) & (EV > tau_)
            # pool-size bands = quintiles of the (outcome-free) gross place pool over dev races
            gp = d.net_place_pool.values[st]
            pbins = [0] + [float(x) for x in np.quantile(gp[grp == 0], [.2, .4, .6, .8])] + [1e12]
            pp = np.digitize(d.net_place_pool.values, pbins) - 1
            rep["value_rule"][v]["by_place_pool"] = {lab: [{"net_place_pool_eur": f"{pbins[i]:,.0f}-{pbins[i + 1]:,.0f}".replace("1,000,000,000,000", "max"),
                                                            **roi_summary(ret[sel & (g_row == gi) & (pp == i)])} for i in range(len(pbins) - 1)]
                                                     for lab, gi in (("dev", 0), ("test", 1))}
            print("    by place pool:", {lab: [(x["net_place_pool_eur"], x["bets"], x.get("roi")) for x in rows]
                                         for lab, rows in rep["value_rule"][v]["by_place_pool"].items()}, flush=True)
            # rule restricted, ON DEV, to the bottom pool quintiles while each is profitable on dev; own stake X on the payout:
            # payout' = 1 + (net + (R - 1) X) / (m (S + X)), net = gross pool - stakes on the placed, S = stake on the horse, R ~ 0.85
            dev_q = rep["value_rule"][v]["by_place_pool"]["dev"]
            nq = 0
            while nq < len(dev_q) and dev_q[nq].get("bets") and dev_q[nq]["roi"] > 0:
                nq += 1
            cap = pbins[nq] if nq > 0 else 0.0
            rep["value_rule"][v]["pool_cap_chosen_on_dev_eur"] = round(cap, 0)
            small = sel & (d.net_place_pool.values < cap)
            S_h = d.stake_eur.values
            placed_stakes = pd.Series(np.where(y > 0, S_h, 0.0)).groupby(d.rid.values).transform("sum").values
            net_pool = d.net_place_pool.values - placed_stakes
            mm_ = np.repeat(m, sz)
            out_small = {}
            for lab, gi in (("dev", 0), ("test", 1)):
                k = small & (g_row == gi)
                blk = {"no_stake_effect": roi_summary(ret[k], day_row[k])}
                for X_ in (10, 50):  # official payout minus the drop of the unrounded payout (fixed 2026-11-07: no second rounding)
                    Sk = np.where(ret[k] > 0, S_h[k], 1.0)
                    drop = net_pool[k] / (mm_[k] * Sk) - (net_pool[k] + (0.85 - 1) * X_) / (mm_[k] * (Sk + X_))
                    blk[f"stake_{X_}eur"] = roi_summary(np.where(ret[k] > 0, np.maximum(ret[k] - drop, 1.05), 0.0), day_row[k])
                out_small[lab] = blk
            rep["value_rule"][v]["small_pools"] = out_small
            print(f"    pools < {cap:,.0f} EUR (chosen on dev):", {lab: {kk: (vv["bets"], vv["roi"], vv.get("roi_ci95")) for kk, vv in b_.items()} for lab, b_ in out_small.items()}, flush=True)
        # bridge-jumper check: heavy favourites (minimum guaranteed payout)
        bj = {}
        for thr in (0.4, 0.5, 0.6, 0.7):
            for lab_g, gi in (("dev", 0), ("test", 1)):
                mm = (g_row == gi) & (d[f"mkt_{v}"].values >= thr)
                bj[f"win_prob>={thr:g}_{lab_g}"] = {**roi_summary(ret[mm], day_row[mm]), "share_paid_minimum": round(float(
                    (ret[mm][ret[mm] > 0] <= 1.10).mean()), 4) if (ret[mm] > 0).any() else None}
        rep.setdefault("heavy_favourites", {})[v] = bj
    rep["pool_model"] = pool
    json.dump(rep, open(RES / "place_report.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> results/place_report.json", flush=True)


if __name__ == "__main__":
    main()
