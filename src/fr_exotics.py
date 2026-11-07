"""France: combination bets (couplé ordre = exacta, couplé gagnant = quinella, trio ordre = trifecta, trio).

  stages  order-of-finish models, walk-forward by year (train on 2018..Y-1, predict Y, Y = 2019..2026):
            per variant (ref = decided 30 min out on the end model at reference odds, final = upper bound with final odds)
            and per stage (2nd given the winner, 3rd given the first two), the per-horse log-strength of
              mdisc   market probability ^ lambda          (discounted Harville on the market)
              disc    end-model probability ^ lambda       (discounted Harville on the end model)
              full    conditional logit on [log end prob, fundamental score, log market prob] (x trot)
              stage   full + residual LightGBM (place-specific features)
          -> data/processed/fr_stage.parquet
  eval    accuracy of every order model, pool comparison (implied probabilities after takeout), top combination vs payout vs
          profit, ROI by price and by model edge, and a value rule chosen on 2019-2024 then applied once to 2025-2026
          -> results/exotics_report.json
Run: python -I src/fr_exotics.py stages | eval
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exotics import PB, RB, combo_table, run_pool  # noqa: E402
from fr_eval import RES_GBM, RESID_FEATS  # noqa: E402
from racemodel import fit_clogit, fit_gbm, predict_clogit, starts_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, RAW = ROOT / "data" / "processed", ROOT / "results", ROOT / "data" / "raw" / "fr"
YEARS = list(range(2019, 2027))
DEV, TEST = (2019, 2024), (2025, 2026)
PLACE_FEATS = ["place_rate", "place_rate_rk", "mus_top3_5", "mus_dq5", "h_dq3", "mus_fail5", "nombrePlacesSecond", "nombrePlacesTroisieme", "n_starts",
               "mus_last", "h_last_pos", "h_avg_pos3"]
# takeout: official average return to players (OFDT / PMU): couplé 74.79 %, trio 77.89 % (ordre variants assumed equal); checked below
POOLS = {"CO": ("COUPLE_ORDRE", "couplé ordre", 2, True, .7479), "CG": ("COUPLE_GAGNANT", "couplé gagnant", 2, False, .7479),
         "TO": ("TRIO_ORDRE", "trio ordre", 3, True, .7789), "TR": ("TRIO", "trio", 3, False, .7789)}
NAMES = {"CO": "Couplé ordre (exacta: first two in order)", "CG": "Couplé gagnant (quinella: first two, any order)",
         "TO": "Trio ordre (trifecta: first three in order)", "TR": "Trio (first three, any order)"}
rng = np.random.default_rng(0)


def load():
    pr = pd.read_parquet(PROC / "fr_preds.parquet")
    cols = ["rid", "numPmu", "raceKey", "pos", "ts", "mkt_overr", "mkt_overf"] + [c for c in RESID_FEATS + PLACE_FEATS if c not in ("f_score", "lmr", "mr_rank")]
    cols = [c for c in dict.fromkeys(cols) if c in ("rid", "numPmu") or c not in pr.columns]
    f = pd.read_parquet(PROC / "fr_feats.parquet", columns=cols)
    d = pr.merge(f, on=["rid", "numPmu"], how="inner").merge(pd.read_parquet(PROC / "fr_oof.parquet")[["rid", "numPmu", "f_score"]], on=["rid", "numPmu"])
    d = d.sort_values(["ts", "rid", "numPmu"]).reset_index(drop=True)
    d["year"] = d.date.dt.year
    d["t"] = d.specialite.astype(str).str.startswith("TROT").astype(float)
    d["lmr"], d["lmf"] = np.log(d.mkt_ref), np.log(d.mkt_final)
    d["move"] = d.lmf - d.lmr
    d["mr_rank"] = d.groupby("rid").mkt_ref.rank(ascending=False)
    d["mf_rank"] = d.groupby("rid").mkt_final.rank(ascending=False)
    for v in ("ref", "final"):
        d[f"lp_{v}"] = np.log(np.clip(d[f"end_{v}"], 1e-9, 1))
        d[f"p_rank_{v}"] = d.groupby("rid")[f"end_{v}"].rank(ascending=False)
    # unique 1st / 2nd / 3rd per race
    for k in (1, 2, 3):
        d[f"n{k}"] = (d.pos == k).groupby(d.rid).transform("sum")
    return d


def Zs(d, v, kind):
    t, lp, lm = d.t.values, d[f"lp_{v}"].values, (d.lmr if v == "ref" else d.lmf).values
    if kind == "mdisc":
        return np.c_[lm, lm * t]
    if kind == "disc":
        return np.c_[lp, lp * t]
    return np.c_[lp, lp * t, d.f_score, d.f_score * t, lm, lm * t]


def gbm_X(d, v):
    X = d[RESID_FEATS + PLACE_FEATS].copy()
    X["lp"], X["p_rank"] = d[f"lp_{v}"], d[f"p_rank_{v}"]
    if v == "final":
        X["lmf"], X["mf_rank"], X["move"], X["mkt_overf"] = d.lmf, d.mf_rank, d.move, d.mkt_overf
    return X


def stage_rows(d, s):
    """Rows of the softmax for stage s (2 = second given the winner, 3 = third given the first two)."""
    ok = (d.n1 == 1) & (d.n2 == 1) & ((d.n3 == 1) if s == 3 else True)
    keep = ok & ~d.pos.isin(range(1, s))
    return keep.values


def stages():
    d = load()
    print(f"{len(d):,} starters, {d.rid.nunique():,} races, {d.date.min().date()} .. {d.date.max().date()}", flush=True)
    out = d[["rid", "numPmu"]].copy()
    rounds = {}
    for v in ("ref", "final"):
        for s in (2, 3):
            m = stage_rows(d, s)
            g = d[m]
            y = (g.pos == s).astype(float).values
            tr, va = (g.year <= 2021).values, (g.year == 2022).values
            w = fit_clogit(Zs(g[tr], v, "full"), y[tr], g.rid.values[tr])
            _, s_tr = predict_clogit(w, Zs(g[tr], v, "full"), g.rid.values[tr])
            _, s_va = predict_clogit(w, Zs(g[va], v, "full"), g.rid.values[va])
            b = fit_gbm(gbm_X(g[tr], v), y[tr], g.rid.values[tr], RES_GBM, 3000, init=s_tr,
                        valid=(gbm_X(g[va], v), y[va], g.rid.values[va], s_va))
            rounds[f"{v}_{s}"] = max(int(b.best_iteration), 50)
            print(f"  rounds {v} stage {s}: {rounds[f'{v}_{s}']}", flush=True)
    json.dump(rounds, open(RES / "exotics_rounds.json", "w"))
    for v in ("ref", "final"):
        for s in (2, 3):
            for kind in ("mdisc", "disc", "full", "stage"):
                out[f"{kind}_{v}_{s}"] = np.nan
            m = stage_rows(d, s)
            for yr in YEARS:
                t0 = time.time()
                trm = m & (d.year < yr).values
                te = d.year.values == yr
                if not te.any():
                    continue
                g = d[trm]
                y = (g.pos == s).astype(float).values
                for kind in ("mdisc", "disc", "full"):
                    w = fit_clogit(Zs(g, v, kind), y, g.rid.values)
                    out.loc[te, f"{kind}_{v}_{s}"] = Zs(d[te], v, kind) @ w
                    if kind == "full":
                        s_tr = Zs(g, v, kind) @ w
                        bst = fit_gbm(gbm_X(g, v), y, g.rid.values, RES_GBM, rounds[f"{v}_{s}"], init=s_tr)
                        out.loc[te, f"stage_{v}_{s}"] = out.loc[te, f"full_{v}_{s}"].values + bst.predict(gbm_X(d[te], v), raw_score=True)
                    if yr == YEARS[-1]:
                        print(f"    {v} s{s} {kind} weights (last fold): {np.round(w, 3).tolist()}", flush=True)
                print(f"  {v} stage {s} {yr}: train rows {len(g):,} ({time.time() - t0:.0f}s)", flush=True)
    out.to_parquet(PROC / "fr_stage.parquet", index=False)


# ------------------------------------------------------------------------------------------------ evaluation
def payouts(pool, keys):
    """Winning combinations of a pool: main rows only (exact label), digits only. Races with a refund, or where nobody held the
    winning combination (dividend 0: the pool goes to consolation combinations), are excluded (gpay = -1)."""
    tp, lab, k, ordered, _ = POOLS[pool]
    rp = pd.concat([pd.read_parquet(p, columns=["raceKey", "typePari", "libelle", "combinaison", "dividendePourUnEuro", "nombreGagnants", "miseBase"])
                    for p in sorted(RAW.glob("rapports-*.parquet"))], ignore_index=True)
    rp = rp[(rp.typePari == tp) & rp.raceKey.isin(keys)]
    bad = set(rp.raceKey[rp.libelle.str.lower().str.contains("rembours")])
    m = rp[rp.libelle.str.lower().str.strip().eq(lab)].copy()
    bad |= set(m.raceKey[m.dividendePourUnEuro <= 0])
    toks = m.combinaison.str.split("-")
    m = m[toks.map(lambda t: len(t) == k and all(x.isdigit() for x in t))]
    m = m[~m.raceKey.isin(bad)]
    m["pay"] = m.dividendePourUnEuro / 100.0
    return m, bad


def build_pool_arrays(d, st, pool):
    """W (races, 4, 3) local indices, PAY (races, 4), NW, NG (number of winning units of the first combination), gpay mask."""
    tp, lab, k, ordered, _ = POOLS[pool]
    races = d.raceKey.values[st]
    m, bad = payouts(pool, set(races))
    ridx = pd.Series(np.arange(len(st)), index=races)
    li = pd.Series(d.li.values, index=pd.MultiIndex.from_arrays([d.raceKey.values, d.numPmu.values]))
    nr = len(st)
    W = np.full((nr, 4, 3), -1, np.int64)
    PAY, NW, NG, valid = np.zeros((nr, 4)), np.zeros(nr, np.int64), np.zeros(nr), np.zeros(nr, bool)
    for rk, gr in m.groupby("raceKey", sort=False):
        r = ridx[rk]
        ok = True
        for w, (cmb, pay, ng) in enumerate(zip(gr.combinaison, gr.pay, gr.nombreGagnants)):
            if w >= 4:
                break
            try:
                loc = [int(li[(rk, int(x))]) for x in cmb.split("-")]
            except KeyError:  # a horse of the winning combination is not among our starters
                ok = False
                break
            if not ordered:
                loc = sorted(loc)
            W[r, w, :k] = loc
            PAY[r, w] = pay
            if w == 0:
                NG[r] = ng
            NW[r] = w + 1
        valid[r] = ok and NW[r] > 0
    return W, PAY, NW, NG, valid


def strengths(d, model):
    """(p, a, b) per row for an order model name like 'end_stage_ref' or 'mkt_harv_final'."""
    src, kind, v = model.split("_")
    p = d[f"{'mkt' if src == 'mkt' else 'end'}_{v}"].values
    if kind == "harv":
        return p, p, p
    col = ("mdisc" if src == "mkt" else kind) + f"_{v}"
    st, sz = starts_of(d.rid.values)

    def norm(x):
        e = np.exp(x - np.repeat(np.maximum.reduceat(x, st), sz))
        return e
    return p, norm(d[f"{col}_2"].values), norm(d[f"{col}_3"].values)


ORDER_MODELS = {"ref": ["mkt_harv_ref", "mkt_disc_ref", "end_harv_ref", "end_disc_ref", "end_full_ref", "end_stage_ref"],
                "final": ["mkt_harv_final", "mkt_disc_final", "end_harv_final", "end_disc_final", "end_full_final", "end_stage_final"]}


def day_boot(vals, day, n=1000):
    u, inv = np.unique(day, return_inverse=True)
    s, c = np.bincount(inv, vals), np.bincount(inv)
    idx = rng.integers(0, len(u), (n, len(u)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return [round(float(x), 4) for x in np.quantile(b, [.025, .975])]


def roi_summary(ret, day=None):
    ret = np.asarray(ret, float)
    if len(ret) == 0:
        return {"bets": 0}
    o = {"bets": int(len(ret)), "hits": int((ret > 0).sum()), "hit_rate": round(float((ret > 0).mean()), 4),
         "avg_payout_when_hit": round(float(ret[ret > 0].mean()), 2) if (ret > 0).any() else None, "roi": round(float(ret.mean() - 1), 4)}
    if day is not None and len(ret) >= 50:
        o["roi_ci95"] = [round(x - 1, 4) for x in day_boot(ret, day)]
    return o


def main_eval():
    d = load()
    stg = pd.read_parquet(PROC / "fr_stage.parquet")
    d = d.merge(stg, on=["rid", "numPmu"], how="left")
    d = d[d.year >= YEARS[0]].sort_values(["ts", "rid", "numPmu"]).reset_index(drop=True)
    st, sz = starts_of(d.rid.values)
    d["li"] = np.arange(len(d)) - np.repeat(st, sz)
    nr = len(st)
    yr = d.year.values[st]
    day = d.date.values[st]
    grp = np.where((yr >= DEV[0]) & (yr <= DEV[1]), 0, np.where(yr >= TEST[0], 1, -1)).astype(np.int64)
    act = np.full((nr, 3), -1, np.int64)
    rr = np.repeat(np.arange(nr), sz)
    for k in (1, 2, 3):
        ok = (d.pos == k).values & (d[f"n{k}"] == 1).values
        act[rr[ok], k - 1] = d.li.values[ok]
    xr = np.c_[d.t.values[st], np.log(sz)].astype(float)
    print(f"evaluation frame: {nr:,} races ({d.date.min().date()} .. {d.date.max().date()}); races with a clear 1-2-3: {(act >= 0).all(1).mean():.3f}", flush=True)
    rep = {"takeout_assumed_return_to_players": {p: POOLS[p][4] for p in POOLS}}
    S = {m: strengths(d, m) for v in ORDER_MODELS for m in ORDER_MODELS[v]}
    nocoef = np.array([0., 0, 0, 0, 0, 1])
    for pool, (tp, lab, k, ordered, trj) in POOLS.items():
        t0 = time.time()
        C, off = combo_table(k, ordered)
        W, PAY, NW, NG, valid = build_pool_arrays(d, st, pool)
        gpay = np.where(valid, grp, -1)
        R = {"races_with_valid_payout": {"dev": int((gpay == 0).sum()), "test": int((gpay == 1).sum())}}
        # ---------------- accuracy of each order model (finishing order, all races)
        acc, per = {}, {}
        for v, models in ORDER_MODELS.items():
            mk = S[f"mkt_harv_{v}"]
            for mname in models:
                p, a, b = S[mname]
                o = run_pool(st, sz, p, a, b, *mk, k, ordered, C, off, act, W, PAY, NW, grp, gpay, nocoef, xr, np.inf, 1.0, 1)
                per[mname] = o
                acc[mname] = {}
                for lab_g, gi in (("dev", 0), ("test", 1)):
                    sel = (grp == gi) & np.isfinite(o[0])
                    acc[mname][lab_g] = {"races": int(sel.sum()), "logloss": round(float(-np.log(o[0][sel]).mean()), 4),
                                         "top_combo_hit": round(float(o[5][sel].mean()), 4)}
        R["accuracy"] = acc
        # differences vs the market's own Harville (bootstrap by race day)
        R["vs_market"] = {}
        for v in ORDER_MODELS:
            base = per[f"mkt_harv_{v}"]
            for mname in ORDER_MODELS[v][1:]:
                o = per[mname]
                sel = (grp == 1) & np.isfinite(o[0]) & np.isfinite(base[0])
                dl = -np.log(o[0][sel]) + np.log(base[0][sel])
                dh = o[5][sel].astype(float) - base[5][sel]
                R["vs_market"][mname] = {"logloss_diff": round(float(dl.mean()), 4), "ll_ci95": day_boot(dl, day[sel]),
                                         "top_hit_diff": round(float(dh.mean()), 4), "top_ci95": day_boot(dh, day[sel])}
        # calibration of the main model's combination probabilities
        o = per["end_stage_ref"]
        cal = o[10]
        R["calibration_end_stage_ref"] = {lab_g: [{"bin": f"{PB[i]:g}-{PB[i + 1]:g}", "n": int(cal[gi, i, 0]),
                                                   "predicted": round(float(cal[gi, i, 1] / max(cal[gi, i, 0], 1)), 5),
                                                   "actual": round(float(cal[gi, i, 2] / max(cal[gi, i, 0], 1)), 5)}
                                                  for i in range(len(PB) - 1) if cal[gi, i, 0] > 0] for lab_g, gi in (("dev", 0), ("test", 1))}
        # ---------------- the pool itself: implied probability of the winning combination = return-to-players / payout
        R["pool"] = {}
        for lab_g, gi in (("dev", 0), ("test", 1)):
            sel = (gpay == gi)
            pay = PAY[sel, 0]
            blk = {"races": int(sel.sum()), "median_payout": round(float(np.median(pay)), 2),
                   "pool_logloss_at_assumed_takeout": round(float(-np.log(trj / pay).mean()), 4)}
            for mname in ("mkt_harv_ref", "end_stage_ref", "mkt_harv_final", "end_stage_final"):
                Pw = per[mname][2][sel]
                ok = np.isfinite(Pw)
                g_ = np.log(Pw[ok] * pay[ok])
                blk[mname] = {"logloss": round(float(-np.log(Pw[ok]).mean()), 4),
                              "growth_bet_all_combos_in_proportion": round(float(g_.mean()), 4), "growth_ci95": day_boot(g_, day[sel][ok]),
                              "implied_return_to_players_ipw": round(float(1 / np.mean(1 / (Pw[ok] * pay[ok]))), 4)}
            R["pool"][lab_g] = blk
        # ---------------- top combination of each model vs payout vs profit (1 unit per race)
        R["top_combo"] = {}
        for mname in ("mkt_harv_ref", "end_harv_ref", "end_stage_ref", "mkt_harv_final", "end_stage_final"):
            o = per[mname]
            R["top_combo"][mname] = {}
            for lab_g, gi in (("dev", 0), ("test", 1)):
                sel = gpay == gi
                ret, tp_, ag = o[6][sel], o[4][sel], o[9][sel].astype(bool)
                blk = {"all": roi_summary(ret, day[sel]), "same_as_market_top": roi_summary(ret[ag]), "differs_from_market_top": roi_summary(ret[~ag]),
                       "share_same": round(float(ag.mean()), 4), "by_model_prob": {}}
                edges = [0, .02, .05, .1, .2, .3, .5, 1.01] if k == 2 else [0, .01, .02, .05, .1, .2, 1.01]
                bb = np.digitize(tp_, edges) - 1
                for i in range(len(edges) - 1):
                    if (bb == i).sum():
                        blk["by_model_prob"][f"{edges[i]:g}-{min(edges[i + 1], 1):g}"] = roi_summary(ret[bb == i])
                R["top_combo"][mname][lab_g] = blk
        # ---------------- ROI of betting EVERY combination, by market price and by model edge (pari-mutuel efficiency)
        for v in ("ref", "final"):
            o = per[f"end_stage_{v}"]
            am, ar = o[11], o[12]
            R[f"roi_by_market_prob_{v}"] = {lab_g: [{"market_prob": f"{PB[i]:g}-{PB[i + 1]:g}", "combos": int(am[gi, i, 0]),
                                                     "avg_market_prob": round(float(am[gi, i, 2] / am[gi, i, 0]), 5),
                                                     "roi": round(float(am[gi, i, 1] / am[gi, i, 0] - 1), 4)}
                                                    for i in range(len(PB) - 1) if am[gi, i, 0] > 0] for lab_g, gi in (("dev", 0), ("test", 1))}
            R[f"roi_by_edge_{v}"] = {lab_g: [{"model_over_market": f"{RB[i]:g}-{RB[i + 1]:g}", "combos": int(ar[gi, i, 0]),
                                              "avg_model_prob": round(float(ar[gi, i, 2] / ar[gi, i, 0]), 5),
                                              "roi": round(float(ar[gi, i, 1] / ar[gi, i, 0] - 1), 4)}
                                             for i in range(len(RB) - 1) if ar[gi, i, 0] > 0] for lab_g, gi in (("dev", 0), ("test", 1))}
        # ---------------- value rule: payout model fit on dev winners, threshold chosen on dev, applied once to test
        R["value_rule"] = {}
        for v in ("ref", "final"):
            mname = f"end_stage_{v}"
            o = per[mname]
            sel = (gpay == 0) & np.isfinite(o[2])
            Pw, Hw, pay = o[2][sel], o[3][sel], PAY[sel, 0]
            X = np.c_[np.ones(sel.sum()), np.log(Hw), np.log(Pw / Hw), xr[sel]]
            beta, *_ = np.linalg.lstsq(X, np.log(pay), rcond=None)
            res = np.log(pay) - X @ beta
            coef = np.r_[beta, np.mean(np.exp(res))]
            p, a, b = S[mname]
            oc = run_pool(st, sz, p, a, b, *S[f"mkt_harv_{v}"], k, ordered, C, off, act, W, PAY, NW, grp, gpay, coef, xr, -0.3, 1e-4, 30_000_000)
            cr, cp, ch, cev, cret, full = oc[13:]
            cg = gpay[cr]
            cday = day[cr]
            # is the estimated EV right? realised ROI by estimated-EV bin
            eb = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.3, 1e9]
            bi_ = np.digitize(cev, eb) - 1
            ev_cal = {lab_g: [{"est_ev": f"{eb[i]:g}..{eb[i + 1]:g}", "mean_est_ev": round(float(cev[(cg == gi) & (bi_ == i)].mean()), 4)
                               if ((cg == gi) & (bi_ == i)).any() else None, **roi_summary(cret[(cg == gi) & (bi_ == i)])}
                              for i in range(len(eb) - 1)] for lab_g, gi in (("dev", 0), ("test", 1))}
            grid = [(pm, tau) for pm in (1e-3, 5e-3, 2e-2) for tau in (-0.2, -0.1, 0.0, .1, .2, .3, .5, 1.0)]
            dev_res = {}
            for pm, tau in grid:
                mm = (cg == 0) & (cp >= pm) & (cev > tau)
                dev_res[f"p>={pm:g},ev>{tau:g}"] = roi_summary(cret[mm])
            min_bets = 1000 if pool in ("CG", "TR") else 300
            okk = [(key, r_) for key, r_ in dev_res.items() if r_["bets"] >= min_bets]
            best = max(okk, key=lambda kv: kv[1]["roi"])[0] if okk else None
            test_res = {}
            for pm, tau in grid:
                mm = (cg == 1) & (cp >= pm) & (cev > tau)
                test_res[f"p>={pm:g},ev>{tau:g}"] = roi_summary(cret[mm], cday[mm])
            R["value_rule"][v] = {"payout_model": {"coef_const_logH_logEdge_trot_logField": [round(float(x), 4) for x in beta],
                                                   "smearing": round(float(coef[5]), 4), "resid_sd": round(float(res.std()), 4)},
                                  "candidates_capped": bool(full), "min_dev_bets": min_bets, "roi_by_estimated_ev": ev_cal,
                                  "dev": dev_res, "chosen_on_dev": best,
                                  "test_chosen": test_res.get(best) if best else None, "test_all": test_res}
            print(f"  {pool} value rule ({v}): chosen {best}; dev {dev_res.get(best)}; test {test_res.get(best)}", flush=True)
        rep[pool] = R
        a_ = R["accuracy"]
        print(f"{pool} ({time.time() - t0:.0f}s): test log-loss mkt_harv_ref {a_['mkt_harv_ref']['test']['logloss']} end_stage_ref "
              f"{a_['end_stage_ref']['test']['logloss']} | mkt_harv_final {a_['mkt_harv_final']['test']['logloss']} end_stage_final "
              f"{a_['end_stage_final']['test']['logloss']} | top hit {a_['mkt_harv_ref']['test']['top_combo_hit']} vs {a_['end_stage_ref']['test']['top_combo_hit']}",
              flush=True)
        print(f"   pool test: {json.dumps(R['pool']['test'])}", flush=True)
        json.dump(rep, open(RES / "exotics_report.json", "w"), indent=1)
    json.dump(rep, open(RES / "exotics_report.json", "w"), indent=1)


if __name__ == "__main__":
    {"stages": stages, "eval": main_eval}[sys.argv[1]]()
