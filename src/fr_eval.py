"""France: market stackers on top of the out-of-fold fundamental model, and the accuracy report.

Models (all walk-forward by year; stackers trained only on earlier years' out-of-fold predictions):
  mkt_final      final pool odds, margin removed proportionally          (the benchmark; known only at the off)
  mkt_ref        reference odds ~30 min before the off, same treatment
  mkt_ref_cal    conditional logit on log(ref prob) (x trot): corrects the favourite-longshot bias
  fund           fundamental model only (never sees this race's odds)
  combo_ref      conditional logit on [log ref prob, fundamental score] (x trot)  -- Benter-style
  end_ref        LightGBM boosted from combo_ref (learns where the combination is still wrong)   = END MODEL (usable 30 min out)
  combo_final / end_final   same with the final odds (+ ref->final move): upper bound "does anything beat the final market?"

dev = 2018-2024 (stacker rounds picked on 2022 only), test = 2025-01-01 .. 2026-05 (looked at once).
Run: python -I src/fr_eval.py   -> results/fr_report.json, results/fr_report.md, data/processed/fr_preds.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from racemodel import fit_clogit, fit_gbm, group_softmax, pick_report, predict_clogit, predict_gbm, race_logloss, starts_of, top1_acc, topk_acc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
EVAL_YEARS = list(range(2018, 2027))
DEV, TEST = (2018, 2024), (2025, 2026)
rng = np.random.default_rng(0)
RES_GBM = dict(learning_rate=0.02, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
               lambda_l2=20.0, max_cat_to_onehot=8, seed=0)
RESID_FEATS = ["f_score", "lmr", "mr_rank", "mkt_overr", "field", "specialite", "discipline", "quinte", "autostart", "hour", "avisEntraineur",
               "deferre", "supplement", "driverChange", "days_since", "mus_avg3", "h_avg_mkt3", "h_last_mkt", "drv_wr", "trn_wr", "drv_ae", "trn_ae",
               "earn_per_start_rk", "win_rate_rk", "numPmu", "dist_handicap", "placeCorde", "handicapValeur_dm", "prize_log", "numReunion"]


def load():
    cols = ["rid", "numPmu", "date", "ts", "won", "n_winners", "mkt_pf", "mkt_pr", "mkt_qf", "mkt_qr", "mkt_overf", "mkt_overr", "odf_rapport",
            "odr_rapport", "reunionKey"] + [c for c in RESID_FEATS if c not in ("f_score", "lmr", "mr_rank")]
    d = pd.read_parquet(PROC / "fr_feats.parquet", columns=list(dict.fromkeys(cols)))
    d = d[d.n_winners >= 1]
    d["y"] = d.won / d.n_winners
    d = d.merge(pd.read_parquet(PROC / "fr_oof.parquet"), on=["rid", "numPmu"], how="inner")
    full = d.mkt_pf.notna() & d.mkt_pr.notna()
    d = d[full.groupby(d.rid).transform("all")]  # races where both markets are complete
    d = d.sort_values(["ts", "rid", "numPmu"]).reset_index(drop=True)
    d["year"] = d.date.dt.year
    d["is_trot"] = d.specialite.astype(str).str.startswith("TROT").astype(float)
    d["lmf"], d["lmr"] = np.log(d.mkt_pf), np.log(d.mkt_pr)
    d["move"] = d.lmf - d.lmr
    d["mr_rank"] = d.groupby("rid").mkt_pr.rank(ascending=False)
    d["mf_rank"] = d.groupby("rid").mkt_pf.rank(ascending=False)
    return d


def Z(d, kind):
    t = d.is_trot.values
    if kind == "mkt_ref_cal":
        return np.c_[d.lmr, d.lmr * t]
    if kind == "mkt_final_cal":
        return np.c_[d.lmf, d.lmf * t]
    if kind == "combo_ref":
        return np.c_[d.lmr, d.lmr * t, d.f_score, d.f_score * t]
    if kind == "combo_final":
        return np.c_[d.lmf, d.lmf * t, d.f_score, d.f_score * t, d.lmr, d.lmr * t]
    raise ValueError(kind)


def resid_X(d, final):
    X = d[RESID_FEATS].copy()
    if final:
        X["lmf"], X["mf_rank"], X["move"], X["mkt_overf"] = d.lmf, d.mf_rank, d.move, d.mkt_overf
    return X


def walk(d, rounds):
    for c in ("mkt_ref_cal", "mkt_final_cal", "combo_ref", "combo_final", "end_ref", "end_final"):
        d[c] = np.nan
    for y in EVAL_YEARS:
        tr, te = d[d.year < y], d[d.year == y]
        if te.empty:
            continue
        for k in ("mkt_ref_cal", "mkt_final_cal", "combo_ref", "combo_final"):
            w = fit_clogit(Z(tr, k), tr.y.values, tr.rid.values)
            d.loc[te.index, k], s = predict_clogit(w, Z(te, k), te.rid.values)
            if k in ("combo_ref", "combo_final"):
                _, s_tr = predict_clogit(w, Z(tr, k), tr.rid.values)
                fin = k == "combo_final"
                b = fit_gbm(resid_X(tr, fin), tr.y.values, tr.rid.values, RES_GBM, rounds[k], init=s_tr)
                d.loc[te.index, "end_final" if fin else "end_ref"], _ = predict_gbm(b, resid_X(te, fin), te.rid.values, init=s)
        print(f"  {y} done", flush=True)
    return d


def tune_rounds(d):
    """Residual-booster rounds: train on OOF 2017-2021, early-stop on 2022 (dev only)."""
    out = {}
    tr, va = d[d.year.between(2017, 2021)], d[d.year == 2022]
    for k in ("combo_ref", "combo_final"):
        w = fit_clogit(Z(tr, k), tr.y.values, tr.rid.values)
        _, s_tr = predict_clogit(w, Z(tr, k), tr.rid.values)
        _, s_va = predict_clogit(w, Z(va, k), va.rid.values)
        fin = k == "combo_final"
        b = fit_gbm(resid_X(tr, fin), tr.y.values, tr.rid.values, RES_GBM, 3000, init=s_tr,
                    valid=(resid_X(va, fin), va.y.values, va.rid.values, s_va))
        out[k] = max(int(b.best_iteration), 50)
        print(f"  residual booster on {k}: {out[k]} rounds", flush=True)
    return out


MODELS = ["mkt_final", "mkt_ref", "mkt_ref_cal", "mkt_final_cal", "fund", "combo_ref", "end_ref", "combo_final", "end_final"]


def metrics(g, col):
    st, sz = starts_of(g.rid.values)
    p, y = g[col].values, g.y.values
    return {"races": int(len(st)), "logloss": round(race_logloss(p, y, st, sz), 4), "top1": round(top1_acc(p, y, st, sz), 4),
            "top3": round(topk_acc(p, y, st, sz, 3), 4)}


def boot_diff(g, a, b, n=1000):
    """Bootstrap by race day: CI of (log-loss a - log-loss b) and (top-1 a - top-1 b)."""
    st, sz = starts_of(g.rid.values)
    y = g.y.values
    ll = lambda col: -np.add.reduceat(y * np.log(np.clip(g[col].values, 1e-12, 1)), st)
    def first_hit(col):
        rid = np.repeat(np.arange(len(st)), sz)
        order = np.lexsort((-g[col].values, rid))
        return (y[order[st]] > 0).astype(float)
    dl, da = ll(a) - ll(b), first_hit(a) - first_hit(b)
    day = g.date.values[st]
    u, inv = np.unique(day, return_inverse=True)
    sl, sa, cnt = np.bincount(inv, dl), np.bincount(inv, da), np.bincount(inv)
    idx = rng.integers(0, len(u), (n, len(u)))
    bl = sl[idx].sum(1) / cnt[idx].sum(1)
    ba = sa[idx].sum(1) / cnt[idx].sum(1)
    return {"logloss_diff": round(float(dl.mean()), 4), "ll_ci95": [round(float(x), 4) for x in np.quantile(bl, [.025, .975])],
            "top1_diff": round(float(da.mean()), 4), "top1_ci95": [round(float(x), 4) for x in np.quantile(ba, [.025, .975])]}


def calibration(g, col):
    bins = pd.cut(g[col], [0, .02, .05, .1, .15, .2, .3, .4, .6, 1])
    c = g.groupby(bins, observed=True).agg(n=("y", "size"), predicted=(col, "mean"), actual=("y", "mean")).round(4)
    return c.reset_index().astype({col: str}).to_dict(orient="records")


def betting(g, col, t):
    """Pari-mutuel check: decide 30 min out (model prob x reference odds > 1 + t), get paid the FINAL odds."""
    ev = g[col] * g.odr_rapport - 1
    b = g[(ev > t) & (ev < 1.0)]
    prof = np.where(b.y > 0, b.odf_rapport * b.y * b.n_winners - 1, -1.0)  # dead heat pays at the published odds share
    return {"bets": int(len(b)), "roi": round(float(prof.mean()), 4) if len(b) else None, "hit": round(float((b.y > 0).mean()), 4) if len(b) else None,
            "avg_odds": round(float(b.odf_rapport.mean()), 2) if len(b) else None}


def main():
    d = load()
    print(f"races with both markets and out-of-fold predictions: {d.rid.nunique():,} ({d.date.min().date()} .. {d.date.max().date()})", flush=True)
    rounds = tune_rounds(d)
    d = walk(d, rounds)
    d["fund"] = d.f_p
    d["mkt_final"], d["mkt_ref"] = d.mkt_pf, d.mkt_pr
    ev = d[d.year >= EVAL_YEARS[0]].copy()
    ev[["rid", "numPmu", "date", "y", "odf_rapport", "odr_rapport", "specialite"] + MODELS].to_parquet(PROC / "fr_preds.parquet", index=False)
    rep = {"rounds": rounds}
    for lab, yrs in (("dev_2018_2024", DEV), ("test_2025_2026", TEST)):
        g = ev[ev.year.between(*yrs)]
        rep[lab] = {"overall": {m: metrics(g, m) for m in MODELS},
                    "vs_market_final": {m: boot_diff(g, m, "mkt_final") for m in ("fund", "end_ref", "end_final", "combo_ref")},
                    "vs_market_ref": {m: boot_diff(g, m, "mkt_ref") for m in ("fund", "end_ref", "mkt_ref_cal")},
                    "by_specialite": {str(s): {m: metrics(gs, m) for m in ("mkt_final", "mkt_ref", "fund", "end_ref", "end_final")}
                                      for s, gs in g.groupby("specialite", observed=True) if gs.rid.nunique() >= 500}}
    # top pick of each model, 1 unit per race, paid at the FINAL pool odds (pari-mutuel: you always get the final odds)
    rep["top_pick"] = {lab: {m: pick_report(ev[ev.year.between(*yrs)], m, "odf_rapport", "mkt_pf") for m in MODELS}
                       for lab, yrs in (("dev_2018_2024", DEV), ("test_2025_2026", TEST))}
    rep["per_year"] = {int(y): {m: metrics(gy, m) for m in ("mkt_final", "mkt_ref", "fund", "end_ref", "end_final")} for y, gy in ev.groupby("year")}
    t = ev[ev.year.between(*TEST)]
    rep["calibration_test_end_ref"] = calibration(t, "end_ref")
    # betting sanity check: threshold on dev, applied once on test
    grid = (0.0, 0.05, 0.1, 0.2, 0.3)
    dv = ev[ev.year.between(*DEV)]
    bd = {tt: betting(dv, "end_ref", tt) for tt in grid}
    best = max((tt for tt in grid if bd[tt]["bets"] >= 2000), key=lambda tt: bd[tt]["roi"], default=0.0)
    rep["betting_parimutuel"] = {"dev": {str(k): v for k, v in bd.items()}, "threshold_chosen_on_dev": best,
                                 "test": betting(t, "end_ref", best), "test_all_thresholds": {str(tt): betting(t, "end_ref", tt) for tt in grid},
                                 "test_mkt_ref_cal_same_rule": betting(t, "mkt_ref_cal", best)}
    json.dump(rep, open(RES / "fr_report.json", "w"), indent=1)
    for lab in ("dev_2018_2024", "test_2025_2026"):
        print(f"\n{lab}")
        for m, v in rep[lab]["overall"].items():
            print(f"  {m:14s} races {v['races']:6,}  log-loss {v['logloss']:.4f}  top-1 {v['top1']:.4f}  top-3 {v['top3']:.4f}")
        for k in ("vs_market_final", "vs_market_ref"):
            for m, v in rep[lab][k].items():
                print(f"  {m:12s} {k}: log-loss {v['logloss_diff']:+.4f} {v['ll_ci95']}  top-1 {v['top1_diff']:+.4f} {v['top1_ci95']}")
    print("\nbetting (pari-mutuel, decide at ref odds, paid final odds):", json.dumps(rep["betting_parimutuel"], indent=1))
    print("\nTOP PICK, 1 unit per race at final odds (test):")
    for m, v in rep["top_pick"]["test_2025_2026"].items():
        a, dis = v["all"], v["differs_from_market_favourite"]
        print(f"  {m:14s} hit {a['hit_rate']:.3f} avg odds {a['avg_odds']:5.2f} ROI {a['roi']:+.3f} | agrees with fav {v['share_agreeing']:.2f}; "
              f"when it differs: n={dis.get('n', 0)} hit {dis.get('hit_rate', 0):.3f} odds {dis.get('avg_odds', 0):.2f} ROI {dis.get('roi', 0):+.3f}")


if __name__ == "__main__":
    main()
