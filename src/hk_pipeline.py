"""Hong Kong: the same method as France on a second market (replication).

  feats  pre-race features from data/processed/hk_runners.parquet (src/hk_checks.py) -> hk_feats.parquet
         programme: draw, official rating (carried into the race), weight carried, declared body weight, gear
         history (earlier races only): days since, placings, beaten lengths, time behind the winner per km, early position,
         how the market rated the horse before, class / distance / rating / body-weight changes, EW jockey and trainer stats
  dev    boosting rounds: train 1985-2009, early-stop on 2010
  wf     fundamental out-of-fold 1995..2018 (train 1985..Y-1)
  eval   stackers on the final odds (the only market here) + report; dev 1996-2016, test 2017-2018
Run: python -I src/hk_pipeline.py feats | dev | wf | eval
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_features import ew_before  # noqa: E402
from racemodel import fit_clogit, fit_gbm, pick_report, predict_clogit, predict_gbm, race_logloss, starts_of, top1_acc, topk_acc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIRST_TRAIN, OOF_YEARS, EVAL_YEARS = "1985-01-01", list(range(1995, 2019)), list(range(1996, 2019))
DEV, TEST = (1996, 2016), (2017, 2018)
rng = np.random.default_rng(0)
GBM = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=10.0, max_cat_to_onehot=8, seed=0)
RES_GBM = dict(learning_rate=0.02, num_leaves=7, min_data_in_leaf=1000, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=20.0, seed=0)
SHORT = {"NOSE": 0.05, "SH": 0.1, "HD": 0.2, "SN": 0.15, "N": 0.3, "-": 0.0, "ML": 0.0, "DH": 0.0}
NOT_FEATURES = {"rid", "race_date", "horse_id", "horse_name", "jockey_name", "trainer_name", "winning_odds", "final_placing", "finish_s",
                "length_behind_winner", "running_positions", "sections_time", "won", "over", "y", "n_winners", "pos_pct", "bl", "tbw", "early",
                "exp_win", "q", "pm", "day", "jt"}


def blen(s):
    if s is None or s != s:
        return np.nan
    s = str(s).strip().upper()
    if s in SHORT:
        return SHORT[s]
    m = re.match(r"^(\d+)?-?(\d)/(\d)$", s)
    if m:
        return (float(m.group(1)) if m.group(1) else 0.0) + float(m.group(2)) / float(m.group(3))
    try:
        return float(s.rstrip("L"))
    except ValueError:
        return np.nan


def cls_num(c):
    c = str(c).strip().upper()
    if c.isdigit():
        return float(c)
    if c in ("1&2",):
        return 1.5
    if c.startswith(("G", "HKG")) and "GRIFFIN" not in c:
        return 0.0
    return 8.0 if "GRIFFIN" in c else np.nan


def people(d, col, tag, base):
    g = d[[col, "day", "won", "exp_win"]].assign(n=1.0)
    a = g.groupby([col, "day"], sort=True)[["n", "won", "exp_win"]].sum().reset_index()
    codes = a[col].astype("category").cat.codes.values.astype(np.int64)
    v = ew_before(codes, a.day.values.astype(np.float64), a[["n", "won", "exp_win"]].values.astype(np.float64), 180.0 / np.log(2))
    a[f"{tag}_n"] = v[:, 0]
    a[f"{tag}_wr"] = (v[:, 1] + 10 * base) / (v[:, 0] + 10)
    a[f"{tag}_ae"] = (v[:, 1] + 2.0) / (v[:, 2] + 2.0)
    return d.merge(a[[col, "day", f"{tag}_n", f"{tag}_wr", f"{tag}_ae"]], on=[col, "day"], how="left")


def feats():
    d = pd.read_parquet(PROC / "hk_runners.parquet")
    d = d[d.race_date >= "1979-01-01"].copy()
    d["day"] = (d.race_date - pd.Timestamp("1979-01-01")).dt.days
    d["n_winners"] = d.groupby("rid").won.transform("sum")
    d["field"] = d.groupby("rid").horse_id.transform("size")
    d["pos_pct"] = (d.final_placing - 1) / (d.field - 1).clip(lower=1)
    d["bl"] = d.length_behind_winner.map(blen).where(d.final_placing > 1, 0.0).where(d.final_placing.notna())
    d["tbw"] = (d.finish_s - d.groupby("rid").finish_s.transform("min")) / d.distance * 1000  # seconds behind the winner per km
    d["early"] = pd.to_numeric(d.running_positions.astype(str).str.split().str[0], errors="coerce") / d.field
    d["q"] = 1 / d.winning_odds
    full = d.q.notna().groupby(d.rid).transform("all")
    d["pm"] = (d.q / d.q.groupby(d.rid).transform("sum")).where(full)
    d["exp_win"] = d.pm.fillna(1 / d.field)
    d["cls"] = d.race_class.map(cls_num)
    d["location"] = d.race_location.astype("category")
    for c in ("track", "course", "going", "gears"):
        d[c] = d[c].astype(str).astype("category")
    d["first_time_gear"] = d.gears.astype(str).str.contains("1").astype(float)
    d["month"] = d.race_date.dt.month
    d = d.sort_values(["horse_id", "race_date", "race_no"]).reset_index(drop=True)
    gh = d.groupby("horse_id", sort=False)
    d["h_n"] = gh.cumcount()
    d["days_since"] = (d.race_date - gh.race_date.shift(1)).dt.days
    d["days_since_first"] = (d.race_date - gh.race_date.transform("min")).dt.days
    for c in ("pos_pct", "bl", "tbw", "early", "exp_win", "won", "cls", "distance", "rating", "on_date_weight", "gears"):
        for lag in (1, 2, 3):
            d[f"_{c}_{lag}"] = gh[c].shift(lag)
    d["h_last_pos"], d["h_avg_pos3"] = d._pos_pct_1, d[["_pos_pct_1", "_pos_pct_2", "_pos_pct_3"]].mean(axis=1)
    d["h_last_bl"], d["h_avg_bl3"] = d._bl_1, d[["_bl_1", "_bl_2", "_bl_3"]].mean(axis=1)
    d["h_last_tbw"], d["h_best_tbw3"] = d._tbw_1, d[["_tbw_1", "_tbw_2", "_tbw_3"]].min(axis=1)
    d["h_avg_early3"] = d[["_early_1", "_early_2", "_early_3"]].mean(axis=1)
    d["h_last_mkt"], d["h_avg_mkt3"] = d._exp_win_1, d[["_exp_win_1", "_exp_win_2", "_exp_win_3"]].mean(axis=1)
    d["h_wins"] = gh.won.cumsum() - d.won
    d["h_ae"] = (d.h_wins + 1.0) / (gh.exp_win.cumsum() - d.exp_win + 1.0)
    d["h_class_change"] = d.cls - d._cls_1
    d["h_dist_change"] = d.distance - d._distance_1
    d["h_rating_change"] = d.rating - d._rating_1
    d["bw_change"] = d.on_date_weight - d._on_date_weight_1
    d["gear_change"] = (d.gears.astype(str) != d._gears_1.astype(str)).astype(float).where(d._gears_1.notna())
    d["h_same_dist"] = d.groupby(["horse_id", "distance"]).cumcount()
    d["h_same_loc"] = d.groupby(["horse_id", "race_location"]).cumcount()
    d = d.drop(columns=[c for c in d.columns if c.startswith("_")])
    base = float(1 / d.field.mean())
    d["jt"] = d.jockey_name.astype(str) + "|" + d.trainer_name.astype(str)
    for col, tag in (("jockey_name", "jky"), ("trainer_name", "trn"), ("jt", "jt")):
        d[col] = d[col].fillna("?")
        d = people(d, col, tag, base)
    rel = ["rating", "actual_weight", "draw", "on_date_weight", "h_avg_pos3", "h_avg_bl3", "h_best_tbw3", "h_avg_mkt3", "h_last_mkt", "jky_wr",
           "jky_ae", "trn_wr", "trn_ae", "days_since", "h_n", "h_ae", "h_avg_early3", "h_rating_change"]
    gr = d.groupby("rid")
    for c in rel:
        d[f"{c}_rk"] = gr[c].rank(pct=True)
        d[f"{c}_dm"] = d[c] - gr[c].transform("mean")
    d = d.drop(columns=["race_location", "race_class"])
    d = d.sort_values(["race_date", "rid", "horse_no"]).reset_index(drop=True)
    d.to_parquet(PROC / "hk_feats.parquet", index=False)
    print(f"-> hk_feats.parquet {d.shape}")


def load():
    d = pd.read_parquet(PROC / "hk_feats.parquet")
    d = d[(d.n_winners >= 1)].copy()
    d["y"] = d.won / d.n_winners
    return d.sort_values(["race_date", "rid", "horse_no"]).reset_index(drop=True)


def F_of(d):
    return [c for c in d.columns if c not in NOT_FEATURES]


def metrics(g, col):
    st, sz = starts_of(g.rid.values)
    p, y = g[col].values, g.y.values
    return {"races": int(len(st)), "logloss": round(race_logloss(p, y, st, sz), 4), "top1": round(top1_acc(p, y, st, sz), 4),
            "top3": round(topk_acc(p, y, st, sz, 3), 4)}


def boot_diff(g, a, b, n=1000):
    st, sz = starts_of(g.rid.values)
    y = g.y.values
    ll = lambda col: -np.add.reduceat(y * np.log(np.clip(g[col].values, 1e-12, 1)), st)
    rid = np.repeat(np.arange(len(st)), sz)
    hit = lambda col: (y[np.lexsort((-g[col].values, rid))[st]] > 0).astype(float)
    dl, da = ll(a) - ll(b), hit(a) - hit(b)
    day = g.race_date.values[st]
    u, inv = np.unique(day, return_inverse=True)
    sl, sa, cnt = np.bincount(inv, dl), np.bincount(inv, da), np.bincount(inv)
    idx = rng.integers(0, len(u), (n, len(u)))
    return {"logloss_diff": round(float(dl.mean()), 4), "ll_ci95": [round(float(x), 4) for x in np.quantile(sl[idx].sum(1) / cnt[idx].sum(1), [.025, .975])],
            "top1_diff": round(float(da.mean()), 4), "top1_ci95": [round(float(x), 4) for x in np.quantile(sa[idx].sum(1) / cnt[idx].sum(1), [.025, .975])]}


def main(mode):
    if mode == "feats":
        return feats()
    d = load()
    F = F_of(d)
    if mode == "dev":
        tr, va = d[(d.race_date >= FIRST_TRAIN) & (d.race_date < "2010-01-01")], d[d.race_date.dt.year == 2010]
        b = fit_gbm(tr[F], tr.y.values, tr.rid.values, GBM, 4000, valid=(va[F], va.y.values, va.rid.values, None))
        print(f"{len(F)} features; best iteration {b.best_iteration}")
        print(pd.Series(b.feature_importance("gain"), index=F).sort_values(ascending=False).head(25).round(0).to_string())
        json.dump({"best_iteration": int(b.best_iteration)}, open(RES / "hk_dev_rounds.json", "w"))
        return
    if mode == "wf":
        rounds = int(json.load(open(RES / "hk_dev_rounds.json"))["best_iteration"] * 1.1)
        out = []
        for y in OOF_YEARS:
            tr, te = d[(d.race_date >= FIRST_TRAIN) & (d.race_date < f"{y}-01-01")], d[d.race_date.dt.year == y]
            b = fit_gbm(tr[F], tr.y.values, tr.rid.values, GBM, rounds)
            p, s = predict_gbm(b, te[F], te.rid.values)
            out.append(pd.DataFrame({"rid": te.rid.values, "horse_id": te.horse_id.values, "f_score": s, "f_p": p}))
            print(f"  {y} done", flush=True)
        pd.concat(out).to_parquet(PROC / "hk_oof.parquet", index=False)
        return
    # ---------------------------------------------------------------- eval
    d = d.merge(pd.read_parquet(PROC / "hk_oof.parquet"), on=["rid", "horse_id"], how="inner")
    d = d[d.pm.notna()].sort_values(["race_date", "rid", "horse_no"]).reset_index(drop=True)
    d["year"] = d.race_date.dt.year
    d["lm"] = np.log(d.pm)
    d["m_rank"] = d.groupby("rid").pm.rank(ascending=False)
    rx = ["f_score", "lm", "m_rank", "over", "field", "location", "track", "going", "cls", "distance", "draw", "rating_dm", "h_rating_change",
          "days_since", "bw_change", "gear_change", "first_time_gear", "jky_wr", "trn_wr", "h_avg_mkt3", "h_last_pos", "actual_weight_dm"]
    for c in ("mkt_cal", "combo", "end"):
        d[c] = np.nan
    tr0, va0 = d[d.year.between(1995, 2009)], d[d.year == 2010]
    w0 = fit_clogit(np.c_[tr0.lm, tr0.f_score], tr0.y.values, tr0.rid.values)
    _, s_tr0 = predict_clogit(w0, np.c_[tr0.lm, tr0.f_score], tr0.rid.values)
    _, s_va0 = predict_clogit(w0, np.c_[va0.lm, va0.f_score], va0.rid.values)
    b0 = fit_gbm(tr0[rx], tr0.y.values, tr0.rid.values, RES_GBM, 3000, init=s_tr0, valid=(va0[rx], va0.y.values, va0.rid.values, s_va0))
    rounds = max(int(b0.best_iteration), 50)
    print(f"residual booster rounds (dev 2010): {rounds}", flush=True)
    for y in EVAL_YEARS:
        tr, te = d[d.year < y], d[d.year == y]
        w = fit_clogit(np.c_[tr.lm], tr.y.values, tr.rid.values)
        d.loc[te.index, "mkt_cal"], _ = predict_clogit(w, np.c_[te.lm], te.rid.values)
        w = fit_clogit(np.c_[tr.lm, tr.f_score], tr.y.values, tr.rid.values)
        d.loc[te.index, "combo"], s = predict_clogit(w, np.c_[te.lm, te.f_score], te.rid.values)
        _, s_tr = predict_clogit(w, np.c_[tr.lm, tr.f_score], tr.rid.values)
        b = fit_gbm(tr[rx], tr.y.values, tr.rid.values, RES_GBM, rounds, init=s_tr)
        d.loc[te.index, "end"], _ = predict_gbm(b, te[rx], te.rid.values, init=s)
    d["mkt_final"], d["fund"] = d.pm, d.f_p
    ev = d[d.year >= EVAL_YEARS[0]]
    ev[["rid", "horse_id", "race_date", "y", "winning_odds", "mkt_final", "mkt_cal", "fund", "combo", "end"]].to_parquet(PROC / "hk_preds.parquet", index=False)
    M = ["mkt_final", "mkt_cal", "fund", "combo", "end"]
    rep = {"residual_rounds": rounds, "combo_weights_last_fit": [round(float(x), 3) for x in w]}
    for lab, yrs in (("dev_1996_2016", DEV), ("test_2017_2018", TEST)):
        g = ev[ev.year.between(*yrs)]
        rep[lab] = {"overall": {m: metrics(g, m) for m in M}, "vs_market": {m: boot_diff(g, m, "mkt_final") for m in ("fund", "combo", "end", "mkt_cal")}}
    rep["top_pick"] = {lab: {m: pick_report(ev[ev.year.between(*yrs)], m, "winning_odds", "mkt_final") for m in M}
                       for lab, yrs in (("dev_1996_2016", DEV), ("test_2017_2018", TEST))}
    rep["per_period"] = {f"{a}-{b}": {m: metrics(ev[ev.year.between(a, b)], m) for m in M} for a, b in ((1996, 2000), (2001, 2005), (2006, 2010), (2011, 2016), (2017, 2018))}
    # pari-mutuel check: final odds are only known at the off; this is an upper bound (decide with final odds)
    t, dv = ev[ev.year.between(*TEST)], ev[ev.year.between(*DEV)]
    def bet(g, tt):
        e = g.end * g.winning_odds - 1
        b_ = g[(e > tt) & (e < 1.0)]
        prof = np.where(b_.y > 0, b_.winning_odds - 1, -1.0)
        return {"bets": int(len(b_)), "roi": round(float(prof.mean()), 4) if len(b_) else None}
    rep["betting_upper_bound"] = {"dev": {str(tt): bet(dv, tt) for tt in (0.0, 0.1, 0.2, 0.3)}, "test": {str(tt): bet(t, tt) for tt in (0.0, 0.1, 0.2, 0.3)}}
    json.dump(rep, open(RES / "hk_report.json", "w"), indent=1)
    for lab in ("dev_1996_2016", "test_2017_2018"):
        print(f"\n{lab}")
        for m, v in rep[lab]["overall"].items():
            print(f"  {m:10s} races {v['races']:6,}  log-loss {v['logloss']:.4f}  top-1 {v['top1']:.4f}  top-3 {v['top3']:.4f}")
        for m, v in rep[lab]["vs_market"].items():
            print(f"  {m:10s} vs market: log-loss {v['logloss_diff']:+.4f} {v['ll_ci95']}  top-1 {v['top1_diff']:+.4f} {v['top1_ci95']}")
    print(json.dumps(rep["per_period"], indent=0)[:3000])
    print(json.dumps(rep["betting_upper_bound"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "feats")
