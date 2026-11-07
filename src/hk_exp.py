"""Hong Kong replication of the fundamental-model experiments (src/fr_exp.py). Dev only: stack evaluated 2000-2016; 2017-18 never read.

  ratings        Plackett-Luce ratings of horse, jockey, trainer (same code as France), tuned on 1990-1994 -> hk_ratings.parquet
  wf NAME        walk-forward out-of-fold 1995..2018 (train 1985..Y-1), rounds tuned on 2010 -> data/processed/exp/hk_NAME.parquet
                 NAME = win | pl3 [+rt]
  stack N1 ..    dev log-loss of the market stack (conditional logit on [log final-odds prob, scores]), walk-forward 2000-2016
Run: python -I src/hk_exp.py ratings | wf win+rt pl3+rt | stack base win+rt pl3+rt
"""
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_exp import EXP, pl_objective, pl_ratings  # noqa: E402
from hk_pipeline import F_of, FIRST_TRAIN, GBM, OOF_YEARS, load  # noqa: E402
from racemodel import fit_clogit, group_softmax, race_logloss, starts_of, top1_acc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
DEV_YEARS = list(range(2000, 2017))
RT = ["rt_h", "rt_d", "rt_t", "rt_nh", "rt_sum", "rt_sum_dm", "rt_sum_rk", "rt_p", "rt_h_dm"]
rng = np.random.default_rng(0)


def ratings():
    d = pd.read_parquet(PROC / "hk_feats.parquet", columns=["rid", "horse_id", "race_date", "race_no", "day", "jockey_name", "trainer_name",
                                                            "final_placing", "won", "n_winners"])
    d = d.sort_values(["day", "race_no", "rid", "horse_id"]).reset_index(drop=True)
    ent = np.column_stack([d[c].astype(str).where(d[c].astype(str).ne("?")).astype("category").cat.codes.values
                           for c in ("horse_id", "jockey_name", "trainer_name")]).astype(np.int64)
    st, sz = starts_of(d.rid.values)
    pos = d.final_placing.fillna(0).values.astype(np.float64)
    rday = d.day.values[st].astype(np.float64)
    y = (d.won / d.n_winners.clip(lower=1)).fillna(0).values
    yr = d.race_date.dt.year.values[st]
    ok = (yr >= 1990) & (yr <= 1994) & (np.add.reduceat(y, st) > 0)
    best = None
    for eh in (0.6, 0.8, 1.2, 2.0):
        for ed in (0.1, 0.2, 0.35):
            for rho in (0.8, 1.0):
                for kmax in (4, 6, 14):
                    out = pl_ratings(st, sz, rday, ent, pos, np.array([eh, ed, ed]), np.array([5.0, 50.0, 50.0]),
                                     np.array([730.0, 365.0, 365.0]), rho, kmax)
                    p = group_softmax(out[:, :3].sum(1), st, sz)
                    ll = float(-np.add.reduceat(y * np.log(np.clip(p, 1e-12, 1)), st)[ok].mean())
                    if best is None or ll < best[0]:
                        best = (ll, eh, ed, rho, kmax)
    ll, eh, ed, rho, kmax = best
    print(f"chosen eta_h {eh} eta_people {ed} rho {rho} kmax {kmax}: winner log-loss 1990-94 {ll:.4f}", flush=True)
    out = pl_ratings(st, sz, rday, ent, pos, np.array([eh, ed, ed]), np.array([5.0, 50.0, 50.0]), np.array([730.0, 365.0, 365.0]), rho, kmax)
    f = pd.DataFrame({"rid": d.rid.values, "horse_id": d.horse_id.values, "rt_h": out[:, 0], "rt_d": out[:, 1], "rt_t": out[:, 2],
                      "rt_nh": np.log1p(out[:, 3])})
    f["rt_sum"] = f.rt_h + f.rt_d + f.rt_t
    g = f.groupby("rid")
    f["rt_sum_dm"] = f.rt_sum - g.rt_sum.transform("mean")
    f["rt_h_dm"] = f.rt_h - g.rt_h.transform("mean")
    f["rt_sum_rk"] = g.rt_sum.rank(pct=True)
    f["rt_p"] = group_softmax(f.rt_sum.values, st, sz)
    f.to_parquet(PROC / "hk_ratings.parquet", index=False)
    json.dump({"eta_horse": eh, "eta_people": ed, "rho": rho, "kmax": kmax, "logloss_1990_94": ll}, open(RES / "exp_hk_ratings.json", "w"))


def wf(name):
    t0 = time.time()
    objective, _, featset = name.partition("+")
    d = load()
    F = F_of(d)
    if featset == "rt":
        d = d.merge(pd.read_parquet(PROC / "hk_ratings.parquet"), on=["rid", "horse_id"], how="left")
        d = d.sort_values(["race_date", "rid", "horse_no"]).reset_index(drop=True)
        F = F + RT
    K = {"win": 1, "pl3": 3}[objective]

    def train(g, rounds, valid=None):
        st, sz = starts_of(g.rid.values)
        p = {**GBM, "verbose": -1, "num_threads": int(os.environ.get("NTHREADS", 28)),
             "objective": pl_objective(st, sz, g.y.values, g.final_placing.fillna(0).values, K, 0.7)}
        ds = lgb.Dataset(g[F], g.y.values, free_raw_data=False)
        if valid is None:
            return lgb.train(p, ds, num_boost_round=rounds)
        sv, zv = starts_of(valid.rid.values)
        dv = lgb.Dataset(valid[F], valid.y.values, reference=ds, free_raw_data=False)
        fe = lambda preds, data: ("race_ll", race_logloss(group_softmax(preds, sv, zv), valid.y.values, sv, zv), False)
        return lgb.train(p, ds, num_boost_round=rounds, valid_sets=[dv], feval=fe, callbacks=[lgb.early_stopping(150, verbose=False)])
    b = train(d[(d.race_date >= FIRST_TRAIN) & (d.race_date < "2010-01-01")], 4000, valid=d[d.race_date.dt.year == 2010])
    rounds = int(b.best_iteration * 1.1)
    print(f"hk {name}: {len(F)} features, best iteration {b.best_iteration} -> {rounds}", flush=True)
    parts = []
    for y in OOF_YEARS:
        tr, te = d[(d.race_date >= FIRST_TRAIN) & (d.race_date < f"{y}-01-01")], d[d.race_date.dt.year == y]
        bb = train(tr, rounds)
        parts.append(pd.DataFrame({"rid": te.rid.values, "horse_id": te.horse_id.values, "f_score": bb.predict(te[F], raw_score=True)}))
    pd.concat(parts).to_parquet(EXP / f"hk_{name}.parquet", index=False)
    print(f"-> exp/hk_{name}.parquet ({time.time() - t0:.0f}s)", flush=True)


def stack(names):
    d = pd.read_parquet(PROC / "hk_feats.parquet", columns=["rid", "horse_id", "race_date", "won", "n_winners", "pm"])
    d = d[d.n_winners >= 1]
    d["y"] = d.won / d.n_winners
    for nm in names:
        v1 = PROC / "hk_oof_v1.parquet"
        src = (v1 if v1.exists() else PROC / "hk_oof.parquet") if nm == "base" else EXP / f"hk_{nm}.parquet"
        d = d.merge(pd.read_parquet(src)[["rid", "horse_id", "f_score"]].rename(columns={"f_score": f"s_{nm}"}), on=["rid", "horse_id"])
    d = d[d.pm.notna().groupby(d.rid).transform("all")].sort_values(["race_date", "rid"]).reset_index(drop=True)
    d["year"] = d.race_date.dt.year
    d = d[d.year <= DEV_YEARS[-1]]
    lm = np.log(d.pm.values)
    configs = {f"fund:{nm}": [d[f"s_{nm}"].values] for nm in names}
    configs.update({f"mkt+{nm}": [lm, d[f"s_{nm}"].values] for nm in names})
    if len(names) > 1:
        configs["mkt+all"] = [lm] + [d[f"s_{nm}"].values for nm in names]
    st, sz = starts_of(d.rid.values)
    yr, day, y = d.year.values[st], d.race_date.values[st], d.y.values
    ll = {}
    for cname, xs in configs.items():
        Z = np.column_stack(xs)
        s = np.zeros(len(d))
        for Y in DEV_YEARS:
            trm, tem = (d.year < Y).values & (d.year >= 1996).values, (d.year == Y).values
            s[tem] = Z[tem] @ fit_clogit(Z[trm], y[trm], d.rid.values[trm])
        p = group_softmax(s, st, sz)
        ll[cname] = -np.add.reduceat(y * np.log(np.clip(p, 1e-12, 1)), st)
    ok = np.isin(yr, DEV_YEARS)
    u, inv = np.unique(day[ok], return_inverse=True)
    idx = rng.integers(0, len(u), (1000, len(u)))
    cnt = np.bincount(inv)
    out = {}
    for cname in configs:
        kind = cname.split(":")[0].split("+")[0]
        ref = f"{kind}:{names[0]}" if kind == "fund" else f"{kind}+{names[0]}"
        dl = ll[cname][ok] - ll[ref][ok]
        b = np.bincount(inv, dl)[idx].sum(1) / cnt[idx].sum(1)
        out[cname] = {"dev_logloss": round(float(ll[cname][ok].mean()), 5), "diff_vs_first": round(float(dl.mean()), 5),
                      "ci95": [round(float(x), 5) for x in np.quantile(b, [.025, .975])]}
        print(f"  hk {cname:20s} dev log-loss {out[cname]['dev_logloss']:.5f}  diff {out[cname]['diff_vs_first']:+.5f} {out[cname]['ci95']}", flush=True)
    print(f"  ({ok.sum():,} dev races)")
    allr = json.load(open(RES / "exp_hk_stack.json")) if (RES / "exp_hk_stack.json").exists() else {}
    allr[" ".join(names)] = out
    json.dump(allr, open(RES / "exp_hk_stack.json", "w"), indent=1)


def blend(names):
    """Walk-forward blend of fundamental variants (conditional logit on the scores, earlier years only; first year in-sample, it is
    only ever stacker training data) -> hk_oof.parquet (previous kept as hk_oof_v1.parquet)."""
    import shutil
    d = pd.read_parquet(PROC / "hk_feats.parquet", columns=["rid", "horse_id", "race_date", "won", "n_winners"])
    d = d[d.n_winners >= 1]
    d["y"] = d.won / d.n_winners
    for nm in names:
        src = PROC / "hk_oof_v1.parquet" if nm == "base" else EXP / f"hk_{nm}.parquet"
        if nm == "base" and not src.exists():
            src = PROC / "hk_oof.parquet"
        d = d.merge(pd.read_parquet(src)[["rid", "horse_id", "f_score"]].rename(columns={"f_score": f"s_{nm}"}), on=["rid", "horse_id"], how="left")
    d = d[d[f"s_{names[0]}"].notna()].sort_values(["race_date", "rid"]).reset_index(drop=True)
    d["year"] = d.race_date.dt.year
    S = d[[f"s_{nm}" for nm in names]].values
    race_all = pd.Series(~np.isnan(S).any(1)).groupby(d.rid.values).transform("all").values
    out = np.zeros(len(d))
    for Y in sorted(d.year.unique()):
        te = (d.year == Y).values
        trm = (d.year < Y).values if Y > d.year.min() else te
        for cols, mask in ((list(range(len(names))), race_all), ([0], ~race_all)):
            if not (te & mask).any():
                continue
            w = fit_clogit(S[trm & race_all][:, cols], d.y.values[trm & race_all], d.rid.values[trm & race_all])
            out[te & mask] = np.nan_to_num(S[te & mask][:, cols]) @ w
        if Y % 4 == 0:
            print(f"  {Y}: weights {np.round(w, 3).tolist()}", flush=True)
    st, sz = starts_of(d.rid.values)
    if not (PROC / "hk_oof_v1.parquet").exists():
        shutil.copy(PROC / "hk_oof.parquet", PROC / "hk_oof_v1.parquet")
    pd.DataFrame({"rid": d.rid.values, "horse_id": d.horse_id.values, "f_score": out, "f_p": group_softmax(out, st, sz)}).to_parquet(
        PROC / "hk_oof.parquet", index=False)
    print(f"-> hk_oof.parquet {len(d):,} rows", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "blend":
        blend(sys.argv[2:])
        sys.exit()
    if cmd == "ratings":
        ratings()
    elif cmd == "wf":
        for nm in sys.argv[2:]:
            wf(nm)
    else:
        stack(sys.argv[2:])
