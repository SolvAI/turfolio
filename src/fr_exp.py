"""Experiments on the fundamental model. Selection uses the dev years only (stack evaluated on 2018-2024); test is never read here.

  ratings         opponent-adjusted Plackett-Luce ratings of horse, driver/jockey and trainer, updated after each race DAY from the
                  full finishing order (features of day D use results of days < D only) -> data/processed/fr_ratings.parquet
  wf NAME         walk-forward out-of-fold scores of a fundamental-model variant (rounds tuned on 2022, like fr_model.py)
                  -> data/processed/exp/NAME.parquet.  NAME = <objective>[+<feature set>]: objective win | pl3 | pl5,
                  feature sets: base, rt (base + ratings)
  stack N1 [N2..] dev log-loss of the market stack (conditional logit on [log market prob, scores] x trot), walk-forward 2018-2024,
                  each variant alone and all together, with day-bootstrap CIs against the first one
Run: python -I src/fr_exp.py ratings | wf pl3+rt | stack base win+rt pl3+rt
"""
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_model import FIRST_TRAIN, GBM, NOT_FEATURES, OOF_YEARS, features  # noqa: E402
from racemodel import fit_clogit, group_softmax, predict_clogit, race_logloss, starts_of, top1_acc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
EXP = PROC / "exp"
EXP.mkdir(exist_ok=True)
DEV_YEARS = list(range(2018, 2025))
rng = np.random.default_rng(0)
RT_COLS = ["rt_h", "rt_d", "rt_t", "rt_nh", "rt_sum", "rt_sum_dm", "rt_sum_rk", "rt_p", "rt_h_dm"]


# ------------------------------------------------------------------------------------------------ ratings
@njit(cache=True)
def pl_ratings(rst, rsz, rday, ent, pos, eta, ns, tau, rho, kmax):
    """ent (rows, 3) entity codes (-1 = unknown); pos (rows,) finishing position (0 = not classified).
    Returns (rows, 6): pre-race ratings of the 3 entities and their numbers of earlier races."""
    nrow = len(pos)
    ne = np.zeros(3, np.int64)
    for e in range(3):
        ne[e] = ent[:, e].max() + 1
    R = np.zeros((3, ne.max()))
    N = np.zeros((3, ne.max()))
    last = np.zeros((3, ne.max()))
    out = np.zeros((nrow, 6))
    nr = len(rst)
    r = 0
    while r < nr:
        r2 = r
        while r2 < nr and rday[r2] == rday[r]:
            r2 += 1
        for q in range(r, r2):
            for i in range(rst[q], rst[q] + rsz[q]):
                for e in range(3):
                    c = ent[i, e]
                    if c < 0:
                        continue
                    if N[e, c] > 0:
                        R[e, c] *= np.exp(-(rday[q] - last[e, c]) / tau[e])
                    last[e, c] = rday[q]
                    out[i, e] = R[e, c]
                    out[i, 3 + e] = N[e, c]
        for q in range(r, r2):
            o, n = rst[q], rsz[q]
            s = out[o:o + n, 0] + out[o:o + n, 1] + out[o:o + n, 2]
            act = np.ones(n, np.bool_)
            g = np.zeros(n)
            order = np.argsort(np.where(pos[o:o + n] > 0, pos[o:o + n], 1e9))
            nfin = 0
            for i in range(n):
                if pos[o + i] > 0:
                    nfin += 1
            w = 1.0
            for k in range(min(nfin, kmax)):
                wi = order[k]
                mx = -1e18
                for i in range(n):
                    if act[i] and s[i] > mx:
                        mx = s[i]
                z = 0.0
                for i in range(n):
                    if act[i]:
                        z += np.exp(s[i] - mx)
                for i in range(n):
                    if act[i]:
                        g[i] -= w * np.exp(s[i] - mx) / z
                g[wi] += w
                act[wi] = False
                w *= rho
            for i in range(n):
                for e in range(3):
                    c = ent[o + i, e]
                    if c < 0:
                        continue
                    R[e, c] += eta[e] / np.sqrt(1.0 + N[e, c] / ns[e]) * g[i]
                    N[e, c] += 1
        r = r2
    return out


def rating_inputs():
    d = pd.read_parquet(PROC / "fr_feats.parquet", columns=["rid", "numPmu", "ts", "day", "hk", "driver", "entraineur", "pos", "won", "n_winners"])
    d = d.sort_values(["day", "ts", "rid", "numPmu"]).reset_index(drop=True)
    ent = np.column_stack([d[c].where(d[c].ne("?")).astype("category").cat.codes.values for c in ("hk", "driver", "entraineur")]).astype(np.int64)
    st, sz = starts_of(d.rid.values)
    return d, ent, st, sz


def rating_logloss(d, out, st, sz, years):
    s = out[:, :3].sum(1)
    yr = d.ts.dt.year.values
    p = group_softmax(s, st, sz)
    y = (d.won / d.n_winners.clip(lower=1)).fillna(0).values
    ok = np.isin(yr[st], years) & (np.add.reduceat(y, st) > 0)
    ll = -np.add.reduceat(y * np.log(np.clip(p, 1e-12, 1)), st)
    return float(ll[ok].mean())


def ratings():
    t0 = time.time()
    d, ent, st, sz = rating_inputs()
    pos = d.pos.fillna(0).values.astype(np.float64)
    rday = d.day.values[st].astype(np.float64)
    # small tuning on 2016-2017 (before any dev year of the stack): learning rates, shrink, decay, stage weights
    best = None
    grid = []
    for eh in (0.8, 1.2, 2.0):
        for ed in (0.2, 0.35, 0.6):
            for rho in (1.0, 0.8):
                for kmax in (3,):
                    grid.append((eh, ed, rho, kmax))
    for eh, ed, rho, kmax in grid:
        out = pl_ratings(st, sz, rday, ent, pos, np.array([eh, ed, ed]), np.array([5.0, 50.0, 50.0]), np.array([730.0, 365.0, 365.0]), rho, kmax)
        ll = rating_logloss(d, out, st, sz, [2016, 2017])
        print(f"  eta_h {eh} eta_people {ed} rho {rho} kmax {kmax}: winner log-loss 2016-17 {ll:.4f}", flush=True)
        if best is None or ll < best[0]:
            best = (ll, eh, ed, rho, kmax)
    ll, eh, ed, rho, kmax = best
    print(f"chosen {best} ({time.time() - t0:.0f}s)", flush=True)
    out = pl_ratings(st, sz, rday, ent, pos, np.array([eh, ed, ed]), np.array([5.0, 50.0, 50.0]), np.array([730.0, 365.0, 365.0]), rho, kmax)
    f = pd.DataFrame({"rid": d.rid.values, "numPmu": d.numPmu.values, "rt_h": out[:, 0], "rt_d": out[:, 1], "rt_t": out[:, 2],
                      "rt_nh": np.log1p(out[:, 3])})
    f["rt_sum"] = f.rt_h + f.rt_d + f.rt_t
    g = f.groupby("rid")
    f["rt_sum_dm"] = f.rt_sum - g.rt_sum.transform("mean")
    f["rt_h_dm"] = f.rt_h - g.rt_h.transform("mean")
    f["rt_sum_rk"] = g.rt_sum.rank(pct=True)
    f["rt_p"] = group_softmax(f.rt_sum.values, st, sz)
    f.to_parquet(PROC / "fr_ratings.parquet", index=False)
    json.dump({"eta_horse": eh, "eta_people": ed, "rho": rho, "kmax": kmax, "dev_logloss_2016_17": ll}, open(RES / "exp_ratings.json", "w"))
    print(f"-> fr_ratings.parquet ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ fundamental-model variants
def pl_objective(st, sz, y, pos, K, rho):
    """Plackett-Luce over the first K places: stage 1 = winner (dead heats shared, as the win model), stage s >= 2 = s-th place among the
    horses not yet placed, used only when places 1..s are each held by exactly one horse. Stage weights rho^(s-1)."""
    A, Y, W = [np.ones(len(y), bool)], [y], [1.0]
    active = pos != 1
    valid = np.add.reduceat((pos == 1).astype(int), st) == 1
    for s in range(2, K + 1):
        valid = valid & (np.add.reduceat((pos == s).astype(int), st) == 1)
        vr = np.repeat(valid, sz)
        A.append(active & vr)
        Y.append(((pos == s) & vr).astype(float))
        W.append(rho ** (s - 1))
        active = active & (pos != s)

    def f(preds, data):
        mx = np.repeat(np.maximum.reduceat(preds, st), sz)
        e0 = np.exp(preds - mx)
        g = np.zeros(len(preds))
        h = np.zeros(len(preds))
        for a, yy, w in zip(A, Y, W):
            e = np.where(a, e0, 0.0)
            z = np.repeat(np.add.reduceat(e, st), sz)
            p = np.where(z > 0, e / np.maximum(z, 1e-300), 0.0)
            g += w * (p - yy)
            h += w * p * (1 - p)
        return g, np.maximum(h, 1e-6)
    return f


def load_variant(featset, all_races=False):
    """all_races=True also keeps races without a recorded result (y = 0 there): used only to SCORE new races (fr_v3_final.py)."""
    d = pd.read_parquet(PROC / "fr_feats.parquet")
    if not all_races:
        d = d[d.n_winners >= 1].copy()
    d["y"] = (d.won / d.n_winners.where(d.n_winners >= 1)).fillna(0.0)
    F = features(d)  # never includes "labelled" (added below) or y
    if featset == "rt":
        d = d.merge(pd.read_parquet(PROC / "fr_ratings.parquet")[["rid", "numPmu"] + RT_COLS], on=["rid", "numPmu"], how="left")
        F = F + RT_COLS
    d["labelled"] = (d.n_winners >= 1).values
    d = d.sort_values(["ts", "rid", "numPmu"]).reset_index(drop=True)
    return d, F


def train(d, F, objective, rounds, valid=None):
    st, sz = starts_of(d.rid.values)
    K = {"win": 1, "pl3": 3, "pl5": 5}[objective]
    obj = pl_objective(st, sz, d.y.values, d.pos.fillna(0).values, K, 0.6 if K == 5 else 0.7)
    p = {**GBM, "verbose": -1, "num_threads": int(os.environ.get("NTHREADS", 28)), "objective": obj}
    ds = lgb.Dataset(d[F], d.y.values, free_raw_data=False)
    if valid is None:
        return lgb.train(p, ds, num_boost_round=rounds)
    sv, zv = starts_of(valid.rid.values)
    dv = lgb.Dataset(valid[F], valid.y.values, reference=ds, free_raw_data=False)

    def feval(preds, data):
        return "race_ll", race_logloss(group_softmax(preds, sv, zv), valid.y.values, sv, zv), False
    return lgb.train(p, ds, num_boost_round=rounds, valid_sets=[dv], feval=feval, callbacks=[lgb.early_stopping(100, verbose=False)])


def wf(name):
    t0 = time.time()
    objective, _, featset = name.partition("+")
    featset = featset or "base"
    d, F = load_variant(featset)
    print(f"{name}: {len(d):,} rows, {len(F)} features", flush=True)
    tr = d[(d.date >= FIRST_TRAIN) & (d.date < "2022-01-01")]
    va = d[d.date.dt.year == 2022]
    b = train(tr, F, objective, 3000, valid=va)
    rounds = int(b.best_iteration * 1.1)
    print(f"  best iteration {b.best_iteration} (2022 race log-loss {b.best_score['valid_0']['race_ll']:.4f}) -> {rounds} rounds "
          f"({time.time() - t0:.0f}s)", flush=True)
    parts = []
    for y in OOF_YEARS:
        tr = d[(d.date >= FIRST_TRAIN) & (d.date < f"{y}-01-01")]
        te = d[d.date.dt.year == y]
        if te.empty:
            continue
        b = train(tr, F, objective, rounds)
        s = b.predict(te[F], raw_score=True)
        st, sz = starts_of(te.rid.values)
        p = group_softmax(s, st, sz)
        if y <= 2024:
            print(f"  {y}: race log-loss {race_logloss(p, te.y.values, st, sz):.4f}, top-1 {top1_acc(p, te.y.values, st, sz):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        parts.append(pd.DataFrame({"rid": te.rid.values, "numPmu": te.numPmu.values, "f_score": s}))
        if y == OOF_YEARS[-1]:
            imp = pd.Series(b.feature_importance("gain"), index=F).sort_values(ascending=False)
            print("  top features (last fold):", ", ".join(f"{k} {v / imp.sum():.3f}" for k, v in imp.head(15).items()), flush=True)
    pd.concat(parts).to_parquet(EXP / f"{name}.parquet", index=False)
    json.dump({"rounds": rounds}, open(EXP / f"{name}.json", "w"))
    print(f"-> exp/{name}.parquet ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ dev stack evaluation
def stack(names):
    cols = ["rid", "numPmu", "date", "won", "n_winners", "mkt_pf", "mkt_pr", "specialite"]
    d = pd.read_parquet(PROC / "fr_feats.parquet", columns=cols)
    d = d[d.n_winners >= 1]
    d["y"] = d.won / d.n_winners
    for nm in names:
        v1 = PROC / "fr_oof_v1.parquet"  # after fr_blend.py, fr_oof.parquet holds the v2 blend; the v1 GBM is kept here
        src = (v1 if v1.exists() else PROC / "fr_oof.parquet") if nm == "base" else EXP / f"{nm}.parquet"
        o = pd.read_parquet(src)[["rid", "numPmu", "f_score"]].rename(columns={"f_score": f"s_{nm}"})
        d = d.merge(o, on=["rid", "numPmu"], how="inner")
    full = d.mkt_pf.notna() & d.mkt_pr.notna()
    d = d[full.groupby(d.rid).transform("all")].sort_values(["date", "rid", "numPmu"]).reset_index(drop=True)
    d = d[d.date.dt.year <= DEV_YEARS[-1]]
    d["year"] = d.date.dt.year
    t = d.specialite.astype(str).str.startswith("TROT").astype(float).values
    lmr, lmf = np.log(d.mkt_pr.values), np.log(d.mkt_pf.values)
    configs = {f"fund:{nm}": [d[f"s_{nm}"].values] for nm in names}
    configs.update({f"ref+{nm}": [lmr, d[f"s_{nm}"].values] for nm in names})
    configs.update({f"final+{nm}": [lmf, lmr, d[f"s_{nm}"].values] for nm in names})
    if len(names) > 1:
        configs["ref+all"] = [lmr] + [d[f"s_{nm}"].values for nm in names]
        configs["final+all"] = [lmf, lmr] + [d[f"s_{nm}"].values for nm in names]
    st, sz = starts_of(d.rid.values)
    yr = d.year.values[st]
    day = d.date.values[st]
    y = d.y.values
    ll = {}
    for cname, xs in configs.items():
        Z = np.column_stack([c for x in xs for c in (x, x * t)])
        s = np.zeros(len(d))
        for Y in DEV_YEARS:
            trm, tem = (d.year < Y).values, (d.year == Y).values
            w = fit_clogit(Z[trm], y[trm], d.rid.values[trm])
            s[tem] = Z[tem] @ w
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
        sdl = np.bincount(inv, dl)
        b = sdl[idx].sum(1) / cnt[idx].sum(1)
        out[cname] = {"dev_logloss": round(float(ll[cname][ok].mean()), 5), "diff_vs_first": round(float(dl.mean()), 5),
                      "ci95": [round(float(x), 5) for x in np.quantile(b, [.025, .975])],
                      "by_year": {int(Y): round(float(ll[cname][yr == Y].mean()), 4) for Y in DEV_YEARS}}
        print(f"  {cname:28s} dev log-loss {out[cname]['dev_logloss']:.5f}  diff {out[cname]['diff_vs_first']:+.5f} {out[cname]['ci95']}", flush=True)
    print(f"  ({ok.sum():,} dev races)")
    allr = json.load(open(RES / "exp_stack.json")) if (RES / "exp_stack.json").exists() else {}
    allr[" ".join(names)] = out
    json.dump(allr, open(RES / "exp_stack.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ratings":
        ratings()
    elif cmd == "wf":
        for nm in sys.argv[2:]:
            wf(nm)
    elif cmd == "stack":
        stack(sys.argv[2:])
