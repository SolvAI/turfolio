"""France: the v3 model fitted on every labelled race (the walk-forward pipeline's components, frozen), and prediction for any date.

Components (each refitted on all labelled races up to --through, with the round/epoch counts tuned on dev):
  fundamental   LightGBM win+rt (fr_exp.py) + networks nn_L20 and nn_L20_mus (fr_nn.py, 4 seeds each), blended by a conditional logit
                fitted on the walk-forward out-of-fold scores (fr_blend.py)                               -> p_fundamental (no odds)
  end model     conditional logit on [log 30-min market prob, fundamental score] (x trot) + residual LightGBM (fr_eval.py)
                                                                                                         -> p_win (needs the 30-min odds)
  place         stage-2/3 order models (fr_exotics.py), place-pool model and payout regression fitted on 2019-2024 (fr_place.py)
                -> p_place, est_place_payout, place_ev; rule_B = p_place >= 0.6 and place_ev > 0.1 (the rule tested out of sample)
Run:
  python -I src/fr_v3_final.py train [--through YYYY-MM-DD] [--out DIR]       (default: all data -> models/v3)
  python -I src/fr_v3_final.py predict FROM [TO] [--model DIR]                 -> results/v3_predictions_FROM_TO.csv
  Races to predict must be in data/processed/fr_feats.parquet: append the day's programme (src/fr_fresh.py download + build),
  then rerun src/fr_features.py and `src/fr_exp.py ratings`. Results are not needed; p_win needs the 30-min reference odds.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_eval import RES_GBM, Z, load as load_eval, resid_X  # noqa: E402
from fr_exotics import gbm_X, load as load_exotics, stage_rows, Zs  # noqa: E402
from fr_exp import RT_COLS, load_variant, train as train_variant  # noqa: E402
from fr_model import FIRST_TRAIN  # noqa: E402
from fr_place import place_ev  # noqa: E402
from racemodel import fit_clogit, fit_gbm, group_softmax, predict_clogit, starts_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, EXP = ROOT / "data" / "processed", ROOT / "results", ROOT / "data" / "processed" / "exp"
MOD = ROOT / "models" / "v3"
NETS = ["nn_L20", "nn_L20_mus"]
VARIANTS = ["win+rt"] + NETS
RULE_B = (0.6, 0.1)


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def cats_of(X):
    return {c: [str(v) for v in X[c].cat.categories] for c in X.columns if str(X[c].dtype) == "category"}


def set_cats(X, cats):
    X = X.copy()
    for c, v in cats.items():
        X[c] = pd.Categorical(X[c].astype(str).where(X[c].notna()), categories=v)
    return X


def train(through, out):
    out.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(through)
    man = {"trained_through": through, "variants": VARIANTS}
    # 1. LightGBM win+rt
    d, F = load_variant("rt")
    tr = d[(d.date >= FIRST_TRAIN) & (d.date <= cut)]
    rounds = json.load(open(EXP / "win+rt.json"))["rounds"]
    b = train_variant(tr, F, "win", rounds)
    b.save_model(str(out / "gbm_win_rt.txt"))
    json.dump({"features": F, "rounds": rounds, "categories": cats_of(tr[F]), "train_rows": len(tr)}, open(out / "gbm_win_rt.json", "w"))
    man["trained_through"] = str(tr.date.max().date())  # last race actually used
    print(f"win+rt GBM: {rounds} rounds on {len(tr):,} starters", flush=True)
    # 2. networks (separate processes: fr_nn keeps its configuration in module globals)
    for tag in NETS:
        subprocess.run([sys.executable, "-I", str(ROOT / "src" / "fr_nn.py"), "final", tag, "0,1,2,3", through, str(out)], check=True)
    # 3. blend weights on the walk-forward out-of-fold scores (as fr_blend.py, all years up to the cut)
    e = pd.read_parquet(PROC / "fr_feats.parquet", columns=["rid", "numPmu", "date", "won", "n_winners", "specialite"])
    e = e[(e.n_winners >= 1) & (e.date <= cut)]
    e["y"] = e.won / e.n_winners
    for nm in VARIANTS:
        e = e.merge(pd.read_parquet(EXP / f"{nm}.parquet")[["rid", "numPmu", "f_score"]].rename(columns={"f_score": f"s_{nm}"}), on=["rid", "numPmu"], how="left")
    e = e[e[f"s_{VARIANTS[0]}"].notna()].sort_values(["date", "rid", "numPmu"]).reset_index(drop=True)
    full = e[[f"s_{nm}" for nm in VARIANTS]].notna().all(1).groupby(e.rid).transform("all").values
    t = e.specialite.astype(str).str.startswith("TROT").astype(float).values
    S = e[[f"s_{nm}" for nm in VARIANTS]].values
    Zf = np.column_stack([c for j in range(len(VARIANTS)) for c in (S[:, j], S[:, j] * t)])
    w_full = fit_clogit(Zf[full], e.y.values[full], e.rid.values[full])
    w_first = fit_clogit(np.c_[S[:, 0], S[:, 0] * t][full], e.y.values[full], e.rid.values[full])
    man["blend"] = {"w_full": w_full.tolist(), "w_first_only": w_first.tolist(), "order": "s_j, s_j*trot for j in variants"}
    print(f"blend weights {np.round(w_full, 3).tolist()}", flush=True)
    # 4. end model: combo_ref + residual booster on the out-of-fold v3 fundamental score (fr_oof.parquet)
    g = load_eval()
    g = g[g.date <= cut]
    w = fit_clogit(Z(g, "combo_ref"), g.y.values, g.rid.values)
    _, s = predict_clogit(w, Z(g, "combo_ref"), g.rid.values)
    rr = json.load(open(RES / "fr_report.json"))["rounds"]["combo_ref"]
    X = resid_X(g, False)
    br = fit_gbm(X, g.y.values, g.rid.values, RES_GBM, rr, init=s)
    br.save_model(str(out / "end_ref_resid.txt"))
    man["end_ref"] = {"combo_ref_weights": w.tolist(), "resid_rounds": rr, "resid_features": list(X.columns), "resid_categories": cats_of(X)}
    print(f"end model: combo weights {np.round(w, 3).tolist()}, residual {rr} rounds on {g.rid.nunique():,} races", flush=True)
    # 5. place: stage-2/3 order models (ref variant) on the out-of-fold end_ref
    x = load_exotics()
    x = x[x.date <= cut]
    er = json.load(open(RES / "exotics_rounds.json"))
    man["place"] = {}
    for st_ in (2, 3):
        m = stage_rows(x, st_)
        gg = x[m]
        y = (gg.pos == st_).astype(float).values
        w_full_s = fit_clogit(Zs(gg, "ref", "full"), y, gg.rid.values)
        w_md = fit_clogit(Zs(gg, "ref", "mdisc"), y, gg.rid.values)
        Xs = gbm_X(gg, "ref")
        bs = fit_gbm(Xs, y, gg.rid.values, RES_GBM, er[f"ref_{st_}"], init=Zs(gg, "ref", "full") @ w_full_s)
        bs.save_model(str(out / f"place_stage{st_}_resid.txt"))
        man["place"][f"stage{st_}"] = {"w_full": w_full_s.tolist(), "w_mdisc": w_md.tolist(), "rounds": er[f"ref_{st_}"],
                                       "features": list(Xs.columns), "categories": cats_of(Xs)}
        print(f"place stage {st_}: {gg.rid.nunique():,} races", flush=True)
    man["place"]["pool"] = json.load(open(RES / "place_pool_params.json"))["ref"]
    man["place"]["rule_B"] = {"p_place_min": RULE_B[0], "ev_min": RULE_B[1], "min_payout": 1.05}
    json.dump(man, open(out / "manifest.json", "w"), indent=1)
    print(f"-> {out}", flush=True)


def fundamentals(lo, hi, moddir):
    """p_fundamental (no odds) for every starter of the races dated lo..hi in fr_feats.parquet; returns (rows, manifest)."""
    import fr_nn
    man = json.load(open(moddir / "manifest.json"))
    d = pd.read_parquet(PROC / "fr_feats.parquet", filters=[("date", ">=", pd.Timestamp(lo)), ("date", "<=", pd.Timestamp(hi))])
    d = d.merge(pd.read_parquet(PROC / "fr_ratings.parquet")[["rid", "numPmu"] + RT_COLS], on=["rid", "numPmu"], how="left")
    d = d.sort_values(["ts", "rid", "numPmu"]).reset_index(drop=True)
    if d.empty:
        raise SystemExit(f"no races {lo}..{hi} in fr_feats.parquet")
    t = d.specialite.astype(str).str.startswith("TROT").astype(float).values
    # fundamental
    gm = json.load(open(moddir / "gbm_win_rt.json"))
    d["s_win+rt"] = lgb.Booster(model_file=str(moddir / "gbm_win_rt.txt")).predict(set_cats(d[gm["features"]], gm["categories"]), raw_score=True)
    for tag in NETS:
        d = d.merge(fr_nn.score(tag, lo, hi, moddir).rename(columns={"f_score": f"s_{tag}"}), on=["rid", "numPmu"], how="left")
    S = d[[f"s_{nm}" for nm in man["variants"]]].values
    full = pd.Series(~np.isnan(S).any(1)).groupby(d.rid.values).transform("all").values
    Zf = np.column_stack([c for j in range(S.shape[1]) for c in (S[:, j], S[:, j] * t)])
    d["f_score"] = np.where(full, np.nan_to_num(Zf) @ np.array(man["blend"]["w_full"]),
                            np.c_[S[:, 0], S[:, 0] * t] @ np.array(man["blend"]["w_first_only"]))
    st, sz = starts_of(d.rid.values)
    d["p_fundamental"] = group_softmax(d.f_score.values, st, sz)
    return d, man


def stack(d, man, moddir):
    """End model (p_win) and place model (p_place, est_place_payout, place_ev, rule_B) on the 30-min odds (mkt_pr, mkt_overr).
    d: rows from fundamentals(), sorted by race; races without complete 30-min odds keep p_win = p_fundamental."""
    d = d.copy()
    t = d.specialite.astype(str).str.startswith("TROT").astype(float).values
    # end model where the race's 30-min odds are complete
    has = d.mkt_pr.notna().groupby(d.rid).transform("all").values
    d["p_win"], d["used_odds"] = d.p_fundamental, has
    d["p_place"], d["est_place_payout"], d["place_ev"], d["rule_B"] = np.nan, np.nan, np.nan, False
    if has.any():
        e = d[has].copy()
        e["is_trot"] = t[has]
        e["lmr"] = np.log(e.mkt_pr)
        e["mr_rank"] = e.groupby("rid").mkt_pr.rank(ascending=False)
        er = man["end_ref"]
        s = Z(e, "combo_ref") @ np.array(er["combo_ref_weights"])
        s = s + lgb.Booster(model_file=str(moddir / "end_ref_resid.txt")).predict(set_cats(resid_X(e, False), er["resid_categories"]), raw_score=True)
        es, ez = starts_of(e.rid.values)
        e["end_ref"] = group_softmax(s, es, ez)
        d.loc[has, "p_win"] = e.end_ref.values
        # place
        e["t"] = e.is_trot
        e["lp_ref"] = np.log(np.clip(e.end_ref, 1e-9, 1))
        e["p_rank_ref"] = e.groupby("rid").end_ref.rank(ascending=False)
        pl = man["place"]

        def strength(score):
            return np.exp(score - np.repeat(np.maximum.reduceat(score, es), ez))
        A, B, Am, Bm = [], [], [], []
        for k, (lst, lstm) in ((2, (A, Am)), (3, (B, Bm))):
            p_ = pl[f"stage{k}"]
            sc = Zs(e, "ref", "full") @ np.array(p_["w_full"])
            sc = sc + lgb.Booster(model_file=str(moddir / f"place_stage{k}_resid.txt")).predict(set_cats(gbm_X(e, "ref"), p_["categories"]), raw_score=True)
            lst.append(strength(sc))
            lstm.append(strength(Zs(e, "ref", "mdisc") @ np.array(p_["w_mdisc"])))
        m = np.where(ez <= 7, 2, 3).astype(np.int64)
        n = len(e)
        zero, one = np.zeros(n), np.ones(len(es))
        P, _ = place_ev(es, ez, e.end_ref.values, A[0], B[0], m, zero, one, one, False)
        Pm, _ = place_ev(es, ez, e.mkt_pr.values, Am[0], Bm[0], m, zero, one, one, False)
        po = pl["pool"]
        tr_ = e.t.values
        gvec = np.where(tr_ == 1, po["gamma_trot"], po["gamma_flat"])
        sh = np.exp(gvec * np.log(e.mkt_pr.values))
        sh = sh / np.repeat(np.add.reduceat(sh, es), ez)
        Rr = np.where(tr_[es] == 1, po["R_trot"], po["R_flat"])
        minpay = np.full(len(es), pl["rule_B"]["min_payout"])
        _, Ehat = place_ev(es, ez, e.end_ref.values, A[0], B[0], m, sh, Rr, minpay, True)
        X = np.c_[np.ones(n), np.log(Ehat - 1 + 1e-3), np.log(P / Pm), tr_, np.log(np.repeat(ez, ez))]
        Ecor = np.maximum(1 + np.exp(X @ np.array(po["beta"])) * po["smear"], pl["rule_B"]["min_payout"])
        ok = np.repeat(ez >= 4, ez)  # no place betting below 4 starters
        d.loc[has, "p_place"] = np.where(ok, P, np.nan)
        d.loc[has, "est_place_payout"] = np.where(ok, Ecor, np.nan)
        d.loc[has, "place_ev"] = np.where(ok, P * Ecor - 1, np.nan)
        d.loc[has, "rule_B"] = ok & (P >= pl["rule_B"]["p_place_min"]) & (P * Ecor - 1 > pl["rule_B"]["ev_min"])
    d["horse"] = d.hk.astype(str).str.split("|").str[0]
    return d


def predict(lo, hi, moddir):
    d, man = fundamentals(lo, hi, moddir)
    d = stack(d, man, moddir)
    cols = ["date", "raceKey", "numPmu", "horse", "p_fundamental", "p_win", "used_odds", "p_place", "est_place_payout", "place_ev", "rule_B"]
    out = d[cols + ["rid"]].copy()
    f = RES / f"v3_predictions_{lo}_{hi}.csv"
    out[cols].round(5).to_csv(f, index=False)
    print(f"-> {f}: {out.raceKey.nunique()} races, {int(out.rule_B.sum())} rule-B place bets", flush=True)
    return out


if __name__ == "__main__":
    os.environ.setdefault("NTHREADS", "12")
    if sys.argv[1] == "train":
        train(arg("--through", "2100-01-01"), Path(arg("--out", str(MOD))))
    else:
        lo = sys.argv[2]
        hi = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else lo
        predict(lo, hi, Path(arg("--model", str(MOD))))
