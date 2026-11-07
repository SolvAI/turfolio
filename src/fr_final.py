"""France: the end model, fit on everything, and prediction for new races.

  train    fundamental LightGBM on all labelled races 2015 -> latest; the combination (conditional logit on
           [reference odds, fundamental score]) and its residual booster on all out-of-fold years (fr_oof.parquet)
           -> models/fr_fundamental.txt, models/fr_resid_ref.txt, models/fr_final.json
  predict  python -I src/fr_final.py predict 2026-05-05 [--no-odds]
           Scores every race of that date in fr_feats.parquet (append the new programme to fr_runners/fr_races and
           rerun src/fr_features.py first; results are not needed). Uses the reference odds if the race has them,
           otherwise the fundamental model alone. -> results/predictions_<date>.csv
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_eval import RES_GBM, RESID_FEATS, Z, load as load_eval, resid_X  # noqa: E402
from fr_model import FIRST_TRAIN, GBM, features, load  # noqa: E402
from racemodel import fit_clogit, fit_gbm, group_softmax, predict_clogit, starts_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, MOD = ROOT / "data" / "processed", ROOT / "results", ROOT / "models"


def train():
    MOD.mkdir(exist_ok=True)
    d = load()
    F = features(d)
    rounds = int(json.load(open(RES / "fr_dev_rounds.json"))["best_iteration"] * 1.15)
    tr = d[d.date >= FIRST_TRAIN]
    b = fit_gbm(tr[F], tr.y.values, tr.rid.values, GBM, rounds)
    b.save_model(str(MOD / "fr_fundamental.txt"))
    e = load_eval()
    w = fit_clogit(Z(e, "combo_ref"), e.y.values, e.rid.values)
    _, s = predict_clogit(w, Z(e, "combo_ref"), e.rid.values)
    rr = json.load(open(RES / "fr_report.json"))["rounds"]["combo_ref"]
    br = fit_gbm(resid_X(e, False), e.y.values, e.rid.values, RES_GBM, rr, init=s)
    br.save_model(str(MOD / "fr_resid_ref.txt"))
    cats = {c: list(tr[c].cat.categories) for c in F if str(tr[c].dtype) == "category"}
    json.dump({"features": F, "combo_ref_weights": list(map(float, w)), "fundamental_rounds": rounds, "resid_rounds": rr,
               "trained_through": str(d.date.max().date()), "categories": cats}, open(MOD / "fr_final.json", "w"))
    print(f"saved: fundamental ({rounds} rounds on {len(tr):,} starters), combo weights {np.round(w, 3)}, residual {rr} rounds")


def predict(date, use_odds=True):
    meta = json.load(open(MOD / "fr_final.json"))
    F = meta["features"]
    d = pd.read_parquet(PROC / "fr_feats.parquet")
    d = d[d.date == pd.Timestamp(date)].sort_values(["ts", "rid", "numPmu"]).reset_index(drop=True)
    if d.empty:
        raise SystemExit(f"no races on {date} in fr_feats.parquet")
    for c, cats in meta["categories"].items():
        d[c] = pd.Categorical(d[c].astype(object), categories=cats)
    st, sz = starts_of(d.rid.values)
    fb = lgb.Booster(model_file=str(MOD / "fr_fundamental.txt"))
    d["f_score"] = fb.predict(d[F], raw_score=True)
    d["p_fundamental"] = group_softmax(d.f_score.values, st, sz)
    d["p"] = d.p_fundamental
    has = d.mkt_pr.notna().groupby(d.rid).transform("all") if use_odds else pd.Series(False, index=d.index)
    if has.any():
        e = d[has].copy()
        e["is_trot"] = e.specialite.astype(str).str.startswith("TROT").astype(float)
        e["lmr"] = np.log(e.mkt_pr)
        e["mr_rank"] = e.groupby("rid").mkt_pr.rank(ascending=False)
        s = Z(e, "combo_ref") @ np.array(meta["combo_ref_weights"])
        br = lgb.Booster(model_file=str(MOD / "fr_resid_ref.txt"))
        X = resid_X(e, False)
        s = s + br.predict(X, raw_score=True)
        es, ez = starts_of(e.rid.values)
        d.loc[has, "p"] = group_softmax(s, es, ez)
    d["fair_odds"] = 1 / d.p
    d["horse"] = d.hk.str.split("|").str[0]
    out = d[["raceKey", "numPmu", "horse", "p", "p_fundamental", "fair_odds", "odr_rapport"]].rename(columns={"odr_rapport": "reference_odds"})
    out["used_odds"] = has.values
    f = RES / f"predictions_{date}.csv"
    out.round(4).to_csv(f, index=False)
    print(out.head(20).to_string(index=False))
    print(f"-> {f} ({out.raceKey.nunique()} races)")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        predict(sys.argv[2], use_odds="--no-odds" not in sys.argv)
