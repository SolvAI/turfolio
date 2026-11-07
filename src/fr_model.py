"""France: fundamental model (no odds of the race) + market stackers, walk-forward by year.

  dev   pick the number of boosting rounds: train 2015-2021, early-stop on 2022 (never looks at 2023+)
  wf    fundamental model out-of-fold: for each year Y in 2017..2026 train on 2015..Y-1, predict Y -> fr_oof.parquet
Run: python -I src/fr_model.py dev | wf
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from racemodel import fit_gbm, predict_gbm, race_logloss, starts_of, top1_acc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIRST_TRAIN = "2015-01-01"
OOF_YEARS = list(range(2017, 2027))
NOT_FEATURES = {
    # ids / keys / time
    "date", "raceKey", "reunionKey", "ts", "day", "rid", "hk", "driver", "entraineur", "proprietaire", "nomPere", "drv_trn", "heureDepart",
    # outcomes of THIS race (only ever used shifted, as history of later races)
    "pos", "won", "dq", "fail", "pos_pct", "rk", "rk_rel", "bl", "n_winners", "y",
    # market of THIS race (benchmark / stacker only)
    "odf_rapport", "odr_rapport", "exp_win",
}
GBM = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=300, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1,
           lambda_l2=10.0, max_bin=255, cat_smooth=20, cat_l2=10, max_cat_to_onehot=8, seed=0)


def load():
    d = pd.read_parquet(PROC / "fr_feats.parquet")
    d = d[d.n_winners >= 1].copy()  # races with a recorded winner
    d["y"] = d.won / d.n_winners
    return d.sort_values(["ts", "rid", "numPmu"]).reset_index(drop=True)


def features(d):
    return [c for c in d.columns if c not in NOT_FEATURES and not c.startswith("mkt_")]


def main(mode):
    d = load()
    F = features(d)
    print(f"{len(d):,} starters, {d.rid.nunique():,} races, {len(F)} features", flush=True)
    if mode == "dev":
        tr = d[(d.date >= FIRST_TRAIN) & (d.date < "2022-01-01")]
        va = d[d.date.dt.year == 2022]
        b = fit_gbm(tr[F], tr.y.values, tr.rid.values, GBM, 3000, valid=(va[F], va.y.values, va.rid.values, None))
        p, _ = predict_gbm(b, va[F], va.rid.values)
        st, sz = starts_of(va.rid.values)
        print(f"best iteration {b.best_iteration}; 2022 race log-loss {race_logloss(p, va.y.values, st, sz):.4f}, top-1 {top1_acc(p, va.y.values, st, sz):.4f}")
        imp = pd.Series(b.feature_importance("gain"), index=F).sort_values(ascending=False)
        print(imp.head(40).round(0).to_string())
        json.dump({"best_iteration": int(b.best_iteration)}, open(RES / "fr_dev_rounds.json", "w"))
        return
    if mode == "wf":
        rounds = int(json.load(open(RES / "fr_dev_rounds.json"))["best_iteration"] * 1.1)
        parts = PROC / "fr_oof_parts"
        parts.mkdir(exist_ok=True)
        for y in OOF_YEARS:
            tr = d[(d.date >= FIRST_TRAIN) & (d.date < f"{y}-01-01")]
            te = d[d.date.dt.year == y]
            if te.empty or (parts / f"{y}.parquet").exists():
                continue
            b = fit_gbm(tr[F], tr.y.values, tr.rid.values, GBM, rounds)
            p, s = predict_gbm(b, te[F], te.rid.values)
            st, sz = starts_of(te.rid.values)
            print(f"  {y}: train {len(tr):,} rows -> race log-loss {race_logloss(p, te.y.values, st, sz):.4f}, top-1 {top1_acc(p, te.y.values, st, sz):.4f}", flush=True)
            pd.DataFrame({"rid": te.rid.values, "numPmu": te.numPmu.values, "f_score": s, "f_p": p}).to_parquet(parts / f"{y}.parquet", index=False)
        pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("*.parquet"))]).to_parquet(PROC / "fr_oof.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
