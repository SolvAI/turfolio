"""Hong Kong replication of the neural race model (src/fr_nn.py): same network and loss, HK features and history tokens.

  tune   train 1985-2009, validate 2010 (HK dev protocol) -> number of epochs
  wf     for each year Y in 1995..2018 train on 1985..Y-1, predict Y -> data/processed/exp/hk_<tag>.parquet
Run: python -I src/hk_nn.py tune | wf [nn | nn_L20]   (nn_L20 = v3 recipe: 20-run history, 4 seeds; epochs from the tune of nn)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fr_nn  # noqa: E402
from fr_exp import EXP  # noqa: E402
from hk_exp import RT  # noqa: E402
from hk_pipeline import F_of, FIRST_TRAIN, OOF_YEARS, load  # noqa: E402
from racemodel import starts_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
L, M = fr_nn.L, 14


def build():
    t0 = time.time()
    d = load()
    F = F_of(d)
    d = d.merge(pd.read_parquet(PROC / "hk_ratings.parquet"), on=["rid", "horse_id"], how="left")
    d = d.sort_values(["race_date", "rid", "horse_no"]).reset_index(drop=True)
    F = F + RT
    cats = [c for c in F if str(d[c].dtype) in ("category", "object")]
    nums = [c for c in F if c not in cats]
    ref = d[d.race_date < "1995-01-01"]
    R = ref[nums].astype(np.float32).values
    lo, hi = np.nanquantile(R, [.01, .99], axis=0)
    mu, sd = np.nanmean(np.clip(R, lo, hi), 0), np.nanstd(np.clip(R, lo, hi), 0) + 1e-6
    X = (np.clip(d[nums].astype(np.float32).values, lo, hi) - mu) / sd
    nan_cols = np.flatnonzero(np.isnan(X).mean(0) > 0.01)
    X = np.c_[np.nan_to_num(X), np.isnan(X[:, nan_cols])].astype(np.float32)
    C = np.column_stack([d[c].astype("category").cat.codes.values + 1 for c in cats]).astype(np.int64) if cats else np.zeros((len(d), 0), np.int64)
    card = [int(C[:, j].max()) + 1 for j in range(C.shape[1])]
    h = pd.read_parquet(PROC / "hk_feats.parquet", columns=["rid", "horse_id", "race_date", "race_no", "day", "pos_pct", "won", "final_placing",
                                                            "bl", "tbw", "early", "exp_win", "field", "cls", "distance", "rating", "actual_weight",
                                                            "draw", "location", "track", "jockey_name"])
    h = h.merge(pd.read_parquet(PROC / "hk_ratings.parquet")[["rid", "horse_id", "rt_sum", "rt_sum_dm"]], on=["rid", "horse_id"], how="left")
    h["field_rt"] = h.rt_sum - h.rt_sum_dm
    h = h.sort_values(["horse_id", "race_date", "race_no"]).reset_index(drop=True)
    tok = np.c_[h.pos_pct.fillna(1.1), h.pos_pct.isna(), h.won, (h.final_placing <= 3), np.log1p(h.bl.clip(0, 30)).fillna(0), h.bl.isna(),
                h.tbw.clip(-1, 10).fillna(0), h.tbw.isna(), h.early.fillna(0.5), np.log(h.exp_win.clip(1e-3, 1)), np.log(h.field), h.cls.fillna(4) / 4,
                h.distance / 1000 - 1.5, (h.rating.fillna(60) - 60) / 20, (h.actual_weight - 120) / 10, h.draw.fillna(7) / 7,
                h.rt_sum.fillna(0), h.field_rt.fillna(0), h.rt_sum_dm.fillna(0)].astype(np.float32)
    info = np.c_[h.day, h.distance / 1000, h.cls.fillna(4), h.location.astype("category").cat.codes,
                 h.jockey_name.astype("category").cat.codes, h.track.astype("category").cat.codes].astype(np.float32)
    gs = np.r_[0, np.flatnonzero(h.horse_id.values[1:] != h.horse_id.values[:-1]) + 1]
    gid = np.repeat(np.arange(len(gs)), np.diff(np.r_[gs, len(h)]))
    within = np.arange(len(h)) - gs[gid]
    H = np.full((len(h), L), -1, np.int64)
    for k in range(1, L + 1):
        ok = within >= k
        H[ok, k - 1] = np.flatnonzero(ok) - k
    pos_in_h = pd.Series(np.arange(len(h)), index=pd.MultiIndex.from_arrays([h.rid.values, h.horse_id.values]))
    ti = pos_in_h.reindex(pd.MultiIndex.from_arrays([d.rid.values, d.horse_id.values])).values.astype(np.int64)
    st, sz = starts_of(d.rid.values)
    keep = sz <= M
    RI = np.full((len(st), M), -1, np.int64)
    for j in range(M):
        ok = keep & (sz > j)
        RI[ok, j] = st[ok] + j
    print(f"built: {len(d):,} rows, {X.shape[1]} numeric, {len(cats)} categoricals ({time.time() - t0:.0f}s)", flush=True)
    return dict(d=d[["rid", "horse_id"]].reset_index(drop=True), X=X, C=C, card=card, y=d.y.values.astype(np.float32),
                pos=d.final_placing.fillna(0).values.astype(np.int64), ti=ti, H=H, tok=tok, info=info, RI=RI,
                year=d.race_date.dt.year.values[st], keep=keep)


def main(mode, tag="nn"):
    global L
    seeds = (0, 1)
    if "L20" in tag:  # v3 recipe: 20-run history, 4 seeds (HK has no form string)
        L = fr_nn.L = 20
        seeds = (0, 1, 2, 3)
    b = build()
    data = fr_nn.Data(b)
    yr, keep = b["year"], b["keep"]
    first = int(FIRST_TRAIN[:4])
    fr_nn.CFG.update(bs=128)
    if mode == "tune":
        tr = np.flatnonzero(keep & (yr >= first) & (yr <= 2009))
        va = np.flatnonzero(keep & (yr == 2010))
        res = {}
        for E in (4, 7, 10, 14):
            net, _ = fr_nn.run(data, b, tr, va, E, 0)
            res[E] = float(np.mean(fr_nn.predict(net, data, va)[1]))
            print(f"  {E} epochs: validation (2010) winner log-loss {res[E]:.4f}", flush=True)
        best = min(res, key=res.get)
        json.dump({"epochs": best, "val_by_epochs": res}, open(RES / "exp_hk_nn_tune.json", "w"))
        print(f"best epochs {best}", flush=True)
        return
    ep = json.load(open(RES / "exp_hk_nn_tune.json"))["epochs"]
    parts = []
    for Y in OOF_YEARS:
        tr = np.flatnonzero(keep & (yr >= first) & (yr < Y))
        te = np.flatnonzero(keep & (yr == Y))
        S = 0
        for sd in seeds:
            net, _ = fr_nn.run(data, b, tr, te, ep, sd)
            S = S + fr_nn.predict(net, data, te)[0] / len(seeds)
        ri = b["RI"][te]
        ok = ri >= 0
        parts.append(pd.DataFrame({"rid": b["d"].rid.values[ri[ok]], "horse_id": b["d"].horse_id.values[ri[ok]], "f_score": S[ok]}))
        print(f"  {Y} done", flush=True)
    pd.concat(parts).to_parquet(EXP / f"hk_{tag}.parquet", index=False)
    print(f"-> exp/hk_{tag}.parquet", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "nn")
