"""France: blend fundamental-model variants into one out-of-fold score (still never sees this race's odds) -> fr_oof.parquet.

score = conditional logit on [s_1, s_1 x trot, s_2, s_2 x trot, ...], weights fitted walk-forward on earlier years' out-of-fold scores
(for the first year, 2017, which is only ever used as stacker training data, the weights are fitted on 2017 itself).
Rows a variant could not score (races of more than 24 runners for the network) get the blend of the remaining variants.
The previous fr_oof.parquet is kept as fr_oof_v1.parquet.
Run: python -I src/fr_blend.py win+rt nn
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from racemodel import fit_clogit, group_softmax, starts_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EXP = PROC / "exp"


def main(names):
    d = pd.read_parquet(PROC / "fr_feats.parquet", columns=["rid", "numPmu", "date", "won", "n_winners", "specialite"])
    d = d[d.n_winners >= 1]
    d["y"] = d.won / d.n_winners
    for nm in names:
        src = PROC / "fr_oof_v1.parquet" if nm == "base" else EXP / f"{nm}.parquet"
        d = d.merge(pd.read_parquet(src)[["rid", "numPmu", "f_score"]].rename(columns={"f_score": f"s_{nm}"}), on=["rid", "numPmu"], how="left")
    d = d[d[f"s_{names[0]}"].notna()].sort_values(["date", "rid", "numPmu"]).reset_index(drop=True)
    d["year"] = d.date.dt.year
    t = d.specialite.astype(str).str.startswith("TROT").astype(float).values
    S = d[[f"s_{nm}" for nm in names]].values
    have = ~np.isnan(S)
    race_all = pd.Series(have.all(1)).groupby(d.rid.values).transform("all").values
    Z_full = np.column_stack([c for j in range(len(names)) for c in (S[:, j], S[:, j] * t)])
    Z_first = np.column_stack([S[:, 0], S[:, 0] * t])
    out = np.zeros(len(d))
    for Y in sorted(d.year.unique()):
        te = (d.year == Y).values
        trm = (d.year < Y).values if Y > d.year.min() else te
        for Z, mask in ((Z_full, race_all), (Z_first, ~race_all)):
            ok_tr, ok_te = trm & race_all, te & mask  # weights always fitted on fully scored races
            if not ok_te.any():
                continue
            w = fit_clogit(Z[ok_tr], d.y.values[ok_tr], d.rid.values[ok_tr])
            out[ok_te] = np.nan_to_num(Z[ok_te]) @ w
            if Z is Z_full:
                print(f"  {Y}: weights {np.round(w, 3).tolist()}", flush=True)
    st, sz = starts_of(d.rid.values)
    p = group_softmax(out, st, sz)
    if not (PROC / "fr_oof_v1.parquet").exists():
        shutil.copy(PROC / "fr_oof.parquet", PROC / "fr_oof_v1.parquet")
    pd.DataFrame({"rid": d.rid.values, "numPmu": d.numPmu.values, "f_score": out, "f_p": p}).to_parquet(PROC / "fr_oof.parquet", index=False)
    print(f"-> fr_oof.parquet: {len(d):,} rows; {(~race_all).sum():,} rows in races not scored by every variant", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
