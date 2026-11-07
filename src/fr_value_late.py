"""Win value rule with a better payout model: the network's second head (fr_nn.py tag nn_L20_mus_mt, exp/nn_L20_mus_mt_head2.parquet)
predicts the late move (final market given the 30-min market). The rule of fr_value.py bets when EV = p * E[final odds | win] - 1 > tau
(p = end_ref, decided 30 min out, paid at the final pool odds). Here E[final odds | win] comes from the winners regression of fr_value.py
with or without head 2's predicted late move as two extra inputs.
  dev   fit the payout model and choose (p_min, tau) on 2018-2021, evaluate both versions on 2022-2024 (dev validation)
  test  fit + choose on 2018-2024 (as fr_value.py), evaluate once on 2025-2026, snapshot odds and official dividends
Run: python -I src/fr_value_late.py dev | test
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_exotics import roi_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, RAW = ROOT / "data" / "processed", ROOT / "results", ROOT / "data" / "raw"
GRID = [(pm, tau) for pm in (0.0, 0.02, 0.05, 0.1) for tau in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)]


def load():
    d = pd.read_parquet(PROC / "fr_preds.parquet")
    d = d.merge(pd.read_parquet(PROC / "fr_feats.parquet", columns=["rid", "numPmu", "raceKey", "field"]), on=["rid", "numPmu"])
    h = pd.read_parquet(PROC / "exp" / "nn_L20_mus_mt_head2.parquet").rename(columns={"f_score": "s2"})
    d = d.merge(h, on=["rid", "numPmu"], how="left")
    d = d[(d.odr_rapport > 1) & (d.odf_rapport > 1) & d.mkt_ref.notna() & d.s2.notna()].copy()
    d["year"] = d.date.dt.year
    d["t"] = d.specialite.astype(str).str.startswith("TROT").astype(float)
    lmr = np.log(d.mkt_ref.values)
    z = pd.Series(lmr + d.s2.values)
    g = z.groupby(d.rid.values)
    z = z - g.transform("max") - np.log(np.exp(z - g.transform("max")).groupby(d.rid.values).transform("sum"))
    d["m2"] = z.values - lmr  # head 2's predicted late move in log probability (renormalised over the runners kept)
    d["ret"] = np.where(d.y > 0, d.odf_rapport, 0.0)
    rp = pd.concat([pd.read_parquet(q, columns=["raceKey", "typePari", "libelle", "combinaison", "dividendePourUnEuro"])
                    for q in sorted((RAW / "fr").glob("rapports-*.parquet"))], ignore_index=True)
    sg = rp[(rp.typePari == "SIMPLE_GAGNANT") & (rp.libelle.astype(str).str.lower() == "simple gagnant")].copy()
    bad = set(rp.raceKey[(rp.typePari == "SIMPLE_GAGNANT") & rp.libelle.astype(str).str.lower().str.contains("rembours")])
    sg = sg[~sg.raceKey.isin(bad)]
    sg["numPmu"] = pd.to_numeric(sg.combinaison, errors="coerce")
    sg["div"] = sg.dividendePourUnEuro / 100
    d = d.merge(sg[["raceKey", "numPmu", "div"]].drop_duplicates(["raceKey", "numPmu"]), on=["raceKey", "numPmu"], how="left")
    d["has_div"] = d.raceKey.isin(set(sg.raceKey))
    d["ret_official"] = d["div"].fillna(0.0)
    return d


def design(d, late):
    lo = np.log(d.odr_rapport.values)
    X = np.c_[np.ones(len(d)), lo, np.log(d.end_ref.values / d.mkt_ref.values), d.t, np.log(d.field), lo * d.t]
    return np.c_[X, d.m2, d.m2 * d.t] if late else X


def ev_of(d, late, fit):
    X, y = design(d, late), np.log(d.odf_rapport.values)
    wn = fit & (d.y.values > 0)
    b, *_ = np.linalg.lstsq(X[wn], y[wn], rcond=None)
    smear = float(np.mean(np.exp(y[wn] - X[wn] @ b)))
    return d.end_ref.values * np.exp(X @ b) * smear - 1


def choose(d, ev, m, min_bets):
    res = {}
    for pm, tau in GRID:
        k = m & (d.end_ref.values >= pm) & (ev > tau) & (ev < 1.0)
        if k.sum() >= min_bets:
            res[(pm, tau)] = d.ret.values[k].mean() - 1
    return max(res, key=res.get)


def main(mode):
    d = load()
    day = d.date.values
    yr = d.year.values
    rep = {}
    if mode == "dev":
        fit, ev_m = (yr >= 2018) & (yr <= 2021), (yr >= 2022) & (yr <= 2024)
        for late in (False, True):
            ev = ev_of(d, late, fit)
            pm, tau = choose(d, ev, fit, 1000)
            k = ev_m & (d.end_ref.values >= pm) & (ev > tau) & (ev < 1.0)
            name = "with_late_head" if late else "payout_model_v3"
            rep[name] = {"chosen_on_2018_21": f"p>={pm:g},ev>{tau:g}", "fit_years": roi_summary(d.ret.values[fit & (d.end_ref.values >= pm) & (ev > tau) & (ev < 1)]),
                         "eval_2022_24": roi_summary(d.ret.values[k], day[k]),
                         "eval_2022_24_by_year": {int(Y): roi_summary(d.ret.values[k & (yr == Y)]) for Y in (2022, 2023, 2024)},
                         "eval_same_tau_0.15": roi_summary(d.ret.values[ev_m & (ev > 0.15) & (ev < 1)], day[ev_m & (ev > 0.15) & (ev < 1)])}
            print(name, json.dumps(rep[name]), flush=True)
    else:
        dev, test = (yr >= 2018) & (yr <= 2024), (yr >= 2025)
        for late in (False, True):
            ev = ev_of(d, late, dev)
            pm, tau = choose(d, ev, dev, 2000)
            k = test & (d.end_ref.values >= pm) & (ev > tau) & (ev < 1.0)
            ko = k & d.has_div.values
            name = "with_late_head" if late else "payout_model_v3"
            rep[name] = {"chosen_on_dev": f"p>={pm:g},ev>{tau:g}", "test_snapshot_odds": roi_summary(d.ret.values[k], day[k]),
                         "test_official_dividends": roi_summary(d.ret_official.values[ko], day[ko]),
                         "test_by_year": {int(Y): roi_summary(d.ret.values[k & (yr == Y)]) for Y in sorted(set(yr[test]))}}
            fw = k & (d.date.values >= np.datetime64("2026-05-06"))  # races after the HF archive (fresh/ run only)
            if fw.any():
                rep[name]["fresh_window_from_2026_05_06"] = {"snapshot_odds": roi_summary(d.ret.values[fw], day[fw]),
                                                             "official_dividends": roi_summary(d.ret_official.values[fw & d.has_div.values], day[fw & d.has_div.values])}
            print(name, json.dumps(rep[name]), flush=True)
    json.dump(rep, open(RES / f"value_late_{mode}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
