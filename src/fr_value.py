"""France: win-bet value rule that anticipates the late money (decide 30 min out, paid at the final pool odds).

The naive rule uses EV = p * reference odds - 1. But late money moves toward what the model sees (the final market is much
sharper), so the final odds of a horse the model likes are usually shorter. Here the final odds are predicted first:
  log final odds ~ [1, log ref odds, log(model p / ref market p), trot, log field, log ref odds x trot]
fitted on dev runners (2018-2024), with smearing. What pays is the final odds of a WINNER, and late money is informed (winners
shorten more), so the main rule uses the model fitted on dev winners only: EV = p * E[final odds | win] - 1. (The all-runner fit,
E[final odds], is kept for comparison: it overstates the payout.) Threshold (and a minimum model probability) chosen on dev, applied once to test 2025-26.
Run: python -I src/fr_value.py -> results/value_report.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_exotics import day_boot, roi_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
DEV, TEST = (2018, 2024), (2025, 2026)


def main():
    d = pd.read_parquet(PROC / "fr_preds.parquet")
    d["year"] = d.date.dt.year
    d["t"] = d.specialite.astype(str).str.startswith("TROT").astype(float)
    d["field"] = d.groupby("rid").rid.transform("size")
    d = d[(d.odr_rapport > 1) & (d.odf_rapport > 1)].copy()
    d["ret"] = np.where(d.y > 0, d.odf_rapport, 0.0)
    dev, test = d.year.between(*DEV).values, d.year.between(*TEST).values
    day = d.date.values
    rep = {}
    for v, p_col in (("ref", "end_ref"), ("fund", "fund")):
        p = d[p_col].values
        lo = np.log(d.odr_rapport.values)
        X = np.c_[np.ones(len(d)), lo, np.log(p / d.mkt_ref.values), d.t, np.log(d.field), lo * d.t]
        y = np.log(d.odf_rapport.values)
        beta, *_ = np.linalg.lstsq(X[dev], y[dev], rcond=None)
        res = y[dev] - X[dev] @ beta
        smear = float(np.mean(np.exp(res)))
        Ef = np.exp(X @ beta) * smear
        # E[final odds | the horse wins]: late money is informed, winners shorten more than losers, so fit on dev WINNERS only
        wn = dev & (d.y.values > 0)
        bw, *_ = np.linalg.lstsq(X[wn], y[wn], rcond=None)
        Ew = np.exp(X @ bw) * float(np.mean(np.exp(y[wn] - X[wn] @ bw)))
        evs = {"naive_ref_odds": p * d.odr_rapport.values - 1, "predicted_final_odds": p * Ef - 1, "predicted_final_odds_given_win": p * Ew - 1}
        R = {"payout_model": {"coef_const_logRefOdds_logEdge_trot_logField_logRefOddsXtrot": [round(float(x), 4) for x in beta],
                              "smearing": round(smear, 4), "resid_sd": round(float(res.std()), 4)},
             "payout_model_winners": [round(float(x), 4) for x in bw]}
        for name, ev in evs.items():
            eb = [-1, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1.0, 1e9]
            bi = np.digitize(ev, eb) - 1
            R[f"roi_by_estimated_ev_{name}"] = {lab: [{"est_ev": f"{eb[i]:g}..{eb[i + 1]:g}", **roi_summary(d.ret.values[g & (bi == i)])}
                                                      for i in range(len(eb) - 1)] for lab, g in (("dev", dev), ("test", test))}
            grid = [(pm, tau) for pm in (0.0, 0.02, 0.05, 0.1) for tau in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)]
            dres, tres = {}, {}
            for pm, tau in grid:
                key = f"p>={pm:g},ev>{tau:g}"
                m = (p >= pm) & (ev > tau) & (ev < 1.0)
                dres[key] = roi_summary(d.ret.values[dev & m])
                tres[key] = roi_summary(d.ret.values[test & m], day[test & m])
            ok = [(k, r) for k, r in dres.items() if r["bets"] >= 2000]
            best = max(ok, key=lambda kv: kv[1]["roi"])[0]
            R[name] = {"dev": dres, "chosen_on_dev": best, "dev_chosen": dres[best], "test_chosen": tres[best], "test_all": tres}
            print(f"{v} {name}: chosen {best}; dev {dres[best]}; test {tres[best]}", flush=True)
        # per-year ROI of the chosen predicted-final-odds rule (stability)
        best = R["predicted_final_odds_given_win"]["chosen_on_dev"]
        pm, tau = [float(x.split(">")[-1].lstrip("=")) for x in best.split(",")]
        ev = evs["predicted_final_odds_given_win"]
        m = (p >= pm) & (ev > tau) & (ev < 1.0)
        R["per_year_chosen"] = {int(Y): roi_summary(d.ret.values[m & (d.year.values == Y)]) for Y in sorted(d.year.unique())}
        rep[v] = R
    json.dump(rep, open(RES / "value_report.json", "w"), indent=1)
    print(json.dumps({k: rep["ref"]["per_year_chosen"][k] for k in rep["ref"]["per_year_chosen"]}, indent=0))


if __name__ == "__main__" and not sys.argv[1:]:
    main()


def checks():
    """Realism checks of the chosen rule: official win dividends (dead heats, payout vs odds snapshot) and the effect of our own stake
    on the pool (win pool rebuilt from the winner's dividend and number of winning stakes)."""
    rep = json.load(open(RES / "value_report.json"))
    d = pd.read_parquet(PROC / "fr_preds.parquet")
    f = pd.read_parquet(PROC / "fr_feats.parquet", columns=["rid", "numPmu", "raceKey", "h_n", "n_starts", "specialite", "quinte", "field"])
    d = d.merge(f.drop(columns=["specialite"]), on=["rid", "numPmu"])
    d["year"] = d.date.dt.year
    d["t"] = d.specialite.astype(str).str.startswith("TROT").astype(float)
    d = d[(d.odr_rapport > 1) & (d.odf_rapport > 1)].copy()
    rp = pd.concat([pd.read_parquet(p, columns=["raceKey", "typePari", "libelle", "combinaison", "dividendePourUnEuro", "nombreGagnants", "miseBase"])
                    for p in sorted((ROOT / "data" / "raw" / "fr").glob("rapports-*.parquet"))], ignore_index=True)
    sg = rp[rp.typePari == "SIMPLE_GAGNANT"]
    bad = set(sg.raceKey[sg.libelle.str.lower().str.contains("rembours")])
    sg = sg[(sg.libelle.astype(str).str.lower() == "simple gagnant") & ~sg.raceKey.isin(bad)].copy()
    sg["numPmu"] = pd.to_numeric(sg.combinaison, errors="coerce")
    sg["div"] = sg.dividendePourUnEuro / 100
    sg["stake_eur"] = sg.nombreGagnants * sg.miseBase / 100
    sg["net_pool"] = sg["div"] * sg.stake_eur   # what the pool pays out = return rate x pool
    d = d[d.raceKey.isin(set(sg.raceKey))]
    d = d.merge(sg[["raceKey", "numPmu", "div"]], on=["raceKey", "numPmu"], how="left")
    d = d.merge(sg.groupby("raceKey").net_pool.first().rename("net_pool"), on="raceKey", how="left")
    d["ret_official"] = d["div"].fillna(0.0)
    d["ret_snapshot"] = np.where(d.y > 0, d.odf_rapport, 0.0)
    # rule = same payout model as main(), refit on dev
    p = d.end_ref.values
    lo = np.log(d.odr_rapport.values)
    X = np.c_[np.ones(len(d)), lo, np.log(p / d.mkt_ref.values), d.t, np.log(d.field), lo * d.t]
    dev, test = d.year.between(*DEV).values, d.year.between(*TEST).values
    wn = dev & (d.y.values > 0)
    beta, *_ = np.linalg.lstsq(X[wn], np.log(d.odf_rapport.values[wn]), rcond=None)
    smear = float(np.mean(np.exp(np.log(d.odf_rapport.values[wn]) - X[wn] @ beta)))
    ev = p * np.exp(X @ beta) * smear - 1
    best = rep["ref"]["predicted_final_odds_given_win"]["chosen_on_dev"]
    pm, tau = [float(x.split(">")[-1].lstrip("=")) for x in best.split(",")]
    m = (p >= pm) & (ev > tau) & (ev < 1.0)
    out = {"rule": best, "races_with_official_dividend": int(d.rid.nunique())}
    day = d.date.values
    for lab, g in (("dev", dev), ("test", test)):
        mm = g & m
        out[lab] = {"snapshot_odds": roi_summary(d.ret_snapshot.values[mm], day[mm]), "official_dividend": roi_summary(d.ret_official.values[mm], day[mm])}
        # own stake: payout per euro becomes (net pool + 0.835 X) / (stake on the horse + X); stake on horse = net pool / final odds
        # winners: stake = net pool / official dividend (exact); others: from the final odds snapshot (only used for the median)
        S = np.where(d.ret_official.values[mm] > 0, d.net_pool.values[mm] / np.maximum(d.ret_official.values[mm], 1e-9),
                     d.net_pool.values[mm] / d.odf_rapport.values[mm])
        for X_ in (2, 10, 50, 200):
            new = (d.net_pool.values[mm] + 0.835 * X_) / (S + X_)
            r_ = np.where(d.ret_official.values[mm] > 0, new, 0.0)
            out[lab][f"official_with_own_stake_{X_}eur"] = roi_summary(r_, day[mm])
        # by size of the win pool (net of takeout): thin pools are where the market is weakest and where our stake moves the price
        bins = [0, 2e4, 5e4, 1e5, 3e5, 1e12]
        pb = np.digitize(d.net_pool.values[mm], bins) - 1
        rows = []
        for i in range(len(bins) - 1):
            k = pb == i
            row = {"net_win_pool_eur": f"{bins[i]:,.0f}-{bins[i + 1]:,.0f}".replace("1,000,000,000,000", "max"), "bets": int(k.sum())}
            for X_ in (0, 2, 10, 50):
                new = (d.net_pool.values[mm][k] + 0.835 * X_) / (S[k] + X_)
                r_ = np.where(d.ret_official.values[mm][k] > 0, new, 0.0)
                row[f"roi_stake_{X_}eur"] = round(float(r_.mean() - 1), 4) if k.any() else None
            rows.append(row)
        out[lab]["by_pool_size"] = rows
        out[lab]["median_stake_on_horse_eur"] = round(float(np.nanmedian(S)), 0)
        out[lab]["median_net_win_pool_eur"] = round(float(np.nanmedian(d.net_pool.values[mm])), 0)
        sel = d[mm]
        out[lab]["profile"] = {"median_ref_odds": round(float(sel.odr_rapport.median()), 2), "median_final_odds": round(float(sel.odf_rapport.median()), 2),
                               "median_model_p": round(float(sel.end_ref.median()), 4), "median_ref_market_p": round(float(sel.mkt_ref.median()), 4),
                               "trot_share": round(float(sel.t.mean()), 3), "quinte_share": round(float(sel.quinte.mean()), 3),
                               "debut_share(no start in data)": round(float((sel.h_n == 0).mean()), 3),
                               "by_specialite": {str(k): roi_summary(d.ret_official.values[mm][sel.specialite.astype(str).values == k])
                                                 for k in sorted(sel.specialite.astype(str).unique())}}
    rep["checks"] = out
    json.dump(rep, open(RES / "value_report.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__" and sys.argv[1:] == ["checks"]:
    checks()


def liquidity():
    """Is the model's edge over the market concentrated in thin pools? Accuracy and value-rule ROI by size of the win pool."""
    rep = json.load(open(RES / "value_report.json"))
    d = pd.read_parquet(PROC / "fr_preds.parquet")
    f = pd.read_parquet(PROC / "fr_feats.parquet", columns=["rid", "numPmu", "raceKey", "field"])
    d = d.merge(f, on=["rid", "numPmu"])
    d["year"] = d.date.dt.year
    rp = pd.concat([pd.read_parquet(p, columns=["raceKey", "typePari", "libelle", "combinaison", "dividendePourUnEuro", "nombreGagnants", "miseBase"])
                    for p in sorted((ROOT / "data" / "raw" / "fr").glob("rapports-*.parquet"))], ignore_index=True)
    sg = rp[(rp.typePari == "SIMPLE_GAGNANT") & (rp.libelle.astype(str).str.lower() == "simple gagnant")]
    net = (sg.dividendePourUnEuro / 100 * sg.nombreGagnants * sg.miseBase / 100).groupby(sg.raceKey).first().rename("net_pool")
    d = d.merge(net, left_on="raceKey", right_index=True, how="inner").sort_values(["date", "rid", "numPmu"]).reset_index(drop=True)
    from racemodel import starts_of
    st, sz = starts_of(d.rid.values)
    y = d.y.values
    ll = {m: -np.add.reduceat(y * np.log(np.clip(d[m].values, 1e-12, 1)), st) for m in ("mkt_ref", "mkt_final", "fund", "end_ref", "end_final")}
    pool, yr, day = d.net_pool.values[st], d.year.values[st], d.date.values[st]
    bins = [0, 2e4, 5e4, 1e5, 2e5, 1e12]
    pb = np.digitize(pool, bins) - 1
    out = {}
    # last-minute upper bound: decide on the final odds with the final-odds model (the final odds are what pays)
    d["ret"] = np.where(d.y > 0, d.odf_rapport, 0.0)
    evf = d.end_final.values * d.odf_rapport.values - 1
    for lab, (a, b) in (("dev", DEV), ("test", TEST)):
        rows = []
        for i in range(len(bins) - 1):
            k = (pb == i) & (yr >= a) & (yr <= b)
            if not k.any():
                continue
            u, inv = np.unique(day[k], return_inverse=True)
            idx = np.random.default_rng(0).integers(0, len(u), (500, len(u)))
            dl = ll["end_final"][k] - ll["mkt_final"][k]
            bb = np.bincount(inv, dl)[idx].sum(1) / np.bincount(inv)[idx].sum(1)
            kr = np.repeat(k, sz)
            mm = kr & (evf > 0.2) & (evf < 1.0)
            rows.append({"net_win_pool_eur": f"{bins[i]:,.0f}-{bins[i + 1]:,.0f}".replace("1,000,000,000,000", "max"), "races": int(k.sum()),
                         **{f"logloss_{m}": round(float(ll[m][k].mean()), 4) for m in ll},
                         "end_final_minus_mkt_final": round(float(dl.mean()), 4), "ci95": [round(float(x), 4) for x in np.quantile(bb, [.025, .975])],
                         "end_ref_minus_mkt_ref": round(float((ll["end_ref"][k] - ll["mkt_ref"][k]).mean()), 4),
                         "last_minute_rule_ev>0.2": roi_summary(d.ret.values[mm])})
        out[lab] = rows
    rep["by_liquidity"] = out
    json.dump(rep, open(RES / "value_report.json", "w"), indent=1)
    for lab in out:
        print(lab)
        for r in out[lab]:
            print("  ", {k: v for k, v in r.items() if not k.startswith("logloss_")})


if __name__ == "__main__" and sys.argv[1:] == ["liquidity"]:
    liquidity()


def liquid_rule(min_pool=5e4):
    """The payout-aware 30-minute rule restricted to pools where our stake does not matter; threshold re-chosen on dev."""
    rep = json.load(open(RES / "value_report.json"))
    d = pd.read_parquet(PROC / "fr_preds.parquet")
    f = pd.read_parquet(PROC / "fr_feats.parquet", columns=["rid", "numPmu", "raceKey", "field"])
    d = d.merge(f, on=["rid", "numPmu"])
    d["year"] = d.date.dt.year
    d["t"] = d.specialite.astype(str).str.startswith("TROT").astype(float)
    rp = pd.concat([pd.read_parquet(p, columns=["raceKey", "typePari", "libelle", "combinaison", "dividendePourUnEuro", "nombreGagnants", "miseBase"])
                    for p in sorted((ROOT / "data" / "raw" / "fr").glob("rapports-*.parquet"))], ignore_index=True)
    sg = rp[(rp.typePari == "SIMPLE_GAGNANT") & (rp.libelle.astype(str).str.lower() == "simple gagnant")].copy()
    sg["numPmu"] = pd.to_numeric(sg.combinaison, errors="coerce")
    sg["div"] = sg.dividendePourUnEuro / 100
    net = (sg["div"] * sg.nombreGagnants * sg.miseBase / 100).groupby(sg.raceKey).first().rename("net_pool")
    d = d[(d.odr_rapport > 1) & (d.odf_rapport > 1)].merge(net, left_on="raceKey", right_index=True)
    d = d.merge(sg[["raceKey", "numPmu", "div"]], on=["raceKey", "numPmu"], how="left")
    d["ret"] = d["div"].fillna(0.0)
    p = d.end_ref.values
    lo = np.log(d.odr_rapport.values)
    X = np.c_[np.ones(len(d)), lo, np.log(p / d.mkt_ref.values), d.t, np.log(d.field), lo * d.t]
    dev, test = d.year.between(*DEV).values, d.year.between(*TEST).values
    wn = dev & (d.y.values > 0)
    beta, *_ = np.linalg.lstsq(X[wn], np.log(d.odf_rapport.values[wn]), rcond=None)
    smear = float(np.mean(np.exp(np.log(d.odf_rapport.values[wn]) - X[wn] @ beta)))
    ev = p * np.exp(X @ beta) * smear - 1
    liq = d.net_pool.values >= min_pool
    day = d.date.values
    dres, tres = {}, {}
    for pm in (0.0, 0.02, 0.05, 0.1):
        for tau in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
            k = f"p>={pm:g},ev>{tau:g}"
            m = liq & (p >= pm) & (ev > tau) & (ev < 1.0)
            dres[k] = roi_summary(d.ret.values[dev & m])
            tres[k] = roi_summary(d.ret.values[test & m], day[test & m])
    best = max(((k, r) for k, r in dres.items() if r["bets"] >= 2000), key=lambda kv: kv[1]["roi"])[0]
    rep["liquid_rule"] = {"min_pool_eur": min_pool, "chosen_on_dev": best, "dev_chosen": dres[best], "test_chosen": tres[best], "test_all": tres, "dev": dres}
    json.dump(rep, open(RES / "value_report.json", "w"), indent=1)
    print(f"liquid pools >= {min_pool:,.0f}: chosen {best}; dev {dres[best]}; test {tres[best]}")


if __name__ == "__main__" and sys.argv[1:] == ["liquid"]:
    liquid_rule()


def locked():
    """Hypothetical: the same 30-minute decisions (EV = p x ref odds - 1) paid at the 30-minute price, as if it could be locked
    (fixed odds / exchange), vs paid at the final pool odds. v1 and v2 predictions."""
    res = {}
    for tag, f in (("v1", PROC / "fr_preds_v1.parquet"), ("v2", PROC / "fr_preds_v2.parquet"), ("current", PROC / "fr_preds.parquet")):
        if not f.exists():
            continue
        g = pd.read_parquet(f)
        g["year"] = g.date.dt.year
        g = g[(g.odr_rapport > 1) & (g.odf_rapport > 1)]
        gd, gt = g.year.between(*DEV).values, g.year.between(*TEST).values
        ret_lock = np.where(g.y > 0, g.odr_rapport, 0.0)
        ret_pm = np.where(g.y > 0, g.odf_rapport, 0.0)
        ev = g.end_ref.values * g.odr_rapport.values - 1
        rows = []
        for tau in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
            m = (ev > tau) & (ev < 1.0)
            rows.append({"ev>": tau, "dev_locked": roi_summary(ret_lock[gd & m])["roi"], "test_locked": roi_summary(ret_lock[gt & m], g.date.values[gt & m]),
                         "test_parimutuel": roi_summary(ret_pm[gt & m])["roi"]})
        res[tag] = rows
    json.dump(res, open(RES / "locked_odds_report.json", "w"), indent=1)
    print(json.dumps(res)[:2000])


if __name__ == "__main__" and sys.argv[1:] == ["locked"]:
    locked()
