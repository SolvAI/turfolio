"""France data checks (no modelling).

  1 coverage by year: races, runners, starters, odds coverage
  2 which odds field is the final price: compare with the official SIMPLE_GAGNANT payout of the winner; snapshot times vs the off
  3 are career stats (starts, wins, earnings, form string) as of the race, or a later snapshot (= leak)?
  4 market efficiency: pool margin, favourite win rate, calibration, market log-loss, by discipline
  5 which fields are post-race (must never be features)

Run: python -I src/fr_checks.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "fr"
D, R = "dernierRapportDirect", "dernierRapportReference"


def main():
    ru = pd.read_parquet(PROC / "fr_runners.parquet")
    ra = pd.read_parquet(PROC / "fr_races.parquet")
    rp = pd.concat([pd.read_parquet(p) for p in sorted(RAW.glob("rapports-*.parquet"))], ignore_index=True)
    print("runner columns:", [c for c in ru.columns])
    print("race columns:", [c for c in ra.columns])
    ru["year"] = ru.date.str[:4].astype(int)
    st = ru.statut.eq("PARTANT")
    print("\n1 COVERAGE")
    cov = ru[st].groupby("year").agg(runners=("raceKey", "size"), races=("raceKey", "nunique"),
                                     odds_direct=(f"{D}.rapport", lambda s: s.notna().mean()),
                                     odds_ref=(f"{R}.rapport", lambda s: s.notna().mean()),
                                     placing=("raw.ordreArrivee", lambda s: s.notna().mean()))
    print(cov.round(3).to_string())
    print("statut values:", ru.statut.value_counts().to_dict())

    print("\n2 ODDS SEMANTICS")
    sg = rp[rp.typePari.eq("SIMPLE_GAGNANT")][["raceKey", "combinaison", "dividendePourUnEuro"]].copy()
    sg["pay"] = sg.dividendePourUnEuro / 100
    w = ru[st & ru["raw.ordreArrivee"].eq(1)].copy()
    w["combinaison"] = w.numPmu.astype(int).astype(str)
    w = w.merge(sg, on=["raceKey", "combinaison"], how="inner")
    for k, lab in ((D, "direct"), (R, "reference")):
        o = w[f"{k}.rapport"]
        ok = o.notna() & w.pay.notna() & (w.pay > 0)
        rat = o[ok] / w.pay[ok]
        print(f"  {lab:9s}: winners with both {ok.sum():,}; odds/payout median {rat.median():.3f}; within 5 % {np.mean(np.abs(rat - 1) < .05):.3f}; "
              f"within 20 % {np.mean(np.abs(rat - 1) < .2):.3f}")
    start = pd.to_datetime(ra.set_index("raceKey").heureDepart, utc=True, errors="coerce")
    start = (start - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() * 1000
    for k, lab in ((D, "direct"), (R, "reference")):
        t = pd.to_numeric(ru[f"{k}.dateRapport"], errors="coerce")
        mins = (t - ru.raceKey.map(start)) / 60000
        q = mins.dropna().quantile([.05, .25, .5, .75, .95]).round(1).to_dict()
        print(f"  {lab:9s}: snapshot time minus scheduled off (minutes) quantiles {q}")

    print("\n3 ARE CAREER STATS AS OF THE RACE?")
    key = ru["idCheval"].astype(str) if "idCheval" in ru and ru.idCheval.notna().mean() > .9 else \
        ru.nom.astype(str) + "|" + ru.get("raw.nomPere", pd.Series("", index=ru.index)).astype(str) + "|" + ru.get("raw.nomMere", pd.Series("", index=ru.index)).astype(str)
    h = ru.assign(hk=key)[st].sort_values(["hk", "date"])
    h["d_n"] = h.groupby("hk").nombreCourses.diff()
    h["d_g"] = h.groupby("hk").gainsCarriere.diff()
    dd = h.d_n.dropna()
    print(f"  consecutive starts of the same horse: change in 'nombreCourses' = 0: {np.mean(dd == 0):.3f}, = 1: {np.mean(dd == 1):.3f}, "
          f"> 1: {np.mean(dd > 1):.3f}, < 0: {np.mean(dd < 0):.3f}")
    print(f"  earnings never decrease: {np.mean(h.d_g.dropna() >= 0):.3f}")
    # the form string should gain this race's result at the front at the next start
    h["next_mus"] = h.groupby("hk").musique.shift(-1)
    h["place"] = h["raw.ordreArrivee"]
    m = h.dropna(subset=["next_mus", "place"])
    m = m[m.place.between(1, 9)]
    lead = m.next_mus.str.extract(r"^\(?\d*\)?(\d)")[0]
    print(f"  next start's form string begins with this race's placing (1-9): {np.mean(lead.astype(float) == m.place):.3f}")
    first = h.groupby("hk").head(1)
    print(f"  first start in the data: nombreCourses == 0 for {np.mean(first.nombreCourses == 0):.3f} (horses debuting after 2014 should be 0)")

    print("\n4 MARKET")
    s = ru[st].copy()
    s = s[s.groupby("raceKey")[f"{D}.rapport"].transform(lambda x: x.notna().all())]
    s["won"] = s["raw.ordreArrivee"].eq(1).astype(int)
    s = s[s.groupby("raceKey").won.transform("sum") == 1]
    for k, lab in ((D, "direct"), (R, "reference")):
        if s[f"{k}.rapport"].isna().any():
            ss = s[s.groupby("raceKey")[f"{k}.rapport"].transform(lambda x: x.notna().all())]
        else:
            ss = s
        ss = ss.copy()
        ss["q"] = 1 / ss[f"{k}.rapport"]
        ss["over"] = ss.groupby("raceKey").q.transform("sum")
        ss["pm"] = ss.q / ss.over
        ss["fav"] = ss.groupby("raceKey")[f"{k}.rapport"].rank(method="first") == 1
        ss = ss.merge(ra[["raceKey", "discipline"]], on="raceKey", how="left")
        print(f"  {lab}: races {ss.raceKey.nunique():,}; sum 1/odds median {ss.drop_duplicates('raceKey').over.median():.3f}")
        for disc, g in ss.groupby("discipline"):
            if g.raceKey.nunique() < 2000:
                continue
            wn = g[g.won == 1]
            print(f"    {disc:13s} races {g.raceKey.nunique():6,} field {g.groupby('raceKey').size().median():.0f}  favourite wins {g[g.fav].won.mean():.3f}  "
                  f"market log-loss {-np.log(wn.pm).mean():.4f}  uniform {np.log(g.groupby('raceKey').size()).mean():.4f}")
        if k == D:
            bins = pd.cut(ss.pm, [0, .02, .05, .1, .15, .2, .3, .4, .6, 1])
            print("    calibration (all disciplines):\n" + ss.groupby(bins, observed=True).agg(n=("won", "size"), market=("pm", "mean"), actual=("won", "mean")).round(4).to_string())

    print("\n5 FIELDS THAT LOOK POST-RACE (never use as features)")
    sus = [c for c in ru.columns if any(t in c.lower() for t in ("arrivee", "temps", "reduction", "incident", "commentaire", "distancecheval", "ecart", "place", "rapport"))]
    print(" ", sus)


if __name__ == "__main__":
    main()
