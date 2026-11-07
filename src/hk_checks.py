"""Hong Kong data checks (no modelling): coverage, duplicates, market efficiency, and whether 'rating' is pre-race.

Writes data/processed/hk_runners.parquet (HK only, one row per runner, cleaned) and prints the checks.
Run: python -I src/hk_checks.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hk"
OUT = ROOT / "data" / "processed"
KEY = ["race_date", "race_location", "race_no"]


def ftime(s):
    try:
        p = [float(x) for x in str(s).replace(".", ":").split(":")]
        return p[0] * 60 + p[1] + p[2] / 100 if len(p) == 3 else np.nan
    except Exception:
        return np.nan


def main():
    p = pd.read_csv(RAW / "performances.csv.gz", low_memory=False)
    p = p[p.race_country == "HK"].copy()
    p["race_date"] = pd.to_datetime(p.race_date, errors="coerce").dt.normalize()
    p["rating"] = pd.to_numeric(p.rating.astype(str).str.strip(), errors="coerce")
    p["finish_s"] = p.finish_time.map(ftime)
    n0 = len(p)
    p = p.drop_duplicates(["horse_id"] + KEY)
    print(f"HK runner rows {n0:,}, after dropping duplicate horse-race rows {len(p):,}")
    p["rid"] = p.groupby(KEY, sort=True).ngroup()
    g = p.groupby("rid")
    p["n_field"] = g.horse_id.transform("size")
    p["won"] = (p.final_placing == 1).astype(int)
    p["q"] = 1 / p.winning_odds
    p["over"] = g.q.transform("sum")
    races = g.agg(date=("race_date", "first"), n=("horse_id", "size"), winners=("won", "sum"), over=("q", "sum"),
                  odds_na=("winning_odds", lambda s: s.isna().mean()))
    print(f"races {len(races):,} ({races.date.min().date()} .. {races.date.max().date()}); median field {races.n.median():.0f}")
    print(f"races with exactly one winner {np.mean(races.winners == 1):.3f}, none {np.mean(races.winners == 0):.3f}, dead heats {np.mean(races.winners > 1):.3f}")
    ok = races[(races.winners == 1) & (races.odds_na == 0)]
    print(f"clean races (one winner, odds for all) {len(ok):,}; sum of 1/odds median {ok.over.median():.3f} "
          f"(IQR {ok.over.quantile(.25):.3f}-{ok.over.quantile(.75):.3f})")
    c = p[p.rid.isin(ok.index)].copy()
    c["pm"] = c.q / c.over  # market probability, margin removed proportionally
    c["fav"] = c.groupby("rid").winning_odds.rank(method="first") == 1
    for lab, yrs in (("1979-1999", (1979, 1999)), ("2000-2009", (2000, 2009)), ("2010-2018", (2010, 2018))):
        s = c[c.race_date.dt.year.between(*yrs)]
        w = s[s.won == 1]
        print(f"  {lab}: races {s.rid.nunique():6,}  favourite wins {s[s.fav].won.mean():.3f}  "
              f"market log-loss (winner) {-np.log(w.pm).mean():.4f}  uniform {np.log(s.groupby('rid').size()).mean():.4f}")
    s = c[c.race_date.dt.year >= 2010]
    bins = pd.cut(s.pm, [0, .02, .05, .1, .15, .2, .3, .4, .6, 1])
    cal = s.groupby(bins, observed=True).agg(n=("won", "size"), market=("pm", "mean"), actual=("won", "mean"))
    print("  calibration of the market 2010-18 (favourite-longshot bias check):\n" + cal.round(4).to_string())

    # is 'rating' the rating carried INTO the race? Ratings move after a race according to the result.
    h = p.sort_values(["horse_id", "race_date"]).copy()
    h["next_rating"] = h.groupby("horse_id").rating.shift(-1)
    h["next_place"] = h.groupby("horse_id").final_placing.shift(-1)
    h["d"] = h.next_rating - h.rating
    m = h.dropna(subset=["d", "final_placing", "next_place"])
    m = m[m.race_date.dt.year >= 2000]
    print(f"rating timing: corr(rating change t->t+1, placing in race t) {np.corrcoef(m.d, m.final_placing)[0, 1]:+.3f}, "
          f"with placing in race t+1 {np.corrcoef(m.d, m.next_place)[0, 1]:+.3f}  (pre-race rating => the first is the strong one)")
    print(f"winners' next rating change: mean {m[m.final_placing == 1].d.mean():+.2f} vs others {m[m.final_placing > 3].d.mean():+.2f}")

    # live odds (2016-18): how far before the off is the first snapshot, and does the last one match the final odds?
    lo = pd.read_csv(RAW / "live_odds.csv.gz")
    lo["race_date"] = pd.to_datetime(lo.race_date).dt.normalize()
    lo["capture_time"] = pd.to_datetime(lo.capture_time)
    lo = lo.sort_values("capture_time")
    first, last = lo.groupby(KEY).first(), lo.groupby(KEY).last()
    nsnap = lo.groupby(KEY).size()
    print(f"live odds: {len(nsnap):,} races, snapshots per race median {nsnap.median():.0f}")
    rows = []
    for k, r in last.iterrows():
        win = json.loads(r.data).get("win", {})
        f_win = json.loads(first.loc[k].data).get("win", {})
        for num, o in win.items():
            rows.append((*k, float(num), pd.to_numeric(o, errors="coerce"), pd.to_numeric(f_win.get(num), errors="coerce"), r.capture_time, first.loc[k].capture_time))
    lv = pd.DataFrame(rows, columns=KEY + ["horse_no", "last_odds", "first_odds", "t_last", "t_first"])
    j = p.merge(lv, on=KEY + ["horse_no"], how="inner")
    print(f"  matched {len(j):,} runners; last snapshot == final odds for {np.mean(np.isclose(j.last_odds, j.winning_odds)):.3f}; "
          f"median |log ratio| first vs final {np.nanmedian(np.abs(np.log(j.first_odds / j.winning_odds))):.3f}")
    lead = (j.drop_duplicates(KEY).t_first - j.drop_duplicates(KEY).race_date).dt.total_seconds() / 3600
    print(f"  first snapshot taken (hours after midnight of race day; negative = day before): median {lead.median():.1f}")
    keep = ["rid", "race_date", "race_location", "race_no", "race_class", "distance", "course", "track", "going", "horse_id", "horse_name",
            "horse_no", "draw", "rating", "actual_weight", "on_date_weight", "gears", "jockey_name", "trainer_name", "winning_odds",
            "final_placing", "finish_s", "length_behind_winner", "running_positions", "sections_time", "n_field", "won", "over"]
    p[keep].to_parquet(OUT / "hk_runners.parquet", index=False)
    print(f"-> hk_runners.parquet {len(p):,} rows")


if __name__ == "__main__":
    main()
