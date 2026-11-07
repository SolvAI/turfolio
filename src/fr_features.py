"""France: pre-race feature table, one row per starter (statut PARTANT) -> data/processed/fr_feats.parquet

Sources of information, all known before the off:
  programme   what PMU publishes before the race: age, sex, breed, draw, distance handicap, weight, official handicap
              rating, shoes/blinkers, trainer's opinion, supplement, career starts/wins/places/earnings (as of the race),
              the form string ("musique", which also covers races outside this data)
  history     computed here from earlier races in the data only (strictly earlier start time for the horse, strictly
              earlier days for people): days since last run, last placings, speed (trot: time per km vs the field),
              beaten lengths (flat/jumps), class and distance changes, how the market rated the horse before, results vs
              market expectation; exponentially weighted win rates and actual/expected of driver/jockey, trainer, sire, owner
  market      final pool odds and reference odds (~30 min before the off) of THIS race: kept in separate columns
              (mkt_*), never mixed into the fundamental features
Post-race fields (finishing order, times, margins, incidents, comments) are only used as history for LATER races.
"""
import re
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
HALF_LIFE = 180.0  # days, for people / sire statistics

LENGTHS = {"NEZ": 0.05, "CTE TETE": 0.1, "TETE": 0.2, "CTE ENC.": 0.25, "ENCOLURE": 0.3, "Ex-Aequo": 0.0, "LOIN": 25.0}


def lengths(lbl):
    if lbl is None or lbl != lbl:
        return np.nan
    if lbl in LENGTHS:
        return LENGTHS[lbl]
    m = re.match(r"^(\d+)?\s*L?\s*(1/4|1/2|3/4)?\s*L?$", lbl.replace(" L ", " ").strip())
    if not m:
        return np.nan
    whole = float(m.group(1)) if m.group(1) else 0.0
    frac = {"1/4": .25, "1/2": .5, "3/4": .75}.get(m.group(2), 0.0)
    return whole + frac


def parse_musique(s):
    """'4p6p5p173p' -> list of (result, discipline, season_break_before). Two digits at a token boundary are a year marker."""
    out, i, brk = [], 0, False
    if not isinstance(s, str):
        return out
    s = s.replace("(", "").replace(")", "")
    while i < len(s) - 1:
        if s[i].isdigit() and s[i + 1].isdigit():
            brk, i = True, i + 2
            continue
        out.append((s[i], s[i + 1].lower(), brk))
        brk, i = False, i + 2
    return out


def musique_feats(series):
    rows = []
    for s in series.values:
        t = parse_musique(s)
        pl = [10.0 if r == "0" else float(r) if r.isdigit() else np.nan for r, _, _ in t]
        k5 = pl[:5]
        first_break = next((j for j, (_, _, b) in enumerate(t) if b), len(t))
        rows.append((
            len(t),
            pl[0] if pl else np.nan,
            np.nanmean(pl[:3]) if any(x == x for x in pl[:3]) else np.nan,
            np.nanmean(k5) if any(x == x for x in k5) else np.nan,
            sum(1 for x in k5 if x == 1), sum(1 for x in k5 if x == x and x <= 3),
            sum(1 for r, _, _ in t[:5] if r == "D"), sum(1 for r, _, _ in t[:5] if not r.isdigit() and r != "D"),
            first_break,
            t[0][1] if t else None,
        ))
    cols = ["mus_n", "mus_last", "mus_avg3", "mus_avg5", "mus_w5", "mus_top3_5", "mus_dq5", "mus_fail5", "mus_runs_this_season", "mus_last_disc"]
    return pd.DataFrame(rows, columns=cols, index=series.index)


@numba.njit(cache=True)
def ew_before(ent, day, x, tau):
    """Rows sorted by (ent, day), one row per entity-day with that day's totals x (n x k).
    Returns the decayed sums as of the START of each day (only earlier days)."""
    n, k = x.shape
    out = np.zeros((n, k))
    s = np.zeros(k)
    for i in range(n):
        if i == 0 or ent[i] != ent[i - 1]:
            s[:] = 0.0
        else:
            f = np.exp(-(day[i] - day[i - 1]) / tau)
            s *= f
        out[i] = s
        s += x[i]
    return out


def people_stats(d, col, tag, base):
    """EW starts / wins / market-expected wins of an entity (driver, trainer, ...), as of the race day (earlier days only)."""
    g = d[[col, "day", "won", "exp_win"]].copy()
    g["n"] = 1.0
    a = g.groupby([col, "day"], sort=True)[["n", "won", "exp_win"]].sum().reset_index()
    codes = a[col].astype("category").cat.codes.values.astype(np.int64)
    v = ew_before(codes, a.day.values.astype(np.float64), a[["n", "won", "exp_win"]].values.astype(np.float64), HALF_LIFE / np.log(2))
    a[f"{tag}_n"], w, e = v[:, 0], v[:, 1], v[:, 2]
    a[f"{tag}_wr"] = (w + 10 * base) / (a[f"{tag}_n"] + 10)
    a[f"{tag}_ae"] = (w + 2.0) / (e + 2.0)
    return d.merge(a[[col, "day", f"{tag}_n", f"{tag}_wr", f"{tag}_ae"]], on=[col, "day"], how="left")


def main():
    cols = ["date", "raceKey", "numPmu", "nom", "age", "sexe", "race", "statut", "placeCorde", "handicapDistance", "driver", "entraineur",
            "proprietaire", "eleveur", "musique", "nombreCourses", "nombreVictoires", "nombrePlaces", "gainsCarriere", "gainsVictoires",
            "gainsPlace", "deferre", "oeilleres", "incident", "raw.driverChange", "raw.indicateurInedit", "raw.gainsParticipant.gainsAnneeEnCours",
            "raw.gainsParticipant.gainsAnneePrecedente", "raw.nomPere", "raw.nomMere", "raw.nomPereMere", "raw.ordreArrivee", "raw.jumentPleine",
            "raw.engagement", "raw.supplement", "raw.reductionKilometrique", "raw.handicapPoids", "raw.distanceChevalPrecedent.libelleCourt",
            "raw.poidsConditionMonte", "raw.tauxReclamation", "raw.nombrePlacesSecond", "raw.nombrePlacesTroisieme", "raw.handicapValeur",
            "raw.paysEntrainement", "raw.avisEntraineur", "raw.pays", "raw.poidsConditionMonteChange",
            "dernierRapportDirect.rapport", "dernierRapportReference.rapport", "dernierRapportDirect.dateRapport", "dernierRapportReference.dateRapport"]
    d = pd.read_parquet(PROC / "fr_runners.parquet", columns=cols)
    d.columns = [c.replace("raw.gainsParticipant.", "").replace("raw.", "").replace("dernierRapportDirect.", "odf_").replace("dernierRapportReference.", "odr_")
                 for c in d.columns]
    rc = ["raceKey", "reunionKey", "heureDepart", "hippodromeCode", "distance", "discipline", "specialite", "categorieParticularite", "conditionAge",
          "conditionSexe", "montantPrix", "nombreDeclaresPartants", "corde", "raw.typePiste", "raw.penetrometre.valeurMesure",
          "raw.penetrometre.intitule", "numReunion", "numCourse", "raw.paris", "conditions", "raw.categorieStatut", "raw.courseTrackee"]
    r = pd.read_parquet(PROC / "fr_races.parquet", columns=rc)
    r.columns = [c.replace("raw.penetrometre.", "going_").replace("raw.", "") for c in r.columns]
    r = r[r.categorieStatut.ne("ANNULEE")]
    r["quinte"] = r.paris.fillna("").str.contains("QUINTE").astype(np.int8)
    r["autostart"] = (r.categorieParticularite.eq("AUTOSTART") | r.conditions.fillna("").str.lower().str.contains("autostart")).astype(np.int8)
    r["ts"] = pd.to_datetime(r.heureDepart, utc=True, errors="coerce").dt.tz_localize(None)
    r["hour"] = r.ts.dt.hour + r.ts.dt.minute / 60
    r["going_val"] = pd.to_numeric(r.going_valeurMesure, errors="coerce")
    r = r.drop(columns=["paris", "conditions", "heureDepart", "categorieStatut", "going_valeurMesure"])
    d = d[d.statut.eq("PARTANT")].merge(r, on="raceKey", how="inner")
    d["date"] = pd.to_datetime(d.date)
    d["day"] = (d.date - pd.Timestamp("2014-01-01")).dt.days
    d = d.sort_values(["ts", "raceKey", "numPmu"]).reset_index(drop=True)
    print(f"starters {len(d):,} in {d.raceKey.nunique():,} races", flush=True)

    # ---------------------------------------------------------------- outcomes (targets; history for later races only)
    d["pos"] = d.ordreArrivee
    d["won"] = (d.pos == 1).astype(np.int8)
    d["dq"] = d.incident.fillna("").str.startswith("DISQUALIFIE").astype(np.int8)
    d["fail"] = d.incident.isin(["ARRETE", "TOMBE", "DEROBE", "RESTE_AU_POTEAU", "DISTANCE"]).astype(np.int8)
    d["field"] = d.groupby("raceKey").numPmu.transform("size")
    d["n_winners"] = d.groupby("raceKey").won.transform("sum")
    d["pos_pct"] = (d.pos - 1) / (d.field - 1).clip(lower=1)
    d["rk"] = d.reductionKilometrique / 1000.0  # seconds per km (trot)
    d["rk_rel"] = d.rk - d.groupby("raceKey").rk.transform("median")
    d = d.rename(columns={"distanceChevalPrecedent.libelleCourt": "gap_lbl"})
    d["gap"] = d.gap_lbl.map(lengths)
    o = d.sort_values(["raceKey", "pos"])
    o["bl"] = o.groupby("raceKey").gap.cumsum().where(o.pos > 1, 0.0)  # beaten lengths behind the winner
    d["bl"] = o.bl.reindex(d.index).where(d.pos.notna())

    # ---------------------------------------------------------------- market of THIS race (benchmark / stacker input only)
    for k, src in (("f", "odf_rapport"), ("r", "odr_rapport")):
        q = 1 / d[src].where(d[src] > 1.0)
        full = q.notna().groupby(d.raceKey).transform("all")
        d[f"mkt_q{k}"] = q
        d[f"mkt_p{k}"] = (q / q.groupby(d.raceKey).transform("sum")).where(full)
        d[f"mkt_over{k}"] = q.groupby(d.raceKey).transform("sum").where(full)
    d["exp_win"] = (d.mkt_qf / d.mkt_qf.groupby(d.raceKey).transform("sum")).fillna(1 / d.field)  # for history

    # ---------------------------------------------------------------- programme features
    d["hk"] = d.nom.str.upper().str.strip() + "|" + d.nomPere.fillna("").str.upper() + "|" + d.nomMere.fillna("").str.upper()
    d["is_trot"] = d.specialite.str.startswith("TROT").astype(np.int8)
    d["n_starts"] = d.nombreCourses
    d["wins"] = d.nombreVictoires
    d["win_rate"] = (d.nombreVictoires + 0.5) / (d.nombreCourses + 5)
    d["place_rate"] = (d.nombrePlaces + 1.5) / (d.nombreCourses + 5)
    d["earn_log"] = np.log1p(d.gainsCarriere.fillna(0) / 100)
    d["earn_per_start"] = np.log1p(d.gainsCarriere.fillna(0) / 100 / (d.nombreCourses + 1))
    d["earn_year_log"] = np.log1p(d.gainsAnneeEnCours.fillna(0) / 100)
    d["earn_prev_year_log"] = np.log1p(d.gainsAnneePrecedente.fillna(0) / 100)
    d["prize_log"] = np.log1p(d.montantPrix.fillna(0))
    d["dist_handicap"] = d.handicapDistance - d.groupby("raceKey").handicapDistance.transform("min")
    d["weight"] = d.handicapPoids / 10.0
    d["weight_rel"] = d.weight - d.groupby("raceKey").weight.transform("mean")
    d["supplement"] = d.supplement.fillna(0)
    d["foreign_trained"] = d.paysEntrainement.notna().astype(np.int8)
    d["claim_log"] = np.log1p(d.tauxReclamation.fillna(0))
    for c in ("driverChange", "indicateurInedit", "jumentPleine", "engagement", "poidsConditionMonteChange"):
        d[c] = d[c].astype(float)
    d = pd.concat([d, musique_feats(d.musique)], axis=1)
    d["mus_same_disc"] = (d.mus_last_disc == d.specialite.map({"TROT_ATTELE": "a", "TROT_MONTE": "m", "PLAT": "p"})).astype(float)

    # ---------------------------------------------------------------- horse history inside the data (earlier starts only)
    d = d.sort_values(["hk", "ts"]).reset_index(drop=True)
    gh = d.groupby("hk", sort=False)
    d["h_n"] = gh.cumcount()
    prev_ts = gh.ts.shift(1)
    d["days_since"] = (d.ts - prev_ts).dt.total_seconds() / 86400
    for c in ("pos_pct", "rk_rel", "bl", "won", "dq", "exp_win", "prize_log", "distance", "rk"):
        for lag in (1, 2, 3):
            d[f"_{c}_{lag}"] = gh[c].shift(lag)
    d["h_last_pos"] = d._pos_pct_1
    d["h_avg_pos3"] = d[["_pos_pct_1", "_pos_pct_2", "_pos_pct_3"]].mean(axis=1)
    d["h_last_rk_rel"] = d._rk_rel_1
    d["h_best_rk_rel3"] = d[["_rk_rel_1", "_rk_rel_2", "_rk_rel_3"]].min(axis=1)
    d["h_last_rk"] = d._rk_1
    d["h_best_rk3"] = d[["_rk_1", "_rk_2", "_rk_3"]].min(axis=1)
    d["h_last_bl"] = d._bl_1
    d["h_avg_bl3"] = d[["_bl_1", "_bl_2", "_bl_3"]].mean(axis=1)
    d["h_dq3"] = d[["_dq_1", "_dq_2", "_dq_3"]].sum(axis=1, min_count=1)
    d["h_last_mkt"] = d._exp_win_1
    d["h_avg_mkt3"] = d[["_exp_win_1", "_exp_win_2", "_exp_win_3"]].mean(axis=1)
    d["h_class_change"] = d.prize_log - d._prize_log_1
    d["h_dist_change"] = d.distance - d._distance_1
    resid = (d.won - d.exp_win)
    d["h_ae"] = (gh.won.cumsum() - d.won + 1.0) / (gh.exp_win.cumsum() - d.exp_win + 1.0)  # actual / expected before this start
    d["h_resid_sum"] = resid.groupby(d.hk).cumsum() - resid
    d["h_same_track"] = d.groupby(["hk", "hippodromeCode"]).cumcount()
    d["h_same_driver"] = d.groupby(["hk", "driver"]).cumcount()
    d = d.drop(columns=[c for c in d.columns if c.startswith("_")])

    # ---------------------------------------------------------------- people & sire (EW, earlier days only)
    base = float(1 / d.field.mean())
    for col, tag in (("driver", "drv"), ("entraineur", "trn"), ("nomPere", "sire"), ("proprietaire", "own")):
        d[col] = d[col].fillna("?")
        d = people_stats(d, col, tag, base)
    d["drv_trn"] = d.driver + "|" + d.entraineur
    d = people_stats(d, "drv_trn", "dt", base)

    # ---------------------------------------------------------------- relative to the field
    rel = ["win_rate", "place_rate", "earn_per_start", "earn_year_log", "earn_log", "n_starts", "age", "mus_avg3", "mus_avg5", "h_avg_pos3",
           "h_best_rk_rel3", "h_best_rk3", "h_avg_bl3", "h_avg_mkt3", "h_last_mkt", "drv_wr", "drv_ae", "trn_wr", "trn_ae", "sire_ae", "handicapValeur",
           "weight", "dist_handicap", "days_since", "h_ae", "own_ae", "dt_ae", "placeCorde"]
    gr = d.groupby("raceKey")
    for c in rel:
        d[f"{c}_rk"] = gr[c].rank(pct=True)
        d[f"{c}_dm"] = d[c] - gr[c].transform("mean")
    d = d.sort_values(["ts", "raceKey", "numPmu"]).reset_index(drop=True)
    d["rid"] = d.groupby("raceKey", sort=False).ngroup()
    for c in ("sexe", "race", "deferre", "oeilleres", "avisEntraineur", "discipline", "specialite", "categorieParticularite", "conditionAge",
              "conditionSexe", "corde", "typePiste", "going_intitule", "hippodromeCode", "mus_last_disc", "pays"):
        d[c] = d[c].astype("category")
    drop = ["nom", "statut", "musique", "incident", "ordreArrivee", "reductionKilometrique", "gap_lbl", "gap", "nomMere", "nomPereMere", "eleveur",
            "odf_dateRapport", "odr_dateRapport", "paysEntrainement", "tauxReclamation", "handicapPoids", "nombreVictoires", "nombreCourses",
            "gainsCarriere", "gainsVictoires", "gainsPlace", "gainsAnneeEnCours", "gainsAnneePrecedente"]
    d = d.drop(columns=[c for c in drop if c in d.columns])
    d.to_parquet(PROC / "fr_feats.parquet", index=False)
    print(f"-> fr_feats.parquet {d.shape}", flush=True)


if __name__ == "__main__":
    main()
