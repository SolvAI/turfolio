"""Fresh hold-out: the French races after the HF archive's end (2026-05-05), downloaded from PMU's own API and rebuilt in the archive's
normalized format, so that the frozen pipeline can run on them unchanged.

The archive was built from the same API (dataset_summary.json): programme per day (French meetings only), then per race the participants
and rapports-definitifs endpoints. A normalized record = a few top-level fields + 'raw' = the API object (src/fr_extract.py flattens it
the same way: top-level copies win over raw.*).
  download FROM TO OUTDIR   -> OUTDIR/{courses,participants}.jsonl (normalized) and OUTDIR/rapports.jsonl (one row per dividend)
  compare OUTDIR            -> field-by-field agreement with the archive on overlapping races (validation of the rebuild)
  build OUTDIR NEWROOT      -> NEWROOT/data/processed/fr_runners|fr_races.parquet = archive + fresh races, NEWROOT/data/raw/fr/rapports-fresh.parquet
  evaluate NEWROOT [LO HI]  -> accuracy of the frozen models on the fresh window (after the pipeline has run in NEWROOT)
Run: python -I src/fr_fresh.py download 2026-05-06 2026-11-05 data/raw/fr_fresh
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_epool import API, get  # noqa: E402
from fr_extract import flat, tidy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()
COURSE_TOP = ["libelle", "libelleCourt", "montantPrix", "distance", "distanceUnit", "corde", "discipline", "specialite", "categorieParticularite",
              "conditionAge", "conditionSexe", "nombreDeclaresPartants", "statut", "arriveeDefinitive", "ordreArrivee", "incidents",
              "rapportsDefinitifsDisponibles", "conditions"]
PART_TOP = ["nom", "idCheval", "age", "sexe", "race", "statut", "placeCorde", "poids", "handicapDistance", "driver", "jockey", "entraineur",
            "proprietaire", "eleveur", "musique", "nombreCourses", "nombreVictoires", "nombrePlaces"]
PART_TAIL = ["deferre", "oeilleres", "incident", "dernierRapportDirect", "dernierRapportReference"]


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def race(job, out):
    d, ddmmyyyy, reu, c = job
    R, C = c["numReunion"], c["numOrdre"]
    rk = f"{d}_R{R}_C{C}"
    ids = dict(date=d, ddmmyyyy=ddmmyyyy, raceKey=rk)
    course = dict(ids, reunionKey=f"{d}_R{reu['numExterne']}", numReunion=R, numExterneReunion=c.get("numExterneReunion", reu["numExterne"]),
                  numCourse=C, numOrdre=C, heureDepart=iso(c["heureDepart"]),
                  **{k: c.get(k) for k in COURSE_TOP[:2]}, hippodromeCode=c["hippodrome"]["codeHippodrome"], hippodrome=c["hippodrome"]["libelleCourt"],
                  **{k: c.get(k) for k in COURSE_TOP[2:]}, raw=c)
    for k, v in (("incidents", []), ("ordreArrivee", []), ("arriveeDefinitive", False)):  # the archive's defaults when the API omits them
        if course[k] is None:
            course[k] = v
    s1, p = get(f"{API}/{ddmmyyyy}/R{R}/C{C}/participants")
    s2, rp = get(f"{API}/{ddmmyyyy}/R{R}/C{C}/rapports-definitifs")
    parts = []
    for x in (p or {}).get("participants", []):
        g = x.get("gainsParticipant") or {}
        parts.append(dict(ids, numReunion=R, numExterneReunion=course["numExterneReunion"], numCourse=C, hippodrome=reu["hippodrome"]["libelleCourt"],
                          numPmu=x.get("numPmu"), **{k: x.get(k) for k in PART_TOP},
                          gainsCarriere=g.get("gainsCarriere"), gainsVictoires=g.get("gainsVictoires"), gainsPlace=g.get("gainsPlace"),
                          **{k: x.get(k) for k in PART_TAIL}, raw=x))
    divs = [dict(date=d, raceKey=rk, numReunion=R, numCourse=C, hippodrome=course["hippodrome"], typePari=b.get("typePari"), miseBase=b.get("miseBase"),
                 libelle=y.get("libelle"), dividende=y.get("dividende"), dividendePourUnEuro=y.get("dividendePourUnEuro"),
                 combinaison=y.get("combinaison"), nombreGagnants=y.get("nombreGagnants"))
            for b in (rp if isinstance(rp, list) else []) for y in b.get("rapports") or []]
    with LOCK:
        for name, rows in (("courses", [course]), ("participants", parts), ("rapports", divs)):
            with open(out / f"{name}.jsonl", "a") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out / "status.jsonl", "a") as f:
            f.write(json.dumps(dict(raceKey=rk, st=[s1, s2], n=len(parts), divs=len(divs))) + "\n")
    return rk


def download(lo, hi, outdir, workers="8"):
    out = Path(outdir) if Path(outdir).is_absolute() else ROOT / outdir
    out.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["raceKey"] for l in open(out / "status.jsonl")} if (out / "status.jsonl").exists() else set()
    days = [date.fromisoformat(lo) + timedelta(days=i) for i in range((date.fromisoformat(hi) - date.fromisoformat(lo)).days + 1)]
    jobs = []
    for dd in days:
        d, ddmmyyyy = dd.isoformat(), dd.strftime("%d%m%Y")
        s, prog = get(f"{API}/{ddmmyyyy}")
        if not prog:
            print(f"{d}: programme status {s}", flush=True)
            continue
        for reu in prog["programme"]["reunions"]:
            if (reu.get("pays") or {}).get("code") != "FRA":
                continue
            for c in reu["courses"]:
                if f"{d}_R{c['numReunion']}_C{c['numOrdre']}" not in done:
                    jobs.append((d, ddmmyyyy, reu, c))
    print(f"{len(days)} days, {len(jobs)} races to fetch", flush=True)
    with ThreadPoolExecutor(int(workers)) as ex:
        for i, rk in enumerate(ex.map(lambda j: race(j, out), jobs)):
            if i % 500 == 0:
                print(f"  {rk}: {i}/{len(jobs)}", flush=True)
    print("done", flush=True)


def flat_rows(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        raw = r.pop("raw", None) or {}
        row = flat(r)
        for k, v in flat(raw, "raw.").items():
            if k[4:] not in row:
                row[k] = v
        rows.append(row)
    return tidy(pd.DataFrame(rows))


def compare(outdir):
    """Agreement with the archive on races present in both (the overlap window), for every column the archive has."""
    out = ROOT / outdir
    for name, key in (("races", ["raceKey"]), ("runners", ["raceKey", "numPmu"])):
        f = flat_rows(out / f"{'courses' if name == 'races' else 'participants'}.jsonl")
        a = pd.read_parquet(ROOT / "data" / "processed" / f"fr_{name}.parquet", filters=[("raceKey", "in", list(f.raceKey.unique()))])
        m = a.merge(f, on=key, suffixes=("_a", "_f"))
        print(f"{name}: archive {len(a)}, fresh {len(f)}, matched {len(m)}; columns only in archive: "
              f"{sorted(set(a.columns) - set(f.columns))}; only in fresh: {sorted(set(f.columns) - set(a.columns))}")
        bad = []
        for c in sorted(set(a.columns) & set(f.columns) - set(key)):
            x, y = m[c + "_a"], m[c + "_f"]
            same = (x.astype(str) == y.astype(str)) | (x.isna() & y.isna())
            num = pd.to_numeric(x.astype(object).map(lambda v: float(v) if isinstance(v, (bool, int, float)) else v), errors="coerce").astype(float), pd.to_numeric(y.astype(object).map(lambda v: float(v) if isinstance(v, (bool, int, float)) else v), errors="coerce").astype(float)
            same |= (num[0] - num[1]).abs() < 1e-9
            if same.mean() < 0.999:
                ex = m.loc[~same, [c + "_a", c + "_f"]].head(2).values.tolist()
                bad.append((c, round(float(same.mean()), 4), round(float(x.notna().mean()), 3), round(float(y.notna().mean()), 3), ex))
        for b in bad:
            print(f"   {b[0]}: agree {b[1]}, non-null archive {b[2]} fresh {b[3]}, e.g. {str(b[4])[:160]}")


def build(outdir, newroot):
    out, new = ROOT / outdir, Path(newroot)
    (new / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (new / "data" / "raw" / "fr").mkdir(parents=True, exist_ok=True)
    for name, src in (("races", "courses"), ("runners", "participants")):
        a = pd.read_parquet(ROOT / "data" / "processed" / f"fr_{name}.parquet")
        f = flat_rows(out / f"{src}.jsonl")
        f = f[~f.raceKey.isin(set(a.raceKey))]
        f = f[[c for c in f.columns if c in a.columns]]  # same columns as the archive (a few new API fields are dropped)
        df = tidy(pd.concat([a, f], ignore_index=True))
        df.to_parquet(new / "data" / "processed" / f"fr_{name}.parquet", index=False)
        print(f"{name}: archive {len(a):,} + fresh {len(f):,} rows ({f.raceKey.nunique():,} races, {f.date.min()}..{f.date.max()})", flush=True)
    r = pd.DataFrame([json.loads(l) for l in open(out / "rapports.jsonl")])
    r.to_parquet(new / "data" / "raw" / "fr" / "rapports-fresh.parquet", index=False)
    print(f"rapports: {len(r):,} rows", flush=True)


def evaluate(newroot, lo="2026-05-06", hi="2026-12-31"):
    """Accuracy on the fresh window from NEWROOT/data/processed/fr_preds.parquet: winner log-loss and top-pick hit rate of each model and
    market, and paired day-bootstrap CIs of the key differences (the claims of README.md, re-tested on races never used for anything)."""
    import numpy as np
    d = pd.read_parquet(Path(newroot) / "data" / "processed" / "fr_preds.parquet")
    d = d[d.date.astype(str).str[:10].between(lo, hi)]
    cols = ["mkt_ref", "mkt_final", "fund", "end_ref", "end_final"]
    d = d[d[cols].notna().all(1).groupby(d.rid).transform("all")]
    g = d.groupby("rid")
    w = d[d.y > 0]
    ll = {c: -(w.y * np.log(w[c].clip(1e-12))).groupby(w.rid).sum() for c in cols}
    top = {c: d.loc[g[c].idxmax()].set_index("rid").y.gt(0) for c in cols}
    day = d.groupby("rid").date.first().reindex(ll[cols[0]].index).values
    u, inv = np.unique(day, return_inverse=True)
    idx = np.random.default_rng(0).integers(0, len(u), (1000, len(u)))
    cnt = np.bincount(inv)
    rep = {"window": [lo, hi], "races": int(len(ll[cols[0]])), "days": int(len(u)),
           "logloss": {c: round(float(ll[c].mean()), 4) for c in cols}, "top_pick_hit": {c: round(float(top[c].mean()), 4) for c in cols}}
    for a, b in (("fund", "mkt_ref"), ("end_ref", "mkt_ref"), ("end_ref", "mkt_final"), ("end_final", "mkt_final")):
        dl = (ll[a] - ll[b]).values
        bs = np.bincount(inv, dl)[idx].sum(1) / cnt[idx].sum(1)
        rep[f"{a}_minus_{b}"] = dict(diff=round(float(dl.mean()), 4), ci95=[round(float(x), 4) for x in np.quantile(bs, [.025, .975])])
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(Path(newroot) / "results" / f"fresh_eval_{lo}_{hi}.json", "w"), indent=1)


if __name__ == "__main__":
    {"download": download, "compare": compare, "build": build, "evaluate": evaluate}[sys.argv[1]](*sys.argv[2:])
