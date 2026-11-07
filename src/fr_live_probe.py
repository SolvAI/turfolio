"""Live check of what Canalturf's odds series really are: record, minute by minute around the off of every French race today,
  - PMU's live odds (participants endpoint: dernierRapportDirect / dernierRapportReference with their timestamps) and the race status,
  - the Canalturf odds page of the same race (its 10-minute series and its snapshot).
Afterwards we can see when each Canalturf label first appears in real time, whether its value is later revised, and which PMU price it equals.
Output: data/raw/canalturf/live/<date>.jsonl, one line per poll. Window: from 20 min before the scheduled start to 25 min after.
Run: python -I src/fr_live_probe.py [YYYY-MM-DD] [--pmu-only]   (waits until 10:30 Paris time)
  --pmu-only: daily forward recorder: PMU main + internet pools only, every 30 s, -20..+20 min, to data/raw/pmu_live/<date>.jsonl
"""
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_canalturf import BASE, LINK, parse_cotes, slug  # noqa: E402

PARIS = ZoneInfo("Europe/Paris")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "canalturf" / "live"
UA = "Mozilla/5.0 (X11; Linux x86_64) research-crawler"
PMU = "https://online.turfinfo.api.pmu.fr/rest/client/61/programme"


def fetch(url, js=False):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            t = r.read().decode("utf-8", errors="replace")
        return json.loads(t) if js else t
    except Exception as e:
        return {"error": f"{type(e).__name__} {e}"} if js else None


def now():
    return datetime.now(PARIS)


def main(*args):
    global OUT
    pmu_only = "--pmu-only" in args
    args = [a for a in args if not a.startswith("--")]
    day = args[0] if args else now().strftime("%Y-%m-%d")
    step, lo_m, hi_m = (30, -20, 20) if pmu_only else (60, -20, 25)
    if pmu_only:
        OUT = ROOT / "data" / "raw" / "pmu_live"
    OUT.mkdir(parents=True, exist_ok=True)
    start_at = datetime.fromisoformat(day + "T10:30").replace(tzinfo=PARIS)  # restarting later in the day starts at once
    while now() < start_at:
        time.sleep(min(600, (start_at - now()).total_seconds() + 1))
    ddmmyyyy = datetime.fromisoformat(day).strftime("%d%m%Y")
    prog = fetch(f"{PMU}/{ddmmyyyy}", js=True)
    races = []
    for r in prog["programme"]["reunions"]:
        if r.get("pays", {}).get("code") != "FRA":
            continue
        for c in r["courses"]:
            races.append(dict(R=r["numOfficiel"], C=c["numOrdre"], name=c["libelle"], hippo=r["hippodrome"]["libelleCourt"],
                              start=datetime.fromtimestamp(c["heureDepart"] / 1000, PARIS)))
    t = "" if pmu_only else (fetch(f"{BASE}/courses_archives.php?date={day}") or "")
    links = sorted({m.groups() for m in LINK.finditer(t) if m.group(1) == day})
    for x in races:
        c = [l for l in links if l[3].lower() == slug(x["name"])]
        x["ct"] = f"{BASE}/cotes/{day}/{c[0][1]}/{c[0][2]}_{c[0][3]}.html" if c else None
    print(f"{len(races)} French races, {sum(x['ct'] is not None for x in races)} with a Canalturf page", flush=True)
    for x in races:
        print(f"  R{x['R']}C{x['C']} {x['start']:%H:%M} {x['name']} -> {x['ct']}", flush=True)
    end = max(x["start"] for x in races) + timedelta(minutes=hi_m + 1)
    while now() < end:
        t0 = time.time()
        for x in races:
            m = (now() - x["start"]).total_seconds() / 60
            if not (lo_m <= m <= hi_m):
                continue
            base = f"{PMU}/{ddmmyyyy}/R{x['R']}/C{x['C']}"
            course = fetch(base, js=True)
            parts = fetch(base + "/participants", js=True)
            eparts = fetch(base + "/participants?specialisation=INTERNET", js=True)  # PMU's separate internet pool (E_SIMPLE_GAGNANT)
            rec = dict(poll=now().isoformat(), R=x["R"], C=x["C"], start=x["start"].isoformat(), mins_after_start=round(m, 2),
                       course={k: course.get(k) for k in ("statut", "categorieStatut", "departImminent", "arriveeDefinitive", "dureeCourse",
                                                          "heureDepart", "ordreArrivee", "error") if k in course},
                       pmu=[dict(num=p["numPmu"], direct=p.get("dernierRapportDirect"), ref=p.get("dernierRapportReference"))
                            for p in parts.get("participants", [])],
                       epmu=[dict(num=p["numPmu"], direct=p.get("dernierRapportDirect"), ref=p.get("dernierRapportReference"))
                             for p in eparts.get("participants", [])])
            if x["ct"]:
                page = fetch(x["ct"])
                rec["ct"] = parse_cotes(page) if page else None
            with open(OUT / f"{day}.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        time.sleep(max(5, step - (time.time() - t0)))
    print("done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
