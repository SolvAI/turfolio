"""Download PMU's internet pool (E_SIMPLE_GAGNANT / E_SIMPLE_PLACE) for every race in our dataset, newest first, from PMU's own API:
  participants?specialisation=INTERNET        -> per runner: final ('direct') and 30-min reference odds of the internet pool, with timestamps
  rapports-definitifs?specialisation=INTERNET -> official e-Simple gagnant / e-Simple placé dividends with stake counts (nombreGagnants)
Output data/raw/pmu_internet/epool.jsonl, one line per race. Resumable (skips raceKeys already written). 2 workers, ~2-3 requests/s.
Run: python -I src/fr_epool.py [from YYYY-MM-DD] [workers, default 2]   |   python -I src/fr_epool.py --refetch
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "pmu_internet"
API = "https://online.turfinfo.api.pmu.fr/rest/client/61/programme"
UA = "Mozilla/5.0 (X11; Linux x86_64) research-crawler"
LOCK = threading.Lock()
KEEP = ("typePari", "rapport", "dateRapport", "favoris")


def get(url, tries=5):
    for k in range(tries):
        time.sleep(0.3)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                body = r.read()
                if r.status == 204 or not body.strip():  # no internet pool for this race (e.g. foreign meetings): not an error
                    return 204, None
                return r.status, json.loads(body.decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            if e.code in (204, 404):
                return e.code, None
            time.sleep(30 * (k + 1) if e.code in (403, 429, 503) else 5 * (k + 1))
            print(f"  HTTP {e.code} {url}", flush=True)
        except Exception as e:
            print(f"  {type(e).__name__} {url}", flush=True)
            time.sleep(10 * (k + 1))
    return None, None


def one(rk, out="epool.jsonl"):
    d, R, C = rk.split("_")
    base = f"{API}/{d[8:10]}{d[5:7]}{d[0:4]}/{R}/{C}"
    s1, p = get(base + "/participants?specialisation=INTERNET")
    s2, r = get(base + "/rapports-definitifs?specialisation=INTERNET")
    rec = dict(raceKey=rk, st=[s1, s2])
    if p:
        rec["runners"] = [dict(num=x.get("numPmu"), statut=x.get("statut"),
                               direct={k: (x.get("dernierRapportDirect") or {}).get(k) for k in KEEP},
                               ref={k: (x.get("dernierRapportReference") or {}).get(k) for k in KEEP}) for x in p.get("participants", [])]
    if isinstance(r, list):
        rec["rapports"] = [x for x in r if x.get("typePari") in ("E_SIMPLE_GAGNANT", "E_SIMPLE_PLACE")]
    with LOCK:
        with open(OUT / out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rk


def main(start="2014-01-01", workers="2"):
    OUT.mkdir(parents=True, exist_ok=True)
    r = pd.read_parquet(ROOT / "data" / "processed" / "fr_races.parquet", columns=["raceKey", "date"])
    keys = sorted(r.raceKey[r.date.astype(str) >= start].unique(), reverse=True)
    done = set()
    if (OUT / "epool.jsonl").exists():
        done = {json.loads(l)["raceKey"] for l in open(OUT / "epool.jsonl")}
    todo = [k for k in keys if k not in done]
    print(f"{len(keys)} races, {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(int(workers)) as ex:
        for i, rk in enumerate(ex.map(one, todo)):
            if i % 500 == 0:
                print(f"{rk}: {i}/{len(todo)} races, {(time.time() - t0) / 3600:.2f} h", flush=True)
    print("done", flush=True)


def refetch():
    """Refetch races whose earlier fetch failed (a None status) into epool_fix.jsonl; readers let a fix record override the original."""
    bad = [json.loads(l)["raceKey"] for l in open(OUT / "epool.jsonl") if None in json.loads(l)["st"]]
    fixed = {json.loads(l)["raceKey"] for l in open(OUT / "epool_fix.jsonl")} if (OUT / "epool_fix.jsonl").exists() else set()
    todo = [k for k in bad if k not in fixed]
    print(f"{len(bad)} failed records, {len(todo)} to refetch", flush=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda k: one(k, "epool_fix.jsonl"), todo))
    still = [json.loads(l)["raceKey"] for l in open(OUT / "epool_fix.jsonl") if None in json.loads(l)["st"]]
    print(f"done; still failing: {len(still)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--refetch"]:
        refetch()
    else:
        main(*sys.argv[1:])
