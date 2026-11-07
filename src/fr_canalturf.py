"""Download Canalturf odds-evolution pages for the French races in our dataset (2022-01 -> end of data), newest day first.

For each day: fetch courses_archives.php?date=D, take every race link, match it to our races of that day by slugified race name
(venue slug as tie-break), fetch /cotes/<date>/<venue>/<id>_<slug>.html, and store
  data/raw/canalturf/html/<id>.html.gz         raw page (re-parse later if needed)
  data/raw/canalturf/odds.jsonl                one line per race: ids, snapshot time, per-horse {num, name, ops{PMU,BetClic,...}, series[[hh:mm, v]]}
  data/raw/canalturf/days.jsonl                one line per day: all race links seen + which of our raceKeys were matched / unmatched
Resumable: days already in days.jsonl are skipped. Polite: ~1 request/s with jitter, backoff on errors.
Run: python -I src/fr_canalturf.py [start YYYY-MM-DD] [end YYYY-MM-DD]
"""
import gzip
import html as H
import json
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "canalturf"
BASE = "https://www.canalturf.com"
UA = "Mozilla/5.0 (X11; Linux x86_64) research-crawler (1 req/s)"
DELAY = 1.0


def slug(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def get(url, tries=5):
    for k in range(tries):
        time.sleep(DELAY + random.random() * 0.5)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "fr-FR,fr;q=0.9"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, ""
            wait = 30 * (k + 1) if e.code in (429, 503) else 5 * (k + 1)
            print(f"  HTTP {e.code} {url} -> wait {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:  # network hiccup
            print(f"  {type(e).__name__} {url} -> retry", flush=True)
            time.sleep(10 * (k + 1))
    return None, ""


LINK = re.compile(r"/pronostics-PMU/(\d{4}-\d\d-\d\d)/([^/\"']+)/(\d+)_([^/\"']+)\.html")
PT = re.compile(r'\{\s*"elapsed":\s*"([^"]*)",\s*"value":\s*([0-9.]*)\s*\}')


def parse_cotes(t):
    snap = re.findall(r"Cote à (\d{1,2})h(\d\d)", t)
    # points can have an empty value ('"value": }'): kept as None
    series = {k: [dict(elapsed=e, value=float(x) if x else None) for e, x in PT.findall(v)]
              for k, v in re.findall(r"var cote_data_(\d+) = \[(.*?)\];", t, flags=re.S)}
    horses = []
    for blk in re.split(r'<div class="row mar-top">', t)[1:]:
        m = re.search(r'badge-primary">(\d+)</span><strong>\s*([^<]+)</strong>', blk)
        if not m:
            continue
        ops = {o: float(v) for o, v in re.findall(r"<br/>\s*([A-Za-z]+) : ([0-9]+(?:\.[0-9]+)?)", blk)}
        el = re.search(r'id="mline_cotes_(\d+)"', blk)
        s = series.get(el.group(1)) if el else None
        horses.append(dict(num=int(m.group(1)), name=H.unescape(m.group(2).strip()), ops=ops,
                           series=[[d["elapsed"], d["value"]] for d in s] if s else None))
    return dict(snap=f"{snap[0][0]}:{snap[0][1]}" if snap else None, horses=horses)


def main(start="2022-01-01", end=None):
    (OUT / "html").mkdir(parents=True, exist_ok=True)
    r = pd.read_parquet(ROOT / "data" / "processed" / "fr_races.parquet", columns=["date", "raceKey", "libelle", "hippodrome", "numReunion", "numCourse"])
    r["d"] = pd.to_datetime(r.date).dt.date.astype(str)
    end = end or r.d.max()
    by_day = {d: g for d, g in r[(r.d >= start) & (r.d <= end)].groupby("d")}
    done = set()
    if (OUT / "days.jsonl").exists():
        done = {json.loads(l)["date"] for l in open(OUT / "days.jsonl")}
    have = set()  # races already saved (a crash mid-day leaves some of that day's races written)
    if (OUT / "odds.jsonl").exists():
        have = {json.loads(l)["cid"] for l in open(OUT / "odds.jsonl")}
    days = sorted(by_day, reverse=True)
    print(f"{len(days)} days {days[-1]}..{days[0]}, {len(done)} already done", flush=True)
    t0, n_pages = time.time(), 0
    for i, D in enumerate(days):
        if D in done:
            continue
        st, t = get(f"{BASE}/courses_archives.php?date={D}")
        if st != 200:
            print(f"{D}: day page failed ({st})", flush=True)
            continue
        links = sorted({m.groups() for m in LINK.finditer(t) if m.group(1) == D})
        ours = by_day[D]
        name_idx = {}
        for _, row in ours.iterrows():
            name_idx.setdefault(slug(row.libelle), []).append(row)
        matched, seen = [], set()
        for d_, venue, cid, rslug in links:
            cands = [c for c in name_idx.get(rslug.lower(), []) if c.raceKey not in seen]
            if len(cands) > 1:  # same race name twice on a day: use the venue slug
                v = [c for c in cands if slug(c.hippodrome) in venue or venue in slug(c.hippodrome)]
                cands = v or cands[:1]
            if cands:
                seen.add(cands[0].raceKey)
                matched.append((cands[0], venue, cid, rslug))
        for row, venue, cid, rslug in matched:
            if int(cid) in have:
                continue
            st, t = get(f"{BASE}/cotes/{D}/{venue}/{cid}_{rslug}.html")
            n_pages += 1
            if st != 200:
                continue
            with gzip.open(OUT / "html" / f"{cid}.html.gz", "wt", encoding="utf-8") as f:
                f.write(t)
            try:
                parsed = parse_cotes(t)
            except Exception as e:  # never stop the run on one odd page; the raw page is kept for re-parsing
                print(f"  parse error {cid}: {type(e).__name__} {e}", flush=True)
                parsed = dict(snap=None, horses=[], parse_error=str(e))
            rec = dict(date=D, raceKey=row.raceKey, R=int(row.numReunion), C=int(row.numCourse), cid=int(cid), venue=venue, **parsed)
            with open(OUT / "odds.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(OUT / "days.jsonl", "a") as f:
            f.write(json.dumps(dict(date=D, links=len(links), ours=len(ours), matched=len(matched),
                                    unmatched=sorted(set(ours.raceKey) - seen),
                                    all_links=[f"{v}/{c}_{s}" for _, v, c, s in links])) + "\n")
        el = time.time() - t0
        print(f"{D}: {len(links)} links, ours {len(ours)}, matched {len(matched)} | {i + 1}/{len(days)} days, {n_pages} pages, {el / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
