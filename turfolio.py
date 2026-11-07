#!/usr/bin/env python3
"""Turfolio: the v3 model's place bets on today's French races (PMU), a paper-trading bank, and a one-command check of the record.

  turfolio setup              one time: race history 2014 -> yesterday (Hugging Face archive + PMU's public API), ~10-30 min
  turfolio bets [DATE]        the bets of a race day (default: today, Paris time). A race's bets are known from 30 minutes before
                              its start, when PMU fixes the reference odds the model reads. A past DATE replays that day with results.
      --watch                 keep running and print each race's bets the moment its odds are fixed (leave it on all afternoon)
      --stake 10              stake per bet for the paper log (EUR)
      --all                   also show the model's numbers for every runner of every race, not only the bets
      --rebuild               rebuild the day's model inputs (programme changes) instead of using the cached ones
  turfolio bank               settle the paper bets logged by `bets` with PMU's official dividends and print your bank
  turfolio verify             recompute the published out-of-sample record (1,069 place bets, 2025-01 -> 2026-11-05) yourself

Bets are 'simple placé' (place) bets in PMU's main pool: the horse must finish in the first 3 (first 2 with 4-7 starters).
Paper money only. No guarantee of any kind; see README.md.
"""
import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

DATA = ROOT / "data"
RAW_FR = DATA / "raw" / "fr"        # Hugging Face archive (2014-01-01 -> 2026-05-05) + dividends
PMU_DIR = DATA / "raw" / "pmu"      # PMU API: every complete day after the archive
LIVE = DATA / "live"                # PMU API: the programme of the day being bet, model inputs cache
PROC = DATA / "processed"
STATE = DATA / "state.json"
MODEL = ROOT / "models" / "v3"
LEDGER = ROOT / "bets.csv"
PARIS = ZoneInfo("Europe/Paris")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
HF = "https://huggingface.co/datasets/annaelmoussa/horse-racing-france/resolve/main"
API = "https://online.turfinfo.api.pmu.fr/rest/client/61/programme"
UA = "Mozilla/5.0 (compatible; turfolio)"
ODDS_LEAD = timedelta(minutes=30)   # PMU fixes the reference odds 30 min before the scheduled start
LEDGER_COLS = ["date", "raceKey", "start", "track", "race", "numPmu", "horse", "p_place", "est_payout", "edge", "stake", "logged_at",
               "status", "payout", "profit"]

# ------------------------------------------------------------------------------------------------------------------------------ output
if os.name == "nt":
    os.system("")  # turn on ANSI colours in the Windows console
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(txt, code):
    return f"\033[{code}m{txt}\033[0m" if COLOR else str(txt)


def bold(t): return c(t, "1")
def green(t): return c(t, "32;1")
def red(t): return c(t, "31")
def dim(t): return c(t, "2")
def yellow(t): return c(t, "33;1")


def say(msg=""):
    print(msg, flush=True)


def step(msg):
    say(f"{c('>', '36;1')} {msg}")


def fail(msg):
    say(red(f"error: {msg}"))
    sys.exit(1)


def eur(x):
    return f"EUR {x:,.2f}"


# ------------------------------------------------------------------------------------------------------------------------------ state
def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kw):
    s = state()
    s.update(kw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


def paris_today():
    return datetime.now(PARIS).date()


def need_setup():
    if not state().get("archive_end") or not (PROC / "fr_runners.parquet").exists():
        fail("run `turfolio setup` first (one-time download of the race history)")


@contextlib.contextmanager
def quiet():
    """Silence the research scripts' progress prints (they stay available with TURFOLIO_VERBOSE=1)."""
    if os.environ.get("TURFOLIO_VERBOSE"):
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def run_script(*args):
    """Run a research script in its own process (keeps the memory peak of each step separate)."""
    env = dict(os.environ, NTHREADS=os.environ.get("NTHREADS", str(os.cpu_count() or 4)))
    out = None if os.environ.get("TURFOLIO_VERBOSE") else subprocess.DEVNULL
    r = subprocess.run([sys.executable, "-I", str(SRC / args[0]), *args[1:]], cwd=ROOT, env=env, stdout=out)
    if r.returncode:
        fail(f"{args[0]} failed (exit {r.returncode}); rerun with TURFOLIO_VERBOSE=1 for details")


# ------------------------------------------------------------------------------------------------------------------------------ http
def get_json(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                body = r.read()
                return json.loads(body) if body.strip() else None
        except urllib.error.HTTPError as e:
            if e.code in (204, 404):
                return None
            time.sleep(5 * (k + 1))
        except Exception:
            time.sleep(5 * (k + 1))
    return None


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done, t0 = 0, time.time()
        with open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and sys.stdout.isatty():
                    rate = done / max(time.time() - t0, 1e-6) / 1e6
                    print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB  ({rate:.1f} MB/s)  ", end="", flush=True)
    if total and sys.stdout.isatty():
        print()
    tmp.replace(dest)


# ------------------------------------------------------------------------------------------------------------------------------ setup
def cmd_setup(a):
    import pandas as pd
    t0 = time.time()
    say(bold("Turfolio setup") + dim("  (one time; everything goes to ./data)"))
    tar = RAW_FR / "normalized.tar.gz"
    for x in (RAW_FR, PMU_DIR, PROC, LIVE):
        x.mkdir(parents=True, exist_ok=True)
    if not (PROC / "fr_runners.parquet").exists():
        if not tar.exists():
            step("Downloading the PMU race archive 2014-2026 from Hugging Face (620 MB)")
            download(f"{HF}/data/source/normalized.tar.gz", tar)
        step("Unpacking 157,000 races and 2 million runners (5-15 min)")
        run_script("fr_extract.py")
    for k in range(5):
        f = RAW_FR / f"rapports-{k:04d}.parquet"
        if not f.exists():
            if k == 0:
                step("Downloading the archive's official dividends (30 MB)")
            download(f"{HF}/data/parquet/rapports-{k:04d}.parquet", f)
    end = str(pd.read_parquet(PROC / "fr_races.parquet", columns=["date"]).date.max())[:10]
    save_state(archive_end=end)
    if tar.exists() and not a.keep_archive:
        tar.unlink()  # the parquet tables hold everything the model needs
    sync_history(paris_today() - timedelta(days=1))
    say(green(f"Setup done in {(time.time() - t0) / 60:.0f} min.") + "  Next: " + bold("turfolio bets") + " (or " + bold("turfolio verify") + ")")


def sync_history(until):
    """Download every complete race day after the archive from PMU's API (resumable), then refresh the flat tables."""
    import fr_fresh
    st = state()
    first = date.fromisoformat(st.get("synced_through") or st["archive_end"]) + timedelta(days=1)
    if first <= until:
        n = (until - first).days + 1
        step(f"Downloading {n} race day{'s' if n > 1 else ''} from PMU ({first} -> {until})" + (dim(f"  ~{max(1, n // 9)} min") if n > 3 else ""))
        fr_fresh.download(first.isoformat(), until.isoformat(), str(PMU_DIR), "8")  # prints its own progress
        save_state(synced_through=until.isoformat())
    flat = PROC / "fr_runners_extra1_pmu.parquet"
    status = PMU_DIR / "status.jsonl"
    if status.exists() and (not flat.exists() or status.stat().st_mtime > flat.stat().st_mtime):
        step("Indexing the downloaded days")
        flatten(PMU_DIR, "extra1_pmu")
        import pandas as pd
        rp = [json.loads(x) for x in open(PMU_DIR / "rapports.jsonl")] if (PMU_DIR / "rapports.jsonl").exists() else []
        if rp:
            pd.DataFrame(rp).drop_duplicates(["raceKey", "typePari", "libelle", "combinaison"], keep="last").to_parquet(RAW_FR / "rapports-pmu.parquet", index=False)


def flatten(src, tag):
    """PMU API jsonl (src/fr_fresh.py format) -> data/processed/fr_{races,runners}_{tag}.parquet, last copy of each race kept."""
    import fr_fresh
    import pandas as pd
    for name, fn, key in (("races", "courses", ["raceKey"]), ("runners", "participants", ["raceKey", "numPmu"])):
        f = src / f"{fn}.jsonl"
        df = fr_fresh.flat_rows(f) if f.exists() and f.stat().st_size else pd.DataFrame(columns=key)
        if len(df):
            df = df.drop_duplicates(key, keep="last")
        df.to_parquet(PROC / f"fr_{name}_{tag}.parquet", index=False)


# ------------------------------------------------------------------------------------------------------------------------------ model inputs
def build_day(day, rebuild=False):
    """Model inputs and the no-odds model scores for every race of `day` -> data/live/DAY/fund.parquet (cached)."""
    import pandas as pd
    today = paris_today()
    sync_history(today - timedelta(days=1))
    live_dir = LIVE / day.isoformat()
    fund = live_dir / "fund.parquet"
    stamp = {"history": state().get("synced_through"), "day": day.isoformat()}
    if fund.exists() and not rebuild and json.loads((live_dir / "stamp.json").read_text()) == stamp:
        return pd.read_parquet(fund)
    in_history = day.isoformat() <= (state().get("synced_through") or "")
    extra2 = [PROC / "fr_races_extra2_live.parquet", PROC / "fr_runners_extra2_live.parquet"]
    for f in extra2:
        f.unlink(missing_ok=True)
    if not in_history:
        step(f"Fetching the programme of {day:%a %d %b %Y} from PMU")
        prog = live_dir / "programme"
        shutil.rmtree(prog, ignore_errors=True)
        import fr_fresh
        with quiet():
            fr_fresh.download(day.isoformat(), day.isoformat(), str(prog), "8")
        if not (prog / "courses.jsonl").exists():
            fail(f"PMU has no French programme for {day}")
        flatten(prog, "extra2_live")
    step("Computing form, ratings and history for every runner (2-5 min)")
    run_script("fr_features.py")
    run_script("fr_exp.py", "ratings")
    step("Scoring the field with the v3 model")
    import fr_v3_final
    with quiet():
        d, _ = fr_v3_final.fundamentals(day.isoformat(), day.isoformat(), MODEL)
    keep = [x for x in d.columns if not x.startswith("s_")]
    live_dir.mkdir(parents=True, exist_ok=True)
    d[keep].to_parquet(fund, index=False)
    (live_dir / "stamp.json").write_text(json.dumps(stamp))
    return d[keep]


# ------------------------------------------------------------------------------------------------------------------------------ bets
def programme(day):
    """Today's French races from PMU: R, C, scheduled start (UTC), status, names."""
    p = get_json(f"{API}/{day:%d%m%Y}")
    out = {}
    for reu in (p or {}).get("programme", {}).get("reunions", []):
        if (reu.get("pays") or {}).get("code") != "FRA":
            continue
        for co in reu.get("courses", []):
            rk = f"{day.isoformat()}_R{co['numReunion']}_C{co['numOrdre']}"
            out[rk] = dict(R=co["numReunion"], C=co["numOrdre"], start=datetime.fromtimestamp(co["heureDepart"] / 1000, timezone.utc),
                           statut=co.get("statut") or "", cat=co.get("categorieStatut") or "", track=reu["hippodrome"]["libelleCourt"],
                           name=co.get("libelle") or "", disc=co.get("specialite") or "")
    return out


def live_odds(day, R, C):
    """PMU's reference (30-min) win odds and status of every runner of a race."""
    p = get_json(f"{API}/{day:%d%m%Y}/R{R}/C{C}/participants")
    out = {}
    for x in (p or {}).get("participants", []):
        ref = x.get("dernierRapportReference") or {}
        out[x["numPmu"]] = (x.get("statut"), ref.get("rapport"))
    return out


def with_odds(rows, odds):
    """Put the live reference odds into a race's rows (as src/fr_features.py does from the archive); drop late non-runners."""
    import numpy as np
    r = rows[rows.numPmu.map(lambda n: (odds.get(n) or ("PARTANT", None))[0] == "PARTANT")].copy()
    o = r.numPmu.map(lambda n: (odds.get(n) or (None, None))[1]).astype(float)
    if len(r) < 2 or o.isna().any():
        return None
    q = 1 / o.where(o > 1.0)
    if q.isna().any():
        return None
    r["odr_rapport"], r["mkt_qr"], r["mkt_pr"], r["mkt_overr"] = o.values, q.values, (q / q.sum()).values, float(q.sum())
    r["field"] = len(r)
    s = np.exp(r.f_score - r.f_score.max())
    r["p_fundamental"] = s / s.sum()
    return r


def score(rows):
    import fr_v3_final
    import pandas as pd
    man = json.load(open(MODEL / "manifest.json"))
    d = pd.concat(rows, ignore_index=True).sort_values(["ts", "rid", "numPmu"]).reset_index(drop=True)
    return fr_v3_final.stack(d, man, MODEL)


def dividends(day, keys):
    """Official place dividends (per EUR 1) of past races: {raceKey: {numPmu: payout}}, {} for a race with dividends but no placing."""
    import pandas as pd
    files = sorted(RAW_FR.glob("rapports-*.parquet"))
    if not files:
        return {}
    rp = pd.concat([pd.read_parquet(f, columns=["raceKey", "typePari", "libelle", "combinaison", "dividendePourUnEuro"]) for f in files])
    rp = rp[(rp.typePari == "SIMPLE_PLACE") & rp.raceKey.isin(keys) & rp.libelle.str.lower().str.startswith("simple placé")]
    out = {}
    for rk, g in rp.groupby("raceKey"):
        out[rk] = {int(n): v / 100 for n, v in zip(pd.to_numeric(g.combinaison, errors="coerce"), g.dividendePourUnEuro) if n == n}
    return out


def race_header(info, n, note=""):
    t = info["start"].astimezone(PARIS).strftime("%H:%M")
    rc = "R%dC%d" % (info["R"], info["C"])
    name = info["name"][:38]
    return f"{bold(t)}  {bold(rc.ljust(6))} {info['track'][:16].ljust(16)} {dim(name.ljust(38))} {dim(str(n) + ' runners')}{note}"


def bet_line(r, stake, result=None):
    horse = r.horse.title()[:24].ljust(24)
    edge = green(("+%.0f%%" % (r.place_ev * 100)).ljust(5))
    s = (f"   {green('BET')}  #{int(r.numPmu):<2} {bold(horse)}  place {r.p_place * 100:3.0f}%  "
         f"est. payout {r.est_place_payout:4.2f}  edge {edge}  stake {eur(stake)}")
    if result is not None:
        s += "   " + (green("PLACED, paid %.2f" % result) if result > 0 else red("unplaced"))
    return s


def cmd_bets(a):
    import pandas as pd
    need_setup()
    day = date.fromisoformat(a.date) if a.date else paris_today()
    today = paris_today()
    if day > today + timedelta(days=1):
        fail("PMU publishes the programme one day ahead at most")
    replay = day < today
    say(bold(f"Turfolio  {day:%A %d %B %Y}") + dim("  - v3 model, place bets in PMU's main pool") + (yellow("  [replay]") if replay else ""))
    trained = json.load(open(MODEL / "manifest.json"))["trained_through"]
    if replay and day.isoformat() <= trained:
        say(yellow(f"Note: the shipped model was trained on every race up to {trained}, so a replay of {day} is in-sample and flatters it."))
        say(yellow("      The honest out-of-sample record is `turfolio verify`; replays are out-of-sample from the day after."))
    fund = build_day(day, a.rebuild)
    if fund.empty:
        fail(f"no French races on {day}")
    prog = programme(day) if not replay else {}
    if replay:  # past day: the reference odds and results are in the history tables
        for rk, g in fund.groupby("raceKey", sort=False):
            r0 = g.iloc[0]
            prog[rk] = dict(R=int(r0.numReunion), C=int(r0.numCourse), start=pd.Timestamp(r0.ts).tz_localize("UTC").to_pydatetime(),
                            statut="", cat="", track="", name="", disc=str(r0.specialite))
        meta = pd.read_parquet(PROC / "fr_races_extra1_pmu.parquet" if day.isoformat() > state()["archive_end"] else PROC / "fr_races.parquet",
                               columns=["raceKey", "hippodrome", "libelle"], filters=[("date", "==", day.isoformat())])
        for x in meta.itertuples():
            if x.raceKey in prog:
                prog[x.raceKey].update(track=x.hippodrome or "", name=x.libelle or "")
    in_fund = set(fund.raceKey)
    races = [rk for rk in sorted(prog, key=lambda k: prog[k]["start"]) if rk in in_fund]
    say(dim(f"{len(races)} French races. A race's bets appear 30 min before its start, when PMU fixes the reference odds.") + "\n")
    logged = ledger_keys()
    done, total_bets, total_stake, results = set(), 0, 0.0, []
    divs = dividends(day, set(races)) if replay else {}
    last_wait, first = None, True
    while True:
        now = datetime.now(timezone.utc)
        if not replay and not first:
            prog.update({k: v for k, v in programme(day).items() if k in prog})  # status changes, cancellations
        first = False
        ready, waiting = [], []
        for rk in races:
            if rk in done:
                continue
            info = prog[rk]
            over = info["cat"] in ("ARRIVEE", "ANNULEE") or info["statut"].startswith("ARRIVEE") or "ANNUL" in info["statut"]
            if not replay and over and not a.all:
                done.add(rk)  # already run: nothing to bet
                continue
            (ready if replay or now >= info["start"] - ODDS_LEAD else waiting).append(rk)
        rows = []
        for rk in ready:
            g = fund[fund.raceKey == rk]
            r = (g if g.mkt_pr.notna().all() else None) if replay else with_odds(g, live_odds(day, prog[rk]["R"], prog[rk]["C"]))
            if r is None:
                if replay or now >= prog[rk]["start"]:
                    done.add(rk)  # the reference odds never became complete: no bet
                continue
            rows.append(r)
        if rows:
            d = score(rows)
            scored = set(d.raceKey)
            for rk in [k for k in races if k in scored]:
                g = d[d.raceKey == rk].sort_values("p_place", ascending=False)
                info = prog[rk]
                bets = g[g.rule_B]
                done.add(rk)
                if bets.empty and not a.all:
                    continue
                late = not replay and now >= info["start"]
                say(race_header(info, len(g), yellow("  started: too late to bet") if late and len(bets) else ""))
                for r in bets.itertuples():
                    res = None
                    if replay and rk in divs:
                        res = divs[rk].get(int(r.numPmu), 0.0)
                        results.append(res)
                    say(bet_line(r, a.stake, res))
                    if late:
                        continue
                    total_bets += 1
                    total_stake += a.stake
                    if not replay and (rk, int(r.numPmu)) not in logged:
                        log_bet(day, rk, info, r, a.stake)
                        logged.add((rk, int(r.numPmu)))
                if a.all:
                    for r in g.itertuples():
                        tag = "BET" if r.rule_B else "   "
                        say(dim(f"        {tag} #{int(r.numPmu):<2} {r.horse.title()[:24]:<24} win {r.p_win * 100:4.1f}%  place {r.p_place * 100:3.0f}%  "
                                f"est. payout {r.est_place_payout:4.2f}  edge {r.place_ev * 100:+4.0f}%"))
                say()
        if not a.watch or replay or not waiting:
            break
        k = min(waiting, key=lambda k: prog[k]["start"])
        nxt = prog[k]["start"] - ODDS_LEAD
        if nxt != last_wait:
            say(dim(f"... waiting for R{prog[k]['R']}C{prog[k]['C']} {prog[k]['track']}: odds at {nxt.astimezone(PARIS):%H:%M} Paris time, "
                    f"{len(waiting)} races to go. Ctrl+C to stop."))
            last_wait = nxt
        time.sleep(max(15, min(60, (nxt - datetime.now(timezone.utc)).total_seconds() + 20)))
    # summary
    if replay:
        if results:
            profit = sum(r - 1 for r in results) * a.stake
            say(bold(f"Replay {day}: {len(results)} bets, {sum(r > 0 for r in results)} placed, profit ") +
                (green if profit >= 0 else red)(f"{eur(profit)} ({profit / (len(results) * a.stake) * 100:+.1f}% on {eur(len(results) * a.stake)} staked)"))
        else:
            say(bold(f"Replay {day}: {total_bets} bets") + dim(" (no dividends in the history for this day)"))
        return
    waiting = [k for k in races if k not in done and prog[k]["start"] > datetime.now(timezone.utc)]
    say(bold(f"{total_bets} bet{'s' if total_bets != 1 else ''} so far today") + (f" ({eur(total_stake)} staked, logged to bets.csv)" if total_bets else ""))
    if waiting:
        k = min(waiting, key=lambda k: prog[k]["start"])
        say(dim(f"{len(waiting)} races still to come; next odds at {(prog[k]['start'] - ODDS_LEAD).astimezone(PARIS):%H:%M}. ")
            + "Leave " + bold("turfolio bets --watch") + " running to get each bet as it appears.")
    say(dim("Tomorrow: ") + bold("turfolio bank") + dim(" settles the paper bets with the official dividends."))


# ------------------------------------------------------------------------------------------------------------------------------ paper bank
def ledger_rows():
    if not LEDGER.exists():
        return []
    with open(LEDGER, newline="") as f:
        return list(csv.DictReader(f))


def ledger_keys():
    return {(r["raceKey"], int(r["numPmu"])) for r in ledger_rows()}


def write_ledger(rows):
    with open(LEDGER, "w", newline="") as f:
        w = csv.DictWriter(f, LEDGER_COLS)
        w.writeheader()
        w.writerows(rows)


def log_bet(day, rk, info, r, stake):
    rows = ledger_rows()
    rows.append(dict(date=day.isoformat(), raceKey=rk, start=info["start"].astimezone(PARIS).strftime("%H:%M"), track=info["track"],
                     race=f"R{info['R']}C{info['C']}", numPmu=int(r.numPmu), horse=r.horse.title(), p_place=round(float(r.p_place), 4),
                     est_payout=round(float(r.est_place_payout), 2), edge=round(float(r.place_ev), 4), stake=stake,
                     logged_at=datetime.now(PARIS).strftime("%Y-%m-%d %H:%M"), status="open", payout="", profit=""))
    write_ledger(rows)


def settle(r):
    """Official result of one logged bet: ('won', payout per EUR 1) / ('lost', 0) / ('refund', 1) / None while not final."""
    d = date.fromisoformat(r["date"])
    R, C = r["race"][1:].split("C")
    parts = get_json(f"{API}/{d:%d%m%Y}/R{R}/C{C}/participants") or {}
    me = next((x for x in parts.get("participants", []) if x.get("numPmu") == int(r["numPmu"])), None)
    if me is not None and me.get("statut") != "PARTANT":
        return "refund", 1.0  # non-runner: stake returned
    rp = get_json(f"{API}/{d:%d%m%Y}/R{R}/C{C}/rapports-definitifs")
    for b in rp if isinstance(rp, list) else []:
        if b.get("typePari") != "SIMPLE_PLACE":
            continue
        pays = {str(y.get("combinaison")): y.get("dividendePourUnEuro", 0) / 100 for y in b.get("rapports") or []
                if (y.get("libelle") or "").lower().startswith("simple placé")}
        if pays:
            v = pays.get(str(r["numPmu"]))
            return ("won", v) if v else ("lost", 0.0)
    if (datetime.now(PARIS).date() - d).days >= 3:
        return "refund", 1.0  # no place dividends days later: race cancelled or no place pool
    return None


def cmd_bank(a):
    rows = ledger_rows()
    if not rows:
        say("No paper bets yet. Run " + bold("turfolio bets --watch") + " on a race day; every bet it prints is logged to bets.csv.")
        return
    n_new = 0
    for r in rows:
        if r["status"] == "open":
            s = settle(r)
            if s:
                r["status"], pay = s
                r["payout"] = f"{pay:.2f}"
                r["profit"] = f"{(pay - 1) * float(r['stake']):.2f}"
                n_new += 1
    write_ledger(rows)
    closed = [r for r in rows if r["status"] != "open"]
    say(bold("Turfolio paper bank") + dim(f"  ({len(rows)} bets logged, {n_new} settled just now, file bets.csv)") + "\n")
    for r in rows[-25:]:
        st = r["status"]
        txt = {"won": "PLACED %.2f" % float(r["payout"] or 0), "lost": "unplaced", "refund": "refunded"}.get(st, "open").ljust(12)
        res = {"won": green, "lost": red, "refund": yellow}.get(st, dim)(txt)
        prof = f"{float(r['profit']):+8.2f}" if r["profit"] else "        "
        say(f"  {r['date']} {r['start']}  {r['race']:<6} {r['track'][:14]:<14} #{r['numPmu']:<2} {r['horse'][:22]:<22}  place {float(r['p_place']) * 100:3.0f}%  {res} {prof}")
    if closed:
        staked = sum(float(r["stake"]) for r in closed if r["status"] != "refund")
        profit = sum(float(r["profit"]) for r in closed)
        won = sum(r["status"] == "won" for r in closed)
        lost = sum(r["status"] == "lost" for r in closed)
        start = a.start
        say()
        say(bold(f"{won + lost} settled bets: {won} placed ({won / max(won + lost, 1) * 100:.0f}%)") +
            f"   staked {eur(staked)}   profit " + (green if profit >= 0 else red)(f"{eur(profit)} ({profit / max(staked, 1e-9) * 100:+.1f}%)"))
        say(f"Bank: {eur(start)} -> " + bold(eur(start + profit)))
        if won + lost < 300:
            say(dim("(Below a few hundred bets this is mostly luck: the measured edge is about +5% per bet, the swing of a single bet about 60%.)"))


# ------------------------------------------------------------------------------------------------------------------------------ verify
def cmd_verify(a):
    import numpy as np
    import pandas as pd
    need_setup()
    sync_history(min(paris_today() - timedelta(days=1), date(2026, 11, 5)))
    import fr_place
    p = pd.read_parquet(ROOT / "predictions" / "walkforward_predictions.parquet")
    p["date"] = pd.to_datetime(p.date)
    field = p.groupby("raceKey").size()
    b = p[p.rule_B & (p.date >= "2025-01-01") & (p.date <= "2026-11-05")].copy()
    dv = fr_place.place_dividends(set(b.raceKey), field)
    b = b.merge(dv[["raceKey", "numPmu", "div"]], on=["raceKey", "numPmu"], how="left")
    have = set(dv.raceKey)
    b = b[b.raceKey.isin(have)]
    b["ret"] = b["div"].fillna(0.0)
    say(bold("Turfolio verify") + dim("  - the published place-bet record, recomputed from the shipped predictions and PMU's official dividends") + "\n")
    say(dim("Every bet below was chosen by the walk-forward model (trained only on earlier years) 30 minutes before the off,"))
    say(dim("by a rule fixed on 2019-2024. Paid at the official 'simple placé' dividend, EUR 10 per bet.") + "\n")
    rng = np.random.default_rng(0)
    for name, lo, hi in (("Hold-out 2025-01 -> 2026-05-05", "2025-01-01", "2026-05-05"), ("Fresh window 2026-05-06 -> 11-05", "2026-05-06", "2026-11-05"),
                         ("Both (all out-of-sample bets)", "2025-01-01", "2026-11-05")):
        x = b[(b.date >= lo) & (b.date <= hi)].sort_values("date")
        if x.empty:
            continue
        days = x.date.values
        u, inv = np.unique(days, return_inverse=True)
        s, n = np.bincount(inv, x.ret.values - 1), np.bincount(inv)
        idx = rng.integers(0, len(u), (2000, len(u)))
        boot = s[idx].sum(1) / n[idx].sum(1)
        lo_ci, hi_ci = np.quantile(boot, [0.025, 0.975])
        roi = x.ret.mean() - 1
        bank = 1000 + np.cumsum(10 * (x.ret.values - 1))
        say(f"  {bold(f'{name:<34}')} {len(x):>5,} bets  placed {(x.ret > 0).mean() * 100:3.0f}%  ROI " +
            (green if roi > 0 else red)(f"{roi * 100:+.1f}%") + dim(f" [{lo_ci * 100:+.1f}, {hi_ci * 100:+.1f}]") +
            f"  bank EUR 1,000 -> {bold(f'{bank[-1]:,.0f}')}")
    say()
    say(dim("Brackets: 95% bootstrap interval over race days. The published figures (README) were computed by src/fr_place_conf.py"))
    say(dim("on the same bets; small differences can come from dividends PMU corrected after the fact."))


# ------------------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(prog="turfolio", description="Place bets on French horse races (PMU) from the v3 model. Paper money only.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="one-time download and build of the race history")
    s.add_argument("--keep-archive", action="store_true", help="keep the 620 MB download after unpacking")
    s.set_defaults(fn=cmd_setup)
    s = sub.add_parser("bets", help="the bets of a race day (default today)")
    s.add_argument("date", nargs="?", help="YYYY-MM-DD; a past date replays that day with results")
    s.add_argument("--watch", action="store_true", help="keep running and print each race's bets as its odds are fixed")
    s.add_argument("--stake", type=float, default=10.0, help="stake per bet in EUR for the paper log (default 10)")
    s.add_argument("--all", action="store_true", help="show every runner, not only the bets")
    s.add_argument("--rebuild", action="store_true", help="rebuild the day's model inputs")
    s.set_defaults(fn=cmd_bets)
    s = sub.add_parser("bank", help="settle the logged paper bets and show the bank")
    s.add_argument("--start", type=float, default=1000.0, help="starting bank in EUR (default 1000)")
    s.set_defaults(fn=cmd_bank)
    s = sub.add_parser("verify", help="recompute the published out-of-sample record")
    s.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    try:
        a.fn(a)
    except KeyboardInterrupt:
        say(dim("\nstopped."))


if __name__ == "__main__":
    main()
