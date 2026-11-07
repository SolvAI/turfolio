"""Flatten the French PMU archive (data/raw/fr/normalized.tar.gz) into parquet, streaming (the jsonl files are ~5 GB).

  participants.jsonl -> data/processed/fr_runners.parquet   one row per declared runner, every scalar field kept
                        (top level + raw.* + both odds snapshots), so later steps can decide what is pre-race
  courses.jsonl      -> data/processed/fr_races.parquet     one row per race, same flattening

Run: python -I src/fr_extract.py   (self-contained; -I keeps the data folder out of the import path)
"""
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAR = ROOT / "data" / "raw" / "fr" / "normalized.tar.gz"
OUT = ROOT / "data" / "processed"
CHUNK = 200_000
DROP = {"urlCasaque", "raw.urlCasaque"}


def flat(o, pre="", out=None):
    out = {} if out is None else out
    for k, v in o.items():
        key = f"{pre}{k}"
        if key in DROP:
            continue
        if isinstance(v, dict):
            flat(v, key + ".", out)
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


def tidy(df):
    for c in df.columns:
        s = df[c]
        if s.dtype == object:
            nn = s.dropna()
            if len(nn) and nn.map(lambda v: isinstance(v, (bool, int, float))).all():
                df[c] = pd.to_numeric(s.map(lambda v: float(v) if v is not None else None), errors="coerce")
            else:
                df[c] = s.map(lambda v: None if v is None or (isinstance(v, float) and v != v) or v == "nan" else str(v))
    return df


def stream(member, name):
    parts, buf, n = [], [], 0
    tmp = OUT / f"_{name}_chunks"
    tmp.mkdir(exist_ok=True)
    with tarfile.open(TAR, "r|gz") as tf:
        for m in tf:
            if m.name != member:
                continue
            f = tf.extractfile(m)
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                raw = r.pop("raw", None) or {}
                row = flat(r)
                for k, v in flat(raw, "raw.").items():
                    if k[4:] not in row:  # keep raw-only fields; top-level copies win
                        row[k] = v
                buf.append(row)
                n += 1
                if len(buf) == CHUNK:
                    p = tmp / f"{len(parts):03d}.parquet"
                    tidy(pd.DataFrame(buf)).to_parquet(p, index=False)
                    parts.append(p); buf = []
                    print(f"  {name}: {n:,} rows", flush=True)
            break
    if buf:
        p = tmp / f"{len(parts):03d}.parquet"
        tidy(pd.DataFrame(buf)).to_parquet(p, index=False)
        parts.append(p)
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = tidy(df)
    df.to_parquet(OUT / f"fr_{name}.parquet", index=False)
    for p in parts:
        p.unlink()
    tmp.rmdir()
    print(f"{name}: {len(df):,} rows, {df.shape[1]} columns -> fr_{name}.parquet", flush=True)
    return df


if __name__ == "__main__":
    if sys.argv[1:] == ["retidy"]:  # re-clean existing outputs (missing strings stored as 'nan' by an earlier version)
        for n in ("races", "runners"):
            tidy(pd.read_parquet(OUT / f"fr_{n}.parquet")).to_parquet(OUT / f"fr_{n}.parquet", index=False)
        sys.exit()
    which = sys.argv[1:] or ["courses", "participants"]
    for w in which:
        stream(f"normalized/{w}.jsonl", {"courses": "races", "participants": "runners"}[w])
