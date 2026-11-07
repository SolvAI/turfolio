"""France: neural race model (fundamental: never sees this race's odds), walk-forward by year like fr_model.py.

Each runner = its pre-race feature vector (the GBM's 157 features + ratings) + the sequence of its last L starts in the data
(result, speed vs field, beaten lengths, market opinion then, opponent strength = mean rating of that field, distance/class/track
changes relative to today). A small attention pool summarises the history; a transformer over the field lets runners see each
other; the loss is Plackett-Luce over the first 3 places (stage 1 = winner, dead heats shared).

  tune   train 2015-2021, validate 2022 -> number of epochs (dev only)
  wf     for each year Y in 2017..2026 train on 2015..Y-1, predict Y -> data/processed/exp/nn.parquet (f_score)
Run: python -I src/fr_nn.py tune | wf [tag]
Variants by tag: L20 (20-run history), mus (form-string tokens), kdX (training-only distillation of the final market's probabilities, weight X),
  off (score = residual over the 30-min market, which enters the loss as an offset, never as an input)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_exp import EXP, RT_COLS, load_variant  # noqa: E402
from fr_model import FIRST_TRAIN, OOF_YEARS  # noqa: E402
from racemodel import starts_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
L, M = 10, 24
dev = torch.device("cuda")
CFG = dict(d=192, heads=4, layers=2, hd=64, drop=0.1, lr=1e-3, wd=1e-4, bs=256, max_epochs=14, nohist=False)


def build(all_races=False, meta=None):
    """all_races: also races without a result (to score new races). meta: the preprocessing saved with a final model
    (category vocabularies, NaN-flag columns, clipping/standardisation) so that new races are encoded exactly as in training."""
    t0 = time.time()
    d, F = load_variant("rt", all_races)
    if CFG.get("ref"):  # end-to-end variant: this race's 30-minute market as an input (log prob, rank, overround)
        d["lmr_in"] = np.log(d.mkt_pr.clip(lower=1e-4))
        d["mr_rank_in"] = d.groupby("rid").mkt_pr.rank(ascending=False)
        d["mkt_overr_in"] = d.mkt_overr
        F = F + ["lmr_in", "mr_rank_in", "mkt_overr_in"]
    cats = [c for c in F if str(d[c].dtype) in ("category", "object")]
    nums = [c for c in F if c not in cats]
    # numeric: clip at 1/99 % and standardise with 2015-2016 statistics (always inside every training window); NaN -> 0 + flag
    X = d[nums].astype(np.float32).values
    if meta is None:
        ref = d[d.date < "2017-01-01"]
        lo, hi = np.nanquantile(ref[nums].astype(np.float32).values, [.01, .99], axis=0)
        mu, sd = np.nanmean(np.clip(ref[nums].astype(np.float32).values, lo, hi), 0), np.nanstd(np.clip(ref[nums].astype(np.float32).values, lo, hi), 0) + 1e-6
    else:
        assert nums == meta["nums"] and cats == meta["cats"], "feature columns differ from the saved model"
        lo, hi, mu, sd = (np.array(meta[k], np.float32) for k in ("lo", "hi", "mu", "sd"))
    X = np.clip(X, lo, hi)
    X = (X - mu) / sd
    nan_cols = np.flatnonzero(np.isnan(X).mean(0) > 0.01) if meta is None else np.array(meta["nan_cols"], np.int64)
    X = np.c_[np.nan_to_num(X), np.isnan(X[:, nan_cols])].astype(np.float32)
    if meta is None:
        vocab = {c: [str(v) for v in d[c].astype("category").cat.categories] for c in cats}
        C = np.column_stack([d[c].astype("category").cat.codes.values + 1 for c in cats]).astype(np.int64)
        card = [int(C[:, j].max()) + 1 for j in range(C.shape[1])]
    else:  # unseen category values -> 0 (the code of a missing value)
        vocab, card = meta["vocab"], meta["card"]
        C = np.column_stack([pd.Categorical(d[c].astype(str).where(d[c].notna()), categories=vocab[c]).codes + 1 for c in cats]).astype(np.int64)
    pre = {"nums": nums, "cats": cats, "lo": lo.tolist(), "hi": hi.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "nan_cols": nan_cols.tolist(),
           "vocab": vocab, "card": card}
    # history tokens from ALL starts (also races without a recorded winner)
    h = pd.read_parquet(PROC / "fr_feats.parquet", columns=["rid", "numPmu", "hk", "ts", "day", "pos_pct", "won", "pos", "dq", "fail", "rk_rel",
                                                            "bl", "exp_win", "field", "prize_log", "distance", "autostart", "is_trot",
                                                            "specialite", "hippodromeCode", "driver"])
    h = h.merge(pd.read_parquet(PROC / "fr_ratings.parquet")[["rid", "numPmu", "rt_sum", "rt_sum_dm"]], on=["rid", "numPmu"], how="left")
    h["field_rt"] = h.rt_sum - h.rt_sum_dm
    h = h.sort_values(["hk", "ts"]).reset_index(drop=True)
    tok = np.c_[h.pos_pct.fillna(1.1), h.pos_pct.isna(), h.won, (h.pos <= 3).fillna(False), h.dq, h.fail, h.rk_rel.clip(-5, 5).fillna(0),
                h.rk_rel.isna(), np.log1p(h.bl.clip(0, 30)).fillna(0), h.bl.isna(), np.log(h.exp_win.clip(1e-3, 1)), np.log(h.field),
                h.prize_log - 9, h.distance / 1000 - 2.5, h.autostart, h.is_trot, h.rt_sum.fillna(0), h.field_rt.fillna(0), h.rt_sum_dm.fillna(0)
                ].astype(np.float32)
    info = np.c_[h.day, h.distance / 1000, h.prize_log, h.hippodromeCode.astype("category").cat.codes, h.driver.astype("category").cat.codes,
                 h.specialite.astype("category").cat.codes].astype(np.float32)
    grp_start = np.r_[0, np.flatnonzero(h.hk.values[1:] != h.hk.values[:-1]) + 1]
    gid = np.repeat(np.arange(len(grp_start)), np.diff(np.r_[grp_start, len(h)]))
    within = np.arange(len(h)) - grp_start[gid]
    H = np.full((len(h), L), -1, np.int64)
    for k in range(1, L + 1):
        ok = within >= k
        H[ok, k - 1] = np.flatnonzero(ok) - k
    pos_in_h = pd.Series(np.arange(len(h)), index=pd.MultiIndex.from_arrays([h.rid.values, h.numPmu.values]))
    ti = pos_in_h.reindex(pd.MultiIndex.from_arrays([d.rid.values, d.numPmu.values])).values.astype(np.int64)
    # races padded to M runners
    st, sz = starts_of(d.rid.values)
    keep = sz <= M
    RI = np.full((len(st), M), -1, np.int64)
    for j in range(M):
        ok = keep & (sz > j)
        RI[ok, j] = st[ok] + j
    pos = d.pos.fillna(0).values.astype(np.int64)
    MU = None
    if CFG.get("mus"):
        MU = musique_tokens(d)
    # distillation target (kd variants only): the FINAL market's probabilities of this race, used as a training target, never as an input
    full = d.mkt_pf.notna().groupby(d.rid).transform("all").values
    q = np.where(full, d.mkt_pf.fillna(0).values / d.groupby("rid").mkt_pf.transform("sum").replace(0, np.nan).fillna(1).values, 0.0)
    # residual variants (off): the 30-min market's log-probabilities are a fixed OFFSET added to the network's score in the loss (not an input)
    fr = d.mkt_pr.notna().groupby(d.rid).transform("all").values
    lmr = np.where(fr, np.log(d.mkt_pr.clip(lower=1e-4).fillna(1).values / d.groupby("rid").mkt_pr.transform("sum").replace(0, np.nan).fillna(1).values), 0.0)
    print(f"built: {len(d):,} rows, {X.shape[1]} numeric inputs, {len(cats)} categoricals, {len(h):,} history tokens ({time.time() - t0:.0f}s)", flush=True)
    return dict(d=d[["rid", "numPmu", "date"]].reset_index(drop=True), X=X, C=C, card=card, y=d.y.values.astype(np.float32), pos=pos, ti=ti,
                H=H, tok=tok, info=info, RI=RI, year=d.date.dt.year.values[st], keep=keep, MU=MU,
                q=q.astype(np.float32), qv=full[st], lmr=lmr.astype(np.float32), rv=fr[st],
                rdate=d.date.values[st], labelled=d.labelled.values[st], pre=pre)


KM = 12
MUS_RES = {**{str(i): i for i in range(1, 10)}, "0": 10, "D": 11, "A": 12, "T": 13, "R": 14}
MUS_DISC = {"a": 0, "m": 1, "p": 2, "h": 3, "s": 4, "c": 5}


def musique_tokens(d):
    """The form string (published before the race; it also covers races outside the data): last KM results as tokens
    [one-hot result (1..9, 0 = 10th or worse, D disqualified, A stopped, T fell, R/other), one-hot discipline, season break before it]."""
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from fr_features import parse_musique
    mus = pd.read_parquet(PROC / "fr_runners.parquet", columns=["raceKey", "numPmu", "musique"])
    m = d[["raceKey", "numPmu"]].merge(mus, on=["raceKey", "numPmu"], how="left").musique.values
    nf = 15 + 6 + 1
    out = np.zeros((len(d), KM, nf), np.float16)
    for i, sm in enumerate(m):
        for k, (r, dc, brk) in enumerate(parse_musique(sm)[:KM]):
            out[i, k, MUS_RES.get(r, 14)] = 1
            out[i, k, 15 + MUS_DISC.get(dc, 2)] = 1
            out[i, k, 21] = float(brk)
    return out


class Net(nn.Module):
    def __init__(self, nx, card, ntok, c):
        super().__init__()
        D = c["d"]
        self.emb = nn.ModuleList([nn.Embedding(k, min(8, k)) for k in card])
        ne = sum(min(8, k) for k in card)
        self.inp = nn.Sequential(nn.Linear(nx + ne, D), nn.GELU(), nn.Dropout(c["drop"]), nn.Linear(D, D))
        self.tok = nn.Sequential(nn.Linear(ntok + 6 + L, c["hd"]), nn.GELU(), nn.Linear(c["hd"], c["hd"]))
        self.tatt = nn.TransformerEncoder(nn.TransformerEncoderLayer(c["hd"], 2, 2 * c["hd"], c["drop"], batch_first=True, norm_first=True), 1)
        self.tq = nn.Linear(c["hd"], 1)
        self.mus = (nn.Sequential(nn.Linear(22 + KM, 32), nn.GELU(), nn.Linear(32, 32)) if c.get("mus") else None)
        self.mix = nn.Sequential(nn.Linear(D + c["hd"] + (32 if c.get("mus") else 0), D), nn.GELU())
        self.race = (nn.TransformerEncoder(nn.TransformerEncoderLayer(D, c["heads"], 2 * D, c["drop"], batch_first=True, norm_first=True), c["layers"])
                     if c["layers"] > 0 else None)
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 1))
        # multi-task variants (mt): a second head predicts the late move = the final market given the 30-min market (as an offset)
        self.out2 = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 1)) if c.get("mt") else None

    def forward(self, x, cat, tk, tmask, rmask, mu=None):
        B, Mx = x.shape[:2]
        e = torch.cat([x] + [emb(cat[..., j]) for j, emb in enumerate(self.emb)], -1)
        u = self.inp(e)
        t = self.tok(tk.view(B * Mx, L, -1))
        tm = tmask.view(B * Mx, L)
        empty = tm.all(1)
        tm2 = tm.clone()
        tm2[empty, 0] = False  # avoid all-masked rows in attention
        t = self.tatt(t, src_key_padding_mask=tm2)
        a = self.tq(t).squeeze(-1).masked_fill(tm2, -1e4)
        hv = (torch.softmax(a, 1).unsqueeze(-1) * t).sum(1)
        hv = hv.masked_fill(empty.unsqueeze(-1), 0.0).view(B, Mx, -1)
        parts = [u, hv]
        if self.mus is not None:
            mm = (mu[..., :15].sum(-1) > 0).float()                     # (B, M, KM) token present
            pos_oh = torch.eye(KM, device=mu.device).expand(*mu.shape[:2], KM, KM)
            e_mu = self.mus(torch.cat([mu.float(), pos_oh], -1))         # (B, M, KM, 32)
            parts.append((e_mu * mm.unsqueeze(-1)).sum(2) / mm.sum(2, keepdim=True).clamp(min=1))
        z = self.mix(torch.cat(parts, -1))
        if self.race is not None:
            z = self.race(z, src_key_padding_mask=rmask)
        if self.out2 is not None:
            return self.out(z).squeeze(-1), self.out2(z).squeeze(-1)
        return self.out(z).squeeze(-1)


class Data:
    """Everything on the GPU; batches gathered by race index."""
    def __init__(self, b):
        self.X = torch.tensor(b["X"], device=dev)
        self.C = torch.tensor(b["C"], device=dev)
        self.y = torch.tensor(b["y"], device=dev)
        self.pos = torch.tensor(b["pos"], device=dev)
        self.ti = torch.tensor(b["ti"], device=dev)
        self.H = torch.tensor(b["H"], device=dev)
        self.tok = torch.tensor(b["tok"], device=dev)
        self.info = torch.tensor(b["info"], device=dev)
        self.RI = torch.tensor(b["RI"], device=dev)
        self.MU = torch.tensor(b["MU"], device=dev) if b.get("MU") is not None else None
        self.q = torch.tensor(b["q"], device=dev)
        self.qv = torch.tensor(b["qv"], device=dev)
        self.lmr = torch.tensor(b["lmr"], device=dev)
        self.rv = torch.tensor(b["rv"], device=dev)

    def offset(self, races):
        ri = self.RI[races]
        return self.lmr[ri.clamp(min=0)] * (ri >= 0)

    def kd_target(self, races):
        ri = self.RI[races]
        return self.q[ri.clamp(min=0)] * (ri >= 0), self.qv[races]

    def batch(self, races):
        ri = self.RI[races]                      # (B, M)
        rmask = ri < 0
        r = ri.clamp(min=0)
        x, cat, y, pos = self.X[r], self.C[r], self.y[r] * (~rmask), self.pos[r] * (~rmask)
        cur = self.ti[r]                         # (B, M) index in the token table
        hi = self.H[cur]                         # (B, M, L)
        tmask = (hi < 0) | rmask.unsqueeze(-1)
        if CFG["nohist"]:
            tmask = torch.ones_like(tmask)
        hc = hi.clamp(min=0)
        tk = self.tok[hc]
        ci, pi = self.info[cur].unsqueeze(2), self.info[hc]
        rel = torch.stack([torch.log1p((ci[..., 0] - pi[..., 0]).clamp(min=0)) / 3, ci[..., 1] - pi[..., 1], ci[..., 2] - pi[..., 2],
                           (ci[..., 3] == pi[..., 3]).float(), (ci[..., 4] == pi[..., 4]).float(), (ci[..., 5] == pi[..., 5]).float()], -1)
        recency = torch.eye(L, device=dev).expand(*hi.shape[:2], L, L)
        tk = torch.cat([tk, rel, recency], -1).masked_fill(tmask.unsqueeze(-1), 0.0)
        mu = self.MU[r] if self.MU is not None else None
        return x, cat, tk, tmask, rmask, y, pos, mu


def pl_loss(s, rmask, y, pos, K=3, rho=0.7):
    s = s.float().masked_fill(rmask, -1e4)
    tot = -(y * torch.log_softmax(s, 1)).sum(1)                 # stage 1: winner (dead heats shared)
    act = ~rmask & (pos != 1)
    valid = (pos == 1).sum(1) == 1
    w = 1.0
    for k in range(2, K + 1):
        valid = valid & ((pos == k).sum(1) == 1)
        w *= rho
        ls = torch.log_softmax(s.masked_fill(~act, -1e4), 1)
        tk = (pos == k).float()
        tot = tot + w * valid.float() * -(tk * ls).sum(1)
        act = act & (pos != k)
    return tot.mean()


def kd_loss(s, rmask, q, qv):
    """cross-entropy of the network's race softmax against the final market's probabilities (races fully priced only)"""
    ce = -(q * torch.log_softmax(s.float().masked_fill(rmask, -1e4), 1)).sum(1)
    return (ce * qv).sum() / qv.sum().clamp(min=1)


def winner_ll(s, rmask, y):
    s = s.float().masked_fill(rmask, -1e4)
    return -(y * torch.log_softmax(s, 1)).sum(1)


def run(data, b, train_idx, eval_idx, epochs, seed, log_val=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(b["X"].shape[1], b["card"], b["tok"].shape[1], CFG).to(dev)
    if CFG.get("off"):  # residual on the 30-min market: train on races where it is complete
        train_idx = train_idx[b["rv"][train_idx]]
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    steps = epochs * int(np.ceil(len(train_idx) / CFG["bs"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=steps, pct_start=0.1)
    hist = []
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(train_idx)
        for i in range(0, len(perm), CFG["bs"]):
            x, cat, tk, tmask, rmask, y, pos, mu = data.batch(torch.tensor(perm[i:i + CFG["bs"]], device=dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = net(x, cat, tk, tmask, rmask, mu)
            s2 = None
            if CFG.get("mt"):
                s, s2 = s
            if CFG.get("off"):
                s = s.float() + data.offset(torch.tensor(perm[i:i + CFG["bs"]], device=dev))
            loss = pl_loss(s, rmask, y, pos)
            if CFG.get("mt"):  # head 2: final market given the 30-min market (offset), races where both markets are complete
                bi = torch.tensor(perm[i:i + CFG["bs"]], device=dev)
                q, qv = data.kd_target(bi)
                loss = loss + CFG.get("kd", 1.0) * kd_loss(s2.float() + data.offset(bi), rmask, q, (qv & data.rv[bi]).float())
            elif CFG.get("kd"):
                q, qv = data.kd_target(torch.tensor(perm[i:i + CFG["bs"]], device=dev))
                loss = loss + CFG["kd"] * kd_loss(s, rmask, q, qv.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        if log_val:
            s_ev = predict(net, data, eval_idx)
            ll = float(np.mean(s_ev[1]))
            hist.append(ll)
            print(f"    epoch {ep + 1}: validation winner log-loss {ll:.4f}", flush=True)
    return net, hist


@torch.no_grad()
def predict(net, data, idx):
    net.eval()
    S, LL, S2 = [], [], []
    for i in range(0, len(idx), 1024):
        x, cat, tk, tmask, rmask, y, pos, mu = data.batch(torch.tensor(idx[i:i + 1024], device=dev))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = net(x, cat, tk, tmask, rmask, mu)
        if CFG.get("mt"):
            s, s2 = s
            S2.append(s2.float().masked_fill(rmask, np.nan).cpu().numpy())
        s = s.float()
        S.append(s.masked_fill(rmask, np.nan).cpu().numpy())
        LL.append(winner_ll(s, rmask, y).cpu().numpy())
    predict.s2 = np.concatenate(S2) if S2 else None  # second head (mt variants)
    return np.concatenate(S), np.concatenate(LL)


MOD = ROOT / "models" / "v3"


def final(tag, seeds, through, moddir):
    """Fit on every labelled race up to `through` (all by default) with the tuned number of epochs; save weights + preprocessing."""
    t0 = time.time()
    b = build(all_races=True)
    data = Data(b)
    ep = json.load(open(RES / f"exp_{tag}_tune.json"))["epochs"]
    tr = np.flatnonzero(b["keep"] & b["labelled"] & (b["year"] >= int(FIRST_TRAIN[:4])) & (b["rdate"] <= np.datetime64(through)))
    moddir.mkdir(parents=True, exist_ok=True)
    meta = {**b["pre"], "tag": tag, "L": L, "cfg": CFG, "epochs": ep, "seeds": seeds, "nx": int(b["X"].shape[1]), "ntok": int(b["tok"].shape[1]),
            "train_races": int(len(tr)), "trained_through": str(pd.Timestamp(b["rdate"][tr].max()).date())}
    json.dump(meta, open(moddir / f"{tag}.json", "w"))
    for sd in seeds:
        net, _ = run(data, b, tr, tr[:1], ep, sd)
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, moddir / f"{tag}_seed{sd}.pt")
        print(f"  {tag} seed {sd}: {len(tr):,} races, {ep} epochs ({time.time() - t0:.0f}s)", flush=True)


def score(tag, lo, hi, moddir=MOD):
    """Network score (seed average) of every race dated lo..hi, with or without a result, from a saved final model."""
    global L
    meta = json.load(open(moddir / f"{tag}.json"))
    L = meta["L"]
    CFG.clear()
    CFG.update(meta["cfg"])
    b = build(all_races=True, meta=meta)
    data = Data(b)
    idx = np.flatnonzero(b["keep"] & (b["rdate"] >= np.datetime64(lo)) & (b["rdate"] <= np.datetime64(hi)))
    S = 0
    for sd in meta["seeds"]:
        net = Net(meta["nx"], meta["card"], meta["ntok"], CFG).to(dev)
        net.load_state_dict(torch.load(moddir / f"{tag}_seed{sd}.pt", map_location=dev))
        S = S + predict(net, data, idx)[0] / len(meta["seeds"])
    ri = b["RI"][idx]
    ok = ri >= 0
    return pd.DataFrame({"rid": b["d"].rid.values[ri[ok]], "numPmu": b["d"].numPmu.values[ri[ok]], "f_score": S[ok]})


def main(mode, tag="nn"):
    global L
    if "L20" in tag:
        L = 20
    if "big" in tag:
        CFG.update(d=256, hd=96)
    if "ref" in tag:
        CFG["ref"] = True
    if "mus" in tag:
        CFG["mus"] = True
    if "nohist" in tag:
        CFG["nohist"] = True
    if "norace" in tag:
        CFG["layers"] = 0
    if "mt" in tag:  # multi-task: head 1 = fundamental (results only), head 2 = late move (final market given the 30-min market)
        CFG["mt"] = True
    if "off" in tag:  # residual on the 30-min market (offset); the saved f_score is the residual only
        CFG["off"] = True
    if "kd" in tag:  # e.g. nn_L20_mus_kd1, nn_L20_mus_kd0p3: weight of the final-market distillation term
        CFG["kd"] = float(tag.split("kd")[-1].replace("p", "."))
    if mode == "final":  # python -I src/fr_nn.py final TAG [seeds] [through YYYY-MM-DD] [model dir]
        seeds = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2,3").split(",")]
        return final(tag, seeds, sys.argv[4] if len(sys.argv) > 4 else "2100-01-01", Path(sys.argv[5]) if len(sys.argv) > 5 else MOD)
    b = build()
    data = Data(b)
    yr, keep = b["year"], b["keep"]
    first = int(FIRST_TRAIN[:4])
    if mode == "tune":
        tr = np.flatnonzero(keep & (yr >= first) & (yr <= 2021))
        va = np.flatnonzero(keep & (yr == 2022))
        t0 = time.time()
        res = {}
        for E in ((4, 7, 10) if tag == "nn" else (10,)):  # full one-cycle schedules of different lengths: the end-of-schedule validation score decides
            net, _ = run(data, b, tr, va, E, 0)
            res[E] = float(np.mean(predict(net, data, va)[1]))
            print(f"  {E} epochs: validation winner log-loss {res[E]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        best = min(res, key=res.get)
        if tag != "nn":  # keep the 2022 validation scores of the last schedule for comparisons
            s_va = predict(net, data, va)[0]
            ri = b["RI"][va]
            ok = ri >= 0
            out = pd.DataFrame({"rid": b["d"].rid.values[ri[ok]], "numPmu": b["d"].numPmu.values[ri[ok]], "f_score": s_va[ok]})
            if predict.s2 is not None:
                out["f_score2"] = predict.s2[ok]
            out.to_parquet(EXP / f"{tag}_val2022.parquet", index=False)
        json.dump({"epochs": best, "val_by_epochs": res, "cfg": CFG}, open(RES / f"exp_{tag}_tune.json", "w"))
        print(f"best epochs {best}", flush=True)
        return
    ep = json.load(open(RES / f"exp_{tag}_tune.json"))["epochs"]
    seeds = [0, 1] if len(sys.argv) < 4 else [int(x) for x in sys.argv[3].split(",")]
    parts, parts2 = [], []
    for Y in OOF_YEARS:
        t0 = time.time()
        tr = np.flatnonzero(keep & (yr >= first) & (yr < Y))
        te = np.flatnonzero(keep & (yr == Y))
        if len(te) == 0:
            continue
        S, S2 = 0, 0
        for sd in seeds:
            net, _ = run(data, b, tr, te, ep, sd)
            s, ll = predict(net, data, te)
            S = S + s / len(seeds)
            if predict.s2 is not None:
                S2 = S2 + predict.s2 / len(seeds)
        ri = b["RI"][te]
        ok = ri >= 0
        rows = ri[ok]
        parts.append(pd.DataFrame({"rid": b["d"].rid.values[rows], "numPmu": b["d"].numPmu.values[rows], "f_score": S[ok]}))
        if CFG.get("mt"):
            parts2.append(pd.DataFrame({"rid": b["d"].rid.values[rows], "numPmu": b["d"].numPmu.values[rows], "f_score": S2[ok]}))
        ss = S - np.nanmax(S, 1, keepdims=True)
        p = np.exp(ss) / np.nansum(np.exp(ss), 1, keepdims=True)
        yy = b["y"][np.maximum(ri, 0)] * ok
        print(f"  {Y}: winner log-loss {float(-(yy * np.log(np.where(ok, p, 1))).sum(1).mean()):.4f} ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(parts).to_parquet(EXP / f"{tag}.parquet", index=False)
    if parts2:  # second head (late move), averaged over the same seeds
        pd.concat(parts2).to_parquet(EXP / f"{tag}_head2.parquet", index=False)
    print(f"-> exp/{tag}.parquet", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "nn")
