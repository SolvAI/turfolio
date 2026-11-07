"""Shared race-level modelling: softmax over each race's field (conditional logit), for LightGBM and for small linear stackers.

A race is a group of consecutive rows (sorted by race id). p_i = exp(s_i) / sum_{j in race} exp(s_j).
"""
import os

import numpy as np
import lightgbm as lgb
from scipy.optimize import minimize


def starts_of(rid):
    rid = np.asarray(rid)
    st = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1]
    return st, np.diff(np.r_[st, len(rid)])


def group_softmax(s, st, sz):
    m = np.maximum.reduceat(s, st)
    e = np.exp(s - np.repeat(m, sz))
    z = np.add.reduceat(e, st)
    return e / np.repeat(z, sz)


def race_logloss(p, y, st, sz):
    """Mean over races of -sum_i y_i log p_i (y sums to 1 per race; dead heats share)."""
    return float(-np.add.reduceat(y * np.log(np.clip(p, 1e-12, 1)), st).mean())


def top1_acc(p, y, st, sz):
    rid = np.repeat(np.arange(len(st)), sz)
    order = np.lexsort((-p, rid))  # within race, highest p first
    first = order[st]
    return float((y[first] > 0).mean())


def topk_acc(p, y, st, sz, k=3):
    rid = np.repeat(np.arange(len(st)), sz)
    order = np.lexsort((-p, rid))
    rank = np.empty(len(p), int)
    rank[order] = np.arange(len(p)) - np.repeat(st, sz)
    return float(np.add.reduceat(((rank < k) & (y > 0)).astype(float), st).clip(0, 1).mean())


def softmax_objective(st, sz, y):
    def f(preds, data):
        p = group_softmax(preds, st, sz)
        return p - y, np.maximum(p * (1 - p), 1e-6)
    return f


def fit_gbm(X, y, rid, params, rounds, init=None, cat="auto", valid=None):
    """valid = (Xv, yv, ridv, initv) -> early stopping on race log-loss; returns booster (best iteration kept)."""
    st, sz = starts_of(rid)
    ds = lgb.Dataset(X, y, init_score=init, categorical_feature=cat, free_raw_data=False)
    p = {**dict(verbose=-1, num_threads=int(os.environ.get("NTHREADS", 28)), objective=softmax_objective(st, sz, y)), **params}
    if valid is None:
        return lgb.train(p, ds, num_boost_round=rounds)
    Xv, yv, rv, iv = valid
    sv, zv = starts_of(rv)
    dv = lgb.Dataset(Xv, yv, init_score=iv, categorical_feature=cat, reference=ds, free_raw_data=False)

    def feval(preds, data):
        return "race_ll", race_logloss(group_softmax(preds, sv, zv), yv, sv, zv), False
    return lgb.train(p, ds, num_boost_round=rounds, valid_sets=[dv], feval=feval,
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])


def predict_gbm(b, X, rid, init=None):
    s = b.predict(X, raw_score=True, num_iteration=b.best_iteration or None)
    if init is not None:
        s = s + init
    st, sz = starts_of(rid)
    return group_softmax(s, st, sz), s


def fit_clogit(Z, y, rid, l2=1e-4):
    """Linear conditional logit: score = Z @ w (no intercept: it cancels within a race)."""
    st, sz = starts_of(rid)
    Z = np.asarray(Z, float)

    def obj(w):
        s = Z @ w
        p = group_softmax(s, st, sz)
        ll = -np.add.reduceat(y * np.log(np.clip(p, 1e-12, 1)), st).sum()
        # gradient: sum over races of sum_i (p_i - y_i) z_i
        g = Z.T @ (p - y)
        n = len(st)
        return ll / n + l2 * (w @ w), g / n + 2 * l2 * w
    w = minimize(obj, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return w


def predict_clogit(w, Z, rid):
    st, sz = starts_of(rid)
    s = np.asarray(Z, float) @ w
    return group_softmax(s, st, sz), s


# ------------------------------------------------------------------ top-pick analysis: accuracy vs odds vs profit
ODDS_BINS = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 1e9]
PROB_BINS = [0.0, 0.15, 0.25, 0.35, 0.5, 0.7, 1.01]


def top_rows(p, rid, st, sz):
    r = np.repeat(np.arange(len(st)), sz)
    return np.lexsort((-np.asarray(p), r))[st]


def _summ(won, odds):
    n = len(won)
    if n == 0:
        return {"n": 0}
    prof = np.where(won, odds - 1, -1.0)
    return {"n": int(n), "hit_rate": round(float(won.mean()), 4), "avg_odds": round(float(np.nanmean(odds)), 2),
            "breakeven_hit_rate": round(float(np.nanmean(1 / odds)), 4), "roi": round(float(np.nanmean(prof)), 4)}


def pick_report(g, pcol, pay_col, mkt_col):
    """Bet 1 unit on the model's top pick in every race, paid at pay_col (pool odds). Splits:
    agree / disagree with the market favourite (mkt_col), by the pick's odds, by the model's probability for it."""
    st, sz = starts_of(g.rid.values)
    pk = top_rows(g[pcol].values, g.rid.values, st, sz)
    fv = top_rows(g[mkt_col].values, g.rid.values, st, sz)
    won = g.y.values[pk] > 0
    odds = g[pay_col].values[pk].astype(float)
    ok = np.isfinite(odds) & (odds > 1)
    won, odds, p, agree = won[ok], odds[ok], g[pcol].values[pk][ok], (pk == fv)[ok]
    out = {"all": _summ(won, odds), "agrees_with_market_favourite": _summ(won[agree], odds[agree]),
           "differs_from_market_favourite": _summ(won[~agree], odds[~agree]),
           "share_agreeing": round(float(agree.mean()), 4)}
    ob = np.digitize(odds, ODDS_BINS) - 1
    out["by_odds"] = {f"{ODDS_BINS[i]:g}-{ODDS_BINS[i + 1]:g}": _summ(won[ob == i], odds[ob == i]) for i in range(len(ODDS_BINS) - 1)}
    pb = np.digitize(p, PROB_BINS) - 1
    out["by_model_prob"] = {f"{PROB_BINS[i]:.2f}-{min(PROB_BINS[i + 1], 1):.2f}": _summ(won[pb == i], odds[pb == i]) for i in range(len(PROB_BINS) - 1)}
    return out
