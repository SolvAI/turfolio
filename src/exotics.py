"""Combination-bet engine (exacta / quinella / trifecta / trio) on top of per-horse strengths.

Order-of-finish model (Plackett-Luce with stage-specific strengths, Benter 1994 / Lo-Bacon-Shone):
  P(1st = i)               = p_i                                   (win model)
  P(2nd = j | 1st = i)     = a_j / (sum_k a_k - a_i)                (stage-2 strengths a)
  P(3rd = l | i, j)        = b_l / (sum_k b_k - b_i - b_j)          (stage-3 strengths b)
Harville = a = b = p. All combinations of every race are enumerated in numba.

Pools: k = 2 or 3 horses, ordered (exacta / trifecta) or not (quinella / trio).
"""
from itertools import combinations, permutations

import numpy as np
from numba import njit

MAXN = 24
PB = np.array([0, .0005, .001, .002, .005, .01, .02, .05, .1, .2, .3, 1.01])  # combination-probability bins
RB = np.array([0, .5, .67, .8, .9, 1.1, 1.25, 1.5, 2, 3, 1e9])                # model / market probability ratio bins


def combo_table(k, ordered):
    """Flat table of all combinations for field sizes 0..MAXN: rows (i, j, l) with l = -1 when k = 2; offsets per field size."""
    rows, off = [], [0]
    for n in range(MAXN + 1):
        it = permutations(range(n), k) if ordered else combinations(range(n), k)
        for c in it:
            rows.append(list(c) + [-1] * (3 - k))
        off.append(len(rows))
    return np.array(rows, dtype=np.int64).reshape(-1, 3), np.array(off, dtype=np.int64)


@njit(cache=True)
def _ex(i, j, p, a, S2):
    return p[i] * a[j] / (S2 - a[i])


@njit(cache=True)
def _tri(i, j, l, p, a, b, S2, S3):
    return p[i] * a[j] / (S2 - a[i]) * b[l] / (S3 - b[i] - b[j])


@njit(cache=True)
def _prob(k, ordered, i, j, l, p, a, b, S2, S3):
    if k == 2:
        e = _ex(i, j, p, a, S2)
        if not ordered:
            e += _ex(j, i, p, a, S2)
        return e
    if ordered:
        return _tri(i, j, l, p, a, b, S2, S3)
    return (_tri(i, j, l, p, a, b, S2, S3) + _tri(i, l, j, p, a, b, S2, S3) + _tri(j, i, l, p, a, b, S2, S3)
            + _tri(j, l, i, p, a, b, S2, S3) + _tri(l, i, j, p, a, b, S2, S3) + _tri(l, j, i, p, a, b, S2, S3))


@njit(cache=True)
def _bin(x, edges):
    for t in range(len(edges) - 1):
        if x < edges[t + 1]:
            return t
    return len(edges) - 2


@njit(cache=True)
def run_pool(st, sz, p, a, b, pm, am, bm, k, ordered, C, off, act, W, PAY, NW, gacc, gpay, coef, xr, tau_min, pmin, cap):
    """Enumerate every combination of every race.

    p, a, b   model strengths (p normalised per race);  pm, am, bm  market strengths (benchmark for prices)
    act       (races, 3) local index of the 1st, 2nd, 3rd finisher (-1 if unknown)  -> accuracy
    W, PAY, NW  winning combinations of the pool (local indices, sorted if unordered), payout per 1 unit, count
    gacc      accuracy group per race (0 dev, 1 test, -1 skip);  gpay  pool group (-1 = no valid payout)
    coef, xr  payout model: log payout_hat = coef . [1, log H, log(P/H), xr[r, 0], xr[r, 1]] (coef[5] = smearing)
    returns per-race outputs, accumulators by bins, and candidate bets with EV > tau_min (decision rule search)
    """
    nr = len(st)
    P_act = np.full(nr, np.nan)
    H_act = np.full(nr, np.nan)
    P_w = np.full(nr, np.nan)
    H_w = np.full(nr, np.nan)
    top_p = np.full(nr, np.nan)
    top_hit = np.zeros(nr, np.int8)
    top_ret = np.full(nr, np.nan)
    mtop_hit = np.zeros(nr, np.int8)
    mtop_ret = np.full(nr, np.nan)
    agree = np.zeros(nr, np.int8)
    nP, nR = len(PB) - 1, len(RB) - 1
    acc_cal = np.zeros((2, nP, 3))   # by model prob: count, sum p, hits (accuracy groups)
    acc_m = np.zeros((2, nP, 3))     # by market prob: count, sum payout, sum H (pool groups)
    acc_r = np.zeros((2, nR, 3))     # by ratio model/market: count, sum payout, sum p
    c_r = np.zeros(cap, np.int32)
    c_p = np.zeros(cap, np.float32)
    c_h = np.zeros(cap, np.float32)
    c_ev = np.zeros(cap, np.float32)
    c_ret = np.zeros(cap, np.float32)
    nc = 0
    for r in range(nr):
        s, n = st[r], sz[r]
        if n < k or n > MAXN:
            continue
        pr, ar, br = p[s:s + n], a[s:s + n], b[s:s + n]
        qm, qa, qb = pm[s:s + n], am[s:s + n], bm[s:s + n]
        S2, S3, T2, T3 = ar.sum(), br.sum(), qa.sum(), qb.sum()
        # actual combination (from the finishing order) in this pool's convention
        x0, x1, x2 = act[r, 0], act[r, 1], act[r, 2]
        have = x0 >= 0 and x1 >= 0 and (k == 2 or x2 >= 0)
        if have and not ordered:
            if k == 2:
                if x1 < x0:
                    x0, x1 = x1, x0
            else:
                if x0 > x1:
                    x0, x1 = x1, x0
                if x1 > x2:
                    x1, x2 = x2, x1
                if x0 > x1:
                    x0, x1 = x1, x0
        g, gp = gacc[r], gpay[r]
        best, bi, mbest, mbi = -1.0, -1, -1.0, -1
        for c in range(off[n], off[n + 1]):
            i, j, l = C[c, 0], C[c, 1], C[c, 2]
            P = _prob(k, ordered, i, j, l, pr, ar, br, S2, S3)
            H = _prob(k, ordered, i, j, l, qm, qa, qb, T2, T3)
            if P > best:
                best, bi = P, c
            if H > mbest:
                mbest, mbi = H, c
            hit = have and i == x0 and j == x1 and (k == 2 or l == x2)
            if hit:
                P_act[r], H_act[r] = P, H
            if g >= 0:
                t = _bin(P, PB)
                acc_cal[g, t, 0] += 1
                acc_cal[g, t, 1] += P
                acc_cal[g, t, 2] += hit
            if gp >= 0:
                ret = 0.0
                for w in range(NW[r]):
                    if W[r, w, 0] == i and W[r, w, 1] == j and (k == 2 or W[r, w, 2] == l):
                        ret += PAY[r, w]
                        if w == 0:
                            P_w[r], H_w[r] = P, H
                t = _bin(H, PB)
                acc_m[gp, t, 0] += 1
                acc_m[gp, t, 1] += ret
                acc_m[gp, t, 2] += H
                t = _bin(P / H, RB)
                acc_r[gp, t, 0] += 1
                acc_r[gp, t, 1] += ret
                acc_r[gp, t, 2] += P
                if P >= pmin:
                    lh = np.log(H)
                    lp = coef[0] + coef[1] * lh + coef[2] * (np.log(P) - lh) + coef[3] * xr[r, 0] + coef[4] * xr[r, 1]
                    ev = P * np.exp(lp) * coef[5] - 1.0
                    if ev > tau_min and nc < cap:
                        c_r[nc], c_p[nc], c_h[nc], c_ev[nc], c_ret[nc] = r, P, H, ev, ret
                        nc += 1
        top_p[r] = best
        bi0, bi1, bi2 = C[bi, 0], C[bi, 1], C[bi, 2]
        top_hit[r] = have and bi0 == x0 and bi1 == x1 and (k == 2 or bi2 == x2)
        agree[r] = bi == mbi
        if gp >= 0:
            rt, mrt = 0.0, 0.0
            for w in range(NW[r]):
                if W[r, w, 0] == bi0 and W[r, w, 1] == bi1 and (k == 2 or W[r, w, 2] == bi2):
                    rt += PAY[r, w]
                if W[r, w, 0] == C[mbi, 0] and W[r, w, 1] == C[mbi, 1] and (k == 2 or W[r, w, 2] == C[mbi, 2]):
                    mrt += PAY[r, w]
            top_ret[r], mtop_ret[r] = rt, mrt
        m0, m1, m2 = C[mbi, 0], C[mbi, 1], C[mbi, 2]
        mtop_hit[r] = have and m0 == x0 and m1 == x1 and (k == 2 or m2 == x2)
    return (P_act, H_act, P_w, H_w, top_p, top_hit, top_ret, mtop_hit, mtop_ret, agree, acc_cal, acc_m, acc_r,
            c_r[:nc], c_p[:nc], c_h[:nc], c_ev[:nc], c_ret[:nc], nc == cap)
