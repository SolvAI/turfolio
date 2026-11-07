"""Builds results/REPORT.md from results/fr_report.json and results/hk_report.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
NAMES = {"mkt_final": "Market, final odds (known only at the off)", "mkt_ref": "Market, reference odds (~30 min before)",
         "mkt_ref_cal": "Market, reference odds, longshot bias corrected", "mkt_final_cal": "Market, final odds, longshot bias corrected",
         "mkt_cal": "Market, final odds, longshot bias corrected", "fund": "Fundamental model (no odds of the race)",
         "combo_ref": "Fundamental + reference odds (conditional logit)", "end_ref": "**END MODEL**: fundamental + reference odds + residual booster",
         "combo_final": "Fundamental + final odds (conditional logit)", "end_final": "Fundamental + final odds + residual booster",
         "combo": "Fundamental + final odds (conditional logit)", "end": "**END MODEL**: fundamental + final odds + residual booster"}


def table(block, models):
    rows = ["| Model | Races | Top-1 (winner picked) | Winner in top 3 | Log-loss |", "|---|---|---|---|---|"]
    for m in models:
        v = block[m]
        rows.append(f"| {NAMES.get(m, m)} | {v['races']:,} | {v['top1']:.1%} | {v['top3']:.1%} | {v['logloss']:.4f} |")
    return "\n".join(rows)


def diffs(block):
    rows = ["| Model vs benchmark | Log-loss diff [95 % CI] | Top-1 diff [95 % CI] |", "|---|---|---|"]
    for m, v in block.items():
        rows.append(f"| {NAMES.get(m, m)} | {v['logloss_diff']:+.4f} [{v['ll_ci95'][0]:+.4f}, {v['ll_ci95'][1]:+.4f}] | "
                    f"{v['top1_diff']:+.2%} [{v['top1_ci95'][0]:+.2%}, {v['top1_ci95'][1]:+.2%}] |")
    return "\n".join(rows)


def picks(tp, models, end):
    rows = ["| Model | Top-pick hit rate | Avg odds of the pick | Break-even hit rate | ROI betting the pick every race | Same pick as market favourite | When it differs: races / hit / odds / ROI |",
            "|---|---|---|---|---|---|---|"]
    for m in models:
        v = tp[m]
        a, d = v["all"], v["differs_from_market_favourite"]
        dd = f"{d['n']:,} / {d['hit_rate']:.1%} / {d['avg_odds']:.2f} / {d['roi']:+.1%}" if d.get("n") else "-"
        rows.append(f"| {NAMES.get(m, m)} | {a['hit_rate']:.1%} | {a['avg_odds']:.2f} | {a['breakeven_hit_rate']:.1%} | {a['roi']:+.1%} | {v['share_agreeing']:.0%} | {dd} |")
    out = ["\n".join(rows), "", f"End model's top pick, split by its odds and by its own probability:", "",
           "| Pick's final odds | Races | Hit rate | Break-even | ROI |", "|---|---|---|---|---|"]
    for k, v in tp[end]["by_odds"].items():
        if v.get("n"):
            out.append(f"| {k.replace('-1e+09', '+')} | {v['n']:,} | {v['hit_rate']:.1%} | {v['breakeven_hit_rate']:.1%} | {v['roi']:+.1%} |")
    out += ["", "| Model probability of its pick | Races | Hit rate | Avg odds | ROI |", "|---|---|---|---|---|"]
    for k, v in tp[end]["by_model_prob"].items():
        if v.get("n"):
            out.append(f"| {k} | {v['n']:,} | {v['hit_rate']:.1%} | {v['avg_odds']:.2f} | {v['roi']:+.1%} |")
    return "\n".join(out)


EX_NAMES = {"CO": "Couplé ordre (exacta)", "CG": "Couplé gagnant (quinella)", "TO": "Trio ordre (trifecta)", "TR": "Trio (any order)",
            "QIN": "Quinella", "TCE": "Tierce (trifecta)"}


def _ci(v):
    return f"[{v[0]:+.1%}, {v[1]:+.1%}]" if v else ""


def _roi(r):
    if not r or not r.get("bets"):
        return "0 bets"
    return f"{r['bets']:,} bets, {r['hits']:,} hits, ROI {r['roi']:+.1%} {_ci(r.get('roi_ci95'))}"


def exotics_section():
    p = RES / "exotics_report.json"
    if not p.exists():
        return []
    ex = json.load(open(p))
    hk = json.load(open(RES / "hk_exotics_report.json")) if (RES / "hk_exotics_report.json").exists() else None
    P = ("CO", "CG", "TO", "TR")
    out = ["## Combination bets: couplé (first two) and trio (first three)", "",
           "Order of finish: a Plackett-Luce model. The win probability comes from the end model. The 2nd and 3rd places use their own strengths "
           "(a conditional logit plus a small residual LightGBM per stage, walk-forward by year: train on 2018..Y-1, predict Y). "
           "Harville (2nd/3rd = the win probability renormalised) is the textbook baseline. Every combination of every race is enumerated. "
           "Dev = 2019-2024, test = 2025-01 -> 2026-05. *ref* = decided 30 minutes before the off (reference odds); *final* = with the final "
           "win odds (upper bound, not usable to bet). Payouts are the official PMU dividends per 1 EUR (paid at the final pool). Races where "
           "nobody held the winning combination (the pool then pays consolation combinations) or that were refunded are left out of the payout "
           "analysis (about 2 % of couplé races, a conservative choice for long-shot strategies).", ""]
    cg, tr = ex["CG"], ex["TR"]
    _qd = ex["CG"]["pool"]["test"]["end_stage_final"]["logloss"] - ex["CG"]["pool"]["test"]["pool_logloss_at_assumed_takeout"]
    qverb = "beats" if _qd < -0.01 else ("roughly ties" if abs(_qd) <= 0.01 else "is still worse than")
    tops = [ex[k]["top_combo"]["end_stage_ref"]["test"]["all"]["roi"] for k in P]
    rule_txt, hits = [], []
    for k in P:
        for v in ("ref", "final"):
            tc = ex[k]["value_rule"][v]["test_chosen"]
            if tc and tc.get("bets"):
                hits.append(tc["hits"])
                rule_txt.append(f"{EX_NAMES[k].split(' (')[0]} {'30 min' if v == 'ref' else 'final odds'} {tc['roi']:+.0%} ({tc['hits']} hits)")
    out += ["**Summary**", "",
            f"- **The model prices combinations much better than the textbook method.** Against Harville on the 30-minute market, the stage "
            f"models cut log-loss by {-cg['vs_market']['end_stage_ref']['logloss_diff']:.3f} (quinella) and {-tr['vs_market']['end_stage_ref']['logloss_diff']:.3f} (trio); "
            f"the combination probabilities are well calibrated (table below). For the quinella most of the gain comes from the better win model; for the "
            f"trio about two thirds come from modelling 2nd and 3rd place separately. Harville overrates favourites for the minor places: the fitted "
            f"discount exponent is about 0.8 for 2nd and 0.7 for 3rd (lower in trotting).",
            f"- **But the combination pools are sharper still.** On the test races where the pool ran, the trio pool's own implied log-loss "
            f"is {tr['pool']['test']['pool_logloss_at_assumed_takeout']:.3f}, against {tr['pool']['test']['end_stage_ref']['logloss']:.3f} for the model 30 minutes out "
            f"and {tr['pool']['test']['end_stage_final']['logloss']:.3f} even with the final win odds. For the quinella the model with final odds "
            f"{qverb} the pool ({cg['pool']['test']['end_stage_final']['logloss']:.3f} vs {cg['pool']['test']['pool_logloss_at_assumed_takeout']:.3f}), "
            f"but 30 minutes out it is worse ({cg['pool']['test']['end_stage_ref']['logloss']:.3f}). "
            "Exotic bettors do not price like Harville on the win odds.",
            f"- **No reliable profit.** Betting the model's top combination returns {min(tops):+.0%} to {max(tops):+.0%} (30-minute version). "
            f"The value rules chosen on dev give, on test: " + "; ".join(rule_txt) + ". Every confidence interval spans zero widely: the rules rest on "
            f"{min(hits)} to {max(hits)} winning tickets. The takeout on combinations (22-25 %) is larger than on win bets (about 16.5 %).",
            "- **Hong Kong** gives the same picture: the model adds nothing to the quinella pool and only matches the tierce pool, "
            "and every model-edge ROI bin loses.", "",
            "### Accuracy of the combination probabilities (test, all races with a clear 1-2-3)", "",
           "Log-loss of the actual combination (lower is better) / hit rate of the model's single most likely combination.", "",
           "| Pool | Races | Harville on the 30-min market | End model + Harville | **End model + stage models (ref)** | Harville on the final market | End model + stage models (final) |",
           "|---|---|---|---|---|---|---|"]
    for k in P:
        a = ex[k]["accuracy"]
        c = lambda m: f"{a[m]['test']['logloss']:.3f} / {a[m]['test']['top_combo_hit']:.1%}"
        out.append(f"| {EX_NAMES[k]} | {a['end_stage_ref']['test']['races']:,} | {c('mkt_harv_ref')} | {c('end_harv_ref')} | **{c('end_stage_ref')}** | "
                   f"{c('mkt_harv_final')} | {c('end_stage_final')} |")
    out += ["", "Stage models vs Harville on the same market (test; bootstrap by race day):", "",
            "| Pool | ref: log-loss diff [95 % CI] | ref: top-combination hit diff | final: log-loss diff [95 % CI] |", "|---|---|---|---|"]
    for k in P:
        vr, vf = ex[k]["vs_market"]["end_stage_ref"], ex[k]["vs_market"]["end_stage_final"]
        out.append(f"| {EX_NAMES[k]} | {vr['logloss_diff']:+.3f} [{vr['ll_ci95'][0]:+.3f}, {vr['ll_ci95'][1]:+.3f}] | {vr['top_hit_diff']:+.2%} "
                   f"[{vr['top_ci95'][0]:+.2%}, {vr['top_ci95'][1]:+.2%}] | {vf['logloss_diff']:+.3f} [{vf['ll_ci95'][0]:+.3f}, {vf['ll_ci95'][1]:+.3f}] |")
    out += ["", "Calibration of the end model's combination probabilities (test, ref): mean predicted vs actual frequency.", "",
            "| Predicted | Quinella combos | Quinella predicted / actual | Trio combos | Trio predicted / actual |", "|---|---|---|---|---|"]
    qc = {c["bin"]: c for c in ex["CG"]["calibration_end_stage_ref"]["test"]}
    tc = {c["bin"]: c for c in ex["TR"]["calibration_end_stage_ref"]["test"]}
    for b in qc:
        q, t = qc[b], tc.get(b)
        out.append(f"| {b} | {q['n']:,} | {q['predicted']:.4f} / {q['actual']:.4f} | " + (f"{t['n']:,} | {t['predicted']:.4f} / {t['actual']:.4f} |" if t else "- | - |"))
    out += ["", "### Against the combination pool itself (test, races where the pool ran)", "",
            "The pool's probability for the winning combination = return to players / payout (returns assumed: couplé 74.8 %, trio 77.9 %, "
            "OFDT/PMU averages; the ordre variants assumed equal). *Growth* = mean log return of spreading the bankroll over every combination "
            "in proportion to the model's probability: it does not depend on the assumed takeout, and > 0 would mean the model beats the pool after takeout.", "",
            "| Pool | Races | Median payout | Pool log-loss | Model (ref) log-loss / growth | Harville final market log-loss / growth | Model (final) log-loss / growth |",
            "|---|---|---|---|---|---|---|"]
    for k in P:
        b = ex[k]["pool"]["test"]
        g = lambda m: f"{b[m]['logloss']:.3f} / {b[m]['growth_bet_all_combos_in_proportion']:+.3f}"
        out.append(f"| {EX_NAMES[k]} | {b['races']:,} | {b['median_payout']:.1f} | {b['pool_logloss_at_assumed_takeout']:.3f} | {g('end_stage_ref')} | "
                   f"{g('mkt_harv_final')} | {g('end_stage_final')} |")
    co = {k: ex[k]["value_rule"]["ref"]["payout_model"]["coef_const_logH_logEdge_trot_logField"] for k in P}
    out += ["", "**The pools already price most of what the model knows.** Regressing the log payout of the winning combination on the "
            "market-Harville probability H and the model's edge P/H (dev winners) gives an edge coefficient of "
            + ", ".join(f"{EX_NAMES[k].split(' (')[0]} {co[k][2]:+.2f}" for k in P)
            + " (0 = the pool ignores what the model knows, -1 = the pool prices all of it). So, 30 minutes before the off, roughly 70-80 % "
            "of the model's apparent edge over Harville is already in the final combination payouts.", "",
            "### Top combination vs payout vs profit (test, 1 unit per race)", "",
            "| Pool | Model | Hit rate | Avg payout when hit | ROI [95 % CI] | Same as market's top | When different: bets / hit / ROI |", "|---|---|---|---|---|---|---|"]
    for k in P:
        for m in ("mkt_harv_ref", "end_stage_ref", "end_stage_final"):
            t = ex[k]["top_combo"][m]["test"]
            a, d = t["all"], t["differs_from_market_top"]
            dd = f"{d['bets']:,} / {d['hit_rate']:.1%} / {d['roi']:+.1%}" if d.get("bets") else "-"
            out.append(f"| {EX_NAMES[k]} | {m} | {a['hit_rate']:.1%} | {a['avg_payout_when_hit']:.2f} | {a['roi']:+.1%} {_ci(a.get('roi_ci95'))} | {t['share_same']:.0%} | {dd} |")
    out += ["", "### Where the money goes: ROI of betting every combination, by price and by model edge (ref)", "",
            "By the market-Harville probability of the combination (the exotic favourite-longshot bias):", "",
            "| Market prob of the combination | Quinella dev | Quinella test | Trio dev | Trio test |", "|---|---|---|---|---|"]
    rows = {}
    for k in ("CG", "TR"):
        for lab in ("dev", "test"):
            for x in ex[k]["roi_by_market_prob_ref"][lab]:
                rows.setdefault(x["market_prob"], {})[(k, lab)] = f"{x['roi']:+.0%} ({x['combos']:,})"
    for b, v in rows.items():
        out.append(f"| {b} | {v.get(('CG', 'dev'), '-')} | {v.get(('CG', 'test'), '-')} | {v.get(('TR', 'dev'), '-')} | {v.get(('TR', 'test'), '-')} |")
    out += ["", "By the model's edge (model probability / market-Harville probability):", "",
            "| Model / market | Quinella dev | Quinella test | Trio dev | Trio test |", "|---|---|---|---|---|"]
    rows = {}
    for k in ("CG", "TR"):
        for lab in ("dev", "test"):
            for x in ex[k]["roi_by_edge_ref"][lab]:
                rows.setdefault(x["model_over_market"].replace("-1e+09", "+"), {})[(k, lab)] = f"{x['roi']:+.0%} ({x['combos']:,})"
    for b, v in rows.items():
        out.append(f"| {b} | {v.get(('CG', 'dev'), '-')} | {v.get(('CG', 'test'), '-')} | {v.get(('TR', 'dev'), '-')} | {v.get(('TR', 'test'), '-')} |")
    out += ["", "### Value rule (bet when estimated EV > threshold; chosen on dev, applied once to test)", "",
            "EV = model probability x expected payout. Expected payout comes from the regression above (fit on dev winners). Threshold and minimum "
            "model probability are picked on dev, best ROI with at least 1,000 dev bets (300 for the small ordre pools).", "",
            "| Pool | Variant | Rule chosen on dev | Dev | Test |", "|---|---|---|---|---|"]
    for k in P:
        for v in ("ref", "final"):
            V = ex[k]["value_rule"][v]
            out.append(f"| {EX_NAMES[k]} | {v} | {V['chosen_on_dev']} | {_roi(V['dev'].get(V['chosen_on_dev']))} | {_roi(V['test_chosen'])} |")
    out += ["", "Realised ROI by estimated EV (test, ref; is the EV estimate right?):", "", "| Estimated EV | Quinella | Trio |", "|---|---|---|"]
    cg, tr = ex["CG"]["value_rule"]["ref"]["roi_by_estimated_ev"]["test"], ex["TR"]["value_rule"]["ref"]["roi_by_estimated_ev"]["test"]
    for a, b in zip(cg, tr):
        f = lambda x: f"{x['roi']:+.1%} ({x['bets']:,})" if x.get("bets") else "-"
        out.append(f"| {a['est_ev'].replace('..1e+09', '+')} | {f(a)} | {f(b)} |")
    if hk:
        out += ["", "### Hong Kong replication (final odds only; dividends exist for 2016-09 -> 2018-06)", "",
                "Same order models (conditional logits per stage, walk-forward by year). The value rule is France's (final variant), not re-tuned; "
                "the payout model is fit on the 239 HK races of 2016.", "",
                "| Pool | Test races | Harville on market | Discounted Harville on market | End model + stages | Pool log-loss (takeout 17.5 % / 25 %) | Model growth | Top combo ROI (model) | France rule on HK test |",
                "|---|---|---|---|---|---|---|---|---|"]
        for k in ("QIN", "TCE"):
            R = hk[k]
            a = R["accuracy"]
            c = lambda m: f"{a[m]['test']['logloss']:.3f} / {a[m]['test']['top_combo_hit']:.1%}"
            b = R["pool"]["test"]
            out.append(f"| {EX_NAMES[k]} | {b['races']:,} | {c('mkt_harv')} | {c('mkt_disc')} | {c('end_full')} | {b['pool_logloss_at_assumed_takeout']:.3f} | "
                       f"{b['end_full']['growth_bet_all_combos_in_proportion']:+.3f} | {R['top_combo']['end_full']['test']['all']['roi']:+.1%} | "
                       f"{R['value_rule']['rule_from_france_dev']}: {_roi(R['value_rule']['test_rule_from_france'])} |")
    return out



def _j(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def _r(r, ci=True):
    if not r or not r.get("bets"):
        return "-"
    s = f"{r['bets']:,} bets, {r['roi']:+.1%}"
    if ci and r.get("roi_ci95"):
        s += f" [{r['roi_ci95'][0]:+.1%}, {r['roi_ci95'][1]:+.1%}]"
    return s


def v2_section():
    st, hs = _j("exp_stack.json"), _j("exp_hk_stack.json")
    v1f, v2f = _j("v1/fr_report.json"), _j("fr_report.json")
    v1h, v2h = _j("v1/hk_report.json"), _j("hk_report.json")
    out = ["## Model v2 / v3: opponent-adjusted ratings and a neural race model", "",
           "v1 used one LightGBM fundamental model (157 hand-made features, at most 3 past runs per horse). v2 adds two things, and the "
           "end model, the combination models and the place study were all rebuilt on top of it.",
           "- **Ratings** (`src/fr_exp.py ratings`): a Plackett-Luce rating for every horse, driver/jockey and trainer, updated after each "
           "race day from the full finishing order (gradient step on the finishing-order likelihood, step size shrinking with experience, "
           "slow decay). Beating a strong field counts more than beating a weak one. Features of day D only use results of earlier days.",
           "- **A neural race model** (`src/fr_nn.py`): each runner is its feature vector plus the sequence of its last 10 starts "
           "(result, speed vs the field, beaten lengths, market opinion then, the field's mean rating = opponent strength, distance / class / "
           "track / driver changes relative to today). An attention pool summarises the history, a transformer runs over the field, and the "
           "loss is Plackett-Luce over the first 3 places. Walk-forward by year like the GBM (2 seeds, 10 epochs, chosen on 2022).",
           "- The fundamental score is a walk-forward blend (conditional logit, no odds) of the GBM with ratings and the network(s) "
           "(`src/fr_blend.py`); the network gets most of the weight.",
           "- **v3** (current; chosen on dev only): the network with a 20-run history, and a second one that also reads the form string "
           "(the last 12 placings, including races outside the data), each averaged over 4 seeds, blended with the GBM with ratings. "
           "It improves the 30-minute stack by a further 0.003 on dev.", ""]
    if st and "base win+rt nn" in st:
        s = dict(st["base win+rt nn"])
        s3 = st.get("base win+rt nn_L20 nn_L20_mus", {})
        for k_, v_ in s3.items():
            if "nn_L20" in k_:
                s[k_] = v_
            if k_.endswith("+all"):
                s[k_.replace("all", "v3all")] = v_
        out += ["### Development experiments (France 2018-2024; the hold-out was not used for any choice)", "",
                "Race log-loss of the winner, walk-forward; difference vs v1 with a day-bootstrap 95 % CI.", "",
                "| Fundamental variant | Alone (no odds) | Stacked with 30-min odds | Stacked with final odds |", "|---|---|---|---|"]
        lab = {"base": "v1 GBM", "win+rt": "GBM + ratings", "nn": "Neural race model (10 runs, 2 seeds) = v2 network", "all": "v2 blend (all three)",
               "nn_L20": "Network, 20 runs, 4 seeds", "nn_L20_mus": "Network, 20 runs + form string, 4 seeds", "v3all": "**v3 blend** (GBM + ratings and both)"}
        for nm in ("base", "win+rt", "nn", "all", "nn_L20", "nn_L20_mus", "v3all"):
            cells = []
            for kind in ("fund", "ref", "final"):
                k = f"{kind}:{nm}" if kind == "fund" else f"{kind}+{nm}"
                v = s.get(k)
                cells.append("-" if v is None else f"{v['dev_logloss']:.4f}" + ("" if nm == "base" else f" ({v['diff_vs_first']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}])"))
            out.append(f"| {lab[nm]} | " + " | ".join(cells) + " |")
    if hs and "base win+rt pl3+rt nn" in hs:
        s = hs["base win+rt pl3+rt nn"]
        out += ["", "Hong Kong (dev 2000-2016, final odds only):", "", "| Fundamental variant | Alone | Stacked with final odds |", "|---|---|---|"]
        lab = {"base": "v1 GBM", "win+rt": "GBM + ratings", "pl3+rt": "GBM + ratings, top-3 Plackett-Luce objective", "nn": "Neural race model", "all": "All"}
        for nm in ("base", "win+rt", "pl3+rt", "nn", "all"):
            cells = []
            for kind in ("fund", "mkt"):
                k = f"{kind}:{nm}" if kind == "fund" else f"{kind}+{nm}"
                v = s.get(k)
                cells.append("-" if v is None else f"{v['dev_logloss']:.4f}" + ("" if nm == "base" else f" ({v['diff_vs_first']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}])"))
            out.append(f"| {lab[nm]} | " + " | ".join(cells) + " |")
    abl = [("v1 GBM (157 features)", 2.0400), ("GBM + top-3 Plackett-Luce objective", 2.0353), ("GBM + ratings", 2.0241),
           ("GBM + ratings + top-3 objective", 2.0255)]
    for tag, name in (("exp_nn_tune.json", "Network (10-run history + field transformer)"), ("exp_nn_nohist_tune.json", "Network without the history"),
                      ("exp_nn_norace_tune.json", "Network without the field transformer"), ("exp_nn_L20_tune.json", "Network with a 20-run history"),
                      ("exp_nn_L20_big_tune.json", "Network, 20-run history, wider"),
                      ("exp_nn_mus_tune.json", "Network + form-string tokens"), ("exp_nn_L20_mus_tune.json", "Network, 20-run history + form string"),
                      ("exp_nn_ref_tune.json", "Network given this race's 30-min odds too (end-to-end; compare with the stack: 1.9425 on the same 2022 races, the end-to-end network 1.9565)")):
        j = _j(tag)
        if j:
            abl.append((name, j["val_by_epochs"]["10"]))
    out += ["", "### Where the gain comes from (France, train 2015-2021, validate 2022, fundamental only)", "",
            "| Variant | 2022 race log-loss |", "|---|---|"] + [f"| {n} | {v:.4f} |" for n, v in abl]
    out += ["", "The gain is the history sequence: without it the network is no better than the GBM with ratings; the transformer over the field "
            "adds nothing; 20 runs instead of 10 and the form string add a little (both used in v3).", "",
            "**The gain grows with the length of the history** (dev 2018-2024, network minus v1 GBM, race log-loss):", ""]
    gh = _j("exp_nn_gain_by_history.json")
    if gh:
        out += ["| Winner's earlier starts in the data | " + " | ".join(r_["bin"] for r_ in gh["winner_prior_starts"]) + " |",
                "|---|" + "---|" * len(gh["winner_prior_starts"]),
                "| Races | " + " | ".join(f"{r_['races']:,}" for r_ in gh["winner_prior_starts"]) + " |",
                "| Network vs GBM | " + " | ".join(f"{r_['nn_minus_gbm']:+.3f}" for r_ in gh["winner_prior_starts"]) + " |",
                "| GBM + ratings vs GBM | " + " | ".join(f"{r_['gbm_rt_minus_gbm']:+.3f}" for r_ in gh["winner_prior_starts"]) + " |", "",
                "The same holds by the field's average history, and in every discipline (" +
                ", ".join(f"{k.lower()} {v_['nn_minus_gbm']:+.3f}" for k, v_ in gh["by_specialite"].items()) + "). For debutants the network "
                "is slightly worse; for horses with 20+ earlier starts it gains 0.07. Three hand-made lags and a few aggregates cannot use the "
                "detail of the last 10 runs (who was beaten, how fast, in what company, under what conditions); attention over them can."]
    out += ["", "Leakage checks for a gain this large: the history holds only strictly earlier starts of the same horse (of 1.76M consecutive "
            "same-horse pairs, none share a race and 175 share a start time); every input of the current race is pre-race (the same features as "
            "the GBM); the gain disappears when the history is masked; and against the final odds it shrinks to -0.009, the size one expects "
            "from better information rather than from a leaked result (which would beat the final market by far more)."]
    v2s = _j("v2/fr_report.json")
    if v1f and v2f:
        a, b = v1f["test_2025_2026"], v2f["test_2025_2026"]
        c2 = (v2s or v2f)["test_2025_2026"]
        g = lambda blk, path: path(blk)
        rows = [("Fundamental alone: log-loss", lambda q: q["overall"]["fund"]["logloss"], "{:.4f}"),
                ("Fundamental alone: top pick wins", lambda q: q["overall"]["fund"]["top1"], "{:.1%}"),
                ("End model, 30 min out: log-loss", lambda q: q["overall"]["end_ref"]["logloss"], "{:.4f}"),
                ("End model, 30 min out: top pick wins", lambda q: q["overall"]["end_ref"]["top1"], "{:.1%}"),
                ("End model, 30 min out, vs 30-min market", lambda q: q["vs_market_ref"]["end_ref"]["logloss_diff"], "{:+.4f}"),
                ("End model with final odds, vs final market", lambda q: q["vs_market_final"]["end_final"]["logloss_diff"], "{:+.4f}")]
        out += ["", "### v1 vs v2 vs v3 on the France hold-out (2025-01 -> 2026-05)", "", "| | v1 | v2 | v3 (current) |", "|---|---|---|---|"]
        out += [f"| {n} | {f.format(fn(a))} | {f.format(fn(c2))} | {f.format(fn(b))} |" for n, fn, f in rows]
        out += [f"| Value rule at 30 min (naive, chosen on dev): ROI | {v1f['betting_parimutuel']['test']['roi']:+.1%} | "
                f"{(v2s or v2f)['betting_parimutuel']['test']['roi']:+.1%} | {v2f['betting_parimutuel']['test']['roi']:+.1%} |",
                f"| Market: 30-min / final odds log-loss | {a['overall']['mkt_ref']['logloss']:.4f} / {a['overall']['mkt_final']['logloss']:.4f} | same | same |",
                "", "The hold-out was evaluated once per version; every choice between versions was made on dev."]
    if v1h and v2h:
        a, b = v1h["test_2017_2018"], v2h["test_2017_2018"]
        out += ["", f"Hong Kong hold-out 2017-18 ({b['overall']['end']['races']:,} races): fundamental {a['overall']['fund']['logloss']:.4f} -> "
                f"{b['overall']['fund']['logloss']:.4f}; end model vs final market {a['vs_market']['end']['logloss_diff']:+.4f} -> "
                f"{b['vs_market']['end']['logloss_diff']:+.4f} [{b['vs_market']['end']['ll_ci95'][0]:+.4f}, {b['vs_market']['end']['ll_ci95'][1]:+.4f}]. "
                "The network improves the HK fundamental model as much as in France, but the HK final odds already contain it. The v3 changes "
                "(20 runs, 4 seeds) improve the HK fundamental a little more but not the market stack (dev -0.0012 either way), so HK stays on the v2 recipe."]
    return out


def value_section():
    v = _j("value_report.json")
    if not v:
        return []
    r = v["ref"]
    pm = r["payout_model"]["coef_const_logRefOdds_logEdge_trot_logField_logRefOddsXtrot"]
    out = ["## Win bets: late money, pool size and the closing line (France, v3)", "",
           "In a pari-mutuel pool every bet is paid at the final odds. Late money moves toward what the model sees, so a horse the model "
           "likes at the 30-minute odds usually starts shorter. `src/fr_value.py` first predicts the final odds (regression on dev runners: "
           f"log final odds = {pm[1]:.2f} log ref odds {pm[2]:+.2f} log(model p / market p) + ...; i.e. the final odds shrink by edge^{abs(pm[2]):.2f}), "
           "then bets when p x predicted final odds - 1 exceeds a threshold chosen on 2018-24.", "",
           "| Rule (decided 30 min out, paid at final odds) | Chosen on dev | Dev 2018-24 | Hold-out 2025-26 |", "|---|---|---|---|"]
    for name, lab in (("naive_ref_odds", "EV at the 30-min odds"), ("predicted_final_odds", "EV at the predicted final odds (fit on all runners)"),
                      ("predicted_final_odds_given_win", "**EV at the predicted final odds of a winner** (fit on winners)")):
        if name in r:
            x = r[name]
            out.append(f"| {lab} | {x['chosen_on_dev']} | {_r(x['dev_chosen'], False)} | {_r(x['test_chosen'])} |")
    if "payout_model_winners" in r:
        out += ["", "What pays is the final odds of a winner, and the late money is informed: winners shorten more than losers "
                f"(edge exponent {r['payout_model_winners'][2]:+.2f} for winners against {pm[2]:+.2f} for all runners). The winner fit is the "
                "correct expected payout, and its estimated EV matches the realised return on dev. It was adopted after the all-runner rule "
                "had been run once on the hold-out, so the two hold-out numbers should be read together: +2 % to +12 %, every interval "
                "spanning zero. The tables below use the winner fit."]
    c = v.get("checks")
    if c:
        out += ["", f"Same rule ({c['rule']}) paid with the official PMU win dividends: dev {_r(c['dev']['official_dividend'])}; "
                f"hold-out {_r(c['test']['official_dividend'])}. Per year: " +
                ", ".join(f"{y} {x['roi']:+.0%}" for y, x in r["per_year_chosen"].items() if x.get("bets")) + ".",
                "", "**Where the profit is: by size of the win pool** (net of takeout, rebuilt from the winner's dividend x winning stakes). "
                "Own stake = the bet's effect on its own payout.", "",
                "| Net win pool | Dev bets | Dev ROI (no stake effect / 10 EUR / 50 EUR) | Hold-out bets | Hold-out ROI (no stake effect / 10 EUR / 50 EUR) |", "|---|---|---|---|---|"]
        for a, b in zip(c["dev"]["by_pool_size"], c["test"]["by_pool_size"]):
            f = lambda q: f"{q['roi_stake_0eur']:+.0%} / {q['roi_stake_10eur']:+.0%} / {q['roi_stake_50eur']:+.0%}" if q["bets"] else "-"
            out.append(f"| {a['net_win_pool_eur']} EUR | {a['bets']:,} | {f(a)} | {b['bets']:,} | {f(b)} |")
        thin = v.get("by_liquidity", {}).get("test", [{}])[0]
        share = thin.get("races", 0) / max(sum(r_["races"] for r_ in v.get("by_liquidity", {}).get("test", [])), 1)
        out += ["", f"Pools under 20k EUR ({share:.1%} of the hold-out races, mostly regional meetings late in the programme) carry the largest "
                "returns, and that is where a stake of a few euros already moves the price. Outside them the returns scatter around zero from one "
                "band to the next (see the liquid-pool rule below). Dev returns are optimistic: the threshold was chosen on them."]
        lr = v.get("liquid_rule")
        if lr:
            out += ["", f"**Liquid pools only** (net win pool >= {lr['min_pool_eur']:,.0f} EUR, threshold re-chosen on dev): {lr['chosen_on_dev']}; "
                    f"dev {_r(lr['dev_chosen'], False)}; hold-out {_r(lr['test_chosen'])}."]
    lq = v.get("by_liquidity")
    if lq:
        out += ["", "**Accuracy by pool size, and the closing line.** The last column bets at the final odds with the final-odds model "
                "(EV > 20 %): an upper bound, since nobody can bet at the exact final odds (the French data has no snapshot between 30 minutes "
                "and the off to measure how much the odds still move).", "",
                "| Net win pool | Races (hold-out) | End model (final odds) vs final market [CI] | End model (30 min) vs 30-min market | "
                "Bet at the final odds, EV > 20 %: dev | hold-out |", "|---|---|---|---|---|---|"]
        for a, b in zip(lq["dev"], lq["test"]):
            out.append(f"| {b['net_win_pool_eur']} EUR | {b['races']:,} | {b['end_final_minus_mkt_final']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | "
                       f"{b['end_ref_minus_mkt_ref']:+.4f} | {_r(a['last_minute_rule_ev>0.2'], False)} | {_r(b['last_minute_rule_ev>0.2'], False)} |")
        out += ["", "The model beats the final market in every pool size, most in thin pools. Given the final odds it would be profitable in "
                "pools up to 200k EUR on dev, and on the hold-out except one 27-bet cell; 30 minutes out, most of that edge is taken by the late money."]
    return out


def timing_section():
    lk, hk = _j("locked_odds_report.json"), _j("hk_timing.json")
    out = []
    if lk:
        out += ["### The opening edge: what if the 30-minute price could be locked?", "",
                "In a pari-mutuel pool an early bet does not lock its price: it is paid at the final odds. If the 30-minute PMU price could be "
                "locked (as with a fixed-odds bookmaker or an exchange quoting the same prices), the same 30-minute decisions would return:", "",
                "| Model EV at the 30-min odds | v1: locked 30-min price (hold-out) | v3: locked 30-min price (hold-out) | v3: paid at final odds (reality) |",
                "|---|---|---|---|"]
        cur = lk.get("current", lk.get("v2"))
        v1rows = lk.get("v1") or cur
        for a, b in zip(v1rows, cur):
            ta, tb = a["test_locked"], b["test_locked"]
            out.append(f"| > {b['ev>']:.0%} | {ta['bets']:,} bets, {ta['roi']:+.0%} [{ta['roi_ci95'][0]:+.0%}, {ta['roi_ci95'][1]:+.0%}] | "
                       f"{tb['bets']:,} bets, {tb['roi']:+.0%} [{tb['roi_ci95'][0]:+.0%}, {tb['roi_ci95'][1]:+.0%}] | {b['test_parimutuel']:+.1%} |")
        lo_c, hi_c = min(r_["test_locked"]["roi"] for r_ in cur), max(r_["test_locked"]["roi"] for r_ in cur)
        lo_1, hi_1 = min(r_["test_locked"]["roi"] for r_ in v1rows), max(r_["test_locked"]["roi"] for r_ in v1rows)
        out += ["", f"The model knows 30 minutes early where the late money will go. Against a locked early price that is worth {lo_c:+.0%} to "
                f"{hi_c:+.0%} (even v1: {lo_1:+.0%} to {hi_1:+.0%}); in the pari-mutuel pool the late money takes it, and the same bets are "
                "about break-even."]
    if hk:
        m = hk["share_of_30min_to_final_gain_still_missing"]
        out += ["", "### How late is the late money? (Hong Kong live odds, " + f"{hk['races']:,} races 2016-18)", "",
                "Market log-loss of the winner k minutes before the odds freeze (the off), against the final odds "
                f"({hk['final_odds_logloss']:.4f}). Snapshots near the off are about {hk['median_seconds_between_last_two_snapshots_before_freeze'] / 60:.0f} minutes apart.", "",
                "| Minutes before the off | 30 | 15 | 10 | 5 | 3 | last snapshot before the off |", "|---|---|---|---|---|---|---|",
                "| Log-loss | " + " | ".join(f"{hk['logloss_k_min_before_last_snapshot'][k]:.4f}" for k in ("30", "15", "10", "5", "3", "0.5")) + " |",
                "| Share of the 30-min -> final gain still missing | " + " | ".join(f"{m[k]:.0%}" for k in ("30", "15", "10", "5", "3", "0.5")) + " |", "",
                "In Hong Kong about half of the late information reaches the odds only in the last few minutes. If France is similar, the "
                "closing-line edge is only reachable by betting in the final seconds, and only in part."]
    return out


def place_section():
    p = _j("place_report.json")
    if not p:
        return []
    out = ["## Place bets (simple placé, France, v3)", "",
           "A place bet wins if the horse finishes in the first 3 (first 2 with 4-7 starters). The pool is split equally between the placed "
           "horses: payout = 1 + (R x pool - stakes on the placed) / (m x stake on the horse), rounded down to 0.10, minimum 1.10 (1.05 "
           "in some races since 2021). PMU publishes each placed horse's payout and number of winning stakes, so (payout - 1) x stakes is "
           "equal across the placed horses of a race (checked: median deviation 2.5 %). `src/fr_place.py` prices every horse with the order "
           "models of the combination study (P(place) from Plackett-Luce with stage-2/3 strengths) and a structural model of the place pool.", ""]
    pm = p["pool_model"]
    out += [f"- **The place pool is flatter than the win pool:** stakes rise as (win-market share)^{pm['final']['gamma_trot']:.2f} in trotting and "
            f"^{pm['final']['gamma_flat']:.2f} in flat/jumps. Estimated return rate {pm['final']['return_rate_trot']:.1%} (trot) / {pm['final']['return_rate_flat']:.1%} (flat).",
            f"- **The payouts already price most of the 30-minute edge:** regressing log(payout - 1) of placed horses on the structural "
            f"prediction and log(model / market P(place)) gives {pm['ref']['payout_regression_const_logEhatm1_logEdge_trot_logField'][2]:+.2f} at 30 min "
            f"and {pm['final']['payout_regression_const_logEhatm1_logEdge_trot_logField'][2]:+.2f} with the final odds.", "",
            "| P(place) model (hold-out) | Log-loss per horse | Top place pick places | Top place pick ROI |", "|---|---|---|---|"]
    names = {"mkt_harv_ref": "Harville, 30-min market", "mkt_disc_ref": "Discounted Harville, 30-min market", "end_stage_ref": "End model + stages, 30 min",
             "mkt_harv_final": "Harville, final market", "mkt_disc_final": "Discounted Harville, final market", "end_stage_final": "End model + stages, final odds"}
    for k, n in names.items():
        a = p["accuracy"][k]["test"]
        out.append(f"| {n} | {a['logloss_per_horse']:.4f} | {a['top_pick_places']:.1%} | {a['top_pick_roi']:+.1%} |")
    out += ["", "ROI of every place bet by the horse's final-odds win probability (hold-out): " +
            ", ".join(f"{x['win_prob_final']} {x['roi']:+.0%}" for x in p["roi_by_win_prob_final"]["test"]) + ".",
            "Heavy favourites, where the minimum payout binds, come close to break-even but do not beat it.", "",
            "| Value rule (chosen on dev) | Chosen | Dev | Hold-out |", "|---|---|---|---|"]
    for v in ("ref", "final"):
        x = p["value_rule"][v]
        out.append(f"| {'30 min out' if v == 'ref' else 'final odds (upper bound)'} | {x['chosen_on_dev']} | {_r(x['dev'][x['chosen_on_dev']], False)} | {_r(x['test_chosen'])} |")
    bp = p["value_rule"]["ref"].get("by_place_pool")
    if bp:
        out += ["", "The 30-minute rule by size of the place pool (gross payout pool = return rate x pool, which does not depend on the result; "
                "bands = quintiles of dev races):", "", "| Place pool (EUR) | Dev | Hold-out |", "|---|---|---|"]
        for a, b in zip(bp["dev"], bp["test"]):
            out.append(f"| {a['net_place_pool_eur']} | {_r(a, False)} | {_r(b, False)} |")
        sp = p["value_rule"]["ref"].get("small_pools", {})
        if sp:
            out += ["", f"No pool-size pattern. (A first version filtered on the *net* place pool, R x pool minus the stakes on the placed "
                    "horses. That quantity shrinks when favourites place, so filtering on it selects races where favourites placed: it showed "
                    "a spurious +15 % on dev and hold-out alike for these favourite bets. With the outcome-free measure the effect is gone: "
                    f"the cap chosen on dev gives {_r(sp['test']['no_stake_effect'])} on the hold-out.)"]
    out += ["", "Realised ROI by estimated EV (hold-out): " + "; ".join(
        f"{'30 min' if v == 'ref' else 'final'}: " + ", ".join(f"{x['est_ev']} {x['roi']:+.0%} ({x['bets']:,})" for x in p["value_rule"][v]["roi_by_estimated_ev"]["test"][3:])
        for v in ("ref", "final"))]
    return out


def main():
    fr = json.load(open(RES / "fr_report.json"))
    out = ["# HorseCash accuracy report", "",
           "All numbers are out of sample: every prediction comes from models fit only on earlier years (yearly walk-forward).",
           "Log-loss = mean of -log(probability given to the winner); lower is better. Top-1 = the model's most likely horse won.", ""]
    t = fr["test_2025_2026"]
    o, vr, vf = t["overall"], t["vs_market_ref"]["end_ref"], t["vs_market_final"]["end_final"]
    tp = fr["top_pick"]["test_2025_2026"]
    bp = fr["betting_parimutuel"]
    vfund = t["vs_market_ref"]["fund"]
    val, plc, hk_ = _j("value_report.json"), _j("place_report.json"), _j("hk_report.json")
    out += ["## Key findings (France hold-out 2025-01 -> 2026-05, " + f"{o['mkt_final']['races']:,} races; model v3)", "",
            f"- **The fundamental model alone now beats the 30-minute market.** Without seeing this race's odds, model v3 (a neural race model that "
            f"reads each horse's last 20 runs and its form string, plus opponent-adjusted ratings) scores log-loss {o['fund']['logloss']:.3f} against {o['mkt_ref']['logloss']:.3f} "
            f"for the reference odds ({vfund['logloss_diff']:+.3f}, CI {vfund['ll_ci95'][0]:+.3f} to {vfund['ll_ci95'][1]:+.3f}). v1 (gradient boosting) scored 2.038.",
            f"- **Usable 30 minutes before the off:** the end model's top pick wins {o['end_ref']['top1']:.1%} against {o['mkt_ref']['top1']:.1%} for the favourite "
            f"at that time (+{vr['top1_diff']:.1%}, CI {vr['top1_ci95'][0]:+.1%} to {vr['top1_ci95'][1]:+.1%}); log-loss {o['end_ref']['logloss']:.3f} vs {o['mkt_ref']['logloss']:.3f}. "
            f"The final market ({o['mkt_final']['logloss']:.3f}) is still sharper: the money of the last 30 minutes is informed.",
            f"- **The model beats the closing line.** Given the final odds too, it beats the final market: log-loss {vf['logloss_diff']:+.4f} "
            f"(CI {vf['ll_ci95'][0]:+.4f} to {vf['ll_ci95'][1]:+.4f}), in every pool size, most in thin pools.",
            f"- **Profit is a question of liquidity and timing.** Betting the top pick in every race returns {tp['end_ref']['all']['roi']:+.1%} "
            f"(the pool keeps about 16.5 %). The naive 30-minute value rule returns {bp['test']['roi']:+.1%} on {bp['test']['bets']:,} bets."]
    if val and val.get("checks"):
        c = val["checks"]["test"]["official_dividend"]
        a = val["ref"]["predicted_final_odds"]["test_chosen"]
        lk_ = _j("locked_odds_report.json") or {}
        cur_ = lk_.get("current", lk_.get("v2", [{"test_locked": {"roi": float("nan")}}]))
        lk_lo, lk_hi = min(r_["test_locked"]["roi"] for r_ in cur_), max(r_["test_locked"]["roi"] for r_ in cur_)
        out[-1] += (f" Rules that predict the final odds first return {c['roi']:+.1%} ({c['bets']:,} bets, CI {c['roi_ci95'][0]:+.0%} to "
                    f"{c['roi_ci95'][1]:+.0%}) to {a['roi']:+.1%} ({a['bets']:,} bets) on the hold-out, depending on the payout model: break-even "
                    "to modestly positive, never significant, and the best returns sit in pools under 20k EUR where a few euros move the price. "
                    f"Against a locked 30-minute price the same decisions would return {lk_lo:+.0%} to {lk_hi:+.0%}; betting at the final odds would also be "
                    "clearly profitable. Both are out of reach in a pari-mutuel pool (see 'Win bets').")
    if plc:
        x = plc["value_rule"]["ref"]["test_chosen"]
        out.append(f"- **Place bets** (simple placé) 30 minutes out: the rule chosen on dev returns {x['roi']:+.1%} on {x['bets']:,} hold-out bets "
                   f"(CI {x['roi_ci95'][0]:+.1%} to {x['roi_ci95'][1]:+.1%}); positive on dev too, not significant.")
    out.append("- **Combination bets** (couplé, trio): the model prices them far better than Harville, but the pools are about as sharp and keep "
               "22-25 %; no rule is reliably profitable.")
    if hk_:
        h = hk_["test_2017_2018"]["vs_market"]["end"]
        out.append(f"- **Hong Kong**, a much bigger and tighter market: the network improves the fundamental model as much as in France, but the "
                   f"end model adds little to the final odds ({h['logloss_diff']:+.4f}, CI {h['ll_ci95'][0]:+.4f} to {h['ll_ci95'][1]:+.4f}).")
    out.append("")
    out += ["## France (PMU, 2014-2026)", "", "### Hold-out: 2025-01-01 -> 2026-05-05 (never used for any choice)", "",
            table(fr["test_2025_2026"]["overall"], ["mkt_final", "mkt_ref", "mkt_ref_cal", "fund", "combo_ref", "end_ref", "end_final"]), "",
            "Against the final market:", "", diffs(fr["test_2025_2026"]["vs_market_final"]), "",
            "Against the reference market (what is known 30 minutes before the off):", "", diffs(fr["test_2025_2026"]["vs_market_ref"]), "",
            "### Development years 2018-2024", "",
            table(fr["dev_2018_2024"]["overall"], ["mkt_final", "mkt_ref", "mkt_ref_cal", "fund", "combo_ref", "end_ref", "end_final"]), "",
            "### By discipline (hold-out)", "", "| Discipline | Races | Final market | Ref market | Fundamental | End model (ref) | End model (final) |", "|---|---|---|---|---|---|---|"]
    for s, v in fr["test_2025_2026"]["by_specialite"].items():
        out.append(f"| {s} | {v['mkt_final']['races']:,} | {v['mkt_final']['top1']:.1%} / {v['mkt_final']['logloss']:.3f} | "
                   f"{v['mkt_ref']['top1']:.1%} / {v['mkt_ref']['logloss']:.3f} | {v['fund']['top1']:.1%} / {v['fund']['logloss']:.3f} | "
                   f"{v['end_ref']['top1']:.1%} / {v['end_ref']['logloss']:.3f} | {v['end_final']['top1']:.1%} / {v['end_final']['logloss']:.3f} |")
    out += ["", "(top-1 / log-loss)", "", "### Per year", "", "| Year | Races | Final market | Ref market | Fundamental | End model (ref) | End model (final) |", "|---|---|---|---|---|---|---|"]
    for y, v in fr["per_year"].items():
        out.append(f"| {y} | {v['mkt_final']['races']:,} | {v['mkt_final']['top1']:.1%} / {v['mkt_final']['logloss']:.3f} | "
                   f"{v['mkt_ref']['top1']:.1%} / {v['mkt_ref']['logloss']:.3f} | {v['fund']['top1']:.1%} / {v['fund']['logloss']:.3f} | "
                   f"{v['end_ref']['top1']:.1%} / {v['end_ref']['logloss']:.3f} | {v['end_final']['top1']:.1%} / {v['end_final']['logloss']:.3f} |")
    out += ["", "### Calibration of the end model (hold-out)", "", "| Predicted | Runners | Mean predicted | Actual win rate |", "|---|---|---|---|"]
    for r in fr["calibration_test_end_ref"]:
        out.append(f"| {r['end_ref']} | {r['n']:,} | {r['predicted']:.3f} | {r['actual']:.3f} |")
    out += ["", "### Top pick vs odds vs profit (hold-out)", "",
            "One unit on each model's top pick in every race, paid at the final pool odds (in a pari-mutuel pool every bet gets the final odds).",
            "Break-even hit rate = average of 1/odds of the picks: the hit rate needed to make money at those prices.", "",
            picks(fr["top_pick"]["test_2025_2026"], ["mkt_final", "mkt_ref", "fund", "combo_ref", "end_ref", "end_final"], "end_ref"), "",
            "Development years 2018-2024:", "",
            picks(fr["top_pick"]["dev_2018_2024"], ["mkt_final", "mkt_ref", "fund", "end_ref", "end_final"], "end_ref")]
    b = fr["betting_parimutuel"]
    out += ["", "### Betting check (secondary)", "",
            "Decide 30 minutes before the off (end-model probability x reference odds - 1 > threshold), get paid the final pool odds.",
            f"Threshold chosen on 2018-24: {b['threshold_chosen_on_dev']}. Hold-out: {b['test']['bets']:,} bets, ROI {b['test']['roi']:+.1%} "
            f"(market-only rule, same threshold: {b['test_mkt_ref_cal_same_rule']['bets']:,} bets, ROI {b['test_mkt_ref_cal_same_rule']['roi']:+.1%}).", ""]
    hk_path = RES / "hk_report.json"
    if hk_path.exists():
        hk = json.load(open(hk_path))
        out += ["## Hong Kong (HKJC, 1979-2018): replication on a second market", "", "Only final odds exist here, so the end model uses them.", "",
                "### Hold-out: 2017-2018", "", table(hk["test_2017_2018"]["overall"], ["mkt_final", "mkt_cal", "fund", "combo", "end"]), "",
                diffs(hk["test_2017_2018"]["vs_market"]), "", "### Development years 1996-2016", "",
                table(hk["dev_1996_2016"]["overall"], ["mkt_final", "mkt_cal", "fund", "combo", "end"]), "", diffs(hk["dev_1996_2016"]["vs_market"]), "",
                "### Top pick vs odds vs profit (hold-out 2017-2018)", "",
                "Paid at the final odds. The end model here uses the final odds to choose, so its ROI is an upper bound; the fundamental model never sees odds.", "",
                picks(hk["top_pick"]["test_2017_2018"], ["mkt_final", "fund", "combo", "end"], "end"), "",
                "### By period", "", "| Period | Races | Market | Fundamental | End model |", "|---|---|---|---|---|"]
        for p, v in hk["per_period"].items():
            out.append(f"| {p} | {v['mkt_final']['races']:,} | {v['mkt_final']['top1']:.1%} / {v['mkt_final']['logloss']:.3f} | "
                       f"{v['fund']['top1']:.1%} / {v['fund']['logloss']:.3f} | {v['end']['top1']:.1%} / {v['end']['logloss']:.3f} |")
    out += [""] + v2_section() + [""] + value_section() + [""] + timing_section() + [""] + place_section() + [""] + exotics_section()
    (RES / "REPORT.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
