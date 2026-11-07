#!/usr/bin/env bash
# Full HorseCash pipeline (France + Hong Kong). Raw data must be in data/raw/{fr,hk} (see README). The neural models need a CUDA GPU.
set -euo pipefail
cd "$(dirname "$0")"
P="python -I"
$P src/fr_extract.py            # archive -> fr_runners / fr_races parquet   (~5 min)
$P src/fr_checks.py  > results/fr_checks.log
$P src/hk_checks.py  > results/hk_checks.log
$P src/fr_features.py           # pre-race features                          (~1 min)
$P src/fr_model.py dev          # boosting rounds on 2022                    (~5 min)
rm -rf data/processed/fr_oof_parts data/processed/fr_oof_v1.parquet data/processed/hk_oof_v1.parquet
$P src/fr_model.py wf           # v1 fundamental (GBM) out-of-fold 2017-2026 (~5-10 min)
# ---- model v2: ratings + neural race model, blended into the fundamental score (fr_oof.parquet; v1 kept as fr_oof_v1.parquet)
$P src/fr_exp.py ratings        > results/exp_ratings.log
$P src/fr_exp.py wf win+rt      > results/exp_wf.log        # GBM + ratings out-of-fold     (~7 min)
$P src/fr_nn.py tune nn         > results/exp_nn_tune.log   # epochs on 2022                (~3 min, GPU)
$P src/fr_nn.py wf nn           > results/exp_nn_wf.log     # v2 network out-of-fold 2017-2026 (~15 min, GPU)
$P src/fr_nn.py tune nn_L20 > results/exp_nn_L20.log && $P src/fr_nn.py wf nn_L20 0,1,2,3 > results/exp_nn_L20_wf.log          # v3 (~30 min)
$P src/fr_nn.py tune nn_L20_mus > results/exp_nn_L20_mus.log && $P src/fr_nn.py wf nn_L20_mus 0,1,2,3 > results/exp_nn_L20_mus_wf.log
cp data/processed/fr_oof.parquet data/processed/fr_oof_v1.parquet
$P src/fr_blend.py win+rt nn_L20 nn_L20_mus                 # v3 fundamental -> fr_oof.parquet
$P src/fr_eval.py               # stackers + evaluation                      (~10 min)
$P src/hk_pipeline.py feats && $P src/hk_pipeline.py dev && $P src/hk_pipeline.py wf
$P src/hk_exp.py ratings && $P src/hk_exp.py wf win+rt > results/exp_hk_wf.log
$P src/hk_nn.py tune > results/exp_hk_nn_tune.log && $P src/hk_nn.py wf > results/exp_hk_nn_wf.log
$P src/hk_exp.py blend win+rt nn
$P src/hk_pipeline.py eval
$P src/fr_exotics.py stages > results/exotics_stages.log   # 2nd/3rd-place models, walk-forward (~20 min)
$P src/fr_exotics.py eval   > results/exotics_eval.log     # couplé / trio evaluation (~10 min)
$P src/fr_place.py          > results/place.log            # simple placé
$P src/fr_value.py          > results/value.log            # win value rules (late money)
$P src/fr_value.py checks   >> results/value.log           # official dividends, own-stake effect, pool size
$P src/fr_value.py liquidity >> results/value.log          # accuracy and closing-line rule by pool size
$P src/fr_value.py liquid   >> results/value.log           # rule restricted to liquid pools
$P src/fr_value.py locked   >> results/value.log           # hypothetical: 30-min price locked (v1 needs fr_preds_v1.parquet)
$P src/hk_exotics.py        > results/hk_exotics.log       # Hong Kong quinella / tierce
$P src/hk_timing.py         > results/hk_timing.log        # Hong Kong live odds: late information
# ---- development experiments shown in the report (dev years only)
$P src/fr_exp.py wf pl3 pl3+rt && $P src/hk_exp.py wf pl3+rt
$P src/fr_exp.py stack base win+rt nn && $P src/fr_exp.py stack base win+rt nn_L20 nn_L20_mus && $P src/hk_exp.py stack base win+rt pl3+rt nn
for t in nn_nohist nn_norace nn_L20_big nn_mus nn_ref; do $P src/fr_nn.py tune $t > results/exp_$t.log; done
# ---- late prices, internet pool, late-move head, confidence and place rules (2026-11-05..07; see fresh/README.md)
$P src/fr_canalturf.py 2022-06-01 2024-12-31                # Canalturf quarter-hour internet-pool series (see README: its terms)
$P src/fr_late_ct.py 2022-06-01 2024-12-31 > results/late_ct_full.log
$P src/fr_epool.py 2017-01-01 8 > results/epool.log && $P src/fr_epool.py --refetch    # PMU internet pool from the API
$P src/fr_pools.py 2018-01-01 2024-12-31 > results/pools_dev_full.log
$P src/fr_pools.py 2025-01-01 2026-05-31 --test > results/pools_test.log
$P src/fr_nn.py tune nn_L20_mus_mt > results/exp_nn_L20_mus_mt.log && $P src/fr_nn.py wf nn_L20_mus_mt 0,1,2,3 > results/exp_nn_L20_mus_mt_wf.log
$P src/fr_value_late.py dev && $P src/fr_value_late.py test
$P src/fr_confidence.py dev                                  # pre-registers the win confidence rule
$P src/fr_place_conf.py dev                                  # pre-registers place rules A and B
# fresh window: download 2026-05-06..11-05 from PMU's API and rerun the frozen pipeline in fresh/ (see fresh/README.md), then
#   $P src/fr_confidence.py test; $P src/fr_place_conf.py test; $P src/fr_nearoff.py
$P src/fr_v3_final.py train                                  # the v3 model on all data -> models/v3
$P src/make_report.py           # -> results/REPORT.md
