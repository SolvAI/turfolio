#!/usr/bin/env bash
# Frozen v3 pipeline on archive + fresh races (see README.md). Tuned rounds/epochs come from the copied results/*.json; the steps
# that re-tune (GBM variant rounds, ratings) do so on 2015-2022 only, which the fresh months cannot change.
set -euo pipefail
cd "$(dirname "$0")"
P="python -I"
export NTHREADS=12 OMP_NUM_THREADS=12
$P src/fr_features.py                                    > results/features.log
$P src/fr_model.py wf                                    > results/model_wf.log      # v1 GBM out-of-fold 2017-2026
$P src/fr_exp.py ratings                                 > results/exp_ratings.log
$P src/fr_exp.py wf win+rt                               > results/exp_wf.log
$P src/fr_nn.py wf nn_L20 0,1,2,3                        > results/exp_nn_L20_wf.log
$P src/fr_nn.py wf nn_L20_mus 0,1,2,3                    > results/exp_nn_L20_mus_wf.log
cp data/processed/fr_oof.parquet data/processed/fr_oof_v1.parquet
$P src/fr_blend.py win+rt nn_L20 nn_L20_mus              > results/blend.log
$P src/fr_eval.py                                        > results/fr_eval.log
$P src/fr_value.py                                       > results/value.log
$P src/fr_epool.py 2026-05-06 8                          > results/epool_fresh.log   # internet pool for the fresh races
$P src/fr_fresh.py evaluate "$(pwd)" 2026-05-06 2026-12-31 > results/fresh_eval.log
$P src/fr_pools.py 2026-05-06 2026-12-31 --test           > results/pools_fresh.log
# added 2026-11-06..07 (run in this order, after the steps above)
$P src/fr_nn.py wf nn_L20_mus_mt 0,1,2,3                 > results/exp_nn_L20_mus_mt_wf.log  # late-move head (pre-registered test)
$P src/fr_value_late.py test                             > results/value_late_test.log
$P src/fr_exotics.py stages                              > results/exotics_stages.log
$P src/fr_place.py                                       > results/place.log
$P src/fr_v3_final.py train --through 2026-05-05 --out models/v3_through_2026-05-05 > results/v3_final_check_train.log   # check of the
$P src/fr_v3_final.py predict 2026-05-06 2026-11-05 --model models/v3_through_2026-05-05 > results/v3_final_check_predict.log  # final code
$P src/fr_v3_final.py train --out models/v3              > results/v3_final_train.log        # the shipped model (all races to 11-05)
echo "fresh pipeline done"
