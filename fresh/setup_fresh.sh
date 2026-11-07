#!/usr/bin/env bash
# Build fresh/: a frozen copy of the pipeline on the archive + the races after it, downloaded from PMU's public API.
# Run from anywhere, after ../run_all.sh. Then: fresh/run_fresh.sh
set -euo pipefail
cd "$(dirname "$0")/.."
FROM=${1:-2026-05-06}; TO=${2:-2026-11-05}
python -I src/fr_fresh.py download "$FROM" "$TO" data/raw/fr_fresh        # resumable; ~10 min for 6 months
mkdir -p fresh/data/raw/fr fresh/results
rm -rf fresh/src && cp -r src fresh/src
for f in data/raw/fr/*; do ln -sfn "../../../../$f" "fresh/data/raw/fr/$(basename "$f")"; done
ln -sfn ../../../data/raw/pmu_internet fresh/data/raw/pmu_internet
python -I src/fr_fresh.py build data/raw/fr_fresh fresh                    # archive + fresh races, rapports-fresh.parquet
cp results/{fr_dev_rounds,exp_ratings,exp_nn_L20_tune,exp_nn_L20_mus_tune,exp_nn_L20_mus_mt_tune}.json fresh/results/
