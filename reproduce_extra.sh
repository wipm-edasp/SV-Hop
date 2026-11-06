#!/usr/bin/env bash
# Reproduce the target-held-out, ablation, C1 and SI results.
set -euo pipefail
cd "$(dirname "$0")"

[ -e data ] || ln -sfn dataset data
[ -e results ] || ln -sfn artifacts results

PY="${PY:-python3}"
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"

echo "== 1. Target-held-out (full lexicon) =="
$PY -u code/target_holdout.py
$PY -u code/target_holdout_ablation.py
$PY -u code/make_fig_target_holdout.py

echo "== 2. PDB-only lexicon ablation =="
USE_CHEMBL=0 OUT_DIR=results/target_holdout_pdbonly $PY -u code/target_holdout.py
OUT_DIR=results/target_holdout_pdbonly $PY -u code/target_holdout_ablation.py

echo "== 3. Criterion validation =="
$PY -u code/c1_validation.py

echo "== 4. Supporting Information =="
$PY -u code/make_si.py

echo "done"
