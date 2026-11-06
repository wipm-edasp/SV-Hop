#!/usr/bin/env bash
# Reproduce the main scaffold-held-out SV-Hop results from this archive.
set -euo pipefail
cd "$(dirname "$0")"

# Scripts use data/ and results/ relative to the working directory.
[ -e data ] || ln -sfn dataset data
[ -e results ] || ln -sfn artifacts results

PY="${PY:-python3}"
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"

echo "== 0. PDB structures =="
if [ ! -d data/pdb_hops/pdb ] || [ -z "$(ls -A data/pdb_hops/pdb 2>/dev/null)" ]; then
  $PY code/fetch_pdb.py --benchmark data/pdb_hops/benchmark.json --out data/pdb_hops/pdb
fi

echo "== 1. Benchmark =="
$PY -u code/mine_pdb_hops.py
$PY -u code/build_svhop_benchmark.py

echo "== 2. ChEMBL panel and rules =="
if [ ! -f data/chembl/panel.jsonl ]; then
  $PY -u code/download_chembl_parallel.py
fi
$PY -u code/mine_chembl_rules.py

echo "== 3. Core-swap candidates =="
$PY -u code/core_lexicon_rank.py

echo "== 4. Crystal-pose transfer =="
$PY -u code/crystal_pose_transfer.py

echo "== 5. Contact recovery =="
LIMIT=2000 $PY -u code/candidate_contact.py

echo "== 6. Ranker and CIs =="
$PY -u code/interaction_ranker.py
$PY -u code/bootstrap_svhop.py

echo "== 7. Figures =="
$PY -u code/make_figures_svhop.py

echo "done"
