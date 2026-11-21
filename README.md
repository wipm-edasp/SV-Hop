# SV-Hop v0.1.3

Benchmark, lexicon, model artifacts, EDIA ligand-quality audit and code for
**"Scaffold hopping from deposited crystal pairs: a docking-free contact-conservation
benchmark and a structure-validated method"**.

## Contents

| Path | Contents |
|---|---|
| `code/` | Pipeline scripts (PDB mining -> benchmark -> EDIA gate -> lexicon -> candidates -> evaluation) |
| `dataset/` | Benchmark (2,886 pairs, splits, per-pair EDIA scores), ligand tables, EDIA tables, ChEMBL panel, core rules |
| `artifacts/` | Result tables and JSON summaries for every number in the manuscript |
| `dataset/edia/` | EDIA_m per ligand, per-entry service status, per-pair scores, failure list, audit summary |
| `checksums/SHA256SUMS` | SHA-256 of every file |

## Key results

- Contact-conservation criterion: true hop recovers 80.6% of reference contacts; 97% of
  pairs above 0.6. It tracks pharmacophore overlap (rho = 0.51, n = 50) but not the potency
  gap (rho = -0.13).
- Scaffold-held-out: SV-Hop 62.1% joint top-1 (185/298) vs property 50.3%, 2D 44.3%,
  random 27.0%, copy 0%.
- Target-held-out: 65.3% vs 60.0% (+5.3 pp); ranker transfer rho = 0.13.
- Lexicon ablation: a 388-rule PDB-only set reaches 81.9% vs 65.3%; the equal-diversity
  control reduces the gap from 18.9 to 13.1 pp, so candidate diversity explains part, not
  all, of the difference.
- Ligand electron density (EDIA_m; 528 instances, 522 entries): 364 (68.9%) well supported
  (>= 0.8), 74 moderate, 12 poor, 78 not scorable (19 in 19 entries without structure
  factors, 59 not reported as a molecule). At pair level 1,587 of 2,886 (55.0%; 66.1% of the
  2,402 pairs with both ligands scored) pass the strict gate. On the 186 of 298 scaffold-test
  pairs that pass, joint top-1 is 60.2% vs 62.1% on all 298.

## The EDIA gate

`code/build_svhop_benchmark.py` admits a pair only when both ligands have
EDIA_m >= 0.8 (`--edia`, default 0.8). The gate is **on by default** for new
constructions. The benchmark analysed in the manuscript was assembled before the
gate existed, so it is shipped with ids and splits intact (2,886 pairs), now
annotated with `ediam1/ediam2/edia_min/edia_pass`, and can be rebuilt with
`--no-edia`. The gated benchmark is written separately:

```bash
python code/build_svhop_benchmark.py --no-edia                              # 2,886 pairs
python code/build_svhop_benchmark.py --edia 0.8 --out data/pdb_hops/benchmark_edia08.json
python code/analyze_edia.py                                                 # audit + sensitivity
```

Scores come from the ProteinsPlus EDIA REST service
(<https://proteins.plus/api/edia_rest>), computed on 17 November 2026; `code/fetch_edia.py`
re-fetches them if the cache is missing. Unscored ligands (no deposited structure
factors, or a ligand the service does not report as a separate molecule) fail the
gate rather than being assumed acceptable.

## Rule-count definitions

- 60,069 rules: full lexicon (59,735 ChEMBL + 334 PDB), used in the scaffold-held-out run.
- 54,123 rules: same set minus ChEMBL rules observed on the six held-out targets, used in the
  target-held-out run.
- 4,029 rules: compact subset with support >= 3, released as `dataset/top_rules.json`.

## Reproduce

```bash
pip install -r code/requirements.txt
bash reproduce.sh          # structures, EDIA table, benchmark, lexicon, main evaluation, audit
bash reproduce_extra.sh    # ablations, target-held-out, diversity control
```

All manuscript numbers are read from `artifacts/`; no value is typed by hand. Figures and the
SI markdown are written to `out/`, which the scripts create.

## License and citation

MIT (see `LICENSE`). Citation metadata in `CITATION.cff`. The release is distributed
through this repository (tag `sv-hop-0.1.3`); no separate archive DOI has been assigned,
so cite the repository URL together with the tag and the checksums in
`checksums/SHA256SUMS`.

## Contact

Correspondence about the benchmark, the lexicon, or the code: Xiongjie Xiao (xiaoxj007@outlook.com).
