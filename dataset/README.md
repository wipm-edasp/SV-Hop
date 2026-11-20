# Dataset

## Files

| file | content |
|---|---|
| `pdb_hops/benchmark.json` | 2,886 same-pocket, different-scaffold pairs (27 targets, five families) with target/scaffold splits, plus per-pair `ediam1/ediam2/edia_min/edia_pass` |
| `pdb_hops/ligands.csv` | 915 PDB ligands (SMILES, Murcko scaffold, heavy atoms, contact residues) |
| `pdb_hops/pairs_filtered.csv` | legacy v0.1.1 intermediate (1,861 pairs); superseded by `benchmark.json` and read by no script — kept for provenance only |
| `pdb_hops/summary.json` | mining statistics |
| `edia/edia_ligands.csv` | EDIA_m per (PDB, CCD): 528 ligand instances, best copy, copy count, service status |
| `edia/edia_status.csv` | 522 entries: 503 scored, 19 without deposited structure factors |
| `edia/edia_pairs.csv` | per-pair EDIA scores, pass flag and splits (2,886 rows) |
| `edia/edia_failures.csv` | 164 ligands below EDIA_m 0.8 or unscorable, with the pairs they affect |
| `edia/edia_summary.json` | audit counts and the EDIA-restricted sensitivity analysis |
| `chembl/panel.jsonl` | 28,754 ChEMBL 37 measurements, 20 targets (pChEMBL, assay type, year) |
| `chembl_core_rules.jsonl` | 59,735 deduplicated `(S1, S2, k)` core-replacement rules |
| `top_rules.json` | compact rule table with support >= 3 (4,029 rules) |

## PDB coordinates

Raw `.pdb` files are not bundled (469 MB). Fetch the 763 entries with:

```bash
python code/fetch_pdb.py \
  --benchmark dataset/pdb_hops/benchmark.json \
  --out dataset/pdb_hops/pdb
```

The downloader reads the unique `pdb1`/`pdb2` fields and skips files already
present. All structures are X-ray entries at ≤2.5 Å used by the benchmark.

## Benchmark format

`benchmark.json`:

- `pairs`: list of `{id, target, family, pdb1, ccd1, pdb2, ccd2, sp, scaf1, scaf2,
  jaccard, n1, n2, ediam1, ediam2, edia_min, edia_pass}`; each pair is a
  same-pocket, different-scaffold ligand pair. `ediam*` are null when the ligand
  could not be scored.
- `splits.scaffold`: `{train, val, test}` pair IDs, scaffold-disjoint.
- `splits.target`: `{train, val, test}` pair IDs, target-disjoint.
- `edia`: gate threshold, whether the gate was applied to this file, and the
  number of pairs scored / passing.
- `stats`: per-family and per-target counts.

The shipped file is the set analysed in the manuscript: 2,886 pairs, ids and
splits unchanged, assembled before the EDIA gate existed (`"edia": {"applied":
false}`). The gate is on by default in `code/build_svhop_benchmark.py`; run it
with `--no-edia` to rebuild this file, or with `--edia 0.8 --out
data/pdb_hops/benchmark_edia08.json` for the gated set (1,587 pairs).

## EDIA tables

`edia_ligands.csv` holds the molecule-level score and the number of copies of the
ligand in the entry; the best-supported copy is used. `status` is `ok` when the
entry had a structure-score table and `no_density` when the service reported no
electron density file. Ligands can also be unscorable inside a scored entry when
the service does not report them as separate molecules; those rows have an empty
`ediam`. Both cases fail the gate. Scores were computed on 17 November 2026 with
the ProteinsPlus EDIA REST service (<https://proteins.plus/api/edia_rest>);
`code/fetch_edia.py` reproduces them.

## Provenance and licenses

- **RCSB PDB**: coordinates and chemical components; subject to RCSB usage terms.
- **UniProt**: REST API mappings.
- **ChEMBL 37**: panel and rules; CC BY-SA 3.0 (EMBL-EBI).
- Benchmark construction, ligand table, EDIA tables, derived rules and result
  tables: CC BY 4.0.

If you redistribute `dataset/chembl/` or `chembl_core_rules.jsonl`, keep the
ChEMBL CC BY-SA 3.0 attribution.
