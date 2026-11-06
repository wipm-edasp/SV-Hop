# Dataset

## Files

| file | content |
|---|---|
| `pdb_hops/benchmark.json` | 2,886 same-pocket, different-scaffold pairs (27 targets, five families) with target/scaffold splits |
| `pdb_hops/ligands.csv` | 915 PDB ligands (SMILES, Murcko scaffold, heavy atoms, contact residues) |
| `pdb_hops/pairs_filtered.csv` | mined candidate pairs after filtering (benchmark source pairs) |
| `pdb_hops/summary.json` | mining statistics |
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

- `pairs`: list of `{id, target, family, pdb1, ccd1, pdb2, ccd2, scaffold1,
  scaffold2, ...}`; each pair is a same-pocket, different-scaffold ligand pair.
- `splits.scaffold`: `{train, val, test}` pair IDs, scaffold-disjoint.
- `splits.target`: `{train, val, test}` pair IDs, target-disjoint.
- `stats`: per-family and per-target counts.

## Provenance and licenses

- **RCSB PDB**: coordinates and chemical components; subject to RCSB usage terms.
- **UniProt**: REST API mappings.
- **ChEMBL 37**: panel and rules; CC BY-SA 3.0 (EMBL-EBI).
- Benchmark construction, ligand table, derived rules and result tables:
  CC BY 4.0.

If you redistribute `dataset/chembl/` or `chembl_core_rules.jsonl`, keep the
ChEMBL CC BY-SA 3.0 attribution.
