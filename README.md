# SV-Hop v0.1.2

Benchmark, lexicon, model artifacts and code for **"Scaffold hopping from deposited crystal
pairs: a docking-free contact-conservation benchmark and a structure-validated method"**.

## Contents

| Path | Contents |
|---|---|
| `code/` | Pipeline scripts (PDB mining -> benchmark -> lexicon -> candidates -> evaluation) |
| `dataset/` | Benchmark (2,886 pairs, splits), ligand tables, ChEMBL panel, core rules |
| `artifacts/` | Result tables and JSON summaries for every number in the manuscript |
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

## Rule-count definitions

- 60,069 rules: full lexicon (59,735 ChEMBL + 334 PDB), used in the scaffold-held-out run.
- 54,123 rules: same set minus ChEMBL rules observed on the six held-out targets, used in the
  target-held-out run.
- 4,029 rules: compact rule table with support >= 3 (released as a resource).

## Reproduce

```bash
pip install -r code/requirements.txt
bash reproduce.sh          # benchmark, lexicon and main evaluation
bash reproduce_extra.sh    # ablations, target-held-out, diversity control
```

All manuscript numbers are read from `artifacts/`; no value is typed by hand. Figures and the SI markdown are written to `out/`, which the scripts create.

## License and citation

MIT (see `LICENSE`). Citation metadata in `CITATION.cff`.

## Contact

Correspondence about the benchmark, the lexicon, or the code: Xiongjie Xiao (xiaoxj007@outlook.com).
