# SV-Hop prospective assay package

## Candidates

22 compounds in `results/prospective/candidates.csv`:

| target | n | contact recovery range |
|---|---|---|
| A2A | 6 | 0.73–0.89 |
| EGFR | 7 | 0.82–0.93 |
| CDK2 | 3 | — |
| AKT1 | 3 | — |
| AR | 2 | — |
| CA2 | 1 | — |

Selection: constrained interaction ranker (property + SA + novelty as hard filters,
predicted contact as ranking objective), top-12 per target, deduplicated by
generic scaffold.

## Assays

| target | primary assay | counter-screen |
|---|---|---|
| A2A | cAMP / radioligand binding (HEK293-A2A) | A1, A2B |
| EGFR | ADP-Glo kinase assay (wild-type + L858R/T790M/C797S) | EGFR family selectivity |
| CDK2/AKT1/AR/CA2 | target-specific biochemical assay | family panel |

- 8-point dose-response, n=3, reference compound on every plate.
- Report IC50/Ki with 95% CI; ≥70% inhibition at top concentration defines a hit.

## Controls (assayed in parallel, same plates)

1. **Reference** `ccd1` for each pair (positive control; known active).
2. **Copy** — reference ligand itself on its original core (the trivial baseline).
3. **2D-selected** candidate — highest core Tanimoto, property-filtered.
4. **Random** core-swap candidate — random rule, property-filtered.
5. Vehicle.

## Go / No-Go

- Success: **≥3 novel-scaffold compounds with IC50 < 10 µM** and a hit rate above
  the copy and 2D arms.
- Strong: **≥1 series with IC50 < 1 µM**.
- Failure: no novel-scaffold hit, or the copy/2D arms match or beat the method.
  In that case the claim is limited to the retrospective/structural result.

## Synthesis

- Commercial building blocks (Enamine/WuXi); target ≤3 steps.
- SA ≤ 3.7 for all 22; confirm with AiZynthFinder before ordering.
- Order 2–5 mg per compound; purity >95% (LC-MS).

## Provenance and authorship

- All raw assay data shared; negative results publishable.
- Computational + medicinal chemistry + assay authorship agreed before synthesis.
- Structures (PDB IDs) and reference SMILES included in the candidate table.
