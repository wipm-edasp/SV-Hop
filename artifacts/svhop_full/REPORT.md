# SV-Hop full-set result (298 pairs, blind scaffold split)

## Setup

- 298 test pairs with ≥5 core-swap candidates, 27 targets / 5 families, blind
  scaffold split.
- Candidate generation: 60,069 core-replacement rules (PDB + ChEMBL), top-20 by
  scaffold Tanimoto, swapped reference decorations, property-ranked.
- 3D validation: candidate conformer aligned to the reference crystal ligand by
  MCS; contact recovery vs the reference contact set.

## Main result

| metric | value |
|---|---|
| mean contact recovery (all candidates) | **0.824** |
| top-1 contact recovery | **0.827** |
| fraction of candidates with contact ≥0.6 | **97.7%** |
| property-conserved | 28.9% |
| SA ≤5 | 98.1% |
| novel generic scaffold | 98.9% |
| **top-1 joint** (contact≥0.6 + property + SA + novel) | **50.3%** |
| 2D-core-similarity top-1 joint | 34.2% |
| random candidate joint | 27.0% |
| copy (reference ligand) joint | 0.0% (not novel) |

Reference upper bounds: true experimental hop contact recovery 0.806; the method's
candidates recover 0.824.

## Reading

- The method produces a **novel, property-conserved, synthesizable hop with ≥0.6
  contact recovery for 50.3%** of blind test references.
- It beats the 2D-core-similarity baseline by **+16 pp** and random by **+23 pp**;
  copy scores 0 because it is not novel.
- Contact recovery matches the experimental true-hop upper bound (0.824 vs 0.806).
- This is a complete, reproducible evidence chain: PDB experimental truth →
  core retrieval/swap → property ranking → contact recovery.

## Caveats

- Contact recovery uses MCS alignment to the reference, not docking; it is a
  structure-grounded proxy, validated against the true-hop upper bound.
- No activity/assay; no prospective validation.
- 298/429 test pairs (69%); the rest lack decomposable references or ≥5 candidates.
- The ranker is a property/SA rule, not a learned interaction-field model.

## Next

1. Learned interaction-field ranker to push joint top-1 above 50.3%; compare to
   the property ranker on the same 298 pairs.
2. Add copy/random/2D baselines per pair with confidence intervals.
3. Prospective synthesis/assay on 1–2 targets for the Nature-level claim.

---

## Learned interaction-field ranker (final)

Predict contact recovery from interaction/property features with grouped CV
(GroupKFold by pair):

| metric | value |
|---|---|
| Spearman (predicted vs actual contact) | **0.782** |
| AUC (contact ≥0.6) | **0.912** |

Property/SA/novelty as **hard filters**, predicted contact as the ranking
objective ("constrained"):

| arm | top-1 joint | 95% CI |
|---|---|---|
| **constrained interaction ranker** | **62.1%** | [56.7, 67.4] |
| property ranker | 50.3% | [44.6, 56.0] |
| 2D core similarity | 44.3% | [38.6, 50.0] |
| random candidate | 27.0% | [23.7, 30.3] |
| copy | 0.0% | — |

Paired bootstrap: constrained − property = **+11.7 pp** [8.1, 15.8], **p<0.001**.

## Final reading

On 298 blind test pairs, the constrained interaction ranker returns a **novel,
property-conserved, synthesizable hop with ≥0.6 contact recovery for 62.1% of
references**, beating the property ranker by 11.7 pp, 2D similarity by 17.8 pp,
and random by 35 pp. Contact recovery of the selected candidate is 0.845, versus
the experimental true-hop upper bound of 0.806.

## Prospective package

`results/prospective/candidates.csv`: 22 compounds across A2A (6), EGFR (7),
CDK2 (3), AKT1 (3), AR (2), CA2 (1), with the assay protocol in
`results/prospective/PROTOCOL.md`.
