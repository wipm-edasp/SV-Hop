# SV-Hop core retrieval + swap — positive pilot

## Setup

- 334 core-replacement rules mined from the training split (`S1 -> S2`, equal
  attachment count `k`).
- For each test reference `S1`, rank training rules by scaffold Morgan Tanimoto
  to `S1` (same `k`), take the top-20 unique candidate cores.
- Swap the reference decorations onto each candidate core; check validity,
  scaffold match, property conservation, SA.

## Results (77 evaluable test pairs; 18% of 429 test pairs)

| metric | value |
|---|---|
| **exact core recall@20** | **58.4%** |
| mean best retrieved-core Tanimoto to true core | 0.710 |
| mean valid swaps per pair (of 20) | 18.6 (93%) |
| mean SA≤5 swaps per pair | 17.6 (88%) |
| **true-scaffold generated (≥1 per pair)** | **57.1%** |
| mean property-conserved swaps per pair | 3.0 (15%) |

Per attachment count: k=1:4, k=2:31, k=3:24, k=4:12, k=5:5, k=6:1.

## Reading

- **Core retrieval works**: scaffold similarity over a cross-target rule set puts
  the exact experimental core in the top-20 for 58% of evaluable pairs.
- **The swap works**: 93% of retrieved cores accept the reference decorations and
  reproduce the target Murcko scaffold.
- **Property conservation is the bottleneck**: only ~3 of 20 swaps stay within the
  bioisosteric property window. The method must **rank/select** those 3, and add
  interaction-field and novelty objectives.
- Coverage is 18% because rules require equal attachment counts and successful
  decomposition; expanding the rule set (ChEMBL scaffold-hop pairs) is the next
  data step.

## Next

1. Save candidate-level records and rank by property distance + SA + novelty;
   measure top-1/top-3 drug-like hops.
2. Add the pocket interaction-field score; compare to copy/random/2D/docking.
3. Expand rules from ChEMBL scaffold-hop pairs to raise coverage.
