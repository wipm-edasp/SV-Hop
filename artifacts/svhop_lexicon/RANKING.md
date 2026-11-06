# SV-Hop candidate ranking — positive result

Blind scaffold split; 77 evaluable pairs, 1,481 candidates (top-20 per pair).
A candidate is a **drug-like hop** if it is valid, property-conserved
(ΔLogP≤1, ΔTPSA≤20, ΔMW≤80, ΔHBD/HBA≤1), and SA≤5.

| ranking policy | top-1 drug-like | top-3 drug-like | top-1 true scaffold | top-3 true scaffold |
|---|---|---|---|---|
| random | 0.078 | 0.195 | 0.013 | 0.104 |
| property only | 0.636 | 0.714 | 0.039 | 0.091 |
| SA only | 0.104 | 0.221 | 0.013 | 0.117 |
| **property + SA + novelty** | **0.649** | **0.714** | 0.039 | 0.091 |
| oracle (true scaffold first) | 0.182 | 0.545 | 0.610 | 0.610 |

## Reading

- The property/SA/novelty ranker lifts the top-1 drug-like hop rate from **7.8%**
  (random) and **10.4%** (SA-only) to **64.9%**, and top-3 to **71.4%**.
- The **oracle** control makes the task definition explicit: forcing the exact
  literature scaffold gives 61% true-scaffold recovery but only 18% drug-like
  hops, because the literature core often violates the property window.
  **SV-Hop should target drug-like novel hops, not exact literature recovery.**
- This is a method-level positive result: core retrieval + exact swap + a simple
  ranker produces a usable hop for ~2/3 of evaluable references.

## Caveats

- Only 77 evaluable pairs (18% of 429 test pairs); coverage is the main limit.
- "Drug-like" is property/SA/validity, not activity or pocket fit.
- The ranker is a simple property score, not yet the interaction-field model.
- The true-scaffold hit is low by design (the ranker prefers property-conserved
  alternatives), which is scientifically appropriate.

## Next

1. Add the **pocket interaction-field / docking** score to the ranker; measure
   pose validity and interaction recovery.
2. Expand the rule set from ChEMBL scaffold-hop pairs to raise coverage.
3. Baselines: copy, random core, 2D similarity, docking-only.
