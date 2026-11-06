# Release notes

## v0.1.2 (2026-10-28)

Manuscript repositioned as a benchmark paper and prepared for *Journal of Chemical Information and Modeling*.

Added
- `code/svhop_diversity_control.py` and `artifacts/target_holdout/equal_diversity.json`:
  per-pair scaffold counting and rarefaction to equal diversity.
- SI Table S9; manuscript section 3.8.1.

Corrected
- The per-family joint top-1 in the manuscript was carrying the property-arm rates; the
  SV-Hop rates are 57.8 (kinase), 79.1 (enzyme_other), 50.0 (protease), 63.3
  (nuclear_receptor), 70.0 (gpcr); overall 62.1% (185/298).
- The lexicon claim: candidate diversity explains about one third of the PDB-only advantage
  (18.9 -> 13.1 pp), not all of it; rule composition accounts for the rest.
- Rule-count definitions are now stated explicitly (60,069 / 54,123 / 4,029).
- SI S6 CI harmonised to [60.3, 70.0]; per-family summary rows added.

## v0.1.1 (2026-10-25)

- Target-held-out evaluation, lexicon ablation (PDB-only), criterion validation.
- Prospective 22-compound panel and protocol.

## v0.1.0 (2026-10-24)

- Initial benchmark, lexicon and scaffold-held-out evaluation.
