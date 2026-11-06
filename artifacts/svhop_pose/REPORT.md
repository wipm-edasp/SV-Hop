# Crystal-pose transfer — 3D benchmark validation passes

## Method

Superpose pdb2 onto pdb1 using shared Cα atoms (Kabsch); transfer the true hop's
crystal coordinates into the pdb1 frame; compute its contact residues in pdb1 and
compare with the reference ligand's contact set.

No docking, no receptor prep, no box.

## Results (40 test pairs)

| metric | value |
|---|---|
| mean Cα RMSD after superposition | **0.55 Å** |
| mean true-hop contact recall | **0.806** |
| mean contact Jaccard | 0.700 |
| fraction of pairs with recall ≥ 0.6 | **97%** |

One outlier: CA2 6T81/2FOU recall 0.00 (different binding mode/ligand copy).

## Reading

- The PDB scaffold-hop pairs are **structurally coherent**: after superposition,
  the true experimental hop recovers ~80% of the reference contacts in 97% of
  pairs. This is a 3D ground truth that does not depend on docking.
- It also establishes an **upper bound** for candidate scoring: a method that
  proposes a different hop need not match the exact literature ligand, but its
  pose should recover a comparable fraction of reference contacts.
- This validates the **benchmark**, not yet the candidate generator/ranker.

## Next

1. **Candidate contact recovery without docking**: generate conformers, align to
   the reference ligand by shape/pharmacophore, compute contact recall in pdb1;
   compare to the true-hop upper bound (0.806) and to copy.
2. **Interaction-field ranker**: train a 3D pocket+ligand model to predict contact
   recovery on the blind scaffold split; use it to rerank the property-selected
   candidates.
3. Report the joint metric: property-conserved + contact-recovery ≥ threshold +
   scaffold novelty.
