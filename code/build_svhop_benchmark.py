"""Build the SV-Hop benchmark: filter PDB scaffold-hop pairs and define blind splits.

Input : data/pdb_hops/pairs.csv (from mine_pdb_hops.py)
Output: data/pdb_hops/benchmark.json

Quality filter: >=2 rings per scaffold, neither scaffold is a substructure of the
other, scaffold Morgan Tanimoto <= 0.6.  Deduplicate by (target, scaffold pair),
keeping the best-Jaccard instance.

Splits:
  target   : all pairs of a target in the same split (70/15/15 by target hash)
  scaffold : all pairs sharing a scaffold pair in the same split (70/15/15),
             so no scaffold pair leaks across train/test.
"""
from __future__ import annotations

import collections
import hashlib
import json

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors

from mine_pdb_hops import FAMILIES

RDLogger.DisableLog("rdApp.*")


def hfrac(s, salt="", mod=10000):
    return int(hashlib.md5((s + salt).encode()).hexdigest(), 16) % mod


def scaffold_ok(s1, s2):
    m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        return False
    if rdMolDescriptors.CalcNumRings(m1) < 2 or rdMolDescriptors.CalcNumRings(m2) < 2:
        return False
    if m1.HasSubstructMatch(m2) or m2.HasSubstructMatch(m1):
        return False
    f1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(m1, 2, 2048)
    f2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(m2, 2, 2048)
    return DataStructs.TanimotoSimilarity(f1, f2) <= 0.6


def main():
    df = pd.read_csv("data/pdb_hops/pairs.csv")
    print(f"[raw] {len(df)} pairs")
    keep = []
    for r in df.itertuples():
        if scaffold_ok(r.scaf1, r.scaf2):
            keep.append(r)
    df = pd.DataFrame(keep)
    print(f"[filter] {len(df)} pairs")

    # dedupe by (target, scaffold pair), keep best Jaccard
    df["sp"] = df.apply(lambda r: "|".join(sorted([r.scaf1, r.scaf2])), axis=1)
    df = df.sort_values(["target", "sp", "jaccard"], ascending=[True, True, False])
    df = df.drop_duplicates(["target", "sp"], keep="first").reset_index(drop=True)

    fam_of = {}
    for fam, targets in FAMILIES.items():
        for t in targets:
            fam_of[t] = fam
    df["family"] = df.target.map(fam_of).fillna("other")

    # target split
    tsplit = {}
    for t in sorted(df.target.unique()):
        x = hfrac(t, "t")
        tsplit[t] = "train" if x < 7000 else ("val" if x < 8500 else "test")

    # scaffold split (global by scaffold pair)
    ssplit = {}
    for sp in sorted(df.sp.unique()):
        x = hfrac(sp, "s")
        ssplit[sp] = "train" if x < 7000 else ("val" if x < 8500 else "test")

    pairs = []
    for i, r in df.iterrows():
        pairs.append({
            "id": i, "target": r.target, "family": r.family,
            "pdb1": r.pdb1, "ccd1": r.ccd1, "pdb2": r.pdb2, "ccd2": r.ccd2,
            "sp": r.sp,
            "scaf1": r.scaf1, "scaf2": r.scaf2, "jaccard": float(r.jaccard),
            "n1": int(r.n1), "n2": int(r.n2),
        })

    def split_ids(key):
        return {
            s: [p["id"] for p in pairs if key(p) == s]
            for s in ("train", "val", "test")
        }

    bench = {
        "pairs": pairs,
        "splits": {
            "target": split_ids(lambda p: tsplit[p["target"]]),
            "scaffold": split_ids(lambda p: ssplit[p["sp"]]),
        },
        "family": {p["id"]: p["family"] for p in pairs},
        "stats": {
            "pairs": len(pairs),
            "targets": int(df.target.nunique()),
            "families": int(df.family.nunique()),
            "unique_scaffold_pairs": int(df.sp.nunique()),
            "per_family": dict(collections.Counter(df.family)),
            "per_target": dict(collections.Counter(df.target)),
        },
    }
    json.dump(bench, open("data/pdb_hops/benchmark.json", "w"), indent=2)

    print(json.dumps(bench["stats"], indent=2))
    for name in ("target", "scaffold"):
        print(name, {s: len(v) for s, v in bench["splits"][name].items()})


if __name__ == "__main__":
    main()
