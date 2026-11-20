"""Build the SV-Hop benchmark: filter PDB scaffold-hop pairs and define blind splits.

Input : data/pdb_hops/pairs.csv (from mine_pdb_hops.py)
        data/edia/edia_ligands.csv (per (PDB, CCD) EDIA_m; from fetch_edia.py)
Output: data/pdb_hops/benchmark.json

Quality filter: >=2 rings per scaffold, neither scaffold is a substructure of the
other, scaffold Morgan Tanimoto <= 0.6.  Deduplicate by (target, scaffold pair),
keeping the best-Jaccard instance.

Ligand electron-density gate: a pair is admitted only when both ligands have
EDIA_m >= threshold (default 0.8; Methods 2.1.4).  The gate is ON by default, so
new constructions do not inherit ligands that the electron density does not
support.  `--no-edia` rebuilds the set analysed in the manuscript (2,886 pairs),
which was assembled before the gate existed and is audited pair by pair in
Tables S11 and S12.  Every pair carries its two EDIA scores either way.

Splits:
  target   : all pairs of a target in the same split (70/15/15 by target hash)
  scaffold : all pairs sharing a scaffold pair in the same split (70/15/15),
             so no scaffold pair leaks across train/test.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors

from mine_pdb_hops import FAMILIES

RDLogger.DisableLog("rdApp.*")

DEFAULT_PAIRS = "data/pdb_hops/pairs.csv"
DEFAULT_OUT = "data/pdb_hops/benchmark.json"
DEFAULT_EDIA = "data/edia/edia_ligands.csv"
DEFAULT_THRESHOLD = 0.8


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


def load_edia(path):
    """(PDB, CCD) -> (EDIA_m or None, status)."""
    table = {}
    if not os.path.exists(path):
        print(f"[edia] no table at {path}; every pair will be left unscored")
        return table
    for r in csv.DictReader(open(path, encoding="utf-8")):
        key = (r["pdb"].upper(), r["ccd"])
        table[key] = (float(r["ediam"]) if r["ediam"] else None, r.get("status", ""))
    return table


def annotate_edia(df, table):
    """Attach ediam1/ediam2/edia_min to every pair.

    A pair passes the gate only when both ligands are scored and both are
    >= threshold.  Unscored ligands (no deposited structure factors, or a ligand
    the service does not report as a separate molecule) fail the gate instead of
    being assumed acceptable.
    """
    e1, e2 = [], []
    for r in df.itertuples():
        e1.append(table.get((str(r.pdb1).upper(), r.ccd1), (None, "missing"))[0])
        e2.append(table.get((str(r.pdb2).upper(), r.ccd2), (None, "missing"))[0])
    df = df.copy()
    df["ediam1"] = e1
    df["ediam2"] = e2
    df["edia_min"] = [None if a is None or b is None else min(a, b) for a, b in zip(e1, e2)]
    return df


def main():
    ap = argparse.ArgumentParser(description="Build the SV-Hop benchmark.")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS, help="mined pairs (pairs.csv)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="benchmark JSON to write")
    ap.add_argument("--edia", type=float, default=DEFAULT_THRESHOLD,
                    help="EDIA_m threshold applied to both ligands (default 0.8)")
    ap.add_argument("--no-edia", action="store_true",
                    help="disable the gate and rebuild the analysed (pre-gate) set")
    ap.add_argument("--edia-table", default=DEFAULT_EDIA, help="per-ligand EDIA table")
    args = ap.parse_args()

    df = pd.read_csv(args.pairs)
    print(f"[raw] {len(df)} pairs")
    keep = [r for r in df.itertuples() if scaffold_ok(r.scaf1, r.scaf2)]
    df = pd.DataFrame(keep)
    print(f"[filter] {len(df)} pairs")

    # dedupe by (target, scaffold pair), keep best Jaccard
    df["sp"] = df.apply(lambda r: "|".join(sorted([r.scaf1, r.scaf2])), axis=1)
    df = df.sort_values(["target", "sp", "jaccard"], ascending=[True, True, False])
    df = df.drop_duplicates(["target", "sp"], keep="first").reset_index(drop=True)

    # dedupe first, gate second: each pair keeps the instance that survived the
    # scaffold filters, so --no-edia rebuilds exactly the analysed 2,886 pairs
    table = load_edia(args.edia_table)
    if not args.no_edia and not table:
        raise SystemExit("[edia] the gate is on but no EDIA table was found at "
                         f"{args.edia_table}; run code/fetch_edia.py first, or pass --no-edia "
                         "to rebuild the analysed pre-gate set")
    df = annotate_edia(df, table)
    n_all = len(df)
    n_scored = int(df["edia_min"].notna().sum())
    passes = [m is not None and not pd.isna(m) and float(m) >= args.edia
              for m in df["edia_min"]]
    n_pass = int(sum(passes))
    applied = not args.no_edia
    if applied:
        df = df.loc[[bool(x) for x in passes]].reset_index(drop=True)
        print(f"[edia] gate {args.edia}: kept {n_pass} of {n_all} pairs "
              f"({n_scored} scored for both ligands)")
        if os.path.normpath(args.out) == os.path.normpath(DEFAULT_OUT):
            print("[edia] NOTE: this overwrites the shipped 2,886-pair benchmark. "
                  "Use --no-edia to rebuild the analysed set, or --out to write the gated set elsewhere")
    else:
        print(f"[edia] gate disabled: {n_all} pairs, {n_pass} would pass {args.edia}")

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
            "ediam1": None if pd.isna(r.ediam1) else float(r.ediam1),
            "ediam2": None if pd.isna(r.ediam2) else float(r.ediam2),
            "edia_min": None if pd.isna(r.edia_min) else float(r.edia_min),
            "edia_pass": int((not pd.isna(r.edia_min)) and r.edia_min >= args.edia),
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
        "edia": {
            "threshold": args.edia,
            "applied": applied,
            "scored_both": n_scored,
            "pairs_before_gate": n_all,
            "pairs_passing_gate": n_pass,
            "pairs_after_gate": len(pairs),
        },
        "stats": {
            "pairs": len(pairs),
            "targets": int(df.target.nunique()),
            "families": int(df.family.nunique()),
            "unique_scaffold_pairs": int(df.sp.nunique()),
            "per_family": dict(collections.Counter(df.family)),
            "per_target": dict(collections.Counter(df.target)),
        },
    }
    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)
    json.dump(bench, open(args.out, "w", encoding="utf-8"), indent=2)

    print(json.dumps(bench["stats"], indent=2))
    for name in ("target", "scaffold"):
        print(name, {s: len(v) for s, v in bench["splits"][name].items()})


if __name__ == "__main__":
    main()
