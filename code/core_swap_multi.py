"""Multi-attachment core swap.

For PDB pairs with equal numbers of attachment points, transplant all reference
decorations onto the true core and check validity, scaffold match, property
conservation, and SA.
"""
from __future__ import annotations

import json
import os

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from chem import murcko_smiles, props, sa_score

RDLogger.DisableLog("rdApp.*")


def decompose_multi(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    core0 = MurckoScaffold.GetScaffoldForMol(m)
    if core0 is None or core0.GetNumAtoms() == 0:
        return None
    core = Chem.RenumberAtoms(core0, [a.GetIdx() for a in core0.GetAtoms()])
    match = m.GetSubstructMatch(core)
    if not match or len(match) != core.GetNumAtoms():
        return None
    cs = set(match)
    att = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (i in cs) != (j in cs):
            att.append(i if i in cs else j)
    att = list(dict.fromkeys(att))
    if not att:
        return None
    dec = Chem.ReplaceCore(m, core)
    if dec is None:
        return None
    frags = Chem.GetMolFrags(dec, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    dec_all = frags[0]
    for f in frags[1:]:
        dec_all = Chem.CombineMols(dec_all, f)
    rw = Chem.RWMol(dec_all)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetIsotope(0)
    dec_d = rw.GetMol()
    try:
        Chem.SanitizeMol(dec_d)
    except Exception:
        return None
    rw = Chem.RWMol(core)
    for c in att:
        pos = list(match).index(c)
        d = rw.AddAtom(Chem.Atom(0))
        rw.GetAtomWithIdx(d).SetIsotope(0)
        rw.AddBond(pos, d, Chem.BondType.SINGLE)
    core_d = rw.GetMol()
    try:
        Chem.SanitizeMol(core_d)
    except Exception:
        return None
    return core_d, dec_d, len(att), Chem.MolToSmiles(core)


def join_multi(core_d, dec_d):
    d1 = [a.GetIdx() for a in core_d.GetAtoms() if a.GetAtomicNum() == 0]
    d2 = [a.GetIdx() for a in dec_d.GetAtoms() if a.GetAtomicNum() == 0]
    if len(d1) != len(d2) or not d1:
        return None
    combo = Chem.CombineMols(core_d, dec_d)
    rw = Chem.RWMol(combo)
    off = core_d.GetNumAtoms()
    d2 = [x + off for x in d2]
    pairs = list(zip(d1, d2))
    info = []
    for i1, i2 in pairs:
        n1 = rw.GetAtomWithIdx(i1).GetNeighbors()[0].GetIdx()
        n2 = rw.GetAtomWithIdx(i2).GetNeighbors()[0].GetIdx()
        info.append((i1, i2, n1, n2))
    rem = sorted([x for p in info for x in p[:2]], reverse=True)
    def shift(x):
        return x - sum(1 for r in rem if r < x)
    info2 = [(shift(n1), shift(n2)) for _, _, n1, n2 in info]
    for r in rem:
        rw.RemoveAtom(r)
    for n1, n2 in info2:
        rw.AddBond(n1, n2, Chem.BondType.SINGLE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def main():
    bench = json.load(open("data/pdb_hops/benchmark.json"))
    lig = pd.read_csv("data/pdb_hops/ligands.csv")
    smi = {(r.pdb, r.ccd): r.smiles for r in lig.itertuples()}
    rows = []
    for p in bench["pairs"]:
        ref = smi.get((p["pdb1"], p["ccd1"]))
        true = smi.get((p["pdb2"], p["ccd2"]))
        if not ref or not true:
            continue
        d1 = decompose_multi(ref)
        d2 = decompose_multi(true)
        if d1 is None or d2 is None:
            continue
        if d1[2] != d2[2] or d1[2] < 2:
            continue
        prod = join_multi(d2[0], d1[1])
        if prod is None:
            rows.append({"target": p["target"], "family": p["family"], "k": d1[2],
                         "valid": 0, "scaffold_match": 0, "sa": None,
                         "dlogp": None, "dtpsa": None, "dmw": None})
            continue
        sm = Chem.MolToSmiles(prod)
        pr, ps = props(ref), props(sm)
        rows.append({
            "target": p["target"], "family": p["family"], "k": d1[2],
            "valid": 1,
            "scaffold_match": int(murcko_smiles(prod) == d2[3]),
            "sa": sa_score(sm),
            "dlogp": (ps["logp"] - pr["logp"]) if ps and pr else None,
            "dtpsa": (ps["tpsa"] - pr["tpsa"]) if ps and pr else None,
            "dmw": (ps["mw"] - pr["mw"]) if ps and pr else None,
            "swapped": sm,
        })
    df = pd.DataFrame(rows)
    os.makedirs("results/svhop_coreswap", exist_ok=True)
    df.to_csv("results/svhop_coreswap/multi_swaps.csv", index=False)
    n = len(df)
    summary = {
        "pairs_attempted": n,
        "valid": float(df.valid.mean()) if n else None,
        "scaffold_match": float(df.scaffold_match.mean()) if n else None,
        "mean_sa": float(df.sa.mean()) if n else None,
        "frac_sa_le_5": float((df.sa <= 5).mean()) if n else None,
        "frac_property_conserved": float(((df.dlogp.abs() <= 1)
                                          & (df.dtpsa.abs() <= 20)
                                          & (df.dmw.abs() <= 80)).mean()) if n else None,
        "per_k": df.groupby("k").size().to_dict(),
        "per_target": df.groupby("target").size().to_dict(),
    }
    json.dump(summary, open("results/svhop_coreswap/multi_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    ok = df[(df.valid == 1)]
    print("\nexamples:")
    for r in ok.head(5).itertuples():
        print(f"  {r.target} k={r.k} {r.swapped[:100]}")


if __name__ == "__main__":
    main()



