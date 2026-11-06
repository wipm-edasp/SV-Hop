"""Candidate contact recovery without docking.

For top property-ranked core-swap candidates, generate a conformer, align it to
the reference crystal ligand by MCS, and compute contact recovery in the
reference pocket.  Compare to the true-hop upper bound (~0.806).
"""
from __future__ import annotations

import json
import os

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS

from dock_copy_check2 import contacts_from_atoms, crystal_ligand, select_copy

RDLogger.DisableLog("rdApp.*")


def main():
    bench = json.load(open("data/pdb_hops/benchmark.json"))
    lig = pd.read_csv("data/pdb_hops/ligands.csv")
    smi = {(r.pdb, r.ccd): r.smiles for r in lig.itertuples()}
    df = pd.read_csv("results/svhop_lexicon/candidates.csv")
    df = df[df.smiles.notna()].copy()
    df["prop"] = -(df.dlogp.abs() + df.dtpsa.abs() / 20 + df.dmw.abs() / 80
                   + df.dhbd.abs() + df.dhba.abs()) - 0.1 * df.sa.fillna(10)
    counts = df.groupby("pair").size()
    test_ids = set(bench["splits"]["scaffold"]["test"])
    pairs = [p for p in bench["pairs"] if p["id"] in test_ids and counts.get(p["id"], 0) >= 5]
    os.makedirs("results/svhop_contact", exist_ok=True)
    rows = []
    LIMIT = int(os.environ.get("LIMIT", "60"))
    for p in pairs[:LIMIT]:
        pdb = f"data/pdb_hops/pdb/{p['pdb1']}.pdb"
        ref_smi = smi.get((p["pdb1"], p["ccd1"]))
        if not ref_smi:
            continue
        sel = select_copy(pdb, p["ccd1"])
        if sel is None:
            continue
        ref = crystal_ligand(pdb, p["ccd1"], sel["chain"], sel["seqid"], ref_smi)
        if ref is None:
            continue
        g = df[df.pair == p["id"]].nlargest(5, "prop")
        for rank, r in enumerate(g.itertuples()):
            cand = Chem.MolFromSmiles(r.smiles)
            if cand is None:
                continue
            cand = Chem.AddHs(cand)
            if AllChem.EmbedMolecule(cand, randomSeed=0) != 0:
                continue
            try:
                AllChem.MMFFOptimizeMolecule(cand)
                cand = Chem.RemoveHs(cand)
                mcs = rdFMCS.FindMCS([ref, cand], timeout=5,
                                     atomCompare=rdFMCS.AtomCompare.CompareAny,
                                     bondCompare=rdFMCS.BondCompare.CompareAny)
                if mcs.numAtoms < 3:
                    continue
                patt = Chem.MolFromSmarts(mcs.smartsString)
                m1 = ref.GetSubstructMatch(patt)
                m2 = cand.GetSubstructMatch(patt)
                if not m1 or not m2:
                    continue
                AllChem.AlignMol(cand, ref, atomMap=list(zip(m2, m1)))
            except Exception:
                continue
            conf = cand.GetConformer()
            atoms = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
                     for i in range(cand.GetNumAtoms()) if cand.GetAtomWithIdx(i).GetAtomicNum() != 1]
            cs = contacts_from_atoms(pdb, atoms)
            recall = len(cs & sel["contacts"]) / max(1, len(sel["contacts"]))
            rows.append({"target": p["target"], "pair": p["id"], "rank": rank,
                         "contact_recall": recall, "mcs_atoms": mcs.numAtoms,
                         "n_contacts": len(cs)})
        print(f"[{p['target']}] {p['id']} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/svhop_contact/per_candidate.csv", index=False)
    summary = {
        "n_pairs": int(out.pair.nunique()) if len(out) else 0,
        "n_candidates": len(out),
        "mean_recall": float(out.contact_recall.mean()) if len(out) else None,
        "top1_recall": float(out[out["rank"] == 0].contact_recall.mean()) if len(out) else None,
        "frac_recall_ge_06": float((out.contact_recall >= 0.6).mean()) if len(out) else None,
        "mean_mcs_atoms": float(out.mcs_atoms.mean()) if len(out) else None,
    }
    json.dump(summary, open("results/svhop_contact/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


