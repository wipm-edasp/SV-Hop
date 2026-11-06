"""Crystal-pose transfer for 3D validation.

Superpose pdb2 onto pdb1 using shared C-alpha atoms, transfer the true hop's
crystal coordinates into the pdb1 frame, and measure interaction recovery against
the reference ligand's contact set.

This gives an upper bound for a real experimental hop without docking.
"""
from __future__ import annotations

import json
import os

import gemmi
import numpy as np
import pandas as pd

from dock_copy_check2 import contacts_from_atoms, select_copy


def ca_dict(st, chain_name=None):
    d = {}
    for ch in st[0]:
        if chain_name and ch.name != chain_name:
            continue
        for res in ch:
            if res.het_flag == "H":
                continue
            ca = res.find_atom("CA", "*")
            if ca is not None:
                d.setdefault((res.seqid.num, res.name), ca.pos)
    return d


def kabsch(P, Q):
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    return R, t


def ligand_atoms(st, ccd):
    best = None
    for ch in st[0]:
        for res in ch:
            if res.het_flag == "H" and res.name == ccd:
                heavy = [a for a in res if a.element.name != "H"]
                if len(heavy) >= 6 and (best is None or len(heavy) > len(best)):
                    best = heavy
    if best is None:
        return None
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for a in best])


def main():
    bench = json.load(open("data/pdb_hops/benchmark.json"))
    lig = pd.read_csv("data/pdb_hops/ligands.csv")
    test_ids = set(bench["splits"]["scaffold"]["test"])
    pairs = [p for p in bench["pairs"] if p["id"] in test_ids]
    pairs = [p for p in pairs if os.path.exists(f"data/pdb_hops/pdb/{p['pdb1']}.pdb")
             and os.path.exists(f"data/pdb_hops/pdb/{p['pdb2']}.pdb")]
    os.makedirs("results/svhop_pose", exist_ok=True)
    rows = []
    for p in pairs[:40]:
        p1 = f"data/pdb_hops/pdb/{p['pdb1']}.pdb"
        p2 = f"data/pdb_hops/pdb/{p['pdb2']}.pdb"
        sel = select_copy(p1, p["ccd1"])
        if sel is None:
            continue
        st1 = gemmi.read_structure(p1); st1.setup_entities()
        st2 = gemmi.read_structure(p2); st2.setup_entities()
        d1 = ca_dict(st1, sel["chain"])
        d2 = ca_dict(st2, sel["chain"])
        common = sorted(set(d1) & set(d2))
        if len(common) < 20:
            d2 = ca_dict(st2, None)
            common = sorted(set(d1) & set(d2))
        if len(common) < 20:
            print(f"[skip] {p['target']} {p['pdb1']} ca={len(common)}", flush=True)
            continue
        P = np.array([[d1[k].x, d1[k].y, d1[k].z] for k in common])
        Q = np.array([[d2[k].x, d2[k].y, d2[k].z] for k in common])
        R, t = kabsch(P, Q)
        xyz = ligand_atoms(st2, p["ccd2"])
        if xyz is None:
            continue
        xyz_t = xyz @ R - t @ R
        # RMSD of CA after fit
        fit = P @ R.T + t
        ca_rmsd = float(np.sqrt(((fit - Q) ** 2).sum(1).mean()))
        atoms = [(x, y, z) for x, y, z in xyz_t]
        cs = contacts_from_atoms(p1, atoms)
        ref = sel["contacts"]
        recall = len(cs & ref) / max(1, len(ref))
        jac = len(cs & ref) / max(1, len(cs | ref))
        rows.append({"target": p["target"], "family": p["family"],
                     "pdb1": p["pdb1"], "pdb2": p["pdb2"],
                     "ca_n": len(common), "ca_rmsd": ca_rmsd,
                     "true_recall": recall, "jaccard": jac,
                     "n_true_contacts": len(cs), "n_ref_contacts": len(ref)})
        print(f"[{p['target']}] {p['pdb1']}/{p['pdb2']} ca_rmsd={ca_rmsd:.2f} "
              f"true_recall={recall:.2f} jac={jac:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/svhop_pose/transfer.csv", index=False)
    if len(df):
        print(df[["ca_rmsd", "true_recall", "jaccard"]].mean().to_string())
        print(f"recall>=0.6: {(df.true_recall >= 0.6).mean():.2f}  n={len(df)}")


if __name__ == "__main__":
    main()

