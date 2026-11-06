"""Copy positive control with correct ligand-copy selection.

For each (pdb, ccd), enumerate all copies; choose the one with the most protein
contacts; dock the copy ligand into that pocket and check RMSD / contact recall.
"""
from __future__ import annotations

import collections
import json
import os
import subprocess

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

RDLogger.DisableLog("rdApp.*")
SMINA = "./tools/smina.static"


def select_copy(pdb, ccd):
    st = gemmi.read_structure(pdb)
    st.setup_entities()
    ns = gemmi.NeighborSearch(st[0], st.cell, 5.0).populate()
    best = None
    for ch in st[0]:
        for res in ch:
            if res.het_flag != "H" or res.name != ccd:
                continue
            heavy = [a for a in res if a.element.name != "H"]
            if len(heavy) < 6:
                continue
            cs = set()
            for a in heavy:
                for mk in ns.find_atoms(a.pos, "\0", radius=5.0):
                    r = st[0][mk.chain_idx][mk.residue_idx]
                    if r.het_flag != "H":
                        cs.add(f"{r.seqid.num}:{r.name}")
            if best is None or len(cs) > len(best["contacts"]):
                xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in heavy])
                best = {
                    "chain": ch.name, "seqid": res.seqid.num,
                    "center": xyz.mean(0),
                    "size": min(max(float(np.max(xyz.max(0) - xyz.min(0)) + 8.0), 16.0), 26.0),
                    "contacts": cs,
                }
    return best


def crystal_ligand(pdb, ccd, chain, seqid, smi):
    m = Chem.MolFromPDBFile(pdb, sanitize=False, removeHs=False)
    if m is None:
        return None
    frags = Chem.SplitMolByPDBResidues(m)
    target = None
    for key, mol in frags.items():
        parts = key.split(":")
        if len(parts) >= 3 and parts[0] == chain and parts[1].isdigit() and int(parts[1]) == seqid and parts[2] == ccd:
            target = mol
            break
    if target is None:
        for key, mol in frags.items():
            if key.endswith(ccd):
                target = mol
                break
    if target is None:
        return None
    try:
        target = Chem.RemoveHs(target)
        tmpl = Chem.MolFromSmiles(smi)
        target = AllChem.AssignBondOrdersFromTemplate(tmpl, target)
    except Exception:
        return None
    return target


def contacts_from_atoms(pdb, atoms, ns=None, st=None):
    if ns is None or st is None:
        st = gemmi.read_structure(pdb)
        st.setup_entities()
        ns = gemmi.NeighborSearch(st[0], st.cell, 5.0).populate()
    out = set()
    for x, y, z in atoms:
        for mk in ns.find_atoms(gemmi.Position(x, y, z), "\0", radius=5.0):
            r = st[0][mk.chain_idx][mk.residue_idx]
            if r.het_flag != "H":
                out.add(f"{r.seqid.num}:{r.name}")
    return out


def main():
    bench = json.load(open("data/pdb_hops/benchmark.json"))
    lig = pd.read_csv("data/pdb_hops/ligands.csv")
    smi = {(r.pdb, r.ccd): r.smiles for r in lig.itertuples()}
    test_ids = set(bench["splits"]["scaffold"]["test"])
    pairs = [p for p in bench["pairs"] if p["id"] in test_ids]
    # one pair per target, up to 10
    picked, seen = [], collections.Counter()
    for p in pairs:
        if seen[p["target"]] >= 5:
            continue
        if not os.path.exists(f"data/pdb_hops/pdb/{p['pdb1']}.pdb"):
            continue
        picked.append(p)
        seen[p["target"]] += 1
        if len(picked) >= 50:
            break
    os.makedirs("results/svhop_copycheck", exist_ok=True)
    rows = []
    for p in picked:
        pdb = f"data/pdb_hops/pdb/{p['pdb1']}.pdb"
        ref_smi = smi.get((p["pdb1"], p["ccd1"]))
        if not ref_smi:
            continue
        sel = select_copy(pdb, p["ccd1"])
        if sel is None:
            print(f"[skip] {p['target']} no copy", flush=True)
            continue
        crystal = crystal_ligand(pdb, p["ccd1"], sel["chain"], sel["seqid"], ref_smi)
        if crystal is None:
            print(f"[skip] {p['target']} no crystal mol", flush=True)
            continue
        sdf = f"results/svhop_copycheck/copy_{p['id']}.sdf"
        w = Chem.SDWriter(sdf)
        m = Chem.AddHs(Chem.MolFromSmiles(ref_smi))
        AllChem.EmbedMolecule(m, randomSeed=0)
        m.SetProp("_Name", "copy")
        w.write(m)
        w.close()
        out = f"results/svhop_copycheck/dock_{p['id']}.sdf"
        c = sel["center"]
        s = sel["size"]
        cmd = [SMINA, "-r", pdb, "-l", sdf,
               "--center_x", f"{c[0]:.3f}", "--center_y", f"{c[1]:.3f}", "--center_z", f"{c[2]:.3f}",
               "--size_x", f"{s:.1f}", "--size_y", f"{s:.1f}", "--size_z", f"{s:.1f}",
               "--out", out, "--num_modes", "1", "--seed", "0", "--cpu", "4"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=1200)
        except Exception:
            continue
        pose = next((x for x in Chem.SDMolSupplier(out, sanitize=False) if x is not None), None)
        if pose is None:
            continue
        try:
            pose = Chem.RemoveHs(pose)
            rmsd = float(rdMolAlign.GetBestRMS(pose, crystal))
        except Exception:
            rmsd = np.nan
        conf = pose.GetConformer()
        atoms = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
                 for i in range(pose.GetNumAtoms())]
        cs = contacts_from_atoms(pdb, atoms)
        recall = len(cs & sel["contacts"]) / max(1, len(sel["contacts"]))
        rows.append({"pair_id": p["id"], "target": p["target"], "pdb": p["pdb1"], "chain": sel["chain"],
                     "seqid": sel["seqid"], "n_contacts": len(sel["contacts"]),
                     "rmsd": rmsd, "recall": recall})
        print(f"[{p['target']}] {p['pdb1']} contacts={len(sel['contacts'])} "
              f"rmsd={rmsd:.2f} recall={recall:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/svhop_copycheck/per_target.csv", index=False)
    if len(df):
        print(df[["rmsd", "recall"]].mean().to_string())
        print(f"pass (rmsd<2 & recall>0.6): {((df.rmsd < 2) & (df.recall > 0.6)).mean():.2f}")


if __name__ == "__main__":
    main()



