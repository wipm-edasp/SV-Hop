"""C1: orthogonal validation of the contact-conservation criterion.

Two checks on pairs whose two crystal ligands are both measured in ChEMBL:
1. External activity: does contact recovery track the experimental potency gap?
2. Pharmacophore overlap: an independent, feature-based measure of the transferred
   experimental hop in the reference frame.
"""
from __future__ import annotations

import json
import os

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures
from scipy.stats import spearmanr

from crystal_pose_transfer import ca_dict, kabsch
from dock_copy_check2 import contacts_from_atoms, crystal_ligand, select_copy

RDLogger.DisableLog("rdApp.*")
OUT = "results/c1_validation"
os.makedirs(OUT, exist_ok=True)

FDEF = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
FACT = ChemicalFeatures.BuildFeatureFactory(FDEF)


def ikey(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToInchiKey(m) if m else None


def feature_points(mol):
    conf = mol.GetConformer()
    out = []
    for f in FACT.GetFeaturesForMol(mol):
        ids = list(f.GetAtomIds())
        if not ids:
            continue
        pos = np.mean([list(conf.GetAtomPosition(i)) for i in ids], axis=0)
        out.append((f.GetFamily(), pos))
    return out


def pharm_overlap(fa, fb, tol):
    if not fa:
        return float("nan")
    used, hit = set(), 0
    for fam, p in fa:
        best, bd = None, 1e9
        for j, (fam2, q) in enumerate(fb):
            if j in used or fam2 != fam:
                continue
            d = float(np.linalg.norm(p - q))
            if d < bd:
                bd, best = d, j
        if best is not None and bd <= tol:
            used.add(best)
            hit += 1
    return hit / len(fa)


def perm_p(x, y, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    vals = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n)])
    return float((np.abs(vals) >= abs(obs)).mean())


def activity(panel, key, target):
    recs = [x for x in panel.get(key, []) if x[0] == target]
    same = True
    if not recs:
        recs = panel.get(key, [])
        same = False
    if not recs:
        return None, None
    return float(np.median([x[1] for x in recs])), same


def main():
    bench = json.load(open("data/pdb_hops/benchmark.json"))
    lig = pd.read_csv("data/pdb_hops/ligands.csv")
    smi = {(r.pdb, r.ccd): r.smiles for r in lig.itertuples()}
    keys = {(r.pdb, r.ccd): ikey(r.smiles) for r in lig.itertuples() if r.smiles}

    panel = {}
    for line in open("data/chembl/panel.jsonl", encoding="utf-8"):
        r = json.loads(line)
        k = ikey(r["smiles"])
        if k and r.get("pchembl") is not None:
            panel.setdefault(k, []).append((r["target"], float(r["pchembl"])))
    print(f"[panel] keys={len(panel)}")

    rows = []
    for p in bench["pairs"]:
        k1 = keys.get((p["pdb1"], p["ccd1"]))
        k2 = keys.get((p["pdb2"], p["ccd2"]))
        if k1 not in panel or k2 not in panel:
            continue
        p1v, s1 = activity(panel, k1, p["target"])
        p2v, s2 = activity(panel, k2, p["target"])
        if p1v is None or p2v is None:
            continue
        ref_smi = smi.get((p["pdb1"], p["ccd1"]))
        true_smi = smi.get((p["pdb2"], p["ccd2"]))
        f1 = f"data/pdb_hops/pdb/{p['pdb1']}.pdb"
        f2 = f"data/pdb_hops/pdb/{p['pdb2']}.pdb"
        if not (os.path.exists(f1) and os.path.exists(f2)):
            continue
        sel1 = select_copy(f1, p["ccd1"])
        sel2 = select_copy(f2, p["ccd2"])
        if sel1 is None or sel2 is None:
            continue
        mol1 = crystal_ligand(f1, p["ccd1"], sel1["chain"], sel1["seqid"], ref_smi)
        mol2 = crystal_ligand(f2, p["ccd2"], sel2["chain"], sel2["seqid"], true_smi)
        if mol1 is None or mol2 is None:
            continue
        st1 = gemmi.read_structure(f1); st1.setup_entities()
        st2 = gemmi.read_structure(f2); st2.setup_entities()
        d1 = ca_dict(st1, sel1["chain"])
        d2 = ca_dict(st2, sel1["chain"])
        common = sorted(set(d1) & set(d2))
        if len(common) < 20:
            d2 = ca_dict(st2, None)
            common = sorted(set(d1) & set(d2))
        if len(common) < 20:
            continue
        P = np.array([[d1[k].x, d1[k].y, d1[k].z] for k in common])
        Q = np.array([[d2[k].x, d2[k].y, d2[k].z] for k in common])
        R, t = kabsch(P, Q)

        conf = mol2.GetConformer()
        xyz = []
        for i in range(mol2.GetNumAtoms()):
            v = np.array(list(conf.GetAtomPosition(i)))
            w = v @ R - t @ R
            conf.SetAtomPosition(i, tuple(w))
            xyz.append(tuple(w))
        cs = contacts_from_atoms(f1, xyz)
        recall = len(cs & sel1["contacts"]) / max(1, len(sel1["contacts"]))
        fa = feature_points(mol1)
        fb = feature_points(mol2)
        rows.append({"pair": p["id"], "target": p["target"], "family": p["family"],
                     "pdb1": p["pdb1"], "pdb2": p["pdb2"],
                     "contact": recall, "p_ref": p1v, "p_hop": p2v,
                     "dp": p1v - p2v, "abs_dp": abs(p1v - p2v),
                     "same_target": bool(s1 and s2),
                     "pharm_2A": pharm_overlap(fa, fb, 2.0),
                     "pharm_1.5A": pharm_overlap(fa, fb, 1.5),
                     "n_feat_ref": len(fa)})
        if len(rows) % 10 == 0:
            print(f"[pairs] {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/pairs.csv", index=False)
    ok = df[df.same_target & df.contact.notna()]
    res = {"n_pairs": int(len(df)), "n_same_target": int(len(ok)),
           "n_contact_lt_0.6": int((ok.contact < 0.6).sum()),
           "median_abs_dp": float(ok.abs_dp.median())}
    for name, x, y in [("contact_vs_abs_dp", ok.contact, ok.abs_dp),
                       ("contact_vs_dp", ok.contact, ok.dp),
                       ("contact_vs_pharm2A", ok.contact, ok.pharm_2A),
                       ("pharm2A_vs_abs_dp", ok.pharm_2A, ok.abs_dp)]:
        sub = pd.DataFrame({"x": x, "y": y}).dropna()
        res[name] = {"n": int(len(sub)),
                     "rho": float(spearmanr(sub.x, sub.y).statistic)}
        if name in ("contact_vs_abs_dp", "contact_vs_pharm2A"):
            res[name]["p_perm"] = perm_p(sub.x, sub.y)
    res["mean_contact"] = float(ok.contact.mean())
    weak = ok[ok.dp <= -1.0]
    same = ok[ok.abs_dp < 0.5]
    res["weak_hop"] = {"n": int(len(weak)), "mean_contact": float(weak.contact.mean()) if len(weak) else None,
                       "mean_abs_dp": float(weak.abs_dp.mean()) if len(weak) else None}
    res["similar_potency"] = {"n": int(len(same)), "mean_contact": float(same.contact.mean()) if len(same) else None}
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
