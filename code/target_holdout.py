"""Target-held-out evaluation for SV-Hop.

Train targets and test targets are disjoint. Rules come from PDB target-train
pairs plus ChEMBL rules whose targets are not in the test set. Candidates and
contact recovery are generated for both target-train (ranker training) and
target-test (evaluation).
"""
from __future__ import annotations

import collections
import json
import os
import random

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

from chem import sa_score
from core_swap_multi import decompose_multi, join_multi
from dock_copy_check2 import contacts_from_atoms, crystal_ligand, select_copy

RDLogger.DisableLog("rdApp.*")

OUT = os.environ.get("OUT_DIR", "results/target_holdout")
USE_CHEMBL = os.environ.get("USE_CHEMBL", "1") == "1"
_EMBED = AllChem.ETKDGv3()
_EMBED.randomSeed = 0
_EMBED.timeout = 10
_C = {}


def fp(smi, n=1024):
    key = (smi, n)
    if key in _C:
        return _C[key]
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        v = np.zeros(n, dtype=np.float32)
    else:
        bv = rdChemFp(m, n)
        v = (np.frombuffer(bv.ToBitString().encode(), dtype=np.uint8) - 48).astype(np.float32)
    _C[key] = v
    return v


def rdChemFp(m, n):
    from rdkit.Chem import rdMolDescriptors
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=n)


def dec(smi):
    if smi not in _C:
        _C[smi] = decompose_multi(smi)
    return _C[smi]


def props(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    from rdkit.Chem import Descriptors, Lipinski
    return (Descriptors.MolLogP(m), Descriptors.TPSA(m), Lipinski.NumHDonors(m),
            Lipinski.NumHAcceptors(m), Descriptors.MolWt(m))


def gen_scaf(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    s = MurckoScaffold.GetScaffoldForMol(m)
    if s is None or s.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(s))


def build_rules(bench, smi, train_ids, test_targets):
    rules = []
    for p in bench["pairs"]:
        if p["id"] not in train_ids:
            continue
        ref, true = smi.get((p["pdb1"], p["ccd1"])), smi.get((p["pdb2"], p["ccd2"]))
        if not ref or not true:
            continue
        d1, d2 = dec(ref), dec(true)
        if d1 is None or d2 is None or d1[2] != d2[2] or d1[3] == d2[3]:
            continue
        rules.append({"s1": d1[3], "s2": d2[3], "k": d1[2], "true_smi": true})
    for line in (open("data/chembl_core_rules.jsonl", encoding="utf-8") if USE_CHEMBL else []):
        r = json.loads(line)
        if r["target"] in test_targets:
            continue
        rules.append({"s1": r["s1"], "s2": r["s2"], "k": r["k"], "true_smi": r["rep_smi"]})
    seen, uniq = set(), []
    for r in rules:
        key = (r["s1"], r["s2"], r["k"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    by_k = collections.defaultdict(list)
    for r in uniq:
        by_k[r["k"]].append(r)
    for k in by_k:
        by_k[k] = {"rules": by_k[k], "F": np.stack([fp(r["s1"]) for r in by_k[k]])}
    return by_k, len(uniq)


_sel_cache, _xtal_cache = {}, {}


def get_ref(pdb, ccd, ref_smi):
    key = (pdb, ccd)
    if key not in _sel_cache:
        _sel_cache[key] = select_copy(pdb, ccd)
    sel = _sel_cache[key]
    if sel is None:
        return None
    xkey = (pdb, ccd, sel["chain"], sel["seqid"])
    if xkey not in _xtal_cache:
        _xtal_cache[xkey] = crystal_ligand(pdb, ccd, sel["chain"], sel["seqid"], ref_smi)
    return sel, _xtal_cache[xkey]


def process_pair(p, split, smi, by_k):
    ref = smi.get((p["pdb1"], p["ccd1"]))
    if not ref:
        return []
    d1 = dec(ref)
    if d1 is None or d1[2] not in by_k:
        return []
    k = d1[2]
    P = by_k[k]
    f1 = fp(d1[3])
    inter = np.minimum(P["F"], f1).sum(1)
    union = np.maximum(P["F"], f1).sum(1)
    sims = inter / np.maximum(union, 1e-9)
    order = np.argsort(-sims)
    ref_pr = props(ref)
    if ref_pr is None:
        return []
    pdb = f"data/pdb_hops/pdb/{p['pdb1']}.pdb"
    refpair = get_ref(pdb, p["ccd1"], ref)
    if refpair is None:
        return []
    sel, xtal = refpair
    if xtal is None:
        return []
    seen, cands = set(), []
    for idx in order:
        r = P["rules"][int(idx)]
        sim = float(sims[int(idx)])
        if r["s2"] == d1[3] or r["s2"] in seen:
            continue
        seen.add(r["s2"])
        if len(seen) > 40:
            break
        dd = dec(r["true_smi"])
        if dd is None or dd[2] != k:
            continue
        prod = join_multi(dd[0], d1[1])
        if prod is None:
            continue
        sm = Chem.MolToSmiles(prod)
        pp = props(sm)
        if pp is None or pp[4] > 700:
            continue
        d = [pp[i] - ref_pr[i] for i in range(5)]
        cons = int(abs(d[0]) <= 1 and abs(d[1]) <= 20 and abs(d[2]) <= 1
                   and abs(d[3]) <= 1 and abs(d[4]) <= 80)
        s = sa_score(sm)
        score = -(abs(d[0]) + abs(d[1]) / 20 + abs(d[4]) / 80
                  + abs(d[2]) + abs(d[3])) - 0.1 * s
        cands.append({"smiles": sm, "prop": score, "conserved": cons,
                      "sa_ok": int(s <= 5), "sa": s, "sim": sim,
                      "d": d, "k": k})
    if not cands:
        return []
    cands.sort(key=lambda x: -x["prop"])
    top = cands[:5]
    rows = []
    for rank, c in enumerate(top):
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, _EMBED) != 0:
            continue
        try:
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            mcs = rdFMCS.FindMCS([xtal, mol], timeout=5,
                                 atomCompare=rdFMCS.AtomCompare.CompareAny,
                                 bondCompare=rdFMCS.BondCompare.CompareAny)
            if mcs.numAtoms < 3:
                continue
            patt = Chem.MolFromSmarts(mcs.smartsString)
            m1 = xtal.GetSubstructMatch(patt)
            m2 = mol.GetSubstructMatch(patt)
            if not m1 or not m2:
                continue
            AllChem.AlignMol(mol, xtal, atomMap=list(zip(m2, m1)))
        except Exception:
            continue
        conf = mol.GetConformer()
        atoms = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
                 for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
        cs = contacts_from_atoms(pdb, atoms)
        rec = len(cs & sel["contacts"]) / max(1, len(sel["contacts"]))
        novel = int(gen_scaf(c["smiles"]) != gen_scaf(ref))
        rows.append({"split": split, "pair": p["id"], "target": p["target"],
                     "family": p["family"], "rank": rank, "smiles": c["smiles"],
                     "contact_recall": rec, "mcs_atoms": mcs.numAtoms,
                     "scaf_sim_ref": c["sim"], "dlogp": c["d"][0], "dtpsa": c["d"][1],
                     "dhbd": c["d"][2], "dhba": c["d"][3], "dmw": c["d"][4],
                     "sa": c["sa"], "conserved": c["conserved"], "sa_ok": c["sa_ok"],
                     "novel": novel, "prop": c["prop"], "k": c["k"]})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    bench = json.load(open("data/pdb_hops/benchmark.json"))
    lig = pd.read_csv("data/pdb_hops/ligands.csv")
    smi = {(r.pdb, r.ccd): r.smiles for r in lig.itertuples()}
    tr_ids = set(bench["splits"]["target"]["train"])
    te_ids = set(bench["splits"]["target"]["test"])
    tr_pairs = [p for p in bench["pairs"] if p["id"] in tr_ids]
    te_pairs = [p for p in bench["pairs"] if p["id"] in te_ids]
    test_targets = {p["target"] for p in te_pairs}
    print(f"[split] train pairs {len(tr_pairs)} test pairs {len(te_pairs)} "
          f"test targets {len(test_targets)}")

    by_k, n_rules = build_rules(bench, smi, tr_ids, test_targets)
    print(f"[rules] {n_rules} (ChEMBL test targets excluded)")

    rng = random.Random(0)
    rng.shuffle(tr_pairs)
    tr_pairs = tr_pairs[: int(os.environ.get("TRAIN_N", "700"))]
    te_pairs = te_pairs[: int(os.environ.get("TEST_N", "504"))]
    rows = []
    for i, p in enumerate(tr_pairs):
        rows.extend(process_pair(p, "train", smi, by_k))
        if i % 50 == 0:
            print(f"[train] {i}/{len(tr_pairs)} rows={len(rows)}", flush=True)
    for i, p in enumerate(te_pairs):
        rows.extend(process_pair(p, "test", smi, by_k))
        if i % 50 == 0:
            print(f"[test] {i}/{len(te_pairs)} rows={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/rows.csv", index=False)
    print(f"[rows] {len(df)} train={sum(df.split=='train')} test={sum(df.split=='test')}")

    # ranker: train on target-train, test on target-test
    feats = ["mcs_atoms", "scaf_sim_ref", "dlogp", "dtpsa", "dhbd", "dhba", "dmw",
             "sa", "conserved", "sa_ok", "k"]
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    tr = df[df.split == "train"]
    te = df[df.split == "test"].copy()
    reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                        max_leaf_nodes=31, l2_regularization=1.0,
                                        random_state=0)
    reg.fit(tr[feats].fillna(0).values, tr.contact_recall.values)
    te["pred_contact"] = np.clip(reg.predict(te[feats].fillna(0).values), 0, 1)
    ybin = (te.contact_recall >= 0.6).astype(int)
    spearman = float(spearmanr(te.contact_recall, te.pred_contact).statistic)
    auc = float(roc_auc_score(ybin, te.pred_contact)) if ybin.nunique() > 1 else float("nan")
    print(f"[ranker] holdout spearman={spearman:.3f} auc={auc:.3f}")

    # arms
    arm_rows = []
    for pid, g in te.groupby("pair"):
        g = g.copy()
        prop_top = g.sort_values("prop", ascending=False).iloc[0]
        sim_top = g.sort_values("scaf_sim_ref", ascending=False).iloc[0]
        pred_top = g.sort_values("pred_contact", ascending=False).iloc[0]
        ok = g[(g.conserved == 1) & (g.sa_ok == 1) & (g.novel == 1)]
        cons_top = ok.sort_values("pred_contact", ascending=False).iloc[0] if len(ok) else prop_top
        arm_rows.append({
            "pair": pid, "family": g.family.iloc[0],
            "property": int(prop_top.contact_recall >= 0.6 and prop_top.conserved == 1
                            and prop_top.sa_ok == 1 and prop_top.novel == 1),
            "sim2d": int(sim_top.contact_recall >= 0.6 and sim_top.conserved == 1
                         and sim_top.sa_ok == 1 and sim_top.novel == 1),
            "constrained": int(cons_top.contact_recall >= 0.6 and cons_top.conserved == 1
                               and cons_top.sa_ok == 1 and cons_top.novel == 1),
            "random_expected": float(((g.contact_recall >= 0.6) & (g.conserved == 1)
                                      & (g.sa_ok == 1) & (g.novel == 1)).mean()),
            "copy": 0,
        })
    ad = pd.DataFrame(arm_rows)
    rng = np.random.default_rng(0)
    out = {"n_train_pairs": int(tr.pair.nunique()), "n_test_pairs": int(te.pair.nunique()),
           "n_train_targets": int(tr.target.nunique()),
           "n_test_targets": int(te.target.nunique()),
           "spearman": spearman, "auc": auc, "arms": {}}
    for arm in ["property", "sim2d", "constrained", "random_expected", "copy"]:
        v = ad[arm].values
        bs = v[rng.integers(0, len(v), size=(4000, len(v)))].mean(1)
        out["arms"][arm] = {"mean": float(v.mean()),
                            "ci_lo": float(np.percentile(bs, 2.5)),
                            "ci_hi": float(np.percentile(bs, 97.5))}
    out["by_family"] = {f: {a: float(g[a].mean()) for a in ["property", "sim2d", "constrained"]}
                        for f, g in ad.groupby("family")}
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


