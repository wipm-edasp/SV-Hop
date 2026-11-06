"""Rank SV-Hop core-swap candidates.

For each test pair, retrieve top-20 cores, swap decorations, record candidate
features, and compare ranking policies:
  random, property, SA, property+SA+novelty, oracle (true scaffold first).

Outcome: top-1/top-3 drug-like hop = valid & property-conserved & SA<=5.
"""
from __future__ import annotations

import collections
import json
import os
import random

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors

from chem import murcko_smiles, props, sa_score
from core_swap_multi import decompose_multi, join_multi

RDLogger.DisableLog("rdApp.*")

_CACHE = {}


def fp(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def dec(smi):
    if smi not in _CACHE:
        _CACHE[smi] = decompose_multi(smi)
    return _CACHE[smi]


def main():
    bench = json.load(open("data/pdb_hops/benchmark.json"))
    lig = pd.read_csv("data/pdb_hops/ligands.csv")
    smi = {(r.pdb, r.ccd): r.smiles for r in lig.itertuples()}
    train_ids = set(bench["splits"]["scaffold"]["train"])
    test_ids = set(bench["splits"]["scaffold"]["test"])

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
    for line in open("data/chembl_core_rules.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rules.append({"s1": r["s1"], "s2": r["s2"], "k": r["k"], "true_smi": r["rep_smi"]})
    seen_rule = set()
    uniq = []
    for r in rules:
        key = (r["s1"], r["s2"], r["k"])
        if key in seen_rule:
            continue
        seen_rule.add(key)
        uniq.append(r)
    rules = uniq
    by_k = collections.defaultdict(list)
    for r in rules:
        by_k[r["k"]].append(r)
    for k in by_k:
        by_k[k] = {"rules": by_k[k], "F": np.stack([fp(r["s1"]) for r in by_k[k]])}
    print(f"[rules] {len(rules)}")

    rows = []
    for p in bench["pairs"]:
        if p["id"] not in test_ids:
            continue
        ref = smi.get((p["pdb1"], p["ccd1"]))
        if not ref:
            continue
        d1 = dec(ref)
        if d1 is None or d1[2] not in by_k:
            continue
        k = d1[2]
        f1 = fp(d1[3])
        if f1 is None:
            continue
        P = by_k[k]
        inter = np.minimum(P["F"], f1).sum(1)
        union = np.maximum(P["F"], f1).sum(1)
        sims = inter / np.maximum(union, 1e-9)
        order = np.argsort(-sims)
        ref_pr = props(ref)
        seen_s2 = set()
        n = 0
        for idx in order:
            r = P["rules"][int(idx)]
            sim = float(sims[int(idx)])
            if r["s2"] == d1[3] or r["s2"] in seen_s2:
                continue
            seen_s2.add(r["s2"])
            n += 1
            if n > 20:
                break
            d = dec(r["true_smi"])
            if d is None or d[2] != k:
                continue
            prod = join_multi(d[0], d1[1])
            if prod is None:
                rows.append({"pair": p["id"], "target": p["target"], "k": k,
                             "cand_s2": r["s2"], "scaf_sim_ref": sim,
                             "valid": 0, "conserved": 0, "sa_ok": 0,
                             "true_scaf": 0, "sa": None,
                             "dlogp": None, "dtpsa": None, "dmw": None,
                             "dhbd": None, "dhba": None, "smiles": None})
                continue
            sm = Chem.MolToSmiles(prod)
            pp = props(sm)
            s = sa_score(sm)
            cons = 0
            if pp and ref_pr:
                cons = int(abs(pp["logp"] - ref_pr["logp"]) <= 1
                           and abs(pp["tpsa"] - ref_pr["tpsa"]) <= 20
                           and abs(pp["mw"] - ref_pr["mw"]) <= 80
                           and abs(pp["hbd"] - ref_pr["hbd"]) <= 1
                           and abs(pp["hba"] - ref_pr["hba"]) <= 1)
            rows.append({
                "pair": p["id"], "target": p["target"], "k": k,
                "cand_s2": r["s2"], "scaf_sim_ref": sim,
                "valid": 1, "conserved": cons, "sa_ok": int(s <= 5),
                "true_scaf": 0,
                "sa": s,
                "dlogp": (pp["logp"] - ref_pr["logp"]) if pp and ref_pr else None,
                "dtpsa": (pp["tpsa"] - ref_pr["tpsa"]) if pp and ref_pr else None,
                "dmw": (pp["mw"] - ref_pr["mw"]) if pp and ref_pr else None,
                "dhbd": (pp["hbd"] - ref_pr["hbd"]) if pp and ref_pr else None,
                "dhba": (pp["hba"] - ref_pr["hba"]) if pp and ref_pr else None,
                "smiles": sm,
            })

    df = pd.DataFrame(rows)
    os.makedirs("results/svhop_lexicon", exist_ok=True)
    df.to_csv("results/svhop_lexicon/candidates.csv", index=False)
    print(f"[candidates] {len(df)} over {df.pair.nunique()} pairs")

    def score_policy(g, name):
        g = g.copy()
        if name == "random":
            g["score"] = np.random.default_rng(0).random(len(g))
        elif name == "prop":
            g["score"] = -(g.dlogp.abs() / 1.0 + g.dtpsa.abs() / 20
                           + g.dmw.abs() / 80 + g.dhbd.abs() + g.dhba.abs())
        elif name == "sa":
            g["score"] = -g.sa.fillna(10)
        elif name == "prop_sa":
            g["score"] = (-(g.dlogp.abs() / 1.0 + g.dtpsa.abs() / 20
                            + g.dmw.abs() / 80 + g.dhbd.abs() + g.dhba.abs())
                          - 0.1 * g.sa.fillna(10) + 0.2 * (1 - g.scaf_sim_ref))
        elif name == "oracle":
            g["score"] = g.true_scaf * 10 - g.dlogp.abs().fillna(9)
        return g.sort_values("score", ascending=False)

    policies = ["random", "prop", "sa", "prop_sa", "oracle"]
    out = {}
    for name in policies:
        t1 = {"valid": [], "conserved": [], "sa_ok": [], "druglike": [], "true_scaf": []}
        t3 = {"druglike": [], "true_scaf": []}
        for pid, g in df.groupby("pair"):
            if len(g) == 0:
                continue
            r = score_policy(g, name).iloc[0]
            t1["valid"].append(r.valid)
            t1["conserved"].append(r.conserved)
            t1["sa_ok"].append(r.sa_ok)
            t1["druglike"].append(int(r.valid and r.conserved and r.sa_ok))
            t1["true_scaf"].append(r.true_scaf)
            top3 = score_policy(g, name).head(3)
            t3["druglike"].append(int(((top3.valid == 1) & (top3.conserved == 1)
                                       & (top3.sa_ok == 1)).any()))
            t3["true_scaf"].append(int(top3.true_scaf.max() > 0))
        out[name] = {
            "top1": {k: float(np.mean(v)) for k, v in t1.items()},
            "top3": {k: float(np.mean(v)) for k, v in t3.items()},
        }
    summary = {"n_pairs": int(df.pair.nunique()), "n_candidates": len(df),
               "policies": out}
    json.dump(summary, open("results/svhop_lexicon/ranking_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()











