"""Learned interaction-field ranker for SV-Hop.

Predict contact recovery from cheap interaction/property features, then rerank
candidates.  Grouped CV by pair prevents leakage.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")


def main():
    bench = json.load(open("data/pdb_hops/benchmark.json"))
    lig = pd.read_csv("data/pdb_hops/ligands.csv")
    smi = {(r.pdb, r.ccd): r.smiles for r in lig.itertuples()}
    cand = pd.read_csv("results/svhop_lexicon/candidates.csv")
    cand = cand[cand.smiles.notna()].copy()
    cand["prop"] = -(cand.dlogp.abs() + cand.dtpsa.abs() / 20 + cand.dmw.abs() / 80
                     + cand.dhbd.abs() + cand.dhba.abs()) - 0.1 * cand.sa.fillna(10)
    cand["prank"] = cand.groupby("pair")["prop"].rank(ascending=False, method="first").astype(int) - 1
    pc = pd.read_csv("results/svhop_contact/per_candidate.csv")
    m = pc.merge(cand[["pair", "prank", "smiles", "conserved", "sa_ok", "scaf_sim_ref",
                       "dlogp", "dtpsa", "dmw", "dhbd", "dhba", "sa", "k", "prop"]],
                 left_on=["pair", "rank"], right_on=["pair", "prank"], how="inner")

    def gen(s):
        mm = Chem.MolFromSmiles(s) if s else None
        if mm is None:
            return None
        sc = MurckoScaffold.GetScaffoldForMol(mm)
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc)) if sc and sc.GetNumAtoms() else None

    pm = {p["id"]: p for p in bench["pairs"]}
    m["novel"] = m.apply(lambda r: int(gen(r.smiles) != gen(smi.get((pm[r.pair]["pdb1"], pm[r.pair]["ccd1"]), ""))), axis=1)
    m["joint"] = ((m.contact_recall >= 0.6) & (m.conserved == 1) & (m.sa_ok == 1) & (m.novel == 1)).astype(int)

    feats = ["mcs_atoms", "scaf_sim_ref", "dlogp", "dtpsa", "dmw", "dhbd", "dhba",
             "sa", "conserved", "sa_ok", "k"]
    X = m[feats].fillna(0).values
    y = m.contact_recall.values
    groups = m.pair.values
    oof = np.zeros(len(m))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                            max_leaf_nodes=31, l2_regularization=1.0,
                                            random_state=0)
        reg.fit(X[tr], y[tr])
        oof[te] = reg.predict(X[te])
    m["pred_contact"] = np.clip(oof, 0, 1)
    print(f"[rank] n={len(m)} spearman={spearmanr(y, oof).statistic:.3f} "
          f"auc>=0.6={roc_auc_score((y >= 0.6).astype(int), oof):.3f}")

    # per-pair arms
    rows = []
    sel_rows = []
    for pid, g in m.groupby("pair"):
        g = g.copy()
        g["comb"] = (g.pred_contact.rank(pct=True) + g.prop.rank(pct=True)) / 2
        prop_top = g[g.prank == 0].iloc[0]
        pred_top = g.loc[g.pred_contact.idxmax()]
        comb_top = g.loc[g.comb.idxmax()]
        ok = g[(g.conserved == 1) & (g.sa_ok == 1) & (g.novel == 1)]
        cons_top = ok.loc[ok.pred_contact.idxmax()] if len(ok) else prop_top
        rows.append({
            "pair": pid, "family": pm[pid]["family"],
            "property": int(prop_top.joint),
            "pred_contact": int(pred_top.joint),
            "combined": int(comb_top.joint),
            "constrained": int(cons_top.joint),
            "contact_prop": float(prop_top.contact_recall),
            "contact_pred": float(pred_top.contact_recall),
            "contact_comb": float(comb_top.contact_recall),
            "contact_cons": float(cons_top.contact_recall),
        })
        sel_rows.append({"pair": pid, "target": pm[pid]["target"], "family": pm[pid]["family"],
                         "smiles": cons_top.smiles, "ref_smiles": smi.get((pm[pid]["pdb1"], pm[pid]["ccd1"])),
                         "pdb1": pm[pid]["pdb1"], "ccd1": pm[pid]["ccd1"], "ccd2": pm[pid]["ccd2"],
                         "contact_recall": float(cons_top.contact_recall),
                         "pred_contact": float(cons_top.pred_contact),
                         "conserved": int(cons_top.conserved), "sa_ok": int(cons_top.sa_ok),
                         "novel": int(cons_top.novel), "sa": float(cons_top.sa)})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    out = {"n_pairs": len(df), "spearman": float(spearmanr(y, oof).statistic),
           "auc": float(roc_auc_score((y >= 0.6).astype(int), oof)), "arms": {}}
    for arm in ["property", "pred_contact", "combined", "constrained"]:
        v = df[arm].values
        bs = v[rng.integers(0, len(v), size=(4000, len(v)))].mean(1)
        out["arms"][arm] = {"mean": float(v.mean()),
                            "ci_lo": float(np.percentile(bs, 2.5)),
                            "ci_hi": float(np.percentile(bs, 97.5))}
    for arm in ["contact_prop", "contact_pred", "contact_comb", "contact_cons"]:
        out[arm] = float(df[arm].mean())
    # paired combined - property
    d = df.combined.values - df.property.values
    bs = d[rng.integers(0, len(d), size=(4000, len(d)))].mean(1)
    out["combined_minus_property"] = {"mean": float(d.mean()),
                                      "ci_lo": float(np.percentile(bs, 2.5)),
                                      "ci_hi": float(np.percentile(bs, 97.5))}
    d = df.constrained.values - df.property.values
    bs = d[rng.integers(0, len(d), size=(4000, len(d)))].mean(1)
    out["constrained_minus_property"] = {"mean": float(d.mean()),
                                         "ci_lo": float(np.percentile(bs, 2.5)),
                                         "ci_hi": float(np.percentile(bs, 97.5)),
                                         "p": float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))}
    pd.DataFrame(sel_rows).to_csv("results/svhop_full/selected.csv", index=False)
    json.dump(out, open("results/svhop_full/interaction_ranker.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()












