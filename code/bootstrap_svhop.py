"""Bootstrap CIs and full baseline table for SV-Hop (298 pairs)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

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
    cand["simrank"] = cand.groupby("pair")["scaf_sim_ref"].rank(ascending=False, method="first").astype(int) - 1
    pc = pd.read_csv("results/svhop_contact/per_candidate.csv")
    m = pc.merge(cand[["pair", "prank", "simrank", "smiles", "conserved", "sa_ok", "scaf_sim_ref"]],
                 left_on=["pair", "rank"], right_on=["pair", "prank"], how="inner")
    m["simrank"] = m.groupby("pair")["scaf_sim_ref"].rank(ascending=False, method="first").astype(int) - 1

    def gen(s):
        mm = Chem.MolFromSmiles(s) if s else None
        if mm is None:
            return None
        sc = MurckoScaffold.GetScaffoldForMol(mm)
        if sc is None or sc.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc))

    pm = {p["id"]: p for p in bench["pairs"]}
    m["novel"] = m.apply(
        lambda r: int(gen(r.smiles) != gen(smi.get((pm[r.pair]["pdb1"], pm[r.pair]["ccd1"]), ""))), axis=1)
    m["joint"] = ((m.contact_recall >= 0.6) & (m.conserved == 1) & (m.sa_ok == 1) & (m.novel == 1)).astype(int)

    # per-pair arm values
    rows = []
    for pid, g in m.groupby("pair"):
        r = {"pair": pid, "family": pm[pid]["family"]}
        r["property"] = float(g[g.prank == 0].joint.iloc[0])
        r["sim2d"] = float(g[g.simrank == 0].joint.iloc[0])
        r["random_expected"] = float(g.joint.mean())
        r["copy"] = 0.0
        r["contact_property"] = float(g[g.prank == 0].contact_recall.iloc[0])
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"[pairs] {len(df)}")

    rng = np.random.default_rng(0)
    out = {"n_pairs": len(df), "arms": {}}
    for arm in ["property", "sim2d", "random_expected", "copy"]:
        vals = df[arm].values
        boot = vals[rng.integers(0, len(vals), size=(4000, len(vals)))].mean(1)
        out["arms"][arm] = {
            "mean": float(vals.mean()),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
        }
    # contact recovery CI (top-1)
    vals = df.contact_property.values
    boot = vals[rng.integers(0, len(vals), size=(4000, len(vals)))].mean(1)
    out["contact_top1"] = {"mean": float(vals.mean()),
                           "ci_lo": float(np.percentile(boot, 2.5)),
                           "ci_hi": float(np.percentile(boot, 97.5))}
    out["by_family"] = {
        fam: {arm: float(g[arm].mean()) for arm in ["property", "sim2d", "random_expected"]}
        for fam, g in df.groupby("family")
    }
    json.dump(out, open("results/svhop_full/bootstrap.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

