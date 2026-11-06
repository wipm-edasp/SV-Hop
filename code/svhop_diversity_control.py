"""Equal-diversity control for the SV-Hop lexicon ablation.

Question: is the PDB-only advantage (81.9% vs 65.3% joint top-1 on the target-held-out
split) explained by candidate diversity rather than by rule quality?

Method: for every test pair, compute the number of distinct generic Murcko scaffolds among
the candidates in each arm. Then, per pair, subsample both arms down to the smaller of the
two diversity counts (one candidate per scaffold, highest predicted contact), and recompute
the joint outcome. Repeat 200 times and average. If the arm difference disappears at equal
diversity, diversity is the driver.

Inputs: rows.csv from the full-lexicon and PDB-only target-held-out runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import HistGradientBoostingRegressor

RDLogger.DisableLog("rdApp.*")

PASS_COLS = ("conserved", "sa_ok", "novel")
FEATS = ["mcs_atoms", "scaf_sim_ref", "dlogp", "dtpsa", "dhbd", "dhba", "dmw",
         "sa", "conserved", "sa_ok", "k"]


def generic_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:  # noqa: BLE001
        return None
    if not s:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(Chem.MolFromSmiles(s)))
    except Exception:  # noqa: BLE001
        return s


def add_scaffold(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cache: dict[str, str | None] = {}
    vals = []
    for s in out["smiles"]:
        if s not in cache:
            cache[s] = generic_scaffold(s)
        vals.append(cache[s])
    out["scaffold"] = vals
    return out


def pair_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, g in df.groupby("pair"):
        passed = (
            g[list(PASS_COLS)].fillna(0).astype(int).all(axis=1)
            & (g["contact_recall"].astype(float) >= 0.6)
        )
        rows.append({
            "pair": pair,
            "n": len(g),
            "d": g["scaffold"].nunique(dropna=True),
            "joint": int(passed.any()),
            "joint_count": int(passed.sum()),
        })
    return pd.DataFrame(rows)


def rarefy(g: pd.DataFrame, d_target: int, rng, n_draws: int) -> float:
    """Mean joint success when the pair is reduced to d_target distinct scaffolds."""
    if d_target <= 0:
        return float("nan")
    by_scaf: dict[str, pd.Series] = {}
    for _, row in g.iterrows():
        s = row["scaffold"]
        if s is None:
            continue
        if s not in by_scaf or float(row.get("pred_contact", 0) or 0) > float(by_scaf[s].get("pred_contact", 0) or 0):
            by_scaf[s] = row
    scafs = list(by_scaf)
    if len(scafs) <= d_target:
        chosen = scafs
        draws = 1
    else:
        chosen = None
        draws = n_draws
    hits = 0
    total = 0
    for _ in range(draws):
        pick = scafs if chosen is not None else list(rng.choice(scafs, size=d_target, replace=False))
        ok = False
        for s in pick:
            r = by_scaf[s]
            if all(int(r[c] or 0) == 1 for c in PASS_COLS) and float(r["contact_recall"]) >= 0.6:
                ok = True
                break
        hits += int(ok)
        total += 1
    return hits / total if total else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--pdbonly", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    full = add_scaffold(pd.read_csv(args.full))
    pdb = add_scaffold(pd.read_csv(args.pdbonly))
    full = full[full["split"] == "test"].reset_index(drop=True)
    pdb = pdb[pdb["split"] == "test"].reset_index(drop=True)

    tf, tp = pair_table(full), pair_table(pdb)
    m = tf.merge(tp, on="pair", suffixes=("_full", "_pdb"))
    print(f"test pairs: full {len(tf)}, pdb-only {len(tp)}, matched {len(m)}")
    print(f"joint: full {tf.joint.mean():.3f}, pdb-only {tp.joint.mean():.3f}")
    print(f"mean distinct scaffolds per pair: full {tf.d.mean():.2f}, pdb-only {tp.d.mean():.2f}")

    rng = np.random.default_rng(args.seed)
    rare_full, rare_pdb = [], []
    for _, row in m.iterrows():
        d_target = int(min(row.d_full, row.d_pdb))
        gf = full[full.pair == row.pair]
        gp = pdb[pdb.pair == row.pair]
        rare_full.append(rarefy(gf, d_target, rng, args.draws))
        rare_pdb.append(rarefy(gp, d_target, rng, args.draws))
    m["rarefied_full"] = rare_full
    m["rarefied_pdb"] = rare_pdb

    same = m[m.d_full == m.d_pdb]
    print(f"\nequal-diversity pairs (same d): {len(same)}")
    if len(same):
        print(f"  joint full {same.joint_full.mean():.3f} vs pdb {same.joint_pdb.mean():.3f}")

    print(f"\nrarefied to the pair's smaller diversity:")
    print(f"  full-lexicon {m.rarefied_full.mean():.3f} vs PDB-only {m.rarefied_pdb.mean():.3f}")
    print(f"  mean d after rarefaction: {m[['d_full','d_pdb']].min(axis=1).mean():.2f}")

    by_d = []
    for d in sorted(m.d_pdb.unique()):
        s = m[m.d_pdb == d]
        if len(s) >= 3:
            by_d.append({"d": int(d), "n_pairs": int(len(s)),
                         "joint_full": float(s.joint_full.mean()),
                         "joint_pdb": float(s.joint_pdb.mean())})

    out = {
        "n_pairs": int(len(m)),
        "joint_full": float(tf.joint.mean()),
        "joint_pdb": float(tp.joint.mean()),
        "mean_d_full": float(tf.d.mean()),
        "mean_d_pdb": float(tp.d.mean()),
        "median_d_full": float(tf.d.median()),
        "median_d_pdb": float(tp.d.median()),
        "equal_d_pairs": int(len(same)),
        "equal_d_joint_full": float(same.joint_full.mean()) if len(same) else None,
        "equal_d_joint_pdb": float(same.joint_pdb.mean()) if len(same) else None,
        "rarefied_joint_full": float(m.rarefied_full.mean()),
        "rarefied_joint_pdb": float(m.rarefied_pdb.mean()),
        "by_d": by_d,
        "per_pair": m[[
            "pair", "d_full", "d_pdb", "joint_full", "joint_pdb",
            "rarefied_full", "rarefied_pdb",
        ]].to_dict(orient="records"),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
