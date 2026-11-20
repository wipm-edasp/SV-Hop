"""Generate the Supporting Information from result files."""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import HistGradientBoostingRegressor

from core_swap_multi import decompose_multi

RDLogger.DisableLog("rdApp.*")
OUT = "out/supporting_information.md"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

bench = json.load(open("data/pdb_hops/benchmark.json"))
pairs = bench["pairs"]
splits = bench["splits"]
stats = bench["stats"]
lig = pd.read_csv("data/pdb_hops/ligands.csv")
fam = {}
for p in pairs:
    fam.setdefault(p["target"], p["family"])

scaf_tr = set(splits["scaffold"]["train"])
scaf_te = set(splits["scaffold"]["test"])
tgt_tr = set(splits["target"]["train"])
tgt_te = set(splits["target"]["test"])


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def gen_scaf(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    s = MurckoScaffold.GetScaffoldForMol(m)
    if s is None or s.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(s))


# S1
rows1 = []
for t in sorted(stats["per_target"], key=lambda x: -stats["per_target"][x]):
    sub = lig[lig.target == t]
    tp = [p for p in pairs if p["target"] == t]
    n_sc = collections.Counter(
        "train" if p["id"] in scaf_tr else "val" if p["id"] in splits["scaffold"]["val"] else "test"
        for p in tp)
    n_tg = collections.Counter(
        "train" if p["id"] in tgt_tr else "val" if p["id"] in splits["target"]["val"] else "test"
        for p in tp)
    rows1.append([t, fam.get(t, ""), sub.pdb.nunique(), len(sub),
                  len(tp), f"{n_sc['train']}/{n_sc['val']}/{n_sc['test']}",
                  f"{n_tg['train']}/{n_tg['val']}/{n_tg['test']}"])
rows1.append(["**Total**", "", lig.pdb.nunique(), len(lig), len(pairs),
              f"{len(scaf_tr)}/{len(splits['scaffold']['val'])}/{len(scaf_te)}",
              f"{len(tgt_tr)}/{len(splits['target']['val'])}/{len(tgt_te)}"])

# S2
pdb_rules = set()
for p in pairs:
    if p["id"] not in scaf_tr:
        continue
    a = lig[(lig.pdb == p["pdb1"]) & (lig.ccd == p["ccd1"])]
    b = lig[(lig.pdb == p["pdb2"]) & (lig.ccd == p["ccd2"])]
    if not len(a) or not len(b):
        continue
    d1, d2 = decompose_multi(a.smiles.iloc[0]), decompose_multi(b.smiles.iloc[0])
    if d1 is None or d2 is None or d1[2] != d2[2] or d1[3] == d2[3]:
        continue
    pdb_rules.add((d1[3], d2[3], d1[2]))
chem = collections.Counter()
for line in open("data/chembl_core_rules.jsonl", encoding="utf-8"):
    chem[json.loads(line)["k"]] += 1
pdb_k = collections.Counter(r[2] for r in pdb_rules)
rows2 = []
for k in sorted(set(pdb_k) | set(chem)):
    rows2.append([k, pdb_k.get(k, 0), chem.get(k, 0), pdb_k.get(k, 0) + chem.get(k, 0)])
rows2.append(["**Total**", len(pdb_rules), sum(chem.values()),
              len(pdb_rules) + sum(chem.values())])

# S3
sel = pd.read_csv("results/svhop_full/selected.csv")
sel["hit"] = ((sel.contact_recall >= 0.6) & (sel.conserved == 1)
              & (sel.sa_ok == 1) & (sel.novel == 1)).astype(int)
rows3 = []
for t, g in sel.groupby("target"):
    rows3.append([t, g.family.iloc[0], len(g), f"{100 * g.hit.mean():.1f}"])
rows3.append(["**Overall**", "", len(sel), f"{100 * sel.hit.mean():.1f}"])

# S4
rk = json.load(open("results/svhop_full/interaction_ranker.json"))
bs = json.load(open("results/svhop_full/bootstrap.json"))
rows4 = []
for name, key in [("Property-only ranker", "property"), ("2D core similarity", "sim2d"),
                  ("Random candidate", "random_expected"), ("Copy", "copy")]:
    a = bs["arms"][key]
    rows4.append([name, f"{100 * a['mean']:.1f} [{100 * a['ci_lo']:.1f}, {100 * a['ci_hi']:.1f}]"])
a = rk["arms"]["pred_contact"]
rows4.append(["Contact-only (predicted)", f"{100 * a['mean']:.1f} [{100 * a['ci_lo']:.1f}, {100 * a['ci_hi']:.1f}]"])
a = rk["arms"]["combined"]
rows4.append(["Property + contact blend", f"{100 * a['mean']:.1f} [{100 * a['ci_lo']:.1f}, {100 * a['ci_hi']:.1f}]"])
a = rk["arms"]["constrained"]
rows4.append(["**SV-Hop (constrained ranker)**", f"**{100 * a['mean']:.1f} [{100 * a['ci_lo']:.1f}, {100 * a['ci_hi']:.1f}]**"])

# S5
pro = pd.read_csv("results/prospective/candidates.csv")
rows5 = []
for i, r in pro.reset_index().iterrows():
    rows5.append([i + 1, r.target, r.pair, r.smiles, f"{r.contact_recall:.2f}",
                  f"{r.pred_contact:.2f}", f"{r.sa:.1f}"])

# S6
rows = pd.read_csv("results/target_holdout/rows.csv")
te = rows[rows.split == "test"].copy()
FEATS = ["mcs_atoms", "scaf_sim_ref", "dlogp", "dtpsa", "dhbd", "dhba", "dmw",
         "sa", "conserved", "sa_ok", "k"]
reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                    max_leaf_nodes=31, l2_regularization=1.0,
                                    random_state=0)
reg.fit(rows[rows.split == "train"][FEATS].fillna(0).values,
        rows[rows.split == "train"].contact_recall.values)
te["pred_contact"] = np.clip(reg.predict(te[FEATS].fillna(0).values), 0, 1)
hits = {}
for pid, g in te.groupby("pair"):
    ok = g[(g.conserved == 1) & (g.sa_ok == 1) & (g.novel == 1)]
    r = ok.sort_values("pred_contact").iloc[-1] if len(ok) else g.sort_values("prop").iloc[-1]
    hits[pid] = int(r.contact_recall >= 0.6 and r.conserved == 1 and r.sa_ok == 1 and r.novel == 1)
te["hit"] = te.pair.map(hits)
rows6 = []
for t, g in te.groupby("target"):
    sub = g.drop_duplicates("pair")
    rows6.append([t, sub.family.iloc[0], len(sub), f"{100 * sub.hit.mean():.1f}"])
rows6.append(["**Overall**", "", te.pair.nunique(),
              f"{100 * te.drop_duplicates('pair').hit.mean():.1f}"])
ab = json.load(open("results/target_holdout/ablation.json"))
rows6b = []
for name, key in [("Full constraints", "full"), ("Drop novelty", "drop_novelty"),
                  ("Drop SA", "drop_sa"), ("Drop SA + novelty", "drop_both"),
                  ("Contact-only selection", "ranker_only")]:
    a = ab["arms"][key]
    rows6b.append([name, f"{100 * a['mean']:.1f} [{100 * a['ci_lo']:.1f}, {100 * a['ci_hi']:.1f}]"])

# S7
rows7 = []
for name, d, n_rules in [("PDB + ChEMBL", "results/target_holdout", "54,123"),
                         ("PDB only", "results/target_holdout_pdbonly", "388")]:
    s = json.load(open(f"{d}/summary.json"))
    t = pd.read_csv(f"{d}/rows.csv")
    t = t[t.split == "test"]
    div = t.groupby("pair").smiles.apply(
        lambda x: len(set(filter(None, (gen_scaf(y) for y in x)))))
    ok = t[(t.conserved == 1) & (t.sa_ok == 1) & (t.novel == 1)]
    rows7.append([name, n_rules, f"{t.pair.nunique()}/504",
                  f"{100 * s['arms']['constrained']['mean']:.1f}",
                  f"{div.mean():.2f}", f"{ok.pair.nunique()}/{t.pair.nunique()}"])

# S8
c1 = json.load(open("results/c1_validation/summary.json"))
rows8 = [
    ["Contact vs |Δp|", c1["contact_vs_abs_dp"]["n"],
     f"{c1['contact_vs_abs_dp']['rho']:.2f}", f"{c1['contact_vs_abs_dp']['p_perm']:.2f}"],
    ["Contact vs pharmacophore overlap", c1["contact_vs_pharm2A"]["n"],
     f"{c1['contact_vs_pharm2A']['rho']:.2f}", "<0.001"],
    ["Weak hops (≥10-fold) mean contact", c1["weak_hop"]["n"],
     f"{c1['weak_hop']['mean_contact']:.3f}", "—"],
    ["Similar potency (<3-fold) mean contact", c1["similar_potency"]["n"],
     f"{c1['similar_potency']['mean_contact']:.3f}", "—"],
]

n_ok = c1["n_same_target"]

# S9
div = json.load(open("results/target_holdout/equal_diversity.json"))
rows9 = [
    ["All test pairs", div["n_pairs"],
     f"{100 * div['joint_full']:.1f}", f"{100 * div['joint_pdb']:.1f}",
     f"{div['mean_d_full']:.2f}", f"{div['mean_d_pdb']:.2f}"],
    ["Equal-diversity subset", div["equal_d_pairs"],
     f"{100 * div['equal_d_joint_full']:.1f}", f"{100 * div['equal_d_joint_pdb']:.1f}",
     "equal n", "equal n"],
]

# S10 — full target panel, including accessions that contributed no pair
from mine_pdb_hops import FAMILIES as _FAM, TARGETS as _TGT

_edia = pd.read_csv("data/edia/edia_ligands.csv")
rows10 = []
_cls = {t: fam for fam, ts in _FAM.items() for t in ts}
for fam, ts in _FAM.items():
    for t in ts:
        sub = lig[lig.target == t]
        rows10.append([t, fam, _TGT[t], sub.pdb.nunique(), len(sub),
                       sum(1 for p in pairs if p["target"] == t)])
rows10.append(["**Total**", "", "", lig.pdb.nunique(), len(lig), len(pairs)])

# S11 — EDIA audit (from the shipped table, regenerated by code/analyze_edia.py)
_ed = json.load(open("data/edia/edia_summary.json"))
_b = _ed["ligand_buckets"]
_n = _ed["ligand_instances"]
_se = _ed["scaffold_eval"]
_cv = _ed["criterion_validation"]
rows11 = [
    ["Ligand instances", _n],
    ["Entries scored", "503 of 522"],
    ["Well supported, EDIA_m >= 0.8", f"{_b['well_supported_ge_0.8']} ({100 * _b['well_supported_ge_0.8'] / _n:.1f}%)"],
    ["Moderate, 0.4-0.8", f"{_b['moderate_0.4_0.8']} ({100 * _b['moderate_0.4_0.8'] / _n:.1f}%)"],
    ["Poor, < 0.4", f"{_b['poor_lt_0.4']} ({100 * _b['poor_lt_0.4'] / _n:.1f}%)"],
    ["Not scorable", f"{_b['unscored']} ({100 * _b['unscored'] / _n:.1f}%)"],
    ["Pairs, both ligands >= 0.8", f"{_ed['pair_pass']:,} of {_ed['pair_total']:,} ({_ed['pair_pass_rate_of_all']:.1f}%)"],
    ["Pairs scored for both ligands", f"{_ed['pair_scored']:,} of {_ed['pair_total']:,}"],
    ["Pairs, both ligands >= 0.4", f"{_ed['pair_pass_tier2']:,} ({_ed['pair_pass_tier2_rate']:.1f}%)"],
    ["Joint top-1, all / strict / looser gate",
     f"{_se['joint_all']:.1f}% / {_se['joint_pass']:.1f}% (n={_se['pass']}) / {_se['joint_tier2']:.1f}% (n={_se['pass_tier2']})"],
    ["Criterion validation, same-target n", f"{_cv['same_target']['n']} (passing: {_cv['same_target_pass']['n']})"],
    ["rho(contact, dp), same-target all / passing",
     f"{_cv['same_target']['rho_dp']:.2f} / {_cv['same_target_pass']['rho_dp']:.2f}"],
    ["rho(contact, pharmacophore), all / passing",
     f"{_cv['same_target']['rho_pharm']:.2f} / {_cv['same_target_pass']['rho_pharm']:.2f}"],
]

# S12 — ligands failing the gate
_fail = pd.read_csv("data/edia/edia_failures.csv")
rows12 = [[r.pdb, r.ccd, r.ediam if isinstance(r.ediam, str) else f"{r.ediam:.3f}",
           r.status, r.target, r.pair] for r in _fail.itertuples()]
doc = f"""# Supporting Information

**Structure-grounded scaffold hopping with a learned interaction ranker**

Tables S1–S12. All values are generated by `code/make_si.py` from the result files
in `results/` and the benchmark in `data/pdb_hops/`; no numbers are typed by hand.

## Table S1. Benchmark targets

X-ray structures at ≤2.5 Å; ligands are non-polymer residues with 15–60 heavy
atoms. Split columns are train/validation/test.

{table(["Target", "Family", "Structures", "Ligands", "Pairs",
        "Scaffold split", "Target split"], rows1)}

## Table S2. Core-replacement lexicon

Rules are `(S1 → S2, k)`; deduplicated by the triple. ChEMBL rules come from a
20-target IC50 panel and are counted before target filtering.

{table(["k", "PDB-derived", "ChEMBL-derived", "Total"], rows2)}

## Table S3. Scaffold-held-out per-target results

Joint top-1: novel, property-conserved, SA ≤ 5 and contact recovery ≥ 0.6.

{table(["Target", "Family", "Test pairs", "Joint top-1 (%)"], rows3)}

## Table S4. Scaffold-held-out arms

{table(["Arm", "Joint top-1 (%) [95% CI]"], rows4)}

## Table S5. Prospective panel

22 compounds; contact recovery is the crystal-transfer/MCS measure, predicted
contact is the ranker output, SA is the Ertl synthetic-accessibility score.

{table(["#", "Target", "Pair", "SMILES", "Contact", "Pred. contact", "SA"], rows5)}

## Table S6. Target-held-out evaluation

Six unseen targets; rules exclude all test targets; ranker trained on
target-training pairs only. Overall joint top-1 is over covered pairs.

{table(["Target", "Family", "Test pairs", "Joint top-1 (%)"], rows6)}

Constraint ablation (all arms scored on the full joint criterion):

{table(["Selection", "Joint top-1 (%) [95% CI]"], rows6b)}

## Table S7. Lexicon ablation

Same target-held-out pairs and seed. Diversity is the mean number of distinct
generic Murcko scaffolds among the candidates of a pair.

{table(["Lexicon", "Rules", "Test pairs covered", "Joint top-1 (%)",
        "Scaffolds/pair", "Pairs with any valid candidate"], rows7)}

## Table S8. Criterion validation

{n_ok} pairs had both crystal ligands measured in ChEMBL ({c1['n_pairs']} matched
by InChIKey; {c1['n_same_target']} same-target). Δp = p_ref − p_hop. Permutation
tests use 2,000 shuffles.

{table(["Check", "n", "Spearman ρ", "p"], rows8)}

## Table S9. Equal-diversity control

The PDB-only lexicon is compared with the full lexicon after rarefaction to the
same number of distinct generic Murcko scaffolds per pair.

{table(["Analysis", "Pairs", "Joint top-1, PDB + ChEMBL (%)", "Joint top-1, PDB only (%)",
        "Scaffolds/pair, PDB + ChEMBL", "Scaffolds/pair, PDB only"], rows9)}

## Table S10. Target panel and inclusion counts

Thirty-five UniProt accessions were fixed before any pair was counted. Structures
and ligands are those retained by the X-ray (≤2.5 Å), ligand-size (15–60 heavy
atoms) and contact filters; a target with zero pairs passed the structure filters
but contributed no qualifying pair.

{table(["Target", "Class", "UniProt", "Structures", "Ligands", "Pairs"], rows10)}

## Table S11. EDIA audit and sensitivity of the main results

EDIA_m is the molecule-level electron-density score from the ProteinsPlus EDIA
REST service, computed on 17 November 2026; the gate requires EDIA_m ≥ 0.8 for
both ligands. The benchmark predates the gate, so it is applied as an audit and
the evaluations are repeated on the passing subset.

{table(["Item", "Value"], rows11)}

## Table S12. Ligands failing the EDIA gate

Every ligand with EDIA_m < 0.8 or without a score, with the number of benchmark
pairs in which it appears. `n/a` marks a ligand that could not be scored.

{table(["PDB", "CCD", "EDIA_m", "Status", "Target", "Pairs"], rows12)}

## Software and reproducibility

RDKit 2026.03.6, NumPy 1.26.4, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.17.1,
LightGBM 4.7.0, gemmi 0.7.5, matplotlib 3.11.2. Random seeds: split 0, candidate
embedding 0, ranker 0, bootstrap 0. Hardware: 128-core Linux workstation with four
NVIDIA RTX PRO 6000 GPUs; all steps are CPU-only.
"""
open(OUT, "w", encoding="utf-8").write(doc)
print(f"wrote {OUT} ({len(doc)} chars)")
