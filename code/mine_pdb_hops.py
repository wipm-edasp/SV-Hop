"""Mine experimentally solved scaffold-hop pairs from the PDB.

For each target (UniProt), take high-resolution X-ray structures, extract
drug-like non-polymer ligands, compute Bemis-Murcko scaffolds and pocket-contact
residue sets, and pair ligands that (i) have different scaffolds and (ii) share
the same pocket (contact Jaccard >= 0.5).

Output: data/pdb_hops/{ligands.csv,pairs.csv,summary.json}
"""
from __future__ import annotations

import collections
import json
import os
import time
import urllib.request

import gemmi
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

TARGETS = {
    # kinases
    "EGFR": "P00533", "ABL1": "P00519", "JAK2": "O60674", "CDK2": "P24941",
    "SRC": "P12931", "MET": "P08581", "p38": "Q16539", "VEGFR2": "P35968",
    "PI3Ka": "P42336", "AKT1": "P31749",
    # proteases
    "Thrombin": "P00734", "FXa": "P00742", "Caspase3": "P42574",
    "CathepsinK": "P43235", "Trypsin": "P07477",
    # GPCRs / channels / transporters
    "A2A": "P29274", "DRD2": "P14416", "M1": "P11229", "CB1": "P21554",
    "SERT": "P31645", "DAT": "Q01959", "hERG": "Q12809",
    # nuclear receptors
    "ERa": "P03372", "PPARg": "P37231", "AR": "P10275", "VDR": "P11473",
    "GR": "P04150",
    # enzymes / other
    "COX2": "P35354", "CA2": "P00918", "HSP90": "P07900", "BCL2": "P10415",
    "DHFR": "P00374", "PDE5": "O76074", "ACE": "P12821", "HMGCR": "P04035",
}
FAMILIES = {
    "kinase": ["EGFR", "ABL1", "JAK2", "CDK2", "SRC", "MET", "p38", "VEGFR2", "PI3Ka", "AKT1"],
    "protease": ["Thrombin", "FXa", "Caspase3", "CathepsinK", "Trypsin"],
    "gpcr": ["A2A", "DRD2", "M1", "CB1"],
    "transporter": ["SERT", "DAT"],
    "ion_channel": ["hERG"],
    "nuclear_receptor": ["ERa", "PPARg", "AR", "VDR", "GR"],
    "enzyme_other": ["COX2", "CA2", "HSP90", "BCL2", "DHFR", "PDE5", "ACE", "HMGCR"],
}
EXCLUDE = {
    "HOH", "WAT", "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "DMS", "ACT",
    "FMT", "ACE", "CL", "NA", "MG", "CA", "ZN", "MN", "FE", "CU", "NI", "CD",
    "HG", "IOD", "BR", "NO3", "MES", "TRS", "HEP", "EPE", "BOG", "LDA", "CIT",
    "TLA", "MPD", "P6G", "1PE", "2PE", "PGE", "SIN", "IMD", "BME", "DTT",
}
ROOT = "data/pdb_hops"
PDB_DIR = f"{ROOT}/pdb"
CCD_CACHE = f"{ROOT}/ccd_cache.json"


def get(url, tries=3, timeout=30):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 + 2 * k)


def uniprot_pdbs(acc):
    path = f"{ROOT}/uniprot_{acc}.json"
    if os.path.exists(path):
        return json.load(open(path))
    d = json.loads(get(f"https://rest.uniprot.org/uniprotkb/{acc}.json"))
    out = []
    for x in d.get("uniProtKBCrossReferences", []):
        if x.get("database") != "PDB":
            continue
        props = {p["key"]: p["value"] for p in x.get("properties", [])}
        if "Resolution" not in props:
            continue
        try:
            res = float(props["Resolution"].split()[0])
        except Exception:
            continue
        if res > 2.5:
            continue
        if props.get("Method", "").lower().startswith("x"):
            out.append({"pdb": x["id"], "res": res})
    out.sort(key=lambda z: z["res"])
    out = out[:40]
    json.dump(out, open(path, "w"))
    return out


def ccd_smiles(ccd, cache):
    if ccd in cache:
        return cache[ccd]
    try:
        d = json.loads(get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd}"))
        desc = d.get("rcsb_chem_comp_descriptor") or {}
        smi = desc.get("SMILES_stereo") or desc.get("SMILES")
    except Exception:
        smi = None
    cache[ccd] = smi
    return smi


def scaffold(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    s = MurckoScaffold.GetScaffoldForMol(m)
    if s is None or s.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(s)


def parse_complex(path, target, pdbid, cache):
    try:
        st = gemmi.read_structure(path)
        st.setup_entities()
    except Exception:
        return []
    model = st[0]
    try:
        ns = gemmi.NeighborSearch(model, st.cell, 5.0).populate()
    except Exception:
        return []
    seen = set()
    out = []
    for ch in model:
        for res in ch:
            if res.het_flag != "H" or res.name in EXCLUDE:
                continue
            heavy = [a for a in res if a.element.name != "H"]
            if not (15 <= len(heavy) <= 60):
                continue
            if res.name in seen:
                continue
            seen.add(res.name)
            smi = ccd_smiles(res.name, cache)
            scaf = scaffold(smi)
            if scaf is None:
                continue
            contacts = set()
            for a in heavy:
                for m in ns.find_atoms(a.pos, "\0", radius=5.0):
                    r = model[m.chain_idx][m.residue_idx]
                    if r.het_flag != "H":
                        contacts.add((r.seqid.num, r.name))
            if len(contacts) < 5:
                continue
            out.append({
                "target": target, "pdb": pdbid, "ccd": res.name,
                "smiles": smi, "scaffold": scaf, "n_heavy": len(heavy),
                "contacts": contacts,
            })
    return out


def main():
    os.makedirs(PDB_DIR, exist_ok=True)
    cache = json.load(open(CCD_CACHE)) if os.path.exists(CCD_CACHE) else {}
    all_lig = []
    for target, acc in TARGETS.items():
        pdbs = uniprot_pdbs(acc)
        print(f"[{target}] {len(pdbs)} X-ray structures <=2.5A", flush=True)
        for rec in pdbs:
            pdbid = rec["pdb"].upper()
            path = f"{PDB_DIR}/{pdbid}.pdb"
            if not os.path.exists(path):
                try:
                    data = get(f"https://files.rcsb.org/download/{pdbid}.pdb", timeout=60)
                    open(path, "wb").write(data)
                except Exception:
                    continue
            all_lig.extend(parse_complex(path, target, pdbid, cache))
        json.dump(cache, open(CCD_CACHE, "w"))
        print(f"[{target}] ligands so far {sum(1 for x in all_lig if x['target']==target)}", flush=True)

    for x in all_lig:
        x["contacts"] = ";".join(f"{c[0]}:{c[1]}" for c in sorted(x["contacts"]))
    with open(f"{ROOT}/ligands.csv", "w", encoding="utf-8") as f:
        f.write("target,pdb,ccd,scaffold,n_heavy,smiles,contacts\n")
        for x in all_lig:
            f.write(f"{x['target']},{x['pdb']},{x['ccd']},{x['scaffold']},{x['n_heavy']},{x['smiles']},{x['contacts']}\n")

    by_t = collections.defaultdict(list)
    for x in all_lig:
        by_t[x["target"]].append(x)
    pairs = []
    for t, ligs in by_t.items():
        uniq = {}
        for x in ligs:
            uniq.setdefault((x["pdb"], x["ccd"]), x)
        ligs = list(uniq.values())
        csets = [set(x["contacts"].split(";")) for x in ligs]
        for i in range(len(ligs)):
            for j in range(i + 1, len(ligs)):
                if ligs[i]["scaffold"] == ligs[j]["scaffold"]:
                    continue
                inter = len(csets[i] & csets[j])
                union = len(csets[i] | csets[j])
                jac = inter / union if union else 0.0
                if jac < 0.5:
                    continue
                pairs.append({
                    "target": t, "pdb1": ligs[i]["pdb"], "ccd1": ligs[i]["ccd"],
                    "pdb2": ligs[j]["pdb"], "ccd2": ligs[j]["ccd"],
                    "scaf1": ligs[i]["scaffold"], "scaf2": ligs[j]["scaffold"],
                    "jaccard": jac, "n1": ligs[i]["n_heavy"], "n2": ligs[j]["n_heavy"],
                })
    pairs.sort(key=lambda x: (-x["jaccard"], x["target"]))
    with open(f"{ROOT}/pairs.csv", "w", encoding="utf-8") as f:
        f.write("target,pdb1,ccd1,pdb2,ccd2,scaf1,scaf2,jaccard,n1,n2\n")
        for p in pairs:
            f.write(f"{p['target']},{p['pdb1']},{p['ccd1']},{p['pdb2']},{p['ccd2']},{p['scaf1']},{p['scaf2']},{p['jaccard']:.3f},{p['n1']},{p['n2']}\n")

    per_t = collections.Counter(p["target"] for p in pairs)
    summary = {
        "targets": len(TARGETS),
        "structures": len(os.listdir(PDB_DIR)),
        "ligands": len(all_lig),
        "pairs": len(pairs),
        "targets_with_pairs": len(per_t),
        "per_target": dict(per_t),
        "unique_scaffolds": len({x["scaffold"] for x in all_lig}),
    }
    json.dump(summary, open(f"{ROOT}/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
