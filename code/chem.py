"""Chemical utilities for RepairHop.

Single-attachment Bemis-Murcko decomposition:

    molecule M  ->  core T (ring systems + linkers) + decoration R

Both T and R carry one dummy atom `[1*]` at the attachment point, so the
product is reconstructed by joining the two dummy atoms.  We keep only
molecules whose decomposition has exactly one attachment point, which makes
the repair target unambiguous and the join operation exact.
"""
from __future__ import annotations

import os
import sys

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors, RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_SA_READY = False


def _ensure_sa():
    global _SA_READY
    if not _SA_READY:
        sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_dir not in sys.path:
            sys.path.append(sa_dir)
        _SA_READY = True


def largest_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    return max(frags, key=lambda m: m.GetNumAtoms())


def canon(mol_or_smi) -> str | None:
    mol = Chem.MolFromSmiles(mol_or_smi) if isinstance(mol_or_smi, str) else mol_or_smi
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def murcko_smiles(mol: Chem.Mol) -> str | None:
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core is None or core.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(core)


def _attachment_bonds(mol: Chem.Mol, core_set: set[int]):
    out = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (i in core_set) != (j in core_set):
            out.append((i if i in core_set else j, b.GetBondType()))
    return out


def _one_dummy(mol: Chem.Mol) -> Chem.Mol | None:
    dd = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dd) != 1:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(dd[0]).SetIsotope(1)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m


def decompose(smiles: str, min_heavy: int = 8) -> dict | None:
    """Return {smi, core, core_d, dec, dec_d, dec_n, core_n, att_bt} or None."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    mol = largest_fragment(mol)
    if mol is None or mol.GetNumAtoms() < min_heavy:
        return None

    core0 = MurckoScaffold.GetScaffoldForMol(mol)
    if core0 is None or core0.GetNumAtoms() == 0:
        return None
    # make core atom indices contiguous so match position == core atom index
    core = Chem.RenumberAtoms(core0, [a.GetIdx() for a in core0.GetAtoms()])
    match = mol.GetSubstructMatch(core)
    if not match or len(match) != core.GetNumAtoms():
        return None

    core_set = set(match)
    att = _attachment_bonds(mol, core_set)
    if len(att) != 1:
        return None
    att_atom, att_bt = att[0]
    pos = list(match).index(att_atom)

    dec = Chem.ReplaceCore(mol, core)
    if dec is None:
        return None
    frags = Chem.GetMolFrags(dec, asMols=True, sanitizeFrags=True)
    if len(frags) != 1:
        return None
    dec_d = _one_dummy(frags[0])
    if dec_d is None:
        return None

    rw = Chem.RWMol(core)
    d = rw.AddAtom(Chem.Atom(0))
    rw.GetAtomWithIdx(d).SetIsotope(1)
    rw.AddBond(pos, d, att_bt)
    core_d = rw.GetMol()
    try:
        Chem.SanitizeMol(core_d)
    except Exception:
        return None

    return {
        "smi": Chem.MolToSmiles(mol),
        "core": Chem.MolToSmiles(core),
        "core_d": Chem.MolToSmiles(core_d),
        "dec": Chem.MolToSmiles(frags[0]),
        "dec_d": Chem.MolToSmiles(dec_d),
        "dec_n": frags[0].GetNumAtoms(),
        "core_n": core.GetNumAtoms(),
        "att_bt": str(att_bt),
    }


def _dummy_pair(a: Chem.Mol, b: Chem.Mol):
    da = [x for x in a.GetAtoms() if x.GetAtomicNum() == 0]
    db = [x for x in b.GetAtoms() if x.GetAtomicNum() == 0]
    if len(da) != 1 or len(db) != 1:
        return None
    return da[0].GetIdx(), db[0].GetIdx()


def join(core_d: Chem.Mol, dec_d: Chem.Mol) -> Chem.Mol | None:
    """Join a core and a decoration at their single dummy atoms."""
    if core_d is None or dec_d is None:
        return None
    pair = _dummy_pair(core_d, dec_d)
    if pair is None:
        return None
    i1, i2 = pair
    if len(Chem.GetMolFrags(core_d)) != 1 or len(Chem.GetMolFrags(dec_d)) != 1:
        return None
    combo = Chem.CombineMols(core_d, dec_d)
    rw = Chem.RWMol(combo)
    off = core_d.GetNumAtoms()
    i2 = i2 + off
    nb1 = rw.GetAtomWithIdx(i1).GetNeighbors()
    nb2 = rw.GetAtomWithIdx(i2).GetNeighbors()
    if len(nb1) == 0 or len(nb2) == 0:
        return None
    n1 = nb1[0].GetIdx()
    n2 = nb2[0].GetIdx()
    bt = rw.GetBondBetweenAtoms(i1, n1).GetBondType()
    rem = sorted([i1, i2], reverse=True)
    n1n = n1 - sum(1 for r in rem if r < n1)
    n2n = n2 - sum(1 for r in rem if r < n2)
    for r in rem:
        rw.RemoveAtom(r)
    rw.AddBond(n1n, n2n, bt)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def join_smiles(core_d_smi: str, dec_smi: str) -> str | None:
    a = Chem.MolFromSmiles(core_d_smi) if core_d_smi else None
    b = Chem.MolFromSmiles(dec_smi) if dec_smi else None
    if a is None or b is None:
        return None
    m = join(a, b)
    return Chem.MolToSmiles(m) if m is not None else None


def morgan(mol: Chem.Mol):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def fp_of(smi: str):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return morgan(m) if m is not None else None


def tanimoto(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(a, b))


def qed(smi: str) -> float:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return float("nan")
    try:
        return float(QED.qed(m))
    except Exception:
        return float("nan")


def sa_score(smi: str) -> float:
    _ensure_sa()
    import sascorer  # from RDKit Contrib

    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return float("nan")
    try:
        return float(sascorer.calculateScore(m))
    except Exception:
        return float("nan")


def props(smi: str) -> dict | None:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    try:
        return {
            "logp": float(Descriptors.MolLogP(m)),
            "tpsa": float(Descriptors.TPSA(m)),
            "hbd": int(Lipinski.NumHDonors(m)),
            "hba": int(Lipinski.NumHAcceptors(m)),
            "mw": float(Descriptors.MolWt(m)),
        }
    except Exception:
        return None


def bioisostere_pass(a: str, b: str) -> int:
    pa, pb = props(a), props(b)
    if pa is None or pb is None:
        return 0
    return int(
        abs(pb["logp"] - pa["logp"]) <= 1.0
        and abs(pb["tpsa"] - pa["tpsa"]) <= 20.0
        and abs(pb["hbd"] - pa["hbd"]) <= 1
        and abs(pb["hba"] - pa["hba"]) <= 1
        and abs(pb["mw"] - pa["mw"]) <= 80.0
    )


def num_rings(smi: str) -> int:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return rdMolDescriptors.CalcNumRings(m) if m is not None else 0
