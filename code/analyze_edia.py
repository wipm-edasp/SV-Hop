"""Merge EDIA ligand-quality scores into the SV-Hop benchmark and quantify impact.

Inputs
  data/pdb_hops/benchmark.json          2,886 pairs (analysed set)
  data/edia/edia_ligands.csv            per (PDB, CCD) EDIA_m
  results/svhop_full/selected.csv       per test pair, constrained-arm top-1
  results/c1_validation/pairs.csv       criterion-validation pairs

Outputs
  results/edia_qc/edia_pairs.csv        per pair: scores, pass flag, splits
  results/edia_qc/edia_summary.json     counts and sensitivity
  results/edia_qc/edia_failures.csv     ligands below threshold or unscored

The benchmark and its evaluation predate the gate, so the gate is applied here as
an audit: the analysed set stays intact, and the main evaluations are repeated on
the passing subset (Tables S11 and S12 of the Supporting Information).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics

DEFAULT_BENCH = "data/pdb_hops/benchmark.json"
DEFAULT_EDIA = "data/edia/edia_ligands.csv"
DEFAULT_SELECTED = "results/svhop_full/selected.csv"
DEFAULT_C1 = "results/c1_validation/pairs.csv"
DEFAULT_OUT = "results/edia_qc"

T = 0.8
JOINT_CONTACT = 0.6


def load_edia(path):
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d[(r["pdb"].upper(), r["ccd"])] = (float(r["ediam"]) if r["ediam"] else None,
                                           r.get("status", ""))
    return d


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pct(x, n):
    return 100.0 * x / n if n else float("nan")


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description="EDIA audit of the SV-Hop benchmark.")
    ap.add_argument("--benchmark", default=DEFAULT_BENCH)
    ap.add_argument("--edia", default=DEFAULT_EDIA)
    ap.add_argument("--selected", default=DEFAULT_SELECTED)
    ap.add_argument("--c1", default=DEFAULT_C1)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=T)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bench = json.load(open(args.benchmark, encoding="utf-8"))
    edia = load_edia(args.edia)
    pairs = bench["pairs"]
    tsplit = {i: s for s, ids in bench["splits"]["target"].items() for i in ids}
    ssplit = {i: s for s, ids in bench["splits"]["scaffold"].items() for i in ids}

    rows = []
    for p in pairs:
        e1, st1 = edia.get((p["pdb1"].upper(), p["ccd1"]), (None, "missing"))
        e2, st2 = edia.get((p["pdb2"].upper(), p["ccd2"]), (None, "missing"))
        both = e1 is not None and e2 is not None
        mn = min(e1, e2) if both else None
        rows.append({**p, "ediam1": e1, "ediam2": e2, "ediam_min": mn,
                     "pass": int(both and mn >= args.threshold),
                     "pass_tier2": int(both and mn >= 0.4),
                     "scored": int(both), "tsplit": tsplit[p["id"]], "ssplit": ssplit[p["id"]]})

    with open(os.path.join(args.out, "edia_pairs.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "target", "family", "pdb1", "ccd1", "pdb2", "ccd2",
                                          "jaccard", "ediam1", "ediam2", "ediam_min", "pass",
                                          "scored", "pass_tier2", "tsplit", "ssplit"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in w.fieldnames})

    lig_values = [v for v, _ in edia.values() if v is not None]
    buckets = {
        "well_supported_ge_0.8": sum(1 for v in lig_values if v >= 0.8),
        "moderate_0.4_0.8": sum(1 for v in lig_values if 0.4 <= v < 0.8),
        "poor_lt_0.4": sum(1 for v in lig_values if v < 0.4),
        "unscored": sum(1 for v, _ in edia.values() if v is None),
    }

    n = len(rows)
    n_scored = sum(r["scored"] for r in rows)
    n_pass = sum(r["pass"] for r in rows)
    n_pass2 = sum(r["pass_tier2"] for r in rows)
    fam = {}
    for r in rows:
        fam.setdefault(r["family"], [0, 0, 0])
        fam[r["family"]][0] += 1
        fam[r["family"]][1] += r["scored"]
        fam[r["family"]][2] += r["pass"]
    splits = {}
    for key in ("ssplit", "tsplit"):
        splits[key] = {}
        for s in ("train", "val", "test"):
            sub = [r for r in rows if r[key] == s]
            splits[key][s] = {"pairs": len(sub), "pass": sum(r["pass"] for r in sub)}

    # scaffold-held-out evaluation subset
    sel = list(csv.DictReader(open(args.selected, encoding="utf-8")))
    sel_ids = {int(r["pair"]) for r in sel}
    pair_by_id = {r["id"]: r for r in rows}
    joint_all, joint_pass, joint_pass2 = [], [], []
    for r in sel:
        pid = int(r["pair"])
        ok = (int(r["conserved"]) == 1 and int(r["sa_ok"]) == 1 and int(r["novel"]) == 1
              and float(r["contact_recall"]) >= JOINT_CONTACT)
        joint_all.append(int(ok))
        if pair_by_id[pid]["pass"]:
            joint_pass.append(int(ok))
        if pair_by_id[pid]["pass_tier2"]:
            joint_pass2.append(int(ok))
    n_sel_pass = sum(1 for pid in sel_ids if pair_by_id[pid]["pass"])
    n_sel_pass2 = sum(1 for pid in sel_ids if pair_by_id[pid]["pass_tier2"])

    # criterion validation (the paper's n = 50 uses same-target pairs only)
    c1 = list(csv.DictReader(open(args.c1, encoding="utf-8")))
    c1_st = [r for r in c1 if r["same_target"] == "True"]
    c1_pass = [r for r in c1 if pair_by_id.get(int(r["pair"]), {}).get("pass")]
    c1_st_pass = [r for r in c1_st if pair_by_id.get(int(r["pair"]), {}).get("pass")]

    def block(sub, name):
        xs = [float(r["contact"]) for r in sub]
        if not xs:
            return {"name": name, "n": 0}
        dp = [float(r["dp"]) for r in sub]
        ph = [float(r["pharm_2A"]) for r in sub]
        return {"name": name, "n": len(sub),
                "contact_mean": mean(xs),
                "contact_median": statistics.median(xs),
                "n_contact_ge_0.6": sum(1 for x in xs if x >= 0.6),
                "rho_dp": spearman(xs, dp), "rho_pharm": spearman(xs, ph)}

    summary = {
        "ligand_instances": len(edia),
        "ligand_buckets": buckets,
        "ligand_pass_rate": pct(buckets["well_supported_ge_0.8"], len(edia)),
        "pair_total": n,
        "pair_scored": n_scored,
        "pair_pass": n_pass,
        "pair_pass_rate_of_all": pct(n_pass, n),
        "pair_pass_rate_of_scored": pct(n_pass, n_scored),
        "pair_pass_tier2": n_pass2,
        "pair_pass_tier2_rate": pct(n_pass2, n),
        "per_family": {k: {"pairs": v[0], "scored": v[1], "pass": v[2]} for k, v in fam.items()},
        "splits": splits,
        "scaffold_eval": {"pairs": len(sel), "pass": n_sel_pass,
                          "joint_all": pct(sum(joint_all), len(joint_all)),
                          "joint_pass": pct(sum(joint_pass), len(joint_pass)),
                          "selected_contact_mean_all": mean([float(r["contact_recall"]) for r in sel]),
                          "pass_tier2": n_sel_pass2,
                          "joint_tier2": pct(sum(joint_pass2), len(joint_pass2)),
                          "selected_contact_mean_pass": mean(
                              [float(r["contact_recall"]) for r in sel
                               if pair_by_id[int(r["pair"])]["pass"]])},
        "criterion_validation": {"all": block(c1, "all"),
                                 "same_target": block(c1_st, "same_target_50"),
                                 "same_target_pass": block(c1_st_pass, "same_target_pass"),
                                 "edia_pass": block(c1_pass, "edia_pass")},
        "threshold": args.threshold,
    }
    json.dump(summary, open(os.path.join(args.out, "edia_summary.json"), "w"), indent=2)

    # failure list at ligand level
    fails = []
    for p in pairs:
        for pdb, ccd in ((p["pdb1"], p["ccd1"]), (p["pdb2"], p["ccd2"])):
            v, st = edia.get((pdb.upper(), ccd), (None, "missing"))
            if v is None or v < args.threshold:
                fails.append({"pdb": pdb, "ccd": ccd,
                              "ediam": "" if v is None else f"{v:.3f}",
                              "status": st, "target": p["target"], "pair": p["id"]})
    seen, uniq = set(), []
    for r in fails:
        k = (r["pdb"], r["ccd"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    with open(os.path.join(args.out, "edia_failures.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb", "ccd", "ediam", "status", "target", "pair"])
        w.writeheader()
        w.writerows(uniq)

    print(json.dumps({k: summary[k] for k in ("ligand_buckets", "pair_total", "pair_scored",
                                              "pair_pass", "pair_pass_rate_of_all",
                                              "pair_pass_rate_of_scored", "pair_pass_tier2",
                                              "pair_pass_tier2_rate")}, indent=2))
    print(f"wrote {args.out}/edia_pairs.csv, edia_summary.json, edia_failures.csv "
          f"({len(uniq)} failing ligands)")


if __name__ == "__main__":
    main()
