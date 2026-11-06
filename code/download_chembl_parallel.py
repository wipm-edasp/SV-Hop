"""Parallel, offset-sampled ChEMBL panel download.

The EBI API is latency-bound; one sequential target takes minutes.  We sample
three pages spread across each target's activity list (for document diversity)
and fetch targets concurrently.  Output format matches `download_chembl.py`.
"""
from __future__ import annotations

import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://www.ebi.ac.uk"
API = BASE + "/chembl/api/data"

TARGETS = [
    ("kinase", "EGFR", "CHEMBL203"),
    ("kinase", "ABL1", "CHEMBL1862"),
    ("kinase", "JAK2", "CHEMBL2971"),
    ("kinase", "CDK2", "CHEMBL301"),
    ("kinase", "BRAF", "CHEMBL5145"),
    ("gpcr", "A2A", "CHEMBL251"),
    ("gpcr", "DRD2", "CHEMBL234"),
    ("gpcr", "ADRB2", "CHEMBL210"),
    ("gpcr", "OPRM1", "CHEMBL233"),
    ("protease", "Thrombin", "CHEMBL204"),
    ("protease", "FXa", "CHEMBL244"),
    ("protease", "BACE1", "CHEMBL4822"),
    ("ion_channel", "hERG", "CHEMBL240"),
    ("nuclear_receptor", "ERa", "CHEMBL206"),
    ("nuclear_receptor", "AR", "CHEMBL1871"),
    ("nuclear_receptor", "PPARg", "CHEMBL235"),
    ("transporter", "SERT", "CHEMBL228"),
    ("transporter", "DAT", "CHEMBL238"),
    ("enzyme", "COX2", "CHEMBL230"),
    ("enzyme", "CA2", "CHEMBL205"),
]


def page(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "repairhop/0.1"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(4 * (k + 1))
                continue
            raise
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def fetch(family, name, tid, per_page=1000, samples=3):
    base = f"{API}/activity.json?target_chembl_id={tid}&standard_type=IC50&limit={per_page}"
    d0 = page(base)
    total = int(d0["page_meta"]["total_count"])
    offsets = {0}
    for i in range(1, samples):
        off = ((total * i) // samples // per_page) * per_page
        if off < total:
            offsets.add(off)
    raw = list(d0.get("activities", []))
    for off in sorted(offsets - {0}):
        raw.extend(page(base + f"&offset={off}").get("activities", []))

    seen_act, recs = set(), []
    for a in raw:
        aid = a.get("activity_id")
        if aid in seen_act:
            continue
        seen_act.add(aid)
        smi = a.get("canonical_smiles")
        p = a.get("pchembl_value")
        if not smi or p is None:
            continue
        if a.get("standard_relation") not in ("=", None, ""):
            continue
        try:
            p = float(p)
        except Exception:
            continue
        if not (3.0 <= p <= 12.0):
            continue
        recs.append(
            {
                "smiles": smi,
                "pchembl": p,
                "target": name,
                "family": family,
                "assay_type": a.get("assay_type"),
                "year": a.get("document_year"),
            }
        )
    by = {}
    for r in recs:
        by.setdefault(r["smiles"], []).append(r)
    out = []
    for smi, rs in by.items():
        r = dict(rs[0])
        r["pchembl"] = float(statistics.median([x["pchembl"] for x in rs]))
        r["n_measure"] = len(rs)
        out.append(r)
    return out, total


def main():
    outdir = "data/chembl"
    os.makedirs(outdir, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch, f, n, t): (f, n, t) for f, n, t in TARGETS}
        for fut in as_completed(futs):
            f, n, t = futs[fut]
            try:
                rows, total = fut.result()
                for r in rows:
                    r["family"] = f
                with open(os.path.join(outdir, f"{n}.jsonl"), "w", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
                results[n] = (f, len(rows), total)
                print(f"[ok] {f:18s} {n:9s} kept={len(rows):5d} total={total}", flush=True)
            except Exception as e:
                print(f"[ERR] {f}/{n}: {type(e).__name__} {e}", flush=True)

    all_rows = []
    for _, n, _ in TARGETS:
        p = os.path.join(outdir, f"{n}.jsonl")
        if os.path.exists(p):
            all_rows.extend(json.loads(l) for l in open(p, encoding="utf-8"))
    with open(os.path.join(outdir, "panel.jsonl"), "w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    fam = {}
    for r in all_rows:
        fam.setdefault(r["family"], set()).add(r["target"])
    summary = {
        "targets": len(results),
        "rows": len(all_rows),
        "unique_smiles": len({r["smiles"] for r in all_rows}),
        "families": {k: len(v) for k, v in fam.items()},
        "per_target": {n: results[n][1] for n in results},
        "totals": {n: results[n][2] for n in results},
    }
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
