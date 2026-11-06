"""Mine core-replacement rules from the ChEMBL multi-family panel.

For same-target molecule pairs with different Murcko scaffolds and equal
attachment counts, record (S1, S2, k, representative molecule) as a rule.
"""
from __future__ import annotations

import collections
import json
import os
import random

import pandas as pd

from core_swap_multi import decompose_multi


def main():
    rows, seen = [], set()
    for line in open("data/chembl/panel.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["smiles"] in seen:
            continue
        seen.add(r["smiles"])
        rows.append(r)
    print(f"[load] {len(rows)} molecules")

    by_t = collections.defaultdict(list)
    for r in rows:
        d = decompose_multi(r["smiles"])
        if d is None:
            continue
        by_t[r["target"]].append({"s1": d[3], "k": d[2], "smi": r["smiles"],
                                  "family": r["family"]})
    print(f"[targets] {len(by_t)}")

    rng = random.Random(0)
    rules = {}
    for t, lst in by_t.items():
        if len(lst) > 400:
            lst = rng.sample(lst, 400)
        made = 0
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                if a["s1"] == b["s1"] or a["k"] != b["k"] or a["k"] < 1:
                    continue
                key = (min(a["s1"], b["s1"]), max(a["s1"], b["s1"]), a["k"])
                if key not in rules:
                    rules[key] = {"s1": a["s1"], "s2": b["s1"], "k": a["k"],
                                  "target": t, "family": a["family"],
                                  "rep_smi": b["smi"]}
                    made += 1
                if made >= 3000:
                    break
            if made >= 3000:
                break
    print(f"[rules] {len(rules)}")
    os.makedirs("data", exist_ok=True)
    with open("data/chembl_core_rules.jsonl", "w", encoding="utf-8") as f:
        for r in rules.values():
            f.write(json.dumps(r) + "\n")
    summary = {
        "rules": len(rules),
        "targets": len({r["target"] for r in rules.values()}),
        "families": dict(collections.Counter(r["family"] for r in rules.values())),
        "per_k": dict(collections.Counter(r["k"] for r in rules.values())),
        "source_molecules": len(rows),
    }
    json.dump(summary, open("data/chembl_core_rules_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
