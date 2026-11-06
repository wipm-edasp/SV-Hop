"""Download the PDB entries used by the SV-Hop benchmark.

Reads benchmark.json, collects pdb1/pdb2, and fetches each entry from RCSB into
--out. Existing files larger than 1 kB are kept.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="dataset/pdb_hops/benchmark.json")
    ap.add_argument("--out", default="dataset/pdb_hops/pdb")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    bench = json.load(open(args.benchmark))
    ids = sorted({p["pdb1"] for p in bench["pairs"]} | {p["pdb2"] for p in bench["pairs"]})
    os.makedirs(args.out, exist_ok=True)
    print(f"[fetch] {len(ids)} PDB entries -> {args.out}")
    ok = skip = fail = 0
    for i, pid in enumerate(ids, 1):
        dst = os.path.join(args.out, f"{pid}.pdb")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            skip += 1
            continue
        url = f"https://files.rcsb.org/download/{pid}.pdb"
        for attempt in range(args.retries):
            try:
                urllib.request.urlretrieve(url, dst)
                if os.path.getsize(dst) > 1000:
                    ok += 1
                    break
            except Exception as exc:  # noqa: BLE001
                if attempt == args.retries - 1:
                    fail += 1
                    print(f"[fail] {pid}: {exc}")
                else:
                    time.sleep(2 * (attempt + 1))
        if i % 50 == 0:
            print(f"[fetch] {i}/{len(ids)} ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"[done] ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
