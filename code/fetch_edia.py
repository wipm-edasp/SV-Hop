"""Fetch per-ligand EDIA_m scores from the ProteinsPlus EDIA REST service.

Two phases so that ~520 entries finish in minutes rather than hours:
  1. submit every entry (rate-limited to stay under the 30 jobs/minute limit),
     storing the returned job location;
  2. poll the submitted jobs round-robin, download each structure-score CSV as
     soon as it is ready.

Results are cached per entry, so the script can be re-run after an interruption.

Input : data/pdb_hops/benchmark.json
Output: data/edia/raw/<PDB>.csv              structure scores (residues and molecules)
        data/edia/jobs.json                  job locations for pending entries
        data/edia/edia_status.csv            one row per PDB entry
        data/edia/edia_ligands.csv           one row per (PDB, CCD) used by the benchmark

The shipped data/edia/edia_ligands.csv was produced with this script on
17 November 2026 (503 entries scored, 19 without deposited structure factors).
Re-running is only needed if the entry set changes; each submission costs a
round trip to Hamburg and the service computes the map on demand.

ProteinsPlus rate limit: 30 jobs/minute (HTTP 429 when exceeded).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request

API = "https://proteins.plus/api/edia_rest"
DEFAULT_BENCH = "data/pdb_hops/benchmark.json"
DEFAULT_OUT = "data/edia"

MIN_POST_INTERVAL = 3.2   # seconds -> ~27 jobs/min, under the 30/min limit
POLL_GAP = 0.9            # seconds between successive GETs
POLL_TIMEOUT = 900.0      # wall clock allowed for the polling phase


def request(url, data=None, timeout=60):
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.getcode(), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:                      # network hiccup
        return -1, str(e)


def load_status(path):
    d = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path, encoding="utf-8")):
            d[r["pdb"]] = r
    return d


def save_status(path, d):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb", "status", "message"])
        w.writeheader()
        for k in sorted(d):
            w.writerow(d[k])


def parse_molecules(txt):
    """CCD -> best (max) EDIA_m over all copies, plus the copy count."""
    vals = {}
    for r in csv.DictReader(txt.splitlines()):
        if r.get("Structure specifier") != "m":
            continue
        name = (r.get("Name") or "").strip()
        try:
            v = float(r.get("EDIAm"))
        except (TypeError, ValueError):
            continue
        vals.setdefault(name, []).append(v)
    return {k: {"ediam": max(v), "n_copies": len(v)} for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser(description="Fetch EDIA scores for benchmark ligands.")
    ap.add_argument("--benchmark", default=DEFAULT_BENCH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    raw = os.path.join(args.out, "raw")
    jobs_path = os.path.join(args.out, "jobs.json")
    status_path = os.path.join(args.out, "edia_status.csv")
    os.makedirs(raw, exist_ok=True)

    bench = json.load(open(args.benchmark, encoding="utf-8"))
    ligs = {}
    for p in bench["pairs"]:
        ligs.setdefault(p["pdb1"].upper(), set()).add(p["ccd1"])
        ligs.setdefault(p["pdb2"].upper(), set()).add(p["ccd2"])
    pdbs = sorted(ligs)
    print(f"{len(pdbs)} PDB entries, {sum(len(v) for v in ligs.values())} ligand instances",
          flush=True)

    status = load_status(status_path)
    jobs = json.load(open(jobs_path, encoding="utf-8")) if os.path.exists(jobs_path) else {}

    def finished(pdb):
        return os.path.exists(os.path.join(raw, f"{pdb}.csv")) or status.get(pdb, {}).get("status") in {
            "no_density", "submit_failed"}

    # ---------- phase 1: submit ----------
    todo = [p for p in pdbs if not finished(p) and p not in jobs]
    print(f"phase 1: submitting {len(todo)} entries", flush=True)
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    state = {"last": 0.0}

    def submit_one(pdb):
        with lock:
            wait = MIN_POST_INTERVAL - (time.time() - state["last"])
            if wait > 0:
                time.sleep(wait)
            state["last"] = time.time()
        code, body = request(API, {"edia": {"pdbCode": pdb.lower()}})
        tries = 0
        while code == 429 and tries < 4:
            tries += 1
            time.sleep(30)
            code, body = request(API, {"edia": {"pdbCode": pdb.lower()}})
        try:
            j = json.loads(body)
        except Exception:
            return pdb, {"status": "submit_failed", "message": body[:120]}, None
        msg = j.get("message", "")
        loc = j.get("location")
        if code in (200, 202) and loc:
            return pdb, {"status": "queued", "message": msg[:120]}, loc
        if "No electron density" in msg:
            return pdb, {"status": "no_density", "message": msg[:120]}, None
        return pdb, {"status": f"submit_failed_{code}", "message": msg[:120]}, None

    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, (pdb, st, loc) in enumerate(ex.map(submit_one, todo), 1):
            status[pdb] = {"pdb": pdb, "status": st["status"], "message": st["message"]}
            if loc:
                jobs[pdb] = loc
            if i % 40 == 0:
                json.dump(jobs, open(jobs_path, "w"), indent=2)
                save_status(status_path, status)
                print(f"  submitted {i}/{len(todo)}", flush=True)
    json.dump(jobs, open(jobs_path, "w"), indent=2)
    save_status(status_path, status)
    print(f"phase 1 done: {len(jobs)} jobs queued, "
          f"{sum(1 for v in status.values() if v['status'] == 'no_density')} without density",
          flush=True)

    # ---------- phase 2: poll ----------
    pending = [p for p in sorted(jobs) if not os.path.exists(os.path.join(raw, f"{p}.csv"))]
    print(f"phase 2: polling {len(pending)} jobs", flush=True)
    t0 = time.time()

    def poll_one(pdb):
        code, body = request(jobs[pdb], timeout=30)
        if code == 429:
            return pdb, "retry", ""
        if code == 202:
            return pdb, "pending", ""
        if code != 200:
            if "No electron density" in body:
                return pdb, "no_density", body[:120]
            return pdb, f"poll_failed_{code}", body[:120]
        j = json.loads(body)
        url = j.get("structure_scores")
        if not url:
            return pdb, "no_scores", str(j)[:120]
        c, txt = request(url, timeout=60)
        if c != 200 or not txt.strip():
            return pdb, f"download_failed_{c}", ""
        with open(os.path.join(raw, f"{pdb}.csv"), "w", encoding="utf-8") as f:
            f.write(txt)
        return pdb, "ok", ""

    while pending and time.time() - t0 < POLL_TIMEOUT:
        still = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for pdb, st, msg in ex.map(poll_one, pending):
                if st in ("pending", "retry"):
                    still.append(pdb)
                elif st == "ok":
                    status[pdb] = {"pdb": pdb, "status": "ok", "message": ""}
                else:
                    status[pdb] = {"pdb": pdb, "status": st, "message": msg}
        pending = still
        save_status(status_path, status)
        n_ok = sum(1 for v in status.values() if v["status"] == "ok")
        print(f"  polled: ok={n_ok} pending={len(pending)}", flush=True)
        if pending:
            time.sleep(max(POLL_GAP, 2.0))

    # ---------- aggregate ----------
    # an entry is "ok" when its score table exists, whatever a stale status file says
    out_rows, missing = [], []
    for pdb in pdbs:
        path = os.path.join(raw, f"{pdb}.csv")
        have = os.path.exists(path)
        st = "ok" if have else status.get(pdb, {}).get("status", "missing")
        status[pdb] = {"pdb": pdb, "status": st, "message": status.get(pdb, {}).get("message", "")}
        mols = parse_molecules(open(path, encoding="utf-8").read()) if have else {}
        for ccd in sorted(ligs[pdb]):
            if ccd in mols:
                out_rows.append({"pdb": pdb, "ccd": ccd,
                                 "ediam": f"{mols[ccd]['ediam']:.3f}",
                                 "n_copies": mols[ccd]["n_copies"], "status": st})
            else:
                missing.append((pdb, ccd))
                out_rows.append({"pdb": pdb, "ccd": ccd, "ediam": "", "n_copies": 0,
                                 "status": st})

    save_status(status_path, status)
    lig_path = os.path.join(args.out, "edia_ligands.csv")
    with open(lig_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb", "ccd", "ediam", "n_copies", "status"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"ligand rows {len(out_rows)}, without EDIA_m {len(missing)}", flush=True)
    print("missing sample:", missing[:15], flush=True)


if __name__ == "__main__":
    main()
