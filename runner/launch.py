#!/usr/bin/env python
"""Run a job CSV with N parallel single-thread worker processes.

    python runner/launch.py jobs/g0.csv --workers 3
Re-running is safe: run.py skips every (arm, seed) that already has a result file.
Extra flags after '--' are passed to run.py (e.g. -- --upstream /path/thmtj.py).
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLS = {"phase": "--phase", "arm": "--arm", "seed": "--seed", "tau_ps": "--tau-ps",
        "theta_pos": "--theta-pos", "theta_neg": "--theta-neg", "lr": "--lr", "epochs": "--epochs"}


def one(row, extra, log_dir):
    cmd = [sys.executable, str(HERE / "run.py")]
    for k, flag in COLS.items():
        if row.get(k, "") != "":
            cmd += [flag, row[k]]
    if row.get("shuffle_labels") == "1":
        cmd.append("--shuffle-labels")
    cmd += extra
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    name = f"{row['phase']}__{row['arm']}__s{row['seed']}"
    t = time.time()
    with open(log_dir / f"{name}.log", "a") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    print(f"{time.strftime('%H:%M:%S')} {'ok ' if rc == 0 else 'FAIL'} {name} ({time.time()-t:.0f}s)", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.jobs, newline="")))
    log_dir = HERE.parent / "ledger" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} jobs, {a.workers} workers")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rcs = list(ex.map(lambda r: one(r, a.rest, log_dir), rows))
    print(f"finished: {rcs.count(0)} ok, {len(rcs) - rcs.count(0)} failed (failed runs can simply be re-launched)")


if __name__ == "__main__":
    main()
