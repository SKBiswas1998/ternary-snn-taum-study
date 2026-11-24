#!/usr/bin/env python
"""Generate job CSVs. Reserved seed ranges (never reuse across purposes):
   explore 0-4 | G0 0-4 | G3 100-109 | tune 200-201 | confirm: CSPRNG draw at freeze (not here).
Usage: python runner/make_jobs.py g0|g2|g3|explore
"""
import csv
import sys
from pathlib import Path

JOBS = Path(__file__).resolve().parent.parent / "jobs"
HEAD = ["phase", "arm", "seed", "tau_ps", "theta_pos", "theta_neg", "lr", "epochs", "shuffle_labels"]
TAUS = [50, 100, 200, 350, 503.8, 800, 1200]


def write(name, rows):
    JOBS.mkdir(exist_ok=True)
    p = JOBS / f"{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEAD)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {p} ({len(rows)} jobs)")


def row(phase, arm, seed, tau=503.8, tp=0.8, tn=0.9, lr=1e-3, ep=10, sh=0):
    return dict(phase=phase, arm=arm, seed=seed, tau_ps=tau, theta_pos=tp, theta_neg=tn, lr=lr, epochs=ep, shuffle_labels=sh)


what = sys.argv[1] if len(sys.argv) > 1 else ""
if what == "g0":      # replication: committed build (theta_neg=1.6) vs README build (0.9), 5 seeds each
    write("g0", [row("G0", b, s, tn=tn) for b, tn in (("committed", 1.6), ("readme", 0.9)) for s in range(5)])
elif what == "g2":    # negative control: shuffled train labels, 3 tau arms x 2 seeds
    write("g2", [row("G2", f"shuf_tau{t}", s, tau=t, sh=1) for t in (50, 503.8, 1200) for s in range(2)])
elif what == "g3":    # seed noise on the reference arm
    write("g3", [row("G3", "ref", s) for s in range(100, 110)])
elif what == "explore":
    write("explore", [row("explore", f"tau{t}", s, tau=t) for t in TAUS for s in range(5)])
else:
    sys.exit(__doc__)
