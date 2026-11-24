#!/usr/bin/env python
"""Merge per-run JSON files into ledger/ledger.jsonl and print a summary.

BLINDING NOTE: for phase 'confirm' this prints run counts only, never test accuracy.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "ledger" / "runs"
recs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]
(RUNS.parent / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
print(f"{len(recs)} runs merged into ledger/ledger.jsonl\n")

g = defaultdict(list)
for r in recs:
    g[(r["phase"], r["arm"])].append(r)
print(f"{'phase':8}{'arm':18}{'n':>3}{'nan':>4}{'val%(mean±sd)':>20}{'neg-spike%L1/L2':>18}{'time/run':>10}")
for (ph, arm), rs in sorted(g.items()):
    ok = [r for r in rs if r["status"] == "ok"]
    v = [r["final_val"] for r in ok]
    vs = f"{st.mean(v):.2f}±{st.stdev(v) if len(v) > 1 else 0:.2f}" if v else "-"
    neg = f"{100*st.mean(r['stats']['neg1'] for r in ok):.1f}/{100*st.mean(r['stats']['neg2'] for r in ok):.1f}" if ok else "-"
    t = f"{st.mean(r['wall_s'] for r in rs)/60:.1f}m"
    print(f"{ph:8}{arm:18}{len(rs):>3}{len(rs)-len(ok):>4}{vs:>20}{neg:>18}{t:>10}")
    if ph == "G0" and ok:     # G0 is the replication gate: it reports test accuracy by design
        fin = [r["final_test"] for r in ok]
        best = [r["best_test_epoch"][1] for r in ok if "best_test_epoch" in r]
        print(f"        G0 test: final-epoch {st.mean(fin):.2f}%  best-epoch(authors' way) {st.mean(best):.2f}%  "
              f"(best minus final = {st.mean(best)-st.mean(fin):.2f} pp, the winner's-curse gap)")
