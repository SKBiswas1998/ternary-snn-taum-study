#!/usr/bin/env python
"""Live progress for the extra-seed set (jobs\\more.csv: seeds 5-11 on tau 350/503.8/800/2000 ps). Read-only; safe to run in any tab.

    python runner\\watch_more.py --start "2026-11-25 21:50"
    python runner\\watch_more.py --once
Ctrl+C only stops this display, never the runs.
"""
import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS, RUNS = ROOT / "ledger" / "logs", ROOT / "ledger" / "runs"

STAGES = [
    ("More seeds (5-11), 12 seeds total", "explore", ["tau350", "tau503.8", "tau800", "tau2000"], list(range(5, 12))),
]
EP = re.compile(r"ep (\d+)/10 loss ([\d.]+) train ([\d.]+)% val ([\d.]+)%\s+\((\d+)s\)")
FULL, EMPTY, SQ, SQO, RUN = "█", "░", "■", "□", "▶"
G, C, Y, D, R = "\033[92m", "\033[96m", "\033[93m", "\033[90m", "\033[0m"


def bar(f, w=30):
    k = int(round(max(0.0, min(1.0, f)) * w))
    return FULL * k + EMPTY * (w - k)


def hm(h):
    h = max(h, 0.0)
    return f"{int(h)}h {int((h - int(h)) * 60):02d}m"


def snapshot(now, window_min=12):
    fin = {p.stem.split("__lr")[0] for p in RUNS.glob("*.json")}
    run = {}
    for f in LOGS.glob("*.log"):
        if not re.match(r"(explore|G2)__", f.stem) or f.stem in fin:
            continue
        mt = datetime.fromtimestamp(f.stat().st_mtime)
        if mt < now - timedelta(minutes=window_min):
            continue
        last = None
        for line in f.read_text(errors="ignore").splitlines():
            m = EP.search(line)
            if m:
                last = m
        if last:
            ep, val, es = int(last.group(1)), last.group(4), int(last.group(5)) / int(last.group(1))
        else:
            ep, val, es = 0, "-", 290.0
        frac = (ep + min(0.98, (now - mt).total_seconds() / es)) / 10
        run[f.stem] = dict(ep=ep, val=val, frac=frac)
    return fin, run


def render(t0, now):
    fin, run = snapshot(now)
    stage_info, total, work = [], 0, 0.0
    for name, ph, arms, seeds in STAGES:
        n, done, w, rows = len(arms) * len(seeds), 0, 0.0, []
        for a in arms:
            s = ""
            for i in seeds:
                k = f"{ph}__{a}__s{i}"
                if k in fin:
                    s += SQ; done += 1; w += 1
                elif k in run:
                    s += RUN; w += run[k]["frac"]
                else:
                    s += SQO
            rows.append((a, s))
        stage_info.append((name, n, done, w, rows))
        total += n
        work += w
    el = (now - t0).total_seconds() / 3600
    rate = work / el if el > 0.5 and work > 0.8 else 4.2
    out = [f"{Y}Extra-seed runs progress   {now:%a %H:%M:%S}   elapsed {hm(el)}{R}", ""]
    rem_total = total - work
    eta = now + timedelta(hours=rem_total / rate)
    out.append(f"OVERALL  {C}{bar(work / total, 40)}{R} {100 * work / total:4.1f}%   {work:.1f} / {total} runs")
    out.append(f"         finishes {eta:%a %H:%M}  (countdown {hm(rem_total / rate)})   speed {rate:.1f} runs/h"
               + ("" if el > 0.5 and work > 0.8 else "  (assumed, measuring...)"))
    out.append("")
    cum = 0.0
    for name, n, done, w, rows in stage_info:
        cum += n - w
        end = "done" if n - w < 0.01 else f"done ~{now + timedelta(hours=cum / rate):%a %H:%M}"
        col = G if done == n else C
        out.append(f"{name:<36}{col}{bar(w / n, 24)}{R} {done:>2}/{n:<2} {D}{end}{R}")
    out.append("")
    out.append(f"{D}Seeds per arm ({SQ} finished, {RUN} running, {SQO} waiting){R}")
    for name, n, done, w, rows in stage_info:
        for a, s in rows:
            out.append(f"  {a:<10}{s}")
    out.append("")
    out.append("Running now")
    if not run:
        out.append(f"  {D}(none in the last 12 min - check the launch window){R}")
    for k, v in sorted(run.items()):
        out.append(f"  {k:<24}{C}{bar(v['frac'], 20)}{R} {100 * v['frac']:3.0f}%  ep {v['ep']:>2}/10  val {v['val']}%")
    out += ["", f"{D}Ctrl+C stops this display only.{R}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-11-25 21:50")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    t0 = datetime.strptime(a.start, "%Y-%m-%d %H:%M")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.system("")                                   # enable ANSI colours in the Windows console
    while True:
        txt = render(t0, datetime.now())
        if not a.once:
            os.system("cls" if os.name == "nt" else "clear")
        print(txt, flush=True)
        if a.once:
            return
        time.sleep(3)


if __name__ == "__main__":
    main()
