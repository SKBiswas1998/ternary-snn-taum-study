#!/usr/bin/env python
"""Time one training batch on THIS machine (synthetic data) -> minutes per run estimate."""
import argparse, sys, time
from pathlib import Path
import torch, torch.nn as nn
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core_loader import load_core

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--upstream", default=str(HERE.parent / "upstream" / "mtj" / "ternary_mtj" / "thmtj.py"))
a = ap.parse_args()
torch.set_num_threads(a.threads)
core = load_core(a.upstream)
torch.manual_seed(0)
m = core.HardMTJSpikingMLP(num_steps=25)
opt = torch.optim.Adam(m.parameters(), 1e-3); crit = nn.CrossEntropyLoss()
x = torch.rand(64, 1, 28, 28); y = torch.randint(0, 10, (64,))
def step():
    opt.zero_grad(); loss = crit(m(x)[0].sum(0), y); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
for _ in range(2): step()
t = time.time(); n = 8
for _ in range(n): step()
dt = (time.time() - t) / n
ep = dt * 782 / 60
print(f"threads={a.threads}: {dt:.3f} s/batch -> {ep:.1f} min/epoch -> {ep*10:.0f} min per 10-epoch run (+~1 min eval)")
print(f"=> with N parallel single-thread workers expect ~{60/(ep*10+1):.1f} runs/hour/worker (slower if CPU throttles)")
