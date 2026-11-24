#!/usr/bin/env python
"""Cheap checks of the runner itself (the runner is new code, so it must be tested before use)."""
import sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core_loader import load_core
from data import stratified_split, load_synthetic

up = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent / "upstream" / "mtj" / "ternary_mtj" / "thmtj.py")

# 1. split: deterministic, disjoint, exact size, stratified
y = np.random.RandomState(1).randint(0, 10, 60000)
tr, va = stratified_split(y, 10000); tr2, va2 = stratified_split(y, 10000)
assert (tr == tr2).all() and (va == va2).all(), "split not deterministic"
assert len(va) == 10000 and len(tr) == 50000 and not set(tr) & set(va), "split sizes/disjointness"
fr = np.bincount(y[va], minlength=10) / 10000; fr_all = np.bincount(y, minlength=10) / 60000
assert np.abs(fr - fr_all).max() < 0.002, "split not stratified"
print("ok  split: deterministic, disjoint, exact, stratified")

# 2. pairing: same seed -> identical init across arms with different tau/theta; different seed -> different
def init(seed, tau, tn):
    core = load_core(up); core.TAU_LEAK_PS, core.NEG_THRESHOLD = tau, tn
    torch.manual_seed(seed); m = core.HardMTJSpikingMLP(num_steps=25)
    return torch.cat([p.flatten() for p in m.parameters()])
a, b, c = init(7, 50, 0.9), init(7, 1200, 1.6), init(8, 50, 0.9)
assert torch.equal(a, b), "same seed must give same init across arms"
assert not torch.equal(a, c), "different seeds must differ"
print("ok  pairing: same seed -> same init in every arm")

# 3. override reaches the neurons (the sweep would be silently void otherwise)
core = load_core(up); core.TAU_LEAK_PS, core.THRESHOLD, core.NEG_THRESHOLD = 123.0, 0.55, 0.66
m = core.HardMTJSpikingMLP(num_steps=25)
for n in (m.neuron1, m.neuron2, m.neuron3):
    vals = {k: v for k, v in vars(n).items() if isinstance(v, (int, float))}
    assert any(abs(v - 123.0) < 1e-9 for v in vals.values()), f"tau override missing in {vals}"
    assert any(abs(v - 0.55) < 1e-9 for v in vals.values()), f"theta+ override missing in {vals}"
    assert any(abs(v - 0.66) < 1e-9 for v in vals.values()), f"theta- override missing in {vals}"
print("ok  overrides: tau, theta+, theta- reach all three neuron layers")
print("selftest passed. NOTE: G1 (analytic leak check) is NOT implemented yet.")
