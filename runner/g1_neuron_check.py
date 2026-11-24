#!/usr/bin/env python
"""G1 neuron unit test (protocol v0.2, gate G1).

Checks, with no training:
  A. leak() equals the closed form m*B/sqrt(1-m^2(1-B^2)) and is an exact exponential in u=m/sqrt(1-m^2).
  B. integrate() equals Joy's tanh law with the repo's A(J) and tau_r(J) tables (independent re-implementation).
  C. tau_r(J) against Joy's fitted rise times (descriptive overlay).
  D. repo leak iterated at dt=100 ps against Joy's mumax decay data (descriptive overlay).

Usage:  python g1_neuron_check.py UPSTREAM_THMTJ_PY MUMAX_ROOT
  MUMAX_ROOT = the folder that contains 'LIF_equation_current_1' and 'Last week of March 26'
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from core_loader import load_core          # when run from the repo's runner/ folder
except ImportError:
    sys.path.insert(0, "/home/claude/ternary-snn-taum-study/runner")
    from core_loader import load_core

up, root = sys.argv[1], Path(sys.argv[2])
core = load_core(up)
f64 = torch.float64
ok_all = True


def report(name, val, tol=None, unit=""):
    global ok_all
    if tol is None:
        print(f"  {name}: {val:.4g}{unit}")
    else:
        good = val <= tol
        ok_all &= good
        print(f"  {name}: {val:.3e}  (limit {tol:.0e})  {'PASS' if good else 'FAIL'}")


# ---------------------------------------------------------------- A. leak
print("A. leak() vs closed form and exponential-in-u identity (float64)")
m = torch.linspace(-0.999, 0.999, 1999, dtype=f64)
worst_cf, worst_u = 0.0, 0.0
for tau in (50.0, 100.0, 200.0, 350.0, 503.8, 800.0, 1200.0, 2000.0, 5000.0):
    n = core.HardMTJNeuron(tau_leak_ps=tau, dt_ps=100.0)
    B = float(np.exp(-100.0 / tau))
    got = n.leak(m)
    ref = m * B / torch.sqrt(1 - m**2 * (1 - B**2))
    worst_cf = max(worst_cf, float((got - ref).abs().max()))
    u_in = m / torch.sqrt(1 - m**2)
    u_out = got / torch.sqrt(1 - got**2)
    worst_u = max(worst_u, float((u_out - B * u_in).abs().max() / (u_in.abs().max())))
report("max |leak - closed form|", worst_cf, 1e-12)
report("max relative |u_out - B*u_in|", worst_u, 1e-9)
n = core.HardMTJNeuron(tau_leak_ps=503.8)
print(f"  beta at tau=503.8 ps, dt=100 ps: {np.exp(-100/503.8):.4f}   leak preserves sign: "
      f"{bool(((n.leak(m) * m) >= 0).all())}   |leak(m)| < |m|: {bool((n.leak(m).abs() < m.abs() + 1e-15).all())}")

# ---------------------------------------------------------------- B. integrate
print("B. integrate() vs independent implementation of Joy's tanh law")
Jg = np.array([1e11, 2e11, 3e11, 4e11, 5e11, 6e11, 7e11, 8e11, 9e11, 1e12])
Ag = np.array([0.9855, 0.9991, 0.9997, 0.9998, 0.9999, 0.99993, 0.99999, 0.9999, 1.0, 1.0])
n = core.HardMTJNeuron()
worst = 0.0
for J in np.linspace(1e11, 1e12, 37):
    mm = np.linspace(-0.98, 0.98, 99)
    A = np.interp(J, Jg, Ag)
    tr = 386.98 * (J / 1e11) ** -1.223 + 8.88
    ratio = np.clip(mm / (A + 1e-8), -0.999, 0.999)
    ref = A * np.tanh(30.0 / (tr + 1e-8) + np.arctanh(ratio))
    got = n.integrate(torch.tensor(mm, dtype=f64), torch.full((99,), J, dtype=f64),
                      torch.ones(99, dtype=f64)).numpy()
    worst = max(worst, float(np.abs(got - ref).max()))
report("max |integrate - reference|", worst, 1e-9)

# ---------------------------------------------------------------- C. tau_r(J)
print("C. tau_r(J) of the repo vs Joy's fitted rise times (descriptive)")
tr_repo = lambda J: float(n.get_tau_r(torch.tensor(J, dtype=f64)))
for J, fit, note in ((7e11, 45.1, "our fit of her rise table"), (1e12, 31.2, "annotated in her script")):
    r = tr_repo(J)
    print(f"  J={J:.0e}: repo {r:.1f} ps   Joy {fit:.1f} ps ({note})   diff {100*(r-fit)/fit:+.1f}%")
print(f"  A(1e12): repo {float(n.get_A(torch.tensor(1e12, dtype=f64))):.5f}   Joy's fitted rise amplitude 0.99923")


# ---------------------------------------------------------------- D. leak vs mumax
def u_of(mz):
    return mz / np.sqrt(1 - mz**2)


def fit_tau(t, mz, lo=0.02, hi=0.995):
    k = (mz > lo) & (mz < hi)
    s = np.polyfit(t[k], np.log(u_of(mz[k])), 1)[0]
    return -1.0 / s, k.sum()


def repo_leak_curve(m0, tau_ps, steps):
    nn_ = core.HardMTJNeuron(tau_leak_ps=tau_ps, dt_ps=100.0)
    x = torch.tensor([m0], dtype=f64)
    out = [m0]
    for _ in range(steps):
        x = nn_.leak(x)
        out.append(float(x))
    return np.array(out)


print("D1. free decay, alpha=0.01, Ku1=0 (LIF_equation_current_1/Mz_time_decay.out/table.txt)")
d = np.loadtxt(root / "LIF_equation_current_1" / "Mz_time_decay.out" / "table.txt")
t, mz = d[:, 0] * 1e12, d[:, 3]                      # ps
i0 = int(np.argmax(mz <= 0.95))
tfit, npts = fit_tau(t[i0:], mz[i0:], lo=0.02, hi=0.95)
print(f"  fitted tau from {npts} points: {tfit:.0f} ps")
step = int(round(100.0 / (t[1] - t[0])))             # table samples per 100 ps
idx = np.arange(i0, len(t), step)
ts, ms = t[idx] - t[i0], mz[idx]
keep = ms > 0.02
for tau in (503.8, tfit):
    sim = repo_leak_curve(ms[0], tau, len(ms) - 1)
    e = np.abs(sim - ms)[keep]
    print(f"  repo leak with tau={tau:6.1f} ps: max |dmz| {e.max():.4f}, rms {np.sqrt((e**2).mean()):.4f} over {keep.sum()} points")

print("D2. post-pulse tails, Change alpha tables (Ku1=0); tau_leak from ln(u) slope after the mz maximum")
base = root / "Last week of March 26" / "Code" / "Change alpha"
for name, alpha in (("table1.txt", 0.01), ("table2.txt", 0.05), ("table3.txt", 0.1)):
    d = np.loadtxt(base / name)
    t, mz = d[:, 0] * 1e12, d[:, 3]
    p = int(np.argmax(mz))
    if mz[p] < 0.02 or len(t) - p < 4:
        print(f"  alpha={alpha}: no usable tail (max mz {mz[p]:.3f})")
        continue
    tt = t[p:] - t[p]
    k = mz[p:] > 0.0
    s, c = np.polyfit(tt[k], np.log(u_of(mz[p:][k])), 1)
    tau = -1.0 / s
    pred = 503.8 * 0.01 / alpha
    print(f"  alpha={alpha}: peak mz {mz[p]:.3f}, tail {k.sum()} pts over {tt[-1]:.0f} ps, tau {tau:.0f} ps; "
          f"503.8*0.01/alpha = {pred:.0f} ps ({100*(tau-pred)/pred:+.0f}%), alpha*tau = {alpha*tau:.2f} ps")

print("\nG1 unit-test items A and B:", "PASS" if ok_all else "FAIL")
