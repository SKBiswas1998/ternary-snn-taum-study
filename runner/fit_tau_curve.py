"""Empirical curve fits: validation accuracy vs tau_m.  Usage: python runner/fit_tau_curve.py ledger/runs
Exploratory. Fits use the 10 sweep arms (50 ps - 5 ns), weighted by 1/sd^2 of each arm (floor 0.15 pp)."""
import json, glob, re, sys, os
import numpy as np
from scipy.optimize import curve_fit
R = sys.argv[1] if len(sys.argv) > 1 else "ledger/runs"
DT = 100.0
taus = np.array([50, 100, 200, 350, 503.8, 800, 1200, 2000, 3000, 5000.])
def arm(t):
    return np.array([json.load(open(f))["final_val"] for f in sorted(glob.glob(os.path.join(R, "explore__tau%g__s*" % t)))])
vals = [arm(t) for t in taus]
m = np.array([v.mean() for v in vals]); sd = np.maximum([v.std(ddof=1) for v in vals], 0.15); n = np.array([len(v) for v in vals])
sem = sd / np.sqrt(n)

models = {
 "exp   A - B*exp(-tau/t0)":              (lambda t, A, B, t0: A - B*np.exp(-t/t0),            [95.5, 20, 100]),
 "power A - B*(dt/tau)^p":                (lambda t, A, B, p: A - B*(DT/t)**p,                 [95.5, 5, 1.0]),
 "beta  A - B*(1-beta)^p, beta=exp(-dt/tau)": (lambda t, A, B, p: A - B*(1-np.exp(-DT/t))**p,   [95.5, 20, 2.0]),
}
rng = np.random.RandomState(7)
print("Fits to arm means (weighted). rss = weighted residual sum of squares; AICc uses k parameters, 10 points.\n")
res = {}
for name, (f, p0) in models.items():
    try:
        p, _ = curve_fit(f, taus, m, p0=p0, sigma=sem, absolute_sigma=True, maxfev=20000)
    except Exception as e:
        print(name, "fit failed:", e); continue
    pred = f(taus, *p); rss = np.sum(((m - pred) / sem) ** 2); k = len(p); N = len(taus)
    aicc = N*np.log(max(rss, 1e-9)/N) + 2*k + 2*k*(k+1)/(N-k-1)
    # bootstrap over seeds (resample seeds within each arm)
    bp = []
    for _ in range(1500):
        mb = np.array([rng.choice(v, len(v)).mean() for v in vals])
        try:
            q, _ = curve_fit(f, taus, mb, p0=p, sigma=sem, absolute_sigma=True, maxfev=20000); bp.append(q)
        except Exception: pass
    bp = np.array(bp); ci = np.percentile(bp, [2.5, 97.5], axis=0) if len(bp) > 100 else None
    res[name] = (p, rss, aicc, pred)
    print(name)
    for j, v in enumerate(p):
        s = f"   p{j} = {v:8.3f}" + (f"   95% boot [{ci[0][j]:8.3f}, {ci[1][j]:8.3f}]" if ci is not None else "")
        print(s)
    print(f"   weighted rss {rss:6.1f}   AICc {aicc:6.1f}   max |residual| {np.max(np.abs(m-pred)):.2f} pp\n")
best = min(res, key=lambda k: res[k][2]); print("lowest AICc:", best)
p = res[best][0]
print("\nResiduals for the lowest-AICc model (pp):", {("%g" % t): round(float(a), 2) for t, a in zip(taus, m - res[best][3])})
