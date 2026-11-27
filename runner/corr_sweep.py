"""Correlation analysis of the exploratory sweep.  Usage: python runner/corr_sweep.py ledger/runs figures
Uses every explore run (tau 50 ps - 5 ns, seeds as available).  Spearman unless stated.  Exploratory; correlation is not causation:
tau_m sets everything else here, so stats are also analysed WITHIN arm (arm mean removed) to avoid confounding by tau_m."""
import json, glob, re, sys, os
import numpy as np
from scipy import stats
R = sys.argv[1] if len(sys.argv) > 1 else "ledger/runs"; OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(1)
rows = []
for f in sorted(glob.glob(os.path.join(R, "explore__*.json"))):
    d = json.load(open(f)); st = d["stats"]; va = [v for _, v in d["val_acc"]]
    rows.append(dict(tau=d["params"]["tau_ps"], seed=d["params"]["seed"], val=d["final_val"], last3=np.mean(va[-3:]),
                     tr=d["train_acc"][-1], loss=d["train_loss"][-1], v1=va[0], v3=va[2], v5=va[4],
                     fr1=st["fr1"], fr2=st["fr2"], fr3=st["fr3"], neg1=st["neg1"], neg2=st["neg2"], neg3=st["neg3"], out=st["out_abs"]))
n = len(rows); print("runs:", n)
col = lambda k: np.array([r[k] for r in rows]); tau = col("tau"); arm = {t: tau == t for t in np.unique(tau)}
val = col("val")
def boot_rho(x, y, B=2000):
    b = []
    for _ in range(B):
        i = rng.randint(0, len(x), len(x)); b.append(stats.spearmanr(x[i], y[i])[0])
    return np.nanpercentile(b, [2.5, 97.5])
def within(x):                                  # remove each arm's mean
    z = x.astype(float).copy()
    for t, m in arm.items(): z[m] -= z[m].mean()
    return z
print("\n=== 1. Accuracy vs tau_m transforms (Spearman rho, all runs, and by regime) ===")
beta = np.exp(-100 / tau)
for lab, x in (("log10 tau_m", np.log10(tau)), ("leak per step dt/tau", 100 / tau), ("beta = exp(-dt/tau)", beta)):
    r = stats.spearmanr(x, val)[0]; lo = tau >= 800; hi = ~lo
    print(f"  {lab:<22} all: {r:+.2f}   tau<800 ps: {stats.spearmanr(x[hi], val[hi])[0]:+.2f}   tau>=800 ps: {stats.spearmanr(x[lo], val[lo])[0]:+.2f}  (n={lo.sum()} on the plateau)")
x = (100 / tau) ** 1.59; print(f"  Pearson r of accuracy with (dt/tau)^1.59 (the fitted form): {stats.pearsonr(x, val)[0]:+.3f}  -> variance explained {100*stats.pearsonr(x, val)[0]**2:.0f} %")
print("\n=== 2. What else moves with tau_m? Spearman of each run statistic with log10 tau_m (all runs) ===")
names = [("fr1", "layer-1 firing rate"), ("fr2", "layer-2 firing rate"), ("fr3", "layer-3 firing rate"), ("neg1", "layer-1 negative-spike share"),
         ("neg2", "layer-2 negative share"), ("neg3", "layer-3 negative share"), ("out", "output spike scale"), ("tr", "final train accuracy")]
for k, lab in names:
    print(f"  {lab:<30} rho {stats.spearmanr(np.log10(tau), col(k))[0]:+.2f}")
print("\n=== 3. Does a statistic predict accuracy WITHIN an arm? (arm means removed, so tau_m cannot confound) ===")
for k, lab in names[:7] + [("v1", "epoch-1 validation accuracy"), ("v3", "epoch-3 validation accuracy"), ("tr", "final train accuracy")]:
    keep = tau >= 200                                    # exclude the unstable 50/100 ps arms
    zx, zy = within(col(k))[keep], within(val)[keep]
    r, p = stats.spearmanr(zx, zy); ci = boot_rho(zx, zy)
    print(f"  {lab:<30} rho {r:+.2f}  95% boot [{ci[0]:+.2f}, {ci[1]:+.2f}]  p {p:.3f}  (n={keep.sum()}, tau >= 200 ps)")
print("\n=== 4. Is the seed a shared nuisance? Seed effect shared across arms (tau 350/503.8/800/2000, seeds 0-11) ===")
S = list(range(12)); M = np.array([[next(r["val"] for r in rows if r["tau"] == t and r["seed"] == s) for s in S] for t in (350, 503.8, 800, 2000)])
C = np.corrcoef(M); off = C[np.triu_indices(4, 1)]
print(f"  mean pairwise correlation across seeds {off.mean():+.2f}  (range {off.min():+.2f} to {off.max():+.2f}); with 12 seeds a value must exceed about +-0.58 to be significant")
print("\n=== 5. Does early learning speed predict the final result? (within arm, tau >= 200 ps) ===")
for k, lab in (("v1", "epoch 1"), ("v3", "epoch 3"), ("v5", "epoch 5")):
    keep = tau >= 200; r = stats.spearmanr(within(col(k))[keep], within(val)[keep])[0]; print(f"  validation at {lab} vs final: rho {r:+.2f}")
print("\n=== 6. Generalisation gap: train minus validation accuracy by arm ===")
for t in sorted(arm): g = (col("tr") - val)[arm[t]]; print(f"  tau {t:>6g}: train - val = {g.mean():+.2f} pp  (n={arm[t].sum()})")
r = stats.spearmanr(np.log10(tau), col("tr") - val)[0]; print(f"  rho(gap, log tau) = {r:+.2f}")

# Figures are drawn by plot_heatmaps.py (fig5 correlations, fig6 accuracy by tau_m and epoch).
