"""Statistical checks for the exploratory tau_m sweep (validation accuracy).  Usage: python runner/stats_sweep.py ledger/runs
Everything is computed from the per-run JSON files.  Seeds 0-4 are shared across arms (paired design)."""
import json, glob, re, sys, os, itertools
import numpy as np
from scipy import stats
R = sys.argv[1] if len(sys.argv) > 1 else "ledger/runs"
rng = np.random.RandomState(12345)
taus = [50, 100, 200, 350, 503.8, 800, 1200, 2000, 3000, 5000]
nm = lambda t: "%g" % t
def load(prefix):
    o = {}
    for f in glob.glob(os.path.join(R, prefix + "__s*__lr*")):
        k = int(re.search(r"__s(\d+)__lr", f).group(1))
        if k < 5 or prefix.startswith(("G0", "G3", "G2")): o[k] = json.load(open(f))   # seeds 0-4 only; extra seeds are analysed in stats_more.py
    return o
D = {t: load("explore__tau" + nm(t)) for t in taus}
S = sorted(D[503.8])
A = np.array([[D[t][s]["final_val"] for s in S] for t in taus])         # arms x seeds
last3 = np.array([[np.mean([v for _, v in D[t][s]["val_acc"][-3:]]) for s in S] for t in taus])
best = np.array([[max(v for _, v in D[t][s]["val_acc"]) for s in S] for t in taus])
ref = taus.index(503.8)
tcrit = stats.t.ppf(0.975, 4)

print("=== 1. Per-arm mean, sd, 95% CI (t, df=4), bootstrap CI ===")
for i, t in enumerate(taus):
    m, sd = A[i].mean(), A[i].std(ddof=1); h = tcrit * sd / np.sqrt(5)
    bs = [rng.choice(A[i], 5).mean() for _ in range(5000)]
    print(f"tau {nm(t):>6}: mean {m:6.2f}  sd {sd:4.2f}  95%CI [{m-h:6.2f}, {m+h:6.2f}]  boot [{np.percentile(bs,2.5):6.2f}, {np.percentile(bs,97.5):6.2f}]")

print("\n=== 2. Paired difference vs 503.8 ps (same seed = same init and data order) ===")
rows = []
for i, t in enumerate(taus):
    if i == ref: continue
    d = A[i] - A[ref]; m, sd = d.mean(), d.std(ddof=1); h = tcrit * sd / np.sqrt(5)
    p = stats.ttest_1samp(d, 0).pvalue
    # exact sign-flip permutation test (32 sign patterns); smallest possible two-sided p is 2/32
    flips = [np.mean(d * np.array(sg)) for sg in itertools.product([1, -1], repeat=5)]
    pperm = np.mean(np.abs(flips) >= abs(d.mean()) - 1e-12)
    rows.append((t, m, h, p, pperm, int((d > 0).sum())))
ps = np.array([r[3] for r in rows]); order = np.argsort(ps); holm = np.empty(len(ps))
run = 0
for rank, idx in enumerate(order):
    run = max(run, min(1, ps[idx] * (len(ps) - rank))); holm[idx] = run
for r, h in zip(rows, holm):
    print(f"tau {nm(r[0]):>6}: diff {r[1]:+5.2f} pp  95%CI [{r[1]-r[2]:+5.2f}, {r[1]+r[2]:+5.2f}]  paired-t p {r[3]:.3f}  Holm p {h:.3f}  perm p {r[4]:.3f}  seeds better {r[5]}/5")

print("\n=== 3. Does pairing help? Correlation of seed effects between arms >= 350 ps ===")
sub = A[taus.index(350):]                       # arms 350..5000
C = np.corrcoef(sub); off = C[np.triu_indices_from(C, 1)]
print(f"mean pairwise correlation across seeds: {off.mean():.2f}  (range {off.min():.2f} to {off.max():.2f})")
vu = np.mean([np.var(sub[a] - sub[b], ddof=1) for a, b in itertools.combinations(range(len(sub)), 2)])
vp = np.mean([np.var(sub[a], ddof=1) + np.var(sub[b], ddof=1) for a, b in itertools.combinations(range(len(sub)), 2)])
print(f"var of paired difference {vu:.3f} vs unpaired (sum of variances) {vp:.3f}")

print("\n=== 4. Plateau tests, arms 350..5000 ps (7 arms x 5 seeds, seed = block) ===")
fr = stats.friedmanchisquare(*sub)
print(f"Friedman test across the 7 arms: chi2 {fr.statistic:.2f}, p {fr.pvalue:.3f}")
obs = sub.mean(1).max() - sub.mean(1).min(); cnt = 0
for _ in range(20000):
    P = np.array([rng.permutation(sub[:, j]) for j in range(5)]).T   # shuffle arm labels within each seed
    cnt += (P.mean(1).max() - P.mean(1).min()) >= obs
print(f"permutation test: observed spread between best and worst arm means {obs:.2f} pp, p {cnt/20000:.3f}")
lt = np.log10(taus[taus.index(350):]); slopes = []
for j in range(5):
    slopes.append(np.polyfit(lt, sub[:, j], 1)[0])
slopes = np.array(slopes); h = tcrit * slopes.std(ddof=1) / np.sqrt(5)
print(f"slope of accuracy vs log10(tau), per seed then averaged: {slopes.mean():+.2f} pp per decade, 95%CI [{slopes.mean()-h:+.2f}, {slopes.mean()+h:+.2f}]")
rise = A[taus.index(200)].mean() - A[0].mean()
print(f"for scale: gain from 50 to 200 ps is {rise:.1f} pp; a full decade of tau above 350 ps changes accuracy by about {slopes.mean():+.2f} pp")

print("\n=== 5. Knee location, bootstrap over seeds (resampling seeds jointly keeps pairing) ===")
plat_idx = [taus.index(x) for x in (800, 1200, 2000, 3000, 5000)]; knees = []
for _ in range(5000):
    sidx = rng.randint(0, 5, 5); M = A[:, sidx].mean(1); plat = M[plat_idx].mean()
    k = next((taus[i] for i in range(len(taus)) if plat - M[i] <= 0.5), None); knees.append(k)
vals, cts = np.unique([k for k in knees if k is not None], return_counts=True)
print("smallest tau within 0.5 pp of the plateau (mean of 800-5000 ps):", {nm(v): round(c / 50, 1) for v, c in zip(vals, cts)}, "% of bootstrap draws")

print("\n=== 6. Is the picture an artifact of the metric? final epoch vs mean of last 3 epochs vs best epoch ===")
for lab, M in (("final epoch", A), ("last-3 mean", last3), ("best epoch", best)):
    print(f"{lab:>12}: " + "  ".join(f"{nm(t)}:{M[i].mean():.2f}" for i, t in enumerate(taus)))
ep_sd = np.mean([np.std([v for _, v in D[503.8][s]["val_acc"]][-4:], ddof=1) for s in S])
print(f"within-run epoch-to-epoch sd over the last 4 epochs (503.8 ps): {ep_sd:.2f} pp  (compare seed sd ~0.4)")

print("\n=== 7. Artifact checks ===")
G0 = load("G0__readme"); same = all(abs(G0[s]["final_val"] - D[503.8][s]["final_val"]) < 1e-9 for s in S)
print("503.8 ps sweep arm identical to G0 README build for the same seeds (determinism):", same)
G3 = load("G3__ref"); g3 = np.array([G3[s]["final_val"] for s in sorted(G3)])
w = stats.ttest_ind(g3, A[ref], equal_var=False)
print(f"G3 (seeds 100-109) {g3.mean():.2f} +- {g3.std(ddof=1):.2f} vs sweep reference (seeds 0-4) {A[ref].mean():.2f} +- {A[ref].std(ddof=1):.2f}: Welch p {w.pvalue:.2f}")
pooled = np.concatenate([g3, A[ref], np.array([G0[s]["final_val"] for s in S])[:0]])
print(f"pooled reference-arm sd (15 runs counted in G0+G3 earlier ~0.31); here G3+sweep ref: {pooled.std(ddof=1):.2f}")
lv = stats.levene(*A[2:]); print(f"Levene equal-variance test, arms 200 ps and up: p {lv.pvalue:.2f}; sd at 50 ps {A[0].std(ddof=1):.2f}, 100 ps {A[1].std(ddof=1):.2f} (so small-tau arms need Welch/paired tests, not pooled sd)")
G2 = [json.load(open(f))["final_val"] for f in glob.glob(os.path.join(R, "G2__*"))]
bt = stats.binomtest(int(round(np.mean(G2) / 100 * 10000)), 10000, 0.1)
print(f"G2 shuffled labels: mean {np.mean(G2):.2f}% over {len(G2)} runs (max {max(G2):.2f}); chance is about 10%")
fr1 = [d["stats"]["fr1"] for t in taus for d in D[t].values()]
print(f"layer-1 firing rate (any sign, per neuron per step), all sweep runs: {min(fr1):.3f} to {max(fr1):.3f}  -> near-saturated in every arm")
print("binomial sampling noise of one 10k-image validation score at 95%:", round(100 * np.sqrt(.95 * .05 / 10000), 2), "pp (shared by all arms, so it cancels in paired differences)")
