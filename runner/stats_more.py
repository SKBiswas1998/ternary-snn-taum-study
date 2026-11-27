"""12-seed comparison on tau 350 / 503.8 / 800 / 2000 ps (seeds 0-11, paired by seed). Usage: python runner/stats_more.py ledger/runs"""
import json, glob, re, sys, os, itertools
import numpy as np
from scipy import stats
R = sys.argv[1] if len(sys.argv) > 1 else "ledger/runs"
rng = np.random.RandomState(2026)
arms = [350, 503.8, 800, 2000]
def load(t):
    o = {}
    for f in glob.glob(os.path.join(R, "explore__tau%g__s*__lr*" % t)):
        o[int(re.search(r"__s(\d+)__lr", f).group(1))] = json.load(open(f))
    return o
D = {t: load(t) for t in arms}
S = sorted(set.intersection(*[set(D[t]) for t in arms])); print("seeds:", S, "n =", len(S))
fin = {t: np.array([D[t][s]["final_val"] for s in S]) for t in arms}
l3 = {t: np.array([np.mean([v for _, v in D[t][s]["val_acc"][-3:]]) for s in S]) for t in arms}
tc = stats.t.ppf(0.975, len(S) - 1)
for lab, M in (("FINAL EPOCH", fin), ("LAST-3-EPOCH MEAN", l3)):
    print(f"\n=== {lab} ===")
    for t in arms:
        v = M[t]; h = tc * v.std(ddof=1) / np.sqrt(len(v))
        print(f"tau {t:>6g}: mean {v.mean():.2f}  sd {v.std(ddof=1):.2f}  95%CI [{v.mean()-h:.2f}, {v.mean()+h:.2f}]  min {v.min():.2f} max {v.max():.2f}")
    pairs = [(503.8, 350), (503.8, 800), (800, 2000), (503.8, 2000), (350, 800), (350, 2000)]
    print("paired differences (second minus first), 95% CI, t-test p, exact sign-flip p (n=12), Holm over 6 pairs")
    out = []
    for a, b in pairs:
        d = M[b] - M[a]; m = d.mean(); h = tc * d.std(ddof=1) / np.sqrt(len(d)); p = stats.ttest_1samp(d, 0).pvalue
        flips = np.array(list(itertools.product([1, -1], repeat=len(d)))); pp = np.mean(np.abs((flips * d).mean(1)) >= abs(m) - 1e-12)
        out.append([a, b, m, h, p, pp, int((d > 0).sum())])
    ps = np.array([o[4] for o in out]); order = np.argsort(ps); holm = np.empty(len(ps)); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1, ps[i] * (len(ps) - rank))); holm[i] = run
    for o, h in zip(out, holm):
        print(f"  {o[1]:>6g} - {o[0]:<6g}: {o[2]:+.2f} pp  [{o[2]-o[3]:+.2f}, {o[2]+o[3]:+.2f}]  t p {o[4]:.3f}  perm p {o[5]:.3f}  Holm {h:.3f}  seeds higher {o[6]}/{len(S)}")
    X = np.array([M[t] for t in arms])
    print(f"  Friedman over the 4 arms: p {stats.friedmanchisquare(*X).pvalue:.3f}")
    plat = np.array([M[800], M[2000]])
    d = M[2000] - M[800]; h90 = stats.t.ppf(0.95, len(S) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    print(f"  equivalence 800 vs 2000 (margin +-0.3 pp): diff {d.mean():+.2f}, 90% CI [{d.mean()-h90:+.2f}, {d.mean()+h90:+.2f}] -> {'equivalent' if abs(d.mean())+h90 < 0.3 else 'not shown equivalent at 0.3'}; (margin +-0.5): {'equivalent' if abs(d.mean())+h90 < 0.5 else 'not shown'}")
print("\n=== seed-level view (final epoch): every seed, arms 350 / 503.8 / 800 / 2000 ===")
for i, s in enumerate(S): print(f"  s{s:<2d}", "  ".join(f"{fin[t][i]:.2f}" for t in arms))
print("\nsd of paired diff 800-503.8 across seeds 0-4 vs seeds 5-11 (does the early edge hold on fresh seeds?)")
d = fin[800] - fin[503.8]; print(f"  seeds 0-4: {d[:5].mean():+.2f}   seeds 5-11 (fresh): {d[5:].mean():+.2f}  (n=7, 95% CI +-{stats.t.ppf(.975,6)*d[5:].std(ddof=1)/np.sqrt(7):.2f})")
d = fin[2000] - fin[503.8]; print(f"  2000-503.8: seeds 0-4: {d[:5].mean():+.2f}   seeds 5-11: {d[5:].mean():+.2f}")
d = fin[350] - fin[503.8]; print(f"  350-503.8: seeds 0-4: {d[:5].mean():+.2f}   seeds 5-11: {d[5:].mean():+.2f}")
