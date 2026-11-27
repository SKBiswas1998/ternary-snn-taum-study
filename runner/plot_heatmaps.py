"""Clean heat maps for the tau_m sweep.  Usage: python runner/plot_heatmaps.py ledger/runs figures
fig5: correlation heat maps (lower triangle).  fig6: mean validation accuracy by tau_m and epoch."""
import json, glob, re, sys, os
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, Normalize
R = sys.argv[1] if len(sys.argv) > 1 else "ledger/runs"; OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
INK, MUTED = "#1f2933", "#6b7280"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

runs = []
for f in sorted(glob.glob(os.path.join(R, "explore__*.json"))):
    d = json.load(open(f)); st = d["stats"]
    runs.append(dict(tau=d["params"]["tau_ps"], seed=d["params"]["seed"], val=d["final_val"], tr=d["train_acc"][-1],
                     fr1=st["fr1"], fr2=st["fr2"], fr3=st["fr3"], neg3=st["neg3"], out=st["out_abs"], curve=[v for _, v in d["val_acc"]]))
tau = np.array([r["tau"] for r in runs]); col = lambda k: np.array([r[k] for r in runs])

# ---- fig 5: correlations ----
keys = ["val", "tr", "fr1", "fr2", "fr3", "neg3", "out"]
names = ["Val.\\nacc.", "Train\\nacc.", "Layer-1\\nfiring", "Layer-2\\nfiring", "Output\\nfiring", "Output\\nneg. share", "Output\\nscale"]
names = [n.replace("\\n", "\n") for n in names]
def within(x):
    z = x.astype(float).copy()
    for t in np.unique(tau): z[tau == t] -= z[tau == t].mean()
    return z
def corr(keep, wi):
    X = np.array([(within(col(k)) if wi else col(k))[keep] for k in keys]).T
    return stats.spearmanr(X)[0]
cm = LinearSegmentedColormap.from_list("div", ["#0b5fa5", "#5aa9e6", "#f7f7f7", "#ff8a65", "#c1121f"])
fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw=dict(wspace=0.12))
for ax, (title, sub, M) in zip(axs, (("All runs", "includes the effect of tau_m", corr(np.ones(len(runs), bool), False)),
                                     ("Within each arm", "tau_m effect removed, tau_m of 200 ps and up", corr(tau >= 200, True)))):
    n = len(keys); T = M[1:, :-1]; m = n - 1
    mask = np.triu(np.ones((m, m), bool), 1)            # keep the lower triangle including the diagonal of the trimmed matrix
    im = ax.imshow(np.ma.array(T, mask=mask), cmap=cm, vmin=-1, vmax=1)
    for i in range(m + 1):
        ax.axhline(i - 0.5, color="white", lw=2); ax.axvline(i - 0.5, color="white", lw=2)
    for i in range(m):
        for j in range(i + 1):
            v = T[i, j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=11.5, color="white" if abs(v) > 0.65 else INK)
    ax.set_xticks(range(m)); ax.set_xticklabels(names[:-1], fontsize=9.5)
    ax.set_yticks(range(m)); ax.set_yticklabels(names[1:], fontsize=9.5)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=34); ax.text(0, 1.035, sub, fontsize=10.5, color=MUTED, transform=ax.transAxes)
    for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.04, pad=0.14, aspect=40); cb.set_label("Spearman correlation", fontsize=10.5); cb.outline.set_visible(False)
fig.text(0.01, -0.03, "78 runs (tau_m 50 ps to 5 ns). Exploratory; correlation, not causation: tau_m drives every statistic here.", fontsize=9.5, color=MUTED)
fig.savefig(os.path.join(OUT, "fig5_correlations.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

# ---- fig 6: accuracy by tau_m and epoch ----
taus = [50, 100, 200, 350, 503.8, 800, 1200, 2000, 3000, 5000]
Z = np.array([[np.mean([r["curve"][ep] for r in runs if r["tau"] == t]) for t in taus] for ep in range(10)])
cmap = plt.get_cmap("viridis")
fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(Z, cmap=cmap, norm=PowerNorm(gamma=3, vmin=62, vmax=96), origin="lower", aspect="auto")
for i in range(11):
    ax.axhline(i - 0.5, color="white", lw=2); ax.axvline(i - 0.5, color="white", lw=2)
for e in range(10):
    for j in range(10):
        v = Z[e, j]; ax.text(j, e, "%.1f" % v, ha="center", va="center", fontsize=10.5, color="white" if cmap(PowerNorm(gamma=3, vmin=62, vmax=96)(v))[:3] != () and sum(cmap(PowerNorm(gamma=3, vmin=62, vmax=96)(v))[:3]) < 1.9 else INK)
ax.set_xticks(range(10)); ax.set_xticklabels(["%g" % t for t in taus]); ax.set_yticks(range(10)); ax.set_yticklabels(range(1, 11))
ax.set_xlabel("membrane time constant tau_m (ps)", labelpad=10); ax.set_ylabel("training epoch", labelpad=10); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Validation accuracy (%) over training", loc="left", fontsize=14, fontweight="bold", pad=34)
ax.text(0, 1.035, "Long time constants reach high accuracy within a few epochs; 50 ps is still behind at epoch 10", fontsize=10.5, color=MUTED, transform=ax.transAxes)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[65, 80, 90, 95]); cb.outline.set_visible(False); cb.set_label("mean validation accuracy (%)", fontsize=10.5)
fig.text(0.01, -0.04, "Mean over seeds (5 per arm; 12 for 350, 503.8, 800, 2000 ps). Colour scale stretched towards the high end. Exploratory.", fontsize=9.5, color=MUTED)
fig.savefig(os.path.join(OUT, "fig6_heatmap_tau_seed.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
print("wrote fig5, fig6")
