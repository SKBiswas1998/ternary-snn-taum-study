"""Plots for the exploratory tau_m sweep.  Usage: python runner/plot_sweep.py ledger/runs figures
Validation accuracy only (final epoch).  Exploratory: 5 seeds per arm, fixed lr/thresholds, 10 epochs."""
import json, glob, re, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs_dir = sys.argv[1] if len(sys.argv) > 1 else "ledger/runs"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(out_dir, exist_ok=True)

INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
BLUE, DARK = "#2a6fb0", "#0b2f57"
REF = 503.8
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

def load(prefix):
    out = {}
    for f in glob.glob(os.path.join(runs_dir, prefix + "__s*__lr*")):
        d = json.load(open(f)); s = int(re.search(r"__s(\d+)__lr", f).group(1)); out[s] = d
    return out

taus = [50, 100, 200, 350, 503.8, 800, 1200, 2000, 3000, 5000]
name = lambda t: ("%g" % t)
data = {t: load("explore__tau" + name(t)) for t in taus}
final = {t: np.array([data[t][s]["final_val"] for s in sorted(data[t])]) for t in taus}
seeds = {t: sorted(data[t]) for t in taus}

# Figure 1: accuracy vs tau_m
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.8)
mean = np.array([final[t].mean() for t in taus]); sd = np.array([final[t].std(ddof=1) for t in taus])
ax.fill_between(taus, mean - sd, mean + sd, color=BLUE, alpha=0.15, lw=0)
for t in taus:
    ax.scatter([t] * len(final[t]), final[t], s=22, color=BLUE, alpha=0.55, edgecolor="white", lw=0.6, zorder=3)
ax.plot(taus, mean, color=DARK, lw=2, marker="o", ms=6, mfc=DARK, mec="white", mew=1.2, zorder=4)
ax.axvline(REF, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(REF * 1.06, 83, "reference\n503.8 ps", color=MUTED, fontsize=9, va="bottom")
ax.set_xscale("log"); ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
ax.set_xlabel("membrane time constant tau_m (ps, log scale)"); ax.set_ylabel("validation accuracy (%)")
ax.set_title("Accuracy rises steeply, then levels off between about 500 and 800 ps", loc="left", color=INK, fontsize=12)
ax.annotate("50 ps arm still improving\nat epoch 10 (may be undertrained)", xy=(50, mean[0]), xytext=(68, 77),
            color=MUTED, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
fig.text(0.01, 0.01, "Dots: individual seeds (5 per arm; 12 for 350, 503.8, 800, 2000 ps). Line: mean. Band: +-1 sd. Exploratory, validation set, final epoch.",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(out_dir, "fig1_accuracy_vs_tau.png")); plt.close(fig)

# Figure 2: learning curves
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.8)
show = [50, 100, 200, 503.8, 1200, 5000]
cmap = matplotlib.colormaps["Blues"]
for i, t in enumerate(show):
    curves = np.array([[v for _, v in data[t][s]["val_acc"]] for s in sorted(data[t])])
    ep = [e for e, _ in data[t][sorted(data[t])[0]]["val_acc"]]
    c = cmap(0.35 + 0.6 * i / (len(show) - 1))
    ax.plot(ep, curves.mean(0), color=c, lw=2, marker="o", ms=4, label="%g ps" % t)
ax.set_xlim(1, 10.2); ax.set_ylim(60, 97); ax.set_xticks(range(1, 11))
ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy (%), mean over seeds")
ax.set_title("Short time constants learn more slowly; long ones converge together", loc="left", color=INK, fontsize=12)
ax.legend(title="tau_m", frameon=False, loc="lower right", fontsize=9, title_fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig2_learning_curves.png")); plt.close(fig)

# Figure 3: paired gain over the reference (same seed, same init and data order)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.8)
ts = [200, 350, 800, 1200, 2000, 3000, 5000]
x = np.arange(len(ts))
for k, t in enumerate(ts):
    d = np.array([data[t][s]["final_val"] - data[REF][s]["final_val"] for s in sorted(data[t])]) if False else \
        np.array([data[t][s]["final_val"] - data[503.8][s]["final_val"] for s in sorted(data[t])])
    ax.scatter(np.full(len(d), k) + np.linspace(-0.12, 0.12, len(d)), d, s=22, color=BLUE, alpha=0.6, edgecolor="white", lw=0.6, zorder=3)
    m, s_ = d.mean(), d.std(ddof=1)
    ax.errorbar(k, m, yerr=s_, color=DARK, lw=1.6, capsize=4, marker="o", ms=6, mfc=DARK, mec="white", zorder=4)
ax.axhline(0, color=MUTED, lw=1)
ax.axhspan(-0.4, 0.4, color=GRID, alpha=0.5, lw=0, zorder=0)
ax.text(len(ts) - 0.5, -0.44, "grey band: about +-0.4 pp, the smallest difference\n5 seeds can show (G3 noise)", color=MUTED, fontsize=8, ha="right", va="top")
ax.set_xticks(x); ax.set_xticklabels(["%g" % t for t in ts])
ax.set_xlabel("tau_m (ps)"); ax.set_ylabel("gain over 503.8 ps (percentage points)")
ax.set_title("Past 503.8 ps the gains are small; 200 ps and below is clearly worse", loc="left", color=INK, fontsize=12)
fig.text(0.01, 0.01, "Paired by seed. Dots: seeds (12 at 350, 800, 2000 ps; 5 elsewhere). Marker and bar: mean +-1 sd of the paired difference. Exploratory.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(out_dir, "fig3_paired_gain.png")); plt.close(fig)
print("wrote 3 figures to", out_dir)

# Figure 4: every run gathered so far, one dot per run (validation accuracy, final epoch)
allruns = {}
for f in glob.glob(os.path.join(runs_dir, "*.json")):
    d = json.load(open(f)); allruns.setdefault((d["phase"], d["arm"]), []).append(d["final_val"])
rows = [("G2 shuffled labels (control)", [v for (p, a), vs in allruns.items() if p == "G2" for v in vs], "ctl"),
        ("G0 committed code", allruns.get(("G0", "committed"), []), "bad"),
        ("G0 README build", allruns.get(("G0", "readme"), []), "ok"),
        ("G3 seed noise, 503.8 ps (10 seeds)", allruns.get(("G3", "ref"), []), "ok")]
for t in taus:
    rows.append(("sweep tau_m = %g ps" % t, allruns.get(("explore", "tau" + name(t)), []), "sw"))
col = {"ctl": MUTED, "bad": "#b4532a", "ok": BLUE, "sw": BLUE}
fig, ax = plt.subplots(figsize=(8.5, 6.2))
ax.set_axisbelow(True); ax.grid(axis="x", color=GRID, lw=0.8)
for i, (lab, vals, kind) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.scatter(vals, np.full(len(vals), y) + np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [y], s=26,
               color=col[kind], alpha=0.75, edgecolor="white", lw=0.6, zorder=3)
    if vals: ax.plot([np.mean(vals)] * 2, [y - 0.3, y + 0.3], color=DARK, lw=2, zorder=4)
ax.set_yticks([len(rows) - 1 - i for i in range(len(rows))]); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.axhline(len(rows) - 4.5, color=GRID, lw=1)
ax.set_xlabel("validation accuracy (%), final epoch"); ax.set_xlim(0, 100)
ax.set_title("All runs so far: controls behave, sweep arms from 350 ps up overlap", loc="left", color=INK, fontsize=12)
fig.text(0.01, 0.01, "One dot per run (56 sweep/control runs + 20 G0/G3 runs). Dark tick: mean. Orange: committed upstream code, mostly fails to learn.",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(out_dir, "fig4_all_runs.png")); plt.close(fig)
print("wrote fig4")

# Figure 6: heat map, mean validation accuracy by tau_m (x, log) and training epoch (y)
EP = np.arange(1, 11)
Z = np.array([[np.mean([[v for e, v in data[t][s_]["val_acc"] if e == ep][0] for s_ in data[t]]) for t in taus] for ep in EP])
lt = np.log10(taus)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("acc", ["#0b1d3a", "#1d4e89", "#2a9d8f", "#e9f5db"])
fig, ax = plt.subplots(figsize=(8.6, 4.8))
X, Y = np.meshgrid(lt, EP)
im = ax.contourf(X, Y, Z, levels=np.linspace(62, 96, 69), cmap=cmap, extend="both")
cs = ax.contour(X, Y, Z, levels=[80, 90, 94, 95], colors="white", linewidths=0.8, alpha=0.8)
ax.clabel(cs, fmt="%g%%", fontsize=8)
ax.set_xticks(lt); ax.set_xticklabels(["%g" % t for t in taus], fontsize=8.5)
ax.set_yticks(EP); ax.tick_params(length=0)
ax.set_xlabel("membrane time constant tau_m (ps)"); ax.set_ylabel("training epoch")
for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[65, 75, 85, 95]); cb.set_label("mean validation accuracy (%)", fontsize=9); cb.outline.set_visible(False)
ax.set_title("Long time constants reach high accuracy early; short ones are still behind at epoch 10", loc="left", fontsize=11, color=INK)
fig.text(0.01, -0.01, "Mean over seeds (5 per arm; 12 for 350, 503.8, 800, 2000 ps). White lines: 80, 90, 94, 95 % accuracy. Exploratory.", fontsize=8, color=MUTED)
fig.savefig(os.path.join(out_dir, "fig6_heatmap_tau_seed.png"), bbox_inches="tight"); plt.close(fig); print("wrote fig6")
