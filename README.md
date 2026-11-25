# ternary-snn-taum-study

Does the membrane time constant **tau_m** change the accuracy of the ternary MTJ-inspired spiking network
(Drexel group, `ternary_mtj/thmtj.py`), and why does accuracy rise and then saturate as the magnetic
anisotropy of the device grows? The design is pre-registered: read **`protocol/protocol.pdf`** first.

> **Status (25 Nov 2026, afternoon).** Protocol **v0.2 is a draft, not frozen**. No confirmatory run has been made.
> Gates G0, G1, G2 and G3 are done and the exploratory tau_m sweep (tau_m 50 ps to 5 ns, 56 runs in all) has
> finished. The sweep is **exploratory**: validation accuracy only, one learning rate, fixed thresholds, 10 epochs,
> 5 seeds per arm. It points to a hypothesis; it does not confirm one.

## What we know so far

| Item | Result |
|---|---|
| **G0 replication, README build** (negative threshold 0.90, 5 seeds) | test accuracy 95.45 % best-epoch (authors' way), 95.27 % final epoch; validation 95.02 +- 0.45 %. Authors report 95.89 % best epoch, 95.22 % at epoch 10. **Narrow miss on the pre-set +-0.3 pp rule** (0.44 pp low on the best-epoch mean); final epoch matches. Kept as "not passed" until a stricter test is run. |
| **G0, committed code** (negative threshold 1.6, 5 seeds) | 65 % final / 68 % best-epoch test; only 1 of 5 seeds learned; 0 % negative spikes. The committed code does not reproduce the README; the README build does. |
| **G1 neuron unit test** | `leak()` and `integrate()` match Joy's closed forms to machine precision. tau_r(J) within 1-3 % of her fits; free-decay tau at alpha = 0.01 fits 523 ps (repo uses 503.8 ps; error in m_z at most 0.019). alpha x tau_leak = 5.0-5.7 ps. |
| **G3 seed noise** (README build, tau_m = 503.8 ps, seeds 100-109) | validation 95.20 +- 0.22 %, range 94.72-95.46 %, no failed seeds. Pooled with G0 (15 runs): sd about 0.31 pp. With 5 seeds per arm only differences of about 0.4 pp or more will stand out. |
| **G2 shuffled-label control** (tau_m 50, 503.8, 1200 ps, 2 seeds each) | validation 9.15-11.54 % in all 6 runs, i.e. chance. No label leakage in the pipeline. |
| **Exploratory tau_m sweep, 50-1200 ps** (validation, final epoch, 5 seeds) | 50 ps 81.65 %, 100 ps 92.49 %, 200 ps 94.19 %, 350 ps 94.84 %, 503.8 ps 95.02 %, 800 ps 95.53 %, 1200 ps 95.29 %. Steep rise to about 200 ps, flat from about 350 ps. The 50 ps arm was still improving at epoch 10 and may be undertrained. No clear optimum: 800 ps leads, 1200 ps is not distinguishable. |
| **Long tau_m, 2 / 3 / 5 ns** (validation, final epoch, 5 seeds; hypothesis H6) | 2 ns 95.61 %, 3 ns 95.42 %, 5 ns 95.43 %. No drop and no change past the 2.5 ns window. Paired gains over 503.8 ps are +0.26 to +0.59 pp with paired sd 0.4-0.6 pp, so not resolved from noise. H6 (window artefact) is not supported in this range. |

Caveats that matter: validation accuracy is measured on a fixed 10k hold-out of a 50k training split (the
authors train on 60k); the runner, not the upstream code, is treated as the main suspect for any gap, and a
stricter replication (unmodified upstream script, seed 42, 60k training) is planned but not yet run.

## Physics behind the tau_m arms (`physics/`)

* `macrospin_tau.py`: LLG macrospin; tau_leak ~ 1 / (alpha (1 - Ku1 / K_c)), K_c about 5.6e5 J/m^3.
* `fit_tau_vs_ku1.py`: fit the leak constant from a mumax3 `table.txt`.
* `Ku1_tau_sweep.mx3`: mumax3 sweep over Ku1. **Untested above Ku1 = 2.5e5**; needs a fast GPU (gate G1b).

Damping alpha covers tau_m at or below 503.8 ps; anisotropy only lengthens it above 503.8 ps. See protocol rows 6, 10, 11, 13-17.

## Setup (Windows, PowerShell)
```powershell
git clone --recurse-submodules https://github.com/SKBiswas1998/ternary-snn-taum-study
cd ternary-snn-taum-study
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```
Upstream (`upstream/mtj`, pinned submodule) is used verbatim: `runner/core_loader.py` slices out the neuron and
network classes and only overrides `THRESHOLD`, `NEG_THRESHOLD`, `TAU_LEAK_PS` before the model is built.

## Running
```powershell
python runner\selftest.py                       # runner checks; must pass
python runner\make_jobs.py g0                   # also: g2, g3, explore, ext
python runner\launch.py jobs\g0.csv --workers 3
python runner\collect.py                        # merge ledger and print summary
python runner\g1_neuron_check.py upstream\mtj\ternary_mtj\thmtj.py "<folder with LIF_equation_current_1>"
```
Job sets: `g0` replication (10), `g3` seed noise (10), `g2` shuffled-label control (6), `explore` tau_m sweep
50-1200 ps x 5 seeds (35), `ext` tau_m = 2, 3, 5 ns x 5 seeds (15, tests the 2.5 ns window, hypothesis H6).
All except `g0` use the README build (negative threshold 0.90). `launch.py` is resumable: finished (arm, seed)
pairs are skipped. About 45 min per run with 3 workers on an i7-10510U, 8 GB; close other apps first
(about 0.85 GB per worker). To stop the laptop sleeping overnight see `powercfg /change standby-timeout-ac 0`.

## Rules the code enforces
* Test accuracy is computed only in phase `confirm` (once, final epoch) and phase `G0` (every epoch, to
  replicate the authors' best-test-epoch protocol). Exploratory and control phases use validation only.
* `collect.py` never prints test accuracy for `confirm` (blinding).
* Same seed gives the same initial weights and data order in every arm (paired design).
* Reserved seeds: explore/G0 0-4, G3 100-109, tune 200-201; confirm seeds are drawn at freeze.
* Finished runs are never re-run (diverged runs count as failures); crashes may be re-run.

## Layout
`protocol/` protocol source and PDF (v0.2) | `runner/` training, job generation, collection, G1 check |
`jobs/` job CSVs | `ledger/` one JSON per run and logs | `physics/` macrospin and mumax3 scripts | `upstream/mtj` pinned upstream code.

## Not done yet
Stricter replication of G0 (unmodified upstream script, seed 42, 60k training; the runner is the main suspect for the
gap, so this will be rerun whatever the sweep shows); gate G1b (mumax3 at Ku1 of 3, 4, 5e5 on a fast GPU);
per-arm learning-rate tuning and more epochs (the 50 ps arm); a T = 50 control and a linear-leak arm (not yet in the
runner); protocol v0.3 (record G0-G3 and sweep results, rule for non-learning seeds); protocol freeze and confirmatory
runs (12 seeds per arm, heavy part on Kaggle); answers to the open questions in protocol Appendix C (which build
produced the authors' number; Joy's accuracy-vs-anisotropy data and which device parameters she varied).
