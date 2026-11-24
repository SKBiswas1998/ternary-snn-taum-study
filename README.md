# ternary-snn-taum-study

Pre-registered test of whether the membrane time constant tau_m changes accuracy in the ternary
MTJ-inspired spiking network (`ternary_mtj/thmtj.py` of the upstream repo). Read `protocol/protocol.pdf` first.
**Protocol v0.1 is a draft; nothing is frozen and no confirmatory run has been made.**

## Setup (Windows, PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy
git submodule add <URL of MTJ-Inspired-Nonlinear-Neuron> upstream/mtj
git submodule update --init
```
Upstream is used verbatim: `runner/core_loader.py` slices out the neuron/network classes and only
overrides `THRESHOLD`, `NEG_THRESHOLD`, `TAU_LEAK_PS` before the model is built.

## First 20 minutes
```powershell
python runner/selftest.py          # runner checks (split, pairing, overrides) - must pass
python runner/bench.py             # minutes per run on THIS machine (decides parallel workers)
python runner/make_jobs.py g0      # also: g2, g3, explore
python runner/launch.py jobs/g0.csv --workers 3     # gate G0: replication, 10 runs
python runner/collect.py           # merge ledger + summary
```
`launch.py` is resumable: re-running skips finished (arm, seed). Close other apps first (each worker ~0.85 GB).

## Rules the code enforces
* Test accuracy is computed only in phase `confirm` (once, final epoch) and phase `G0` (every epoch, to
  replicate the authors' best-test-epoch protocol and measure the winner's-curse gap). Nowhere else.
* `collect.py` never prints test accuracy for `confirm` (blinding).
* Same seed -> same initial weights and data order in every arm (paired design).
* Reserved seeds: explore/G0 0-4, G3 100-109, tune 200-201; confirm seeds are drawn at freeze.
* Finished runs are never re-run (diverged runs count as failures); crashes may be re-run.

## Not done yet
G1 (analytic leak check of the neuron), per-arm learning-rate tuning jobs, the SHD pipeline,
Kaggle notebook, protocol v0.2, blinding key tooling.
