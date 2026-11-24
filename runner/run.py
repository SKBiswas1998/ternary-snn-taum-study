#!/usr/bin/env python
"""Run ONE (arm, seed) training run and write one JSON result file.

Protocol rules enforced here (see protocol/protocol.pdf):
  * test accuracy is computed ONLY in phase 'confirm' (once, final epoch) and phase 'G0'
    (every epoch, to replicate the authors' best-test-epoch protocol). Never elsewhere.
  * a finished run (status ok/nan) is never re-run; crashes leave no result file and may be re-run.
  * same seed -> same initial weights and same data order in every arm (paired design).
"""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from core_loader import load_core          # noqa: E402
from data import load_mnist, load_synthetic  # noqa: E402

TEST_PHASES = {"G0": "every_epoch", "confirm": "final_only"}


def git_hash(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def evaluate(model, x, y, bs=500, want_stats=False):
    model.eval()
    correct, n = 0, 0
    acc = {k: 0.0 for k in ("fr1", "fr2", "fr3", "neg1", "neg2", "neg3", "out_abs")}
    for i in range(0, len(x), bs):
        xb, yb = x[i:i + bs], y[i:i + bs]
        if want_stats:
            s1, s2, s3 = model(xb, return_all=True)[:3]
            out = s3.sum(0)
            for k, s in (("1", s1), ("2", s2), ("3", s3)):
                acc["fr" + k] += s.abs().mean().item() * len(xb)       # firing rate (any sign)
                acc["neg" + k] += (s < 0).float().mean().item() * len(xb)  # negative-spike fraction
            acc["out_abs"] += out.abs().mean().item() * len(xb)         # summed-output-spike scale
        else:
            out = model(xb)[0].sum(0)
        correct += (out.argmax(1) == yb).sum().item()
        n += len(xb)
    stats = {k: v / n for k, v in acc.items()} if want_stats else None
    return 100.0 * correct / n, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["smoke", "G0", "G2", "G3", "tune", "explore", "confirm"])
    ap.add_argument("--arm", required=True, help="arm id or code (used in the file name)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tau-ps", type=float, default=503.8, help="membrane time constant tau_m in ps (dt fixed 100 ps)")
    ap.add_argument("--theta-pos", type=float, default=0.8)
    ap.add_argument("--theta-neg", type=float, default=0.9, help="README value; committed code has 1.6 (unreachable)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--shuffle-labels", action="store_true", help="G2 negative control: shuffle TRAIN labels")
    ap.add_argument("--synthetic", type=int, default=0, help="N fake training samples (smoke/timing ONLY)")
    ap.add_argument("--upstream", default=str(ROOT / "upstream" / "mtj" / "ternary_mtj" / "thmtj.py"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "ledger" / "runs"))
    a = ap.parse_args()

    if a.synthetic and a.phase not in ("smoke", "G3"):
        sys.exit("--synthetic is only allowed with phase smoke (timing)")
    run_id = f"{a.phase}__{a.arm}__s{a.seed}__lr{a.lr:g}__e{a.epochs}"
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run_id}.json"
    if out_file.exists():
        print(f"[skip] {run_id} (already done)")
        return

    t0 = time.time()
    torch.set_num_threads(a.threads)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    core = load_core(a.upstream)
    core.THRESHOLD, core.NEG_THRESHOLD, core.TAU_LEAK_PS = a.theta_pos, a.theta_neg, a.tau_ps

    d = load_synthetic(a.synthetic) if a.synthetic else load_mnist(a.data)
    xtr, ytr = d["xtr"], d["ytr"].clone()
    if a.shuffle_labels:                                   # fixed permutation, independent of run seed
        ytr = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(12345))]

    torch.manual_seed(a.seed)                              # INIT stream: same seed -> same weights in every arm
    np.random.seed(a.seed)
    model = core.HardMTJSpikingMLP(num_steps=a.steps)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    crit = nn.CrossEntropyLoss()
    shuffler = torch.Generator().manual_seed(10_000 + a.seed)   # DATA-ORDER stream, separate from init

    res = dict(train_loss=[], train_acc=[], val_acc=[], test_acc_per_epoch=[])
    status = "ok"
    n = len(xtr)
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=shuffler)
        tot_loss, correct, seen = 0.0, 0, 0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            xb, yb = xtr[idx], ytr[idx]
            opt.zero_grad()
            out = model(xb)[0].sum(0)
            loss = crit(out, yb)
            if not math.isfinite(loss.item()):
                status = "nan"
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            seen += len(xb)
        if status == "nan":
            print(f"[nan] {run_id} diverged in epoch {ep}")
            break
        res["train_loss"].append(tot_loss / seen)
        res["train_acc"].append(100.0 * correct / seen)
        if ep % a.val_every == 0 or ep == a.epochs:
            res["val_acc"].append((ep, evaluate(model, d["xva"], d["yva"])[0]))
        if TEST_PHASES.get(a.phase) == "every_epoch":
            res["test_acc_per_epoch"].append((ep, evaluate(model, d["xte"], d["yte"])[0]))
        print(f"[{run_id}] ep {ep}/{a.epochs} loss {res['train_loss'][-1]:.4f} "
              f"train {res['train_acc'][-1]:.2f}% val {res['val_acc'][-1][1]:.2f}%  ({time.time()-t0:.0f}s)", flush=True)

    final_val, stats, final_test = None, None, None
    if status == "ok":
        final_val, stats = evaluate(model, d["xva"], d["yva"], want_stats=True)
        if a.phase in TEST_PHASES:                          # the ONLY place test data is touched at the end
            final_test = evaluate(model, d["xte"], d["yte"])[0]
    if res["test_acc_per_epoch"]:
        res["best_test_epoch"] = max(res["test_acc_per_epoch"], key=lambda t: t[1])

    record = dict(
        id=run_id, phase=a.phase, arm=a.arm, status=status,
        params=dict(tau_ps=a.tau_ps, beta=math.exp(-100.0 / a.tau_ps) if a.tau_ps > 0 else 0.0,
                    theta_pos=a.theta_pos, theta_neg=a.theta_neg, lr=a.lr, seed=a.seed,
                    epochs=a.epochs, batch=a.batch, steps=a.steps, shuffle_labels=a.shuffle_labels,
                    synthetic=a.synthetic),
        final_val=final_val, final_test=final_test, stats=stats, **res,
        wall_s=time.time() - t0,
        env=dict(python=platform.python_version(), torch=torch.__version__, platform=platform.platform(),
                 machine=platform.machine(), cpu_count=os.cpu_count(), threads=a.threads,
                 runner_commit=git_hash(ROOT), upstream_commit=git_hash(Path(a.upstream).parent),
                 deterministic_flag=True),
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    tmp = out_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, out_file)                               # atomic: a half-written result never appears
    print(f"[done] {run_id} status={status} final_val={final_val} wall={record['wall_s']:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
