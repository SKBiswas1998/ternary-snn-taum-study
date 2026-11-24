"""Fit the leak time constant from a mumax3 relaxation table.

Usage:  python fit_tau_vs_ku1.py path\\to\\table.txt

Model (from the closed form in Joy's LIF_august.m): u = mz/sqrt(1-mz^2) decays as
exp(-t/tau).  So ln(u) is a straight line in t with slope -1/tau.  We fit that line
over 0.02 < mz < 0.995, and also report the first-half and second-half slopes as a
check: a pure exponential in u gives the same tau in both halves.

If the table has a Ku1 column (column 5, from tableadd(Ku1)), runs are split by Ku1.
Otherwise the whole table is treated as one run.
"""
import sys
import numpy as np


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                pass
    return np.array(rows)


def fit_run(t, mz):
    """t in seconds. Returns (tau_all, tau_first_half, tau_second_half, n_points) in ns."""
    mz = np.abs(mz)
    keep = (mz > 0.02) & (mz < 0.995)
    # use only the monotone decay part: from the first kept point until mz first drops below 0.02
    below = np.where(mz <= 0.02)[0]
    if below.size:
        keep[below[0]:] = False
    if keep.sum() < 8:
        return None
    ts = (t[keep] - t[keep][0]) * 1e9
    z = np.log(mz[keep] / np.sqrt(1.0 - mz[keep] ** 2))
    slope = np.polyfit(ts, z, 1)[0]
    h = len(ts) // 2
    s1 = np.polyfit(ts[:h], z[:h], 1)[0]
    s2 = np.polyfit(ts[h:], z[h:], 1)[0]
    f = lambda s: -1.0 / s if s < 0 else float("nan")
    return f(slope), f(s1), f(s2), int(keep.sum())


def main(path):
    D = load(path)
    t, mz = D[:, 0], D[:, 3]
    if D.shape[1] >= 5:
        ku = D[:, 4]
        values = []
        for k in ku:
            if k not in values:
                values.append(k)
    else:
        ku = np.zeros(len(t))
        values = [float("nan")]
    print("Ku1 (J/m3)   tau_all(ns)  tau_1st_half  tau_2nd_half   n_pts")
    for k in values:
        sel = (ku == k) if D.shape[1] >= 5 else np.ones(len(t), bool)
        r = fit_run(t[sel], mz[sel])
        if r is None:
            print("%10.3g   no clean decay (perpendicular stable or too short)" % k)
        else:
            print("%10.3g   %9.3f   %11.3f   %11.3f   %5d" % (k, *r))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
