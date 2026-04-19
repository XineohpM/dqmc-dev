#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot imaginary-time current-current correlator JNJN_xx:
    *JNJN_xx_perbin.npy
    *JNJN_xx_mean.npy
"""

import os
import glob
import argparse
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

def _find_unique(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if len(hits) == 0:
        raise FileNotFoundError(f"No file matched pattern: {pattern}")
    if len(hits) > 1:
        msg = "\n".join(hits)
        raise RuntimeError(f"Multiple files matched pattern: {pattern}\n{msg}\n"
                           f"Please keep exactly one match (or refine your directory/pattern).")
    return hits[0]

def _load_1d_mean(path: str, L_expected: Optional[int] = None) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Mean file must be 1D after squeeze, got shape {arr.shape} from {path}")
    if (L_expected is not None) and (arr.shape[0] != L_expected):
        raise ValueError(f"Mean length mismatch: mean has L={arr.shape[0]} but perbin has L={L_expected}")
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, type=str,
                    help="Directory containing *JNJN_xx_perbin.npy and *_JNJN_xx_mean.npy")
    ap.add_argument("--beta", required=True, type=float, help="Inverse temperature beta")
    ap.add_argument("--dt", required=True, type=float, help="Imaginary-time step dt")
    ap.add_argument("--out", default=None, type=str,
                    help="If provided, save figure to this path (e.g. /path/to/plot.png). Otherwise show interactively.")
    args = ap.parse_args()

    base = os.path.abspath(os.path.expanduser(args.path))
    if not os.path.isdir(base):
        raise NotADirectoryError(f"--path is not a directory: {base}")

    perbin_file = _find_unique(os.path.join(base, "*JNJN_xx_perbin.npy"))
    mean_file   = _find_unique(os.path.join(base, "*JNJN_xx_mean.npy"))

    G_bins = np.load(perbin_file, allow_pickle=False)  # (nbin, L)
    if G_bins.ndim != 2:
        raise ValueError(f"Perbin file must be 2D (nbin,L). Got shape {G_bins.shape} from {perbin_file}")

    nbin, L = G_bins.shape
    G_mean = _load_1d_mean(mean_file, L_expected=L)
    mid = L // 2
    if L % 2 != 0: Lb2 = float((G_mean[mid] + G_mean[mid + 1])/2)
    else: Lb2 = float((G_mean[mid]))
    print("Lambda(beta/2)=", Lb2)

    tau = np.arange(L) * float(args.dt)
    if not np.isclose(float(args.beta), float(args.dt) * L):
        print(f"[WARN] beta != dt*L : beta={args.beta}, dt*L={args.dt}*{L}={float(args.dt)*L}")

    # Error bar from perbin ONLY
    std = G_bins.std(axis=0, ddof=1)
    sem = std / np.sqrt(nbin)

    # ---- plot: mean curve + uncertainty band ----
    plt.figure()
    plt.plot(tau, G_mean, lw=2.0, label="mean JNJN")
    plt.fill_between(tau, G_mean - sem, G_mean + sem, alpha=0.25, label="standard deviation")

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\langle J_x(\tau)\,J_x(0)\rangle$")
    plt.title(f"JNJN_xx  (nbin={nbin}, L={L}, dt={args.dt}, beta={args.beta})")
    plt.legend(loc="best")
    plt.tight_layout()

    if args.out:
        out = os.path.abspath(os.path.expanduser(args.out))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"[OK] Saved plot -> {out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()