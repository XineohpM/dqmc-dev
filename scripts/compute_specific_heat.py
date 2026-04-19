#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_specific_heat.py
phoenixm@stanford.edu

Compute specific heat from a beta scan by reconstructing the total energy in the
same spirit as data_analysis.py:

    E_site = K_site + U * D_site

with K_site reconstructed from meas_eqlt/g00.
File reading and jackknife are done through util.py.
"""

import os
import sys
import glob
from pathlib import Path
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))
import util

def get_meas(dir: str):
    density, sign, g00, nsamp, beta, U, Nx, Ny, double_occ = util.load_file(
        dir, "meas_eqlt/density", "meas_eqlt/sign", "meas_eqlt/g00", 
        "meas_eqlt/n_sample", "metadata/beta", "metadata/U",
        "metadata/Nx", "metadata/Ny", "meas_eqlt/double_occ")
    Nx = int(np.asarray(Nx).reshape(-1)[0])
    Ny = int(np.asarray(Ny).reshape(-1)[0])
    g00 = np.asarray(g00, dtype=float)
    g00 = np.reshape(g00, (-1, Nx, Ny), order='F')
    density = float(np.asarray(density, dtype=float).reshape(-1)[0])
    sign = float(np.asarray(sign, dtype=float).reshape(-1)[0])
    nsamp = int(np.asarray(nsamp).reshape(-1)[0])
    beta = float(np.asarray(beta, dtype=float).reshape(-1)[0])
    U = float(np.asarray(U, dtype=float).reshape(-1)[0])
    Nx = int(np.asarray(Nx).reshape(-1)[0])
    Ny = int(np.asarray(Ny).reshape(-1)[0])
    double_occ = float(np.asarray(double_occ, dtype=float).reshape(-1)[0])
    N = Nx * Ny
    return density, sign, g00, nsamp, beta, U, Nx, Ny, N, double_occ

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True,
                   help="Parent directory containing T_* subfolders.")
    p.add_argument("--subdir_pfx", default="T_*",
                   help="Sub-directory prefix under path.")
    p.add_argument("--output_pfx", default="specific_heat_",
                   help="Prefix for outputs written to --path.")
    p.add_argument("--tp", type=float, default=0.0,
                   help="Next-nearest-neighbor hopping tp used in the g00-based kinetic reconstruction.")
    return p.parse_args()

def main():
    args = parse_args()
    path = args.path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Base path not found or not a directory: {path}")
    subdir_pfx = args.subdir_pfx
    subdirs = sorted(glob.glob(os.path.join(path, subdir_pfx)))
    if not subdirs:
        raise SystemExit(f"No subdirs matching {subdir_pfx} under {path}.")
    tp = float(args.tp)

    T_list = []
    E_mean_list = []
    E_err_list = []
    E_bins_list = []
    sign_bins_list = []
    nsamp_bins_list = []

    for d in subdirs:
        files = sorted(glob.glob(os.path.join(d, "*.h5")))
        if not files:
            raise FileNotFoundError(f"No .h5 files found in {d}")
        
        beta_0 = util.load_firstfile(os.path.join(d, ""), "metadata/beta")
        beta_0 = float(np.asarray(beta_0[0], dtype=float).reshape(-1)[0])
        T = 1/beta_0
        T_list.append(T)

        bins_E = []
        bins_sign = []
        bins_nsamp = []
        for file in files:
            density, sign, g00, nsamp, beta, U, Nx, Ny, N, double_occ = get_meas(file)
            if not np.isclose(beta, beta_0):
                raise ValueError(f"Inconsistent beta across bins in {file}: {beta} vs {beta_0}")
            #factor of 2 appears because g00 = 0.5*(gup + gdn), I think
            k1 = 2*(g00[:,0,1]+g00[:,1,0]+g00[:,0,Ny-1]+g00[:,Nx-1,0]) 
            #TODO: check if tp terms are correct
            k2 = 2*tp*(g00[:,1,1] + g00[:,1,Ny-1] + g00[:,Nx-1,1]+ g00[:,Nx-1,Ny-1])
            v = U * double_occ
            e = float(np.asarray(k1 + k2 + v, dtype=float).reshape(-1)[0])
            bins_E.append(e)
            bins_sign.append(sign)
            bins_nsamp.append(nsamp)
    
        bins_E = np.asarray(bins_E, dtype=float)
        bins_sign = np.asarray(bins_sign, dtype=float)
        bins_nsamp = np.asarray(bins_nsamp, dtype=float)

        if bins_E.size == 0:
            raise ValueError(f"No valid bins found in {d}")
        if np.any(~np.isfinite(bins_E)):
            raise ValueError(f"Non-finite energy encountered in {d}")
        if np.any(~np.isfinite(bins_sign)):
            raise ValueError(f"Non-finite sign encountered in {d}")
        
        jk_sign = util.jackknife_noniid(
            bins_nsamp, bins_sign,
            f=lambda ns, s: (s / ns).real,
        )
        jk_E = util.jackknife_noniid(
            bins_nsamp, bins_sign, bins_E,
            f=lambda ns, s, e: (e / s).real,
        )

        sign_mean = float(np.asarray(jk_sign[0]).reshape(-1)[0])
        sign_err = float(np.asarray(jk_sign[1]).reshape(-1)[0])
        E_mean = float(np.asarray(jk_E[0]).reshape(-1)[0])
        E_err = float(np.asarray(jk_E[1]).reshape(-1)[0])
        E_mean_list.append(E_mean)
        E_err_list.append(E_err)
        E_bins_list.append(bins_E)
        sign_bins_list.append(bins_sign)
        nsamp_bins_list.append(bins_nsamp)

    if not T_list:
        raise SystemExit("No valid temperature points found.")

    T = np.array(T_list, dtype=float)
    E = np.array(E_mean_list, dtype=float)
    E_err = np.array(E_err_list, dtype=float)

    order = np.argsort(T)
    T = T[order]
    E = E[order]
    E_err = E_err[order]
    E_bins_list = [E_bins_list[i] for i in order]
    sign_bins_list = [sign_bins_list[i] for i in order]
    nsamp_bins_list = [nsamp_bins_list[i] for i in order]

    T_mid = 0.5 * (T[1:] + T[:-1])
    T_diff = T[1:] - T[:-1]
    dE = E[1:] - E[:-1]
    dE_err = np.sqrt(E_err[1:]**2 + E_err[:-1]**2)
    C = dE / T_diff
    C_err = dE_err / np.abs(T_diff)

    out_T = os.path.join(path, f"{args.output_pfx}_T.npy")
    # x-axis for specific heat C(T)
    out_T_mid = os.path.join(path, f"{args.output_pfx}_T_mid.npy")
    out_E = os.path.join(path, f"{args.output_pfx}_E_mean.npy")
    out_Ee = os.path.join(path, f"{args.output_pfx}_E_stderr.npy")
    out_C = os.path.join(path, f"{args.output_pfx}_specific_heat.npy")
    out_Ce = os.path.join(path, f"{args.output_pfx}_specific_heat_stderr.npy")
    np.save(out_T, T)
    np.save(out_T_mid, T_mid)
    np.save(out_E, E)
    np.save(out_Ee, E_err)
    np.save(out_C, C)
    np.save(out_Ce, C_err)

    plt.figure()
    plt.errorbar(
        T_mid, C, yerr=C_err,
        fmt="o", ms=4,
        linestyle="none",
        capsize=2,
        elinewidth=1.0,
    )

    if T_mid.size >= 3:
        x = np.log(T_mid)
        xs = np.linspace(x.min(), x.max(), 400)
        pchip = PchipInterpolator(x, C)
        plt.plot(np.exp(xs), pchip(xs), "-", lw=1.5)

    plt.xscale("log")
    plt.xlabel("T")
    plt.ylabel("C")
    plt.grid()
    #plt.legend()
    plt.tight_layout()
    out_fig = os.path.join(path, f"{args.output_pfx}_C_vs_T.png")
    plt.savefig(out_fig, dpi=160)

    print("Saved:", out_T, out_T_mid, out_E, out_Ee, out_C, out_Ce, out_fig)


if __name__ == "__main__":
    main()