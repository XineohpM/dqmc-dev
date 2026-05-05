#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_specific_heat.py
phoenixm@stanford.edu

1.  Compute specific heat from a beta scan by reconstructing the total energy and 
    calculate the derivative:

        E_site = K_site + U * D_site
        C_v = dE / dT

    with K_site reconstructed from eqaul time measuerment meas_eqlt/g00.

2.  Compute specific heat from energy fluctuation:

        C_v = (<E^2> - <E>^2) / T^2

    with equal time energy correlation measurements meas_eqlt/kk, meas_eqlt/kv, 
    meas_eqlt/vk and meas_eqlt/vv.

File reading and jackknife are done through util.py.
"""

import os
import sys
import glob
from pathlib import Path
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

def get_meas_corr(dir: str):
    # v here refers to double occupancy in meas.c
    # Make sure to multiply by U when using energy correlator containing v
    kk, kv, vv, kn, vn, nn, Nx, Ny = util.load_file(dir, "meas_eqlt/kk", "meas_eqlt/kv", 
                                                "meas_eqlt/vv", "meas_eqlt/kn", "meas_eqlt/vn",
                                                "meas_eqlt/nn", "metadata/Nx", "metadata/Ny")
    kk = np.asarray(kk, dtype=float).reshape(-1)
    kv = np.asarray(kv, dtype=float).reshape(-1)
    vv = np.asarray(vv, dtype=float).reshape(-1)
    kn = np.asarray(kn, dtype=float).reshape(-1)
    vn = np.asarray(vn, dtype=float).reshape(-1)
    nn = np.asarray(nn, dtype=float).reshape(-1)
    Nx = int(np.asarray(Nx).reshape(-1)[0])
    Ny = int(np.asarray(Ny).reshape(-1)[0])
    return kk, kv, vv, kn, vn, nn

def V(U, double_occ):
    # Per-site potential energy
    return U*double_occ

def K(g00, Nx, Ny, tp=0):
    # Per-site kinetic energy
    k1 = 2*(g00[:,0,1]+g00[:,1,0]+g00[:,0,Ny-1]+g00[:,Nx-1,0])
    k2 = 2*tp*(g00[:,1,1] + g00[:,1,Ny-1] + g00[:,Nx-1,1]+ g00[:,Nx-1,Ny-1])
    return k1+k2

def energy(file, tp=0):
    # Per-site energy
    density, sign, g00, nsamp, beta, U, Nx, Ny, N, double_occ = get_meas(file)
    e = V(U, double_occ) + K(g00, Nx, Ny, tp)
    e = float(np.asarray(e, dtype=float).reshape(-1)[0])
    return e


def diff(subdirs, tp):
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
            e = energy(file, tp)
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
    E_mean = np.array(E_mean_list, dtype=float)
    E_err = np.array(E_err_list, dtype=float)

    order = np.argsort(T)
    T = T[order]
    E_mean = E_mean[order]
    E_err = E_err[order]
    E_bins_list = [E_bins_list[i] for i in order]
    sign_bins_list = [sign_bins_list[i] for i in order]
    nsamp_bins_list = [nsamp_bins_list[i] for i in order]

    T_mid = 0.5 * (T[1:] + T[:-1])
    T_diff = T[1:] - T[:-1]
    dE = E_mean[1:] - E_mean[:-1]
    dE_err = np.sqrt(E_err[1:]**2 + E_err[:-1]**2)
    C = dE / T_diff
    C_err = dE_err / np.abs(T_diff)
    return T, T_mid, E_mean, E_err, C, C_err

def fluc(subdirs, tp):
    T_list = []
    C_mean_list = []
    C_err_list = []

    for d in subdirs:
        files = sorted(glob.glob(os.path.join(d, "*.h5")))
        if not files:
            raise FileNotFoundError(f"No .h5 files found in {d}")

        beta_0 = util.load_firstfile(os.path.join(d, ""), "metadata/beta")
        beta_0 = float(np.asarray(beta_0[0], dtype=float).reshape(-1)[0])
        T_list.append(1 / beta_0)

        bins_e = []
        bins_h2 = []
        bins_hn = []
        bins_n = []
        bins_n2 = []
        bins_sign = []
        bins_nsamp = []

        Nsite_0 = None

        for file in files:
            density, sign, g00, nsamp, beta, U, Nx, Ny, Nsite, double_occ = get_meas(file)
            kk, kv, vv, kn, vn, nn = get_meas_corr(file)

            if not np.isclose(beta, beta_0):
                raise ValueError(f"Inconsistent beta across bins in {file}: {beta} vs {beta_0}")

            if Nsite_0 is None:
                Nsite_0 = Nsite
            elif Nsite != Nsite_0:
                raise ValueError(f"Inconsistent system size in {file}: {Nsite} vs {Nsite_0}")

            k = float(np.asarray(K(g00, Nx, Ny, tp), dtype=float).reshape(-1)[0])
            e = k + U * double_occ

            h2 = kk.sum() + 2 * U * kv.sum() + U**2 * vv.sum()
            hn = kn.sum() + U * vn.sum()
            n = density
            n2 = nn.sum()

            bins_e.append(e)
            bins_h2.append(h2)
            bins_hn.append(hn)
            bins_n.append(n)
            bins_n2.append(n2)
            bins_sign.append(sign)
            bins_nsamp.append(nsamp)

        bins_e = np.asarray(bins_e, dtype=float)
        bins_h2 = np.asarray(bins_h2, dtype=float)
        bins_hn = np.asarray(bins_hn, dtype=float)
        bins_n = np.asarray(bins_n, dtype=float)
        bins_n2 = np.asarray(bins_n2, dtype=float)
        bins_sign = np.asarray(bins_sign, dtype=float)
        bins_nsamp = np.asarray(bins_nsamp, dtype=float)

        def f_cv(ns, s, e, h2, hn, n, n2):
            e_mean = (e / s).real
            h2_mean = (h2 / s).real
            hn_mean = (hn / s).real
            n_mean = (n / s).real
            n2_mean = (n2 / s).real

            cov_hh = h2_mean - Nsite_0 * e_mean**2
            cov_hn = hn_mean - Nsite_0 * e_mean * n_mean
            cov_nn = n2_mean - Nsite_0 * n_mean**2

            if np.any(np.abs(cov_nn) < 1e-14):
                return np.nan

            return beta_0**2 * (cov_hh - cov_hn**2 / cov_nn)

        jk_C = util.jackknife_noniid(
            bins_nsamp,
            bins_sign,
            bins_e,
            bins_h2,
            bins_hn,
            bins_n,
            bins_n2,
            f=f_cv,
        )

        C_mean = float(np.asarray(jk_C[0]).reshape(-1)[0])
        C_err = float(np.asarray(jk_C[1]).reshape(-1)[0])

        C_mean_list.append(C_mean)
        C_err_list.append(C_err)

    if not T_list:
        raise SystemExit("No valid temperature points found.")

    T = np.array(T_list, dtype=float)
    C = np.array(C_mean_list, dtype=float)
    C_err = np.array(C_err_list, dtype=float)

    order = np.argsort(T)
    T = T[order]
    C = C[order]
    C_err = C_err[order]
    return T, C, C_err

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True,
                   help="Parent directory containing T_* subfolders.")
    p.add_argument("--subdir_pfx", default="T_*",
                   help="Sub-directory prefix under path.")
    p.add_argument("--output_pfx", default="specific_heat",
                   help="Prefix for outputs written to --path.")
    p.add_argument("--tp", type=float, default=0.0,
                   help="Next-nearest-neighbor hopping tp used in the g00-based kinetic reconstruction.")
    p.add_argument("--mode", choices=["fluc", "diff", "both"], required=True,
                   help="fluc -> calculate Cv from energy fluctuation; diff -> calculate Cv with derivative; both -> do both.")
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

    do_diff = False
    do_fluc = False
    if args.mode == "diff" or args.mode == "both": do_diff = True
    if args.mode == "fluc" or args.mode == "both": do_fluc = True

    if do_diff:
        T_d, T_mid_d, E_d, E_err_d, C_d, C_err_d = diff(subdirs, tp)
        out_T = os.path.join(path, f"{args.output_pfx}_T_diff.npy")
        # x-axis for specific heat C(T)
        out_T_mid = os.path.join(path, f"{args.output_pfx}_T_mid_diff.npy")
        out_E = os.path.join(path, f"{args.output_pfx}_E_mean_diff.npy")
        out_Ee = os.path.join(path, f"{args.output_pfx}_E_stderr_diff.npy")
        out_C = os.path.join(path, f"{args.output_pfx}_specific_heat_diff.npy")
        out_Ce = os.path.join(path, f"{args.output_pfx}_specific_heat_stderr_diff.npy")
        np.save(out_T, T_d)
        np.save(out_T_mid, T_mid_d)
        np.save(out_E, E_d)
        np.save(out_Ee, E_err_d)
        np.save(out_C, C_d)
        np.save(out_Ce, C_err_d)

        plt.figure()
        plt.errorbar(
            T_mid_d, C_d, yerr=C_err_d,
            fmt="o", ms=4,
            linestyle="none",
            capsize=2,
            elinewidth=1.0,
        )

        if T_mid_d.size >= 3:
            x = np.log(T_mid_d)
            xs = np.linspace(x.min(), x.max(), 400)
            pchip = PchipInterpolator(x, C_d)
            plt.plot(np.exp(xs), pchip(xs), "-", lw=1.5)

        plt.xscale("log")
        plt.xlabel("T")
        plt.ylabel("C")
        plt.grid()
        #plt.legend()
        plt.tight_layout()
        out_fig = os.path.join(path, f"{args.output_pfx}_C_vs_T_diff.png")
        plt.savefig(out_fig, dpi=160)

        print("Saved:", out_T, out_T_mid, out_E, out_Ee, out_C, out_Ce, out_fig)
    
    if do_fluc:
        T_f, C_f, C_err_f = fluc(subdirs, tp)
        out_T = os.path.join(path, f"{args.output_pfx}_T_fluc.npy")
        out_C = os.path.join(path, f"{args.output_pfx}_specific_heat_fluc.npy")
        out_Ce = os.path.join(path, f"{args.output_pfx}_specific_heat_stderr_fluc.npy")
        np.save(out_T, T_f)
        np.save(out_C, C_f)
        np.save(out_Ce, C_err_f)

        plt.figure()
        plt.errorbar(
            T_f, C_f, yerr=C_err_f,
            fmt="o", ms=4,
            linestyle="none",
            capsize=2,
            elinewidth=1.0,
        )

        if T_f.size >= 3:
            x = np.log(T_f)
            xs = np.linspace(x.min(), x.max(), 400)
            pchip = PchipInterpolator(x, C_f)
            plt.plot(np.exp(xs), pchip(xs), "-", lw=1.5)

        plt.xscale("log")
        plt.xlabel("T")
        plt.ylabel("C")
        plt.grid()
        #plt.legend()
        plt.tight_layout()
        out_fig = os.path.join(path, f"{args.output_pfx}_C_vs_T_fluc.png")
        plt.savefig(out_fig, dpi=160)

        print("Saved:", out_T, out_C, out_Ce, out_fig)

    if args.mode == "both":
        plt.figure()
        plt.errorbar(
            T_f, C_f, yerr=C_err_f,
            fmt="o", ms=4,
            linestyle="none",
            capsize=2,
            elinewidth=1.0,
            label="energy fluctuation"
        )
        plt.errorbar(
            T_mid_d, C_d, yerr=C_err_d,
            fmt="o", ms=4,
            linestyle="none",
            capsize=2,
            elinewidth=1.0,
            label="derivative"
        )

        if T_f.size >= 3:
            x = np.log(T_f)
            xs = np.linspace(x.min(), x.max(), 400)
            pchip = PchipInterpolator(x, C_f)
            plt.plot(np.exp(xs), pchip(xs), "-", lw=1.5)

        if T_mid_d.size >= 3:
            x = np.log(T_mid_d)
            xs = np.linspace(x.min(), x.max(), 400)
            pchip = PchipInterpolator(x, C_d)
            plt.plot(np.exp(xs), pchip(xs), "-", lw=1.5)

        plt.xscale("log")
        plt.xlabel("T")
        plt.ylabel("C")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        out_fig = os.path.join(path, f"{args.output_pfx}_C_vs_T_both.png")
        plt.savefig(out_fig, dpi=160)



if __name__ == "__main__":
    main()