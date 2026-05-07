#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import glob
from pathlib import Path
import numpy as np
import argparse
import matplotlib.pyplot as plt

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))
import util

def parse_n_from_path(path):
    for part in Path(path).parts:
        m = re.fullmatch(r"n([0-9.]+)", part)
        hf = re.fullmatch(r"half_filling", part)
        if m:
            return float(m.group(1))
        elif hf:
            return 1.0
    raise ValueError(f"Cannot parse n from path: {path}")


def parse_T_from_path(path):
    for part in Path(path).parts:
        m = re.match(r"T([0-9.eE+-]+)_beta", part)
        n = re.match(r"T_([0-9.eE+-]+)", part)
        p = re.match(r"beta([0-9.eE+-]+)", part)
        q = re.match(r"beta_([0-9.eE+-]+)", part)
        if m:
            return float(m.group(1))
        elif n:
            return float(n.group(1))
        elif p:
            return 1.0/float(p.group(1))
        elif q:
            return 1.0/float(q.group(1))
    raise ValueError(f"Cannot parse T from path: {path}")

def get_meas_file(file: str):
    density, sign, nsamp, beta, U, Nx, Ny, double_occ = util.load_file(
        file, "meas_eqlt/density", "meas_eqlt/sign", 
        "meas_eqlt/n_sample", "metadata/beta", "metadata/U",
        "metadata/Nx", "metadata/Ny", "meas_eqlt/double_occ")
    Nx = int(np.asarray(Nx).reshape(-1)[0])
    Ny = int(np.asarray(Ny).reshape(-1)[0])
    density = float(np.asarray(density, dtype=float).reshape(-1)[0])
    sign = float(np.asarray(sign, dtype=float).reshape(-1)[0])
    nsamp = int(np.asarray(nsamp).reshape(-1)[0])
    beta = float(np.asarray(beta, dtype=float).reshape(-1)[0])
    U = float(np.asarray(U, dtype=float).reshape(-1)[0])
    double_occ = float(np.asarray(double_occ, dtype=float).reshape(-1)[0])
    N = Nx * Ny
    return density, sign, nsamp, beta, U, Nx, Ny, N, double_occ

def get_meas_dir(dir: str):
    files = sorted(glob.glob(os.path.join(dir, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {dir}")
    nt = parse_n_from_path(dir)
    Tt = parse_T_from_path(dir)
    
    density0, sign0, nsamp0, beta0, U0, Nx0, Ny0, double_occ0 = util.load_firstfile(
        dir, "meas_eqlt/density", "meas_eqlt/sign", "meas_eqlt/n_sample", 
        "metadata/beta", "metadata/U", "metadata/Nx", "metadata/Ny", 
        "meas_eqlt/double_occ"
    )
    Nx0 = int(np.asarray(Nx0).reshape(-1)[0])
    Ny0 = int(np.asarray(Ny0).reshape(-1)[0])
    N0= Nx0*Ny0
    if not np.isclose(1.0/Tt, beta0):
        raise ValueError(f"T mismatch in {dir}: T from path={Tt}, beta={beta0}")
    
    density_list = []
    double_occ_list = []
    sign_list = []
    nsamp_list = []
    for file in files:
        density, sign, nsamp, beta, U, Nx, Ny, N, double_occ = get_meas_file(file)
        if not np.isclose(beta, beta0):
            raise ValueError(f"T mismatch in {dir}: T from path={Tt}, beta={beta0}, 1/beta={1.0/beta0}")
        if not np.isclose(U, U0):
            raise ValueError(f"U mismatch in {file}: U={U}, expected U0={U0}")
        if not np.isclose(N, N0):
            raise ValueError(f"N mismatch in {file}: N={N}, expected N0={N0}")
        density_list.append(density)
        double_occ_list.append(double_occ)
        sign_list.append(sign)
        nsamp_list.append(nsamp)

    density_arr = np.asarray(density_list, dtype=float)
    double_occ_arr = np.asarray(double_occ_list, dtype=float)
    sign_arr = np.asarray(sign_list, dtype=float)
    nsamp_arr = np.asarray(nsamp_list, dtype=float)
    single_occ_arr = density_arr - 2.0 * double_occ_arr

    # Jackknife
    jk_double_occ = util.jackknife_noniid(nsamp_arr, sign_arr, double_occ_arr)
    double_occ_mean = jk_double_occ[0]
    double_occ_err = jk_double_occ[1]
    jk_density = util.jackknife_noniid(nsamp_arr, sign_arr, density_arr)
    density_mean = jk_density[0]
    density_err = jk_density[1]
    jk_single_occ = util.jackknife_noniid(nsamp_arr, sign_arr, single_occ_arr)
    single_occ_mean = jk_single_occ[0]
    single_occ_err = jk_single_occ[1]

    if not np.isclose(density_mean, nt, atol=1e-2):
        raise ValueError(f"density mismatch in {dir}: measured density={density_mean}, target n={nt}")
    
    return double_occ_mean, double_occ_err, single_occ_mean, single_occ_err, nt, Tt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Directory containing HDF5 bin files")
    p.add_argument("--glob", default="n*/T*_beta*_U*/mu*/")
    p.add_argument("--output_path",
                   help="Directory for output, will be created if needed")
    p.add_argument("--out_prefix", default="double_occ", help="Output file prefix")
    args = p.parse_args()
    path = args.path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Base path not found or not a directory: {path}")
    output_path = args.output_path if args.output_path is not None else path
    os.makedirs(output_path, exist_ok=True)

    dirs = sorted(glob.glob(os.path.join(path, args.glob)))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(
            f"No directories found with pattern: {os.path.join(path, args.glob)}"
        )
    results_by_n = {}
    for d in dirs:
        double_occ_mean, double_occ_err, single_occ_mean, single_occ_err, nt, Tt = get_meas_dir(d)
        results_by_n.setdefault(nt, []).append([double_occ_mean, double_occ_err, 
                                                single_occ_mean, single_occ_err, Tt])

    # Save per-n numpy arrays and plot one figure for each n.
    fig_overlay, ax_overlay = plt.subplots(figsize=(6.0, 4.5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (nt, rows) in enumerate(sorted(results_by_n.items())):
        rows = sorted(rows, key=lambda x: x[4])
        double_occ_arr = np.asarray([r[0] for r in rows], dtype=float)
        double_occ_err_arr = np.asarray([r[1] for r in rows], dtype=float)
        single_occ_arr = np.asarray([r[2] for r in rows], dtype=float)
        single_occ_err_arr = np.asarray([r[3] for r in rows], dtype=float)
        T_arr = np.asarray([r[4] for r in rows], dtype=float)

        color = color_cycle[i % len(color_cycle)]
        n_label = f"n={nt:g}"
        n_tag = f"n{nt:g}"

        np.save(os.path.join(output_path, f"{args.out_prefix}_{n_tag}_T.npy"), T_arr)
        np.save(os.path.join(output_path, f"{args.out_prefix}_{n_tag}_double_occ_mean.npy"), double_occ_arr)
        np.save(os.path.join(output_path, f"{args.out_prefix}_{n_tag}_double_occ_err.npy"), double_occ_err_arr)
        np.save(os.path.join(output_path, f"{args.out_prefix}_{n_tag}_single_occ_mean.npy"), single_occ_arr)
        np.save(os.path.join(output_path, f"{args.out_prefix}_{n_tag}_single_occ_err.npy"), single_occ_err_arr)

        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.errorbar(
            T_arr, double_occ_arr, yerr=double_occ_err_arr,
            fmt="o", ms=4, linestyle="-", capsize=2,
            elinewidth=1.0, linewidth=1.2, color=color,
            label=fr"{n_label}, double occ.",
        )
        ax.errorbar(
            T_arr, single_occ_arr, yerr=single_occ_err_arr,
            fmt="s", ms=4, linestyle="--", capsize=2,
            elinewidth=1.0, linewidth=1.2, color=color,
            label=fr"{n_label}, single occ.",
        )
        ax.set_xlabel("T")
        ax.set_ylabel("occupation probability")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, f"{args.out_prefix}_{n_tag}_occ.png"), dpi=200)
        plt.close(fig)

        ax_overlay.errorbar(
            T_arr, double_occ_arr, yerr=double_occ_err_arr,
            fmt="o", ms=4, linestyle="-", capsize=2,
            elinewidth=1.0, linewidth=1.2, color=color,
            label=fr"{n_label}, double occ.",
        )
        ax_overlay.errorbar(
            T_arr, single_occ_arr, yerr=single_occ_err_arr,
            fmt="s", ms=4, linestyle="--", capsize=2,
            elinewidth=1.0, linewidth=1.2, color=color,
            label=fr"{n_label}, single occ.",
        )

    ax_overlay.set_xlabel("T")
    ax_overlay.set_ylabel("occupation probability")
    #ax_overlay.set_title("Double occupancy vs T")
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.legend(frameon=False)
    fig_overlay.tight_layout()
    fig_overlay.savefig(os.path.join(output_path, f"{args.out_prefix}_occ_overlay.png"), dpi=200)
    plt.close(fig_overlay)

if __name__ == "__main__":
    main()