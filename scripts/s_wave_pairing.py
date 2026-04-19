#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
s_wave_pairing.py

Usage Example:
python3 /home/users/phoenixm/scripts/s_wave_pairing.py \
  --path /scratch/users/phoenixm/dqmc_runs/U-6_6x6_tp0_nflux0/n0.6_resistivity/ \
  --output_path /scratch/users/phoenixm/dqmc_runs/U-6_6x6_tp0_nflux0/n0.6_resistivity/ \
  --relpath_list \
    "T0.05_beta20_U-6/mu-0.462713/" \
    "T0.1_beta10_U-6/mu-0.463656/" \
    "T0.125_beta8_U-6/mu-0.464428/" \
    "T0.166667_beta6_U-6/mu-0.466893/" \
    "T0.2_beta5_U-6/mu-0.470034/" \
    "T0.222222_beta4.5_U-6/mu-0.472615/" \
    "T0.25_beta4_U-6/mu-0.47691/" \
    "T0.285714_beta3.5_U-6/mu-0.484274/" \
    "T0.333333_beta3_U-6/mu-0.496175/" \
    "T0.4_beta2.5_U-6/mu-0.515708/"\
    "T0.5_beta2_U-6/mu-0.5489/" \
    "T0.666667_beta1.5_U-6/mu-0.610407/" \
    "T1_beta1_U-6/mu-0.755037/" \
    "T2_beta0.5_U-6/mu-1.3016/" \
    "T4_beta0.25_U-6/mu-2.68617/" \
    "T8_beta0.125_U-6/mu-5.85011/" 
'''

import os, sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))

import util

def get_meas(file: str):
    n, sign, nsamp, beta, U, Nx, Ny, pair_sw, double_occ = util.load_file(
        file, "meas_eqlt/density", "meas_eqlt/sign",
        "meas_eqlt/n_sample", "metadata/beta", "metadata/U",
        "metadata/Nx", "metadata/Ny", "meas_eqlt/pair_sw", "meas_eqlt/double_occ")
    n = float(np.asarray(n, dtype=float).reshape(-1)[0])
    sign = float(np.asarray(sign, dtype=float).reshape(-1)[0])
    nsamp = int(np.asarray(nsamp).reshape(-1)[0])
    beta = float(np.asarray(beta, dtype=float).reshape(-1)[0])
    U = float(np.asarray(U, dtype=float).reshape(-1)[0])
    Nx = int(np.asarray(Nx).reshape(-1)[0])
    Ny = int(np.asarray(Ny).reshape(-1)[0])
    pair_sw = np.asarray(pair_sw, dtype=float).reshape(-1)
    double_occ = float(np.asarray(double_occ, dtype=float).reshape(-1)[0])
    return n, sign, nsamp, beta, U, Nx, Ny, pair_sw, double_occ

def pairing(pair_sw, double_occ, Nx, Ny):
    N = Nx * Ny
    pair_sw = np.asarray(pair_sw, dtype=float).reshape(-1)
    if pair_sw.size != N:
        raise ValueError(
            f"Expected meas_eqlt/pair_sw to have length {N}, got {pair_sw.size}."
        )

    # meas_eqlt/pair_sw is the displacement-averaged equal-time correlator
    # <b_i b_{i+r}^\dagger>, stored over the N translation-inequivalent lattice displacements.
    # For a real, translation-invariant system, the onsite s-wave pair-field structure factor is
    # Ps = 2 * sum_{r != 0} pair_sw[r] + pair_sw[0] + double_occ.
    return 2.0 * np.sum(pair_sw[1:]) + pair_sw[0] + double_occ

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Directory containing HDF5 bin files")
    p.add_argument("--output_path",
                   help="Directory for output, will be created if needed")
    p.add_argument("--out_prefix", default="s_wave_pairing", help="Output file prefix")
    p.add_argument("--relpath_list", nargs="+", required=True,
                    help=("List of relative paths. Example: 'T0.05_beta20_U-6/mu-0.462713/'"))
    args = p.parse_args()
    path = args.path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Base path not found or not a directory: {path}")
    output_path = args.output_path if args.output_path is not None else path
    os.makedirs(output_path, exist_ok=True)

    Ts_all = []
    Ps_all = []
    Ps_err_all = []
    beta_all = []

    for relpath in args.relpath_list:
        files = sorted(glob.glob(os.path.join(path, relpath, "*.h5")))
        if not files:
            raise FileNotFoundError(f"No .h5 files found in {relpath}")

        beta0, U0, Nx0, Ny0 = util.load_file(files[0], "metadata/beta", "metadata/U", "metadata/Nx", "metadata/Ny")
        Nx = int(np.asarray(Nx0).reshape(-1)[0])
        Ny = int(np.asarray(Ny0).reshape(-1)[0])
        N = Nx * Ny

        bins_Ps = []
        bins_sign = []
        bins_nsamp = []
        beta_ref = None
        U_ref = None

        for file in files:
            n, sign, nsamp, beta, U, Nx_f, Ny_f, pair_sw, double_occ = get_meas(file)
            if Nx_f * Ny_f != N:
                raise ValueError(
                    f"Nx and Ny from {file} are not consistent with Nx and Ny from the first bin."
                )
            if beta_ref is None:
                beta_ref = beta
                U_ref = U
            else:
                if not np.isclose(beta, beta_ref):
                    raise ValueError(f"Inconsistent beta across bins in {relpath}: {beta} vs {beta_ref}")
                if not np.isclose(U, U_ref):
                    raise ValueError(f"Inconsistent U across bins in {relpath}: {U} vs {U_ref}")

            Ps = pairing(pair_sw, double_occ, Nx, Ny)
            bins_Ps.append(Ps)
            bins_sign.append(sign)
            bins_nsamp.append(nsamp)

        bins_Ps = np.asarray(bins_Ps, dtype=float)
        bins_sign = np.asarray(bins_sign, dtype=float)
        bins_nsamp = np.asarray(bins_nsamp, dtype=float)

        if bins_Ps.size == 0:
            raise ValueError(f"No valid bins found in {relpath}")
        if np.any(~np.isfinite(bins_Ps)):
            raise ValueError(f"Non-finite Ps encountered in {relpath}")
        if np.any(~np.isfinite(bins_sign)):
            raise ValueError(f"Non-finite sign encountered in {relpath}")

        jk_sign = util.jackknife_noniid(
            bins_nsamp, bins_sign,
            f=lambda ns, s: (s / ns).real,
        )
        jk_Ps = util.jackknife_noniid(
            bins_nsamp, bins_sign, bins_Ps,
            f=lambda ns, s, ps: (ps / s).real,
        )

        sign_mean = float(np.asarray(jk_sign[0]).reshape(-1)[0])
        sign_err = float(np.asarray(jk_sign[1]).reshape(-1)[0])
        Ps_mean = float(np.asarray(jk_Ps[0]).reshape(-1)[0])
        Ps_err = float(np.asarray(jk_Ps[1]).reshape(-1)[0])

        relpath_clean = relpath.strip("/")
        safe_relpath = relpath_clean.replace("/", "__") if relpath_clean else "root"
        #out_txt = os.path.join(output_path, f"{args.out_prefix}__{safe_relpath}.txt")

        header = "relpath beta T U Nx Ny nbin sign sign_err Ps Ps_err"
        T = np.inf if beta_ref == 0 else 1.0 / beta_ref
        #with open(out_txt, "w", encoding="utf-8") as fh:
        #    fh.write(header + "\n")
        #    fh.write(
        #        f"{relpath_clean} {beta_ref:.16g} {T:.16g} {U_ref:.16g} {Nx:d} {Ny:d} "
        #        f"{bins_Ps.size:d} {sign_mean:.16g} {sign_err:.16g} {Ps_mean:.16g} {Ps_err:.16g}\n"
        #    )

        print(
            f"[OK] {relpath_clean}: beta={beta_ref:.6g}, T={T:.6g}, U={U_ref:.6g}, "
            f"Ps={Ps_mean:.8g} ± {Ps_err:.3g}, sign={sign_mean:.8g} ± {sign_err:.3g}"
        )
        Ts_all.append(T)
        Ps_all.append(Ps_mean)
        Ps_err_all.append(Ps_err)
        beta_all.append(beta_ref)

    if len(Ts_all) == 0:
        raise ValueError("No Ps(T) points were collected.")

    Ts_all = np.asarray(Ts_all, dtype=float)
    Ps_all = np.asarray(Ps_all, dtype=float)
    Ps_err_all = np.asarray(Ps_err_all, dtype=float)
    beta_all = np.asarray(beta_all, dtype=float)

    order = np.argsort(Ts_all)
    Ts_all = Ts_all[order]
    Ps_all = Ps_all[order]
    Ps_err_all = Ps_err_all[order]
    beta_all = beta_all[order]

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    ax.errorbar(Ts_all, Ps_all, yerr=Ps_err_all, fmt='o-', capsize=3)
    ax.set_xlabel("T")
    ax.set_ylabel(r"$P_s$")
    ax.set_title(r"$P_s$ vs $T$")
    fig.tight_layout()

    out_png = os.path.join(output_path, f"{args.out_prefix}__Ps_vs_T.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved plot to {out_png}")

if __name__ == "__main__":
    main()
