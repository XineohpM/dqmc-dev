#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from glob import glob
from pathlib import Path
import sys, os, re
import numpy as np
import argparse

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))

import util


def _validate_jackknife_denominators(sign):
    sign = np.asarray(sign)
    total = np.sum(sign)
    total_abs = np.sum(np.abs(sign))
    rtol = 1e-12
    if np.abs(total) <= rtol * total_abs:
        raise ValueError("total accumulated sign is too close to zero")
    leave_one_out = total - sign
    leave_one_out_abs = total_abs - np.abs(sign)
    if np.any(np.abs(leave_one_out) <= rtol * leave_one_out_abs):
        bad = np.flatnonzero(
            np.abs(leave_one_out) <= rtol * leave_one_out_abs
        )
        raise ValueError(
            "jackknife leave-one-out accumulated sign is too close to zero "
            f"when omitting bin(s) {bad[:10].tolist()}"
        )


def get_meas(mu_dir: str):
    nn, n, sign, mu, nsamp, beta, U, Nx, Ny = util.load(
        mu_dir, "meas_eqlt/nn", "meas_eqlt/density", "meas_eqlt/sign",
        "metadata/mu", "meas_eqlt/n_sample", "metadata/beta", "metadata/U",
        "metadata/Nx", "metadata/Ny")
    Nx0 = int(np.asarray(Nx).reshape(-1)[0])
    Ny0 = int(np.asarray(Ny).reshape(-1)[0])
    nsite = Nx0 * Ny0
    return nn, n, sign, mu, nsamp, beta, U, nsite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, 
                   help="Base directory named as n*, containing hdf5 files under n*/T*_beta*_U*/mu*/")
    p.add_argument("--output_path",  
                   help="Directory for output, will be created if needed")
    args = p.parse_args()
    base = os.path.expanduser(args.path)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Base path not found or not a directory: {base}")
    
    # Resolve output location TODO: check prefix
    if args.output_path is None:
        out_dir = "."
        out_prefix = os.path.join(out_dir, "compressibility")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out = os.path.expanduser(args.output_path)
        # If user passes a directory (existing or endswith /), write into it.
        if os.path.isdir(out) or out.endswith(os.sep):
            out_dir = out
            os.makedirs(out_dir, exist_ok=True)
            out_prefix = os.path.join(out_dir, "compressibility")
        else:
            # treat as a prefix
            out_prefix = out
            out_dir = os.path.dirname(out_prefix) or "."
            os.makedirs(out_dir, exist_ok=True)
    
    pattern_n = os.path.join(base, "n*", "T*_beta*_U*", "mu*")
    pattern_hf = os.path.join(base, "half_filling", "T_*")
    mu_dirs = sorted(glob(pattern_n)+glob(pattern_hf))
    if not mu_dirs:
        raise FileNotFoundError(f"No mu directories found with pattern: {pattern_n} and {pattern_hf}")
    
    rows = []
    n_bad = 0

    for mu_dir in mu_dirs:
        mu_dir_h5 = mu_dir if mu_dir.endswith(os.sep) else (mu_dir + os.sep)
        nn, n, sign, mu, nsamp, beta, U, nsite = get_meas(mu_dir_h5)
        
        # Infer target n and T from directory path
        hf = re.search(r"half_filling", mu_dir_h5)
        if hf:
            n_target = 1.0
            mT = re.search(r"/T_([0-9eE+\-\.]+)", mu_dir_h5)
            T = float(mT.group(1)) if mT else np.nan
        else:
            mnt = re.search(r"/n(0\.[0-9]+)(?:/|$)", mu_dir_h5)
            n_target = float(mnt.group(1)) if mnt else np.nan
            mT = re.search(r"/T([0-9eE+\-\.]+)_beta([0-9eE+\-\.]+)_U([0-9eE+\-\.]+)(?:/|$)", mu_dir_h5)
            T = float(mT.group(1)) if mT else np.nan

        # Ensure arrays are 1D over bins
        sign = np.asarray(sign).reshape(-1)
        nsamp = np.asarray(nsamp).reshape(-1)
        n = np.asarray(n).reshape(-1)
        nn = np.asarray(nn)

        mmax = np.max(nsamp)
        mask = (nsamp == mmax)
        if not mask.all():
            print(f"{mu_dir_h5} incomplete: {mask.sum()}/{mask.size}")
        sign = sign[mask]
        nsamp = nsamp[mask]
        n = n[mask]
        nn = nn[mask]

        # Keep the raw signed accumulators.  Physical observables are formed
        # once, as ratios of sums over bins, inside jackknife_noniid.
        density_numerator = n
        nn_numerator = nn.reshape(nn.shape[0], -1).sum(axis=1)

        # A single bin may have zero accumulated sign; only non-finite data or
        # non-positive sample counts make a bin invalid.
        valid = (
            np.isfinite(nsamp)
            & (nsamp > 0)
            & np.isfinite(sign)
            & np.isfinite(density_numerator)
            & np.isfinite(nn_numerator)
        )
        nsamp = nsamp[valid]
        sign = sign[valid]
        density_numerator = density_numerator[valid]
        nn_numerator = nn_numerator[valid]

        if sign.size < 3:
            n_bad += 1
            rows.append((n_target, T, float(beta[0]) if np.size(beta) else np.nan, float(U[0]) if np.size(U) else np.nan,
                         float(mu[0]) if np.size(mu) else np.nan, np.nan, np.nan, np.nan, np.nan,
                         np.nan, mu_dir))
            continue

        # Use a non-i.i.d. ratio jackknife on raw signed accumulators, then
        # form chi:
        # chi = beta * ( <S> - nsite * <n>^2 )
        bval = float(np.asarray(beta).reshape(-1)[0])
        _validate_jackknife_denominators(sign)

        def f(sum_nsamp, sum_s, sum_nn, sum_n):
            S_mean = sum_nn / sum_s
            n_mean = sum_n / sum_s
            return bval * (S_mean - nsite * (n_mean ** 2))

        def f_inverse(sum_nsamp, sum_s, sum_nn, sum_n):
            chi_value = f(sum_nsamp, sum_s, sum_nn, sum_n)
            if not np.isfinite(chi_value) or chi_value == 0.0:
                return np.nan
            return 1.0 / chi_value

        jk_chi = util.jackknife_noniid(
            nsamp,
            sign,
            nn_numerator,
            density_numerator,
            f=f,
        )

        jk_1_over_chi = util.jackknife_noniid(
            nsamp,
            sign,
            nn_numerator,
            density_numerator,
            f=f_inverse,
        )
        chi = float(jk_chi[0])
        chi_err = float(jk_chi[1])
        chi_inv = float(jk_1_over_chi[0])
        chi_inv_err = float(jk_1_over_chi[1])

        # Also report the measured <n> at this mu for sanity
        jk_n = util.jackknife_noniid(
            nsamp,
            sign,
            density_numerator,
        )
        n_mean = float(jk_n[0])
        dn = n_mean - n_target if np.isfinite(n_target) else np.nan

        rows.append((n_target, T, bval, float(np.asarray(U).reshape(-1)[0]), float(np.asarray(mu).reshape(-1)[0]),
                     chi, chi_err, chi_inv, chi_inv_err, dn, mu_dir))

    # Sort by target n then T
    rows.sort(key=lambda r: (r[0], r[1]))

    out_tsv = out_prefix + "_chi.tsv"
    header = "n_target\tT\tbeta\tU\tmu\tchi\tchi_err\tchi_inv\tchi_inv_err\tdn\tmu_dir\n"
    with open(out_tsv, "w") as f:
        f.write(header)
        for (n_target, T, beta, U, mu, chi, chi_err, chi_inv, chi_inv_err, dn, mu_dir) in rows:
            f.write(
                f"{n_target:.12g}\t{T:.12g}\t{beta:.12g}\t{U:.12g}\t{mu:.12g}\t"
                f"{chi:.12g}\t{chi_err:.12g}\t{chi_inv:.12g}\t{chi_inv_err:.12g}\t{dn:.12g}\t{mu_dir}\n"
            )

    n_total = len(rows)
    n_nan_chi = sum((not np.isfinite(r[5])) for r in rows)
    print(f"[OK] wrote {out_tsv}")
    print(f"Total mu dirs: {n_total}")
    print(f"Rows with NaN chi: {n_nan_chi}")
    if n_bad:
        print(f"Rows failed to process (too few valid bins): {n_bad}")

    # Save one compact, self-describing array bundle per target filling.
    by_n = {}
    for (n_target, T, beta, U, mu, chi, chi_err, chi_inv, chi_inv_err, dn, mu_dir) in rows:
        if not (np.isfinite(n_target) and np.isfinite(T)):
            continue
        by_n.setdefault(n_target, []).append(
            (T, chi, chi_err, chi_inv, chi_inv_err)
        )

    for n_target in sorted(by_n):
        points = sorted(by_n[n_target], key=lambda point: point[0])
        values = np.asarray(points, dtype=float)
        filling_tag = np.format_float_positional(n_target, trim="-")
        if "." not in filling_tag:
            filling_tag += ".0"
        output_npz = out_prefix + f"_n{filling_tag}.npz"
        np.savez(
            output_npz,
            n_target=np.asarray(n_target, dtype=float),
            T=values[:, 0],
            chi=values[:, 1],
            chi_err=values[:, 2],
            chi_inv=values[:, 3],
            chi_inv_err=values[:, 4],
        )
        print(f"[OK] wrote {output_npz}")


if __name__ == "__main__":
    main()
