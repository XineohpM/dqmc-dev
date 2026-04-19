#!/usr/bin/env python3
"""
extract_local_moment.py
phoenixm@stanford.edu

Extract local moment <m_z^2>(T) from DQMC HDF5 outputs.

Definition (single-orbital):
    m_z = n_up - n_dn
    <m_z^2> = <n> - 2 <n_up n_dn>

This script scans ROOT/T_*/ for *.h5, reads meas_eqlt results, computes per-file <m_z^2>,
then averages over files at each temperature and plots vs T.

It is tailored to the dqmc-dev HDF5 layout where the following datasets exist:
    /meas_eqlt/n_sample   (scalar)
    /meas_eqlt/double_occ (shape {1} accumulator over samples)
    /meas_eqlt/density    (shape {1} accumulator over samples)
Optionally, /meas_eqlt/density_u and /meas_eqlt/density_d may exist.

Notes on normalization:
For scalar datasets stored as shape {1}, the value is typically accumulated over n_sample;
we divide by n_sample to get the sample average.

Usage example:
python3 /home/users/phoenixm/scripts/extract_local_moment.py \
  --root /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_C \
  --out_prefix U-6_n6x6_C_local_moment \
  --U -6
"""

import os
import glob
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

PATH_NS = "meas_eqlt/n_sample"
PATH_DO = "meas_eqlt/double_occ"      # <n_u n_d> accumulator
PATH_N  = "meas_eqlt/density"         # <n_u + n_d> accumulator
PATH_NU = "meas_eqlt/density_u"
PATH_ND = "meas_eqlt/density_d"
PATH_SIGN = "meas_eqlt/sign"          # optional; may be absent in some builds


def _read_scalar_avg(f: h5py.File, path: str, n_sample: int) -> float:
    """Read dataset at `path` and return sample average as float.

    Supports scalar datasets stored as shape () or (1,) that represent sums over samples.
    Also supports per-sample arrays (shape (n_sample,)) by taking mean.
    """
    x = np.array(f[path][()])
    if x.shape == () or x.size == 1:
        return float(x.reshape(-1)[0]) / float(n_sample)
    return float(np.mean(x))


def _per_file_mz2(fp: str, tol_density: float, tol_spin: float, strict: bool) -> tuple[float, bool, bool]:
    """Compute <m_z^2> for a single HDF5 file.

    Returns:
        mz2: <m_z^2>
        dens_mismatch: whether |(density_u+density_d) - density| exceeds tol_density (if both exist)
        spin_mismatch: whether |density_u - density_d| exceeds tol_spin (if both exist)
    """
    with h5py.File(fp, "r") as f:
        ns = int(np.array(f[PATH_NS]))

        # double occupancy D = <n_up n_dn>
        D = _read_scalar_avg(f, PATH_DO, ns)

        # total density n = <n_up + n_dn>
        dens_mismatch = False
        spin_mismatch = False

        n = None
        if PATH_N in f:
            n = _read_scalar_avg(f, PATH_N, ns)

        nu = nd = None
        if PATH_NU in f and PATH_ND in f:
            nu = _read_scalar_avg(f, PATH_NU, ns)
            nd = _read_scalar_avg(f, PATH_ND, ns)

            # Consistency check: density_u + density_d equals density (if density exists)
            if n is not None:
                dens_mismatch = abs((nu + nd) - n) > float(tol_density)

            # Spin symmetry check
            spin_mismatch = abs(nu - nd) > float(tol_spin)

            if strict and (dens_mismatch or spin_mismatch):
                raise ValueError(
                    f"Check failed for {fp}: n={n}, nu={nu}, nd={nd}, "
                    f"nu+nd={(nu+nd)}, |nu+nd-n|={abs((nu+nd)-(n if n is not None else 0.0))}, "
                    f"|nu-nd|={abs(nu-nd)}"
                )

        # If no total density was read yet, fall back to nu+nd if present
        if n is None:
            if nu is not None and nd is not None:
                n = nu + nd
            else:
                raise KeyError("Missing density dataset: need /meas_eqlt/density (or density_u+density_d)")

        mz2 = n - 2.0 * D

        if not np.isfinite(mz2):
            raise ValueError("mz2 is not finite")

        return float(mz2), dens_mismatch, spin_mismatch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract <m_z^2>(T) from DQMC HDF5 files.")
    p.add_argument("--root", required=True, help="Root directory containing T_* subfolders.")
    p.add_argument("--U", type=float, required=True, help="On-site interaction U used for the atomic-limit reference curve.")
    p.add_argument("--h5_glob", default="*.h5", help="Glob for h5 files inside each T_* folder.")
    p.add_argument("--out_prefix", default="local_moment", help="Prefix for output .npy/.png files.")
    p.add_argument("--skip_missing", action="store_true",
                   help="Skip T_* folders that have zero readable h5 files instead of erroring.")
    p.add_argument("--tol_density", type=float, default=1e-6,
                   help="Abs tolerance for checking density_u+density_d == density (default: 1e-6).")
    p.add_argument("--tol_spin", type=float, default=1e-6,
                   help="Abs tolerance for checking spin symmetry |density_u-density_d| (default: 1e-6).")
    p.add_argument("--strict_checks", action="store_true",
                   help="If set, raise an error on the first density/spin check violation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root

    T_list = []
    mz2_mean_list = []
    mz2_err_list = []

    t_dirs = sorted(glob.glob(os.path.join(root, "T_*")))
    if not t_dirs:
        raise SystemExit(f"No T_* subdirectories found under: {root}")

    for d in t_dirs:
        base = os.path.basename(d)
        try:
            T = float(base.split("_", 1)[1])
        except Exception:
            continue

        fps = sorted(glob.glob(os.path.join(d, args.h5_glob)))
        if not fps:
            if args.skip_missing:
                print(f"[SKIP] {base}: no files matching {args.h5_glob}")
                continue
            raise FileNotFoundError(f"No h5 files matching {args.h5_glob} under {d}")

        vals = []
        bad = 0
        dens_bad = 0
        spin_bad = 0
        for fp in fps:
            try:
                mz2_i, dens_mismatch, spin_mismatch = _per_file_mz2(
                    fp, args.tol_density, args.tol_spin, args.strict_checks
                )
                vals.append(mz2_i)
                if dens_mismatch:
                    dens_bad += 1
                if spin_mismatch:
                    spin_bad += 1
            except Exception:
                bad += 1

        vals = np.array(vals, dtype=float)
        if vals.size == 0:
            if args.skip_missing:
                print(f"[SKIP] {base}: no valid files (failed={bad})")
                continue
            raise RuntimeError(f"{base}: all files failed (failed={bad})")

        mean = float(vals.mean())
        err = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0

        T_list.append(T)
        mz2_mean_list.append(mean)
        mz2_err_list.append(err)

        print(
            f"[OK] {base}: N={vals.size} failed={bad}  <mz2>={mean:.6g}  err={err:.3g}  "
            f"dens_mismatch={dens_bad}  spin_mismatch={spin_bad}"
        )

    # sort by T
    T = np.array(T_list, dtype=float)
    mz2 = np.array(mz2_mean_list, dtype=float)
    mz2err = np.array(mz2_err_list, dtype=float)
    idx = np.argsort(T)
    T, mz2, mz2err = T[idx], mz2[idx], mz2err[idx]

    # save
    np.save(os.path.join(root, f"{args.out_prefix}_T.npy"), T)
    np.save(os.path.join(root, f"{args.out_prefix}_mz2.npy"), mz2)
    np.save(os.path.join(root, f"{args.out_prefix}_mz2_err.npy"), mz2err)

    # plot
    plt.figure()
    plt.errorbar(T, mz2, yerr=mz2err, fmt="o", capsize=2)

    # Smooth-looking connecting curve via interpolation in log10(T)
    x_dense = None
    T_dense = None
    if T.size >= 2:
        x = np.log10(T)
        x_dense = np.linspace(x.min(), x.max(), 400)
        T_dense = 10**x_dense
        y_dense = np.interp(x_dense, x, mz2)
        # Light-blue connecting curve
        plt.plot(T_dense, y_dense, linewidth=2.0, alpha=0.45, color="#4DA3FF")

    # Atomic limit (t=0) reference curve:
    #   <m_z^2> = 1 / (exp(-U/(2T)) + 1)
    # Use a dense T grid if available; otherwise evaluate on the data points.
    if T_dense is None:
        T_dense = T
    mz2_atomic = 1.0 / (np.exp(-args.U / (2.0 * T_dense)) + 1.0)
    plt.plot(T_dense, mz2_atomic, linestyle="--", linewidth=1.8, color="orange", alpha=0.8,
             label=r"atomic limit $t=0$")

    plt.legend(frameon=False)

    plt.xscale("log")
    plt.xlabel("T")
    plt.ylabel(r"$\langle m_z^2 \rangle$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(root, f"{args.out_prefix}_mz2_vs_T.png")
    plt.savefig(out_png, dpi=160)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()