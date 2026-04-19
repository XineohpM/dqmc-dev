'''
extract_1_particle_local_g.py
phoenixm@stanford.edu

Extract imaginary-time single particle local Green's function from DQMC HDF5 outputs.

Usage examples:

'''
import glob
import argparse
import os, sys
import numpy as np
from pathlib import Path

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))

import util

def get_meas(file: str, dt: float):
    gu, gd, g, beta, sign, Nx, Ny, nsamp = util.load_file(
        file,
        "meas_uneqlt/gt0_u",
        "meas_uneqlt/gt0_d",
        "meas_uneqlt/gt0",
        "metadata/beta",
        "meas_uneqlt/sign",
        "metadata/Nx",
        "metadata/Ny",
        "meas_uneqlt/n_sample",
    )
    sign = float(np.asarray(sign, dtype=float).reshape(-1)[0])
    nsamp = int(np.asarray(nsamp).reshape(-1)[0])
    beta = float(np.asarray(beta, dtype=float).reshape(-1)[0])
    Nx = int(np.asarray(Nx).reshape(-1)[0])
    Ny = int(np.asarray(Ny).reshape(-1)[0])
    N = Nx * Ny
    L = int(round(beta / dt))
    try:
        gu = np.asarray(gu, dtype=float).reshape(L, N)
        gd = np.asarray(gd, dtype=float).reshape(L, N)
        g = np.asarray(g, dtype=float).reshape(L, N)
    except Exception as e:
        raise ValueError(
            f"Failed to reshape gt0 datasets to (L={L}, N={N}) for file {file}."
        ) from e
    return gu, gd, g, sign, L, nsamp

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Directory containing HDF5 bin files")
    p.add_argument("--output_path",
                   help="Directory for output, will be created if needed")
    p.add_argument("--out_prefix", default="1_particle_local_", help="Output file prefix")
    p.add_argument("--dt", required=True, help="Imaginary time step dt of the HDF5 files")
    args = p.parse_args()
    path = args.path
    dt = float(args.dt)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Base path not found or not a directory: {path}")
    output_path = args.output_path if args.output_path is not None else path
    os.makedirs(output_path, exist_ok=True)

    bins_g = []
    bins_gu = []
    bins_gd = []
    bins_sign = []
    bins_nsamp = []

    files = sorted(glob.glob(os.path.join(path, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {path}")
    for file in files:
        gu, gd, g, sign, L, nsamp = get_meas(file, dt)
        gu_loc = gu[:, 0]
        gd_loc = gd[:, 0]
        g_loc = 0.5 * (gu_loc + gd_loc)
        bins_g.append(g_loc)
        bins_gu.append(gu_loc)
        bins_gd.append(gd_loc)
        bins_sign.append(sign)
        bins_nsamp.append(nsamp)
    bins_g = np.asarray(bins_g, dtype=float)
    bins_gu = np.asarray(bins_gu, dtype=float)
    bins_gd = np.asarray(bins_gd, dtype=float)
    bins_sign = np.asarray(bins_sign, dtype=float)
    bins_nsamp = np.asarray(bins_nsamp, dtype=float)

    if bins_g.size == 0 or bins_gu.size == 0 or bins_gd.size == 0:
        raise ValueError(f"No valid bins found in {path}")
    if np.any(~np.isfinite(bins_g)):
        raise ValueError(f"Non-finite g encountered in {path}")
    if np.any(~np.isfinite(bins_sign)):
        raise ValueError(f"Non-finite sign encountered in {path}")
    if np.any(bins_nsamp <= 0):
        raise ValueError(f"Non-positive n_sample encountered in {path}")
    if np.any(bins_sign == 0):
        raise ValueError(f"Zero sign encountered in {path}; cannot divide by sign")

    jk_sign = util.jackknife_noniid(
        bins_nsamp, bins_sign,
        f=lambda ns, s: (s / ns).real,
    )
    jk_g = util.jackknife_noniid(
        bins_nsamp, bins_sign, bins_g,
        f=lambda ns, s, g: (g / (s[:, None] if np.ndim(s) else s)).real,
    )
    jk_gu = util.jackknife_noniid(
        bins_nsamp, bins_sign, bins_gu,
        f=lambda ns, s, gu: (gu / (s[:, None] if np.ndim(s) else s)).real,
    )
    jk_gd = util.jackknife_noniid(
        bins_nsamp, bins_sign, bins_gd,
        f=lambda ns, s, gd: (gd / (s[:, None] if np.ndim(s) else s)).real,
    )

    sign_mean = float(np.asarray(jk_sign[0]).reshape(-1)[0])
    sign_err = float(np.asarray(jk_sign[1]).reshape(-1)[0])
    g_mean = np.asarray(jk_g[0]).reshape(-1)
    g_err = np.asarray(jk_g[1]).reshape(-1)
    gu_mean = np.asarray(jk_gu[0]).reshape(-1)
    gu_err = np.asarray(jk_gu[1]).reshape(-1)
    gd_mean = np.asarray(jk_gd[0]).reshape(-1)
    gd_err = np.asarray(jk_gd[1]).reshape(-1)

    g_all = np.asarray(bins_g / bins_sign[:, None], dtype=float)
    gu_all = np.asarray(bins_gu / bins_sign[:, None], dtype=float)
    gd_all = np.asarray(bins_gd / bins_sign[:, None], dtype=float)

    tau = np.arange(L, dtype=float) * dt

    np.save(os.path.join(output_path, f"{args.out_prefix}tau.npy"), tau)
    np.save(os.path.join(output_path, f"{args.out_prefix}g_mean.npy"), g_mean)
    np.save(os.path.join(output_path, f"{args.out_prefix}g_err.npy"), g_err)
    np.save(os.path.join(output_path, f"{args.out_prefix}gu_mean.npy"), gu_mean)
    np.save(os.path.join(output_path, f"{args.out_prefix}gu_err.npy"), gu_err)
    np.save(os.path.join(output_path, f"{args.out_prefix}gd_mean.npy"), gd_mean)
    np.save(os.path.join(output_path, f"{args.out_prefix}gd_err.npy"), gd_err)
    np.save(os.path.join(output_path, f"{args.out_prefix}g_all.npy"), g_all)
    np.save(os.path.join(output_path, f"{args.out_prefix}gu_all.npy"), gu_all)
    np.save(os.path.join(output_path, f"{args.out_prefix}gd_all.npy"), gd_all)

    print(f"Processed {len(files)} bins from {path}")
    print(f"Average sign = {sign_mean:.12g} +/- {sign_err:.12g}")
    print(f"Saved outputs to {output_path}")

if __name__ == "__main__":
    main()