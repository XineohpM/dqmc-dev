import os, sys
import glob
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))

import util


def get_meas(file: str):
    sign, nsamp, beta, U, Nx, Ny = util.load_file(
        file, "meas_eqlt/sign", "meas_eqlt/n_sample", "metadata/beta",
        "metadata/U", "metadata/Nx", "metadata/Ny")
    sign = float(np.asarray(sign, dtype=float).reshape(-1)[0])
    nsamp = int(np.asarray(nsamp).reshape(-1)[0])
    beta = float(np.asarray(beta, dtype=float).reshape(-1)[0])
    U = float(np.asarray(U, dtype=float).reshape(-1)[0])
    Nx0 = int(np.asarray(Nx).reshape(-1)[0])
    Ny0 = int(np.asarray(Ny).reshape(-1)[0])
    nsite = Nx0 * Ny0
    return sign, nsamp, beta, U, nsite, Nx0, Ny0


def n_from_path(path: str):
    parts = Path(path).parts
    if "half_filling" in parts:
        return 1.0
    for part in parts:
        m = re.fullmatch(r"n([0-9]+(?:\.[0-9]+)?)", part)
        if m:
            return float(m.group(1))
    raise ValueError(f"Cannot infer filling n from path: {path}")


def beta_key(beta: float):
    return f"{beta:.12g}"


def process_one_dir(run_path: str):
    files = sorted(glob.glob(os.path.join(run_path, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {run_path}")

    sign0, nsamp0, beta0, U0, Nx0, Ny0 = util.load_firstfile(
        os.path.join(run_path, ""),
        "meas_eqlt/sign", "meas_eqlt/n_sample", "metadata/beta",
        "metadata/U", "metadata/Nx", "metadata/Ny")
    Nx = int(np.asarray(Nx0).reshape(-1)[0])
    Ny = int(np.asarray(Ny0).reshape(-1)[0])
    N = Nx * Ny
    beta = float(np.asarray(beta0, dtype=float).reshape(-1)[0])
    U = float(np.asarray(U0, dtype=float).reshape(-1)[0])
    T = 1.0 / beta
    n = n_from_path(run_path)

    bins_nsamp = []
    bins_sign = []

    for file in files:
        sign, nsamp, beta_i, U_i, nsite, Nx_m, Ny_m = get_meas(file)
        if not np.isclose(beta, beta_i):
            raise ValueError(f"Beta of bin ({beta_i}) does not match first file beta ({beta})")
        if not np.isclose(U, U_i):
            raise ValueError(f"U of bin ({U_i}) does not match first file U ({U})")
        if Nx_m != Nx or Ny_m != Ny or nsite != N:
            raise ValueError(f"Inconsistent lattice size in {file}: got {Nx_m}x{Ny_m}, expected {Nx}x{Ny}")
        bins_nsamp.append(nsamp)
        bins_sign.append(sign)

    bins_nsamp = np.asarray(bins_nsamp, dtype=float)
    bins_sign = np.asarray(bins_sign, dtype=float)

    sign_jk = util.jackknife_noniid(
        bins_nsamp,
        bins_sign,
        f=lambda nsamp, sign: sign / nsamp,
    )
    sign_mean = float(sign_jk[0])
    sign_err = float(sign_jk[1])

    print(
        f"[ok] {run_path} n={n:.12g} T={T:.12g} beta={beta:.12g} "
        f"sign={sign_mean:.12g} err={sign_err:.12g}"
    )
    return {
        "dir": run_path,
        "n": n,
        "T": T,
        "beta": beta,
        "beta_key": beta_key(beta),
        "U": U,
        "sign": sign_mean,
        "sign_err": sign_err,
    }


def run_dirs_from_file_glob(base_path: str, pattern: str):
    files = sorted(glob.glob(os.path.join(base_path, pattern)))
    dirs = sorted({os.path.dirname(f) for f in files if f.endswith(".h5")})
    return dirs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=".", help="Base directory containing half_filling/ and n*/ directories")
    p.add_argument("--half_filling_glob", default="half_filling/T_*/*.h5",
                   help="Glob for half-filled HDF5 bins relative to --path")
    p.add_argument("--doping_glob", default="n*/T*_beta*_U*/mu*/*.h5",
                   help="Glob for doped HDF5 bins relative to --path")
    p.add_argument("--out_prefix", default="sign",
                   help="Output prefix for saved arrays and figure")
    p.add_argument("--out_fig", default=None,
                   help="Output figure path; default is <path>/<out_prefix>_vs_n.png")
    args = p.parse_args()

    base_path = args.path
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found or not a directory: {base_path}")

    run_dirs = []
    run_dirs.extend(run_dirs_from_file_glob(base_path, args.half_filling_glob))
    run_dirs.extend(run_dirs_from_file_glob(base_path, args.doping_glob))
    run_dirs = sorted(set(run_dirs), key=lambda d: (n_from_path(d), d))
    if not run_dirs:
        raise FileNotFoundError(
            f"No .h5 files found with globs {args.half_filling_glob!r} and {args.doping_glob!r}"
        )

    rows = [process_one_dir(run_dir) for run_dir in run_dirs]

    n_values = sorted({r["n"] for r in rows})
    rows_by_n_beta = {(r["n"], r["beta_key"]): r for r in rows}
    beta_keys_by_n = {
        n: {r["beta_key"] for r in rows if np.isclose(r["n"], n)}
        for n in n_values
    }
    common_beta_keys = sorted(
        set.intersection(*beta_keys_by_n.values()),
        key=lambda key: next(r["T"] for r in rows if r["beta_key"] == key),
    )
    if not common_beta_keys:
        raise ValueError("No temperatures are common to all available n values")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(np.linspace(0.0, 1.0, len(common_beta_keys)))

    for color, key in zip(colors, common_beta_keys):
        T = next(r["T"] for r in rows if r["beta_key"] == key)
        xs = []
        ys = []
        yerrs = []
        for n in n_values:
            r = rows_by_n_beta[(n, key)]
            xs.append(n)
            ys.append(r["sign"])
            yerrs.append(r["sign_err"])
        ax.errorbar(
            xs, ys, yerr=yerrs,
            fmt="o-",
            color=color,
            ecolor=color,
            markerfacecolor=color,
            markeredgecolor=color,
            capsize=3,
            linewidth=1.5,
            markersize=4,
            label=f"T/t = {T:.6g}",
        )

    ax.set_xlim(min(n_values), max(n_values))
    ax.set_xticks(n_values)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\langle \text{sign} \rangle$")
    ax.set_yscale("log")
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    out_fig = args.out_fig
    if out_fig is None:
        out_fig = os.path.join(base_path, f"{args.out_prefix}_vs_n.png")
    out_dir = os.path.dirname(out_fig)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    order = np.lexsort((
        np.asarray([r["T"] for r in rows], dtype=float),
        np.asarray([r["n"] for r in rows], dtype=float),
    ))
    n_arr = np.asarray([rows[i]["n"] for i in order], dtype=float)
    T_arr = np.asarray([rows[i]["T"] for i in order], dtype=float)
    sign_arr = np.asarray([rows[i]["sign"] for i in order], dtype=float)
    sign_err_arr = np.asarray([rows[i]["sign_err"] for i in order], dtype=float)

    np.save(os.path.join(base_path, f"{args.out_prefix}_n.npy"), n_arr)
    np.save(os.path.join(base_path, f"{args.out_prefix}_T.npy"), T_arr)
    np.save(os.path.join(base_path, f"{args.out_prefix}_mean.npy"), sign_arr)
    np.save(os.path.join(base_path, f"{args.out_prefix}_err.npy"), sign_err_arr)

    print("Saved sign arrays and figure:")
    print(f"  {os.path.join(base_path, args.out_prefix + '_n.npy')}")
    print(f"  {os.path.join(base_path, args.out_prefix + '_T.npy')}")
    print(f"  {os.path.join(base_path, args.out_prefix + '_mean.npy')}")
    print(f"  {os.path.join(base_path, args.out_prefix + '_err.npy')}")
    print(f"  {out_fig}")


if __name__ == "__main__":
    main()
