'''
plot_charge_order.py

Usage Example:

 python3 /home/users/phoenixm/scripts/plot_charge_order.py \
   --path /scratch/users/phoenixm/dqmc_runs/U-6_8x8_tp0_nflux0/half_filling \
   --glob 'T*_beta*_U*/mu*' \
   --out_prefix charge_order \
   --vlim 0.8
'''

import os, sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))

import util

def get_meas(file: str):
    nn, n, sign, nsamp, beta, U, Nx, Ny = util.load_file(
        file, "meas_eqlt/nn", "meas_eqlt/density", "meas_eqlt/sign",
        "meas_eqlt/n_sample", "metadata/beta", "metadata/U",
        "metadata/Nx", "metadata/Ny")
    nn = np.asarray(nn, dtype=float).reshape(-1)
    n = float(np.asarray(n, dtype=float).reshape(-1)[0])
    sign = float(np.asarray(sign, dtype=float).reshape(-1)[0])
    nsamp = int(np.asarray(nsamp).reshape(-1)[0])
    beta = float(np.asarray(beta, dtype=float).reshape(-1)[0])
    U = float(np.asarray(U, dtype=float).reshape(-1)[0])
    Nx0 = int(np.asarray(Nx).reshape(-1)[0])
    Ny0 = int(np.asarray(Ny).reshape(-1)[0])
    nsite = Nx0 * Ny0
    return nn, n, sign, nsamp, beta, U, nsite, Nx0, Ny0

def process_one_dir(run_path: str, out_prefix: str, vlim=None):
    files = sorted(glob.glob(os.path.join(run_path, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {run_path}")

    nn0, n0, sign0, nsamp0, beta0, U0, Nx0, Ny0 = util.load_firstfile(
        os.path.join(run_path, ""),
        "meas_eqlt/nn", "meas_eqlt/density", "meas_eqlt/sign",
        "meas_eqlt/n_sample", "metadata/beta", "metadata/U",
        "metadata/Nx", "metadata/Ny")
    Nx = int(np.asarray(Nx0).reshape(-1)[0])
    Ny = int(np.asarray(Ny0).reshape(-1)[0])
    N = Nx * Ny
    beta = float(np.asarray(beta0, dtype=float).reshape(-1)[0])
    T = 1.0 / beta

    bins_nsamp = []
    bins_sign = []
    bins_dens = []
    bins_nn = []

    for file in files:
        nn, n, sign, nsamp, beta_i, U, nsite, Nx_m, Ny_m = get_meas(file)
        beta_bin = float(np.asarray(beta_i, dtype=float).reshape(-1)[0])
        if not np.isclose(beta, beta_bin):
            raise ValueError(f"Beta of bin ({beta_bin}) does not match with beta of the firstfile ({beta})")
        if Nx_m != Nx or Ny_m != Ny or nsite != N:
            raise ValueError(f"Inconsistent lattice size in {file}: got {Nx_m}x{Ny_m}, expected {Nx}x{Ny}")
        if nn.shape[0] != N:
            raise ValueError(f"Unexpected nn length in {file}: got {nn.shape[0]}, expected {N}")
        bins_nsamp.append(nsamp)
        bins_sign.append(sign)
        bins_dens.append(n)
        bins_nn.append(nn.copy())

    bins_nsamp = np.asarray(bins_nsamp, dtype=float)
    bins_sign = np.asarray(bins_sign, dtype=float)
    bins_dens = np.asarray(bins_dens, dtype=float)
    bins_nn = np.asarray(bins_nn, dtype=float)

    C_jk = util.jackknife_noniid(
        bins_nsamp,
        bins_sign,
        bins_dens,
        bins_nn,
        f=lambda nsamp, sign, dens, nn: (nn.T / sign).T - ((dens / sign)**2)[..., None],
    )
    # Connected equal-time charge correlation: C(r) = <n_0 n_r> - <n>^2
    C_mean = np.asarray(C_jk[0], dtype=float)
    C_err = np.asarray(C_jk[1], dtype=float)

    C_map = C_mean.reshape(Ny, Nx)
    C_map_err = C_err.reshape(Ny, Nx)
    x = np.arange(Nx)[None, :]
    y = np.arange(Ny)[:, None]
    eta = (-1) ** (x + y)
    eta_flat = eta.reshape(-1)
    C_stag = eta * C_map

    # Equal-time CDW structure factor at Q=(pi,pi), with jackknife error.
    S_cdw_jk = util.jackknife_noniid(
        bins_nsamp,
        bins_sign,
        bins_dens,
        bins_nn,
        f=lambda nsamp, sign, dens, nn: np.sum(
            eta_flat * ((nn.T / sign).T - ((dens / sign)**2)[..., None]),
            axis=-1,
        ),
    )
    S_cdw = float(S_cdw_jk[0])
    S_cdw_err = float(S_cdw_jk[1])

    C_map_plot = np.fft.fftshift(C_map)
    C_stag_plot = np.fft.fftshift(C_stag)
    C_err_map_plot = np.fft.fftshift(C_map_err)

    # Bubble C_map
    xs = np.arange(Nx)
    ys = np.arange(Ny)
    X, Y = np.meshgrid(xs, ys)
    vals = C_map_plot.ravel()
    errs = C_err_map_plot.ravel()
    mask = np.abs(vals) > errs
    vmax = float(np.max(np.abs(vals))) if vlim is None else float(vlim)
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("bwr_r")
    fig, ax = plt.subplots(figsize=(0.9 + 0.7 * Nx, 0.9 + 0.7 * Ny))
    sizes = np.full(vals.shape, 450.0, dtype=float)
    sc = ax.scatter(
        X.ravel()[mask], Y.ravel()[mask],
        s=sizes[mask],
        c=vals[mask],
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        alpha=0.95,
    )
    for x0, y0, v in zip(X.ravel()[mask], Y.ravel()[mask], vals[mask]):
        ax.text(
            x0, y0,
            "+" if v >= 0 else "-",
            ha="center", va="center",
            color="black", fontsize=12, fontweight="bold"
        )
    ax.set_xlim(-0.5, Nx - 0.5)
    ax.set_ylim(-0.5, Ny - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(Nx))
    ax.set_yticks(np.arange(Ny))
    ax.set_xticklabels(np.arange(Nx) - (Nx // 2))
    ax.set_yticklabels(np.arange(Ny) - (Ny // 2))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(run_path, f"{out_prefix}_charge_corr_bubble.png"), dpi=200)
    plt.close(fig)

    plt.figure(figsize=(5, 4))
    plt.imshow(C_stag_plot, origin="lower")
    plt.colorbar(label=r"$(-1)^{x+y}[\langle n_0 n_r\rangle - \langle n\rangle^2]$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(np.arange(Nx), np.arange(Nx) - (Nx // 2))
    plt.yticks(np.arange(Ny), np.arange(Ny) - (Ny // 2))
    plt.title("Average staggered charge correlation map")
    plt.tight_layout()
    plt.savefig(os.path.join(run_path, f"{out_prefix}_charge_corr_stag.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(C_map_plot, origin="lower")
    plt.colorbar(label=r"$\langle n_0 n_r\rangle - \langle n\rangle^2$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(np.arange(Nx), np.arange(Nx) - (Nx // 2))
    plt.yticks(np.arange(Ny), np.arange(Ny) - (Ny // 2))
    plt.title("Average charge correlation map")
    plt.tight_layout()
    plt.savefig(os.path.join(run_path, f"{out_prefix}_charge_corr.png"), dpi=200)
    plt.close()

    print(f"[ok] {run_path} T={T:.12g} beta={beta:.12g} S_cdw={S_cdw:.12g} err={S_cdw_err:.12g}")
    print("Saved:")
    print(f"  {os.path.join(run_path, out_prefix + '_charge_corr.png')}")
    print(f"  {os.path.join(run_path, out_prefix + '_charge_corr_stag.png')}")
    print(f"  {os.path.join(run_path, out_prefix + '_charge_corr_bubble.png')}")

    return {
        "dir": run_path,
        "T": T,
        "beta": beta,
        "S_cdw": S_cdw,
        "S_cdw_err": S_cdw_err,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Base directory, or a single directory containing HDF5 bin files")
    p.add_argument("--glob", default="T_*/", help="Glob pattern for subdirectories under --path, e.g. 'T*_beta*_U*/mu*'")
    p.add_argument("--out_prefix", default="charge_order_avg", help="Output file prefix")
    p.add_argument("--vlim", type=float, default=None,
                   help="Symmetric color limit for bubble plot; default uses data max")
    args = p.parse_args()

    base_path = args.path
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found or not a directory: {base_path}")
    dirs = sorted(glob.glob(os.path.join(base_path, args.glob)))
    run_dirs = [d for d in dirs if os.path.isdir(d) and glob.glob(os.path.join(d, "*.h5"))]
    if not run_dirs:
        raise FileNotFoundError(
            f"No directories containing .h5 files found under {base_path} with glob {args.glob!r}"
        )

    rows = []
    for run_dir in run_dirs:
        rows.append(process_one_dir(run_dir, args.out_prefix, args.vlim))

    order = np.argsort(np.asarray([r["T"] for r in rows], dtype=float))
    T_arr = np.asarray([rows[i]["T"] for i in order], dtype=float)
    S_arr = np.asarray([rows[i]["S_cdw"] for i in order], dtype=float)
    S_err_arr = np.asarray([rows[i]["S_cdw_err"] for i in order], dtype=float)

    np.save(os.path.join(base_path, f"{args.out_prefix}_S_cdw_T.npy"), T_arr)
    np.save(os.path.join(base_path, f"{args.out_prefix}_S_cdw_mean.npy"), S_arr)
    np.save(os.path.join(base_path, f"{args.out_prefix}_S_cdw_err.npy"), S_err_arr)

    print("Saved S_cdw vs T arrays:")
    print(f"  {os.path.join(base_path, args.out_prefix + '_S_cdw_T.npy')}")
    print(f"  {os.path.join(base_path, args.out_prefix + '_S_cdw_mean.npy')}")
    print(f"  {os.path.join(base_path, args.out_prefix + '_S_cdw_err.npy')}")

if __name__ == "__main__":
    main()