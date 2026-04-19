'''
plot_charge_order.py

Usage Example:

 python3 /home/users/phoenixm/scripts/plot_charge_order.py \
   --path /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_conductivity_resistivity_02242026/T_0.2 \
   --out_prefix T0.2

 python3 /home/users/phoenixm/scripts/plot_charge_order.py \
   --path /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_conductivity_resistivity_02242026/T_0.2 \
   --output_path /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_conductivity_resistivity_02242026/T_0.2/plots \
   --vlim 0.8

homedir=$(pwd)
for dir in T*/; do
    [ -d "$dir" ] || continue
    absdir="${homedir}/${dir%/}"
    echo "$absdir"
    python3 /home/users/phoenixm/scripts/plot_charge_order.py \
    --path "$absdir" \
    --out_prefix vlim0p8 \
    --vlim 0.8
done
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Directory containing HDF5 bin files")
    p.add_argument("--output_path",
                   help="Directory for output, will be created if needed")
    p.add_argument("--out_prefix", default="charge_order_avg", help="Output file prefix")
    p.add_argument("--vlim", type=float, default=None,
                   help="Symmetric color limit for bubble plot; default uses data max")
    args = p.parse_args()
    path = args.path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Base path not found or not a directory: {path}")
    output_path = args.output_path if args.output_path is not None else path
    os.makedirs(output_path, exist_ok=True)

    files = sorted(glob.glob(os.path.join(path, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {path}")
    
    N = 0
    Nx = 0
    Ny = 0
    
    nn0, n0, sign0, nsamp0, beta0, U0, Nx0, Ny0 = util.load_firstfile(
        os.path.join(path, ""),
        "meas_eqlt/nn", "meas_eqlt/density", "meas_eqlt/sign",
        "meas_eqlt/n_sample", "metadata/beta", "metadata/U",
        "metadata/Nx", "metadata/Ny")
    Nx = int(np.asarray(Nx0).reshape(-1)[0])
    Ny = int(np.asarray(Ny0).reshape(-1)[0])
    N = Nx * Ny
    
    sum_sign = 0.0
    sum_dens = 0.0
    sum_nn = np.zeros(N, dtype=float)

    bins_nsamp = []
    bins_sign = []
    bins_dens = []
    bins_nn = []

    for file in files:
        nn, n, sign, nsamp, beta, U, nsite, Nx_m, Ny_m = get_meas(file)
        sum_sign += sign
        sum_dens += n 
        sum_nn += nn

        bins_nsamp.append(nsamp)
        bins_sign.append(sign)
        bins_dens.append(n)
        bins_nn.append(nn.copy())

    if sum_sign == 0: 
        raise ZeroDivisionError("Total sign is 0, cannot normalize.")
    
    avg_dens = sum_dens / sum_sign
    avg_nn = sum_nn / sum_sign
    C = avg_nn - avg_dens**2

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
    C_err = np.asarray(C_jk[1], dtype=float)

    C_map = C.reshape(Ny, Nx)
    x = np.arange(Nx)[None, :]
    y = np.arange(Ny)[:, None]
    eta = (-1) ** (x + y)
    C_stag = eta * C_map

    C_map_plot = np.fft.fftshift(C_map)
    C_stag_plot = np.fft.fftshift(C_stag)

    C_err_map = C_err.reshape(Ny, Nx)
    C_err_map_plot = np.fft.fftshift(C_err_map)

    # Bubble C_map
    xs = np.arange(Nx)
    ys = np.arange(Ny)
    X, Y = np.meshgrid(xs, ys)
    vals = C_map_plot.ravel()
    errs = C_err_map_plot.ravel()
    mask = np.abs(vals) > errs
    vmax = float(np.max(np.abs(vals))) if args.vlim is None else float(args.vlim)
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
    fig.savefig(os.path.join(output_path, f"{args.out_prefix}_charge_corr_bubble.png"), dpi=200)
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
    plt.savefig(os.path.join(output_path, f"{args.out_prefix}_charge_corr_stag.png"), dpi=200)
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
    plt.savefig(os.path.join(output_path, f"{args.out_prefix}_charge_corr.png"), dpi=200)
    plt.close()


    print("Saved:")
    print(f"  {os.path.join(output_path, args.out_prefix + '_charge_corr.png')}")
    print(f"  {os.path.join(output_path, args.out_prefix + '_charge_corr_stag.png')}")
    print(f"  {os.path.join(output_path, args.out_prefix + '_charge_corr_bubble.png')}")

if __name__ == "__main__":
    main()