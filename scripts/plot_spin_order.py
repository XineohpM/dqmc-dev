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
    zz, xx, dens_u, dens_d, sign, nsamp, beta, U, Nx, Ny = util.load_file(
        file, "meas_eqlt/zz", "meas_eqlt/xx",
        "meas_eqlt/density_u", "meas_eqlt/density_d", "meas_eqlt/sign",
        "meas_eqlt/n_sample", "metadata/beta", "metadata/U",
        "metadata/Nx", "metadata/Ny")
    zz = np.asarray(zz, dtype=float).reshape(-1)
    xx = np.asarray(xx, dtype=float).reshape(-1)
    dens_u = float(np.asarray(dens_u, dtype=float).reshape(-1)[0])
    dens_d = float(np.asarray(dens_d, dtype=float).reshape(-1)[0])
    sign = float(np.asarray(sign, dtype=float).reshape(-1)[0])
    nsamp = int(np.asarray(nsamp).reshape(-1)[0])
    beta = float(np.asarray(beta, dtype=float).reshape(-1)[0])
    U = float(np.asarray(U, dtype=float).reshape(-1)[0])
    Nx0 = int(np.asarray(Nx).reshape(-1)[0])
    Ny0 = int(np.asarray(Ny).reshape(-1)[0])
    nsite = Nx0 * Ny0
    return zz, xx, dens_u, dens_d, sign, nsamp, beta, U, nsite, Nx0, Ny0

def C_to_Sq(nsamp, sign, dens_u, dens_d, zz, Ny, Nx):
    mz = 0.5 * ((dens_u / sign) - (dens_d / sign))
    C = (zz.T / sign).T - (np.asarray(mz)**2)[..., None]
    if C.ndim == 1:
        C = C.reshape(Ny, Nx)
    else:
        C = C.reshape(C.shape[0], Ny, Nx)
    return np.fft.fft2(C, axes=(-2, -1)).real

def process_one_dir(run_path: str, out_prefix: str, vlim=None):
    '''
    Process a single directory containing HDF5 bins. These bins should ideally be with
    the same T, U, mu, and n. Plotting the bubble maps for staggered and non-staggered 
    spin order, and the 2D map of S(q). Returning T and S_zz.
    '''
    files = sorted(glob.glob(os.path.join(run_path, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {run_path}")

    zz0, xx0, dens_u0, dens_d0, sign0, nsamp0, beta0, U0, Nx0, Ny0 = util.load_firstfile(
        os.path.join(run_path, ""),
        "meas_eqlt/zz", "meas_eqlt/xx",
        "meas_eqlt/density_u", "meas_eqlt/density_d", "meas_eqlt/sign",
        "meas_eqlt/n_sample", "metadata/beta", "metadata/U",
        "metadata/Nx", "metadata/Ny")
    Nx = int(np.asarray(Nx0).reshape(-1)[0])
    Ny = int(np.asarray(Ny0).reshape(-1)[0])
    N = Nx * Ny
    beta = float(np.asarray(beta0, dtype=float).reshape(-1)[0])
    T = 1.0 / beta

    bins_nsamp = []
    bins_sign = []
    bins_dens_u = []
    bins_dens_d = []
    bins_zz = []

    for file in files:
        zz, xx, dens_u, dens_d, sign, nsamp, beta_i, U, nsite, Nx_m, Ny_m = get_meas(file)
        beta_bin = float(np.asarray(beta_i, dtype=float).reshape(-1)[0])
        if not np.isclose(beta, beta_bin):
            raise ValueError(f"Beta of bin ({beta_bin}) does not match with beta of the firstfile ({beta})")
        if Nx_m != Nx or Ny_m != Ny or nsite != N:
            raise ValueError(f"Inconsistent lattice size in {file}: got {Nx_m}x{Ny_m}, expected {Nx}x{Ny}")
        if zz.shape[0] != N:
            raise ValueError(f"Unexpected zz length in {file}: got {zz.shape[0]}, expected {N}")
        bins_nsamp.append(nsamp)
        bins_sign.append(sign)
        bins_dens_u.append(dens_u)
        bins_dens_d.append(dens_d)
        bins_zz.append(zz.copy())

    bins_nsamp = np.asarray(bins_nsamp, dtype=float)
    bins_sign = np.asarray(bins_sign, dtype=float)
    bins_dens_u = np.asarray(bins_dens_u, dtype=float)
    bins_dens_d = np.asarray(bins_dens_d, dtype=float)
    bins_zz = np.asarray(bins_zz, dtype=float)

    C_jk = util.jackknife_noniid(
        bins_nsamp,
        bins_sign,
        bins_dens_u,
        bins_dens_d,
        bins_zz,
        f=lambda nsamp, sign, dens_u, dens_d, zz: (
            (zz.T / sign).T
            - (np.asarray(0.5 * ((dens_u / sign) - (dens_d / sign))) ** 2)[..., None]
        ),
    )
    # Connected equal-time spin correlation: C(r) = <S_0^z S_r^z> - <S^z>^2
    C_mean = np.asarray(C_jk[0], dtype=float)
    C_err = np.asarray(C_jk[1], dtype=float)

    C_map = C_mean.reshape(Ny, Nx)
    C_map_err = C_err.reshape(Ny, Nx)
    x = np.arange(Nx)[None, :]
    y = np.arange(Ny)[:, None]
    eta = (-1) ** (x + y)
    eta_flat = eta.reshape(-1)
    C_stag = eta * C_map

    # Equal-time spin structure factor at Q=(pi,pi), with jackknife error.
    S_zz_jk = util.jackknife_noniid(
        bins_nsamp,
        bins_sign,
        bins_dens_u,
        bins_dens_d,
        bins_zz,
        f=lambda nsamp, sign, dens_u, dens_d, zz: np.sum(
            eta_flat * (
                (zz.T / sign).T
                - (np.asarray(0.5 * ((dens_u / sign) - (dens_d / sign))) ** 2)[..., None]
            ),
            axis=-1,
        ),
    )
    S_zz = float(S_zz_jk[0])
    S_zz_err = float(S_zz_jk[1])

    # S_zz(q)
    S_q_jk = util.jackknife_noniid(
        bins_nsamp,
        bins_sign,
        bins_dens_u,
        bins_dens_d,
        bins_zz,
        f=lambda nsamp, sign, dens_u, dens_d, zz: C_to_Sq(
            nsamp, sign, dens_u, dens_d, zz, Ny, Nx
        ),
    )
    S_q_mean = np.asarray(S_q_jk[0], dtype=float)
    S_q_err = np.asarray(S_q_jk[1], dtype=float)

    # Plot S_zz(q) in momentum space. fftshift moves q=(0,0) to the center;
    # for even Nx, Ny, q=(pi,pi) is equivalent to (-pi,-pi) and appears at the BZ corner.
    S_q_plot = np.fft.fftshift(S_q_mean)
    qx_over_pi = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0)) * 2.0
    qy_over_pi = np.fft.fftshift(np.fft.fftfreq(Ny, d=1.0)) * 2.0

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(S_q_plot, origin="lower", aspect="equal")
    fig.colorbar(im, ax=ax, pad=0.02, label=r"$S_{zz}(q)$")
    ax.set_xlabel(r"$q_x/\pi$")
    ax.set_ylabel(r"$q_y/\pi$")
    ax.set_xticks(np.arange(Nx))
    ax.set_yticks(np.arange(Ny))
    ax.set_xticklabels([f"{v:.2g}" for v in qx_over_pi])
    ax.set_yticklabels([f"{v:.2g}" for v in qy_over_pi])
    ax.tick_params(axis="x", labelrotation=45)
    ax.scatter([0], [0], marker="x", s=80, color="black", linewidths=1.5)
    ax.text(0.15, 0.15, r"$(\pi,\pi)$", color="black", fontsize=9,
            ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(run_path, f"{out_prefix}_S_zz_qspace.png"), dpi=200)
    plt.close(fig)

    np.save(os.path.join(run_path, f"{out_prefix}_S_zz_qspace_mean.npy"), S_q_mean)
    np.save(os.path.join(run_path, f"{out_prefix}_S_zz_qspace_err.npy"), S_q_err)

    S_zz_00 = float(S_q_mean[0, 0])
    S_zz_00_err = float(S_q_err[0, 0])
    chi_mean = beta * S_zz_00
    chi_err = beta * S_zz_00_err

    # Check if S_q peaks at (pi,pi)
    iy_pi = Ny // 2
    ix_pi = Nx // 2
    S_pipi = S_q_mean[iy_pi, ix_pi]
    S_pipi_err = S_q_err[iy_pi, ix_pi]
    max_idx = np.unravel_index(np.argmax(S_q_mean), S_q_mean.shape)
    iy_max, ix_max = max_idx
    qx_max = 2 * np.pi * ix_max / Nx
    qy_max = 2 * np.pi * iy_max / Ny
    qx_max_wrapped = (qx_max + np.pi) % (2*np.pi) - np.pi
    qy_max_wrapped = (qy_max + np.pi) % (2*np.pi) - np.pi
    print(
        f"S(pi,pi)={S_pipi:.8g} +/- {S_pipi_err:.3g}; "
        f"max at ix={ix_max}, iy={iy_max}, "
        f"qx/pi={qx_max_wrapped/np.pi:.3f}, "
        f"qy/pi={qy_max_wrapped/np.pi:.3f}, "
        f"Smax={S_q_mean[max_idx]:.8g}"
    )

    # Move the origin to the center of the map
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
    fig.savefig(os.path.join(run_path, f"{out_prefix}_spin_corr_bubble.png"), dpi=200)
    plt.close(fig)

    plt.figure(figsize=(5, 4))
    plt.imshow(C_stag_plot, origin="lower")
    plt.colorbar(label=r"$(-1)^{x+y}[\langle S_0^z S_r^z\rangle - \langle S^z\rangle^2]$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(np.arange(Nx), np.arange(Nx) - (Nx // 2))
    plt.yticks(np.arange(Ny), np.arange(Ny) - (Ny // 2))
    plt.title("Average staggered spin correlation map")
    plt.tight_layout()
    plt.savefig(os.path.join(run_path, f"{out_prefix}_spin_corr_stag.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(C_map_plot, origin="lower")
    plt.colorbar(label=r"$\langle S_0^z S_r^z\rangle - \langle S^z\rangle^2$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(np.arange(Nx), np.arange(Nx) - (Nx // 2))
    plt.yticks(np.arange(Ny), np.arange(Ny) - (Ny // 2))
    plt.title("Average spin correlation map")
    plt.tight_layout()
    plt.savefig(os.path.join(run_path, f"{out_prefix}_spin_corr.png"), dpi=200)
    plt.close()

    print(f"[ok] {run_path} T={T:.12g} beta={beta:.12g} S_zz={S_zz:.12g} err={S_zz_err:.12g}")
    print("Saved:")
    print(f"  {os.path.join(run_path, out_prefix + '_spin_corr.png')}")
    print(f"  {os.path.join(run_path, out_prefix + '_spin_corr_stag.png')}")
    print(f"  {os.path.join(run_path, out_prefix + '_spin_corr_bubble.png')}")
    print(f"  {os.path.join(run_path, out_prefix + '_S_zz_qspace.png')}")
    print(f"  {os.path.join(run_path, out_prefix + '_S_zz_qspace_mean.npy')}")
    print(f"  {os.path.join(run_path, out_prefix + '_S_zz_qspace_err.npy')}")

    return {
        "dir": run_path,
        "T": T,
        "beta": beta,
        "S_zz": S_zz,
        "S_zz_err": S_zz_err,
        "chi_mean": chi_mean,
        "chi_err": chi_err,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Base directory, or a single directory containing HDF5 bin files")
    p.add_argument("--glob", default="T_*/", help="Glob pattern for subdirectories under --path, e.g. 'T*_beta*_U*/mu*'")
    p.add_argument("--out_prefix", default="spin_order_avg", help="Output file prefix")
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
    S_arr = np.asarray([rows[i]["S_zz"] for i in order], dtype=float)
    S_err_arr = np.asarray([rows[i]["S_zz_err"] for i in order], dtype=float)
    chi_mean_arr = np.asarray([rows[i]["chi_mean"] for i in order], dtype=float)
    chi_err_arr = np.asarray([rows[i]["chi_err"] for i in order], dtype=float)

    np.save(os.path.join(base_path, f"{args.out_prefix}_S_zz_T.npy"), T_arr)
    np.save(os.path.join(base_path, f"{args.out_prefix}_S_zz_mean.npy"), S_arr)
    np.save(os.path.join(base_path, f"{args.out_prefix}_S_zz_err.npy"), S_err_arr)
    np.save(os.path.join(base_path, f"{args.out_prefix}_chi_mean.npy"), chi_mean_arr)
    np.save(os.path.join(base_path, f"{args.out_prefix}_chi_err.npy"), chi_err_arr)

    print("Saved S_zz vs T arrays:")
    print(f"  {os.path.join(base_path, args.out_prefix + '_S_zz_T.npy')}")
    print(f"  {os.path.join(base_path, args.out_prefix + '_S_zz_mean.npy')}")
    print(f"  {os.path.join(base_path, args.out_prefix + '_S_zz_err.npy')}")
    print(f"  {os.path.join(base_path, args.out_prefix + '_chi_mean.npy')}")
    print(f"  {os.path.join(base_path, args.out_prefix + '_chi_err.npy')}")

if __name__ == "__main__":
    main()
