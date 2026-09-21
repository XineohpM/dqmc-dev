'''
plot_dos.py
phoenixm@stanford.edu

Usage example:
python3 /home/users/phoenixm/scripts/plot_dos.py \
        --base /scratch/users/phoenixm/dqmc_runs/U-6_6x6_tp0_nflux0/n0.6_resistivity \
        --items \
            "T0.05_beta20_U-6/mu-0.462713/maxent_out,maxent_dos_,0.05" \
            "T0.1_beta10_U-6/mu-0.463656/maxent_out,maxent_dos_,0.1" \
            "T0.125_beta8_U-6/mu-0.464428/maxent_out,maxent_dos_,0.125" \
            "T0.166667_beta6_U-6/mu-0.466893/maxent_out,maxent_dos_,0.166667" \
            "T0.2_beta5_U-6/mu-0.470034/maxent_out,maxent_dos_,0.2" \
            "T0.222222_beta4.5_U-6/mu-0.472615/maxent_out,maxent_dos_,0.222222" \
            "T0.25_beta4_U-6/mu-0.47691/maxent_out,maxent_dos_,0.25" \
            "T0.285714_beta3.5_U-6/mu-0.484274/maxent_out,maxent_dos_,0.285714" \
            "T0.333333_beta3_U-6/mu-0.496175/maxent_out,maxent_dos_,0.333333" \
            "T0.4_beta2.5_U-6/mu-0.515708/maxent_out,maxent_dos_,0.4" \
            "T0.5_beta2_U-6/mu-0.5489/maxent_out,maxent_dos_,0.5" \
            "T0.666667_beta1.5_U-6/mu-0.610407/maxent_out,maxent_dos_,0.666667" \
            "T1_beta1_U-6/mu-0.755037/maxent_out,maxent_dos_,1.0" \
            "T2_beta0.5_U-6/mu-1.3016/maxent_out,maxent_dos_,2.0" \
            "T4_beta0.25_U-6/mu-2.68617/maxent_out,maxent_dos_,4.0" \
            "T8_beta0.125_U-6/mu-5.85011/maxent_out,maxent_dos_,8.0"
'''

import os, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_sigma(dpath: str, prefix: str):
    """
    Load omega, domega and s_all, drop bootstrap rows that are not finite, then return:
      omega: (Nw,)
      domega: (Nw,)
      sigma_mean: (Nw,)
      sigma_std: (Nw,)  (computed across bootstrap samples)
      n_good, n_tot
    """
    sp = os.path.join(dpath, prefix + "s_all.npy")
    wp = os.path.join(dpath, prefix + "omega.npy")
    dwp = os.path.join(dpath, prefix + "domega.npy")
    if not (os.path.exists(sp) and os.path.exists(wp) and os.path.exists(dwp)):
        raise FileNotFoundError(f"Missing {sp} or {wp} or {dwp}")

    s_all = np.load(sp)         # (bs, Nw)
    omega = np.load(wp)         # (Nw,)
    domega = np.load(dwp)       # (Nw,)

    # Drop failed bootstraps: require whole row finite
    good_mask = np.isfinite(s_all).all(axis=1)
    good = s_all[good_mask] / np.pi
    n_good, n_tot = good.shape[0], s_all.shape[0]

    if n_good == 0:
        raise RuntimeError(f"All bootstrap spectra are invalid in {dpath} (prefix={prefix})")

    s_mean = good.mean(axis=0)
    s_std = np.std(good, axis=0, ddof=1) if n_good > 1 else np.zeros_like(s_mean)
    
    norm = np.sum(s_mean * domega)
    i0 = np.argmin(np.abs(omega))
    i0dos = s_mean[i0]
    print(dpath)
    print("integral of mean DOS =", norm)
    print("closest omega to 0 =", omega[i0])
    print("DOS at omega~0 =", i0dos)

    return omega, domega, s_mean, s_std, n_good, n_tot, i0dos

def main():
    ap = argparse.ArgumentParser(
        description="Plot s(omega) from MaxEnt outputs, automatically dropping failed (NaN) bootstraps."
    )
    ap.add_argument("--base", required=True,
                    help="Base directory that contains the maxent_out folders.")
    ap.add_argument("--items", nargs="+", required=True,
                    help=("List of items: each is 'relpath,prefix,T'. Example: "
                          "'T_0.2/maxent_out,maxent_dos_,0.2'"))
    ap.add_argument("--out",
                    help="Output PNG path (full or relative).")
    ap.add_argument("--output_name", default=None,
                    help="Custom output filename for the DOS vs omega figure. If provided, the DOS-vs-omega plot is saved as this name, the DOS-at-omega~0 vs T plot is saved as '<output_name>_0freq.png', and the associated .npy files use the same root name.")
    ap.add_argument("--xmin", type=float, default=None,
                    help="Optional x-axis min omega. Default: min over all curves.")
    ap.add_argument("--xmax", type=float, default=None,
                    help="Optional x-axis max omega. Default: max over all curves.")
    ap.add_argument("--ymin", type=float, default=None,
                    help="Optional y-axis min. Default: auto.")
    ap.add_argument("--ymax", type=float, default=None,
                    help="Optional y-axis max. Default: auto.")
    ap.add_argument("--no_band", action="store_true",
                    help="If set, do not draw errorbar.")
    args = ap.parse_args()

    i0dos_arr = []
    curves = []
    for item in args.items:
        rel, pfx, Tstr = item.split(",")
        T = float(Tstr)
        dpath = os.path.join(args.base, rel)

        omega, domega, s_mean, s_std, ng, nt, i0dos = load_sigma(dpath, pfx)
        curves.append((T, omega, s_mean, s_std, ng, nt))
        i0dos_arr.append((T, i0dos))

    # Sort by T descending (optional); change to ascending if you prefer
    curves.sort(key=lambda x: x[0], reverse=True)
    i0dos_arr.sort(key=lambda x: x[0], reverse=True)

    # Assign colors: high-T red -> low-T blue (similar to common optical-conductivity plots)
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(np.linspace(0.90, 0.10, len(curves)))

    # Plot
    plt.figure(figsize=(7.4, 5.0))
    xmin = 0.0
    xmax = 0.0
    ymin = +np.inf
    ymax = -np.inf
    tvals = []
    dos0_vals = []
    dos0_errs = []
    omega0_vals = []

    for i, (T, w, s_mean, s_std, ng, nt) in enumerate(curves):
        xmin = min(xmin, float(np.min(w)))
        xmax = max(xmax, float(np.max(w)))
        lower = s_mean - s_std
        upper = s_mean + s_std
        ymin = min(ymin, float(np.min(lower if not args.no_band else s_mean)))
        ymax = max(ymax, float(np.max(upper if not args.no_band else s_mean)))

        i0 = int(np.argmin(np.abs(w)))
        tvals.append(float(T))
        dos0_vals.append(float(s_mean[i0]))
        dos0_errs.append(float(s_std[i0]))
        omega0_vals.append(float(w[i0]))

        # label = f"T = {T:g}  (good {ng}/{nt})"
        label = f"T = {T:g}"
        plt.plot(w, s_mean, lw=2, label=label, color=colors[i])
        if not args.no_band:
            plt.fill_between(w, lower, upper, alpha=0.20, linewidth=0, color=colors[i])

        # Console summary
        print(f"T={T:g}  good={ng}/{nt}  omega_max={w.max():.3f}  "
              f"s: min={s_mean.min():.3e} max={s_mean.max():.3e}  area(trapz)={np.trapz(s_mean,w):.6g}  "
              f"omega~0={w[i0]:.6g}  dos0={s_mean[i0]:.6g} +/- {s_std[i0]:.6g}")

    plt.xlim(args.xmin if args.xmin is not None else xmin, 
             args.xmax if args.xmax is not None else xmax)

    # y-limits
    ylo = args.ymin if args.ymin is not None else min(0.0, ymin*1.05)
    yhi = args.ymax if args.ymax is not None else max(0.10, ymax*1.05)
    plt.ylim(ylo, yhi)

    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$N(\omega)$')
    plt.grid(alpha=0.25, ls=':')
    plt.legend(frameon=False)
    plt.tight_layout()
    
    if args.output_name is not None:
        outpath = os.path.expanduser(args.output_name)
    elif args.out is not None:
        outpath = os.path.expanduser(args.out)
    else:
        outpath = os.path.join(args.base, "dos_vs_T.png")

    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(outpath, dpi=180)

    tvals = np.asarray(tvals, dtype=float)
    dos0_vals = np.asarray(dos0_vals, dtype=float)
    dos0_errs = np.asarray(dos0_errs, dtype=float)
    omega0_vals = np.asarray(omega0_vals, dtype=float)

    outroot, outext = os.path.splitext(outpath)
    if args.output_name is not None:
        outpath_dos0 = outroot + "_0freq.png"
        np.save(outroot + "_0freq_T.npy", tvals)
        np.save(outroot + "_0freq_mean.npy", dos0_vals)
        np.save(outroot + "_0freq_std.npy", dos0_errs)
        np.save(outroot + "_0freq_omega.npy", omega0_vals)
    else:
        outpath_dos0 = outroot + "_dos0_vs_T.png"
        np.save(outroot + "_dos0_T.npy", tvals)
        np.save(outroot + "_dos0_mean.npy", dos0_vals)
        np.save(outroot + "_dos0_std.npy", dos0_errs)
        np.save(outroot + "_dos0_omega.npy", omega0_vals)

    plt.figure(figsize=(6.4, 4.6))
    plt.errorbar(tvals, dos0_vals, yerr=dos0_errs, fmt='o-', lw=1.6, ms=4, capsize=3)
    plt.xlabel(r'$T$')
    plt.ylabel(r'$N(\omega \approx 0)$')
    plt.grid(alpha=0.25, ls=':')
    plt.tight_layout()
    plt.savefig(outpath_dos0, dpi=180)

    print("Wrote", outpath)
    print("Wrote", outpath_dos0)
    if args.output_name is not None:
        print("Wrote", outroot + "_0freq_T.npy")
        print("Wrote", outroot + "_0freq_mean.npy")
        print("Wrote", outroot + "_0freq_std.npy")
        print("Wrote", outroot + "_0freq_omega.npy")
    else:
        print("Wrote", outroot + "_dos0_T.npy")
        print("Wrote", outroot + "_dos0_mean.npy")
        print("Wrote", outroot + "_dos0_std.npy")
        print("Wrote", outroot + "_dos0_omega.npy")


if __name__ == "__main__":
    main()