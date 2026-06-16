#!/usr/bin/env python3
'''
conductivity_plot.py
phoenixm@stanford.edu

Usage example:
    python3 conductivity_plot.py \
        --base /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_conductivity \
        --items \
            "T_4/maxent_out,U-6_T4_jjxx_,4.0" \
            "T_0.5/maxent_out,U-6_T0.5_jjxx_,0.5" \
            "T_0.333333/maxent_out,U-6_T0.333333_jjxx_,0.333333" \
            "T_0.25/maxent_out,U-6_T0.25_jjxx_,0.25" \
            "T_0.2/maxent_out,U-6_T0.2_jjxx_,0.2" \
        --divide_pi \
        --out /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_conductivity/sigma_U-6.png

    python3 /home/users/phoenixm/scripts/conductivity_plot.py \
        --base /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_resistivity \
        --items \
            "T_8/maxent_out,U-6_T8_JNJNxx_,8.0" \
            "T_4/maxent_out,U-6_T4_JNJNxx_,4.0" \
            "T_2/maxent_out,U-6_T2_JNJNxx_,2.0" \
            "T_1/maxent_out,U-6_T1_JNJNxx_,1.0" \
            "T_0.666667/maxent_out,U-6_T0.666667_JNJNxx_,0.666667" \
            "T_0.5/maxent_out,U-6_T0.5_JNJNxx_,0.5" \
            "T_0.4/maxent_out,U-6_T0.4_JNJNxx_,0.4" \
            "T_0.333333/maxent_out,U-6_T0.333333_JNJNxx_,0.333333" \
            "T_0.285714/maxent_out,U-6_T0.285714_JNJNxx_,0.285714" \
            "T_0.25/maxent_out,U-6_T0.25_JNJNxx_,0.25" \
            "T_0.222222/maxent_out,U-6_T0.222222_JNJNxx_,0.222222" \
            "T_0.2/maxent_out,U-6_T0.2_JNJNxx_,0.2" \
        --divide_pi \
        --out /scratch/users/phoenixm/dqmc_runs/U-6_n6x6_resistivity/sigma_U-6.png
'''
import os, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_sigma(dpath: str, prefix: str, divide_pi: bool = True):
    """
    Load omega and s_all, drop bootstrap rows that are not finite, then return:
      omega: (Nw,)
      sigma_mean: (Nw,)
      sigma_stderr: (Nw,)  (standard error across bootstrap samples)
      n_good, n_tot
    """
    sp = os.path.join(dpath, prefix + "s_all.npy")
    wp = os.path.join(dpath, prefix + "omega.npy")
    if not (os.path.exists(sp) and os.path.exists(wp)):
        raise FileNotFoundError(f"Missing {sp} or {wp}")

    s_all = np.load(sp)         # (bs, Nw)
    omega = np.load(wp)         # (Nw,)

    # Drop failed bootstraps: require whole row finite
    good_mask = np.isfinite(s_all).all(axis=1)
    good = s_all[good_mask]
    n_good, n_tot = good.shape[0], s_all.shape[0]

    if n_good == 0:
        raise RuntimeError(f"All bootstrap spectra are invalid in {dpath} (prefix={prefix})")

    # Convert to sigma(omega) if requested
    if divide_pi:
        good = good / np.pi

    sigma_mean = good.mean(axis=0)
    sigma_stderr = good.std(axis=0, ddof=1) / np.sqrt(n_good) if n_good > 1 else np.zeros_like(sigma_mean)

    return omega, sigma_mean, sigma_stderr, n_good, n_tot

def main():
    ap = argparse.ArgumentParser(
        description="Plot sigma(omega) from MaxEnt outputs, automatically dropping failed (NaN) bootstrap spectra."
    )
    ap.add_argument("--base", required=True,
                    help="Base directory that contains the maxent_out folders.")
    ap.add_argument("--items", nargs="+", required=True,
                    help=("List of items: each is 'relpath,prefix,T'. Example: "
                          "'T_0.2/maxent_out,U-6_T0.2_jjxx_,0.2'"))
    ap.add_argument("--out", required=True,
                    help="Output PNG path (full or relative). Must be provided; no default is used.")
    ap.add_argument("--divide_pi", action="store_true",
                    help="If set, plot sigma = s_all/pi. (Recommended if s_all is spectral density times pi.)")
    ap.add_argument("--xmax", type=float, default=None,
                    help="Optional x-axis max omega. Default: max over all curves.")
    ap.add_argument("--ymin", type=float, default=None,
                    help="Optional y-axis min. Default: auto.")
    ap.add_argument("--ymax", type=float, default=None,
                    help="Optional y-axis max. Default: auto.")
    ap.add_argument("--no_band", action="store_true",
                    help="If set, do not draw the standard-error band.")
    args = ap.parse_args()

    curves = []
    for item in args.items:
        rel, pfx, Tstr = item.split(",")
        T = float(Tstr)
        dpath = os.path.join(args.base, rel)

        omega, mu, stderr, ng, nt = load_sigma(dpath, pfx, divide_pi=args.divide_pi)
        curves.append((T, omega, mu, stderr, ng, nt))

    # Sort by T descending (optional); change to ascending if you prefer
    curves.sort(key=lambda x: x[0], reverse=True)

    # Assign colors: high-T red -> low-T blue (similar to common optical-conductivity plots)
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(np.linspace(0.90, 0.10, len(curves)))

    # Plot
    plt.figure(figsize=(4, 6.5))
    xmax = 0.0
    ymin = +np.inf
    ymax = -np.inf

    for i, (T, w, mu, stderr, ng, nt) in enumerate(curves):
        lower = mu - stderr
        upper = mu + stderr
        xmax = max(xmax, float(np.max(w)))
        ymin = min(ymin, float(np.min(lower if not args.no_band else mu)))
        ymax = max(ymax, float(np.max(upper if not args.no_band else mu)))

        # label = f"T = {T:g}  (good {ng}/{nt})"
        label = f"T/t = {T:g}"
        plt.plot(w, mu, lw=2, label=label, color=colors[i])
        if not args.no_band:
            plt.fill_between(w, lower, upper, alpha=0.20, linewidth=0, color=colors[i])

        # Console summary
        print(f"T={T:g}  good={ng}/{nt}  omega_max={w.max():.3f}  "
              f"sigma: min={mu.min():.3e} max={mu.max():.3e}  area(trapz)={np.trapz(mu,w):.6g}")

    plt.xlim(0, args.xmax if args.xmax is not None else xmax)

    # y-limits
    ylo = args.ymin if args.ymin is not None else min(0.0, ymin*1.05)
    yhi = args.ymax if args.ymax is not None else max(0.10, ymax*1.05)
    plt.ylim(ylo, yhi)

    plt.xlabel(r'$\omega/t$')
    plt.ylabel(r'$\sigma_{xx}\ \left[e^{2} / \hbar\right]$')
    #plt.grid(alpha=0.25, ls=':')
    plt.legend(frameon=False)
    plt.tight_layout()
    outpath = os.path.expanduser(args.out)
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(outpath, dpi=180)
    print("Wrote", outpath)

if __name__ == "__main__":
    main()
