#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from glob import glob
from pathlib import Path
import sys, os, re
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

utilpath = Path(__file__).resolve().parents[1]/"util"
sys.path.insert(0, str(utilpath))

import util

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
    p.add_argument("--xrange", default=None,
                   help="Optional x-range for zoom as 'xmin,xmax'. If provided, also creates (1) a full plot with an inset showing this x-range and (2) a separate zoomed plot with xlim set to this range.")
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
    
    # Parse xrange argument if provided
    xlim_zoom = None
    if args.xrange:
        try:
            parts = [p.strip() for p in str(args.xrange).split(',')]
            if len(parts) != 2:
                raise ValueError
            x0, x1 = float(parts[0]), float(parts[1])
            if not (np.isfinite(x0) and np.isfinite(x1)):
                raise ValueError
            if x1 <= x0:
                raise ValueError
            xlim_zoom = (x0, x1)
        except Exception:
            raise ValueError("--xrange must be in the form 'xmin,xmax' with xmin<xmax, e.g. --xrange 0.2,1.0")

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

        # Per-bin normalized observables
        # n_bin is per-site filling <n> for that bin; S_bin is sum_r <n_i n_{i+r}> averaged over i
        n_bin = n / nsamp
        S_bin = nn.sum(axis=1) / nsamp

        # Filter invalid bins (NaN/Inf or zero sign)
        valid = np.isfinite(sign) & np.isfinite(n_bin) & np.isfinite(S_bin) & (sign != 0)
        sign = sign[valid]
        n_bin = n_bin[valid]
        S_bin = S_bin[valid]

        if sign.size < 3:
            n_bad += 1
            rows.append((n_target, T, float(beta[0]) if np.size(beta) else np.nan, float(U[0]) if np.size(U) else np.nan,
                         float(mu[0]) if np.size(mu) else np.nan, np.nan, np.nan, np.nan, float(np.mean(sign)) if sign.size else np.nan,
                         int(sign.size), mu_dir))
            continue

        # Use jackknife on sign-reweighted means, then form chi:
        # chi = beta * ( <S> - nsite * <n>^2 )
        bval = float(np.asarray(beta).reshape(-1)[0])

        def f(sum_s, sum_sS, sum_sn):
            S_mean = sum_sS / sum_s
            n_mean = sum_sn / sum_s
            return bval * (S_mean - nsite * (n_mean ** 2))

        jk = util.jackknife(sign, sign * S_bin, sign * n_bin, f=f)
        chi = float(jk[0])
        chi_err = float(jk[1])

        # Also report the measured <n> at this mu for sanity
        jk_n = util.jackknife(sign, sign * n_bin)
        n_mean = float(jk_n[0])
        n_err = float(jk_n[1])
        dn = n_mean - n_target if np.isfinite(n_target) else np.nan

        rows.append((n_target, T, bval, float(np.asarray(U).reshape(-1)[0]), float(np.asarray(mu).reshape(-1)[0]),
                     chi, chi_err, dn, float(np.mean(sign)), int(sign.size), mu_dir))

    # Sort by target n then T
    rows.sort(key=lambda r: (r[0], r[1]))

    out_tsv = out_prefix + "_chi.tsv"
    header = "n_target\tT\tbeta\tU\tmu\tchi\tchi_err\tdelta_n\tavg_sign\tnbin\tmu_dir\n"
    with open(out_tsv, "w") as f:
        f.write(header)
        for (n_target, T, beta, U, mu, chi, chi_err, dn, avg_sign, nbin, mu_dir) in rows:
            f.write(
                f"{n_target:.12g}\t{T:.12g}\t{beta:.12g}\t{U:.12g}\t{mu:.12g}\t"
                f"{chi:.12g}\t{chi_err:.12g}\t{dn:.12g}\t{avg_sign:.12g}\t{nbin}\t{mu_dir}\n"
            )

    n_total = len(rows)
    n_nan_chi = sum((not np.isfinite(r[5])) for r in rows)
    print(f"[OK] wrote {out_tsv}")
    print(f"Total mu dirs: {n_total}")
    print(f"Rows with NaN chi: {n_nan_chi}")
    if n_bad:
        print(f"Rows failed to process (too few valid bins): {n_bad}")

    # ----------------------------
    def _apply_xrange_and_save(fig, ax, base_png, series_list):
        """If xlim_zoom is set, save a full plot with inset (zoom) and a separate zoom-only plot.

        series_list: list of dicts with keys: x (np.ndarray), y, yerr, label (str), color.
        """
        if xlim_zoom is None:
            return

        x0, x1 = xlim_zoom
        suffix = f"_x_{x0:g}_{x1:g}"

        # Determine y-limits of zoom region based on the series data
        yvals = []
        for s in series_list:
            xd = np.asarray(s["x"], float)
            yd = np.asarray(s["y"], float)
            m = (xd >= x0) & (xd <= x1) & np.isfinite(xd) & np.isfinite(yd)
            if np.any(m):
                yvals.append(yd[m])
        if yvals:
            y_all = np.concatenate(yvals)
            y0, y1 = float(np.min(y_all)), float(np.max(y_all))
            pad = 0.08 * (y1 - y0) if y1 > y0 else 0.1 * (abs(y0) + 1.0)
            y_zoom = (y0 - pad, y1 + pad)
        else:
            y_zoom = None

        # 1) Full plot with inset
        axins = inset_axes(ax, width="42%", height="42%", loc="upper right")
        for s in series_list:
            axins.errorbar(
                s["x"], s["y"], yerr=s["yerr"],
                fmt='o-', capsize=3, color=s["color"], label=s["label"]
            )
        axins.set_xlim(x0, x1)
        if y_zoom is not None:
            axins.set_ylim(*y_zoom)
        axins.grid(True)
        try:
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        except Exception:
            pass

        inset_png = base_png.replace(".png", f"{suffix}_inset.png")
        fig.tight_layout()
        fig.savefig(inset_png, dpi=200)
        print(f"[OK] wrote {inset_png}")

        # 2) Zoom-only plot
        fig2, ax2 = plt.subplots()
        for s in series_list:
            ax2.errorbar(
                s["x"], s["y"], yerr=s["yerr"],
                fmt='o-', capsize=3, color=s["color"], label=s["label"]
            )
        ax2.set_xlim(x0, x1)
        if y_zoom is not None:
            ax2.set_ylim(*y_zoom)
        ax2.set_xlabel(ax.get_xlabel())
        ax2.set_ylabel(ax.get_ylabel())
        ax2.grid(True)
        handles, labels = ax2.get_legend_handles_labels()
        if any(lbl and not lbl.startswith('_') for lbl in labels):
            ax2.legend()
        zoom_png = base_png.replace(".png", f"{suffix}_zoom.png")
        fig2.tight_layout()
        fig2.savefig(zoom_png, dpi=200)
        plt.close(fig2)
        print(f"[OK] wrote {zoom_png}")
    # ----------------------------
    # Plot chi(T) with error bars
    # ----------------------------
    # Collect data by n_target
    by_n = {}
    for (n_target, T, beta, U, mu, chi, chi_err, dn, avg_sign, nbin, mu_dir) in rows:
        if not (np.isfinite(n_target) and np.isfinite(T) and np.isfinite(chi) and np.isfinite(chi_err)):
            continue
        by_n.setdefault(n_target, []).append((T, chi, chi_err))

    if by_n:
        # Overlay plot
        plt.figure()
        series_list = []
        for n_target in sorted(by_n.keys()):
            pts = sorted(by_n[n_target], key=lambda x: x[0])
            Ts = np.array([p[0] for p in pts], float)
            chis = np.array([p[1] for p in pts], float)
            errs = np.array([p[2] for p in pts], float)
            label = rf"$\left<n\right>={n_target:g}$"
            cont = plt.errorbar(Ts, chis, yerr=errs, fmt='o-', capsize=3, label=label)
            color = cont.lines[0].get_color()
            series_list.append({"x": Ts, "y": chis, "yerr": errs, "label": label, "color": color})
        plt.xlabel("T")
        plt.ylabel(r"$\chi$")
        plt.grid(True)
        plt.legend()
        overlay_png = out_prefix + "_chi_overlay.png"
        plt.tight_layout()
        plt.savefig(overlay_png, dpi=200)
        print(f"[OK] wrote {overlay_png}")
        _apply_xrange_and_save(plt.gcf(), plt.gca(), overlay_png, series_list)
        plt.close()

        # Per-n plots
        for n_target in sorted(by_n.keys()):
            pts = sorted(by_n[n_target], key=lambda x: x[0])
            Ts = np.array([p[0] for p in pts], float)
            chis = np.array([p[1] for p in pts], float)
            errs = np.array([p[2] for p in pts], float)
            plt.figure()
            series_list = []
            label = rf"$\left<n\right>={n_target:g}$"
            cont = plt.errorbar(Ts, chis, yerr=errs, fmt='o-', capsize=3, label=label)
            color = cont.lines[0].get_color()
            series_list.append({"x": Ts, "y": chis, "yerr": errs, "label": label, "color": color})
            plt.xlabel("T")
            plt.ylabel(r"$\chi$")
            plt.grid(True)
            plt.legend()
            per_png = out_prefix + f"_chi_n{n_target:g}.png"
            plt.tight_layout()
            plt.savefig(per_png, dpi=200)
            print(f"[OK] wrote {per_png}")
            _apply_xrange_and_save(plt.gcf(), plt.gca(), per_png, series_list)
            plt.close()

if __name__ == "__main__":
    main()
